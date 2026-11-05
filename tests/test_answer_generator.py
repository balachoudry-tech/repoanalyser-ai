
"""
Unit tests for AnthropicAnswerGenerator.

All tests use a fake Anthropic client. No real API calls or API credits
are required.
"""

from types import SimpleNamespace

import pytest

from src.llm.answer_generator import (
    AnthropicAnswerGenerator,
    LLMProviderError,
)


class FakeMessages:
    """Simulate the Anthropic Messages API."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.last_request = None
        self.call_count = 0

    def create(self, **kwargs):
        """Capture the request and return a simulated response."""
        self.call_count += 1
        self.last_request = kwargs

        if self.error:
            raise self.error

        return self.response


class FakeAnthropicClient:
    """Provide a fake client with the Anthropic SDK interface."""

    def __init__(self, response=None, error=None):
        self.messages = FakeMessages(
            response=response,
            error=error,
        )


@pytest.fixture
def context_result():
    """Provide representative ContextBuilder output."""
    return {
        "context": (
            "[S1]\n"
            "File: src/auth/service.py\n"
            "Symbol: authenticate_user\n"
            "Lines: 10-25\n"
            "\n"
            "```python\n"
            "def authenticate_user(username, password):\n"
            "    return verify_password(username, password)\n"
            "```"
        ),
        "sources": [
            {
                "source_id": "S1",
                "file_path": "src/auth/service.py",
                "symbol_name": "authenticate_user",
                "start_line": 10,
                "end_line": 25,
                "language": "python",
            }
        ],
        "source_count": 1,
        "skipped_count": 0,
        "truncated": False,
    }


@pytest.fixture
def fake_response():
    """Provide a simulated successful Anthropic response."""
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text=(
                    "User authentication is handled by "
                    "`authenticate_user` in "
                    "`src/auth/service.py` [S1]."
                ),
            )
        ],
        usage=SimpleNamespace(
            input_tokens=250,
            output_tokens=45,
        ),
        _request_id="req_test_123",
        stop_reason="end_turn",
    )


def test_generate_answer_success(context_result, fake_response):
    """Verify successful answer generation and response metadata."""
    client = FakeAnthropicClient(response=fake_response)
    generator = AnthropicAnswerGenerator(
        client=client,
        model="claude-test-model",
    )

    result = generator.generate_answer(
        "Where is user authentication implemented?",
        context_result,
    )

    assert "authenticate_user" in result["answer"]
    assert "[S1]" in result["answer"]
    assert result["model"] == "claude-test-model"
    assert result["usage"]["input_tokens"] == 250
    assert result["usage"]["output_tokens"] == 45
    assert result["request_id"] == "req_test_123"
    assert result["stop_reason"] == "end_turn"
    assert result["citation_warnings"] == []
    assert result["sources"] == context_result["sources"]


def test_anthropic_request_contains_question_and_context(
    context_result,
    fake_response,
):
    """Verify the question and evidence reach the provider."""
    client = FakeAnthropicClient(response=fake_response)
    generator = AnthropicAnswerGenerator(client=client)

    generator.generate_answer(
        "Where is authentication implemented?",
        context_result,
    )

    request = client.messages.last_request

    assert request["model"] == "claude-sonnet-5-5"
    assert request["max_tokens"] == 1500
    assert "untrusted" in request["system"].lower()

    user_message = request["messages"][0]["content"]

    assert "Where is authentication implemented?" in user_message
    assert "src/auth/service.py" in user_message


def test_empty_sources_do_not_call_anthropic(context_result):
    """Verify no provider request is made when retrieval finds nothing."""
    context_result["sources"] = []

    client = FakeAnthropicClient()
    generator = AnthropicAnswerGenerator(client=client)

    result = generator.generate_answer(
        "Where is authentication implemented?",
        context_result,
    )

    assert "couldn't find relevant source code" in result["answer"].lower()
    assert result["sources"] == []
    assert result["usage"]["input_tokens"] == 0
    assert client.messages.call_count == 0


@pytest.mark.parametrize("question", ["", " ", None, 123])
def test_invalid_question_is_rejected(context_result, question):
    """Verify empty or non-string questions are rejected."""
    generator = AnthropicAnswerGenerator(
        client=FakeAnthropicClient()
    )

    with pytest.raises(ValueError):
        generator.generate_answer(question, context_result)


def test_invalid_context_type_is_rejected():
    """Verify ContextBuilder output must be a dictionary."""
    generator = AnthropicAnswerGenerator(
        client=FakeAnthropicClient()
    )

    with pytest.raises(ValueError, match="dictionary"):
        generator.generate_answer("Explain authentication", [])


def test_invalid_context_field_is_rejected():
    """Verify context must be a string."""
    generator = AnthropicAnswerGenerator(
        client=FakeAnthropicClient()
    )

    with pytest.raises(ValueError, match="Context must be a string"):
        generator.generate_answer(
            "Explain authentication",
            {
                "context": None,
                "sources": [],
            },
        )


def test_invalid_source_metadata_is_rejected():
    """Verify every source must have a source identifier."""
    generator = AnthropicAnswerGenerator(
        client=FakeAnthropicClient()
    )

    with pytest.raises(ValueError, match="source_id"):
        generator.generate_answer(
            "Explain authentication",
            {
                "context": "Some context",
                "sources": [{"file_path": "auth.py"}],
            },
        )


def test_unknown_citations_are_reported(
    context_result,
    fake_response,
):
    """Verify citations absent from retrieved sources are detected."""
    fake_response.content[0].text = (
        "Authentication is implemented here [S1]. "
        "The database layer is in db.py [S9]."
    )

    client = FakeAnthropicClient(response=fake_response)
    generator = AnthropicAnswerGenerator(client=client)

    result = generator.generate_answer(
        "Explain authentication",
        context_result,
    )

    assert result["citation_warnings"] == ["S9"]


def test_empty_anthropic_response_is_rejected(context_result):
    """Verify an empty provider response does not become a valid answer."""
    response = SimpleNamespace(
        content=[],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=0,
        ),
    )

    generator = AnthropicAnswerGenerator(
        client=FakeAnthropicClient(response=response)
    )

    with pytest.raises(LLMProviderError, match="empty text response"):
        generator.generate_answer(
            "Explain authentication",
            context_result,
        )


@pytest.mark.parametrize("max_tokens", [0, -1, 1.5, True])
def test_invalid_max_tokens_are_rejected(max_tokens):
    """Verify output token configuration is validated."""
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicAnswerGenerator(
            client=FakeAnthropicClient(),
            max_tokens=max_tokens,
        )


def test_missing_api_key_is_rejected(monkeypatch):
    """Verify production initialization requires an API key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key is missing"):
        AnthropicAnswerGenerator()
