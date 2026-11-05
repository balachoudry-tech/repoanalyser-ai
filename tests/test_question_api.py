
"""
Integration tests for the repository question-answering API.

The QuestionAnsweringService is replaced with a fake implementation
to test HTTP behavior independently of external dependencies.
"""

import pytest

from fastapi.testclient import TestClient

from src.api.main import create_app


class FakeQuestionAnsweringService:
    """
    Fake question-answering service.

    Records incoming arguments and returns a predictable response.
    """

    def __init__(self):
        """Initialize fake service state."""

        self.calls = []
        self.error = None

    def ask(self, question, repository_id, top_k=5):
        """
        Simulate a question-answering operation.

        Args:
            question: User question.
            repository_id: Target repository.
            top_k: Maximum number of retrieved chunks.

        Returns:
            Structured answer and execution metadata.
        """

        self.calls.append({
            "question": question,
            "repository_id": repository_id,
            "top_k": top_k,
        })

        if self.error:
            raise self.error

        return {
            "answer": (
                "The main function is implemented in "
                "src/main.py [S1]."
            ),
            "sources": [
                {
                    "source_id": "S1",
                    "file_path": "src/main.py",
                    "symbol_name": "main",
                    "start_line": 10,
                    "end_line": 25,
                    "language": "python",
                    "vector_id": "chunk-123",
                    "distance": 0.12,
                }
            ],
            "repository_id": repository_id,
            "retrieval_count": 5,
            "context_source_count": 1,
            "skipped_context_count": 0,
            "model": "claude-test-model",
            "usage": {
                "input_tokens": 532,
                "output_tokens": 494,
            },
            "timing_ms": {
                "retrieval": 150.25,
                "context_building": 8.41,
                "llm_generation": 7840.12,
                "total": 8001.55,
            },
            "request_id": "req-test-123",
            "citation_warnings": [],
        }


class FakeContainer:
    """Fake application container for API tests."""

    def __init__(self):
        """Initialize the fake question-answering service."""

        self.question_answering_service = (
            FakeQuestionAnsweringService()
        )


@pytest.fixture
def api_client():
    """
    Create a FastAPI test client with fake dependencies.

    The application lifespan initializes the fake container.
    """

    container = FakeContainer()

    app = create_app(
        container_factory=lambda: container
    )

    with TestClient(app) as client:
        yield client, container


# ============================================================
# SUCCESS TESTS
# ============================================================


def test_ask_repository_question_success(api_client):
    """Verify successful repository question answering."""

    client, container = api_client

    response = client.post(
        "/api/v1/repositories/sample-repo/ask",
        json={
            "question": "Explain the main function",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["repository_id"] == "sample-repo"
    assert "src/main.py" in body["answer"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source_id"] == "S1"

    assert body["usage"]["input_tokens"] == 532
    assert body["usage"]["output_tokens"] == 494

    assert body["timing_ms"]["retrieval"] == 150.25

    assert len(
        container.question_answering_service.calls
    ) == 1


def test_question_api_passes_correct_arguments(api_client):
    """Verify question parameters reach the service unchanged."""

    client, container = api_client

    client.post(
        "/api/v1/repositories/sample-repo/ask",
        json={
            "question": "Where is authentication implemented?",
            "top_k": 10,
        },
    )

    assert container.question_answering_service.calls == [
        {
            "question": "Where is authentication implemented?",
            "repository_id": "sample-repo",
            "top_k": 10,
        }
    ]


def test_question_api_uses_default_top_k(api_client):
    """Verify the default retrieval count is five."""

    client, container = api_client

    client.post(
        "/api/v1/repositories/sample-repo/ask",
        json={
            "question": "Explain the architecture",
        },
    )

    assert (
        container.question_answering_service.calls[0]["top_k"]
        == 5
    )


# ============================================================
# REQUEST VALIDATION TESTS
# ============================================================


def test_question_api_rejects_empty_question(api_client):
    """Verify empty questions are rejected."""

    client, container = api_client

    response = client.post(
        "/api/v1/repositories/sample-repo/ask",
        json={
            "question": " ",
        },
    )

    assert response.status_code == 422
    assert container.question_answering_service.calls == []


def test_question_api_rejects_missing_question(api_client):
    """Verify question is mandatory."""

    client, container = api_client

    response = client.post(
        "/api/v1/repositories/sample-repo/ask",
        json={},
    )

    assert response.status_code == 422
    assert container.question_answering_service.calls == []


@pytest.mark.parametrize(
    "top_k",
    [0, -1, 21, True],
)
def test_question_api_rejects_invalid_top_k(api_client, top_k):
    """Verify invalid retrieval counts are rejected."""

    client, container = api_client

    response = client.post(
        "/api/v1/repositories/sample-repo/ask",
        json={
            "question": "Explain the architecture",
            "top_k": top_k,
        },
    )

    assert response.status_code == 422
    assert container.question_answering_service.calls == []


def test_question_api_rejects_extra_fields(api_client):
    """Verify unexpected request fields are rejected."""

    client, container = api_client

    response = client.post(
        "/api/v1/repositories/sample-repo/ask",
        json={
            "question": "Explain the architecture",
            "top_k": 5,
            "unexpected": "invalid",
        },
    )

    assert response.status_code == 422
    assert container.question_answering_service.calls == []


# ============================================================
# SERVICE ERROR TESTS
# ============================================================


def test_question_api_handles_invalid_service_input(api_client):
    """Verify service ValueError becomes HTTP 400."""

    client, container = api_client

    container.question_answering_service.error = ValueError(
        "Invalid repository identifier."
    )

    response = client.post(
        "/api/v1/repositories/sample-repo/ask",
        json={
            "question": "Explain the architecture",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid repository identifier."
    )


def test_question_api_handles_service_failure(api_client):
    """Verify unexpected service errors become HTTP 500."""

    client, container = api_client

    container.question_answering_service.error = RuntimeError(
        "Internal provider failure."
    )

    response = client.post(
        "/api/v1/repositories/sample-repo/ask",
        json={
            "question": "Explain the architecture",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Unable to answer the repository question."
    )
