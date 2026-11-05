
"""
Comprehensive unit tests for QuestionAnsweringService.

Coverage:
    1. Successful RAG workflow.
    2. Dependency injection.
    3. Input validation.
    4. Retrieval orchestration.
    5. Context construction orchestration.
    6. LLM generation orchestration.
    7. Empty retrieval handling.
    8. Error handling and exception propagation.
    9. Execution timing.
    10. Logging and observability.

All external dependencies are mocked.
No ChromaDB, embedding model or Anthropic API calls are performed.
"""

import logging

import pytest

from src.retrieval.question_answering_service import (
    QuestionAnsweringError,
    QuestionAnsweringService,
)


# ============================================================
# FAKE DEPENDENCIES
# ============================================================


class FakeRetrievalService:
    """Simulate RetrievalService without accessing ChromaDB."""

    def __init__(self, result=None, error=None):
        self.result = result or {
            "query": "Where is authentication implemented?",
            "repository_id": "repo-123",
            "result_count": 1,
            "results": [
                {
                    "id": "chunk-1",
                    "content": "def authenticate_user(): pass",
                    "metadata": {
                        "file_path": "src/auth/service.py",
                    },
                    "distance": 0.12,
                }
            ],
        }

        self.error = error
        self.calls = []

    def retrieve(self, query, repository_id, top_k=5):
        """Capture retrieval arguments and return simulated results."""

        self.calls.append(
            {
                "query": query,
                "repository_id": repository_id,
                "top_k": top_k,
            }
        )

        if self.error:
            raise self.error

        return self.result


class FakeContextBuilder:
    """Simulate ContextBuilder without building real LLM context."""

    def __init__(self, result=None, error=None):
        self.result = result or {
            "context": "[S1] src/auth/service.py",
            "sources": [
                {
                    "source_id": "S1",
                    "file_path": "src/auth/service.py",
                    "symbol_name": "authenticate_user",
                    "start_line": 10,
                    "end_line": 15,
                    "language": "python",
                }
            ],
            "source_count": 1,
            "skipped_count": 0,
            "truncated": False,
        }

        self.error = error
        self.calls = []

    def build_context(self, retrieval_result):
        """Capture retrieval output and return simulated context."""

        self.calls.append(retrieval_result)

        if self.error:
            raise self.error

        return self.result


class FakeAnswerGenerator:
    """Simulate AnthropicAnswerGenerator without API requests."""

    def __init__(self, result=None, error=None):
        self.model = "claude-test-model"

        self.result = result or {
            "answer": (
                "Authentication is implemented in "
                "src/auth/service.py [S1]."
            ),
            "sources": [
                {
                    "source_id": "S1",
                    "file_path": "src/auth/service.py",
                    "symbol_name": "authenticate_user",
                }
            ],
            "model": self.model,
            "usage": {
                "input_tokens": 200,
                "output_tokens": 50,
            },
            "request_id": "req-test-123",
            "citation_warnings": [],
        }

        self.error = error
        self.calls = []

    def generate_answer(self, question, context_result):
        """Capture the question and context and return a fake answer."""

        self.calls.append(
            {
                "question": question,
                "context_result": context_result,
            }
        )

        if self.error:
            raise self.error

        return self.result


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def dependencies():
    """Create a QuestionAnsweringService with fake dependencies."""

    retrieval = FakeRetrievalService()
    context_builder = FakeContextBuilder()
    answer_generator = FakeAnswerGenerator()

    service = QuestionAnsweringService(
        retrieval_service=retrieval,
        context_builder=context_builder,
        answer_generator=answer_generator,
    )

    return service, retrieval, context_builder, answer_generator


@pytest.fixture
def empty_retrieval_dependencies():
    """Create dependencies representing no matching repository code."""

    retrieval = FakeRetrievalService(
        result={
            "query": "Unknown question",
            "repository_id": "repo-123",
            "result_count": 0,
            "results": [],
        }
    )

    context_builder = FakeContextBuilder(
        result={
            "context": "No relevant source code was found.",
            "sources": [],
            "source_count": 0,
            "skipped_count": 0,
            "truncated": False,
        }
    )

    answer_generator = FakeAnswerGenerator(
        result={
            "answer": "No relevant source code was found.",
            "sources": [],
            "model": "claude-test-model",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "request_id": None,
            "citation_warnings": [],
        }
    )

    service = QuestionAnsweringService(
        retrieval_service=retrieval,
        context_builder=context_builder,
        answer_generator=answer_generator,
    )

    return service, retrieval, context_builder, answer_generator


# ============================================================
# SUCCESSFUL WORKFLOW TESTS
# ============================================================


def test_question_answering_success(dependencies):
    """Verify the complete RAG workflow."""

    service, retrieval, context_builder, answer_generator = dependencies

    result = service.ask(
        question="Where is authentication implemented?",
        repository_id="repo-123",
    )

    assert "src/auth/service.py" in result["answer"]

    assert result["repository_id"] == "repo-123"
    assert result["retrieval_count"] == 1
    assert result["context_source_count"] == 1
    assert result["skipped_context_count"] == 0

    assert result["model"] == "claude-test-model"

    assert result["usage"]["input_tokens"] == 200
    assert result["usage"]["output_tokens"] == 50

    assert result["request_id"] == "req-test-123"
    assert result["citation_warnings"] == []

    assert len(result["sources"]) == 1

    assert result["timing_ms"]["retrieval"] >= 0
    assert result["timing_ms"]["context_building"] >= 0
    assert result["timing_ms"]["llm_generation"] >= 0
    assert result["timing_ms"]["total"] >= 0

    assert len(retrieval.calls) == 1
    assert len(context_builder.calls) == 1
    assert len(answer_generator.calls) == 1


def test_custom_top_k_is_passed_to_retrieval(dependencies):
    """Verify custom top_k reaches RetrievalService."""

    service, retrieval, _, _ = dependencies

    service.ask(
        question="Explain authentication",
        repository_id="repo-123",
        top_k=10,
    )

    assert retrieval.calls[0]["top_k"] == 10


def test_question_and_repository_are_trimmed(dependencies):
    """Verify surrounding whitespace is removed."""

    service, retrieval, _, answer_generator = dependencies

    service.ask(
        question="  Explain authentication  ",
        repository_id="  repo-123  ",
    )

    assert retrieval.calls[0]["query"] == "Explain authentication"

    assert retrieval.calls[0]["repository_id"] == "repo-123"

    assert (
        answer_generator.calls[0]["question"]
        == "Explain authentication"
    )


def test_retrieval_result_is_passed_to_context_builder(dependencies):
    """Verify retrieval output is passed to ContextBuilder."""

    service, retrieval, context_builder, _ = dependencies

    service.ask(
        question="Explain authentication",
        repository_id="repo-123",
    )

    assert context_builder.calls[0] is retrieval.result


def test_context_result_is_passed_to_answer_generator(dependencies):
    """Verify prepared context reaches the answer generator."""

    service, _, context_builder, answer_generator = dependencies

    service.ask(
        question="Explain authentication",
        repository_id="repo-123",
    )

    assert (
        answer_generator.calls[0]["context_result"]
        is context_builder.result
    )


# ============================================================
# INPUT VALIDATION TESTS
# ============================================================


@pytest.mark.parametrize(
    "question",
    ["", " ", None, 123],
)
def test_invalid_question_is_rejected(dependencies, question):
    """Verify invalid questions are rejected before execution."""

    service, retrieval, _, _ = dependencies

    with pytest.raises(ValueError, match="question"):
        service.ask(question, "repo-123")

    assert retrieval.calls == []


@pytest.mark.parametrize(
    "repository_id",
    ["", " ", None, 123],
)
def test_invalid_repository_id_is_rejected(
    dependencies,
    repository_id,
):
    """Verify invalid repository identifiers are rejected."""

    service, retrieval, _, _ = dependencies

    with pytest.raises(ValueError, match="repository_id"):
        service.ask(
            "Explain authentication",
            repository_id,
        )

    assert retrieval.calls == []


@pytest.mark.parametrize(
    "top_k",
    [0, -1, 1.5, True],
)
def test_invalid_top_k_is_rejected(dependencies, top_k):
    """Verify top_k must be a positive integer."""

    service, retrieval, _, _ = dependencies

    with pytest.raises(ValueError, match="top_k"):
        service.ask(
            "Explain authentication",
            "repo-123",
            top_k=top_k,
        )

    assert retrieval.calls == []


@pytest.mark.parametrize(
    "dependency_name",
    [
        "retrieval_service",
        "context_builder",
        "answer_generator",
    ],
)
def test_required_dependencies_are_validated(dependency_name):
    """Verify constructor rejects missing dependencies."""

    dependencies = {
        "retrieval_service": FakeRetrievalService(),
        "context_builder": FakeContextBuilder(),
        "answer_generator": FakeAnswerGenerator(),
    }

    dependencies[dependency_name] = None

    with pytest.raises(ValueError):
        QuestionAnsweringService(**dependencies)


# ============================================================
# EMPTY RETRIEVAL TESTS
# ============================================================


def test_empty_retrieval_result_is_handled(
    empty_retrieval_dependencies,
):
    """Verify the workflow handles an empty retrieval result."""

    service, retrieval, context_builder, answer_generator = (
        empty_retrieval_dependencies
    )

    result = service.ask(
        "Unknown question",
        "repo-123",
    )

    assert result["retrieval_count"] == 0
    assert result["context_source_count"] == 0
    assert result["sources"] == []

    assert result["usage"]["input_tokens"] == 0
    assert result["usage"]["output_tokens"] == 0

    assert len(retrieval.calls) == 1
    assert len(context_builder.calls) == 1
    assert len(answer_generator.calls) == 1


# ============================================================
# ERROR HANDLING TESTS
# ============================================================


def test_retrieval_failure_is_wrapped():
    """Verify retrieval failures become workflow-level errors."""

    service = QuestionAnsweringService(
        retrieval_service=FakeRetrievalService(
            error=RuntimeError("Database unavailable")
        ),
        context_builder=FakeContextBuilder(),
        answer_generator=FakeAnswerGenerator(),
    )

    with pytest.raises(QuestionAnsweringError) as exc_info:
        service.ask(
            "Explain authentication",
            "repo-123",
        )

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )


def test_context_builder_failure_is_wrapped():
    """Verify context construction failures are handled."""

    service = QuestionAnsweringService(
        retrieval_service=FakeRetrievalService(),
        context_builder=FakeContextBuilder(
            error=RuntimeError("Context construction failed")
        ),
        answer_generator=FakeAnswerGenerator(),
    )

    with pytest.raises(QuestionAnsweringError):
        service.ask(
            "Explain authentication",
            "repo-123",
        )


def test_answer_generator_failure_is_wrapped():
    """Verify LLM failures are handled."""

    service = QuestionAnsweringService(
        retrieval_service=FakeRetrievalService(),
        context_builder=FakeContextBuilder(),
        answer_generator=FakeAnswerGenerator(
            error=RuntimeError("LLM unavailable")
        ),
    )

    with pytest.raises(QuestionAnsweringError):
        service.ask(
            "Explain authentication",
            "repo-123",
        )


# ============================================================
# LOGGING TESTS
# ============================================================


def test_service_initialization_is_logged(caplog):
    """Verify service initialization generates an INFO log."""

    with caplog.at_level(logging.INFO):
        QuestionAnsweringService(
            retrieval_service=FakeRetrievalService(),
            context_builder=FakeContextBuilder(),
            answer_generator=FakeAnswerGenerator(),
        )

    assert any(
        "QuestionAnsweringService initialized successfully"
        in record.message
        for record in caplog.records
    )


def test_workflow_start_and_completion_are_logged(
    dependencies,
    caplog,
):
    """Verify workflow start and successful completion are logged."""

    service, _, _, _ = dependencies

    with caplog.at_level(logging.INFO):
        service.ask(
            "Explain authentication",
            "repo-123",
        )

    messages = [record.message for record in caplog.records]

    assert any(
        "Question-answering workflow started" in message
        for message in messages
    )

    assert any(
        "Question-answering workflow completed successfully"
        in message
        for message in messages
    )


def test_pipeline_stages_are_logged(dependencies, caplog):
    """Verify retrieval, context and LLM stages are logged."""

    service, _, _, _ = dependencies

    with caplog.at_level(logging.INFO):
        service.ask(
            "Explain authentication",
            "repo-123",
        )

    messages = [record.message for record in caplog.records]

    assert any("Retrieval started" in message for message in messages)
    assert any("Retrieval completed" in message for message in messages)

    assert any(
        "Context construction started" in message
        for message in messages
    )

    assert any(
        "Context construction completed" in message
        for message in messages
    )

    assert any("LLM generation started" in message for message in messages)
    assert any("LLM generation completed" in message for message in messages)


def test_workflow_failure_is_logged(caplog):
    """Verify workflow exceptions generate ERROR logs with traceback."""

    service = QuestionAnsweringService(
        retrieval_service=FakeRetrievalService(
            error=RuntimeError("Database unavailable")
        ),
        context_builder=FakeContextBuilder(),
        answer_generator=FakeAnswerGenerator(),
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(QuestionAnsweringError):
            service.ask(
                "Explain authentication",
                "repo-123",
            )

    error_records = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR
    ]

    assert len(error_records) >= 1

    assert any(
        "Question-answering workflow failed" in record.message
        for record in error_records
    )

    assert any(
        record.exc_info is not None
        for record in error_records
    )


def test_unknown_citations_generate_warning(dependencies, caplog):
    """Verify unknown citation IDs generate WARNING logs."""

    service, _, _, answer_generator = dependencies

    answer_generator.result["citation_warnings"] = ["S9"]

    with caplog.at_level(logging.WARNING):
        result = service.ask(
            "Explain authentication",
            "repo-123",
        )

    assert result["citation_warnings"] == ["S9"]

    assert any(
        "Unknown citation identifiers detected" in record.message
        for record in caplog.records
    )


# ============================================================
# OBSERVABILITY TESTS
# ============================================================


def test_execution_timings_are_returned(dependencies):
    """Verify all pipeline execution timings are returned."""

    service, _, _, _ = dependencies

    result = service.ask(
        "Explain authentication",
        "repo-123",
    )

    timing = result["timing_ms"]

    assert isinstance(timing, dict)

    assert "retrieval" in timing
    assert "context_building" in timing
    assert "llm_generation" in timing
    assert "total" in timing

    for duration in timing.values():
        assert isinstance(duration, (int, float))
        assert duration >= 0


def test_token_usage_is_returned(dependencies):
    """Verify LLM token usage is included in the response."""

    service, _, _, _ = dependencies

    result = service.ask(
        "Explain authentication",
        "repo-123",
    )

    assert result["usage"] == {
        "input_tokens": 200,
        "output_tokens": 50,
    }


def test_request_id_is_returned(dependencies):
    """Verify Anthropic request ID is included in the response."""

    service, _, _, _ = dependencies

    result = service.ask(
        "Explain authentication",
        "repo-123",
    )

    assert result["request_id"] == "req-test-123"


def test_citation_warnings_are_returned(dependencies):
    """Verify citation validation warnings reach the caller."""

    service, _, _, answer_generator = dependencies

    answer_generator.result["citation_warnings"] = ["S9", "S10"]

    result = service.ask(
        "Explain authentication",
        "repo-123",
    )

    assert result["citation_warnings"] == ["S9", "S10"]


# ============================================================
# RESPONSE STRUCTURE TEST
# ============================================================


def test_response_contains_required_fields(dependencies):
    """Verify the API-facing service response has all expected fields."""

    service, _, _, _ = dependencies

    result = service.ask(
        "Explain authentication",
        "repo-123",
    )

    expected_fields = {
        "answer",
        "sources",
        "repository_id",
        "retrieval_count",
        "context_source_count",
        "skipped_context_count",
        "model",
        "usage",
        "timing_ms",
        "request_id",
        "citation_warnings",
    }

    assert set(result.keys()) == expected_fields
