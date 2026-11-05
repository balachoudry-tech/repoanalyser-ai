
"""
Comprehensive unit tests for CodePilot AI API schemas.

Coverage:
    - Repository indexing request validation.
    - Repository indexing response validation.
    - Question request validation.
    - Source citation validation.
    - Token usage validation.
    - Execution timing validation.
    - Question-answering response validation.
    - Health response validation.
    - Nested model serialization.
    - Rejection of unexpected fields.

No external API, database or embedding model is required.
"""

import pytest

from pydantic import ValidationError

from src.api.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
    ExecutionTiming,
    HealthResponse,
    IndexRepositoryRequest,
    IndexRepositoryResponse,
    SourceCitation,
    TokenUsage,
)


# ============================================================
# TEST DATA
# ============================================================


@pytest.fixture
def valid_index_request():
    """Provide a valid repository indexing request."""

    return {
        "repository_path": "/Users/developer/projects/my-app",
        "repository_id": "my-app",
    }


@pytest.fixture
def valid_ask_request():
    """Provide a valid question-answering request."""

    return {
        "question": "Where is authentication implemented?",
        "top_k": 5,
    }


@pytest.fixture
def valid_source():
    """Provide a valid source citation."""

    return {
        "source_id": "S1",
        "file_path": "src/auth/service.py",
        "symbol_name": "authenticate_user",
        "start_line": 10,
        "end_line": 25,
        "language": "python",
        "vector_id": "chunk-123",
        "distance": 0.12,
    }


@pytest.fixture
def valid_ask_response(valid_source):
    """Provide a complete question-answering response."""

    return {
        "answer": (
            "Authentication is implemented in "
            "src/auth/service.py [S1]."
        ),
        "sources": [valid_source],
        "repository_id": "my-app",
        "retrieval_count": 5,
        "context_source_count": 1,
        "skipped_context_count": 0,
        "model": "claude-sonnet-5-5",
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


# ============================================================
# REPOSITORY INDEXING REQUEST TESTS
# ============================================================


def test_valid_index_repository_request(valid_index_request):
    """Verify valid indexing requests are accepted."""

    request = IndexRepositoryRequest(**valid_index_request)

    assert request.repository_id == "my-app"
    assert request.repository_path == (
        "/Users/developer/projects/my-app"
    )


def test_index_request_trims_whitespace():
    """Verify whitespace around indexing fields is removed."""

    request = IndexRepositoryRequest(
        repository_path="  /tmp/my-app  ",
        repository_id="  my-app  ",
    )

    assert request.repository_path == "/tmp/my-app"
    assert request.repository_id == "my-app"


@pytest.mark.parametrize(
    "field",
    ["repository_path", "repository_id"],
)
def test_index_request_rejects_empty_fields(field):
    """Verify required indexing fields cannot be empty."""

    payload = {
        "repository_path": "/tmp/my-app",
        "repository_id": "my-app",
    }

    payload[field] = "   "

    with pytest.raises(ValidationError):
        IndexRepositoryRequest(**payload)


def test_index_request_rejects_missing_repository_path():
    """Verify repository_path is required."""

    with pytest.raises(ValidationError):
        IndexRepositoryRequest(
            repository_id="my-app"
        )


def test_index_request_rejects_extra_fields():
    """Verify unsupported request properties are rejected."""

    with pytest.raises(ValidationError):
        IndexRepositoryRequest(
            repository_path="/tmp/my-app",
            repository_id="my-app",
            unexpected_field="invalid",
        )


# ============================================================
# REPOSITORY INDEXING RESPONSE TESTS
# ============================================================


def test_valid_index_repository_response():
    """Verify valid indexing response data is accepted."""

    response = IndexRepositoryResponse(
        repository_id="my-app",
        repository_path="/tmp/my-app",
        total_files=20,
        parsed_files=15,
        total_code_units=100,
        total_chunks=120,
        indexed_chunks=120,
        status="completed",
    )

    assert response.repository_id == "my-app"
    assert response.total_files == 20
    assert response.indexed_chunks == 120
    assert response.status == "completed"


@pytest.mark.parametrize(
    "field",
    [
        "total_files",
        "parsed_files",
        "total_code_units",
        "total_chunks",
        "indexed_chunks",
    ],
)
def test_index_response_rejects_negative_counts(field):
    """Verify indexing counters cannot be negative."""

    payload = {
        "repository_id": "my-app",
        "repository_path": "/tmp/my-app",
        "total_files": 20,
        "parsed_files": 15,
        "total_code_units": 100,
        "total_chunks": 120,
        "indexed_chunks": 120,
        "status": "completed",
    }

    payload[field] = -1

    with pytest.raises(ValidationError):
        IndexRepositoryResponse(**payload)


# ============================================================
# QUESTION REQUEST TESTS
# ============================================================


def test_valid_question_request(valid_ask_request):
    """Verify valid question requests are accepted."""

    request = AskQuestionRequest(**valid_ask_request)

    assert request.question == (
        "Where is authentication implemented?"
    )
    assert request.top_k == 5


def test_question_request_uses_default_top_k():
    """Verify top_k defaults to five."""

    request = AskQuestionRequest(
        question="Explain the application architecture."
    )

    assert request.top_k == 5


def test_question_request_trims_whitespace():
    """Verify whitespace around questions is removed."""

    request = AskQuestionRequest(
        question="  Explain authentication  "
    )

    assert request.question == "Explain authentication"


@pytest.mark.parametrize(
    "question",
    ["", " ", "\t", "\n"],
)
def test_question_request_rejects_empty_question(question):
    """Verify empty questions are rejected."""

    with pytest.raises(ValidationError):
        AskQuestionRequest(question=question)


def test_question_request_rejects_missing_question():
    """Verify question is mandatory."""

    with pytest.raises(ValidationError):
        AskQuestionRequest()


@pytest.mark.parametrize(
    "top_k",
    [0, -1, 21],
)
def test_question_request_rejects_invalid_top_k(top_k):
    """Verify top_k stays within the configured range."""

    with pytest.raises(ValidationError):
        AskQuestionRequest(
            question="Explain authentication",
            top_k=top_k,
        )


def test_question_request_rejects_boolean_top_k():
    """Verify boolean values are not accepted as integers."""

    with pytest.raises(ValidationError):
        AskQuestionRequest(
            question="Explain authentication",
            top_k=True,
        )


def test_question_request_rejects_extra_fields():
    """Verify unsupported question parameters are rejected."""

    with pytest.raises(ValidationError):
        AskQuestionRequest(
            question="Explain authentication",
            top_k=5,
            repository_id="my-app",
        )


# ============================================================
# SOURCE CITATION TESTS
# ============================================================


def test_valid_source_citation(valid_source):
    """Verify source citation metadata is accepted."""

    source = SourceCitation(**valid_source)

    assert source.source_id == "S1"
    assert source.file_path == "src/auth/service.py"
    assert source.start_line == 10
    assert source.end_line == 25


def test_source_citation_supports_optional_fields():
    """Verify optional source metadata can be omitted."""

    source = SourceCitation(
        source_id="S1",
        file_path="src/main.py",
    )

    assert source.symbol_name is None
    assert source.start_line is None
    assert source.end_line is None
    assert source.language is None
    assert source.vector_id is None
    assert source.distance is None


def test_source_citation_rejects_missing_file_path():
    """Verify file_path is required."""

    with pytest.raises(ValidationError):
        SourceCitation(
            source_id="S1"
        )


def test_source_citation_rejects_invalid_line_number():
    """Verify source line numbers must be positive."""

    with pytest.raises(ValidationError):
        SourceCitation(
            source_id="S1",
            file_path="src/main.py",
            start_line=0,
        )


def test_source_citation_rejects_extra_fields():
    """Verify unsupported citation metadata is rejected."""

    with pytest.raises(ValidationError):
        SourceCitation(
            source_id="S1",
            file_path="src/main.py",
            unknown_field="invalid",
        )


# ============================================================
# TOKEN USAGE TESTS
# ============================================================


def test_valid_token_usage():
    """Verify valid token usage is accepted."""

    usage = TokenUsage(
        input_tokens=532,
        output_tokens=494,
    )

    assert usage.input_tokens == 532
    assert usage.output_tokens == 494


@pytest.mark.parametrize(
    "field",
    ["input_tokens", "output_tokens"],
)
def test_token_usage_rejects_negative_values(field):
    """Verify token counts cannot be negative."""

    payload = {
        "input_tokens": 100,
        "output_tokens": 50,
    }

    payload[field] = -1

    with pytest.raises(ValidationError):
        TokenUsage(**payload)


# ============================================================
# EXECUTION TIMING TESTS
# ============================================================


def test_valid_execution_timing():
    """Verify execution timings are accepted."""

    timing = ExecutionTiming(
        retrieval=150.25,
        context_building=8.41,
        llm_generation=7840.12,
        total=8001.55,
    )

    assert timing.retrieval == 150.25
    assert timing.context_building == 8.41
    assert timing.llm_generation == 7840.12
    assert timing.total == 8001.55


@pytest.mark.parametrize(
    "field",
    [
        "retrieval",
        "context_building",
        "llm_generation",
        "total",
    ],
)
def test_execution_timing_rejects_negative_values(field):
    """Verify execution durations cannot be negative."""

    payload = {
        "retrieval": 100,
        "context_building": 10,
        "llm_generation": 1000,
        "total": 1110,
    }

    payload[field] = -1

    with pytest.raises(ValidationError):
        ExecutionTiming(**payload)


# ============================================================
# QUESTION ANSWERING RESPONSE TESTS
# ============================================================


def test_valid_question_answering_response(valid_ask_response):
    """Verify the complete answer response is accepted."""

    response = AskQuestionResponse(**valid_ask_response)

    assert response.repository_id == "my-app"
    assert response.model == "claude-sonnet-5-5"

    assert len(response.sources) == 1
    assert response.sources[0].source_id == "S1"

    assert response.usage.input_tokens == 532
    assert response.usage.output_tokens == 494

    assert response.timing_ms.llm_generation == 7840.12

    assert response.request_id == "req-test-123"
    assert response.citation_warnings == []


def test_question_response_supports_empty_sources():
    """Verify answers can be returned without source citations."""

    response = AskQuestionResponse(
        answer="No relevant source code was found.",
        repository_id="my-app",
        retrieval_count=0,
        context_source_count=0,
        skipped_context_count=0,
        model="claude-test-model",
        usage={
            "input_tokens": 0,
            "output_tokens": 0,
        },
        timing_ms={
            "retrieval": 10,
            "context_building": 1,
            "llm_generation": 0,
            "total": 11,
        },
    )

    assert response.sources == []
    assert response.request_id is None
    assert response.citation_warnings == []


def test_question_response_rejects_invalid_nested_usage():
    """Verify nested token usage is validated."""

    payload = {
        "answer": "Authentication is implemented here.",
        "repository_id": "my-app",
        "retrieval_count": 1,
        "context_source_count": 1,
        "skipped_context_count": 0,
        "model": "claude-test-model",
        "usage": {
            "input_tokens": -10,
            "output_tokens": 20,
        },
        "timing_ms": {
            "retrieval": 10,
            "context_building": 1,
            "llm_generation": 100,
            "total": 111,
        },
    }

    with pytest.raises(ValidationError):
        AskQuestionResponse(**payload)


def test_question_response_rejects_invalid_nested_timing():
    """Verify nested execution timings are validated."""

    payload = {
        "answer": "Authentication is implemented here.",
        "repository_id": "my-app",
        "retrieval_count": 1,
        "context_source_count": 1,
        "skipped_context_count": 0,
        "model": "claude-test-model",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
        },
        "timing_ms": {
            "retrieval": -10,
            "context_building": 1,
            "llm_generation": 100,
            "total": 111,
        },
    }

    with pytest.raises(ValidationError):
        AskQuestionResponse(**payload)


def test_question_response_serializes_nested_models(
    valid_ask_response,
):
    """Verify nested schemas serialize into ordinary dictionaries."""

    response = AskQuestionResponse(**valid_ask_response)

    serialized = response.model_dump()

    assert isinstance(serialized, dict)
    assert isinstance(serialized["usage"], dict)
    assert isinstance(serialized["timing_ms"], dict)
    assert isinstance(serialized["sources"], list)

    assert serialized["sources"][0]["source_id"] == "S1"


# ============================================================
# HEALTH RESPONSE TESTS
# ============================================================


def test_valid_health_response():
    """Verify health response structure."""

    response = HealthResponse(
        status="healthy",
        application="CodePilot AI",
        version="0.1.0",
    )

    assert response.status == "healthy"
    assert response.application == "CodePilot AI"
    assert response.version == "0.1.0"


def test_health_response_rejects_extra_fields():
    """Verify unsupported health response fields are rejected."""

    with pytest.raises(ValidationError):
        HealthResponse(
            status="healthy",
            application="CodePilot AI",
            version="0.1.0",
            unexpected="invalid",
        )
