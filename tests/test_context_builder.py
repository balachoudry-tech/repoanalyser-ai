
"""Tests for ContextBuilder."""

import pytest

from src.retrieval.context_builder import ContextBuilder


@pytest.fixture
def retrieval_result():
    """Provide representative semantic retrieval results."""
    return {
        "query": "Where is authentication implemented?",
        "repository_id": "repo-1",
        "result_count": 2,
        "results": [
            {
                "id": "vector-1",
                "content": "def authenticate_user(): pass",
                "metadata": {
                    "file_path": "src/auth/service.py",
                    "symbol_name": "authenticate_user",
                    "start_line": 10,
                    "end_line": 12,
                    "language": "python",
                },
                "distance": 0.12,
            },
            {
                "id": "vector-2",
                "content": "def verify_password(): pass",
                "metadata": {
                    "file_path": "src/auth/security.py",
                    "symbol_name": "verify_password",
                    "start_line": 20,
                    "end_line": 25,
                    "language": "python",
                },
                "distance": 0.18,
            },
        ],
    }


def test_build_context_success(retrieval_result):
    """Context should contain all relevant code and metadata."""
    builder = ContextBuilder()

    result = builder.build_context(retrieval_result)

    assert result["source_count"] == 2
    assert result["skipped_count"] == 0
    assert "[S1]" in result["context"]
    assert "[S2]" in result["context"]
    assert "src/auth/service.py" in result["context"]
    assert "src/auth/security.py" in result["context"]


def test_source_metadata_is_preserved(retrieval_result):
    """Citation metadata should preserve original source locations."""
    builder = ContextBuilder()

    result = builder.build_context(retrieval_result)

    first_source = result["sources"][0]

    assert first_source["source_id"] == "S1"
    assert first_source["file_path"] == "src/auth/service.py"
    assert first_source["symbol_name"] == "authenticate_user"
    assert first_source["start_line"] == 10
    assert first_source["end_line"] == 12


def test_duplicate_chunks_are_removed(retrieval_result):
    """Repeated vector IDs should only appear once."""
    retrieval_result["results"].append(
        retrieval_result["results"][0]
    )

    builder = ContextBuilder()
    result = builder.build_context(retrieval_result)

    assert result["source_count"] == 2
    assert result["context"].count("[S1]") == 1


def test_empty_retrieval_results():
    """An empty retrieval result should return a safe message."""
    builder = ContextBuilder()

    result = builder.build_context({"results": []})

    assert result["source_count"] == 0
    assert result["sources"] == []
    assert "No relevant source code" in result["context"]


def test_context_size_limit(retrieval_result):
    """Chunks exceeding the context budget should be skipped."""
    builder = ContextBuilder(max_context_chars=100)

    result = builder.build_context(retrieval_result)

    assert result["source_count"] == 0
    assert result["skipped_count"] == 2
    assert result["truncated"] is True


def test_invalid_context_size():
    """Context size must be a positive integer."""
    with pytest.raises(ValueError, match="max_context_chars"):
        ContextBuilder(max_context_chars=0)


def test_invalid_retrieval_result():
    """Invalid retrieval input should be rejected."""
    builder = ContextBuilder()

    with pytest.raises(ValueError, match="must be a dictionary"):
        builder.build_context([])


def test_invalid_results_type():
    """Retrieval results must be supplied as a list."""
    builder = ContextBuilder()

    with pytest.raises(ValueError, match="must be a list"):
        builder.build_context({"results": None})


def test_empty_code_chunk_is_skipped(retrieval_result):
    """Empty code content should not be sent to the LLM."""
    retrieval_result["results"].append({
        "id": "vector-3",
        "content": "   ",
        "metadata": {"file_path": "empty.py"},
        "distance": 0.2,
    })

    builder = ContextBuilder()
    result = builder.build_context(retrieval_result)

    assert result["source_count"] == 2
    assert result["skipped_count"] == 1
