
"""Tests for persistent ChromaDB vector storage."""

import pytest

from src.indexing.vector_store import ChromaVectorStore


@pytest.fixture
def vector_store(tmp_path):
    """Create an isolated ChromaDB database for each test."""
    return ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_code",
    )


def make_chunk(chunk_id, content, file_path, symbol_name):
    """Build a representative code chunk."""
    return {
        "chunk_id": chunk_id,
        "content": content,
        "file_path": file_path,
        "symbol_name": symbol_name,
        "symbol_type": "function",
        "language": "python",
        "start_line": 1,
        "end_line": 5,
        "chunk_index": 0,
        "parent_symbol": None,
    }


def test_add_and_search_chunks(vector_store):
    """Stored code should be retrievable using its embedding."""
    chunks = [
        make_chunk(
            "chunk-1",
            "def create_user(): pass",
            "users.py",
            "create_user",
        )
    ]

    embeddings = [[1.0, 0.0, 0.0]]

    inserted = vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        repository_id="repo-1",
    )

    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        repository_id="repo-1",
    )

    assert inserted == 1
    assert len(results) == 1
    assert results[0]["content"] == "def create_user(): pass"
    assert results[0]["metadata"]["symbol_name"] == "create_user"
    assert results[0]["metadata"]["repository_id"] == "repo-1"
    assert results[0]["distance"] == pytest.approx(0.0, abs=0.001)


def test_repository_isolation(vector_store):
    """Search results must not contain another repository's code."""
    chunks = [
        make_chunk("chunk-1", "create user", "users.py", "create_user"),
        make_chunk("chunk-1", "delete order", "orders.py", "delete_order"),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    vector_store.add_chunks(chunks[:1], embeddings[:1], "repo-1")
    vector_store.add_chunks(chunks[1:], embeddings[1:], "repo-2")

    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        repository_id="repo-1",
    )

    assert len(results) == 1
    assert results[0]["content"] == "create user"
    assert results[0]["metadata"]["repository_id"] == "repo-1"


def test_delete_repository(vector_store):
    """Deleting one repository must leave other repositories intact."""
    chunk = make_chunk(
        "chunk-1", "create user", "users.py", "create_user"
    )

    vector_store.add_chunks(
        [chunk], [[1.0, 0.0, 0.0]], "repo-1"
    )
    vector_store.add_chunks(
        [chunk], [[1.0, 0.0, 0.0]], "repo-2"
    )

    assert vector_store.count() == 2

    vector_store.delete_repository("repo-1")

    assert vector_store.count() == 1

    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        repository_id="repo-2",
    )

    assert len(results) == 1


def test_empty_chunks(vector_store):
    """Adding an empty chunk collection should be a no-op."""
    inserted = vector_store.add_chunks(
        chunks=[],
        embeddings=[],
        repository_id="repo-1",
    )

    assert inserted == 0
    assert vector_store.count() == 0


def test_chunk_embedding_count_mismatch(vector_store):
    """Each chunk must have exactly one corresponding embedding."""
    chunk = make_chunk(
        "chunk-1", "create user", "users.py", "create_user"
    )

    with pytest.raises(ValueError, match="must match"):
        vector_store.add_chunks(
            chunks=[chunk],
            embeddings=[],
            repository_id="repo-1",
        )


def test_empty_query_embedding_rejected(vector_store):
    """A semantic search requires a non-empty query embedding."""
    with pytest.raises(ValueError, match="cannot be empty"):
        vector_store.search(
            query_embedding=[],
            repository_id="repo-1",
        )


def test_invalid_result_count_rejected(vector_store):
    """Search result count must be positive."""
    with pytest.raises(ValueError, match="greater than zero"):
        vector_store.search(
            query_embedding=[1.0, 0.0, 0.0],
            repository_id="repo-1",
            n_results=0,
        )
