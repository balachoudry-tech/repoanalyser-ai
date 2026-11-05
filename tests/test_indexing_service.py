
"""Unit tests for the repository indexing pipeline."""

import pytest

from src.indexing.indexing_service import IndexingService


class FakeIngestionService:
    """Return controlled repository ingestion results."""

    def ingest_repository(self, repository_path):
        return {
            "repository": "test-repository",
            "repository_path": repository_path,
            "total_files": 2,
            "parsed_files": 2,
            "unsupported_files": 0,
            "failed_files": 0,
            "total_code_units": 2,
            "code_units": [
                {"symbol_name": "create_user", "content": "def create_user(): pass"},
                {"symbol_name": "delete_user", "content": "def delete_user(): pass"},
            ],
            "errors": [],
        }


class FakeChunker:
    """Return controlled code chunks."""

    def chunk_code_units(self, code_units):
        return [
            {
                "chunk_id": "chunk-1",
                "content": "def create_user(): pass",
            },
            {
                "chunk_id": "chunk-2",
                "content": "def delete_user(): pass",
            },
        ]


class FakeEmbeddingService:
    """Generate predictable vectors for tests."""

    def embed_texts(self, texts):
        return [[1.0, 0.0], [0.0, 1.0]]


class FakeVectorStore:
    """Record chunks sent for persistence."""

    def __init__(self):
        self.saved_chunks = []
        self.saved_repository_id = None

    def add_chunks(self, chunks, embeddings, repository_id):
        self.saved_chunks = chunks
        self.saved_repository_id = repository_id
        assert len(chunks) == len(embeddings)
        return len(chunks)


@pytest.fixture
def indexing_service():
    """Create an indexing service with fake dependencies."""
    return IndexingService(
        ingestion_service=FakeIngestionService(),
        chunker=FakeChunker(),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
    )


def test_index_repository_success(indexing_service, tmp_path):
    """A repository should pass through the entire pipeline."""
    result = indexing_service.index_repository(
        repository_path=str(tmp_path),
        repository_id="repo-1",
    )

    assert result["status"] == "completed"
    assert result["total_files"] == 2
    assert result["parsed_files"] == 2
    assert result["total_code_units"] == 2
    assert result["total_chunks"] == 2
    assert result["indexed_chunks"] == 2


def test_index_repository_empty_id(indexing_service, tmp_path):
    """An empty repository ID must be rejected."""
    with pytest.raises(ValueError, match="Repository ID"):
        indexing_service.index_repository(
            repository_path=str(tmp_path),
            repository_id="",
        )


def test_index_repository_invalid_path(indexing_service, tmp_path):
    """A nonexistent repository path must be rejected."""
    invalid_path = tmp_path / "does-not-exist"

    with pytest.raises(ValueError, match="does not exist"):
        indexing_service.index_repository(
            repository_path=str(invalid_path),
            repository_id="repo-1",
        )


def test_index_repository_with_no_chunks(tmp_path):
    """An empty repository should complete without embedding or storage."""

    class EmptyIngestionService:
        def ingest_repository(self, repository_path):
            return {
                "total_files": 0,
                "parsed_files": 0,
                "total_code_units": 0,
                "code_units": [],
            }

    class EmptyChunker:
        def chunk_code_units(self, code_units):
            return []

    class ShouldNotRunEmbedding:
        def embed_texts(self, texts):
            raise AssertionError("Embedding should not be called")

    class ShouldNotRunVectorStore:
        def add_chunks(self, chunks, embeddings, repository_id):
            raise AssertionError("Vector storage should not be called")

    service = IndexingService(
        ingestion_service=EmptyIngestionService(),
        chunker=EmptyChunker(),
        embedding_service=ShouldNotRunEmbedding(),
        vector_store=ShouldNotRunVectorStore(),
    )

    result = service.index_repository(
        repository_path=str(tmp_path),
        repository_id="empty-repo",
    )

    assert result["status"] == "completed"
    assert result["total_chunks"] == 0
    assert result["indexed_chunks"] == 0
