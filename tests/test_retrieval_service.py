
"""Unit tests for CodePilot AI semantic retrieval."""

import pytest

from src.retrieval.retrieval_service import RetrievalService


class FakeEmbeddingService:
    """Return a predictable query embedding."""

    def __init__(self):
        self.received_query = None

    def embed_query(self, query):
        self.received_query = query
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    """Return controlled semantic search results."""

    def __init__(self):
        self.received_embedding = None
        self.received_repository_id = None
        self.received_top_k = None

    def search(self, query_embedding, repository_id, n_results):
        self.received_embedding = query_embedding
        self.received_repository_id = repository_id
        self.received_top_k = n_results

        return [
            {
                "content": "def authenticate_user(): pass",
                "metadata": {
                    "file_path": "src/auth/service.py",
                    "symbol_name": "authenticate_user",
                    "start_line": 10,
                    "end_line": 12,
                },
                "distance": 0.15,
            }
        ]


@pytest.fixture
def retrieval_service():
    """Create a retrieval service with fake dependencies."""
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()

    service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    return service, embedding_service, vector_store


def test_retrieve_success(retrieval_service):
    """A valid query should return relevant code chunks."""
    service, embedding_service, vector_store = retrieval_service

    result = service.retrieve(
        query="Where is user authentication implemented?",
        repository_id="repo-1",
        top_k=5,
    )

    assert result["repository_id"] == "repo-1"
    assert result["result_count"] == 1

    assert embedding_service.received_query == (
        "Where is user authentication implemented?"
    )

    assert vector_store.received_embedding == [0.1, 0.2, 0.3]
    assert vector_store.received_repository_id == "repo-1"
    assert vector_store.received_top_k == 5

    assert result["results"][0]["metadata"]["file_path"] == (
        "src/auth/service.py"
    )


def test_retrieve_with_custom_top_k(retrieval_service):
    """The caller should be able to customize the result count."""
    service, _, vector_store = retrieval_service

    service.retrieve(
        query="Find database connection logic",
        repository_id="repo-1",
        top_k=10,
    )

    assert vector_store.received_top_k == 10


def test_empty_query_rejected(retrieval_service):
    """An empty natural-language query should be rejected."""
    service, _, _ = retrieval_service

    with pytest.raises(ValueError, match="Query cannot be empty"):
        service.retrieve(
            query=" ",
            repository_id="repo-1",
        )


def test_empty_repository_id_rejected(retrieval_service):
    """A repository identifier is mandatory."""
    service, _, _ = retrieval_service

    with pytest.raises(ValueError, match="Repository ID cannot be empty"):
        service.retrieve(
            query="Find authentication",
            repository_id="",
        )


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k_rejected(retrieval_service, top_k):
    """The number of requested results must be a positive integer."""
    service, _, _ = retrieval_service

    with pytest.raises(ValueError, match="top_k must be"):
        service.retrieve(
            query="Find authentication",
            repository_id="repo-1",
            top_k=top_k,
        )


def test_no_matching_results(retrieval_service):
    """A valid query may return no matching code."""
    service, _, vector_store = retrieval_service

    vector_store.search = lambda **kwargs: []

    result = service.retrieve(
        query="Find an unknown implementation",
        repository_id="repo-1",
    )

    assert result["result_count"] == 0
    assert result["results"] == []
