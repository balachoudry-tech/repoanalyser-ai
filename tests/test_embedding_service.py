
"""
Unit tests for the embedding generation service.

Uses a fake embedding model to validate service behaviour
without downloading or loading the real machine-learning model.
"""


from src.embeddings.embedding_service import EmbeddingService


class FakeEmbeddingModel:
    """
    Lightweight fake model for testing.

    Returns deterministic vectors so tests can validate
    embedding generation without external model dependencies.
    """

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(
        self,
        texts,
        batch_size,
        show_progress_bar,
        convert_to_numpy,
        normalize_embeddings,
    ):
        """Return deterministic three-dimensional vectors."""

        return FakeEmbeddingArray([
            [0.1, 0.2, 0.3]
            for _ in texts
        ])


class FakeEmbeddingArray:
    """Mimics the tolist() behaviour of a NumPy array."""

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def test_embedding_service_initialization():
    """Verify model injection and dimension detection."""

    service = EmbeddingService(model=FakeEmbeddingModel())

    assert service.embedding_dimension == 3


def test_embed_multiple_texts():
    """Verify that multiple texts produce multiple vectors."""

    service = EmbeddingService(model=FakeEmbeddingModel())

    embeddings = service.embed_texts([
        "User authentication",
        "Database connection",
    ])

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 3


def test_embed_single_query():
    """Verify that a single query produces one vector."""

    service = EmbeddingService(model=FakeEmbeddingModel())

    embedding = service.embed_query(
        "Where is authentication implemented?"
    )

    assert len(embedding) == 3


def test_empty_text_collection():
    """Verify that empty input returns an empty collection."""

    service = EmbeddingService(model=FakeEmbeddingModel())

    assert service.embed_texts([]) == []


def test_empty_query_rejected():
    """Verify that empty user queries are rejected."""

    service = EmbeddingService(model=FakeEmbeddingModel())

    try:
        service.embed_query("")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_batch_size():
    """Verify that invalid batch sizes are rejected."""

    try:
        EmbeddingService(
            batch_size=0,
            model=FakeEmbeddingModel(),
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_empty_document_rejected():
    """Verify that blank document content is rejected."""

    service = EmbeddingService(model=FakeEmbeddingModel())

    try:
        service.embed_texts(["valid text", ""])
        assert False, "Expected ValueError"
    except ValueError:
        pass
