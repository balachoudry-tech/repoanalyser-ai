
"""
Embedding generation service for RepoAnalyser AI.

Converts source code chunks and user queries into numerical
vector representations using a Sentence Transformers model.

The generated embeddings are normalized and can be used
for semantic similarity search in a vector database.

The model is loaded once when the service is initialized
and reused for subsequent embedding requests.
"""

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32


class EmbeddingService:
    """
    Generates semantic embeddings for source code and queries.

    Responsibilities:
        1. Load and manage the embedding model.
        2. Generate embeddings for multiple code chunks.
        3. Generate embeddings for individual user queries.
        4. Normalize vectors for cosine similarity search.

    Attributes:
        model: Loaded SentenceTransformer instance.
        model_name: Name of the selected embedding model.
        batch_size: Number of texts processed per batch.
        embedding_dimension: Number of dimensions in each vector.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model=None,
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: Sentence Transformers model identifier.
            batch_size: Number of texts processed in one batch.
            model: Optional preloaded or mock model.

        Dependency injection allows tests to use a fake model
        without downloading the actual embedding model.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.model_name = model_name
        self.batch_size = batch_size

        # Load the model only when one is not injected.
        self.model = model or SentenceTransformer(model_name)

        self.embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple text inputs.

        Args:
            texts: Code chunks or other text documents.

        Returns:
            A list of normalized numerical vectors.

        Raises:
            ValueError: If any input text is empty.
        """

        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip()
               for text in texts):
            raise ValueError("All input texts must be non-empty strings")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a single user query.

        Uses the same embedding model as document indexing
        to ensure both vectors occupy the same vector space.

        Args:
            query: Natural-language question from the user.

        Returns:
            A normalized numerical vector.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        embedding = self.embed_texts([query])

        return embedding[0]
