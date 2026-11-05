
"""
Semantic retrieval service for CodePilot AI.

Converts developer questions into embeddings and retrieves
relevant code chunks from the ChromaDB vector store.
"""

from typing import Any

from src.embeddings.embedding_service import EmbeddingService
from src.indexing.vector_store import ChromaVectorStore


class RetrievalService:
    """
    Coordinate semantic search across an indexed repository.

    This service does not generate answers. It retrieves relevant
    source-code evidence that a downstream LLM can use.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        """
        Initialize retrieval dependencies.

        Args:
            embedding_service: Converts natural-language questions
                into numerical vectors.
            vector_store: Searches previously indexed code chunks.

        Dependencies are injectable to support isolated unit tests.
        """
        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else EmbeddingService()
        )

        self.vector_store = (
            vector_store
            if vector_store is not None
            else ChromaVectorStore()
        )

    def retrieve(
        self,
        query: str,
        repository_id: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Retrieve code chunks relevant to a developer's question.

        Workflow:
            1. Validate the question and repository identifier.
            2. Generate an embedding for the question.
            3. Search ChromaDB within the selected repository.
            4. Return matching code chunks and their metadata.

        Args:
            query: Natural-language question from the developer.
            repository_id: Unique identifier of the indexed repository.
            top_k: Maximum number of matching chunks to retrieve.

        Returns:
            A dictionary containing the original query, repository ID,
            result count and matching code chunks.

        Raises:
            ValueError: If query, repository ID or top_k is invalid.
        """
        # Step 1: Validate user input.
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not isinstance(repository_id, str) or not repository_id.strip():
            raise ValueError("Repository ID cannot be empty.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        # Step 2: Convert the question into a numerical vector.
        query_embedding = self.embedding_service.embed_query(query)

        # Step 3: Search only within the requested repository.
        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            repository_id=repository_id,
            n_results=top_k,
        )

        # Step 4: Return the evidence to the calling component.
        return {
            "query": query,
            "repository_id": repository_id,
            "result_count": len(retrieved_chunks),
            "results": retrieved_chunks,
        }
