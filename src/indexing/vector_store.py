
"""
Persistent vector storage for CodePilot AI.

Stores code chunks and their embeddings in ChromaDB so that
repository code can be retrieved using semantic similarity.
"""

import hashlib
from pathlib import Path
from typing import Any

import chromadb


class ChromaVectorStore:
    """
    Manage persistent code embeddings using ChromaDB.

    Each chunk is stored with its source content and metadata.
    A repository_id is attached to every record so searches
    and deletion operations can be isolated by repository.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "codepilot_code",
    ) -> None:
        """
        Initialize a persistent ChromaDB client and collection.

        Args:
            persist_directory: Local directory where ChromaDB
                stores its database files.
            collection_name: Name of the ChromaDB collection.
        """
        if not collection_name.strip():
            raise ValueError("Collection name cannot be empty.")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _build_metadata(
        chunk: dict[str, Any],
        repository_id: str,
    ) -> dict[str, str | int | float | bool]:
        """
        Build metadata containing only ChromaDB-supported values.

        None values and complex objects are excluded because
        ChromaDB metadata accepts scalar values.
        """
        metadata: dict[str, Any] = {
            "repository_id": repository_id,
        }

        allowed_fields = [
            "symbol_name",
            "symbol_type",
            "parent_symbol",
            "file_path",
            "language",
            "start_line",
            "end_line",
            "chunk_index",
        ]

        for field in allowed_fields:
            value = chunk.get(field)

            if value is not None and isinstance(
                value, (str, int, float, bool)
            ):
                metadata[field] = value

        return metadata

    @staticmethod
    def _build_vector_id(
        repository_id: str,
        chunk_id: str,
    ) -> str:
        """
        Generate a deterministic, repository-specific vector ID.

        This avoids collisions when two repositories contain
        identical chunks with identical chunk IDs.
        """
        raw_id = f"{repository_id}:{chunk_id}"
        return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        repository_id: str,
    ) -> int:
        """
        Insert or update code chunks and embeddings.

        Args:
            chunks: Code chunks produced by CodeChunker.
            embeddings: Embeddings produced by EmbeddingService.
            repository_id: Unique identifier for the repository.

        Returns:
            Number of chunks inserted or updated.

        Raises:
            ValueError: If inputs are invalid or do not align.
        """
        if not repository_id or not repository_id.strip():
            raise ValueError("Repository ID cannot be empty.")

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            content = chunk.get("content")

            if not chunk_id:
                raise ValueError("Every chunk must have a chunk_id.")

            if not isinstance(content, str) or not content.strip():
                raise ValueError("Every chunk must have non-empty content.")

            ids.append(
                self._build_vector_id(repository_id, str(chunk_id))
            )
            documents.append(content)
            metadatas.append(
                self._build_metadata(chunk, repository_id)
            )

        for embedding in embeddings:
            if not embedding:
                raise ValueError("Embeddings cannot be empty.")

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        repository_id: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Find code chunks semantically similar to a query embedding.

        Results are restricted to the requested repository.

        Args:
            query_embedding: Embedding generated from the user's question.
            repository_id: Repository to search within.
            n_results: Maximum number of results to return.

        Returns:
            A list of matching chunks, metadata and cosine distances.
            Lower cosine distance means greater similarity.
        """
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        if not repository_id or not repository_id.strip():
            raise ValueError("Repository ID cannot be empty.")

        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"repository_id": repository_id},
            include=["documents", "metadatas", "distances"],
        )

        matched_results = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for index, vector_id in enumerate(ids):
            matched_results.append(
                {
                    "id": vector_id,
                    "content": documents[index],
                    "metadata": metadatas[index],
                    "distance": distances[index],
                }
            )

        return matched_results

    def delete_repository(self, repository_id: str) -> None:
        """
        Delete all indexed chunks belonging to one repository.

        Other repositories in the same collection are unaffected.
        """
        if not repository_id or not repository_id.strip():
            raise ValueError("Repository ID cannot be empty.")

        self.collection.delete(
            where={"repository_id": repository_id}
        )

    def count(self) -> int:
        """Return the total number of indexed chunks."""
        return self.collection.count()
