
"""
Coordinate repository ingestion, chunking, embedding and indexing.

This module provides the main entry point for indexing a repository
in CodePilot AI.
"""

from pathlib import Path
from typing import Any

from src.ingestion.ingestion_service import IngestionService
from src.indexing.chunker import CodeChunker
from src.indexing.vector_store import ChromaVectorStore
from src.embeddings.embedding_service import EmbeddingService


class IndexingService:
    """
    Orchestrate the complete repository indexing pipeline.

    Dependencies are injected so that each component can be tested
    independently and replaced without changing this service.
    """

    def __init__(
        self,
        ingestion_service: IngestionService | None = None,
        chunker: CodeChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        """
        Initialize the indexing pipeline.

        Optional dependencies support dependency injection in tests.
        """
        self.ingestion_service = (
            ingestion_service or IngestionService()
        )
        self.chunker = chunker or CodeChunker()
        self.embedding_service = (
            embedding_service or EmbeddingService()
        )
        self.vector_store = vector_store or ChromaVectorStore()

    def index_repository(
        self,
        repository_path: str,
        repository_id: str,
    ) -> dict[str, Any]:
        """
        Parse, chunk, embed and persist a repository's code.

        Args:
            repository_path: Local path to the repository.
            repository_id: Unique identifier used to isolate its data.

        Returns:
            A summary of the indexing operation.

        Raises:
            ValueError: If the repository path or ID is invalid.
            Exception: If an indexing component fails.
        """
        if not repository_id or not repository_id.strip():
            raise ValueError("Repository ID cannot be empty.")

        if not Path(repository_path).is_dir():
            raise ValueError(
                f"Repository directory does not exist: {repository_path}"
            )

        # Step 1: Read and parse repository files.
        ingestion_result = (
            self.ingestion_service.ingest_repository(repository_path)
        )

        code_units = ingestion_result["code_units"]

        # Step 2: Split parsed code into manageable chunks.
        chunks = self.chunker.chunk_code_units(code_units)

        # An empty or unsupported repository has nothing to index.
        if not chunks:
            return {
                "repository_id": repository_id,
                "repository_path": str(Path(repository_path).resolve()),
                "total_files": ingestion_result["total_files"],
                "parsed_files": ingestion_result["parsed_files"],
                "total_code_units": ingestion_result["total_code_units"],
                "total_chunks": 0,
                "indexed_chunks": 0,
                "status": "completed",
            }

        # Step 3: Convert all chunks into embedding vectors.
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        # Step 4: Persist code, embeddings and metadata.
        indexed_count = self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            repository_id=repository_id,
        )

        return {
            "repository_id": repository_id,
            "repository_path": str(Path(repository_path).resolve()),
            "total_files": ingestion_result["total_files"],
            "parsed_files": ingestion_result["parsed_files"],
            "total_code_units": ingestion_result["total_code_units"],
            "total_chunks": len(chunks),
            "indexed_chunks": indexed_count,
            "status": "completed",
        }
