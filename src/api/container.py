
"""
Application dependency container for CodePilot AI.

This module constructs and owns the shared application services.

Responsibilities:
    - Initialize shared embedding and vector-store components.
    - Initialize repository ingestion and indexing.
    - Initialize semantic retrieval.
    - Initialize context construction and LLM answer generation.
    - Initialize the question-answering orchestration service.

Services are created once and reused across API requests.
"""

import logging
import os

from src.embeddings.embedding_service import EmbeddingService
from src.indexing.chunker import CodeChunker
from src.indexing.indexing_service import IndexingService
from src.indexing.vector_store import ChromaVectorStore
from src.ingestion.ingestion_service import IngestionService
from src.llm.answer_generator import AnthropicAnswerGenerator
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.question_answering_service import (
    QuestionAnsweringService,
)
from src.retrieval.retrieval_service import RetrievalService


logger = logging.getLogger(__name__)


class ApplicationContainer:
    """
    Hold shared services used by the CodePilot AI application.

    The container provides a central location for accessing application
    dependencies instead of constructing them inside individual routes.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
        indexing_service: IndexingService,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilder,
        answer_generator: AnthropicAnswerGenerator,
        question_answering_service: QuestionAnsweringService,
    ) -> None:
        """Initialize the application dependency container."""

        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.indexing_service = indexing_service
        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.answer_generator = answer_generator
        self.question_answering_service = question_answering_service


def build_container() -> ApplicationContainer:
    """
    Construct all application services and their dependencies.

    Shared dependencies such as EmbeddingService and ChromaVectorStore
    are instantiated once and injected into dependent services.

    Returns:
        A fully initialized ApplicationContainer.

    Raises:
        Exception: If an essential application dependency fails to initialize.
    """

    logger.info("Initializing CodePilot AI application dependencies.")

    # ---------------------------------------------------------
    # 1. Initialize shared infrastructure.
    # ---------------------------------------------------------

    logger.info("Initializing embedding service.")

    embedding_service = EmbeddingService()

    logger.info("Initializing ChromaDB vector store.")

    vector_store = ChromaVectorStore(
        persist_directory=os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            "data/chroma",
        ),
        collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME",
            "codepilot_code",
        ),
    )

    # ---------------------------------------------------------
    # 2. Initialize repository indexing.
    # ---------------------------------------------------------

    logger.info("Initializing repository ingestion service.")

    ingestion_service = IngestionService()

    chunker = CodeChunker()

    indexing_service = IndexingService(
        ingestion_service=ingestion_service,
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # ---------------------------------------------------------
    # 3. Initialize retrieval.
    # ---------------------------------------------------------

    logger.info("Initializing retrieval service.")

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # ---------------------------------------------------------
    # 4. Initialize context construction and LLM.
    # ---------------------------------------------------------

    logger.info("Initializing context builder.")

    context_builder = ContextBuilder()

    logger.info("Initializing Anthropic answer generator.")

    answer_generator = AnthropicAnswerGenerator()

    # ---------------------------------------------------------
    # 5. Initialize question-answering orchestration.
    # ---------------------------------------------------------

    question_answering_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        answer_generator=answer_generator,
    )

    # ---------------------------------------------------------
    # 6. Return the application container.
    # ---------------------------------------------------------

    container = ApplicationContainer(
        embedding_service=embedding_service,
        vector_store=vector_store,
        indexing_service=indexing_service,
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        answer_generator=answer_generator,
        question_answering_service=question_answering_service,
    )

    logger.info(
        "CodePilot AI application dependencies initialized successfully."
    )

    return container
