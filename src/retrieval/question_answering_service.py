
"""
Question answering orchestration for CodePilot AI.

Coordinates the complete Retrieval-Augmented Generation (RAG) workflow.

Workflow:
    1. Validate the user's question and repository identifier.
    2. Retrieve relevant code chunks.
    3. Build context and source citations.
    4. Generate a grounded answer using the configured LLM.
    5. Return the answer, citations and execution metrics.

Logging:
    - Workflow start and completion.
    - Individual pipeline stage execution.
    - Stage-level execution durations.
    - Retrieval and context statistics.
    - LLM token usage.
    - Citation validation warnings.
    - Workflow exceptions.

Sensitive information such as API keys, raw questions and source code
must never be written to application logs.
"""

import logging
from time import perf_counter
from typing import Any

from src.retrieval.context_builder import ContextBuilder
from src.retrieval.retrieval_service import RetrievalService
from src.llm.answer_generator import AnthropicAnswerGenerator


logger = logging.getLogger(__name__)


class QuestionAnsweringError(Exception):
    """Raised when the question-answering workflow fails."""


class QuestionAnsweringService:
    """
    Orchestrate the CodePilot AI question-answering workflow.

    Dependencies are injected to keep the service independently
    testable and avoid coupling it to concrete implementations.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilder,
        answer_generator: AnthropicAnswerGenerator,
    ) -> None:
        """Initialize the question-answering service."""

        if retrieval_service is None:
            raise ValueError("retrieval_service is required.")

        if context_builder is None:
            raise ValueError("context_builder is required.")

        if answer_generator is None:
            raise ValueError("answer_generator is required.")

        self.retrieval_service = retrieval_service
        self.context_builder = context_builder
        self.answer_generator = answer_generator

        logger.info(
            "QuestionAnsweringService initialized successfully."
        )

    @staticmethod
    def _validate_input(
        question: str,
        repository_id: str,
        top_k: int,
    ) -> None:
        """
        Validate user input before starting the RAG workflow.

        A boolean is rejected for top_k because bool is a subclass
        of int in Python.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")

        if not isinstance(repository_id, str) or not repository_id.strip():
            raise ValueError("repository_id must be a non-empty string.")

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer.")

    def ask(
        self,
        question: str,
        repository_id: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Answer a developer's question about an indexed repository.

        Returns:
            A structured response containing the answer, sources,
            repository information, token usage and execution timings.

        Raises:
            ValueError: If the input is invalid.
            QuestionAnsweringError: If a workflow component fails.
        """

        self._validate_input(question, repository_id, top_k)

        question = question.strip()
        repository_id = repository_id.strip()

        workflow_start = perf_counter()

        logger.info(
            "Question-answering workflow started. "
            "repository_id=%s top_k=%s",
            repository_id,
            top_k,
        )

        try:
            # -------------------------------------------------
            # Stage 1: Retrieve relevant repository code.
            # -------------------------------------------------

            retrieval_start = perf_counter()

            logger.info(
                "Retrieval started. repository_id=%s",
                repository_id,
            )

            retrieval_result = self.retrieval_service.retrieve(
                query=question,
                repository_id=repository_id,
                top_k=top_k,
            )

            retrieval_duration = round(
                (perf_counter() - retrieval_start) * 1000,
                2,
            )

            retrieval_count = retrieval_result.get(
                "result_count",
                len(retrieval_result.get("results", [])),
            )

            logger.info(
                "Retrieval completed. "
                "repository_id=%s results=%s duration_ms=%s",
                repository_id,
                retrieval_count,
                retrieval_duration,
            )

            # -------------------------------------------------
            # Stage 2: Build LLM context and citations.
            # -------------------------------------------------

            context_start = perf_counter()

            logger.info(
                "Context construction started. repository_id=%s",
                repository_id,
            )

            context_result = self.context_builder.build_context(
                retrieval_result=retrieval_result,
            )

            context_duration = round(
                (perf_counter() - context_start) * 1000,
                2,
            )

            logger.info(
                "Context construction completed. "
                "repository_id=%s sources=%s skipped=%s "
                "duration_ms=%s",
                repository_id,
                context_result["source_count"],
                context_result["skipped_count"],
                context_duration,
            )

            # -------------------------------------------------
            # Stage 3: Generate answer using Anthropic Claude.
            # -------------------------------------------------

            llm_start = perf_counter()

            logger.info(
                "LLM generation started. "
                "repository_id=%s model=%s",
                repository_id,
                self.answer_generator.model,
            )

            answer_result = self.answer_generator.generate_answer(
                question=question,
                context_result=context_result,
            )

            llm_duration = round(
                (perf_counter() - llm_start) * 1000,
                2,
            )

            usage = answer_result["usage"]

            logger.info(
                "LLM generation completed. "
                "repository_id=%s model=%s "
                "input_tokens=%s output_tokens=%s "
                "duration_ms=%s",
                repository_id,
                answer_result["model"],
                usage["input_tokens"],
                usage["output_tokens"],
                llm_duration,
            )

            # -------------------------------------------------
            # Stage 4: Citation validation information.
            # -------------------------------------------------

            citation_warnings = answer_result.get(
                "citation_warnings",
                [],
            )

            if citation_warnings:
                logger.warning(
                    "Unknown citation identifiers detected. "
                    "repository_id=%s citation_warnings=%s",
                    repository_id,
                    citation_warnings,
                )

            # -------------------------------------------------
            # Stage 5: Prepare final response.
            # -------------------------------------------------

            total_duration = round(
                (perf_counter() - workflow_start) * 1000,
                2,
            )

            result = {
                "answer": answer_result["answer"],
                "sources": answer_result["sources"],
                "repository_id": repository_id,
                "retrieval_count": retrieval_count,
                "context_source_count": context_result["source_count"],
                "skipped_context_count": context_result["skipped_count"],
                "model": answer_result["model"],
                "usage": usage,
                "timing_ms": {
                    "retrieval": retrieval_duration,
                    "context_building": context_duration,
                    "llm_generation": llm_duration,
                    "total": total_duration,
                },
                "request_id": answer_result.get("request_id"),
                "citation_warnings": citation_warnings,
            }

            logger.info(
                "Question-answering workflow completed successfully. "
                "repository_id=%s total_duration_ms=%s "
                "retrieval_count=%s source_count=%s",
                repository_id,
                total_duration,
                retrieval_count,
                context_result["source_count"],
            )

            return result

        except Exception as exc:
            total_duration = round(
                (perf_counter() - workflow_start) * 1000,
                2,
            )

            logger.exception(
                "Question-answering workflow failed. "
                "repository_id=%s duration_ms=%s error_type=%s",
                repository_id,
                total_duration,
                type(exc).__name__,
            )

            raise QuestionAnsweringError(
                "Unable to answer the repository question."
            ) from exc
