
"""
Question-answering API routes for CodePilot AI.

This module exposes repository-specific question answering
through HTTP.

Responsibilities:
    - Validate incoming questions.
    - Resolve the application dependency container.
    - Delegate question answering to QuestionAnsweringService.
    - Return structured answers and source citations.
    - Map service errors to HTTP responses.

The actual RAG pipeline is implemented in the service layer.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.container import ApplicationContainer
from src.api.dependencies import get_container
from src.api.schemas import (
    AskQuestionRequest,
    AskQuestionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/repositories",
    tags=["Question Answering"],
)


@router.post(
    "/{repository_id}/ask",
    response_model=AskQuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about a repository",
    description=(
        "Uses semantic code retrieval and an LLM to answer "
        "questions about an indexed repository with source citations."
    ),
)
def ask_repository_question(
    repository_id: str,
    request: AskQuestionRequest,
    container: ApplicationContainer = Depends(get_container),
) -> AskQuestionResponse:
    """
    Answer a natural-language question about an indexed repository.

    Args:
        repository_id: Unique identifier of the indexed repository.
        request: Validated question and retrieval configuration.
        container: Initialized application dependency container.

    Returns:
        Structured answer containing citations, usage and timings.

    Raises:
        HTTPException: If the question cannot be processed.
    """

    logger.info(
        "Repository question received: repository_id=%s",
        repository_id,
    )

    try:
        result = container.question_answering_service.ask(
            question=request.question,
            repository_id=repository_id,
            top_k=request.top_k,
        )

        response = AskQuestionResponse(**result)

        logger.info(
            (
                "Repository question answered: "
                "repository_id=%s sources=%d"
            ),
            repository_id,
            response.context_source_count,
        )

        return response

    except ValueError as exc:
        logger.warning(
            "Invalid question request: repository_id=%s",
            repository_id,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Question answering failed: repository_id=%s",
            repository_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to answer the repository question.",
        ) from exc
