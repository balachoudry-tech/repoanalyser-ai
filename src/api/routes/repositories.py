
"""
Repository API routes.

This module exposes repository indexing functionality through HTTP.

Responsibilities:
    - Accept and validate indexing requests.
    - Access the application dependency container.
    - Delegate indexing to IndexingService.
    - Return validated API response models.
    - Translate service errors into HTTP responses.

The router contains no ingestion, parsing, embedding or
vector-storage implementation.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.container import ApplicationContainer
from src.api.dependencies import get_container
from src.api.schemas import (
    IndexRepositoryRequest,
    IndexRepositoryResponse,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/repositories",
    tags=["Repositories"],
)


@router.post(
    "/index",
    response_model=IndexRepositoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Index a source-code repository",
    description=(
        "Scans a local repository, extracts code definitions, "
        "generates embeddings and stores searchable chunks."
    ),
)
def index_repository(
    request: IndexRepositoryRequest,
    container: ApplicationContainer = Depends(get_container),
) -> IndexRepositoryResponse:
    """
    Index a repository using the existing IndexingService.

    Args:
        request: Validated repository indexing request.
        container: Application-level dependency container.

    Returns:
        IndexRepositoryResponse containing indexing statistics.

    Raises:
        HTTPException: When repository validation or indexing fails.
    """

    logger.info(
        "Repository indexing request received: repository_id=%s",
        request.repository_id,
    )

    try:
        result = container.indexing_service.index_repository(
            repository_path=request.repository_path,
            repository_id=request.repository_id,
        )

        response = IndexRepositoryResponse(**result)

        logger.info(
            (
                "Repository indexing completed: "
                "repository_id=%s indexed_chunks=%d"
            ),
            request.repository_id,
            response.indexed_chunks,
        )

        return response

    except FileNotFoundError as exc:
        logger.warning(
            "Repository path not found: repository_id=%s",
            request.repository_id,
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository path was not found.",
        ) from exc

    except PermissionError as exc:
        logger.warning(
            "Repository access denied: repository_id=%s",
            request.repository_id,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied while accessing the repository.",
        ) from exc

    except ValueError as exc:
        logger.warning(
            "Invalid repository indexing request: repository_id=%s",
            request.repository_id,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Repository indexing failed: repository_id=%s",
            request.repository_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during repository indexing.",
        ) from exc
