
"""
FastAPI application entry point for CodePilot AI.

Responsibilities:
    - Create the FastAPI application.
    - Initialize application dependencies during startup.
    - Release application references during shutdown.
    - Expose application health information.
    - Provide access to shared application dependencies.

Run locally:
    uvicorn src.api.main:app --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import Callable
from src.api.schemas import HealthResponse
from src.api.dependencies import get_container
from src.api.routes.repositories import router as repositories_router
from fastapi import FastAPI, Request
from src.api.routes.questions import router as questions_router
from pathlib import Path
from fastapi.staticfiles import StaticFiles

from src.api.container import (
    ApplicationContainer,
    build_container,
)


logger = logging.getLogger(__name__)





def create_app(
    container_factory: Callable[[], ApplicationContainer] = build_container,
) -> FastAPI:
    """
    Create and configure the CodePilot AI FastAPI application.

    Args:
        container_factory: Function responsible for constructing
            application dependencies. It can be replaced in tests.

    Returns:
        Configured FastAPI application instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        Manage application startup and shutdown.

        Dependencies are initialized once during startup and remain
        available until the application shuts down.
        """

        logger.info("Starting CodePilot AI application.")

        try:
            app.state.container = container_factory()

            logger.info(
                "Application dependencies are ready."
            )

        except Exception:
            logger.exception(
                "CodePilot AI application startup failed."
            )
            raise

        yield

        logger.info("Shutting down CodePilot AI application.")

        app.state.container = None

        logger.info("CodePilot AI application shutdown completed.")

    app = FastAPI(
        title="CodePilot AI",
        description=(
            "AI-powered codebase documentation and "
            "repository question-answering assistant."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(repositories_router)
    app.include_router(repositories_router)
    app.include_router(questions_router)
    # Serve the CodePilot AI frontend.
    static_directory = Path(__file__).parent / "static"

    
    

    @app.get(
    "/health",
    tags=["Health"],
    response_model=HealthResponse,
    )
    async def health_check():
        """
        Return basic application liveness information.

        This endpoint confirms that the FastAPI application is running.
        It does not independently verify ChromaDB or Anthropic availability.
        """

        return {
            "status": "healthy",
            "application": "CodePilot AI",
            "version": app.version,
        }
    app.mount(
        "/",
        StaticFiles(
            directory=static_directory,
            html=True,
        ),
        name="frontend",
        )

    return app
    


# Application instance used by Uvicorn.
app = create_app()
