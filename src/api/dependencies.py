
"""
Shared FastAPI dependencies.

This module provides access to application-level dependencies
initialized during the FastAPI lifespan.
"""

from fastapi import Request

from src.api.container import ApplicationContainer


def get_container(request: Request) -> ApplicationContainer:
    """
    Retrieve the application dependency container.

    Args:
        request: Incoming FastAPI request.

    Returns:
        ApplicationContainer containing initialized services.

    Raises:
        RuntimeError: If the application container is unavailable.
    """

    container = getattr(request.app.state, "container", None)

    if container is None:
        raise RuntimeError(
            "Application dependencies have not been initialized."
        )

    return container
