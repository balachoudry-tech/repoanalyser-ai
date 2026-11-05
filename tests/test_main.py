
"""
Tests for the CodePilot AI FastAPI application.

These tests verify:
    - Application creation.
    - Application startup.
    - Application shutdown.
    - Health endpoint.
    - Dependency initialization.
    - Dependency access through application state.

No actual embedding model, ChromaDB or Anthropic API is initialized.
"""

from fastapi.testclient import TestClient

from src.api.main import create_app, get_container


class FakeContainer:
    """Represent an initialized application dependency container."""

    def __init__(self):
        self.initialized = True


def test_application_creation():
    """Verify the application factory creates a FastAPI instance."""

    app = create_app(
        container_factory=FakeContainer
    )

    assert app.title == "CodePilot AI"
    assert app.version == "0.1.0"


def test_health_endpoint():
    """Verify the health endpoint returns application information."""

    app = create_app(
        container_factory=FakeContainer
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
        "application": "CodePilot AI",
        "version": "0.1.0",
    }


def test_application_dependencies_initialized_on_startup():
    """Verify the dependency container is created during startup."""

    container = FakeContainer()

    app = create_app(
        container_factory=lambda: container
    )

    with TestClient(app):
        assert app.state.container is container
        assert app.state.container.initialized is True


def test_application_dependencies_cleared_on_shutdown():
    """Verify application state is cleared during shutdown."""

    app = create_app(
        container_factory=FakeContainer
    )

    with TestClient(app):
        assert app.state.container is not None

    assert app.state.container is None


def test_container_factory_called_once_per_application_lifespan():
    """Verify dependencies are initialized only once at startup."""

    call_count = 0

    def fake_container_factory():
        nonlocal call_count
        call_count += 1
        return FakeContainer()

    app = create_app(
        container_factory=fake_container_factory
    )

    with TestClient(app) as client:
        client.get("/health")
        client.get("/health")
        client.get("/health")

    assert call_count == 1


def test_container_factory_failure_stops_startup():
    """Verify application startup fails if dependencies cannot initialize."""

    def failing_factory():
        raise RuntimeError("Dependency initialization failed")

    app = create_app(
        container_factory=failing_factory
    )

    try:
        with TestClient(app):
            pass
    except RuntimeError as exc:
        assert "Dependency initialization failed" in str(exc)
    else:
        raise AssertionError(
            "Expected application startup to fail."
        )


def test_get_container_returns_application_container():
    """Verify the dependency accessor returns the initialized container."""

    container = FakeContainer()

    app = create_app(
        container_factory=lambda: container
    )

    with TestClient(app):
        resolved_container = get_container(
            type(
                "FakeRequest",
                (),
                {"app": app},
            )()
        )

        assert resolved_container is container


def test_get_container_raises_when_not_initialized():
    """Verify accessing unavailable dependencies raises an error."""

    app = create_app(
        container_factory=FakeContainer
    )

    try:
        get_container(
            type(
                "FakeRequest",
                (),
                {"app": app},
            )()
        )
    except RuntimeError as exc:
        assert "not been initialized" in str(exc)
    else:
        raise AssertionError(
            "Expected an error when dependencies are unavailable."
        )
