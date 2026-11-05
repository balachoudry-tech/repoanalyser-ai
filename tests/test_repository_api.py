
"""
Integration tests for the repository indexing API.

The indexing service is mocked so that these tests verify
the HTTP layer independently of the actual indexing pipeline.
"""

import pytest

from fastapi.testclient import TestClient

from src.api.main import create_app


class FakeIndexingService:
    """
    Fake indexing service for API tests.

    Records calls and returns predictable indexing statistics.
    """

    def __init__(self):
        """Initialize fake service state."""

        self.calls = []
        self.error = None

    def index_repository(self, repository_path, repository_id):
        """
        Simulate repository indexing.

        Raises:
            Exception: Configured test exception.
        """

        self.calls.append(
            {
                "repository_path": repository_path,
                "repository_id": repository_id,
            }
        )

        if self.error:
            raise self.error

        return {
            "repository_id": repository_id,
            "repository_path": repository_path,
            "total_files": 20,
            "parsed_files": 15,
            "total_code_units": 100,
            "total_chunks": 120,
            "indexed_chunks": 120,
            "status": "completed",
        }


class FakeContainer:
    """Application container containing a fake indexing service."""

    def __init__(self):
        """Create the fake indexing service."""

        self.indexing_service = FakeIndexingService()


@pytest.fixture
def api_client():
    """
    Create a FastAPI test client with fake dependencies.

    The application lifespan initializes the fake container
    instead of loading real ML models or database connections.
    """

    container = FakeContainer()

    app = create_app(
        container_factory=lambda: container
    )

    with TestClient(app) as client:
        yield client, container


# ============================================================
# SUCCESS TESTS
# ============================================================


def test_index_repository_success(api_client):
    """Verify successful repository indexing."""

    client, container = api_client

    response = client.post(
        "/api/v1/repositories/index",
        json={
            "repository_path": "/tmp/my-app",
            "repository_id": "my-app",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["repository_id"] == "my-app"
    assert body["total_files"] == 20
    assert body["parsed_files"] == 15
    assert body["total_chunks"] == 120
    assert body["indexed_chunks"] == 120
    assert body["status"] == "completed"

    assert len(container.indexing_service.calls) == 1


def test_index_repository_passes_correct_arguments(api_client):
    """Verify request values reach IndexingService unchanged."""

    client, container = api_client

    client.post(
        "/api/v1/repositories/index",
        json={
            "repository_path": "/tmp/example",
            "repository_id": "example",
        },
    )

    assert container.indexing_service.calls == [
        {
            "repository_path": "/tmp/example",
            "repository_id": "example",
        }
    ]


# ============================================================
# REQUEST VALIDATION TESTS
# ============================================================


def test_index_repository_rejects_missing_path(api_client):
    """Verify repository_path is mandatory."""

    client, container = api_client

    response = client.post(
        "/api/v1/repositories/index",
        json={
            "repository_id": "my-app",
        },
    )

    assert response.status_code == 422
    assert container.indexing_service.calls == []


def test_index_repository_rejects_missing_repository_id(api_client):
    """Verify repository_id is mandatory."""

    client, container = api_client

    response = client.post(
        "/api/v1/repositories/index",
        json={
            "repository_path": "/tmp/my-app",
        },
    )

    assert response.status_code == 422
    assert container.indexing_service.calls == []


def test_index_repository_rejects_empty_repository_id(api_client):
    """Verify an empty repository identifier is rejected."""

    client, container = api_client

    response = client.post(
        "/api/v1/repositories/index",
        json={
            "repository_path": "/tmp/my-app",
            "repository_id": " ",
        },
    )

    assert response.status_code == 422
    assert container.indexing_service.calls == []


def test_index_repository_rejects_extra_fields(api_client):
    """Verify unsupported request properties are rejected."""

    client, container = api_client

    response = client.post(
        "/api/v1/repositories/index",
        json={
            "repository_path": "/tmp/my-app",
            "repository_id": "my-app",
            "unknown": "invalid",
        },
    )

    assert response.status_code == 422
    assert container.indexing_service.calls == []


# ============================================================
# SERVICE ERROR MAPPING TESTS
# ============================================================


@pytest.mark.parametrize(
    "error,expected_status",
    [
        (FileNotFoundError("missing"), 404),
        (PermissionError("denied"), 403),
        (ValueError("invalid repository"), 400),
        (RuntimeError("unexpected failure"), 500),
    ],
)
def test_index_repository_maps_service_errors(
    api_client,
    error,
    expected_status,
):
    """Verify service exceptions map to appropriate HTTP statuses."""

    client, container = api_client

    container.indexing_service.error = error

    response = client.post(
        "/api/v1/repositories/index",
        json={
            "repository_path": "/tmp/my-app",
            "repository_id": "my-app",
        },
    )

    assert response.status_code == expected_status
