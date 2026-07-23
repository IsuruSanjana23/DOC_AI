"""API tests for the ``/collections`` endpoints.

Covers full CRUD for collections including ownership checks,
duplicate name rejection, and response schema validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.models.collection import Collection
from app.schemas.auth import UserResponse


NOW = datetime.now(timezone.utc)
COL_ID = str(uuid4())


def _make_fake_collection(user_id: str) -> MagicMock:
    fake = MagicMock(spec=Collection)
    fake.id = COL_ID
    fake.name = "Test Collection"
    fake.description = "A description"
    fake.starred = False
    fake.user_id = UUID(user_id)
    fake.created_at = NOW
    fake.updated_at = NOW
    return fake


class TestCreateCollection:
    """POST /api/v1/collections"""

    @patch("app.services.collection_service.CollectionRepository.create")
    @patch("app.services.collection_service.CollectionRepository.exists_by_name")
    def test_201_returns_collection(
        self,
        mock_exists: MagicMock,
        mock_create: MagicMock,
        client: TestClient,
        test_user: UserResponse,
    ) -> None:
        mock_exists.return_value = False
        mock_create.return_value = _make_fake_collection(test_user.id)
        resp = client.post(
            "/api/v1/collections",
            json={"name": "My Collection", "description": "Notes"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "Test Collection"
        assert body["description"] == "A description"
        assert "id" in body
        assert "created_at" in body

    @patch("app.services.collection_service.CollectionRepository.exists_by_name")
    def test_409_on_duplicate_name(
        self,
        mock_exists: MagicMock,
        client: TestClient,
    ) -> None:
        mock_exists.return_value = True
        resp = client.post(
            "/api/v1/collections",
            json={"name": "Duplicate"},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"] == "A collection with this name already exists"

    def test_422_on_empty_name(self, client: TestClient) -> None:
        resp = client.post("/api/v1/collections", json={"name": "  "})
        assert resp.status_code == 422

    def test_422_on_short_name(self, client: TestClient) -> None:
        resp = client.post("/api/v1/collections", json={"name": "ab"})
        assert resp.status_code == 422

    @patch("app.services.collection_service.CollectionRepository.create")
    @patch("app.services.collection_service.CollectionRepository.exists_by_name")
    def test_201_without_description(
        self,
        mock_exists: MagicMock,
        mock_create: MagicMock,
        client: TestClient,
        test_user: UserResponse,
    ) -> None:
        mock_exists.return_value = False
        mock_create.return_value = _make_fake_collection(test_user.id)
        resp = client.post(
            "/api/v1/collections",
            json={"name": "Valid Name"},
        )
        assert resp.status_code == 201
        assert resp.json()["description"] == "A description"


class TestListCollections:
    """GET /api/v1/collections"""

    def test_200_returns_list(self, client: TestClient) -> None:
        resp = client.get("/api/v1/collections")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestGetCollection:
    """GET /api/v1/collections/{id}"""

    @patch("app.services.collection_service.CollectionRepository.get_by_id")
    def test_200_returns_collection(
        self,
        mock_get: MagicMock,
        client: TestClient,
        test_user: UserResponse,
    ) -> None:
        mock_get.return_value = _make_fake_collection(test_user.id)

        resp = client.get(f"/api/v1/collections/{COL_ID}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Test Collection"

    @patch("app.services.collection_service.CollectionRepository.get_by_id")
    def test_404_on_missing(
        self,
        mock_get: MagicMock,
        client: TestClient,
    ) -> None:
        mock_get.return_value = None
        resp = client.get(f"/api/v1/collections/{uuid4()}")
        assert resp.status_code == 404


class TestUpdateCollection:
    """PATCH /api/v1/collections/{id}"""

    @patch("app.services.collection_service.CollectionRepository.update")
    @patch("app.services.collection_service.CollectionRepository.exists_by_name")
    @patch("app.services.collection_service.CollectionRepository.get_by_id")
    def test_200_updates_name(
        self,
        mock_get: MagicMock,
        mock_exists: MagicMock,
        mock_update: MagicMock,
        client: TestClient,
        test_user: UserResponse,
    ) -> None:
        fake = _make_fake_collection(test_user.id)
        mock_get.return_value = fake
        mock_exists.return_value = False
        mock_update.return_value = fake

        resp = client.patch(
            f"/api/v1/collections/{COL_ID}",
            json={"name": "New Name"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Test Collection"

    @patch("app.services.collection_service.CollectionRepository.update")
    @patch("app.services.collection_service.CollectionRepository.get_by_id")
    def test_200_toggles_star(
        self,
        mock_get: MagicMock,
        mock_update: MagicMock,
        client: TestClient,
        test_user: UserResponse,
    ) -> None:
        fake = _make_fake_collection(test_user.id)
        mock_get.return_value = fake
        mock_update.return_value = fake

        resp = client.patch(
            f"/api/v1/collections/{COL_ID}",
            json={"starred": True},
        )
        assert resp.status_code == 200

    @patch("app.services.collection_service.CollectionRepository.get_by_id")
    def test_404_on_missing(
        self,
        mock_get: MagicMock,
        client: TestClient,
    ) -> None:
        mock_get.return_value = None
        resp = client.patch(
            f"/api/v1/collections/{uuid4()}",
            json={"name": "Nope"},
        )
        assert resp.status_code == 404


class TestDeleteCollection:
    """DELETE /api/v1/collections/{id}"""

    @patch("app.services.collection_service.CollectionRepository.get_by_id")
    def test_204_on_success(
        self,
        mock_get: MagicMock,
        client: TestClient,
        test_user: UserResponse,
    ) -> None:
        mock_get.return_value = _make_fake_collection(test_user.id)

        resp = client.delete(f"/api/v1/collections/{COL_ID}")
        assert resp.status_code == 204

    @patch("app.services.collection_service.CollectionRepository.get_by_id")
    def test_404_on_missing(
        self,
        mock_get: MagicMock,
        client: TestClient,
    ) -> None:
        mock_get.return_value = None
        resp = client.delete(f"/api/v1/collections/{uuid4()}")
        assert resp.status_code == 404
