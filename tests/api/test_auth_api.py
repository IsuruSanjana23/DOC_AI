"""API tests for the ``/auth`` endpoints.

Covers register, login, logout, and me flows including success,
validation errors, duplicate email, and invalid credentials.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.services.exceptions import CredentialsException


class TestRegister:
    """POST /auth/register"""

    @patch("app.services.auth_service.UserRepository.exists_by_email")
    def test_201_on_success(
        self, mock_exists: MagicMock, client: TestClient, mock_db: MagicMock
    ) -> None:
        mock_exists.return_value = False
        resp = client.post(
            "/api/v1/auth/register",
            json={"name": "Alice", "email": "a@b.com", "password": "s3cret"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "id" in body
        assert body["name"] == "Alice"
        assert body["email"] == "a@b.com"

    @patch("app.services.auth_service.UserRepository.exists_by_email")
    def test_409_on_duplicate_email(
        self, mock_exists: MagicMock, client: TestClient, mock_db: MagicMock
    ) -> None:
        mock_exists.return_value = True
        resp = client.post(
            "/api/v1/auth/register",
            json={"name": "Bob", "email": "dup@b.com", "password": "s3cret"},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"] == "A user with this email already exists"

    def test_422_on_missing_fields(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/register", json={})
        assert resp.status_code == 422

    def test_422_on_invalid_email(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/auth/register",
            json={"name": "X", "email": "not-an-email", "password": "x"},
        )
        assert resp.status_code == 422


class TestLogin:
    """POST /api/v1/auth/login"""

    @patch("app.services.auth_service.verify_password")
    @patch("app.services.auth_service.UserRepository.get_by_email")
    def test_200_returns_access_token(
        self,
        mock_get_by_email: MagicMock,
        mock_verify: MagicMock,
        client: TestClient,
        mock_db: MagicMock,
    ) -> None:
        from app.models.user import User

        fake_user = MagicMock(spec=User)
        fake_user.id = "user-uuid"
        mock_get_by_email.return_value = fake_user
        mock_verify.return_value = True

        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "a@b.com", "password": "s3cret"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    def test_401_on_invalid_credentials(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        from app.services.auth_service import AuthService

        original_login = AuthService.login
        AuthService.login = lambda self, email, password: (  # type: ignore[method-assign]
            (_ for _ in ()).throw(CredentialsException())
        )

        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "wrong@b.com", "password": "bad"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid email or password"
        AuthService.login = original_login

    def test_422_on_missing_fields(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/login", json={})
        assert resp.status_code == 422


class TestLogout:
    """POST /api/v1/auth/logout"""

    def test_200_always_succeeds(self, client: TestClient) -> None:
        resp = client.post("/api/v1/auth/logout")
        assert resp.status_code == 200
        assert resp.json()["message"] == "Successfully logged out"


class TestMe:
    """GET /api/v1/auth/me"""

    def test_200_returns_current_user(
        self, client: TestClient, test_user
    ) -> None:
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == test_user.id
        assert body["name"] == test_user.name
        assert body["email"] == test_user.email

    def test_401_without_token(self, anon_client: TestClient) -> None:
        resp = anon_client.get("/api/v1/auth/me")
        assert resp.status_code in (401, 403)
