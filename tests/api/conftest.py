"""Shared fixtures for API integration tests.

Uses FastAPI ``TestClient`` with overridden dependencies so every test
exercises real request/response serialisation, route logic, and status
codes without hitting a real database or auth provider.
"""

from __future__ import annotations

from unittest.mock import MagicMock, create_autospec
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.dependencies.auth import get_current_user, get_db
from app.main import app
from app.schemas.auth import UserResponse


@pytest.fixture
def mock_db() -> MagicMock:
    """Return a mock ``Session`` suitable for ``get_db`` overrides."""
    return create_autospec(Session, instance=True)


@pytest.fixture
def test_user() -> UserResponse:
    """Return a fixed ``UserResponse`` that represents the authenticated user."""
    return UserResponse(
        id=str(uuid4()),
        name="Test User",
        email="test@example.com",
    )


@pytest.fixture
def client(mock_db: MagicMock, test_user: UserResponse) -> TestClient:
    """Yield a ``TestClient`` with ``get_db`` and ``get_current_user`` overridden."""

    def _override_get_db() -> MagicMock:
        return mock_db

    def _override_get_current_user() -> UserResponse:
        return test_user

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = _override_get_current_user

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def anon_client(mock_db: MagicMock) -> TestClient:
    """Yield a ``TestClient`` with only ``get_db`` overridden (no auth)."""

    def _override_get_db() -> MagicMock:
        return mock_db

    app.dependency_overrides[get_db] = _override_get_db
    if get_current_user in app.dependency_overrides:
        del app.dependency_overrides[get_current_user]

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
