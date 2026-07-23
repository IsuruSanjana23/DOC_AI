"""API tests for the ``/documents`` endpoints.

Covers upload, list, get-by-id, and delete flows including file
validation, ownership checks, and error responses.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.schemas.document import DocumentResponse


NOW = datetime.now(timezone.utc)
COL_ID = str(uuid4())
DOC_ID = str(uuid4())


def _make_doc_response(
    doc_id: str = DOC_ID,
    col_id: str = COL_ID,
    filename: str = "test.pdf",
    status: str = "UPLOADED",
) -> DocumentResponse:
    return DocumentResponse(
        id=doc_id,
        original_filename=filename,
        mime_type="application/pdf",
        file_size=1024,
        status=status,
        collection_id=col_id,
        created_at=NOW,
        updated_at=NOW,
    )


class TestUploadDocument:
    """POST /api/v1/documents/upload"""

    @patch("app.services.document_service.DocumentService.upload")
    def test_201_returns_document(
        self,
        mock_upload: MagicMock,
        client: TestClient,
    ) -> None:
        mock_upload.return_value = _make_doc_response()
        resp = client.post(
            "/api/v1/documents/upload",
            data={"collection_id": COL_ID},
            files={"file": ("test.pdf", BytesIO(b"%PDF-1.4 content"), "application/pdf")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["original_filename"] == "test.pdf"
        assert body["status"] == "UPLOADED"

    @patch("app.services.document_service.DocumentService.upload")
    def test_415_on_non_pdf(
        self,
        mock_upload: MagicMock,
        client: TestClient,
    ) -> None:
        from app.services.exceptions import InvalidFileTypeException

        mock_upload.side_effect = InvalidFileTypeException()
        resp = client.post(
            "/api/v1/documents/upload",
            data={"collection_id": COL_ID},
            files={"file": ("doc.txt", BytesIO(b"hello"), "text/plain")},
        )
        assert resp.status_code == 415
        assert resp.json()["detail"] == "Only PDF files are allowed"

    @patch("app.services.document_service.DocumentService.upload")
    def test_413_on_oversized(
        self,
        mock_upload: MagicMock,
        client: TestClient,
    ) -> None:
        from app.services.exceptions import FileTooLargeException

        mock_upload.side_effect = FileTooLargeException()
        resp = client.post(
            "/api/v1/documents/upload",
            data={"collection_id": COL_ID},
            files={"file": ("big.pdf", BytesIO(b"%PDF-1.4 big " * 10**6), "application/pdf")},
        )
        assert resp.status_code == 413

    @patch("app.services.document_service.DocumentService.upload")
    def test_404_on_missing_collection(
        self,
        mock_upload: MagicMock,
        client: TestClient,
    ) -> None:
        from app.services.exceptions import NotFoundException

        mock_upload.side_effect = NotFoundException()
        resp = client.post(
            "/api/v1/documents/upload",
            data={"collection_id": COL_ID},
            files={"file": ("test.pdf", BytesIO(b"%PDF-1.4 content"), "application/pdf")},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Collection not found"


class TestListDocuments:
    """GET /api/v1/collections/{id}/documents"""

    @patch("app.services.document_service.DocumentService.get_all_by_collection")
    def test_200_returns_list(
        self,
        mock_list: MagicMock,
        client: TestClient,
    ) -> None:
        mock_list.return_value = []
        resp = client.get(f"/api/v1/collections/{COL_ID}/documents")
        assert resp.status_code == 200
        assert resp.json() == []

    @patch("app.services.document_service.DocumentService.get_all_by_collection")
    def test_404_on_missing_collection(
        self,
        mock_list: MagicMock,
        client: TestClient,
    ) -> None:
        from app.services.exceptions import NotFoundException

        mock_list.side_effect = NotFoundException()
        resp = client.get(f"/api/v1/collections/{COL_ID}/documents")
        assert resp.status_code == 404


class TestGetDocument:
    """GET /api/v1/documents/{id}"""

    @patch("app.services.document_service.DocumentService.get_by_id")
    def test_200_returns_document(
        self,
        mock_get: MagicMock,
        client: TestClient,
    ) -> None:
        mock_get.return_value = _make_doc_response(
            filename="report.pdf", status="READY"
        )
        resp = client.get(f"/api/v1/documents/{DOC_ID}")
        assert resp.status_code == 200
        assert resp.json()["original_filename"] == "report.pdf"

    @patch("app.services.document_service.DocumentService.get_by_id")
    def test_404_on_missing(
        self,
        mock_get: MagicMock,
        client: TestClient,
    ) -> None:
        from app.services.exceptions import NotFoundException

        mock_get.side_effect = NotFoundException()
        resp = client.get(f"/api/v1/documents/{uuid4()}")
        assert resp.status_code == 404


class TestDeleteDocument:
    """DELETE /api/v1/documents/{id}"""

    @patch("app.services.document_service.DocumentService.delete")
    def test_204_on_success(
        self,
        mock_delete: MagicMock,
        client: TestClient,
    ) -> None:
        mock_delete.return_value = None
        resp = client.delete(f"/api/v1/documents/{DOC_ID}")
        assert resp.status_code == 204

    @patch("app.services.document_service.DocumentService.delete")
    def test_404_on_missing(
        self,
        mock_delete: MagicMock,
        client: TestClient,
    ) -> None:
        from app.services.exceptions import NotFoundException

        mock_delete.side_effect = NotFoundException()
        resp = client.delete(f"/api/v1/documents/{uuid4()}")
        assert resp.status_code == 404
