from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch
from uuid import UUID, uuid4

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentStatus
from app.rag.embedder import EmbeddedChunk
from app.rag.processor import DocumentProcessingError
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.exceptions import ExtractionError, NotFoundException
from app.services.processing_service import ProcessingService


@pytest.fixture
def db() -> MagicMock:
    return create_autospec(Session, instance=True)


@pytest.fixture
def doc_id() -> UUID:
    return uuid4()


@pytest.fixture
def document(doc_id: UUID) -> Document:
    doc = MagicMock(spec=Document)
    doc.id = doc_id
    doc.storage_path = "user/coll/doc.pdf"
    doc.original_filename = "test.pdf"
    return doc


@pytest.fixture
def embedded_chunks() -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(0, "Chunk one", 1, np.array([0.1, 0.2])),
        EmbeddedChunk(1, "Chunk two", 1, np.array([0.3, 0.4])),
    ]


class TestProcessingService:

    @patch("app.services.processing_service.DocumentProcessor")
    @patch("app.services.processing_service.ChunkRepository")
    @patch("app.services.processing_service.DocumentRepository")
    def test_process_document_happy_path(
        self,
        mock_doc_repo_cls: MagicMock,
        mock_chunk_repo_cls: MagicMock,
        mock_processor_cls: MagicMock,
        db: MagicMock,
        doc_id: UUID,
        document: Document,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        mock_doc_repo = mock_doc_repo_cls.return_value
        mock_doc_repo.get_by_id.return_value = document
        mock_chunk_repo = mock_chunk_repo_cls.return_value
        mock_processor = mock_processor_cls.return_value
        mock_processor.process.return_value = embedded_chunks

        service = ProcessingService(db)
        result = service.process_document(doc_id)

        assert result == embedded_chunks
        mock_doc_repo.get_by_id.assert_called_once_with(doc_id)
        mock_processor.process.assert_called_once()
        mock_doc_repo.save_text_content.assert_called_once()
        mock_chunk_repo.save_chunks.assert_called_once_with(
            doc_id, embedded_chunks,
        )
        assert db.commit.call_count == 2

    @patch("app.services.processing_service.DocumentProcessor")
    @patch("app.services.processing_service.ChunkRepository")
    @patch("app.services.processing_service.DocumentRepository")
    def test_process_document_saves_text_content(
        self,
        mock_doc_repo_cls: MagicMock,
        mock_chunk_repo_cls: MagicMock,
        mock_processor_cls: MagicMock,
        db: MagicMock,
        doc_id: UUID,
        document: Document,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        mock_doc_repo = mock_doc_repo_cls.return_value
        mock_doc_repo.get_by_id.return_value = document
        mock_processor = mock_processor_cls.return_value
        mock_processor.process.return_value = embedded_chunks

        service = ProcessingService(db)
        service.process_document(doc_id)

        text_arg = mock_doc_repo.save_text_content.call_args[0][1]
        assert "Chunk one" in text_arg
        assert "Chunk two" in text_arg

    @patch("app.services.processing_service.DocumentProcessor")
    @patch("app.services.processing_service.ChunkRepository")
    @patch("app.services.processing_service.DocumentRepository")
    def test_process_document_updates_status_to_processing(
        self,
        mock_doc_repo_cls: MagicMock,
        mock_chunk_repo_cls: MagicMock,
        mock_processor_cls: MagicMock,
        db: MagicMock,
        doc_id: UUID,
        document: Document,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        mock_doc_repo = mock_doc_repo_cls.return_value
        mock_doc_repo.get_by_id.return_value = document
        mock_processor = mock_processor_cls.return_value
        mock_processor.process.return_value = embedded_chunks

        service = ProcessingService(db)
        service.process_document(doc_id)

        status_calls = [
            c.args[1]
            for c in mock_doc_repo.update_status.call_args_list
        ]
        assert DocumentStatus.PROCESSING in status_calls
        assert DocumentStatus.READY in status_calls

    @patch("app.services.processing_service.DocumentProcessor")
    @patch("app.services.processing_service.ChunkRepository")
    @patch("app.services.processing_service.DocumentRepository")
    def test_process_document_raises_on_missing_doc(
        self,
        mock_doc_repo_cls: MagicMock,
        mock_chunk_repo_cls: MagicMock,
        mock_processor_cls: MagicMock,
        db: MagicMock,
        doc_id: UUID,
    ) -> None:
        mock_doc_repo = mock_doc_repo_cls.return_value
        mock_doc_repo.get_by_id.return_value = None

        service = ProcessingService(db)

        with pytest.raises(NotFoundException):
            service.process_document(doc_id)

    @patch("app.services.processing_service.DocumentProcessor")
    @patch("app.services.processing_service.ChunkRepository")
    @patch("app.services.processing_service.DocumentRepository")
    def test_process_document_sets_failed_on_processor_error(
        self,
        mock_doc_repo_cls: MagicMock,
        mock_chunk_repo_cls: MagicMock,
        mock_processor_cls: MagicMock,
        db: MagicMock,
        doc_id: UUID,
        document: Document,
    ) -> None:
        mock_doc_repo = mock_doc_repo_cls.return_value
        mock_doc_repo.get_by_id.return_value = document
        mock_processor = mock_processor_cls.return_value
        mock_processor.process.side_effect = DocumentProcessingError("boom")

        service = ProcessingService(db)

        with pytest.raises(ExtractionError, match="boom"):
            service.process_document(doc_id)

        mock_doc_repo.update_status.assert_any_call(
            document, DocumentStatus.FAILED,
        )

    @patch("app.services.processing_service.DocumentProcessor")
    @patch("app.services.processing_service.ChunkRepository")
    @patch("app.services.processing_service.DocumentRepository")
    def test_process_document_sets_failed_on_unexpected_error(
        self,
        mock_doc_repo_cls: MagicMock,
        mock_chunk_repo_cls: MagicMock,
        mock_processor_cls: MagicMock,
        db: MagicMock,
        doc_id: UUID,
        document: Document,
    ) -> None:
        mock_doc_repo = mock_doc_repo_cls.return_value
        mock_doc_repo.get_by_id.return_value = document
        mock_processor = mock_processor_cls.return_value
        mock_processor.process.side_effect = RuntimeError("disk full")

        service = ProcessingService(db)

        with pytest.raises(ExtractionError, match="disk full"):
            service.process_document(doc_id)

        mock_doc_repo.update_status.assert_any_call(
            document, DocumentStatus.FAILED,
        )

    @patch("app.services.processing_service.DocumentProcessor")
    @patch("app.services.processing_service.ChunkRepository")
    @patch("app.services.processing_service.DocumentRepository")
    def test_process_document_commits_after_failure_status(
        self,
        mock_doc_repo_cls: MagicMock,
        mock_chunk_repo_cls: MagicMock,
        mock_processor_cls: MagicMock,
        db: MagicMock,
        doc_id: UUID,
        document: Document,
    ) -> None:
        mock_doc_repo = mock_doc_repo_cls.return_value
        mock_doc_repo.get_by_id.return_value = document
        mock_processor = mock_processor_cls.return_value
        mock_processor.process.side_effect = DocumentProcessingError("boom")

        service = ProcessingService(db)

        with pytest.raises(ExtractionError):
            service.process_document(doc_id)

        assert db.commit.call_count == 2

    @patch("app.services.processing_service.DocumentProcessor")
    @patch("app.services.processing_service.ChunkRepository")
    @patch("app.services.processing_service.DocumentRepository")
    def test_process_pending_processes_all_uploaded(
        self,
        mock_doc_repo_cls: MagicMock,
        mock_chunk_repo_cls: MagicMock,
        mock_processor_cls: MagicMock,
        db: MagicMock,
        doc_id: UUID,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        doc1 = MagicMock(spec=Document)
        doc1.id = doc_id
        doc1.original_filename = "a.pdf"
        doc2 = MagicMock(spec=Document)
        doc2.id = uuid4()
        doc2.original_filename = "b.pdf"

        mock_doc_repo = mock_doc_repo_cls.return_value
        mock_doc_repo.get_all_by_status.return_value = [doc1, doc2]
        mock_doc_repo.get_by_id.side_effect = [doc1, doc2]
        mock_processor = mock_processor_cls.return_value
        mock_processor.process.return_value = embedded_chunks

        service = ProcessingService(db)
        results = service.process_pending()

        assert results == ["a.pdf", "b.pdf"]
        mock_doc_repo.get_all_by_status.assert_called_once_with(
            DocumentStatus.UPLOADED,
        )

    @patch("app.services.processing_service.DocumentProcessor")
    @patch("app.services.processing_service.ChunkRepository")
    @patch("app.services.processing_service.DocumentRepository")
    def test_process_pending_continues_on_failure(
        self,
        mock_doc_repo_cls: MagicMock,
        mock_chunk_repo_cls: MagicMock,
        mock_processor_cls: MagicMock,
        db: MagicMock,
        doc_id: UUID,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        doc1 = MagicMock(spec=Document)
        doc1.id = doc_id
        doc1.original_filename = "good.pdf"
        doc2 = MagicMock(spec=Document)
        doc2.id = uuid4()
        doc2.original_filename = "bad.pdf"

        mock_doc_repo = mock_doc_repo_cls.return_value
        mock_doc_repo.get_all_by_status.return_value = [doc1, doc2]
        mock_doc_repo.get_by_id.side_effect = [doc1, doc2]
        mock_processor = mock_processor_cls.return_value
        mock_processor.process.side_effect = [
            embedded_chunks,
            DocumentProcessingError("boom"),
        ]

        service = ProcessingService(db)
        results = service.process_pending()

        assert results == ["good.pdf"]
