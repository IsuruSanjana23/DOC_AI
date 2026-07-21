import os
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import DocumentStatus
from app.rag.extractor import PDFExtractionError, PDFExtractor
from app.repositories.document_repository import DocumentRepository
from app.services.exceptions import ExtractionError, NotFoundException


class ProcessingService:
    def __init__(self, db: Session):
        self.doc_repo = DocumentRepository(db)
        self.extractor = PDFExtractor()

    def extract_text(self, document_id: UUID) -> str:
        doc = self.doc_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundException()

        self.doc_repo.update_status(doc, DocumentStatus.PROCESSING)

        try:
            pages = self.extractor.extract(
                Path(os.path.join(settings.upload_dir, doc.storage_path))
            )
            text = "\n".join(page.text for page in pages)
            self.doc_repo.save_text_content(doc, text)
            self.doc_repo.update_status(doc, DocumentStatus.READY)
            return text
        except PDFExtractionError as e:
            self.doc_repo.update_status(doc, DocumentStatus.FAILED)
            raise ExtractionError(str(e)) from e
        except Exception as e:
            self.doc_repo.update_status(doc, DocumentStatus.FAILED)
            raise ExtractionError(
                f"Unexpected error during extraction: {e}"
            ) from e

    def process_pending(self) -> list[str]:
        documents = self.doc_repo.get_all_by_status(DocumentStatus.UPLOADED)
        results = []
        for doc in documents:
            try:
                self.doc_repo.update_status(doc, DocumentStatus.PROCESSING)
                pages = self.extractor.extract(
                    Path(os.path.join(settings.upload_dir, doc.storage_path))
                )
                text = "\n".join(page.text for page in pages)
                self.doc_repo.save_text_content(doc, text)
                self.doc_repo.update_status(doc, DocumentStatus.READY)
                results.append(doc.original_filename)
            except Exception:
                self.doc_repo.update_status(doc, DocumentStatus.FAILED)
        return results
