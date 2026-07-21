import os
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import DocumentStatus
from app.rag.chunker import Chunker
from app.rag.embedder import EmbeddedChunk, SentenceTransformerEmbedder
from app.rag.extractor import PDFExtractor
from app.rag.processor import DocumentProcessor, DocumentProcessingError
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.exceptions import ExtractionError, NotFoundException


class ProcessingService:
    def __init__(self, db: Session):
        self.db = db
        self.doc_repo = DocumentRepository(db)
        self.chunk_repo = ChunkRepository(db)
        self.processor = DocumentProcessor(
            extractor=PDFExtractor(),
            chunker=Chunker(
                chunk_size=1000,
                chunk_overlap=200,
            ),
            embedder=SentenceTransformerEmbedder(
                model_name=settings.embedding_model_name,
                batch_size=settings.embedding_batch_size,
                device=settings.embedding_device,
            ),
        )

    def process_document(self, document_id: UUID) -> list[EmbeddedChunk]:
        doc = self.doc_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundException()

        self.doc_repo.update_status(doc, DocumentStatus.PROCESSING)
        self.db.commit()

        file_path = Path(os.path.join(settings.upload_dir, doc.storage_path))

        try:
            embedded_chunks = self.processor.process(file_path)

            full_text = "\n".join(c.text for c in embedded_chunks)
            self.doc_repo.save_text_content(doc, full_text)

            self.chunk_repo.save_chunks(document_id, embedded_chunks)

            self.doc_repo.update_status(doc, DocumentStatus.READY)
            self.db.commit()

            return embedded_chunks

        except DocumentProcessingError as e:
            self.doc_repo.update_status(doc, DocumentStatus.FAILED)
            self.db.commit()
            raise ExtractionError(str(e)) from e

        except Exception as e:
            self.doc_repo.update_status(doc, DocumentStatus.FAILED)
            self.db.commit()
            raise ExtractionError(
                f"Unexpected error during processing: {e}"
            ) from e

    def process_pending(self) -> list[str]:
        documents = self.doc_repo.get_all_by_status(DocumentStatus.UPLOADED)
        results: list[str] = []
        for doc in documents:
            try:
                self.process_document(doc.id)
                results.append(doc.original_filename)
            except Exception:
                pass
        return results
