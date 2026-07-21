import os
from uuid import uuid4, UUID

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.repositories.collection_repository import CollectionRepository
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentResponse
from app.services.exceptions import (
    FileTooLargeException,
    InvalidFileTypeException,
    NotFoundException,
)


class DocumentService:
    def __init__(self, db: Session):
        self.doc_repo = DocumentRepository(db)
        self.collection_repo = CollectionRepository(db)

    def upload(
        self,
        file: UploadFile,
        collection_id: UUID,
        user_id: UUID,
    ) -> DocumentResponse:
        collection = self.collection_repo.get_by_id(collection_id)
        if not collection or collection.user_id != user_id:
            raise NotFoundException()

        self._validate_pdf(file)

        content = file.file.read()
        file_size = len(content)

        max_bytes = settings.max_file_size_mb * 1024 * 1024
        if file_size > max_bytes:
            raise FileTooLargeException()

        storage_filename = f"{uuid4().hex}.pdf"
        relative_path = f"{user_id}/{collection_id}/{storage_filename}"
        full_path = os.path.join(settings.upload_dir, relative_path)

        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "wb") as f:
            f.write(content)

        doc = self.doc_repo.create(
            filename=storage_filename,
            original_filename=file.filename or "untitled.pdf",
            mime_type="application/pdf",
            file_size=file_size,
            storage_path=relative_path,
            collection_id=collection_id,
        )

        return DocumentResponse(
            id=str(doc.id),
            original_filename=doc.original_filename,
            mime_type=doc.mime_type,
            file_size=doc.file_size,
            status=doc.status,
            collection_id=str(doc.collection_id),
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )

    def get_by_id(self, document_id: UUID, user_id: UUID) -> DocumentResponse:
        doc = self.doc_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundException()

        collection = self.collection_repo.get_by_id(doc.collection_id)
        if not collection or collection.user_id != user_id:
            raise NotFoundException()

        return DocumentResponse(
            id=str(doc.id),
            original_filename=doc.original_filename,
            mime_type=doc.mime_type,
            file_size=doc.file_size,
            status=doc.status,
            collection_id=str(doc.collection_id),
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )

    def get_all_by_collection(
        self,
        collection_id: UUID,
        user_id: UUID,
    ) -> list[DocumentResponse]:
        collection = self.collection_repo.get_by_id(collection_id)
        if not collection or collection.user_id != user_id:
            raise NotFoundException()

        documents = self.doc_repo.get_all_by_collection(collection_id)
        return [
            DocumentResponse(
                id=str(d.id),
                original_filename=d.original_filename,
                mime_type=d.mime_type,
                file_size=d.file_size,
                status=d.status,
                collection_id=str(d.collection_id),
                created_at=d.created_at,
                updated_at=d.updated_at,
            )
            for d in documents
        ]

    def delete(self, document_id: UUID, user_id: UUID) -> None:
        doc = self.doc_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundException()

        collection = self.collection_repo.get_by_id(doc.collection_id)
        if not collection or collection.user_id != user_id:
            raise NotFoundException()

        full_path = os.path.join(settings.upload_dir, doc.storage_path)
        if os.path.exists(full_path):
            os.remove(full_path)

        self.doc_repo.delete(doc)

    def _validate_pdf(self, file: UploadFile) -> None:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise InvalidFileTypeException()

        if file.content_type != "application/pdf":
            raise InvalidFileTypeException()

        header = file.file.read(5)
        file.file.seek(0)
        if header != b"%PDF-":
            raise InvalidFileTypeException()
