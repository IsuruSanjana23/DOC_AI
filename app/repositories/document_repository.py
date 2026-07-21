from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentStatus


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        filename: str,
        original_filename: str,
        mime_type: str,
        file_size: int,
        storage_path: str,
        collection_id: UUID,
    ) -> Document:
        document = Document(
            filename=filename,
            original_filename=original_filename,
            mime_type=mime_type,
            file_size=file_size,
            storage_path=storage_path,
            collection_id=collection_id,
            status=DocumentStatus.UPLOADED.value,
        )
        self.db.add(document)
        self.db.flush()
        return document

    def get_by_id(self, document_id: UUID) -> Document | None:
        return self.db.get(Document, document_id)

    def get_all_by_collection(self, collection_id: UUID) -> list[Document]:
        stmt = (
            select(Document)
            .where(Document.collection_id == collection_id)
            .order_by(Document.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def update_status(self, document: Document, status: DocumentStatus) -> Document:
        document.status = status.value
        self.db.flush()
        self.db.refresh(document)
        return document

    def delete(self, document: Document) -> None:
        self.db.delete(document)
        self.db.flush()
