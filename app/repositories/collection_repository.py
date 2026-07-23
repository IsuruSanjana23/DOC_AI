from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.models.collection import Collection
from app.models.document import Document


class CollectionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, name: str, description: str | None, user_id: UUID) -> Collection:
        collection = Collection(
            name=name,
            description=description,
            user_id=user_id,
        )
        self.db.add(collection)
        self.db.flush()
        return collection

    def get_by_id(self, collection_id: UUID) -> Collection | None:
        return self.db.get(Collection, collection_id)

    def get_all_by_user(self, user_id: UUID) -> list[Collection]:
        stmt = (
            select(Collection)
            .where(Collection.user_id == user_id)
            .order_by(Collection.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def count_documents(self, collection_id: UUID) -> int:
        stmt = select(func.count()).select_from(Document).where(Document.collection_id == collection_id)
        return self.db.scalar(stmt) or 0

    def update(
        self,
        collection: Collection,
        name: str | None,
        description: str | None,
        starred: bool | None = None,
    ) -> Collection:
        if name is not None:
            collection.name = name
        if description is not None:
            collection.description = description
        if starred is not None:
            if name is None and description is None:
                self.db.execute(
                    text("UPDATE collections SET starred = :starred WHERE id = :id"),
                    {"starred": starred, "id": collection.id},
                )
                self.db.flush()
                self.db.refresh(collection)
                return collection
            collection.starred = starred
        self.db.flush()
        self.db.refresh(collection)
        return collection

    def delete(self, collection: Collection) -> None:
        self.db.delete(collection)
        self.db.flush()

    def exists_by_name(self, user_id: UUID, name: str) -> bool:
        stmt = select(Collection.id).where(
            Collection.user_id == user_id,
            Collection.name == name,
        )
        return self.db.scalar(stmt) is not None
