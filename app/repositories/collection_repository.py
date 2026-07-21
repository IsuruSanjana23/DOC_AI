from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.collection import Collection


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

    def update(
        self,
        collection: Collection,
        name: str | None,
        description: str | None,
    ) -> Collection:
        if name is not None:
            collection.name = name
        if description is not None:
            collection.description = description
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
