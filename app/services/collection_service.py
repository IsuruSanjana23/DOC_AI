from uuid import UUID

from sqlalchemy.orm import Session

from app.repositories.collection_repository import CollectionRepository
from app.schemas.collection import CollectionResponse
from app.services.exceptions import (
    DuplicateCollectionNameException,
    NotFoundException,
)


class CollectionService:
    def __init__(self, db: Session):
        self.repo = CollectionRepository(db)

    def create(
        self,
        name: str,
        description: str | None,
        user_id: UUID,
    ) -> CollectionResponse:
        if self.repo.exists_by_name(user_id, name):
            raise DuplicateCollectionNameException()

        collection = self.repo.create(
            name=name,
            description=description,
            user_id=user_id,
        )
        return CollectionResponse(
            id=str(collection.id),
            name=collection.name,
            description=collection.description,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
        )

    def get_by_id(self, collection_id: UUID, user_id: UUID) -> CollectionResponse:
        collection = self.repo.get_by_id(collection_id)
        if not collection or collection.user_id != user_id:
            raise NotFoundException()
        return CollectionResponse(
            id=str(collection.id),
            name=collection.name,
            description=collection.description,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
        )

    def get_all_by_user(
        self,
        user_id: UUID,
    ) -> list[CollectionResponse]:
        collections = self.repo.get_all_by_user(user_id)
        return [
            CollectionResponse(
                id=str(c.id),
                name=c.name,
                description=c.description,
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in collections
        ]

    def update(
        self,
        collection_id: UUID,
        user_id: UUID,
        name: str | None,
        description: str | None,
    ) -> CollectionResponse:
        collection = self.repo.get_by_id(collection_id)
        if not collection or collection.user_id != user_id:
            raise NotFoundException()

        if name is not None and name != collection.name:
            if self.repo.exists_by_name(user_id, name):
                raise DuplicateCollectionNameException()

        if description is not None and description == "":
            description = None

        collection = self.repo.update(
            collection=collection,
            name=name,
            description=description,
        )
        return CollectionResponse(
            id=str(collection.id),
            name=collection.name,
            description=collection.description,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
        )

    def delete(self, collection_id: UUID, user_id: UUID) -> None:
        collection = self.repo.get_by_id(collection_id)
        if not collection or collection.user_id != user_id:
            raise NotFoundException()
        self.repo.delete(collection)
