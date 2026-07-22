from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.dependencies.auth import get_current_user, get_db
from app.schemas.auth import UserResponse
from app.schemas.collection import (
    CollectionCreate,
    CollectionResponse,
    CollectionUpdate,
)
from app.services.collection_service import CollectionService
from app.services.exceptions import (
    DuplicateCollectionNameException,
    NotFoundException,
)

router = APIRouter(prefix="/collections", tags=["Collections"])


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_collection(
    body: CollectionCreate,
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = CollectionService(db)
    try:
        return service.create(
            name=body.name,
            description=body.description,
            user_id=UUID(current_user.id),
        )
    except DuplicateCollectionNameException:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A collection with this name already exists",
        )


@router.get(
    "",
    response_model=list[CollectionResponse],
)
def list_collections(
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = CollectionService(db)
    return service.get_all_by_user(user_id=UUID(current_user.id))


@router.get(
    "/{collection_id}",
    response_model=CollectionResponse,
)
def get_collection(
    collection_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = CollectionService(db)
    try:
        return service.get_by_id(
            collection_id=collection_id,
            user_id=UUID(current_user.id),
        )
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )


@router.patch(
    "/{collection_id}",
    response_model=CollectionResponse,
)
def update_collection(
    collection_id: UUID,
    body: CollectionUpdate,
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = CollectionService(db)
    try:
        return service.update(
            collection_id=collection_id,
            user_id=UUID(current_user.id),
            name=body.name,
            description=body.description,
            starred=body.starred,
        )
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )
    except DuplicateCollectionNameException:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A collection with this name already exists",
        )


@router.delete(
    "/{collection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_collection(
    collection_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = CollectionService(db)
    try:
        service.delete(
            collection_id=collection_id,
            user_id=UUID(current_user.id),
        )
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )
