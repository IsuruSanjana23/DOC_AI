from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.dependencies.auth import get_current_user, get_db
from app.schemas.auth import UserResponse
from app.schemas.document import DocumentResponse
from app.services.document_service import DocumentService
from app.services.exceptions import (
    ExtractionError,
    FileTooLargeException,
    InvalidFileTypeException,
    NotFoundException,
)
from app.services.processing_service import ProcessingService

router = APIRouter(tags=["Documents"])


@router.post(
    "/documents/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: UploadFile = File(...),
    collection_id: UUID = Form(...),
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = DocumentService(db)
    try:
        doc = service.upload(
            file=file,
            collection_id=collection_id,
            user_id=UUID(current_user.id),
        )

        processing = ProcessingService(db)
        try:
            processing.process_document(UUID(doc.id))
        except ExtractionError:
            pass

        return doc
    except InvalidFileTypeException:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are allowed",
        )
    except FileTooLargeException:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum of {20} MB",
        )
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )


@router.get(
    "/collections/{collection_id}/documents",
    response_model=list[DocumentResponse],
)
def list_documents(
    collection_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = DocumentService(db)
    try:
        return service.get_all_by_collection(
            collection_id=collection_id,
            user_id=UUID(current_user.id),
        )
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found",
        )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
)
def get_document(
    document_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = DocumentService(db)
    try:
        return service.get_by_id(
            document_id=document_id,
            user_id=UUID(current_user.id),
        )
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = DocumentService(db)
    try:
        service.delete(
            document_id=document_id,
            user_id=UUID(current_user.id),
        )
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
