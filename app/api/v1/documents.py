import logging
import threading
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.dependencies.auth import get_current_user, get_db
from app.db.session import SessionLocal
from app.schemas.auth import UserResponse
from app.schemas.document import DocumentResponse
from app.services.document_service import DocumentService
from app.services.exceptions import (
    FileTooLargeException,
    InvalidFileTypeException,
    NotFoundException,
)
from app.services.processing_service import ProcessingService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Documents"])


def process_document_background(document_id: str) -> None:
    logger.info("Background processing started for document %s", document_id)
    db = SessionLocal()
    try:
        service = ProcessingService(db)
        service.process_document(UUID(document_id))
        logger.info("Background processing completed for document %s", document_id)
    except Exception as e:
        logger.error("Background processing failed for document %s: %s", document_id, e, exc_info=True)
    finally:
        db.close()


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

        db.commit()

        logger.info("Spawning background processing for document %s", doc.id)
        threading.Thread(
            target=process_document_background,
            args=(doc.id,),
            daemon=True,
        ).start()

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


@router.post(
    "/documents/{document_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
)
def retry_document(
    document_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = DocumentService(db)
    try:
        doc = service.get_by_id(
            document_id=document_id,
            user_id=UUID(current_user.id),
        )
    except NotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if doc.status != "FAILED":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Only FAILED documents can be retried (current status: {doc.status})",
        )

    logger.info("Spawning background retry for document %s", document_id)
    threading.Thread(
        target=process_document_background,
        args=(str(document_id),),
        daemon=True,
    ).start()

    return {"detail": "Retry initiated", "document_id": str(document_id)}


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
