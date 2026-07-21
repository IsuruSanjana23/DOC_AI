from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import settings
from app.dependencies.auth import get_current_user, get_db
from app.rag.embedder import SentenceTransformerEmbedder
from app.rag.retriever import Retriever
from app.repositories.chunk_repository import ChunkRepository
from app.schemas.auth import UserResponse
from app.schemas.rag import ChatRequest, ChatResponse
from app.services.exceptions import LLMServiceError
from app.services.llm_service import DeepSeekLLMService
from app.services.prompt_builder import PromptBuilder
from app.services.rag_service import RAGService, RAGServiceError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG"])


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformerEmbedder:
    logger.info(
        "Loading embedding model: %s (device=%s)",
        settings.embedding_model_name,
        settings.embedding_device,
    )
    return SentenceTransformerEmbedder(
        model_name=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )


def _get_rag_service(db: Session) -> RAGService:
    embedder = _get_embedder()
    chunk_repo = ChunkRepository(db)
    retriever = Retriever(
        repository=chunk_repo,
        embedder=embedder,
        top_k=settings.retriever_top_k if hasattr(settings, "retriever_top_k") else 5,
        min_score=(
            settings.retriever_min_score
            if hasattr(settings, "retriever_min_score")
            else None
        ),
    )
    prompt_builder = PromptBuilder()
    llm_service = DeepSeekLLMService()
    return RAGService(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
    )


@router.post("/rag/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rag = _get_rag_service(db)
    try:
        result = await rag.answer(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
        )
    except RAGServiceError as e:
        logger.exception("RAG pipeline failed for query=%s", request.query)
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        )

    token_usage = None
    if result.token_usage:
        token_usage = {
            "prompt_tokens": result.token_usage.get("prompt_tokens", 0),
            "completion_tokens": result.token_usage.get("completion_tokens", 0),
            "total_tokens": result.token_usage.get("total_tokens", 0),
        }

    return ChatResponse(
        query=result.query,
        answer=result.answer,
        sources=[
            {
                "source_number": s["source_number"],
                "document_title": s["document_title"],
                "page_number": s["page_number"],
                "text_preview": s["text_preview"],
                "relevance_score": s["relevance_score"],
            }
            for s in result.sources
        ],
        token_usage=token_usage,
    )
