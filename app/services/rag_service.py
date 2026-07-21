from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.rag.retriever import Retriever, RetrieverError
from app.repositories.chunk_repository import SearchResult
from app.services.exceptions import LLMServiceError, PromptBuilderError
from app.services.llm_service import BaseLLMService
from app.services.prompt_builder import Prompt, PromptBuilder

logger = logging.getLogger(__name__)


class RAGServiceError(Exception):
    """Raised when the RAG pipeline fails."""


@dataclass(frozen=True)
class RAGResponse:
    query: str
    answer: str
    sources: list[dict[str, Any]]
    token_usage: dict[str, Any] | None
    prompt: Prompt


class RAGService:

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_service: BaseLLMService,
    ) -> None:
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm_service = llm_service

        logger.debug(
            "RAGService initialized — retriever=%s builder=%s llm=%s",
            type(retriever).__name__,
            type(prompt_builder).__name__,
            type(llm_service).__name__,
        )

    async def answer(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> RAGResponse:
        logger.info(
            "RAG pipeline — query_length=%d top_k=%d min_score=%s",
            len(query),
            top_k,
            min_score,
        )

        # 1. Retrieve relevant chunks
        try:
            results: list[SearchResult] = self._retriever.retrieve(
                query, top_k=top_k
            )
        except RetrieverError as e:
            logger.warning("Retrieval failed: %s", e)
            raise RAGServiceError(f"Retrieval failed: {e}") from e

        logger.info("Retrieved %d chunks", len(results))

        # 2. Build the prompt
        try:
            prompt: Prompt = self._prompt_builder.build(query, results)
        except PromptBuilderError as e:
            logger.warning("Prompt building failed: %s", e)
            raise RAGServiceError(f"Prompt building failed: {e}") from e

        logger.info(
            "Prompt built — tokens=%d chunks=%d",
            prompt.token_count,
            prompt.num_chunks,
        )

        # 3. Generate answer
        try:
            response = await self._llm_service.generate(prompt)
        except LLMServiceError as e:
            logger.warning("LLM generation failed: %s", e)
            raise RAGServiceError(f"LLM generation failed: {e}") from e

        logger.info(
            "LLM response — model=%s response_len=%d",
            response.model,
            len(response.text),
        )

        # 4. Structure sources for the response
        sources = [
            {
                "source_number": idx + 1,
                "document_title": (
                    getattr(r.chunk.document, "original_filename", None)
                    if hasattr(r.chunk, "document")
                    else None
                ),
                "page_number": r.chunk.page_number,
                "text_preview": r.chunk.text[:200],
                "relevance_score": round(r.score, 4),
            }
            for idx, r in enumerate(results)
        ]

        return RAGResponse(
            query=query,
            answer=response.text,
            sources=sources,
            token_usage=response.usage,
            prompt=prompt,
        )
