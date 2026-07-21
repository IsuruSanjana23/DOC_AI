import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.chunker import TextChunk

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


@dataclass(slots=True)
class EmbeddedChunk:
    chunk_index: int
    text: str
    page_number: int | None
    vector: np.ndarray

    @classmethod
    def from_chunk(
        cls, chunk: TextChunk, vector: np.ndarray
    ) -> "EmbeddedChunk":
        return cls(
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            page_number=chunk.page_number,
            vector=vector,
        )


class BaseEmbedder(ABC):

    @abstractmethod
    def embed(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        chunks = [TextChunk(chunk_index=0, text=text, page_number=None)]
        return self.embed(chunks)[0].vector


class SentenceTransformerEmbedder(BaseEmbedder):

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> SentenceTransformer:
        try:
            logger.info(
                "Loading embedding model: %s (device=%s)",
                self.model_name,
                self.device,
            )
            model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Embedding model loaded: %s", self.model_name)
            return model
        except Exception as e:
            logger.exception(
                "Failed to load embedding model %s", self.model_name
            )
            raise EmbeddingError(
                f"Failed to load model {self.model_name}: {e}"
            ) from e

    def embed(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        if not chunks:
            logger.debug("Called embed with empty chunk list")
            return []

        texts = [c.text for c in chunks]

        try:
            vectors: np.ndarray = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.exception(
                "Embedding failed for %d chunks", len(chunks)
            )
            raise EmbeddingError(
                f"Embedding failed for {len(chunks)} chunks: {e}"
            ) from e

        logger.info(
            "Embedded %d chunks (dim=%d, model=%s, batch_size=%d)",
            len(chunks),
            vectors.shape[1],
            self.model_name,
            self.batch_size,
        )

        return [
            EmbeddedChunk.from_chunk(chunk, vector)
            for chunk, vector in zip(chunks, vectors)
        ]
