"""API tests for the ``/rag/chat`` endpoint.

Covers successful RAG responses, error handling, and request validation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


class TestChat:
    """POST /api/v1/rag/chat"""

    @patch("app.api.v1.rag._get_rag_service")
    def test_200_returns_answer(
        self,
        mock_get_service: MagicMock,
        client: TestClient,
    ) -> None:
        from app.services.rag_service import RAGResponse

        mock_service = MagicMock()
        mock_service.answer = AsyncMock()
        mock_service.answer.return_value = RAGResponse(
            query="What is AI?",
            answer="AI stands for Artificial Intelligence.",
            sources=[
                {
                    "source_number": 1,
                    "document_title": "doc.pdf",
                    "page_number": 1,
                    "text_preview": "AI is...",
                    "relevance_score": 0.95,
                }
            ],
            token_usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            prompt=None,
        )
        mock_get_service.return_value = mock_service

        resp = client.post(
            "/api/v1/rag/chat",
            json={"query": "What is AI?", "top_k": 3},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "AI stands for Artificial Intelligence."
        assert len(body["sources"]) == 1
        assert body["sources"][0]["source_number"] == 1
        assert body["token_usage"]["total_tokens"] == 60

    @patch("app.api.v1.rag._get_rag_service")
    def test_200_supports_collection_filter(
        self,
        mock_get_service: MagicMock,
        client: TestClient,
    ) -> None:
        from app.services.rag_service import RAGResponse

        mock_service = MagicMock()
        mock_service.answer = AsyncMock()
        mock_service.answer.return_value = RAGResponse(
            query="filtered",
            answer="Filtered answer",
            sources=[],
            token_usage=None,
            prompt=None,
        )
        mock_get_service.return_value = mock_service

        resp = client.post(
            "/api/v1/rag/chat",
            json={"query": "filtered", "collection_id": "col-1"},
        )
        assert resp.status_code == 200
        mock_service.answer.assert_called_once_with(
            query="filtered",
            top_k=5,
            min_score=None,
            collection_id="col-1",
        )

    @patch("app.api.v1.rag._get_rag_service")
    def test_502_on_rag_error(
        self,
        mock_get_service: MagicMock,
        client: TestClient,
    ) -> None:
        from app.services.rag_service import RAGServiceError

        mock_service = MagicMock()
        mock_service.answer = AsyncMock(side_effect=RAGServiceError("Pipeline failed"))
        mock_get_service.return_value = mock_service

        resp = client.post(
            "/api/v1/rag/chat",
            json={"query": "broken"},
        )
        assert resp.status_code == 502

    def test_502_on_empty_query(self, client: TestClient) -> None:
        resp = client.post("/api/v1/rag/chat", json={"query": ""})
        assert resp.status_code == 502
