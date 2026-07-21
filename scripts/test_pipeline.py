#!/usr/bin/env python
"""End-to-end integration test for the RAG pipeline.

Tests the full Retriever → PromptBuilder → LLMService data flow
with fake retrieval results but a real LLM call.

Usage:
    1. Set your API key:
       $env:LITELLM_API_KEY = "sk-..."  (Windows PowerShell)
       export LITELLM_API_KEY=sk-...     (Linux/macOS)

    2. Run:
       python scripts/test_pipeline.py
"""

import asyncio
import logging
import os
import sys
from uuid import uuid4

# Ensure the backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)

from app.core.config import settings
from app.services.llm_service import DeepSeekLLMService
from app.services.prompt_builder import PromptBuilder
from app.repositories.chunk_repository import SearchResult

# ── Fake data ───────────────────────────────────────────────────────────────


def make_fake_chunk(text: str, page_number: int | None, document_title: str, score: float):
    """Create a fake SearchResult simulating a retrieved chunk."""
    from unittest.mock import MagicMock
    from app.models.chunk import DocumentChunk
    from app.models.document import Document

    chunk = MagicMock(spec=DocumentChunk)
    chunk.text = text
    chunk.page_number = page_number
    chunk.document_id = uuid4()

    doc = MagicMock(spec=Document)
    doc.original_filename = document_title
    chunk.document = doc

    return SearchResult(chunk=chunk, score=score)


def make_sample_results() -> list[SearchResult]:
    return [
        make_fake_chunk(
            text="""Dependency injection is a design pattern in which a class receives its dependencies from external sources rather than creating them internally. In FastAPI, dependency injection is implemented through the Depends function, which allows you to declare dependencies in your path operation functions.""",
            page_number=42,
            document_title="FastAPI User Guide.pdf",
            score=0.95,
        ),
        make_fake_chunk(
            text="""FastAPI's dependency injection system is built on top of Python's type hints. You can use Depends to inject database sessions, authentication credentials, configuration objects, and more into your route handlers.""",
            page_number=43,
            document_title="FastAPI User Guide.pdf",
            score=0.91,
        ),
        make_fake_chunk(
            text="""To use dependency injection in FastAPI, you define a dependency function and then use Depends() as a default value for a parameter in your path operation function. FastAPI will call the dependency function and inject the result automatically.""",
            page_number=44,
            document_title="FastAPI User Guide.pdf",
            score=0.87,
        ),
    ]


# ── Main test ───────────────────────────────────────────────────────────────


async def main():
    query = "What is dependency injection in FastAPI and how do I use it?"

    # 1. Simulate retrieval (normally done by Retriever)
    print("=" * 72)
    print("STEP 1: Simulating retrieval")
    print("=" * 72)
    print(f"Query: {query}")
    results = make_sample_results()
    print(f"Retrieved {len(results)} chunks\n")

    # 2. Build the prompt (PromptBuilder)
    print("=" * 72)
    print("STEP 2: Building prompt (PromptBuilder)")
    print("=" * 72)
    builder = PromptBuilder()
    prompt = builder.build(query, results)
    print(f"System instructions: {len(prompt.system)} chars")
    print(f"Context block: {len(prompt.context)} chars")
    print(f"Question block: {len(prompt.question)} chars")
    print(f"Total prompt: {len(prompt.text)} chars ({prompt.token_count} estimated tokens)")
    print(f"Chunks in context: {prompt.num_chunks}")
    print(f"\n--- FULL PROMPT ---\n{prompt.text}\n-------------------\n")

    # 3. Call the LLM (DeepSeekLLMService)
    print("=" * 72)
    print("STEP 3: Calling LLM (DeepSeekLLMService via LiteLLM)")
    print("=" * 72)

    if not settings.llm_api_key:
        print("ERROR: No LLM API key found in config.")
        print("Add LITELLM_API_KEY=sk-... to backend/.env")
        sys.exit(1)

    service = DeepSeekLLMService()
    response = await service.generate(prompt)

    print(f"\n--- LLM RESPONSE ---")
    print(f"Model: {response.model}")
    print(f"Response text ({len(response.text)} chars):")
    print(response.text)
    if response.usage:
        print(f"\nToken usage: {response.usage}")
    print("--------------------\n")

    # 4. Results
    print("=" * 72)
    print("PIPELINE TEST COMPLETE")
    print("=" * 72)

    return response


if __name__ == "__main__":
    result = asyncio.run(main())
