from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

from app.repositories.chunk_repository import SearchResult

logger = logging.getLogger(__name__)

# ── Default Templates ──────────────────────────────────────────────────────
# These are overridable via constructor injection. Each template uses
# ``str.format()``-style placeholders. Override any of them to change the
# prompt format without modifying the builder itself.

DEFAULT_SYSTEM_TEMPLATE: Final[str] = (
    "You are a precise, factual AI assistant answering questions "
    "about documents.\n\n"
    "Rules:\n"
    "1. Answer ONLY using the provided context below.\n"
    "2. If the context does not contain the answer, say:\n"
    '   "I cannot answer this question based on the available '
    'documents."\n'
    "3. Do NOT fabricate information or use outside knowledge.\n"
    "4. Cite sources using [Source N] after each statement.\n"
    "5. Use markdown formatting for readability."
)

DEFAULT_CONTEXT_TEMPLATE: Final[str] = (
    "Context:\n{context_blocks}"
)

DEFAULT_CHUNK_TEMPLATE: Final[str] = (
    "--------------\n"
    "Source {source_number}\n"
    "Document: {document_title}\n"
    "Page: {page_number}\n"
    "Content:\n"
    "{text}"
)

DEFAULT_QUESTION_TEMPLATE: Final[str] = (
    "Using ONLY the context above, answer the following question:\n\n"
    "Question:\n"
    "{query}\n\n"
    "Answer:"
)

_UNKNOWN_DOCUMENT: Final[str] = "Unknown Document"
_UNAVAILABLE_PAGE: Final[str] = "N/A"


class PromptBuilderError(Exception):
    """Raised when prompt construction fails.

    This is the sole exception type callers should catch when using
    :class:`PromptBuilder`. All input validation errors are wrapped in
    this exception with a descriptive message.
    """


@dataclass(frozen=True)
class Prompt:
    """The immutable result of a :class:`PromptBuilder.build` call.

    :param system: Formatted system instructions.
    :param context: Formatted retrieved-chunk context.
    :param question: Formatted user question (with framing).
    :param text: Fully assembled prompt string (system + context + question).
    :param token_count: Estimated token count (stub — currently character
        count; replace with real tokenizer later).
    :param num_chunks: Number of chunks included in the context.
    """

    system: str
    context: str
    question: str
    text: str
    token_count: int
    num_chunks: int


class PromptBuilder:
    """Constructs RAG prompts from a query and retrieved chunks.

    The builder separates prompt construction into discrete steps
    (system instructions, context formatting, question framing) and
    assembles them into a :class:`Prompt` dataclass.

    All templates are overridable via constructor injection, making the
    builder adaptable to different LLM providers, prompt formats, and
    citation styles without changing the orchestration logic.

    Usage::

        builder = PromptBuilder()
        prompt = builder.build("What is DI?", search_results)
        # prompt.text       → fully assembled string
        # prompt.system     → system instructions
        # prompt.context    → formatted chunks
        # prompt.question   → framed question
        # prompt.token_count → estimated length
    """

    def __init__(
        self,
        system_template: str = DEFAULT_SYSTEM_TEMPLATE,
        context_template: str = DEFAULT_CONTEXT_TEMPLATE,
        chunk_template: str = DEFAULT_CHUNK_TEMPLATE,
        question_template: str = DEFAULT_QUESTION_TEMPLATE,
    ) -> None:
        """Initialise the builder with optional custom templates.

        :param system_template: Template for system instructions.
            Must NOT use any placeholders (static text).
        :param context_template: Template wrapping all chunk blocks.
            Must contain the ``{context_blocks}`` placeholder.
        :param chunk_template: Template for a single chunk.
            Must contain ``{source_number}``, ``{document_title}``,
            ``{page_number}``, and ``{text}`` placeholders.
        :param question_template: Template for the user question.
            Must contain the ``{query}`` placeholder.
        """
        self._system_template = system_template
        self._context_template = context_template
        self._chunk_template = chunk_template
        self._question_template = question_template

        logger.debug(
            "PromptBuilder initialized (system_len=%d, context_len=%d, "
            "chunk_len=%d, question_len=%d)",
            len(system_template),
            len(context_template),
            len(chunk_template),
            len(question_template),
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def build(self, query: str, results: list[SearchResult]) -> Prompt:
        """Build a complete :class:`Prompt` from a query and retrieved chunks.

        :param query: The user's question. Must be non-empty and non-blank.
        :param results: Retrieved chunks from the vector search.
            May be empty — the builder will produce a prompt with no
            context, and the system instructions will handle it.

        :returns: An immutable :class:`Prompt` instance.

        :raises PromptBuilderError: If ``query`` is empty or whitespace-only.

        :raises PromptBuilderError: If any ``SearchResult`` is malformed
            (e.g., missing chunk text).
        """
        query = self._validate_query(query)

        num_chunks = len(results)
        logger.info(
            "Building prompt — query_length=%d num_chunks=%d",
            len(query),
            num_chunks,
        )

        system = self._build_system_prompt()
        context = self._build_context(results)
        question = self._build_question(query)
        text = self._assemble(system, context, question)

        token_count = self._estimate_tokens(text)

        logger.info(
            "Prompt built — system_len=%d context_len=%d "
            "question_len=%d total_len=%d tokens=%d chunks=%d",
            len(system),
            len(context),
            len(question),
            len(text),
            token_count,
            num_chunks,
        )

        return Prompt(
            system=system,
            context=context,
            question=question,
            text=text,
            token_count=token_count,
            num_chunks=num_chunks,
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _validate_query(self, query: str) -> str:
        """Validate and normalize the query string.

        :raises PromptBuilderError: If query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise PromptBuilderError(
                "Query must be a non-empty, non-blank string. "
                f"Received: '{query!r}'"
            )
        return query.strip()

    def _build_system_prompt(self) -> str:
        """Return the system instructions (static template).

        This method exists as a separate helper so that subclasses or
        future versions can inject dynamic content (e.g., date, user
        name, tone override) without changing ``build()``.
        """
        return self._system_template.strip()

    def _build_context(self, results: list[SearchResult]) -> str:
        """Format all retrieved chunks into a single context block.

        :param results: Retrieved search results.
        :returns: Formatted context string. Returns an empty string if
            ``results`` is empty.
        """
        if not results:
            logger.debug("Context is empty — no chunks provided")
            return ""

        context_blocks = "\n\n".join(
            self._format_chunk(idx + 1, result)
            for idx, result in enumerate(results)
        )

        return self._context_template.format(
            context_blocks=context_blocks,
        )

    def _format_chunk(self, source_number: int, result: SearchResult) -> str:
        """Format a single chunk into its template representation.

        :param source_number: 1-based source index for citation.
        :param result: A single search result.
        :returns: Formatted chunk block.

        :raises PromptBuilderError: If the chunk text is empty or None.
        """
        chunk = result.chunk

        if not chunk.text:
            raise PromptBuilderError(
                f"Chunk at source {source_number} has empty text "
                f"(document_id={chunk.document_id})"
            )

        document_title = _UNKNOWN_DOCUMENT
        if hasattr(chunk, "document") and chunk.document is not None:
            doc = chunk.document
            if hasattr(doc, "original_filename") and doc.original_filename:
                document_title = doc.original_filename

        page_number = (
            str(chunk.page_number)
            if chunk.page_number is not None
            else _UNAVAILABLE_PAGE
        )

        return self._chunk_template.format(
            source_number=source_number,
            document_title=document_title,
            page_number=page_number,
            text=chunk.text,
        )

    def _build_question(self, query: str) -> str:
        """Frame the user query inside the question template.

        :param query: The validated, stripped query.
        """
        return self._question_template.format(query=query)

    def _assemble(
        self,
        system: str,
        context: str,
        question: str,
    ) -> str:
        """Concatenate the three prompt sections into one string.

        Sections are separated by blank lines for readability.
        If there is no context, the assembly skips the context
        separator to avoid ``"Context:\\n\\n"`` noise.
        """
        parts = [system]

        if context:
            parts.append(context)

        parts.append(question)

        return "\n\n".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the token count of *text*.

        Current implementation uses character count as a rough proxy.
        Replace with a real tokenizer (``tiktoken``, ``tokenizers``)
        when token-level control is needed.

        :param text: The fully assembled prompt text.
        :returns: Estimated token count.
        """
        return len(text)
