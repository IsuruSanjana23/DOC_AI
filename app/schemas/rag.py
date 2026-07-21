from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float | None = None


class SourceResponse(BaseModel):
    source_number: int
    document_title: str | None
    page_number: int | None
    text_preview: str
    relevance_score: float


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceResponse]
    token_usage: TokenUsage | None
