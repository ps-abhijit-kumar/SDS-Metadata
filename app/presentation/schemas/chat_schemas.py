"""Pydantic schemas for chat endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat query payload."""

    question: str = Field(..., description="User question or prompt.")
    document_id: str = Field("all", description="Document ID or 'all' for multi-document search.")
    conversation_id: str | None = Field(None, description="Optional conversation ID for context history.")


class SourceCitationSchema(BaseModel):
    """Source attribution details."""

    document: str
    page: int
    section: str
    section_title: str


class MetricsSchema(BaseModel):
    """Execution timing metrics."""

    retrieval_ms: float
    llm_ms: float
    total_ms: float


class ChatResponseSchema(BaseModel):
    """Response payload for chat endpoints."""

    answer: str
    grounded: bool
    conversation_id: str
    document_id: str
    sources: list[SourceCitationSchema] = []
    metrics: MetricsSchema
