"""Pydantic response schemas for the FastAPI presentation layer.

These are the public API contracts. They are separate from the internal
DTOs so changes to internal data structures don't automatically change
the API contract.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DebugMetadataSchema(BaseModel):
    """Debug information for RAG pipeline troubleshooting (debug mode only)."""

    stage_timings: dict = Field(description="Execution time (ms) for each of the 10 pipeline stages.")
    total_time_ms: float = Field(description="Total pipeline execution time in milliseconds.")
    retrieved_chunks: list[str] = Field(description="Chunks retrieved from ChromaDB for LLM context.")
    llm_prompt: str = Field(description="Full prompt sent to the LLM.")
    llm_raw_response: str = Field(description="Raw unprocessed response from the LLM.")
    parsed_metadata: dict | None = Field(description="Parsed metadata after extraction (before validation).")


class MetadataResponse(BaseModel):
    """Metadata extracted from a single SDS document."""

    document_id: str = Field(description="Unique identifier assigned at upload time.")
    filename: str = Field(description="Original uploaded filename.")
    status: str = Field(description="Processing status: pending | processing | completed | failed.")
    product_name: str | None = Field(default=None, description="Commercial or trade name of the chemical product.")
    language: str | None = Field(default=None, description="Language the document is written in.")
    jurisdiction: str | None = Field(default=None, description="Regulatory framework this SDS complies with.")
    company_name: str | None = Field(default=None, description="Manufacturer or company responsible for the product.")
    error_message: str | None = Field(default=None, description="Error details if status is 'failed'.")
    created_at: datetime | None = Field(default=None, description="UTC timestamp when the document was uploaded.")
    debug_metadata: DebugMetadataSchema | None = Field(default=None, description="Debug info (only when DEBUG_RAG=true).")

    model_config = {"from_attributes": True}


class BatchExtractionResponse(BaseModel):
    """Response for a batch upload of multiple SDS documents."""

    total: int = Field(description="Number of documents submitted.")
    results: list[MetadataResponse] = Field(description="Per-document extraction results.")


class DocumentListResponse(BaseModel):
    """Response for the document history endpoint."""

    total: int
    documents: list[MetadataResponse]


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    detail: str | None = None
