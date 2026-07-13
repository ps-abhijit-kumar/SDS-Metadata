"""Debug context for RAG pipeline performance monitoring and troubleshooting.

Tracks timing information, retrieved chunks, LLM prompt, and LLM response
across all 10 stages of the extraction pipeline. Only populated when
DEBUG_RAG or LOG_STAGES is enabled in settings.

10 Pipeline Stages:
  1. PDF extraction (PyMuPDF)
  2. Text cleaning & normalization
  3. Semantic chunking
  4. Embedding generation
  5. Chunk storage in ChromaDB
  6. Semantic retrieval from ChromaDB
  7. Prompt building
  8. LLM inference (Ollama)
  9. Metadata parsing & validation
  10. Database persistence
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StageTimer:
    """Tracks execution time for a single pipeline stage."""

    stage_name: str
    duration_ms: float = 0.0
    start_time: float = 0.0

    def __enter__(self) -> StageTimer:
        """Enter context manager — record start time."""
        self.start_time = time.time()
        return self

    def __exit__(self, *args) -> None:
        """Exit context manager — calculate elapsed time in milliseconds."""
        elapsed = time.time() - self.start_time
        self.duration_ms = elapsed * 1000

    def to_dict(self) -> dict:
        return {
            "stage": self.stage_name,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class DebugContext:
    """Collects debug information across the entire RAG pipeline.

    Fields are only populated when debug mode is enabled.
    """

    # ── Timing Information (all 10 stages) ──────────────────────────────────
    stage_timings: dict[str, float] = field(default_factory=dict)

    # ── Retrieval Information ──────────────────────────────────────────────
    retrieved_chunks: list[str] = field(default_factory=list)
    retrieval_query: str = ""

    # ── LLM Interaction ────────────────────────────────────────────────────
    llm_prompt: str = ""
    llm_raw_response: str = ""

    # ── Metadata Extraction Results ────────────────────────────────────────
    parsed_metadata: dict | None = None

    def add_stage_timing(self, stage_name: str, duration_ms: float) -> None:
        """Record the execution time for a stage."""
        self.stage_timings[stage_name] = round(duration_ms, 2)

    def get_total_time_ms(self) -> float:
        """Calculate total pipeline execution time."""
        return sum(self.stage_timings.values())

    def to_dict(self) -> dict:
        """Convert debug context to dictionary for JSON serialization."""
        return {
            "stage_timings": self.stage_timings,
            "total_time_ms": self.get_total_time_ms(),
            "retrieved_chunks": self.retrieved_chunks,
            "llm_prompt": self.llm_prompt,
            "llm_raw_response": self.llm_raw_response,
            "parsed_metadata": self.parsed_metadata,
        }
