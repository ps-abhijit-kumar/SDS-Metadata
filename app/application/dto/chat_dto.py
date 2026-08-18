"""Chat DTOs for RAG conversational system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceCitationDTO:
    """Source attribution details."""

    document: str
    page: int = 1
    section: str = "0"
    section_title: str = "General"
    source_type: str = "document_content"

    def to_dict(self) -> dict:
        return {
            "document": self.document,
            "source_type": self.source_type,
            "page": self.page,
            "section": self.section,
            "section_title": self.section_title,
        }


@dataclass
class ChatResponseDTO:
    """Complete chat response DTO."""

    answer: str
    grounded: bool
    conversation_id: str
    document_id: str
    sources: list[SourceCitationDTO] = field(default_factory=list)
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "grounded": self.grounded,
            "conversation_id": self.conversation_id,
            "document_id": self.document_id,
            "sources": [s.to_dict() for s in self.sources],
            "metrics": {
                "retrieval_ms": round(self.retrieval_ms, 2),
                "llm_ms": round(self.llm_ms, 2),
                "total_ms": round(self.total_ms, 2),
            },
        }
