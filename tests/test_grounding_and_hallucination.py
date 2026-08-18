"""Mandatory Grounding and Anti-Hallucination Tests."""

import pytest

from app.application.services.chat_service import ChatService
from app.application.services.grounding_service import GroundingService
from app.application.services.retrieval_service import RetrievalService, RetrievedChunk
from app.application.use_cases.chat_with_document_use_case import ChatWithDocumentUseCase
from app.domain.entities.chat_message import ChatMessage
from app.domain.repositories.chat_repository import ChatRepository
from app.infrastructure.configuration.settings import Settings


class MockChatRepository(ChatRepository):
    def __init__(self):
        self.messages = []

    def save(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def find_by_document_id(self, document_id: str) -> list[ChatMessage]:
        return [m for m in self.messages if m.document_id == document_id]

    def find_by_conversation_id(self, conversation_id: str) -> list[ChatMessage]:
        return [m for m in self.messages if m.conversation_id == conversation_id]


class MockRetrievalService:
    def __init__(self, sample_chunks):
        self.sample_chunks = sample_chunks

    def retrieve(self, query: str, document_id: str = "all", k=None, threshold=None):
        if "flash point" in query.lower():
            return self.sample_chunks
        # Unrelated query returns empty (below threshold)
        return []


class MockLLMChatService(ChatService):
    def generate_chat_response(self, system_prompt: str, user_prompt: str) -> str:
        if "flash point" in user_prompt.lower():
            return "The flash point is 12°C."
        return "Information not available in the uploaded file."


def test_grounded_answer_and_hallucination_rejection():
    settings = Settings()

    sample_chunks = [
        RetrievedChunk(
            text="Section 9: Physical Properties. Flash point: 12°C. Boiling point: 56°C.",
            document_id="doc-123",
            filename="acetone.pdf",
            page=4,
            section="9",
            section_title="Physical and Chemical Properties",
            score=0.15,
        )
    ]

    retrieval_svc = MockRetrievalService(sample_chunks)
    grounding_svc = GroundingService(settings)
    chat_svc = MockLLMChatService(settings)
    from app.application.services.intent_router import IntentRouter

    repo = MockChatRepository()

    class MockDocRepo:
        def find_by_id(self, doc_id):
            return None
        def find_all(self):
            return []

    use_case = ChatWithDocumentUseCase(
        retrieval_service=retrieval_svc,
        grounding_service=grounding_svc,
        chat_service=chat_svc,
        intent_router=IntentRouter(),
        document_repository=MockDocRepo(),
        chat_repository=repo,
        settings=settings,
    )

    # 1. Valid Grounded Question: "What is the flash point?"
    res1 = use_case.execute("What is the flash point?", document_id="doc-123")
    assert res1.grounded is True
    assert "12°C" in res1.answer
    assert len(res1.sources) == 1
    assert res1.sources[0].document == "acetone.pdf"
    assert res1.sources[0].page == 4

    # 2. Unrelated / Out-of-bounds Question: "When is India's Independence Day?"
    res2 = use_case.execute("When is India's Independence Day?", document_id="doc-123")
    assert res2.grounded is False
    assert res2.answer == "Information not available in the uploaded file."
    assert len(res2.sources) == 0
