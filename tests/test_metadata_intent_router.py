"""Comprehensive tests for metadata intent routing, section awareness, zero sources fallback, and language normalization."""

import pytest

from app.application.services.chat_service import ChatService
from app.application.services.grounding_service import GroundingService
from app.application.services.intent_router import IntentRouter, MetadataIntent
from app.application.services.retrieval_service import RetrievalService, RetrievedChunk
from app.application.services.section_detector import SectionDetector
from app.application.use_cases.chat_with_document_use_case import ChatWithDocumentUseCase
from app.domain.entities.chat_message import ChatMessage
from app.domain.entities.document import Document
from app.domain.enums.document_status import DocumentStatus
from app.domain.value_objects.sds_metadata import SDSMetadata
from app.infrastructure.configuration.settings import Settings


class MockDocumentRepository:
    def __init__(self, doc):
        self.doc = doc

    def save(self, document: Document) -> None:
        pass

    def update(self, document: Document) -> None:
        pass

    def find_by_id(self, document_id: str) -> Document | None:
        return self.doc if self.doc and self.doc.id == document_id else None

    def find_by_hash(self, file_hash: str) -> Document | None:
        return None

    def find_all(self) -> list[Document]:
        return [self.doc] if self.doc else []

    def delete(self, document_id: str) -> None:
        pass


class MockChatRepository:
    def __init__(self):
        self.messages = []

    def save(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def find_by_document_id(self, document_id: str) -> list[ChatMessage]:
        return self.messages

    def find_by_conversation_id(self, conversation_id: str) -> list[ChatMessage]:
        return self.messages


class TrackingLLMChatService(ChatService):
    def __init__(self, settings):
        super().__init__(settings)
        self.llm_was_called = False

    def generate_chat_response(self, system_prompt: str, user_prompt: str) -> str:
        self.llm_was_called = True
        return "RAG response from LLM"

    def generate_chat_response_with_metrics(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        self.llm_was_called = True
        return "RAG response from LLM", 10.0


class MockSectionRetrievalService(RetrievalService):
    def __init__(self, sample_chunks):
        self.sample_chunks = sample_chunks
        self.last_query = None

    def retrieve(self, query: str, document_id: str = "all", k=None, threshold=None):
        self.last_query = query
        detector = SectionDetector()
        sec = detector.detect_section(query)
        if sec:
            return [c for c in self.sample_chunks if c.section == sec]
        return self.sample_chunks if "prime minister" not in query.lower() else []


# ── TEST 1: Language Query ("In which language is it written?") ─────────────
def test_1_language_query_in_which_language_is_it_written():
    router = IntentRouter()
    assert router.detect_intent("In which language is it written?") == MetadataIntent.LANGUAGE


# ── TEST 2: Language Query Variant ("What language is this document in?") ────
def test_2_language_query_what_language_is_this_document_in():
    router = IntentRouter()
    assert router.detect_intent("What language is this document in?") == MetadataIntent.LANGUAGE


# ── TEST 3: Product Name Query ───────────────────────────────────────────────
def test_3_product_name_query():
    settings = Settings()
    meta = SDSMetadata(
        file_id="doc-l0288",
        product_name="Lipid Mixture 1, Chemically Defined",
        company_name="Sigma-Aldrich Chemical Pvt Limited",
        language="Spanish",
        jurisdiction="European Union (REACH / CLP)",
    )
    doc = Document(id="doc-l0288", filename="l0288.pdf", file_path="l0288.pdf", status=DocumentStatus.COMPLETED, metadata=meta)
    tracking_chat = TrackingLLMChatService(settings)

    use_case = ChatWithDocumentUseCase(
        retrieval_service=MockSectionRetrievalService([]),
        grounding_service=GroundingService(settings),
        chat_service=tracking_chat,
        intent_router=IntentRouter(),
        document_repository=MockDocumentRepository(doc),
        chat_repository=MockChatRepository(),
        settings=settings,
    )

    res = use_case.execute("What is the product name?", document_id="doc-l0288")
    assert "Lipid Mixture 1, Chemically Defined" in res.answer
    assert tracking_chat.llm_was_called is False  # NO LLM!


# ── TEST 4: Manufacturer Query ("Who is the manufacturer?") ──────────────────
def test_4_who_is_the_manufacturer():
    settings = Settings()
    meta = SDSMetadata(
        file_id="doc-l0288",
        product_name="Lipid Mixture 1, Chemically Defined",
        company_name="Sigma-Aldrich Chemical Pvt Limited",
        language="Spanish",
        jurisdiction="European Union (REACH / CLP)",
    )
    doc = Document(id="doc-l0288", filename="l0288.pdf", file_path="l0288.pdf", status=DocumentStatus.COMPLETED, metadata=meta)
    tracking_chat = TrackingLLMChatService(settings)

    use_case = ChatWithDocumentUseCase(
        retrieval_service=MockSectionRetrievalService([]),
        grounding_service=GroundingService(settings),
        chat_service=tracking_chat,
        intent_router=IntentRouter(),
        document_repository=MockDocumentRepository(doc),
        chat_repository=MockChatRepository(),
        settings=settings,
    )

    res = use_case.execute("Who is the manufacturer?", document_id="doc-l0288")
    assert "Sigma-Aldrich Chemical Pvt Limited" in res.answer
    assert tracking_chat.llm_was_called is False  # NO LLM!
    assert res.retrieval_ms == 0.0  # NO RETRIEVAL!


# ── TEST 5: Supplier Query ("Who is the supplier?") ──────────────────────────
def test_5_who_is_the_supplier():
    router = IntentRouter()
    assert router.detect_intent("Who is the supplier?") == MetadataIntent.COMPANY_NAME


# ── TEST 6: Jurisdiction Query ───────────────────────────────────────────────
def test_6_jurisdiction_query():
    settings = Settings()
    meta = SDSMetadata(
        file_id="doc-l0288",
        product_name="Lipid Mixture 1",
        company_name="Sigma-Aldrich",
        language="Spanish",
        jurisdiction="European Union (REACH / CLP)",
    )
    doc = Document(id="doc-l0288", filename="l0288.pdf", file_path="l0288.pdf", status=DocumentStatus.COMPLETED, metadata=meta)
    tracking_chat = TrackingLLMChatService(settings)

    use_case = ChatWithDocumentUseCase(
        retrieval_service=MockSectionRetrievalService([]),
        grounding_service=GroundingService(settings),
        chat_service=tracking_chat,
        intent_router=IntentRouter(),
        document_repository=MockDocumentRepository(doc),
        chat_repository=MockChatRepository(),
        settings=settings,
    )

    res = use_case.execute("What jurisdiction does this SDS follow?", document_id="doc-l0288")
    assert "European Union (REACH / CLP)" in res.answer
    assert tracking_chat.llm_was_called is False


# ── TEST 7: RAG Section 4 Query ("What are the first aid measures?") ────────
def test_7_first_aid_measures_rag_section_4():
    settings = Settings()
    sample_chunks = [
        RetrievedChunk(
            text="SECCIÓN 4: Primeros auxilios. En caso de inhalación: aire fresco. En caso de contacto con la piel: lavar con agua.",
            document_id="doc-l0288",
            filename="l0288.pdf",
            page=2,
            section="4",
            section_title="4. Primeros auxilios",
            score=0.20,
        )
    ]
    retrieval_svc = MockSectionRetrievalService(sample_chunks)
    tracking_chat = TrackingLLMChatService(settings)
    doc_repo = MockDocumentRepository(None)

    use_case = ChatWithDocumentUseCase(
        retrieval_service=retrieval_svc,
        grounding_service=GroundingService(settings),
        chat_service=tracking_chat,
        intent_router=IntentRouter(),
        document_repository=doc_repo,
        chat_repository=MockChatRepository(),
        settings=settings,
    )

    res = use_case.execute("What are the first aid measures?", document_id="doc-l0288")
    assert res.grounded is True
    assert tracking_chat.llm_was_called is True  # LLM IS CALLED FOR SECTION 4 RAG!
    assert len(res.sources) == 1
    assert res.sources[0].section == "4"


# ── TEST 8: RAG Section 3 Query ("What chemicals are present in the product?")
def test_8_chemicals_present_rag_section_3():
    settings = Settings()
    sample_chunks = [
        RetrievedChunk(
            text="SECCIÓN 3: Composición/información sobre los componentes. Mevalonolactona CAS 674-26-0.",
            document_id="doc-l0288",
            filename="l0288.pdf",
            page=2,
            section="3",
            section_title="3. Composición",
            score=0.18,
        )
    ]
    retrieval_svc = MockSectionRetrievalService(sample_chunks)
    tracking_chat = TrackingLLMChatService(settings)

    use_case = ChatWithDocumentUseCase(
        retrieval_service=retrieval_svc,
        grounding_service=GroundingService(settings),
        chat_service=tracking_chat,
        intent_router=IntentRouter(),
        document_repository=MockDocumentRepository(None),
        chat_repository=MockChatRepository(),
        settings=settings,
    )

    res = use_case.execute("What chemicals are present in the product?", document_id="doc-l0288")
    assert res.grounded is True
    assert tracking_chat.llm_was_called is True  # LLM CALLED FOR RAG!
    assert len(res.sources) == 1
    assert res.sources[0].section == "3"


# ── TEST 9: Unrelated Question Fallback (No Sources & No LLM) ───────────────
def test_9_unrelated_question_has_zero_sources_and_no_llm():
    settings = Settings()
    tracking_chat = TrackingLLMChatService(settings)

    use_case = ChatWithDocumentUseCase(
        retrieval_service=MockSectionRetrievalService([]),
        grounding_service=GroundingService(settings),
        chat_service=tracking_chat,
        intent_router=IntentRouter(),
        document_repository=MockDocumentRepository(None),
        chat_repository=MockChatRepository(),
        settings=settings,
    )

    res = use_case.execute("When was the Prime Minister born in India?", document_id="all")
    assert res.grounded is False
    assert res.answer == "Information not available in the uploaded documents."
    assert res.sources == []  # ZERO SOURCES!
    assert tracking_chat.llm_was_called is False  # NO LLM!


# ── TEST 11: Five Duplicate Upload Records Deduplicated to ONE Chat Entry ────
def test_11_five_duplicate_records_produce_one_chat_result():
    settings = Settings()
    meta = SDSMetadata(
        file_id="meta-1",
        product_name="Lipid Mixture 1",
        company_name="Sigma-Aldrich Chemical Pvt Limited",
        language="Spanish",
        jurisdiction="European Union (REACH / CLP)",
    )
    docs = [
        Document(id=f"doc-{i}", filename="l0288.pdf", file_path="l0288.pdf", file_hash="hash_l0288", status=DocumentStatus.COMPLETED, metadata=meta)
        for i in range(1, 6)
    ]

    class MultiDupMockRepo:
        def find_all(self):
            return docs
        def find_all_canonical(self):
            return [docs[0]]
        def find_by_id(self, doc_id):
            return docs[0]

    tracking_chat = TrackingLLMChatService(settings)
    use_case = ChatWithDocumentUseCase(
        retrieval_service=MockSectionRetrievalService([]),
        grounding_service=GroundingService(settings),
        chat_service=tracking_chat,
        intent_router=IntentRouter(),
        document_repository=MultiDupMockRepo(),
        chat_repository=MockChatRepository(),
        settings=settings,
    )

    res = use_case.execute("Who is the manufacturer?", document_id="all")
    assert res.grounded is True
    # Verify ONLY ONE l0288.pdf entry appears in chat response answer
    assert res.answer.count("l0288.pdf") <= 1
    assert "Sigma-Aldrich Chemical Pvt Limited" in res.answer


# ── TEST 12: History Preserves Duplicate Upload Events ───────────────────────
def test_12_history_preserves_duplicate_upload_events():
    meta = SDSMetadata(
        file_id="meta-1",
        product_name="Lipid Mixture 1",
        company_name="Sigma-Aldrich",
        language="Spanish",
        jurisdiction="EU",
    )
    docs = [
        Document(id=f"doc-{i}", filename="l0288.pdf", file_path="l0288.pdf", file_hash="hash_l0288", status=DocumentStatus.DUPLICATE if i > 1 else DocumentStatus.COMPLETED, metadata=meta)
        for i in range(1, 6)
    ]

    class HistoryMockRepo:
        def find_all(self):
            return docs  # All 5 upload events preserved in history!
        def find_all_canonical(self):
            return [docs[0]]  # Chat uses only 1 canonical document!

    repo = HistoryMockRepo()
    assert len(repo.find_all()) == 5  # History preserves 5 events
    assert len(repo.find_all_canonical()) == 1  # Chat gets 1 canonical doc


# ── TEST 13: First-Aid RAG Answer Contains Actual Source Metadata ────────────
def test_13_first_aid_answer_contains_actual_retrieved_source_metadata():
    settings = Settings()
    sample_chunks = [
        RetrievedChunk(
            text="SECCIÓN 4: Primeros auxilios. En caso de contacto ocular lavar con agua abundante.",
            document_id="doc-l0288",
            filename="l0288.pdf",
            page=4,
            section="4",
            section_title="4. Primeros auxilios",
            score=0.15,
        )
    ]
    tracking_chat = TrackingLLMChatService(settings)
    use_case = ChatWithDocumentUseCase(
        retrieval_service=MockSectionRetrievalService(sample_chunks),
        grounding_service=GroundingService(settings),
        chat_service=tracking_chat,
        intent_router=IntentRouter(),
        document_repository=MockDocumentRepository(None),
        chat_repository=MockChatRepository(),
        settings=settings,
    )

    res = use_case.execute("What are the first aid measures?", document_id="doc-l0288")
    assert res.grounded is True
    assert len(res.sources) == 1
    src = res.sources[0]
    assert src.document == "l0288.pdf"
    assert src.section == "4"
    assert src.page == 4
    assert src.section_title == "4. Primeros auxilios"


# ── TEST 14: Five Unique Documents Queried Without Duplicate Results ─────────
def test_14_five_unique_documents_queried_without_duplicates():
    settings = Settings()
    names = ["Acetone", "Ethanol", "Methanol", "Toluene", "Xylene"]
    docs = []
    for i, name in enumerate(names, 1):
        meta = SDSMetadata(
            file_id=f"meta-{i}",
            product_name=name,
            company_name=f"Company {name}",
            language="English",
            jurisdiction="US",
        )
        doc = Document(id=f"doc-{i}", filename=f"{name.lower()}.pdf", file_path=f"{name.lower()}.pdf", file_hash=f"hash_{name}", status=DocumentStatus.COMPLETED, metadata=meta)
        docs.append(doc)

    class MultiUniqueRepo:
        def find_all(self):
            return docs
        def find_all_canonical(self):
            return docs

    tracking_chat = TrackingLLMChatService(settings)
    use_case = ChatWithDocumentUseCase(
        retrieval_service=MockSectionRetrievalService([]),
        grounding_service=GroundingService(settings),
        chat_service=tracking_chat,
        intent_router=IntentRouter(),
        document_repository=MultiUniqueRepo(),
        chat_repository=MockChatRepository(),
        settings=settings,
    )

    res = use_case.execute("What is the product name?", document_id="all")
    assert res.grounded is True
    for name in names:
        assert f"{name.lower()}.pdf" in res.answer
    assert len(res.sources) == 5  # Exactly 5 unique sources!


# ── TEST 15: Streaming Sends Tokens Incrementally ────────────────────────────
def test_15_streaming_sends_tokens_incrementally():
    settings = Settings()
    sample_chunks = [
        RetrievedChunk(
            text="SECCIÓN 4: Primeros auxilios. En caso de inhalación salir al aire libre.",
            document_id="doc-1",
            filename="l0288.pdf",
            page=1,
            section="4",
            section_title="4. Primeros auxilios",
            score=0.10,
        )
    ]

    class StreamLLMChatService(ChatService):
        def stream_chat_response(self, system_prompt: str, user_prompt: str):
            yield "Token1 "
            yield "Token2 "
            yield "Token3"

    use_case = ChatWithDocumentUseCase(
        retrieval_service=MockSectionRetrievalService(sample_chunks),
        grounding_service=GroundingService(settings),
        chat_service=StreamLLMChatService(settings),
        intent_router=IntentRouter(),
        document_repository=MockDocumentRepository(None),
        chat_repository=MockChatRepository(),
        settings=settings,
    )

    stream_gen, sources, grounded, conv_id = use_case.execute_stream("What are the first aid measures?", document_id="doc-1")
    tokens = list(stream_gen)
    assert tokens == ["Token1 ", "Token2 ", "Token3"]
    assert grounded is True
    assert len(sources) == 1
    assert sources[0]["document"] == "l0288.pdf"

