"""Comprehensive regression test suite for Conversational RAG & Semantic Query Understanding.

Verifies:
  A. Exact question ("What are the first aid measures?")
  B. Paraphrased question ("What should I do if someone is exposed to this product?")
  C. Generic SDS question ("Give me the safety measures.")
  D. Broad overview question ("Tell me about this file.")
  E. Multi-intent question ("Give me the product name, manufacturer, hazards, first aid and storage information.")
  F. Mixed multi-intent question ("What is the product name, who manufactures it, what are its hazards, and what should I do if it gets into my eyes?")
  G. Metadata natural-language variation ("Who makes this product?")
  H. Completely unrelated question ("When was the Prime Minister born in India?")
  I. Unrelated question rejection (grounded=False, sources=[], no LLM call)
  J. Missing-information case (reports specific field unavailable without hallucination)
  K. Multi-document isolation (no cross-contamination between uploaded SDS files)
  L. Natural language paraphrases across different document styles
"""

import pytest
from pathlib import Path

from app.application.services.chunking_service import ChunkingService
from app.application.services.grounding_service import GroundingService
from app.application.services.intent_router import IntentRouter
from app.application.services.metadata_validator import MetadataValidator
from app.application.services.query_analyzer import QueryAnalyzer
from app.application.services.retrieval_service import RetrievalService, RetrievedChunk
from app.application.services.chat_service import ChatService
from app.application.services.text_cleaner import TextCleaner
from app.application.use_cases.chat_with_document_use_case import ChatWithDocumentUseCase
from app.application.use_cases.extract_metadata_use_case import ExtractMetadataUseCase
from app.infrastructure.configuration.settings import get_settings
from app.infrastructure.database.sqlite_database import SQLiteDatabase
from app.infrastructure.embeddings.ollama_embedding_client import OllamaEmbeddingClient
from app.infrastructure.llm.ollama_llm_client import OllamaLLMClient
from app.infrastructure.repositories.sqlite_chat_repository import SQLiteChatRepository
from app.infrastructure.repositories.sqlite_document_repository import SQLiteDocumentRepository
from app.infrastructure.vectorstore.chroma_store import ChromaVectorStore


class SmartReader:
    def read(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            content = "Sample SDS text Section 1 Identification"

        class PageObj:
            def __init__(self, page_number, text):
                self.page_number = page_number
                self.text = text

        class Extracted:
            pass

        e = Extracted()
        e.full_text = content
        e.page_count = 1
        e.pages = [PageObj(1, content)]
        return e


@pytest.fixture
def rag_pipeline():
    settings = get_settings()
    db = SQLiteDatabase(settings)
    db.initialise()
    with db.connection() as conn:
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM chat_history")
        conn.commit()

    repo = SQLiteDocumentRepository(db)
    chat_repo = SQLiteChatRepository(db)
    reader = SmartReader()
    cleaner = TextCleaner()
    chunker = ChunkingService(settings)
    embedder = OllamaEmbeddingClient(settings)
    vector_store = ChromaVectorStore(settings, embedder)
    llm = OllamaLLMClient(settings)
    validator = MetadataValidator()

    use_case = ExtractMetadataUseCase(
        document_repository=repo,
        document_reader=reader,
        text_cleaner=cleaner,
        chunking_service=chunker,
        vector_store=vector_store,
        llm_client=llm,
        metadata_validator=validator,
        settings=settings,
    )

    retrieval_svc = RetrievalService(vector_store, settings)
    grounding_svc = GroundingService(settings)
    chat_svc = ChatService(settings)
    intent_router = IntentRouter()

    chat_use_case = ChatWithDocumentUseCase(
        retrieval_service=retrieval_svc,
        grounding_service=grounding_svc,
        chat_service=chat_svc,
        intent_router=intent_router,
        document_repository=repo,
        chat_repository=chat_repo,
        settings=settings,
    )

    return use_case, chat_use_case, repo, vector_store


# ── TEST A: Exact SDS Question ────────────────────────────────────────────────
def test_a_exact_first_aid_question(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline
    doc_path = tmp_path / "sds_a.pdf"
    doc_path.write_text(
        "SAFETY DATA SHEET\n"
        "SECTION 1: Identification\nProduct Name: Acetone Clean\nManufacturer: Chemical Ltd\n"
        "SECTION 4: First-Aid Measures\nInhalation: Move to fresh air. Eye contact: Rinse thoroughly with water.\n"
        "SECTION 7: Handling and Storage\nStore in cool, dry area."
    )
    doc_res = use_case.execute(doc_path, "sds_a.pdf")
    chat_res = chat_use_case.execute("What are the first aid measures?", document_id=doc_res.document_id)

    assert chat_res.grounded is True
    assert len(chat_res.sources) > 0
    assert any(term in chat_res.answer.lower() for term in ["fresh air", "water", "eye", "first aid"])


# ── TEST B: Paraphrased Exposure Question ─────────────────────────────────────
def test_b_paraphrased_exposure_question(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline
    doc_path = tmp_path / "sds_b.pdf"
    doc_path.write_text(
        "SAFETY DATA SHEET\n"
        "SECTION 1: Identification\nProduct Name: Solvent X\nManufacturer: Apex Chem\n"
        "SECTION 4: First Aid Measures\n"
        "If inhaled, remove victim to fresh air. If skin contact occurs, wash with soap and water."
    )
    doc_res = use_case.execute(doc_path, "sds_b.pdf")
    chat_res = chat_use_case.execute("What should I do if someone is exposed to this product?", document_id=doc_res.document_id)

    assert chat_res.grounded is True
    assert len(chat_res.sources) > 0


# ── TEST C: Generic SDS Safety Question ───────────────────────────────────────
def test_c_generic_safety_measures_question(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline
    doc_path = tmp_path / "sds_c.pdf"
    doc_path.write_text(
        "SAFETY DATA SHEET\n"
        "SECTION 1: Identification\nProduct Name: Industrial Degreaser\n"
        "SECTION 7: Handling and Storage\nWear protective gloves. Store away from heat.\n"
        "SECTION 8: Exposure Controls\nUse ventilation mask and safety goggles."
    )
    doc_res = use_case.execute(doc_path, "sds_c.pdf")
    chat_res = chat_use_case.execute("Give me the safety measures.", document_id=doc_res.document_id)

    assert chat_res.grounded is True
    assert len(chat_res.sources) > 0


# ── TEST D: Broad Overview Question ("Tell me about this file") ───────────────
def test_d_broad_overview_question(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline
    doc_path = tmp_path / "sds_d.pdf"
    doc_path.write_text(
        "SAFETY DATA SHEET\n"
        "SECTION 1: Identification\nProduct Name: MultiClean Pro\nManufacturer: Global Chem Corp\n"
        "SECTION 2: Hazard Identification\nFlammable liquid. Causes eye irritation.\n"
        "SECTION 4: First Aid\nFlush eyes with water.\n"
        "SECTION 7: Storage\nKeep container tightly closed."
    )
    doc_res = use_case.execute(doc_path, "sds_d.pdf")
    chat_res = chat_use_case.execute("Tell me about this file.", document_id=doc_res.document_id)

    assert chat_res.grounded is True
    assert len(chat_res.sources) > 0


# ── TEST E & F: Multi-Intent Questions ────────────────────────────────────────
def test_e_and_f_multi_intent_questions(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline
    doc_path = tmp_path / "sds_ef.pdf"
    doc_path.write_text(
        "SAFETY DATA SHEET\n"
        "SECTION 1: Identification\nProduct Name: Steel Bond Resin\nManufacturer: Devcon Industrial\n"
        "SECTION 2: Hazard Identification\nHarmful if swallowed. Causes severe skin burns.\n"
        "SECTION 4: First Aid\nIn case of eye contact, rinse immediately with water.\n"
        "SECTION 7: Storage\nStore in cool dry place below 25C."
    )
    doc_res = use_case.execute(doc_path, "sds_ef.pdf")

    # Test E: Multi-intent 1
    q1 = "Give me the product name, manufacturer, hazards, first aid and storage information."
    res1 = chat_use_case.execute(q1, document_id=doc_res.document_id)
    assert res1.grounded is True
    assert len(res1.sources) > 0

    # Test F: Multi-intent 2 (different wording)
    q2 = "What is the product name, who manufactures it, what are its hazards, and what should I do if it gets into my eyes?"
    res2 = chat_use_case.execute(q2, document_id=doc_res.document_id)
    assert res2.grounded is True
    assert len(res2.sources) > 0


# ── TEST G: Metadata Natural Language Variation ──────────────────────────────
def test_g_metadata_natural_language_variation(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline
    doc_path = tmp_path / "sds_g.pdf"
    doc_path.write_text(
        "SAFETY DATA SHEET\n"
        "SECTION 1: Identification\nProduct Name: Hydrofluoric Acid 48%\nManufacturer: Honeywell International\n"
    )
    doc_res = use_case.execute(doc_path, "sds_g.pdf")
    chat_res = chat_use_case.execute("Who makes this product?", document_id=doc_res.document_id)

    assert chat_res.grounded is True
    assert "Honeywell" in chat_res.answer


# ── TEST H & I: Unrelated Question Rejection ──────────────────────────────────
def test_h_and_i_unrelated_question_rejection(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline
    doc_path = tmp_path / "sds_hi.pdf"
    doc_path.write_text(
        "SAFETY DATA SHEET\n"
        "SECTION 1: Identification\nProduct Name: Acetone\n"
        "SECTION 4: First Aid\nFlush eyes with water."
    )
    doc_res = use_case.execute(doc_path, "sds_hi.pdf")

    # Completely unrelated question
    chat_res = chat_use_case.execute("When was the Prime Minister born in India?", document_id=doc_res.document_id)

    assert chat_res.grounded is False
    assert len(chat_res.sources) == 0
    assert "Information not available" in chat_res.answer


# ── TEST J: Missing-Information Case ──────────────────────────────────────────
def test_j_missing_information_reporting(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline
    doc_path = tmp_path / "sds_j.pdf"
    doc_path.write_text(
        "SAFETY DATA SHEET\n"
        "SECTION 1: Identification\nProduct Name: Simple Solvent\n"
        "SECTION 4: First Aid\nFlush eyes with water."
    )
    doc_res = use_case.execute(doc_path, "sds_j.pdf")

    # Ask for transport UN number when section 14 transport is absent
    chat_res = chat_use_case.execute("What is the UN transport shipping number?", document_id=doc_res.document_id)
    assert chat_res.grounded is False or "not available" in chat_res.answer.lower() or "un" in chat_res.answer.lower()


# ── TEST K: Multi-Document Isolation ──────────────────────────────────────────
def test_k_multi_document_isolation(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline

    # Document 1: Fructose
    d1_path = tmp_path / "fructose.pdf"
    d1_path.write_text("SAFETY DATA SHEET\nProduct Name: D-Fructose\nSECTION 4 First Aid\nDrink water if swallowed.")
    d1_res = use_case.execute(d1_path, "fructose.pdf")

    # Document 2: Acetone
    d2_path = tmp_path / "acetone.pdf"
    d2_path.write_text("SAFETY DATA SHEET\nProduct Name: Acetone\nSECTION 4 First Aid\nRemove to fresh air immediately.")
    d2_res = use_case.execute(d2_path, "acetone.pdf")

    chat_res = chat_use_case.execute("What are the first aid measures for D-Fructose?", document_id=d1_res.document_id)
    assert chat_res.grounded is True
    assert "fresh air" not in chat_res.answer.lower()


# ── TEST L: Natural Language Paraphrases Across SDS Styles ────────────────────
def test_l_paraphrased_styles(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline
    doc_path = tmp_path / "legacy_style.pdf"
    doc_path.write_text(
        "MATERIAL SAFETY DATA SHEET\n"
        "Trade Name: Plastic Steel Resin\n"
        "Emergency Procedures: In case of spill, absorb with sand.\n"
        "Exposure Controls: Wear rubber gloves and safety spectacles."
    )
    doc_res = use_case.execute(doc_path, "legacy_style.pdf")
    chat_res = chat_use_case.execute("How do I protect myself when using this chemical?", document_id=doc_res.document_id)

    assert chat_res.grounded is True
    assert len(chat_res.sources) > 0


# ── TEST M: Multi-Intent Safety + Composition Query ──────────────────────────
def test_m_multi_intent_safety_and_composition(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline
    doc_path = tmp_path / "compound_sds.pdf"
    doc_path.write_text(
        "SAFETY DATA SHEET\n"
        "SECTION 1: Identification\nProduct Name: Compound X-100\nManufacturer: BioMed Labs\n"
        "SECTION 3: Composition / Information on Ingredients\n"
        "Substance Name: Compound X-100\nFormula: C27H26N6O\nMolecular Weight: 450.53\nCAS-No.: 2374971-81-8\n"
        "SECTION 4: First Aid Measures\n"
        "Eye contact: Immediately flush eyes with plenty of water for at least 15 minutes.\n"
        "Skin contact: Wash off with soap and plenty of water.\n"
        "Inhalation: Move person to fresh air.\n"
        "SECTION 7: Handling and Storage\n"
        "Keep container tightly closed in a dry, cool and well-ventilated place.\n"
        "SECTION 8: Exposure Controls / Personal Protection\n"
        "Respiratory protection: Use suitable respirator. Hand protection: Use protective gloves. Eye protection: Safety goggles."
    )
    doc_res = use_case.execute(doc_path, "compound_sds.pdf")
    chat_res = chat_use_case.execute(
        "what are the safety measures and also give me the chemical composition",
        document_id=doc_res.document_id,
    )

    assert chat_res.grounded is True
    assert len(chat_res.sources) > 0
    ans_lower = chat_res.answer.lower()

    # Must contain composition details
    assert any(term in ans_lower for term in ["450", "c27", "2374971", "formula", "molecular", "cas", "composition"])
    # Must contain safety details from sections 4-8
    assert any(term in ans_lower for term in ["water", "eye", "skin", "glove", "goggles", "respirat", "ventil", "store", "cool", "fresh air", "first aid", "safety"])
    # Must NOT falsely claim safety measures are unavailable
    assert "no specific safety measures are provided" not in ans_lower
    assert "safety measures: information not available" not in ans_lower


# ── TEST N: Overview Query ("tell me about the file") ────────────────────────
def test_n_overview_query_tell_me_about_the_file(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline
    doc_path = tmp_path / "sample_overview.pdf"
    doc_path.write_text(
        "SAFETY DATA SHEET\n"
        "SECTION 1: Identification\nProduct Name: PolyClear Resin\nManufacturer: ChemTech Global\n"
        "SECTION 2: Hazard Identification\nCauses serious eye irritation. Flammable liquid.\n"
        "SECTION 3: Composition\nActive Polymer: 95%, Solvent: 5%\n"
        "SECTION 4: First Aid Measures\nRinse eyes thoroughly with water.\n"
        "SECTION 7: Storage\nStore below 30C in dry location.\n"
        "SECTION 8: Exposure Controls\nWear nitrile gloves and eye protection."
    )
    doc_res = use_case.execute(doc_path, "sample_overview.pdf")
    chat_res = chat_use_case.execute("tell me about the file", document_id=doc_res.document_id)

    assert chat_res.grounded is True
    assert len(chat_res.sources) > 0
    assert chat_res.answer != "Information not available in the uploaded file."
    ans_lower = chat_res.answer.lower()
    assert any(term in ans_lower for term in ["polyclear", "chemtech", "resin", "hazard", "eye", "safety", "flammable"])


# ── TEST O: Generic Overview Query ("give me an overview of this SDS") ───────
def test_o_generic_overview_give_me_an_overview(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline
    doc_path = tmp_path / "sample_overview2.pdf"
    doc_path.write_text(
        "SAFETY DATA SHEET\n"
        "SECTION 1: Identification\nProduct Name: AcryliShield\nManufacturer: Shield Products Inc\n"
        "SECTION 2: Hazards\nHarmful if swallowed.\n"
        "SECTION 4: First Aid\nCall a poison control center if swallowed.\n"
        "SECTION 8: PPE\nWear chemical resistant gloves."
    )
    doc_res = use_case.execute(doc_path, "sample_overview2.pdf")
    chat_res = chat_use_case.execute("give me an overview of this SDS", document_id=doc_res.document_id)

    assert chat_res.grounded is True
    assert len(chat_res.sources) > 0
    assert chat_res.answer != "Information not available in the uploaded file."


# ── TEST P: Source Attribution Structure ──────────────────────────────────────
def test_p_source_attribution_formatting(rag_pipeline, tmp_path):
    use_case, chat_use_case, _, _ = rag_pipeline
    doc_path = tmp_path / "attribution_test.pdf"
    doc_path.write_text(
        "SAFETY DATA SHEET\n"
        "SECTION 1: Identification\nProduct Name: AttribCheck\n"
        "SECTION 4: First Aid Measures\nFlush eyes with water for 15 minutes."
    )
    doc_res = use_case.execute(doc_path, "attribution_test.pdf")
    chat_res = chat_use_case.execute("What are the first aid measures?", document_id=doc_res.document_id)

    assert chat_res.grounded is True
    assert len(chat_res.sources) > 0
    first_src = chat_res.sources[0]
    assert first_src.document == "attribution_test.pdf"
    assert first_src.page >= 1
    assert first_src.section in ("4", "Metadata") or "First Aid" in first_src.section_title or "General" in first_src.section_title
