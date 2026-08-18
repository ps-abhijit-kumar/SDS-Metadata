"""Unit tests for ChunkingService."""

import pytest
from unittest.mock import MagicMock
from app.application.services.chunking_service import ChunkingService
from app.infrastructure.configuration.settings import Settings


@pytest.fixture
def settings() -> Settings:
    s = MagicMock(spec=Settings)
    s.chunk_size = 400
    s.chunk_overlap = 50
    return s


@pytest.fixture
def service(settings) -> ChunkingService:
    return ChunkingService(settings)


def test_chunk_empty_text_returns_empty(service):
    result = service.chunk("", "doc-001")
    assert result == []


def test_chunk_produces_chunks(service):
    text = "Section 1 Identification\n\n" + ("Product Name: Test Chemical. " * 30)
    chunks = service.chunk(text, "doc-001")
    assert len(chunks) > 0


def test_chunk_assigns_document_id(service):
    text = "Section 1 Identification\n\n" + ("Some SDS text. " * 40)
    chunks = service.chunk(text, "doc-abc")
    for chunk in chunks:
        assert chunk.document_id == "doc-abc"


def test_chunk_assigns_sequential_indices(service):
    text = "Safety Data Sheet\n\n" + ("Content text for testing. " * 50)
    chunks = service.chunk(text, "doc-002")
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_chunk_detects_section_1(service):
    text = (
        "SECTION 1 — IDENTIFICATION\n"
        "Product Name: Aceton\n"
        "Manufacturer: ABC Corp\n" * 30
    )
    chunks = service.chunk(text, "doc-003")
    sections = [c.section_number for c in chunks if c.section_number is not None]
    assert 1 in sections


def test_chunk_detects_section_15_regulatory(service):
    text = (
        "SECTION 15 — REGULATORY INFORMATION\n"
        "This SDS is prepared in accordance with OSHA HazCom 2012.\n" * 30
    )
    chunks = service.chunk(text, "doc-004")
    sections = [c.section_number for c in chunks if c.section_number is not None]
    assert 15 in sections


def test_chunk_metadata_contains_document_id(service):
    text = "SDS content " * 100
    chunks = service.chunk(text, "doc-005")
    for chunk in chunks:
        assert chunk.metadata["document_id"] == "doc-005"


def test_chunk_detects_multilingual_and_generic_section_headers(service):
    # 1. English
    en_chunks = service.chunk("SECTION 4: FIRST AID MEASURES\nFlush eyes with water.\n" * 10, "doc-en")
    assert any(c.section_number == 4 for c in en_chunks)

    # 2. Spanish
    es_chunks = service.chunk("SECCIÓN 4: Primeros auxilios\nLavar los ojos con abundante agua.\n" * 10, "doc-es")
    assert any(c.section_number == 4 for c in es_chunks)

    # 3. German
    de_chunks = service.chunk("ABSCHNITT 4: Erste-Hilfe-Maßnahmen\nAugen gründlich mit Wasser spülen.\n" * 10, "doc-de")
    assert any(c.section_number == 4 for c in de_chunks)

    # 4. Swedish
    sv_chunks = service.chunk("AVSNITT 4: Åtgärder vid första hjälpen\nSkölj ögonen med rikligt med vatten.\n" * 10, "doc-sv")
    assert any(c.section_number == 4 for c in sv_chunks)

    # 5. Number-only format
    num_chunks = service.chunk("4. FIRST AID MEASURES\nRinse thoroughly with clean water.\n" * 10, "doc-num")
    assert any(c.section_number == 4 for c in num_chunks)


def test_repeated_page_headers_do_not_override_active_section(service):
    # Section 7 text followed by a recurring European REACH header should stay Section 7
    text = (
        "AVSNITT 7: Hantering och lagring\n"
        "Skyddsåtgärder för säker hantering.\n"
        "Säkerhetsdatablad enligt förordning (EU) nr 1907/2006 (REACH)\n"
        "DESMODUR L 75\n" * 10
    )
    chunks = service.chunk(text, "doc-header")
    assert all(c.section_number == 7 for c in chunks)

