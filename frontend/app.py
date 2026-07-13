"""AI Document Intelligence Platform — Streamlit Frontend.

This frontend is a pure HTTP client to the FastAPI backend.
It never imports from the app/ package directly.
All communication happens via the REST API.
"""

from __future__ import annotations

import os
from datetime import datetime

import httpx
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_TIMEOUT  = 1200  # seconds — allow time for LLM inference (20 minutes max)

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Document Intelligence Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #2196F3;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }
    .metric-label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.1rem; font-weight: 600; color: #1a1a1a; margin-top: 0.25rem; }
    .status-completed { color: #2e7d32; font-weight: 600; }
    .status-failed    { color: #c62828; font-weight: 600; }
    .status-processing{ color: #e65100; font-weight: 600; }
    .status-pending   { color: #546e7a; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def _api_health() -> bool:
    """Return True if the FastAPI backend is reachable."""
    try:
        r = httpx.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _extract(files: list) -> dict | None:
    """POST files to /api/v1/extract and return the JSON response."""
    try:
        file_tuples = [
            ("files", (f.name, f.getvalue(), "application/pdf"))
            for f in files
        ]
        # Use explicit timeout: (connect, read, write, pool)
        # Read timeout needs to be long for LLM inference
        timeout = httpx.Timeout(timeout=10.0, read=API_TIMEOUT)
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{API_BASE_URL}/api/v1/extract", files=file_tuples)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as exc:
        st.error(f"Request timed out after {API_TIMEOUT}s. The LLM may be taking longer than expected. Try again.")
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
    except httpx.ConnectError:
        st.error("Cannot connect to the API. Make sure the FastAPI server is running.")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
    return None


def _get_documents() -> list[dict]:
    """GET /api/v1/documents and return the list of document records."""
    try:
        r = httpx.get(f"{API_BASE_URL}/api/v1/documents", timeout=15)
        r.raise_for_status()
        return r.json().get("documents", [])
    except Exception:
        return []


def _delete_document(document_id: str) -> bool:
    """DELETE /api/v1/documents/{id}."""
    try:
        r = httpx.delete(f"{API_BASE_URL}/api/v1/documents/{document_id}", timeout=10)
        return r.status_code == 204
    except Exception:
        return False


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    css_class = f"status-{status.lower()}"
    icons = {
        "completed": "✅",
        "failed": "❌",
        "processing": "⏳",
        "pending": "🕐",
    }
    icon = icons.get(status.lower(), "")
    return f'<span class="{css_class}">{icon} {status.upper()}</span>'


def _metadata_card(label: str, value: str | None) -> None:
    display = value if value else "—"
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{display}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_result(result: dict) -> None:
    """Render one extraction result as a card."""
    status = result.get("status", "unknown")

    with st.container():
        st.markdown(f"#### 📄 {result.get('filename', 'Unknown')}")
        col_status, col_id = st.columns([1, 3])
        with col_status:
            st.markdown(_status_badge(status), unsafe_allow_html=True)
        with col_id:
            doc_id = result.get("document_id", "")
            if doc_id:
                st.caption(f"ID: `{doc_id}`")

        if status == "completed":
            c1, c2 = st.columns(2)
            with c1:
                _metadata_card("Product Name", result.get("product_name"))
                _metadata_card("Company Name", result.get("company_name"))
            with c2:
                _metadata_card("Language", result.get("language"))
                _metadata_card("Jurisdiction", result.get("jurisdiction"))

            # Display debug metadata if available
            debug_meta = result.get("debug_metadata")
            if debug_meta:
                with st.expander("🔍 Debug Information", expanded=False):
                    # Timing information
                    stage_timings = debug_meta.get("stage_timings", {})
                    if stage_timings:
                        st.markdown("**Pipeline Timings (ms)**")
                        for stage, timing_ms in sorted(stage_timings.items()):
                            st.caption(f"  {stage}: {timing_ms:.1f}ms")
                        total_ms = debug_meta.get("total_time_ms", 0)
                        st.markdown(f"**Total: {total_ms:.1f}ms**")

                    # Retrieved chunks
                    chunks = debug_meta.get("retrieved_chunks", [])
                    if chunks:
                        st.markdown("**Retrieved Chunks**")
                        for i, chunk in enumerate(chunks, 1):
                            st.text_area(
                                f"Chunk {i}",
                                value=chunk[:500] + ("..." if len(chunk) > 500 else ""),
                                height=100,
                                disabled=True,
                                key=f"chunk_{i}",
                            )

                    # LLM prompt
                    prompt = debug_meta.get("llm_prompt", "")
                    if prompt:
                        st.markdown("**LLM Prompt**")
                        st.text_area(
                            "Full prompt sent to Qwen3:8B",
                            value=prompt[:500] + ("..." if len(prompt) > 500 else ""),
                            height=150,
                            disabled=True,
                        )

                    # LLM response
                    response = debug_meta.get("llm_raw_response", "")
                    if response:
                        st.markdown("**LLM Raw Response**")
                        st.text(response)

                    # Parsed metadata
                    parsed = debug_meta.get("parsed_metadata", {})
                    if parsed:
                        st.markdown("**Parsed Metadata**")
                        st.json(parsed)

        elif status == "failed":
            st.error(f"Extraction failed: {result.get('error_message', 'Unknown error')}")

        st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧪 AI Document\nIntelligence Platform")
    st.caption("Local RAG · Ollama · Qwen3:8B")
    st.divider()

    backend_ok = _api_health()
    if backend_ok:
        st.success("Backend: Online", icon="🟢")
    else:
        st.error("Backend: Offline", icon="🔴")
        st.caption(f"Expected at {API_BASE_URL}")

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("""
    1. Upload PDF  
    2. Extract text (PyMuPDF)  
    3. Clean & normalise  
    4. Semantic chunking  
    5. Embed (nomic-embed-text)  
    6. Store (ChromaDB)  
    7. Retrieve relevant chunks  
    8. Prompt Qwen3:8B via Ollama  
    9. Parse & validate result  
    10. Persist to SQLite  
    """)
    st.divider()
    st.caption("Runs 100% locally. No cloud APIs.")


# ── Main content ──────────────────────────────────────────────────────────────

st.title("SDS Metadata Extraction")
st.caption("Upload Safety Data Sheet PDFs to extract product name, language, and jurisdiction.")

tab_extract, tab_history = st.tabs(["📤 Extract", "📋 History"])

# ── Extract tab ───────────────────────────────────────────────────────────────
with tab_extract:
    if not backend_ok:
        st.warning(
            "The FastAPI backend is not reachable. "
            "Start it with:\n\n```\nuvicorn app.main:app --reload\n```",
            icon="⚠️",
        )

    uploaded_files = st.file_uploader(
        "Upload SDS PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more Safety Data Sheet PDF files.",
        disabled=not backend_ok,
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected.")

        col_btn, col_clear = st.columns([1, 5])
        with col_btn:
            run_btn = st.button(
                "Extract Metadata",
                type="primary",
                disabled=not backend_ok,
                use_container_width=True,
            )

        if run_btn:
            with st.spinner("Running RAG pipeline… this may take up to a few minutes per document."):
                response = _extract(uploaded_files)

            if response:
                results = response.get("results", [])
                completed = sum(1 for r in results if r.get("status") == "completed")
                failed = sum(1 for r in results if r.get("status") == "failed")

                if completed:
                    st.success(f"Extracted metadata for {completed} document(s).")
                if failed:
                    st.warning(f"{failed} document(s) failed.")

                st.markdown("### Results")
                for result in results:
                    _render_result(result)

                # Store in session for display persistence
                st.session_state["last_results"] = results

    elif "last_results" in st.session_state:
        st.markdown("### Last Results")
        for result in st.session_state["last_results"]:
            _render_result(result)

# ── History tab ───────────────────────────────────────────────────────────────
with tab_history:
    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    documents = _get_documents()

    if not documents:
        st.info("No documents have been processed yet.")
    else:
        st.caption(f"{len(documents)} document(s) in history.")

        for doc in documents:
            with st.expander(
                f"{'✅' if doc.get('status') == 'completed' else '❌'} "
                f"{doc.get('filename', 'Unknown')} — {doc.get('status', '').upper()}",
                expanded=False,
            ):
                _render_result(doc)

                doc_id = doc.get("document_id", "")
                if doc_id and st.button(
                    "Delete record",
                    key=f"del_{doc_id}",
                    type="secondary",
                ):
                    if _delete_document(doc_id):
                        st.success("Record deleted.")
                        st.rerun()
                    else:
                        st.error("Failed to delete record.")
