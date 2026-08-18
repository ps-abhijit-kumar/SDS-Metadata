"""SDS Document Intelligence & Conversational RAG Platform — Streamlit Frontend.

A production-oriented UI supporting multi-PDF upload (5+ documents), SHA-256 duplicate detection,
separate 4-field metadata result cards per PDF, and document-grounded streaming RAG chat.
"""

from __future__ import annotations

import json
import os
import httpx
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_TIMEOUT = 1200  # seconds

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SDS Document Intelligence & Conversational RAG",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #2196F3;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .metric-label { font-size: 0.72rem; color: #555; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
    .metric-value { font-size: 1.05rem; font-weight: 600; color: #111; margin-top: 0.2rem; }
    .status-completed { color: #2e7d32; font-weight: 600; background: #e8f5e9; padding: 3px 8px; border-radius: 4px; }
    .status-duplicate { color: #0277bd; font-weight: 600; background: #e1f5fe; padding: 3px 8px; border-radius: 4px; }
    .status-failed    { color: #c62828; font-weight: 600; background: #ffebee; padding: 3px 8px; border-radius: 4px; }
    .status-processing{ color: #e65100; font-weight: 600; background: #fff3e0; padding: 3px 8px; border-radius: 4px; }
    .source-box {
        background: #f1f3f4;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── API Helpers ───────────────────────────────────────────────────────────────

def _get_health() -> dict:
    try:
        r = httpx.get(f"{API_BASE_URL}/health", timeout=4)
        return r.json() if r.status_code == 200 else {"status": "offline"}
    except Exception:
        return {"status": "offline"}


def _extract(files: list) -> dict | None:
    try:
        file_tuples = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
        timeout = httpx.Timeout(timeout=10.0, read=API_TIMEOUT)
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{API_BASE_URL}/api/v1/extract", files=file_tuples)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Upload error: {exc}")
        return None


def _get_documents() -> list[dict]:
    try:
        r = httpx.get(f"{API_BASE_URL}/api/v1/documents", timeout=10)
        r.raise_for_status()
        return r.json().get("documents", [])
    except Exception:
        return []


def _delete_document(document_id: str) -> bool:
    try:
        r = httpx.delete(f"{API_BASE_URL}/api/v1/documents/{document_id}", timeout=10)
        return r.status_code == 204
    except Exception:
        return False


def _stream_chat(question: str, document_id: str):
    """Call SSE stream endpoint and yield tokens while updating sources."""
    url = f"{API_BASE_URL}/api/v1/chat/stream"
    payload = {"question": question, "document_id": document_id}

    sources_acc = []
    grounded = True

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        evt = data.get("event")
                        if evt == "metadata":
                            sources_acc.extend(data.get("sources", []))
                            grounded = data.get("grounded", True)
                        elif evt == "token":
                            yield data.get("token", "")
                    except Exception:
                        continue
    except Exception as exc:
        yield f"\n[Connection Error: {exc}]"

    st.session_state["current_sources"] = sources_acc
    st.session_state["current_grounded"] = grounded


# ── UI Rendering Components ───────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    s = status.lower()
    icons = {"completed": "✅", "duplicate": "⚡", "failed": "❌", "processing": "⏳"}
    icon = icons.get(s, "")
    return f'<span class="status-{s}">{icon} {status.upper()}</span>'


def _metadata_card(label: str, value: str | None) -> None:
    display = value if value else "Not available in document"
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{display}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_document_card(result: dict) -> None:
    status = result.get("status", "unknown")

    with st.container():
        st.markdown(f"### 📄 {result.get('filename', 'Unknown')}")
        col_status, col_id = st.columns([1.5, 3])
        with col_status:
            st.markdown(_status_badge(status), unsafe_allow_html=True)
        with col_id:
            st.caption(f"ID: `{result.get('document_id', '')}`")

        if status in ("completed", "duplicate"):
            c1, c2 = st.columns(2)
            with c1:
                _metadata_card("Product Name", result.get("product_name"))
                _metadata_card("Company Name", result.get("company_name"))
            with c2:
                _metadata_card("Language", result.get("language"))
                _metadata_card("Jurisdiction", result.get("jurisdiction"))
        elif status == "failed":
            st.error(f"Extraction failed: {result.get('error_message', 'Unknown error')}")

        st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧪 SDS Intelligence & Conversational RAG")
    st.caption("FastAPI · Ollama · qwen3:4b-instruct · ChromaDB")
    st.divider()

    health = _get_health()
    if health.get("status") == "ok":
        st.success("Backend API: Online", icon="🟢")
    elif health.get("fastapi") == "online" and health.get("status") == "degraded":
        st.warning("Backend API: Online (Ollama Busy)", icon="🟡")
    else:
        st.error("Backend API: Offline", icon="🔴")

    st.markdown("**System Health Verification**")
    st.caption(f"• SQLite: `{health.get('sqlite', 'check...')}`")
    st.caption(f"• Ollama: `{health.get('ollama', 'check...')}`")
    st.caption(f"• Chat Model: `{health.get('chat_model', 'qwen3:4b-instruct')}`")
    st.caption(f"• Embeddings: `{health.get('embedding_model', 'nomic-embed-text')}`")

    st.divider()
    st.markdown("**Core Guarantees**")
    st.markdown("""
    - **Zero Cloud APIs**: 100% local inference.
    - **SHA-256 Hash**: Fast duplicate detection.
    - **Strict Grounding**: Only answers from uploaded SDS context.
    - **Mandatory Fallback**: *"Information not available in the uploaded file."*
    """)


# ── Main Header & Tabs ────────────────────────────────────────────────────────

st.title("Safety Data Sheet (SDS) Document Intelligence Platform")
st.caption("Upload up to 5+ SDS PDF documents to extract mandatory metadata and execute grounded RAG conversations.")

tab_upload, tab_chat, tab_history = st.tabs(["📤 Upload & Extract", "💬 Conversational RAG Chat", "📋 History & Audit"])


# ── TAB 1: UPLOAD & EXTRACT ───────────────────────────────────────────────────

with tab_upload:
    uploaded_files = st.file_uploader(
        "Upload Safety Data Sheet (SDS) PDF documents (Support 5+ files)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or multiple Safety Data Sheet PDF files.",
    )

    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} PDF file(s) selected.")

        if st.button("Process & Extract Metadata", type="primary", use_container_width=True):
            with st.spinner("Processing documents & executing local RAG metadata extraction..."):
                response = _extract(uploaded_files)

            if response:
                results = response.get("results", [])
                completed = sum(1 for r in results if r.get("status") == "completed")
                duplicates = sum(1 for r in results if r.get("status") == "duplicate")
                failed = sum(1 for r in results if r.get("status") == "failed")

                if completed:
                    st.success(f"Extracted metadata for {completed} document(s).")
                if duplicates:
                    st.info(f"⚡ {duplicates} duplicate document(s) resolved instantly from cache.")
                if failed:
                    st.warning(f"❌ {failed} document(s) failed.")

                st.markdown("## Extracted Metadata Results")
                for res in results:
                    _render_document_card(res)

                st.session_state["last_results"] = results


# ── TAB 2: CONVERSATIONAL RAG CHAT ────────────────────────────────────────────

with tab_chat:
    documents = _get_documents()
    completed_docs = [d for d in documents if d.get("status") in ("completed", "duplicate")]

    if not completed_docs:
        st.warning("No processed SDS documents found. Please upload at least one PDF in the 'Upload & Extract' tab first.")
    else:
        # Document Selector - Deduplicate by filename so Chat uses unique canonical documents
        doc_options = {"All Uploaded Documents": "all"}
        seen_filenames = set()
        for doc in completed_docs:
            fn = doc.get("filename")
            if fn and fn not in seen_filenames:
                seen_filenames.add(fn)
                label = f"{fn} (Product: {doc.get('product_name') or 'N/A'})"
                doc_options[label] = doc.get("document_id")

        selected_label = st.selectbox("💬 Chat Scope:", list(doc_options.keys()))
        selected_doc_id = doc_options[selected_label]

        # Initialize Chat History in Session State
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []

        def _render_source_item(src: dict):
            if not isinstance(src, dict) or not src:
                return
            doc = src.get("document", "Unknown Document")
            source_type = src.get("source_type", "document_content")
            sec_num = str(src.get("section", "")).strip()
            sec_title = str(src.get("section_title", "")).strip()
            page = src.get("page", 1)

            if source_type == "document_metadata" or sec_num == "Metadata" or "metadata" in sec_title.lower():
                st.markdown(f"- **Document:** `{doc}` | **Source:** Extracted Verified Metadata | **Page:** `{page}`")
            else:
                if sec_num and sec_num not in ("0", "N/A", "None") and "section" not in sec_title.lower():
                    sec_display = f"Section {sec_num} — {sec_title}" if sec_title else f"Section {sec_num}"
                else:
                    sec_display = sec_title if sec_title else "General Content"

                st.markdown(
                    f"- **Document:** `{doc}` | **{sec_display}** | **Page:** `{page}`"
                )

        # Display Chat History
        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                valid_sources = [s for s in msg.get("sources", []) if isinstance(s, dict) and s]
                if valid_sources:
                    with st.expander("📌 Source Attribution", expanded=False):
                        for src in valid_sources:
                            _render_source_item(src)

        # Chat Input Prompt
        if user_prompt := st.chat_input("Ask a question about the uploaded SDS document(s)..."):
            st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                st.session_state["current_sources"] = []
                st.session_state["current_grounded"] = True

                # Stream response progressively
                response_text = st.write_stream(_stream_chat(user_prompt, selected_doc_id))

                sources = st.session_state.get("current_sources", [])
                grounded = st.session_state.get("current_grounded", True)
                valid_sources = [s for s in sources if isinstance(s, dict) and s]
                if grounded and valid_sources:
                    with st.expander("📌 Source Attribution", expanded=True):
                        for src in valid_sources:
                            _render_source_item(src)

            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "sources": valid_sources if grounded else [],
                }
            )


# ── TAB 3: HISTORY & AUDIT ────────────────────────────────────────────────────

with tab_history:
    if st.button("🔄 Refresh Document History"):
        st.rerun()

    documents = _get_documents()

    if not documents:
        st.info("No document history available.")
    else:
        st.caption(f"Total documents in storage: {len(documents)}")
        for doc in documents:
            title = f"{'⚡' if doc.get('status') == 'duplicate' else '📄'} {doc.get('filename')} — {doc.get('status', '').upper()}"
            with st.expander(title, expanded=False):
                _render_document_card(doc)
                doc_id = doc.get("document_id")
                if doc_id and st.button("Delete Record & Vectors", key=f"del_{doc_id}"):
                    if _delete_document(doc_id):
                        st.success("Deleted document record and ChromaDB vectors.")
                        st.rerun()
