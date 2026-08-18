"""Chat service for RAG QA generation with streaming and thinking tag removal."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Generator
import httpx

from app.application.dto.chat_dto import SourceCitationDTO
from app.application.services.retrieval_service import RetrievedChunk
from app.infrastructure.configuration.settings import Settings

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_STRICT_SYSTEM_PROMPT = """\
You are an SDS document assistant.

Your only source of truth is the document context supplied with the current request.

Answer ONLY using the supplied context.

Do not use:
- general knowledge
- pretrained knowledge
- assumptions
- external knowledge
- information from unrelated documents
- unsupported inference

If the answer cannot be directly supported by the supplied document context, respond:
{fallback_message}

CRITICAL RULES FOR MULTI-PART / MULTI-INTENT QUESTIONS:
1. Address EVERY requested topic or component in the user's question (e.g. Chemical Composition, Safety Measures, First Aid, Storage, PPE).
2. Use clear markdown sub-headings for each requested topic (e.g. "### Chemical Composition", "### Safety Measures").
3. For "safety measures", summarize all safety-related guidance present in the context (First Aid, Fire Fighting, Accidental Release / Spill, Safe Handling & Storage, PPE / Exposure Controls).
4. If information for a specific requested topic is NOT supported by the context, explicitly state:
   "[Topic]: Information not available in the uploaded document."
   Do NOT omit the requested topic and do NOT guess or use general knowledge.

CRITICAL RULES FOR OVERVIEW / SUMMARY QUESTIONS:
1. When asked for an overview or summary of the document/product/file, provide a structured summary using only the provided metadata and retrieved context:
   - Product & Manufacturer Identification
   - Language & Jurisdiction
   - Chemical Composition / Identification (if present in context)
   - Key Hazards & Classification (if present in context)
   - Key Safety & Emergency Measures (First Aid, Fire, Spills, Handling, PPE if present)
2. Do not invent missing fields; only summarize what is directly supported by the context.

CRITICAL RULES FOR MULTI-DOCUMENT CONTEXT:
1. Keep evidence for each document completely separate.
2. Do NOT merge, combine, or cross-pollinate instructions across different documents.
3. State explicitly which document each statement or instruction belongs to.

Never guess.
Never fabricate values.
Every factual answer must be supported by the supplied SDS context.
Keep answers clear, concise, and directly supported by the evidence.
"""

_CHAT_USER_TEMPLATE = """\
DOCUMENT CONTEXT:
{context}

QUESTION:
{question}
"""


class ChatService:
    """Manages chat prompt construction, LLM calls, and streaming."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_prompt(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        is_multi_doc: bool = False,
        doc_metadata: dict | list[dict] | None = None,
        is_overview: bool = False,
    ) -> tuple[str, str]:
        fallback_msg = (
            self._settings.multi_doc_fallback_response
            if is_multi_doc
            else self._settings.fallback_response
        )
        system_prompt = _STRICT_SYSTEM_PROMPT.format(fallback_message=fallback_msg)

        context_blocks = []

        # Add verified document metadata block if available
        if doc_metadata:
            meta_list = doc_metadata if isinstance(doc_metadata, list) else [doc_metadata]
            for m in meta_list:
                fn = m.get("filename", "Uploaded Document")
                prod = m.get("product_name") or "Not specified in metadata"
                comp = m.get("company_name") or "Not specified in metadata"
                lang = m.get("language") or "Not specified in metadata"
                jur = m.get("jurisdiction") or "Not specified in metadata"
                meta_block = (
                    f"[Document Metadata | Document: {fn}]\n"
                    f"- Product Name: {prod}\n"
                    f"- Manufacturer / Supplier: {comp}\n"
                    f"- Language: {lang}\n"
                    f"- Regulatory Jurisdiction: {jur}"
                )
                context_blocks.append(meta_block)

        for i, c in enumerate(chunks, 1):
            sec_display = str(c.section).strip() if c.section and str(c.section).strip() not in ("0", "") else "N/A"
            block = (
                f"[Source {i} | Document: {c.filename} | Section: {sec_display} | Title: {c.section_title} | Page: {c.page}]\n"
                f"{c.text}"
            )
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)
        user_prompt = _CHAT_USER_TEMPLATE.format(context=context_str, question=question)
        return system_prompt, user_prompt

    def extract_sources(
        self,
        chunks: list[RetrievedChunk],
        doc_metadata: dict | list[dict] | None = None,
    ) -> list[SourceCitationDTO]:
        """Extract unique source citations from retrieved chunks and metadata."""
        seen = set()
        sources: list[SourceCitationDTO] = []

        if doc_metadata:
            meta_list = doc_metadata if isinstance(doc_metadata, list) else [doc_metadata]
            for m in meta_list:
                doc_name = m.get("filename", "document.pdf")
                key = (doc_name, 1, "Metadata", "Extracted Document Metadata")
                if key not in seen:
                    seen.add(key)
                    sources.append(
                        SourceCitationDTO(
                            document=doc_name,
                            page=1,
                            section="Metadata",
                            section_title="Extracted Document Metadata",
                            source_type="document_metadata",
                        )
                    )

        for c in chunks:
            sec_str = str(c.section).strip() if c.section and str(c.section).strip() not in ("0", "") else "N/A"
            sec_title = c.section_title if c.section_title and c.section_title.strip() else ("Section " + sec_str if sec_str != "N/A" else "General Information")
            key = (c.filename, c.page, sec_str, sec_title)
            if key not in seen:
                seen.add(key)
                sources.append(
                    SourceCitationDTO(
                        document=c.filename,
                        page=c.page,
                        section=sec_str,
                        section_title=sec_title,
                    )
                )
        return sources

    def generate_chat_response(self, system_prompt: str, user_prompt: str) -> str:
        """Synchronous chat response using Ollama API."""
        answer, _ = self.generate_chat_response_with_metrics(system_prompt, user_prompt)
        return answer

    def generate_chat_response_with_metrics(self, system_prompt: str, user_prompt: str) -> tuple[str, float]:
        """Synchronous chat response with timing metrics."""
        url = f"{self._settings.ollama_base_url}/api/chat"
        target_model = self._settings.chat_model

        logger.info("CHAT LLM | model=%s | prompt_len=%d", target_model, len(system_prompt) + len(user_prompt))

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "keep_alive": self._settings.ollama_keep_alive,
            "options": {
                "temperature": self._settings.ollama_temperature,
                "num_predict": self._settings.ollama_num_predict,
                "think": False,
            },
        }

        tokens = []
        first_token_ms = 0.0
        start_time = time.time()

        try:
            with httpx.Client(timeout=self._settings.ollama_timeout_seconds) as client:
                with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    in_think_block = False
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk_data = json.loads(line)
                            content = chunk_data.get("message", {}).get("content", "")
                            if not content:
                                continue

                            if "<think>" in content:
                                in_think_block = True
                                continue
                            if "</think>" in content:
                                in_think_block = False
                                continue
                            if in_think_block:
                                continue

                            if not tokens and first_token_ms == 0.0:
                                first_token_ms = (time.time() - start_time) * 1000

                            tokens.append(content)
                        except Exception:
                            continue

            raw_answer = "".join(tokens).strip()
            cleaned_answer = _THINK_RE.sub("", raw_answer).strip()
            if first_token_ms == 0.0:
                first_token_ms = (time.time() - start_time) * 1000
            return cleaned_answer, first_token_ms
        except Exception as exc:
            logger.error("Chat generation failed: %s", exc)
            raise RuntimeError(f"Chat generation failed: {exc}") from exc

    def stream_chat_response(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        """Streaming chat response generator using Ollama SSE API."""
        url = f"{self._settings.ollama_base_url}/api/chat"
        target_model = self._settings.chat_model

        logger.info("CHAT STREAM LLM | model=%s | prompt_len=%d", target_model, len(system_prompt) + len(user_prompt))

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "keep_alive": self._settings.ollama_keep_alive,
            "options": {
                "temperature": self._settings.ollama_temperature,
                "num_predict": self._settings.ollama_num_predict,
                "think": False,
            },
        }

        try:
            with httpx.Client(timeout=self._settings.ollama_timeout_seconds) as client:
                with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    in_think_block = False
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk_data = json.loads(line)
                            content = chunk_data.get("message", {}).get("content", "")
                            if not content:
                                continue

                            # Handle potential inline <think> tags during streaming
                            if "<think>" in content:
                                in_think_block = True
                                continue
                            if "</think>" in content:
                                in_think_block = False
                                continue
                            if in_think_block:
                                continue

                            yield content
                        except Exception:
                            continue
        except Exception as exc:
            logger.error("Chat streaming failed: %s", exc)
            yield f"\n[Error: {exc}]"
