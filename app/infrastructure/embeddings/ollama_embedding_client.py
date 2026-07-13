"""Ollama embedding client using LangChain's OllamaEmbeddings wrapper.

Implements the EmbeddingProvider interface so the application layer
is never aware of which embedding service is used.
"""

from __future__ import annotations

import logging

from langchain_ollama import OllamaEmbeddings

from app.domain.exceptions.base import EmbeddingException
from app.infrastructure.configuration.settings import Settings

logger = logging.getLogger(__name__)


class OllamaEmbeddingClient:
    """Wraps LangChain OllamaEmbeddings and adds error handling + batching."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.ollama_embedding_model
        self._batch_size = settings.embedding_batch_size
        self._client = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
        logger.info(
            "OllamaEmbeddingClient ready | model=%s | base_url=%s",
            settings.ollama_embedding_model,
            settings.ollama_base_url,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text strings.

        Processes in batches to avoid overwhelming the Ollama server.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size

        for batch_index in range(0, len(texts), self._batch_size):
            batch = texts[batch_index : batch_index + self._batch_size]
            batch_num = batch_index // self._batch_size + 1
            logger.debug(
                "Embedding batch %d/%d | size=%d", batch_num, total_batches, len(batch)
            )
            try:
                embeddings = self._client.embed_documents(batch)
                all_embeddings.extend(embeddings)
            except Exception as exc:
                raise EmbeddingException(
                    f"Failed to embed batch {batch_num}/{total_batches}: {exc}"
                ) from exc

        logger.debug("Embedded %d texts -> %d vectors", len(texts), len(all_embeddings))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generate a single query embedding for retrieval."""
        try:
            return self._client.embed_query(text)
        except Exception as exc:
            raise EmbeddingException(f"Failed to embed query: {exc}") from exc

    @property
    def langchain_embeddings(self) -> OllamaEmbeddings:
        """Return the underlying LangChain embeddings object for use with ChromaDB."""
        return self._client
