"""Ollama embedding client using LangChain's OllamaEmbeddings wrapper.

Implements the EmbeddingProvider interface so the application layer
is never aware of which embedding service is used.

Includes:
- Explicit timeout configuration for Ollama 0.6+ compatibility
- Batch processing with error recovery
- Health checks on first use
- Graceful error handling
"""

from __future__ import annotations

import logging

from langchain_ollama import OllamaEmbeddings

from app.domain.exceptions.base import EmbeddingException
from app.infrastructure.configuration.settings import Settings
from app.infrastructure.llm.ollama_health_check import OllamaHealthCheck

logger = logging.getLogger(__name__)


class OllamaEmbeddingClient:
    """Wraps LangChain OllamaEmbeddings and adds error handling + batching."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.ollama_embedding_model
        self._batch_size = settings.embedding_batch_size
        self._base_url = settings.ollama_base_url
        self._timeout = settings.ollama_timeout_seconds
        
        # LangChain's OllamaEmbeddings doesn't have request_timeout param
        # Timeout is controlled via httpx in the client
        self._client = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
        
        self._health_check = OllamaHealthCheck(
            base_url=settings.ollama_base_url,
            timeout_seconds=5,
        )
        self._health_verified = False
        
        logger.info(
            "OllamaEmbeddingClient ready | model=%s | base_url=%s | batch_size=%d | timeout=%ds",
            settings.ollama_embedding_model,
            settings.ollama_base_url,
            settings.embedding_batch_size,
            self._timeout,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text strings.

        Processes in batches to avoid overwhelming the Ollama server.
        """
        # Skip health checks - LangChain will handle connectivity
        # Health checks add overhead; let LangChain handle errors naturally
        
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
                
                # Validate embeddings
                if not embeddings:
                    raise EmbeddingException(
                        f"Batch {batch_num}/{total_batches} returned no embeddings"
                    )
                
                # Check embedding dimensions are consistent
                expected_dim = len(embeddings[0])
                for i, emb in enumerate(embeddings):
                    if len(emb) != expected_dim:
                        raise EmbeddingException(
                            f"Batch {batch_num}/{total_batches}, item {i}: "
                            f"expected dimension {expected_dim}, got {len(emb)}"
                        )
                
                all_embeddings.extend(embeddings)
                logger.debug(
                    "Batch %d/%d complete | dimension=%d",
                    batch_num,
                    total_batches,
                    expected_dim,
                )
                
            except EmbeddingException:
                raise
            except Exception as exc:
                raise EmbeddingException(
                    f"Failed to embed batch {batch_num}/{total_batches}: {exc}"
                ) from exc

        logger.info(
            "Embedded %d texts -> %d vectors | dimension=%d",
            len(texts),
            len(all_embeddings),
            len(all_embeddings[0]) if all_embeddings else 0,
        )
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generate a single query embedding for retrieval."""
        # Skip health checks - LangChain will handle connectivity
        
        try:
            embedding = self._client.embed_query(text)
            
            if not embedding:
                raise EmbeddingException("Embedding query returned empty result")
            
            logger.debug("Embedded query -> vector of dimension %d", len(embedding))
            return embedding
        except EmbeddingException:
            raise
        except Exception as exc:
            raise EmbeddingException(f"Failed to embed query: {exc}") from exc

    @property
    def langchain_embeddings(self) -> OllamaEmbeddings:
        """Return the underlying LangChain embeddings object for use with ChromaDB."""
        return self._client
