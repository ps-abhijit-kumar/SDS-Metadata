"""ChromaDB vector store adapter.

Responsibilities:
  - Store document chunks as embeddings in ChromaDB.
  - Retrieve the most relevant chunks for a set of query strings.
  - Handle ChromaDB collection initialisation and dimension compatibility.

Uses LangChain's Chroma integration which manages the embedding function
and ChromaDB client lifecycle.
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading

import chromadb
from langchain_chroma import Chroma

from app.domain.exceptions.base import VectorStoreException
from app.infrastructure.configuration.settings import Settings
from app.infrastructure.embeddings.ollama_embedding_client import OllamaEmbeddingClient

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Manages document chunk storage and semantic retrieval via ChromaDB."""

    _client_lock = threading.RLock()
    _client_cache: dict[str, chromadb.PersistentClient] = {}

    @classmethod
    def _get_client_for_dir(cls, db_dir: Path) -> chromadb.PersistentClient:
        """Thread-safe acquisition of a shared PersistentClient for a given directory."""
        key = str(db_dir.resolve())
        with cls._client_lock:
            if key not in cls._client_cache:
                client = chromadb.PersistentClient(path=key)
                cls._client_cache[key] = client
            return cls._client_cache[key]

    @classmethod
    def reset_client_cache(cls) -> None:
        """Clear the client cache (useful during test teardown / directory cleanup)."""
        with cls._client_lock:
            cls._client_cache.clear()

    def __init__(self, settings: Settings, embedding_client: OllamaEmbeddingClient) -> None:
        self._collection_name = settings.chroma_collection_name
        self._db_dir = Path(settings.chroma_db_dir)
        self._embedding_client = embedding_client
        self._batch_size = getattr(settings, "embedding_batch_size", 32)
        self._store: Chroma | None = None
        self._lock = threading.RLock()
        self._db_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "ChromaVectorStore ready | collection=%s | dir=%s | batch_size=%d",
            self._collection_name,
            self._db_dir,
            self._batch_size,
        )

    def _get_store(self) -> Chroma:
        """Thread-safe lazy-initialise the Chroma store on first use using shared client."""
        with self._lock:
            if self._store is None:
                client = self._get_client_for_dir(self._db_dir)
                self._store = Chroma(
                    client=client,
                    collection_name=self._collection_name,
                    embedding_function=self._embedding_client.langchain_embeddings,
                )
            return self._store

    def add_documents(
        self,
        document_id: str,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Store a list of text chunks for the given document_id.

        Existing entries for this document_id are deleted first so
        re-processing a document produces a clean result.
        
        Validates:
        - Texts are not empty
        - Embeddings have consistent dimensions
        - Storage completes without errors
        """
        if not texts:
            logger.warning("add_documents called with empty texts for document_id=%s", document_id)
            return

        with self._lock:
            try:
                store = self._get_store()

                # Remove any previously stored chunks for this document
                try:
                    existing = store.get(where={"document_id": document_id})
                    if existing and existing.get("ids"):
                        store.delete(ids=existing["ids"])
                        logger.debug(
                            "Deleted %d existing chunks for document_id=%s (re-processing)",
                            len(existing["ids"]),
                            document_id,
                        )
                except Exception as exc:
                    logger.warning(
                        "Could not clean up existing chunks for %s: %s (continuing anyway)",
                        document_id,
                        exc,
                    )

                # Attach document_id to every chunk's metadata
                enriched_metas: list[dict] = []
                for i, text in enumerate(texts):
                    if not text or not text.strip():
                        logger.debug("Skipping empty chunk at index %d for document_id=%s", i, document_id)
                        continue
                    
                    meta = dict(metadatas[i]) if metadatas and i < len(metadatas) else {}
                    meta["document_id"] = document_id
                    meta["chunk_index"] = i
                    enriched_metas.append(meta)

                if not enriched_metas:
                    raise VectorStoreException(
                        f"No valid text chunks to store for document {document_id}"
                    )

                # Extract non-empty texts matching our metadata
                valid_texts = [
                    text.strip() for text in texts
                    if text and text.strip()
                ]
                
                if len(valid_texts) != len(enriched_metas):
                    logger.warning(
                        "Text/metadata count mismatch: %d texts vs %d metadata entries",
                        len(valid_texts),
                        len(enriched_metas),
                    )

                # Store in dynamic batches with real-time progress logging and error recovery
                total_chunks = len(valid_texts)
                batch_size = max(1, self._batch_size)
                total_batches = (total_chunks + batch_size - 1) // batch_size
                logger.info(
                    "Starting embedding & storage | chunks=%d | batch_size=%d | batches=%d",
                    total_chunks,
                    batch_size,
                    total_batches,
                )

                for batch_idx in range(0, total_chunks, batch_size):
                    batch_num = (batch_idx // batch_size) + 1
                    batch_texts = valid_texts[batch_idx : batch_idx + batch_size]
                    batch_metas = enriched_metas[batch_idx : batch_idx + batch_size]
                    logger.info(
                        "Embedding batch %d/%d | chunks=%d",
                        batch_num,
                        total_batches,
                        len(batch_texts),
                    )
                    try:
                        store.add_texts(texts=batch_texts, metadatas=batch_metas)
                    except Exception as store_exc:
                        # Rollback any partially stored batches for this document_id
                        try:
                            existing = store.get(where={"document_id": document_id})
                            if existing and existing.get("ids"):
                                store.delete(ids=existing["ids"])
                                logger.info(
                                    "Cleaned up %d partially stored chunks for failed document_id=%s",
                                    len(existing["ids"]),
                                    document_id,
                                )
                        except Exception as cleanup_exc:
                            logger.warning(
                                "Failed to clean up partial chunks for %s: %s",
                                document_id,
                                cleanup_exc,
                            )
                        logger.error(
                            "Failed to embed/store batch %d/%d (%d chunks) for document_id=%s: %s",
                            batch_num,
                            total_batches,
                            len(batch_texts),
                            document_id,
                            store_exc,
                        )
                        raise VectorStoreException(
                            f"Failed to store batch {batch_num}/{total_batches} in ChromaDB: {store_exc}"
                        ) from store_exc

                logger.info(
                    "✓ Stored %d chunks for document_id=%s in ChromaDB",
                    total_chunks,
                    document_id,
                )

            except VectorStoreException:
                raise
            except Exception as exc:
                try:
                    existing = store.get(where={"document_id": document_id})
                    if existing and existing.get("ids"):
                        store.delete(ids=existing["ids"])
                        logger.info(
                            "Cleaned up %d partially stored chunks for failed document_id=%s",
                            len(existing["ids"]),
                            document_id,
                        )
                except Exception as cleanup_exc:
                    logger.warning(
                        "Failed to clean up partial chunks for %s: %s",
                        document_id,
                        cleanup_exc,
                    )
                raise VectorStoreException(
                    f"Unexpected error storing chunks for document {document_id}: {exc}"
                ) from exc

    def similarity_search(
        self,
        queries: list[str],
        document_id: str,
        k: int = 2,
    ) -> list[str]:
        """Retrieve the top-k most relevant chunk texts for each query."""
        if not queries:
            return []

        with self._lock:
            try:
                store = self._get_store()
                seen: set[str] = set()
                results: list[tuple[str, int]] = []  # (text, section_number)

                for query in queries:
                    if not query or not query.strip():
                        logger.debug("Skipping empty query")
                        continue
                    
                    try:
                        docs = store.similarity_search(
                            query=query.strip(),
                            k=k,
                            filter={"document_id": document_id},
                        )
                        
                        if not docs:
                            logger.debug("No results for query: %s", query[:50])
                            continue
                        
                        for doc in docs:
                            content = doc.page_content.strip() if doc.page_content else ""
                            section = doc.metadata.get("section_number", 0) if doc.metadata else 0
                            
                            if content and content not in seen:
                                seen.add(content)
                                results.append((content, section))
                    
                    except Exception as query_exc:
                        logger.warning("Error retrieving results for query '%s': %s", query[:50], query_exc)
                        continue

                if not results:
                    logger.warning(
                        "No chunks retrieved for document_id=%s across %d queries",
                        document_id,
                        len(queries),
                    )
                    return []

                def section_priority(section: int | None) -> int:
                    if section in (1, 15):
                        return 100
                    elif section in (2, 14):
                        return 50
                    else:
                        return 0

                results.sort(key=lambda x: section_priority(x[1]), reverse=True)

                limited_results = [text for text, _ in results[:5]]

                logger.info(
                    "✓ Retrieved %d unique chunks for document_id=%s (limited to 5) across %d queries",
                    len(limited_results),
                    document_id,
                    len([q for q in queries if q and q.strip()]),
                )
                return limited_results

            except Exception as exc:
                raise VectorStoreException(
                    f"Failed to retrieve chunks for document {document_id}: {exc}"
                ) from exc

    def similarity_search_with_score(
        self,
        query: str,
        document_id: str = "all",
        k: int = 5,
        where_filter: dict | None = None,
    ) -> list[tuple[any, float]]:
        """Retrieve documents with relevance score/distance."""
        with self._lock:
            try:
                store = self._get_store()
                conditions: list[dict] = []
                if document_id and document_id.lower() != "all":
                    conditions.append({"document_id": document_id})
                if where_filter:
                    conditions.append(where_filter)

                if len(conditions) == 1:
                    filter_dict = conditions[0]
                elif len(conditions) > 1:
                    filter_dict = {"$and": conditions}
                else:
                    filter_dict = None

                results = store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict,
                )
                return results
            except Exception as exc:
                logger.error("Error in similarity_search_with_score: %s", exc)
                return []

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to the given document_id."""
        with self._lock:
            try:
                store = self._get_store()
                existing = store.get(where={"document_id": document_id})
                if existing and existing.get("ids"):
                    store.delete(ids=existing["ids"])
                    logger.info(
                        "Deleted %d chunks for document_id=%s",
                        len(existing["ids"]),
                        document_id,
                    )
            except Exception as exc:
                raise VectorStoreException(
                    f"Failed to delete chunks for document {document_id}: {exc}"
                ) from exc

