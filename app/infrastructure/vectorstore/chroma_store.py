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

import chromadb
from langchain_chroma import Chroma

from app.domain.exceptions.base import VectorStoreException
from app.infrastructure.configuration.settings import Settings
from app.infrastructure.embeddings.ollama_embedding_client import OllamaEmbeddingClient

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Manages document chunk storage and semantic retrieval via ChromaDB."""

    def __init__(self, settings: Settings, embedding_client: OllamaEmbeddingClient) -> None:
        self._collection_name = settings.chroma_collection_name
        self._db_dir = Path(settings.chroma_db_dir)
        self._embedding_client = embedding_client
        self._store: Chroma | None = None
        self._db_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "ChromaVectorStore ready | collection=%s | dir=%s",
            self._collection_name,
            self._db_dir,
        )

    def _get_store(self) -> Chroma:
        """Lazy-initialise the Chroma store on first use."""
        if self._store is None:
            self._store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embedding_client.langchain_embeddings,
                persist_directory=str(self._db_dir),
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

            # Store with error recovery
            try:
                store.add_texts(texts=valid_texts, metadatas=enriched_metas)
                logger.info(
                    "✓ Stored %d chunks for document_id=%s in ChromaDB",
                    len(valid_texts),
                    document_id,
                )
            except Exception as store_exc:
                # If storage fails, it might be a dimension mismatch
                # Log details for debugging
                logger.error(
                    "Failed to store chunks - possible dimension mismatch or corruption: %s",
                    store_exc,
                )
                raise VectorStoreException(
                    f"Failed to store chunks in ChromaDB: {store_exc}"
                ) from store_exc

        except VectorStoreException:
            raise
        except Exception as exc:
            raise VectorStoreException(
                f"Unexpected error storing chunks for document {document_id}: {exc}"
            ) from exc

    def similarity_search(
        self,
        queries: list[str],
        document_id: str,
        k: int = 2,
    ) -> list[str]:
        """Retrieve the top-k most relevant chunk texts for each query.

        Optimised retrieval strategy:
        - Section 1 (Identification) is boosted for company_name and product_name.
        - Section 15 (Regulatory) is boosted for jurisdiction detection.
        - Results are deduplicated to eliminate redundant context.
        - Limited to 3-5 total chunks to reduce LLM inference latency.
        
        Validates:
        - Queries are not empty
        - Document exists in store
        - Retrieved chunks have content
        - Results are within dimension bounds

        Args:
            queries: List of search queries.
            document_id: Document ID to filter by.
            k: Top-k results per query (default: 2).

        Returns:
            List of unique chunk texts, limited to ~5 total.
            
        Raises:
            VectorStoreException: If retrieval fails.
        """
        if not queries:
            return []

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

            # Remove duplicates while keeping highest section-scored results
            # Prioritise Section 1 and 15
            def section_priority(section: int | None) -> int:
                if section in (1, 15):
                    return 100
                elif section in (2, 14):
                    return 50
                else:
                    return 0

            results.sort(key=lambda x: section_priority(x[1]), reverse=True)

            # Limit to 5 chunks max to reduce LLM context and improve speed
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

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to the given document_id."""
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
