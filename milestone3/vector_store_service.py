"""
Vector Store Service — Milestone 3
ChromaDB-based vector store abstraction for semantic similarity search.

ChromaDB runs locally with file persistence — no external service needed.
Each stored document is fully traceable to its source meeting via metadata.

The relational DB (Milestone 2 SQLite) remains the source of truth.
ChromaDB is a search index only and can be rebuilt at any time.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default ChromaDB persistence directory (relative to project root)
DEFAULT_PERSIST_DIR = os.environ.get(
    "CHROMA_PERSIST_DIR",
    str(Path(__file__).resolve().parent.parent / "chroma_db"),
)

# ChromaDB collection name
COLLECTION_NAME = "meeting_intelligence"


class VectorStoreError(Exception):
    """Raised when a vector store operation fails."""
    pass


class VectorStoreService:
    """
    Manages a ChromaDB collection for meeting chunk embeddings.

    Each document in the collection corresponds to one EmbeddingChunk and stores:
        id         = "{meeting_id}_{source_type}_{chunk_index}"
        embedding  = float list (Gemini embedding)
        metadata   = {"meeting_id": ..., "source_type": ..., "content_hash": ..., "chunk_index": ...}
        document   = chunk text

    Args:
        persist_directory: Path to ChromaDB storage directory.
    """

    def __init__(self, persist_directory: str = DEFAULT_PERSIST_DIR) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError(
                "chromadb package is required. Run: pip install chromadb"
            ) from exc

        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "[VectorStoreService] Initialized collection '%s' at '%s'",
            COLLECTION_NAME, persist_directory,
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(
        self,
        chunk_id: str,
        text: str,
        embedding: List[float],
        meeting_id: str,
        source_type: str,
        content_hash: str,
        chunk_index: int,
    ) -> None:
        """
        Inserts or updates a single chunk in the vector store.

        Args:
            chunk_id:     Unique document ID (e.g. "{meeting_id}_{source_type}_{chunk_index}").
            text:         The chunk text (stored as the document).
            embedding:    Dense float vector from EmbeddingService.
            meeting_id:   Source meeting identifier.
            source_type:  Content category (transcript | summary | decision | action_item).
            content_hash: SHA-256 of text for idempotency checking.
            chunk_index:  0-based position within the source.
        """
        try:
            self._collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[
                    {
                        "meeting_id": meeting_id,
                        "source_type": source_type,
                        "content_hash": content_hash,
                        "chunk_index": chunk_index,
                    }
                ],
            )
        except Exception as exc:
            raise VectorStoreError(f"Upsert failed for chunk '{chunk_id}': {exc}") from exc

    def upsert_batch(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Batch upsert for efficiency when indexing many chunks at once.

        Args:
            chunk_ids:  List of unique document IDs.
            texts:      List of chunk texts.
            embeddings: List of float vectors.
            metadatas:  List of metadata dicts (each must have meeting_id, source_type, content_hash, chunk_index).
        """
        if not chunk_ids:
            return
        try:
            self._collection.upsert(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(f"Batch upsert failed: {exc}") from exc

    def delete_meeting(self, meeting_id: str) -> int:
        """
        Deletes all vectors associated with a given meeting_id.

        Returns the number of documents deleted.
        """
        try:
            existing = self._collection.get(
                where={"meeting_id": meeting_id},
                include=[],
            )
            ids_to_delete = existing.get("ids", [])
            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception as exc:
            raise VectorStoreError(f"Delete failed for meeting '{meeting_id}': {exc}") from exc

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        source_type_filter: Optional[str] = None,
        meeting_id_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Finds the top-k most similar chunks to a query embedding.

        Args:
            query_embedding:    Dense float vector of the query.
            top_k:              Maximum number of results.
            source_type_filter: Optional filter by source_type.
            meeting_id_filter:  Optional filter by meeting_id.

        Returns:
            List of dicts, each containing:
                - id, document, metadata (meeting_id, source_type, content_hash, chunk_index)
                - distance (ChromaDB cosine distance; relevance = 1 - distance)
        """
        try:
            where_conditions = []
            if source_type_filter:
                where_conditions.append({"source_type": source_type_filter})
            if meeting_id_filter:
                where_conditions.append({"meeting_id": meeting_id_filter})

            where_filter: Optional[Dict[str, Any]] = None
            if len(where_conditions) == 1:
                where_filter = where_conditions[0]
            elif len(where_conditions) > 1:
                where_filter = {"$and": where_conditions}

            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": min(top_k, max(1, self.count())),
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                query_kwargs["where"] = where_filter

            result = self._collection.query(**query_kwargs)

            if not result or not result.get("ids") or not result["ids"][0]:
                return []

            ids = result["ids"][0]
            docs = result["documents"][0]
            metas = result["metadatas"][0]
            distances = result["distances"][0]

            return [
                {
                    "id": ids[i],
                    "document": docs[i],
                    "metadata": metas[i],
                    "distance": distances[i],
                    # Cosine similarity = 1 - cosine distance (ChromaDB uses cosine distance)
                    "relevance_score": max(0.0, min(1.0, 1.0 - distances[i])),
                }
                for i in range(len(ids))
            ]
        except Exception as exc:
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

    def get_indexed_hashes(self, meeting_id: Optional[str] = None) -> set:
        """
        Returns the set of content_hash values already in the vector store.
        Used for idempotency — skip chunks already indexed.

        Args:
            meeting_id: If provided, restrict to a single meeting's chunks.
        """
        try:
            get_kwargs: Dict[str, Any] = {"include": ["metadatas"]}
            if meeting_id:
                get_kwargs["where"] = {"meeting_id": meeting_id}
            result = self._collection.get(**get_kwargs)
            metas = result.get("metadatas") or []
            return {m["content_hash"] for m in metas if m and "content_hash" in m}
        except Exception as exc:
            logger.warning("[VectorStoreService] Could not fetch indexed hashes: %s", exc)
            return set()

    def count(self) -> int:
        """Returns total number of documents in the collection."""
        try:
            return self._collection.count()
        except Exception:
            return 0

    def count_for_meeting(self, meeting_id: str) -> int:
        """Returns the number of indexed chunks for a specific meeting."""
        try:
            result = self._collection.get(
                where={"meeting_id": meeting_id},
                include=[],
            )
            return len(result.get("ids", []))
        except Exception:
            return 0
