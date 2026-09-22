"""
Semantic Search Service — Milestone 3
End-to-end natural-language query → ranked SearchResult list.

Flow: query text → Gemini embedding → ChromaDB cosine similarity → SearchResult list
Measures and records search_latency_ms against the < 3000 ms requirement.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

try:
    from milestone3.embedding_service import EmbeddingService
    from milestone3.schemas import SearchRequest, SearchResult
    from milestone3.vector_store_service import VectorStoreService
except ImportError:
    from embedding_service import EmbeddingService  # type: ignore
    from schemas import SearchRequest, SearchResult  # type: ignore
    from vector_store_service import VectorStoreService  # type: ignore

logger = logging.getLogger(__name__)

# Performance target from spec
LATENCY_REQUIREMENT_MS = 3000.0


class SemanticSearchService:
    """
    Executes semantic search over indexed meeting content.

    Args:
        embedding_svc:  EmbeddingService instance (or auto-initialized).
        vector_store:   VectorStoreService instance (or auto-initialized).
    """

    def __init__(
        self,
        embedding_svc: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreService] = None,
    ) -> None:
        self._embedder = embedding_svc or EmbeddingService()
        self._vstore = vector_store or VectorStoreService()

    def search(self, request: SearchRequest) -> List[SearchResult]:
        """
        Performs a semantic search for the given query.

        Args:
            request: SearchRequest with query text, top_k, and optional source_type filter.

        Returns:
            List of SearchResult objects, ranked by descending relevance score.
            Each result includes meeting_id, source_type, content, relevance_score,
            and measured search_latency_ms.

        Raises:
            EmbeddingError: If query embedding fails.
            VectorStoreError: If vector similarity search fails.
        """
        t_start = time.perf_counter()

        # Check if there's anything indexed
        total_indexed = self._vstore.count()
        if total_indexed == 0:
            logger.info("[SemanticSearchService] Vector store is empty — no results.")
            return []

        # 1. Embed the query
        query_embedding = self._embedder.embed_text(request.query)

        # 2. Similarity search
        meeting_id_filter = getattr(request, "meeting_id", None)
        raw_results = self._vstore.similarity_search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            source_type_filter=request.source_type,
            meeting_id_filter=meeting_id_filter,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        if elapsed_ms > LATENCY_REQUIREMENT_MS:
            logger.warning(
                "[SemanticSearchService] Search latency %.1f ms exceeded requirement of %.1f ms.",
                elapsed_ms, LATENCY_REQUIREMENT_MS,
            )
        else:
            logger.info("[SemanticSearchService] Search completed in %.1f ms.", elapsed_ms)

        # 3. Map to SearchResult schema
        search_results: List[SearchResult] = []
        for raw in raw_results:
            meta = raw.get("metadata", {})
            search_results.append(
                SearchResult(
                    meeting_id=meta.get("meeting_id", "unknown"),
                    relevance_score=raw.get("relevance_score", 0.0),
                    source_type=meta.get("source_type", "unknown"),
                    content=raw.get("document", ""),
                    search_latency_ms=elapsed_ms,
                )
            )

        # 4. Apply date filtering if requested
        date_from = getattr(request, "date_from", None)
        date_to = getattr(request, "date_to", None)
        if date_from or date_to:
            from datetime import datetime, timezone
            try:
                from milestone3.meeting_repository import MeetingRepository
            except ImportError:
                from meeting_repository import MeetingRepository  # type: ignore

            repo = MeetingRepository()
            filtered: List[SearchResult] = []
            for res in search_results:
                m = repo.get_meeting_by_id(res.meeting_id)
                if m and m.created_at:
                    m_date = m.created_at
                    # Normalize naive / aware
                    if date_from:
                        df = datetime.fromisoformat(date_from)
                        if df.tzinfo and not m_date.tzinfo:
                            m_date = m_date.replace(tzinfo=timezone.utc)
                        elif not df.tzinfo and m_date.tzinfo:
                            df = df.replace(tzinfo=timezone.utc)
                        if m_date < df:
                            continue
                    if date_to:
                        dt = datetime.fromisoformat(date_to)
                        if dt.tzinfo and not m_date.tzinfo:
                            m_date = m_date.replace(tzinfo=timezone.utc)
                        elif not dt.tzinfo and m_date.tzinfo:
                            dt = dt.replace(tzinfo=timezone.utc)
                        if m_date > dt:
                            continue
                filtered.append(res)
            search_results = filtered

        # Sort by relevance descending (ChromaDB should already return sorted, but enforce)
        search_results.sort(key=lambda r: r.relevance_score, reverse=True)

        return search_results
