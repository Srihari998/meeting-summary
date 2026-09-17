"""
Indexing Pipeline — Milestone 3
Orchestrates the full flow: fetch meeting from DB → chunk → embed → upsert into vector store.

Idempotent: already-indexed chunks (matched by SHA-256 content hash) are skipped.
The relational DB is the source of truth; ChromaDB is rebuilt from it at any time.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from milestone3.chunking_service import chunk_meeting_content, EmbeddingChunk
    from milestone3.embedding_service import EmbeddingService
    from milestone3.meeting_repository import MeetingRepository
    from milestone3.vector_store_service import VectorStoreService
except ImportError:
    from chunking_service import chunk_meeting_content, EmbeddingChunk  # type: ignore
    from embedding_service import EmbeddingService  # type: ignore
    from meeting_repository import MeetingRepository  # type: ignore
    from vector_store_service import VectorStoreService  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Summary of an indexing operation."""

    meeting_ids_processed: List[str] = field(default_factory=list)
    chunks_indexed: int = 0
    chunks_skipped: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


def _make_chunk_id(chunk: EmbeddingChunk) -> str:
    """Generates a deterministic, unique ChromaDB document ID for a chunk."""
    return f"{chunk.meeting_id}__{chunk.source_type}__{chunk.chunk_index}"


def _index_single_meeting(
    meeting,
    vector_store: VectorStoreService,
    embedding_svc: EmbeddingService,
    result: IndexingResult,
) -> None:
    """
    Internal helper: indexes all chunks for one Meeting ORM row.
    Modifies `result` in-place.
    """
    meeting_id = meeting.id

    # Collect content to chunk
    transcript = meeting.transcript or ""
    summary = meeting.summary or ""
    decisions = meeting.decisions if hasattr(meeting, "decisions") and meeting.decisions else []
    action_items = [ai.task for ai in meeting.action_items] if meeting.action_items else []

    # Generate all chunks for this meeting
    chunks: List[EmbeddingChunk] = chunk_meeting_content(
        meeting_id=meeting_id,
        transcript=transcript,
        summary=summary,
        decisions=decisions,
        action_items=action_items,
    )

    if not chunks:
        logger.info("[IndexingPipeline] No chunks generated for meeting %s", meeting_id)
        return

    # Idempotency: fetch already-indexed content hashes for this meeting
    already_indexed_hashes = vector_store.get_indexed_hashes(meeting_id=meeting_id)

    # Filter to only new chunks
    new_chunks = [c for c in chunks if c.content_hash not in already_indexed_hashes]
    skipped = len(chunks) - len(new_chunks)
    result.chunks_skipped += skipped

    if not new_chunks:
        logger.info(
            "[IndexingPipeline] Meeting %s: all %d chunks already indexed, skipping.",
            meeting_id, len(chunks),
        )
        return

    logger.info(
        "[IndexingPipeline] Meeting %s: indexing %d new chunks (%d skipped).",
        meeting_id, len(new_chunks), skipped,
    )

    # Embed new chunks
    texts = [c.text for c in new_chunks]
    embeddings = embedding_svc.embed_batch(texts)

    # Upsert to vector store
    valid_ids: List[str] = []
    valid_texts: List[str] = []
    valid_embeddings: List[List[float]] = []
    valid_metadatas: List[dict] = []

    for chunk, emb in zip(new_chunks, embeddings):
        if emb is None:
            logger.warning(
                "[IndexingPipeline] Skipping chunk %s (embedding returned None).",
                _make_chunk_id(chunk),
            )
            continue
        valid_ids.append(_make_chunk_id(chunk))
        valid_texts.append(chunk.text)
        valid_embeddings.append(emb)
        valid_metadatas.append(
            {
                "meeting_id": chunk.meeting_id,
                "source_type": chunk.source_type,
                "content_hash": chunk.content_hash,
                "chunk_index": chunk.chunk_index,
            }
        )

    if valid_ids:
        vector_store.upsert_batch(
            chunk_ids=valid_ids,
            texts=valid_texts,
            embeddings=valid_embeddings,
            metadatas=valid_metadatas,
        )
        result.chunks_indexed += len(valid_ids)
        result.meeting_ids_processed.append(meeting_id)


def index_meeting(
    meeting_id: str,
    db_path: Optional[str] = None,
    vector_store: Optional[VectorStoreService] = None,
    embedding_svc: Optional[EmbeddingService] = None,
) -> IndexingResult:
    """
    Indexes a single meeting by ID into the vector store.

    Idempotent: already-indexed chunks are skipped based on content hash.

    Args:
        meeting_id:    The meeting identifier to index.
        db_path:       Optional path to the SQLite database file.
        vector_store:  Optional pre-initialized VectorStoreService (for testing/DI).
        embedding_svc: Optional pre-initialized EmbeddingService (for testing/DI).

    Returns:
        IndexingResult with counts of indexed, skipped, and any errors.
    """
    result = IndexingResult()
    vstore = vector_store or VectorStoreService()
    embedder = embedding_svc or EmbeddingService()
    repo = MeetingRepository(db_path=db_path)

    meeting = repo.get_meeting_by_id(meeting_id)
    if meeting is None:
        result.errors.append(f"Meeting '{meeting_id}' not found in database.")
        return result

    try:
        _index_single_meeting(meeting, vstore, embedder, result)
    except Exception as exc:
        logger.error("[IndexingPipeline] Error indexing meeting %s: %s", meeting_id, exc)
        result.errors.append(f"Meeting '{meeting_id}': {exc}")

    return result


def index_all_meetings(
    db_path: Optional[str] = None,
    vector_store: Optional[VectorStoreService] = None,
    embedding_svc: Optional[EmbeddingService] = None,
) -> IndexingResult:
    """
    Indexes all meetings stored in the relational database.

    Idempotent: already-indexed chunks are skipped based on content hash.
    Safe to re-run at any time without creating duplicates.

    Args:
        db_path:       Optional path to the SQLite database file.
        vector_store:  Optional pre-initialized VectorStoreService (for testing/DI).
        embedding_svc: Optional pre-initialized EmbeddingService (for testing/DI).

    Returns:
        IndexingResult summarizing the complete indexing operation.
    """
    result = IndexingResult()
    vstore = vector_store or VectorStoreService()
    embedder = embedding_svc or EmbeddingService()
    repo = MeetingRepository(db_path=db_path)

    meetings = repo.get_all_meetings()
    if not meetings:
        logger.info("[IndexingPipeline] No meetings found in database.")
        return result

    logger.info("[IndexingPipeline] Starting indexing of %d meetings.", len(meetings))

    for meeting in meetings:
        try:
            _index_single_meeting(meeting, vstore, embedder, result)
        except Exception as exc:
            logger.error("[IndexingPipeline] Error indexing meeting %s: %s", meeting.id, exc)
            result.errors.append(f"Meeting '{meeting.id}': {exc}")

    logger.info(
        "[IndexingPipeline] Done. %d chunks indexed, %d skipped, %d errors.",
        result.chunks_indexed, result.chunks_skipped, len(result.errors),
    )
    return result


def reindex_meeting(
    meeting_id: str,
    db_path: Optional[str] = None,
    vector_store: Optional[VectorStoreService] = None,
    embedding_svc: Optional[EmbeddingService] = None,
) -> IndexingResult:
    """
    Force re-indexes a meeting by first deleting all its existing vectors,
    then re-embedding and upserting from the source relational DB.

    Use this when a meeting record has been updated in the DB.

    Args:
        meeting_id:    The meeting identifier to re-index.
        db_path:       Optional path to the SQLite database file.
        vector_store:  Optional pre-initialized VectorStoreService.
        embedding_svc: Optional pre-initialized EmbeddingService.

    Returns:
        IndexingResult.
    """
    vstore = vector_store or VectorStoreService()
    deleted = vstore.delete_meeting(meeting_id)
    logger.info("[IndexingPipeline] Deleted %d existing vectors for meeting %s.", deleted, meeting_id)
    return index_meeting(
        meeting_id=meeting_id,
        db_path=db_path,
        vector_store=vstore,
        embedding_svc=embedding_svc,
    )
