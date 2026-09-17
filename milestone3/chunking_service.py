"""
Chunking Service — Milestone 3
Deterministic, overlap-aware text chunking for embedding generation.

This is intentionally separate from milestone2.llm_service.chunk_transcript(),
which is tuned for LLM context-window limits. This chunker is tuned for
embedding granularity (~300 tokens, ~50-token overlap).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Literal

# Approximate characters per token (heuristic)
_CHARS_PER_TOKEN = 4

SourceType = Literal["transcript", "summary", "decision", "action_item"]


@dataclass
class EmbeddingChunk:
    """A text chunk ready for embedding, with full traceability metadata."""

    meeting_id: str
    source_type: str          # transcript | summary | decision | action_item
    chunk_index: int          # 0-based position within its source
    text: str                 # the chunk text to embed
    content_hash: str         # SHA-256 of text — used for idempotency


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_estimate(text: str) -> int:
    """Rough token count estimate: max of char-based and word-based heuristics."""
    if not text:
        return 0
    return max(len(text) // _CHARS_PER_TOKEN, len(text.split()))


def _split_into_units(text: str) -> List[str]:
    """
    Splits text into atomic units (speaker turns → paragraphs → sentences).
    Units are the building blocks assembled into chunks.
    """
    # Try speaker turns first (e.g. "Alice:", "Speaker 1:", "[00:12]")
    speaker_splits = re.split(r"(?=\n[A-Za-z0-9 _\-]+:|\n\[[^\]]+\])", text)
    units: List[str] = []

    for segment in speaker_splits:
        seg = segment.strip()
        if not seg:
            continue
        # If segment is still large, split on double newline (paragraphs)
        if _token_estimate(seg) > 150:
            paras = re.split(r"\n{2,}", seg)
            for para in paras:
                p = para.strip()
                if not p:
                    continue
                # If paragraph is still large, split on sentence boundaries
                if _token_estimate(p) > 100:
                    sentences = re.split(r"(?<=[.!?])\s+", p)
                    units.extend(s.strip() for s in sentences if s.strip())
                else:
                    units.append(p)
        else:
            units.append(seg)

    return [u for u in units if u]


def chunk_for_embedding(
    text: str,
    meeting_id: str,
    source_type: str,
    chunk_tokens: int = 300,
    overlap_tokens: int = 50,
) -> List[EmbeddingChunk]:
    """
    Splits text into overlapping chunks suitable for embedding.

    Args:
        text:          The full text to chunk (transcript, summary, decision, etc.).
        meeting_id:    Meeting identifier for traceability.
        source_type:   Content category (transcript | summary | decision | action_item).
        chunk_tokens:  Target maximum tokens per chunk (~300 default).
        overlap_tokens: Overlap between consecutive chunks (~50 default).

    Returns:
        List of EmbeddingChunk, each with text, metadata, and SHA-256 hash.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Short texts that fit in a single chunk — fast path
    if _token_estimate(text) <= chunk_tokens:
        return [
            EmbeddingChunk(
                meeting_id=meeting_id,
                source_type=source_type,
                chunk_index=0,
                text=text,
                content_hash=_sha256(text),
            )
        ]

    units = _split_into_units(text)

    chunks: List[EmbeddingChunk] = []
    current_parts: List[str] = []
    overlap_buffer: List[str] = []  # trailing units to prepend as overlap

    for unit in units:
        candidate = "\n\n".join(current_parts + [unit]) if current_parts else unit
        if _token_estimate(candidate) <= chunk_tokens:
            current_parts.append(unit)
        else:
            if current_parts:
                chunk_text = "\n\n".join(current_parts)
                chunks.append(
                    EmbeddingChunk(
                        meeting_id=meeting_id,
                        source_type=source_type,
                        chunk_index=len(chunks),
                        text=chunk_text,
                        content_hash=_sha256(chunk_text),
                    )
                )
                # Build overlap buffer: keep trailing units that fit within overlap_tokens
                overlap_buffer = []
                overlap_size = 0
                for part in reversed(current_parts):
                    part_tokens = _token_estimate(part)
                    if overlap_size + part_tokens <= overlap_tokens:
                        overlap_buffer.insert(0, part)
                        overlap_size += part_tokens
                    else:
                        break

            current_parts = overlap_buffer + [unit]
            overlap_buffer = []

    # Flush remaining parts
    if current_parts:
        chunk_text = "\n\n".join(current_parts)
        chunks.append(
            EmbeddingChunk(
                meeting_id=meeting_id,
                source_type=source_type,
                chunk_index=len(chunks),
                text=chunk_text,
                content_hash=_sha256(chunk_text),
            )
        )

    return chunks


def chunk_meeting_content(
    meeting_id: str,
    transcript: str,
    summary: str,
    decisions: List[str],
    action_items: List[str],
) -> List[EmbeddingChunk]:
    """
    Produces all EmbeddingChunks for a complete meeting record.

    Indexes four content types:
        - transcript  → chunked at 300-token granularity with overlap
        - summary     → typically 1–2 chunks
        - decision    → one chunk per decision string
        - action_item → one chunk per action item task string

    Args:
        meeting_id:   Meeting identifier.
        transcript:   Full raw transcript text.
        summary:      Meeting summary text.
        decisions:    List of decision strings.
        action_items: List of action item task strings.

    Returns:
        Flat list of all EmbeddingChunk objects for this meeting.
    """
    all_chunks: List[EmbeddingChunk] = []

    # Transcript
    if transcript:
        all_chunks.extend(chunk_for_embedding(transcript, meeting_id, "transcript"))

    # Summary
    if summary:
        all_chunks.extend(chunk_for_embedding(summary, meeting_id, "summary"))

    # Decisions — index each separately for fine-grained retrieval
    for idx, decision in enumerate(decisions):
        if decision and decision.strip():
            chunk_text = decision.strip()
            all_chunks.append(
                EmbeddingChunk(
                    meeting_id=meeting_id,
                    source_type="decision",
                    chunk_index=idx,
                    text=chunk_text,
                    content_hash=_sha256(chunk_text),
                )
            )

    # Action items — index each separately
    for idx, task in enumerate(action_items):
        if task and task.strip():
            chunk_text = task.strip()
            all_chunks.append(
                EmbeddingChunk(
                    meeting_id=meeting_id,
                    source_type="action_item",
                    chunk_index=idx,
                    text=chunk_text,
                    content_hash=_sha256(chunk_text),
                )
            )

    return all_chunks
