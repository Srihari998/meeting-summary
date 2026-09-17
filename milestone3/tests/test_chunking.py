"""
Tests for milestone3.chunking_service — deterministic overlap-aware chunking.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from milestone3.chunking_service import (
    EmbeddingChunk,
    chunk_for_embedding,
    chunk_meeting_content,
    _sha256,
    _token_estimate,
)


class TestSha256:
    def test_deterministic(self):
        assert _sha256("hello world") == _sha256("hello world")

    def test_different_texts_different_hashes(self):
        assert _sha256("foo") != _sha256("bar")

    def test_returns_hex_string(self):
        h = _sha256("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestTokenEstimate:
    def test_empty(self):
        assert _token_estimate("") == 0

    def test_short_text(self):
        assert _token_estimate("hello") >= 1

    def test_longer_text_more_tokens(self):
        assert _token_estimate("word " * 100) > _token_estimate("word " * 10)


class TestChunkForEmbedding:
    def test_short_text_single_chunk(self):
        text = "This is a short meeting transcript."
        chunks = chunk_for_embedding(text, "mtg-1", "transcript")
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].meeting_id == "mtg-1"
        assert chunks[0].source_type == "transcript"
        assert chunks[0].chunk_index == 0

    def test_empty_text_returns_empty(self):
        chunks = chunk_for_embedding("", "mtg-1", "transcript")
        assert chunks == []

    def test_whitespace_only_returns_empty(self):
        chunks = chunk_for_embedding("   \n\t  ", "mtg-1", "summary")
        assert chunks == []

    def test_content_hash_is_sha256(self):
        text = "A well-formed sentence for testing."
        chunks = chunk_for_embedding(text, "mtg-1", "summary")
        assert len(chunks) == 1
        expected_hash = _sha256(text)
        assert chunks[0].content_hash == expected_hash

    def test_long_text_produces_multiple_chunks(self):
        # Generate text that exceeds chunk_tokens=300
        long_text = ("Speaker 1: " + "This is a long sentence with many words. " * 20 + "\n\n") * 10
        chunks = chunk_for_embedding(long_text, "mtg-2", "transcript", chunk_tokens=100, overlap_tokens=10)
        assert len(chunks) > 1

    def test_chunk_indices_sequential(self):
        long_text = ("Word " * 50 + "\n\n") * 20
        chunks = chunk_for_embedding(long_text, "mtg-3", "transcript", chunk_tokens=100, overlap_tokens=10)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_all_hashes_unique(self):
        long_text = ("Speaker A: " + "Unique sentence number %d. " * 5 + "\n\n") * 20
        # Use format to make unique sentences
        long_text = "\n\n".join(
            f"Speaker A: Unique topic {i}: " + "discussion content here. " * 10
            for i in range(30)
        )
        chunks = chunk_for_embedding(long_text, "mtg-4", "transcript", chunk_tokens=80, overlap_tokens=10)
        hashes = [c.content_hash for c in chunks]
        assert len(hashes) == len(set(hashes)), "All content hashes must be unique"

    def test_chunk_meeting_id_preserved(self):
        chunks = chunk_for_embedding("Some text here.", "meeting-xyz-999", "summary")
        for chunk in chunks:
            assert chunk.meeting_id == "meeting-xyz-999"

    def test_source_type_preserved(self):
        for source_type in ["transcript", "summary", "decision", "action_item"]:
            chunks = chunk_for_embedding("Short content.", "m1", source_type)
            for chunk in chunks:
                assert chunk.source_type == source_type


class TestChunkMeetingContent:
    def test_returns_list_of_embedding_chunks(self):
        chunks = chunk_meeting_content(
            meeting_id="mtg-001",
            transcript="Speaker 1: Hello. Speaker 2: World.",
            summary="A brief hello.",
            decisions=["Agreed to say hello"],
            action_items=["Say hello to everyone"],
        )
        assert all(isinstance(c, EmbeddingChunk) for c in chunks)

    def test_source_types_present(self):
        chunks = chunk_meeting_content(
            meeting_id="mtg-001",
            transcript="Transcript content.",
            summary="Summary content.",
            decisions=["Decision one", "Decision two"],
            action_items=["Task one"],
        )
        source_types = {c.source_type for c in chunks}
        assert "transcript" in source_types
        assert "summary" in source_types
        assert "decision" in source_types
        assert "action_item" in source_types

    def test_empty_decisions_and_action_items(self):
        chunks = chunk_meeting_content(
            meeting_id="mtg-002",
            transcript="Some transcript.",
            summary="Some summary.",
            decisions=[],
            action_items=[],
        )
        source_types = {c.source_type for c in chunks}
        assert "decision" not in source_types
        assert "action_item" not in source_types

    def test_all_chunks_have_meeting_id(self):
        chunks = chunk_meeting_content(
            meeting_id="mtg-003",
            transcript="Talk.",
            summary="Sum.",
            decisions=["D1"],
            action_items=["A1"],
        )
        for chunk in chunks:
            assert chunk.meeting_id == "mtg-003"

    def test_empty_transcript_skipped(self):
        chunks = chunk_meeting_content(
            meeting_id="mtg-004",
            transcript="",
            summary="Summary only.",
            decisions=[],
            action_items=[],
        )
        source_types = {c.source_type for c in chunks}
        assert "transcript" not in source_types
        assert "summary" in source_types
