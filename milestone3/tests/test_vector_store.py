"""
Tests for milestone3.vector_store_service — ChromaDB abstraction.
Uses ChromaDB in-memory (ephemeral) client for isolation — no files written to disk.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import uuid

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _make_in_memory_vector_store():
    """Returns a VectorStoreService backed by an in-memory ChromaDB client with unique collection."""
    import chromadb
    from milestone3.vector_store_service import VectorStoreService

    svc = VectorStoreService.__new__(VectorStoreService)
    client = chromadb.EphemeralClient()
    # Use a unique collection name per call to guarantee full test isolation
    unique_name = f"test_collection_{uuid.uuid4().hex}"
    svc._client = client
    svc._collection = client.get_or_create_collection(
        name=unique_name,
        metadata={"hnsw:space": "cosine"},
    )
    return svc


class TestVectorStoreUpsert:
    def test_upsert_single_chunk(self):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        vstore = _make_in_memory_vector_store()
        vstore.upsert(
            chunk_id="mtg1__transcript__0",
            text="We discussed the project timeline.",
            embedding=[0.1] * 768,
            meeting_id="mtg1",
            source_type="transcript",
            content_hash="abc123",
            chunk_index=0,
        )
        assert vstore.count() == 1

    def test_upsert_overwrites_existing(self):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        vstore = _make_in_memory_vector_store()
        chunk_id = "mtg1__transcript__0"
        vstore.upsert(
            chunk_id=chunk_id, text="Original text.", embedding=[0.1] * 768,
            meeting_id="mtg1", source_type="transcript", content_hash="hash1", chunk_index=0,
        )
        vstore.upsert(
            chunk_id=chunk_id, text="Updated text.", embedding=[0.2] * 768,
            meeting_id="mtg1", source_type="transcript", content_hash="hash2", chunk_index=0,
        )
        assert vstore.count() == 1  # still 1, upserted

    def test_upsert_batch(self):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        vstore = _make_in_memory_vector_store()
        vstore.upsert_batch(
            chunk_ids=["mtg1__transcript__0", "mtg1__summary__0"],
            texts=["Transcript chunk.", "Summary chunk."],
            embeddings=[[0.1] * 768, [0.2] * 768],
            metadatas=[
                {"meeting_id": "mtg1", "source_type": "transcript", "content_hash": "h1", "chunk_index": 0},
                {"meeting_id": "mtg1", "source_type": "summary", "content_hash": "h2", "chunk_index": 0},
            ],
        )
        assert vstore.count() == 2

    def test_upsert_batch_empty_is_noop(self):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        vstore = _make_in_memory_vector_store()
        vstore.upsert_batch([], [], [], [])
        assert vstore.count() == 0


class TestVectorStoreDelete:
    def test_delete_meeting_removes_all_chunks(self):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        vstore = _make_in_memory_vector_store()
        # Insert 3 chunks for mtg1, 2 for mtg2
        for i in range(3):
            vstore.upsert(
                chunk_id=f"mtg1__transcript__{i}", text=f"Chunk {i}", embedding=[0.1] * 768,
                meeting_id="mtg1", source_type="transcript", content_hash=f"h{i}", chunk_index=i,
            )
        for i in range(2):
            vstore.upsert(
                chunk_id=f"mtg2__transcript__{i}", text=f"Other {i}", embedding=[0.2] * 768,
                meeting_id="mtg2", source_type="transcript", content_hash=f"h2{i}", chunk_index=i,
            )

        deleted = vstore.delete_meeting("mtg1")
        assert deleted == 3
        assert vstore.count() == 2
        assert vstore.count_for_meeting("mtg1") == 0
        assert vstore.count_for_meeting("mtg2") == 2

    def test_delete_nonexistent_meeting_returns_zero(self):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        vstore = _make_in_memory_vector_store()
        deleted = vstore.delete_meeting("nonexistent")
        assert deleted == 0


class TestVectorStoreSimilaritySearch:
    def test_similarity_search_returns_results(self):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        vstore = _make_in_memory_vector_store()
        # Insert 3 chunks
        for i in range(3):
            vstore.upsert(
                chunk_id=f"mtg1__transcript__{i}",
                text=f"Meeting content chunk {i}",
                embedding=[float(i + 1) / 10.0] * 768,
                meeting_id="mtg1",
                source_type="transcript",
                content_hash=f"hash{i}",
                chunk_index=i,
            )

        results = vstore.similarity_search(query_embedding=[0.1] * 768, top_k=3)
        assert len(results) > 0
        for r in results:
            assert "meeting_id" in r["metadata"]
            assert "source_type" in r["metadata"]
            assert "relevance_score" in r
            assert 0.0 <= r["relevance_score"] <= 1.0

    def test_similarity_search_empty_store_returns_empty(self):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        vstore = _make_in_memory_vector_store()
        results = vstore.similarity_search(query_embedding=[0.1] * 768, top_k=5)
        assert results == []

    def test_similarity_search_source_type_filter(self):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        vstore = _make_in_memory_vector_store()
        vstore.upsert(
            chunk_id="mtg1__decision__0", text="Decision text",
            embedding=[0.5] * 768, meeting_id="mtg1",
            source_type="decision", content_hash="hd0", chunk_index=0,
        )
        vstore.upsert(
            chunk_id="mtg1__transcript__0", text="Transcript text",
            embedding=[0.5] * 768, meeting_id="mtg1",
            source_type="transcript", content_hash="ht0", chunk_index=0,
        )

        results = vstore.similarity_search(
            query_embedding=[0.5] * 768, top_k=5, source_type_filter="decision"
        )
        assert all(r["metadata"]["source_type"] == "decision" for r in results)


class TestGetIndexedHashes:
    def test_returns_hashes_for_meeting(self):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        vstore = _make_in_memory_vector_store()
        vstore.upsert(
            chunk_id="mtg1__transcript__0", text="Some text",
            embedding=[0.1] * 768, meeting_id="mtg1",
            source_type="transcript", content_hash="abc123", chunk_index=0,
        )

        hashes = vstore.get_indexed_hashes(meeting_id="mtg1")
        assert "abc123" in hashes

    def test_returns_empty_for_unknown_meeting(self):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")

        vstore = _make_in_memory_vector_store()
        hashes = vstore.get_indexed_hashes(meeting_id="nonexistent")
        assert hashes == set()
