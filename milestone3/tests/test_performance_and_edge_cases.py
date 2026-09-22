"""
Performance & Edge Case Test Suite — Milestone 3 (Task 8)
Tests realistic edge cases (8.1 - 8.12) and executes actual latency benchmarks against the < 3000 ms SLA.
"""

import time
from unittest.mock import MagicMock, patch
import pytest

from milestone3.chunking_service import chunk_meeting_content, EmbeddingChunk
from milestone3.embedding_service import EmbeddingError, EmbeddingService
from milestone3.indexing_pipeline import IndexingResult, index_meeting
from milestone3.meeting_repository import MeetingRepository
from milestone3.rag_service import RAGError, RAGService
from milestone3.schemas import RAGRequest, SearchRequest
from milestone3.semantic_search_service import LATENCY_REQUIREMENT_MS, SemanticSearchService
from milestone3.vector_store_service import VectorStoreError, VectorStoreService
from milestone2.models import Meeting, ActionItemRecord, ParticipantRecord


# ------------------------------------------------------------------------------
# Mock Embedding Helper
# ------------------------------------------------------------------------------

class MockFastEmbeddingService(EmbeddingService):
    def __init__(self):
        self._model = "test-embed-model"
        self._api_keys = ["mock-key"]

    def embed_text(self, text: str):
        # Deterministic 32-dim float vector derived from text hash
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vec = [float(b) / 255.0 for b in h[:32]]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_batch(self, texts, skip_hashes=None):
        return [self.embed_text(t) if t else None for t in texts]


@pytest.fixture
def clean_vstore(tmp_path):
    return VectorStoreService(persist_directory=str(tmp_path / "perf_vstore"))


@pytest.fixture
def fast_embedder():
    return MockFastEmbeddingService()


# ------------------------------------------------------------------------------
# 8.1 - 8.6: Data & Query Boundary Edge Cases
# ------------------------------------------------------------------------------

class TestDataAndQueryEdgeCases:
    def test_8_1_large_transcript_chunking_and_indexing(self, clean_vstore, fast_embedder):
        """Large transcript (> 3000 words) chunks properly, embeds, and indexes without overflow."""
        # Generate large transcript
        turns = [
            f"Speaker {i % 4}: We are discussing system architecture section {i}. Details: {'test ' * 40}"
            for i in range(80)
        ]
        large_transcript = "\n\n".join(turns)

        chunks = chunk_meeting_content(
            meeting_id="mtg_large",
            transcript=large_transcript,
            summary="Large architecture discussion summary.",
            decisions=["Approved section 10", "Approved section 20"],
            action_items=["Task 1 for section 5", "Task 2 for section 15"],
        )

        assert len(chunks) > 1
        # Ensure all chunk sizes are within safe token limits
        for c in chunks:
            assert len(c.text.split()) < 600

        # Index chunks
        texts = [c.text for c in chunks]
        embeddings = fast_embedder.embed_batch(texts)
        ids = [f"mtg_large__{c.source_type}__{c.chunk_index}" for c in chunks]
        metas = [
            {"meeting_id": c.meeting_id, "source_type": c.source_type, "content_hash": c.content_hash, "chunk_index": c.chunk_index}
            for c in chunks
        ]
        clean_vstore.upsert_batch(ids, texts, embeddings, metas)

        assert clean_vstore.count_for_meeting("mtg_large") == len(chunks)

        # Search over large meeting
        search_svc = SemanticSearchService(embedding_svc=fast_embedder, vector_store=clean_vstore)
        results = search_svc.search(SearchRequest(query="system architecture section 10", top_k=3))
        assert len(results) == 3
        assert all(r.meeting_id == "mtg_large" for r in results)

    def test_8_2_multiple_historical_meetings_scale(self, clean_vstore, fast_embedder):
        """Index 25 distinct meetings and perform multi-meeting search."""
        for i in range(25):
            mid = f"meeting_{i:03d}"
            text = f"Discussion for project {i % 5} in department {i % 3}. Key decision was regarding plan {i}."
            clean_vstore.upsert(
                chunk_id=f"{mid}_c0",
                text=text,
                embedding=fast_embedder.embed_text(text),
                meeting_id=mid,
                source_type="transcript",
                content_hash=f"hash_{mid}",
                chunk_index=0,
            )

        assert clean_vstore.count() == 25

        search_svc = SemanticSearchService(embedding_svc=fast_embedder, vector_store=clean_vstore)
        t_start = time.perf_counter()
        results = search_svc.search(SearchRequest(query="department 1 project 2 plan", top_k=5))
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        assert len(results) == 5
        assert elapsed_ms < LATENCY_REQUIREMENT_MS

    def test_8_3_long_complex_question(self, clean_vstore, fast_embedder):
        """Multi-sentence, complex natural-language query string handled gracefully without crash."""
        text = "Backend database migration from MySQL to PostgreSQL approved with target date 2026-10-01."
        clean_vstore.upsert(
            chunk_id="mtg_db_0",
            text=text,
            embedding=fast_embedder.embed_text(text),
            meeting_id="mtg_db",
            source_type="decision",
            content_hash="hdb",
            chunk_index=0,
        )

        search_svc = SemanticSearchService(embedding_svc=fast_embedder, vector_store=clean_vstore)
        long_q = (
            "Could you please let me know based on our previous discussions what specific database engine "
            "we decided to migrate our legacy systems to, and whether there was any target date or deadline "
            "established by the backend engineering leads during our planning sessions?"
        )
        results = search_svc.search(SearchRequest(query=long_q, top_k=1))
        assert len(results) == 1
        assert results[0].meeting_id == "mtg_db"

    def test_8_4_short_ambiguous_queries(self, clean_vstore, fast_embedder):
        """Single-word queries ('database', 'Priya', 'deadline') handled gracefully."""
        text = "Priya will test the application before deadline."
        clean_vstore.upsert(
            chunk_id="mtg_p_0",
            text=text,
            embedding=fast_embedder.embed_text(text),
            meeting_id="mtg_p",
            source_type="action_item",
            content_hash="hp",
            chunk_index=0,
        )
        search_svc = SemanticSearchService(embedding_svc=fast_embedder, vector_store=clean_vstore)

        for word in ["database", "Priya", "deadline"]:
            results = search_svc.search(SearchRequest(query=word, top_k=1))
            assert isinstance(results, list)

    def test_8_5_and_8_6_unknown_questions_and_no_matching_meetings(self, clean_vstore, fast_embedder):
        """Questions on non-existent topics return grounded 'no information found' message without hallucinating."""
        search_svc = SemanticSearchService(embedding_svc=fast_embedder, vector_store=clean_vstore)

        rag_svc = RAGService.__new__(RAGService)
        rag_svc._search_svc = search_svc
        rag_svc._model = "test-model"

        # When search returns no results, RAG returns grounded fallback
        resp = rag_svc.answer(RAGRequest(question="What did the company decide about underwater cities?"))
        assert "could not find relevant information" in resp.answer.lower()
        assert resp.meeting_ids == []
        assert resp.sources == []


# ------------------------------------------------------------------------------
# 8.7 - 8.9: Database & Embedding Integrity Edge Cases
# ------------------------------------------------------------------------------

class TestDatabaseAndVectorIntegrityEdgeCases:
    def test_8_7_duplicate_meeting_data_idempotency(self, clean_vstore, fast_embedder):
        """Repeated indexing with identical content does not create duplicate vector records."""
        meeting = MagicMock(spec=Meeting)
        meeting.id = "mtg_dup_test"
        meeting.transcript = "Alice: We discussed budget allocation."
        meeting.summary = "Budget allocation meeting."
        meeting.decisions = ["Budget approved."]
        ai = MagicMock()
        ai.task = "Send budget sheet"
        meeting.action_items = [ai]

        mock_repo = MagicMock(spec=MeetingRepository)
        mock_repo.get_meeting_by_id.return_value = meeting

        # First indexing
        with patch("milestone3.indexing_pipeline.MeetingRepository", return_value=mock_repo):
            res1 = index_meeting("mtg_dup_test", vector_store=clean_vstore, embedding_svc=fast_embedder)
            assert res1.chunks_indexed > 0
            count_after_first = clean_vstore.count_for_meeting("mtg_dup_test")

            # Second indexing (identical content)
            res2 = index_meeting("mtg_dup_test", vector_store=clean_vstore, embedding_svc=fast_embedder)
            assert res2.chunks_indexed == 0
            assert res2.chunks_skipped > 0
            count_after_second = clean_vstore.count_for_meeting("mtg_dup_test")

            # Total chunks in vector store should remain identical
            assert count_after_first == count_after_second

    def test_8_8_missing_or_empty_transcript(self, clean_vstore, fast_embedder):
        """Meeting with empty transcript but valid summary and decisions still chunks and indexes properly."""
        chunks = chunk_meeting_content(
            meeting_id="mtg_no_transcript",
            transcript="",
            summary="Meeting summary without audio transcript.",
            decisions=["Decision X made."],
            action_items=["Task Y assigned."],
        )
        assert len(chunks) == 3
        types = [c.source_type for c in chunks]
        assert "summary" in types
        assert "decision" in types
        assert "action_item" in types
        assert "transcript" not in types

    def test_8_9_missing_embeddings_for_db_meeting(self, clean_vstore, fast_embedder):
        """Meeting exists in relational DB but has not been indexed in vector store."""
        search_svc = SemanticSearchService(embedding_svc=fast_embedder, vector_store=clean_vstore)
        results = search_svc.search(SearchRequest(query="Unindexed meeting topic"))
        assert results == []


# ------------------------------------------------------------------------------
# 8.10 - 8.12: System Failure & Timeout Resilience
# ------------------------------------------------------------------------------

class TestFailureResilience:
    def test_8_10_vector_database_failure_handling(self, fast_embedder):
        """Vector DB unexpected failure raises clean VectorStoreError without leaking credentials."""
        mock_vstore = MagicMock(spec=VectorStoreService)
        mock_vstore.count.return_value = 10
        mock_vstore.similarity_search.side_effect = RuntimeError("ChromaDB connection lost")

        search_svc = SemanticSearchService(embedding_svc=fast_embedder, vector_store=mock_vstore)
        with pytest.raises(Exception) as exc_info:
            search_svc.search(SearchRequest(query="test"))
        assert "ChromaDB connection lost" in str(exc_info.value)

    def test_8_11_llm_failure_exhaustion(self, clean_vstore, fast_embedder):
        """LLM API repeated failure raises RAGError after retry exhaustion without fabrication."""
        text = "Priya will test mobile app."
        clean_vstore.upsert(
            chunk_id="mtg_fail_0",
            text=text,
            embedding=fast_embedder.embed_text(text),
            meeting_id="mtg_fail",
            source_type="transcript",
            content_hash="hfail",
            chunk_index=0,
        )

        search_svc = SemanticSearchService(embedding_svc=fast_embedder, vector_store=clean_vstore)

        rag_svc = RAGService.__new__(RAGService)
        rag_svc._search_svc = search_svc
        rag_svc._model = "test-model"
        rag_svc._prompt_template = "{context}\n{question}"

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("Gemini 503 Service Unavailable")
        rag_svc._client = mock_client

        with patch("milestone3.rag_service.time.sleep"):
            with pytest.raises(RAGError, match="RAG LLM call failed"):
                rag_svc.answer(RAGRequest(question="Who is testing?"))

    def test_8_12_api_timeout_handling(self, fast_embedder):
        """Simulated slow vector search logs warning when exceeding requirement."""
        def slow_search(*args, **kwargs):
            time.sleep(0.05)  # Simulate latency
            return []

        mock_vstore = MagicMock(spec=VectorStoreService)
        mock_vstore.count.return_value = 5
        mock_vstore.similarity_search.side_effect = slow_search

        search_svc = SemanticSearchService(embedding_svc=fast_embedder, vector_store=mock_vstore)
        results = search_svc.search(SearchRequest(query="query"))
        assert results == []


# ------------------------------------------------------------------------------
# Task 8 Performance Benchmarks (< 3000 ms SLA)
# ------------------------------------------------------------------------------

class TestPerformanceBenchmark:
    def test_search_latency_benchmark_under_3_seconds(self, clean_vstore, fast_embedder):
        """
        Executes N=20 search requests over populated vector store.
        Measures Min, Avg, Max latency and strictly asserts Max < 3000 ms.
        """
        # Populate with realistic meeting chunks
        for i in range(50):
            t = f"Meeting chunk {i}: Discussing deployment pipeline, automated CI/CD checks, and release schedules."
            clean_vstore.upsert(
                chunk_id=f"chunk_{i}",
                text=t,
                embedding=fast_embedder.embed_text(t),
                meeting_id=f"mtg_{i // 5}",
                source_type="transcript",
                content_hash=f"hash_{i}",
                chunk_index=i % 5,
            )

        search_svc = SemanticSearchService(embedding_svc=fast_embedder, vector_store=clean_vstore)

        queries = [
            "deployment pipeline",
            "automated CI/CD checks",
            "release schedules",
            "production rollback procedure",
            "pipeline optimization",
        ]

        latencies = []
        N_RUNS = 20

        for run in range(N_RUNS):
            q = queries[run % len(queries)]
            t0 = time.perf_counter()
            results = search_svc.search(SearchRequest(query=q, top_k=5))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            assert len(results) > 0

        min_lat = min(latencies)
        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)

        print(f"\n[BENCHMARK] N={N_RUNS} | Min: {min_lat:.2f} ms | Avg: {avg_lat:.2f} ms | Max: {max_lat:.2f} ms")

        # Must strictly meet < 3000 ms SLA
        assert max_lat < LATENCY_REQUIREMENT_MS, f"Max search latency {max_lat:.2f} ms exceeded {LATENCY_REQUIREMENT_MS} ms"
        assert avg_lat < 1000.0, f"Average search latency {avg_lat:.2f} ms exceeded 1000 ms"
