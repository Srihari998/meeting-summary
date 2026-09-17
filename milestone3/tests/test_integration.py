"""
Integration Test — Milestone 3 end-to-end pipeline.
SKIPPED unless RUN_LIVE_TESTS=1 environment variable is set.

Requires:
    - GEMINI_API_KEY set in environment / .env
    - At least one meeting in the SQLite database
    - chromadb and google-genai packages installed
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

RUN_LIVE = os.environ.get("RUN_LIVE_TESTS", "0") == "1"
skip_reason = "Skipped: set RUN_LIVE_TESTS=1 to run live integration tests."


@pytest.mark.skipif(not RUN_LIVE, reason=skip_reason)
class TestMilestone3Integration:
    """
    Full end-to-end integration test:
    1. Seed a test meeting into the DB
    2. Index it into ChromaDB
    3. Run semantic search
    4. Run RAG question answering
    5. Verify latency < 3000ms
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up temp DB and chroma dir for isolation."""
        import tempfile
        self._tmp_db = tempfile.mktemp(suffix=".db")
        self._tmp_chroma = tempfile.mkdtemp(prefix="chroma_test_")
        os.environ["CHROMA_PERSIST_DIR"] = self._tmp_chroma
        yield
        # Cleanup
        if os.path.exists(self._tmp_db):
            os.remove(self._tmp_db)
        import shutil
        shutil.rmtree(self._tmp_chroma, ignore_errors=True)

    def _seed_meeting(self, db_path: str) -> str:
        """Creates a test meeting in the temp DB and returns its ID."""
        from milestone2.schemas import ActionItem, MeetingIntelligence
        from milestone2.db import save_meeting

        mi = MeetingIntelligence(
            summary="The team decided to launch the new feature in Q4 and assigned John as the lead.",
            key_points=["Feature launch scheduled for Q4", "John assigned as lead developer"],
            decisions=["Launch in Q4", "John leads the integration"],
            action_items=[
                ActionItem(
                    task="Complete backend API integration",
                    assignee="John",
                    deadline="2025-11-30",
                    priority="High",
                    status="Not Started",
                )
            ],
            participants=["Alice", "John", "Sarah"],
        )

        transcript = (
            "Alice: Good morning everyone. Let's discuss the Q4 launch.\n"
            "John: I'm ready to lead the backend integration.\n"
            "Sarah: We should set the deadline for end of November.\n"
            "Alice: Agreed. John, you're the lead. Launch is set for Q4."
        )

        meeting_id = save_meeting(transcript, mi, db_path=db_path)
        return meeting_id

    def test_full_pipeline(self):
        from milestone3.indexing_pipeline import index_meeting
        from milestone3.schemas import RAGRequest, SearchRequest
        from milestone3.semantic_search_service import SemanticSearchService
        from milestone3.rag_service import RAGService
        from milestone3.vector_store_service import VectorStoreService

        # 1. Seed meeting
        meeting_id = self._seed_meeting(self._tmp_db)
        assert meeting_id is not None

        # 2. Index into vector store
        vstore = VectorStoreService(persist_directory=self._tmp_chroma)
        result = index_meeting(meeting_id=meeting_id, db_path=self._tmp_db, vector_store=vstore)

        assert result.success, f"Indexing errors: {result.errors}"
        assert result.chunks_indexed > 0
        assert meeting_id in result.meeting_ids_processed

        # 3. Semantic search
        search_svc = SemanticSearchService(vector_store=vstore)
        search_req = SearchRequest(query="Q4 launch decision", top_k=5)
        search_results = search_svc.search(search_req)

        assert len(search_results) > 0, "Expected at least one search result."
        assert search_results[0].meeting_id == meeting_id
        assert search_results[0].relevance_score > 0.0

        # Latency requirement: < 3000ms
        assert search_results[0].search_latency_ms < 3000.0, (
            f"Search latency {search_results[0].search_latency_ms:.1f} ms exceeded 3000 ms requirement."
        )

        # 4. RAG question answering
        rag_svc = RAGService(search_svc=search_svc)
        rag_resp = rag_svc.answer(RAGRequest(question="Who is leading the backend integration?"))

        assert rag_resp.answer is not None
        assert len(rag_resp.answer) > 0
        assert meeting_id in rag_resp.meeting_ids
        assert len(rag_resp.sources) > 0
        assert rag_resp.latency_ms > 0

        # 5. RAG should NOT fabricate when asked about unknown info
        no_info_resp = rag_svc.answer(RAGRequest(question="What did the CFO say about the dividend payout?"))
        # The answer should indicate no information or be grounded — it should not mention fabricated content
        # We can't assert the exact text, but we verify the structure is valid
        assert isinstance(no_info_resp.answer, str)
        assert len(no_info_resp.answer) > 0

    def test_idempotent_reindex(self):
        """Re-indexing a meeting should not create duplicate chunks."""
        from milestone3.indexing_pipeline import index_meeting
        from milestone3.vector_store_service import VectorStoreService

        vstore = VectorStoreService(persist_directory=self._tmp_chroma)
        meeting_id = self._seed_meeting(self._tmp_db)

        # Index twice
        r1 = index_meeting(meeting_id=meeting_id, db_path=self._tmp_db, vector_store=vstore)
        count_after_first = vstore.count_for_meeting(meeting_id)

        r2 = index_meeting(meeting_id=meeting_id, db_path=self._tmp_db, vector_store=vstore)
        count_after_second = vstore.count_for_meeting(meeting_id)

        assert count_after_first == count_after_second, (
            f"Idempotency violation: count changed from {count_after_first} to {count_after_second}"
        )
        assert r2.chunks_indexed == 0, "Second indexing should skip all already-indexed chunks."
        assert r2.chunks_skipped == count_after_first
