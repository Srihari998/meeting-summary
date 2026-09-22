"""
End-to-End Integration Test Suite — Milestone 3 (Task 9)
Tests the complete multi-milestone intelligence pipeline:
Transcript -> M2 Structured Intelligence Extraction -> SQLite DB Persistence ->
M3 Chunking & Embedding -> ChromaDB Indexing -> Semantic Search -> RAG Grounded QA -> REST API.
"""

import json
import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from milestone2.db import get_session, init_db, save_meeting
from milestone2.models import ActionItemRecord, Meeting, ParticipantRecord
from milestone2.pipeline import process_meeting
from milestone2.schemas import ActionItem, MeetingIntelligence
from milestone3.api import app
from milestone3.chunking_service import chunk_meeting_content
from milestone3.embedding_service import EmbeddingService
from milestone3.indexing_pipeline import index_meeting
from milestone3.meeting_repository import MeetingRepository
from milestone3.rag_service import RAGService
from milestone3.schemas import RAGRequest, RAGResponse, SearchRequest
from milestone3.semantic_search_service import SemanticSearchService
from milestone3.vector_store_service import VectorStoreService


class MockE2EEmbeddingService(EmbeddingService):
    def __init__(self):
        self._model = "e2e-embed-model"
        self._api_keys = ["mock-key"]

    def embed_text(self, text: str):
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vec = [float(b) / 255.0 for b in h[:16]]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_batch(self, texts, skip_hashes=None):
        return [self.embed_text(t) if t else None for t in texts]


@pytest.fixture
def e2e_env(tmp_path):
    """Sets up an isolated SQLite database and ChromaDB vector store for E2E testing."""
    db_file = str(tmp_path / "e2e_meeting.db")
    chroma_dir = str(tmp_path / "e2e_chroma")

    init_db(db_file)
    vstore = VectorStoreService(persist_directory=chroma_dir)
    embedder = MockE2EEmbeddingService()
    repo = MeetingRepository(db_path=db_file)

    return {
        "db_file": db_file,
        "chroma_dir": chroma_dir,
        "vstore": vstore,
        "embedder": embedder,
        "repo": repo,
    }


class TestMilestone3EndToEndPipeline:
    def test_full_pipeline_e2e(self, e2e_env):
        db_file = e2e_env["db_file"]
        vstore = e2e_env["vstore"]
        embedder = e2e_env["embedder"]
        repo = e2e_env["repo"]

        meeting_id = "mtg_e2e_001"
        raw_transcript = (
            "Alice: We need to migrate our database from MySQL to PostgreSQL before Q4.\n"
            "Bob: I agree. I will finalize the PostgreSQL schema migration scripts by 2026-10-15.\n"
            "Alice: Great. Let's make sure Priya reviews the schema changes."
        )

        mock_llm_intelligence = MeetingIntelligence(
            summary="Discussion regarding database migration from MySQL to PostgreSQL for Q4.",
            key_points=[
                "Migrating primary database from MySQL to PostgreSQL",
                "Schema migration scripts required before Q4",
            ],
            decisions=[
                "Approved PostgreSQL as target database engine",
            ],
            action_items=[
                ActionItem(
                    task="Finalize PostgreSQL schema migration scripts",
                    assignee="Bob",
                    deadline="2026-10-15",
                    priority="High",
                    status="Not Started",
                )
            ],
            participants=["Alice", "Bob", "Priya"],
        )

        # ----------------------------------------------------------------------
        # STAGE 1: Process Meeting & Persist to Relational DB (Milestone 2)
        # ----------------------------------------------------------------------
        with patch("milestone2.pipeline.extract_meeting_intelligence", return_value=mock_llm_intelligence):
            intel = process_meeting(raw_transcript, meeting_id=meeting_id, db_path=db_file)
            assert intel is not None
            assert "MySQL to PostgreSQL" in intel.summary

        # Verify DB records
        saved_meeting = repo.get_meeting_by_id(meeting_id)
        assert saved_meeting is not None
        assert "MySQL to PostgreSQL" in saved_meeting.summary
        assert len(saved_meeting.action_items) == 1
        assert saved_meeting.action_items[0].assignee == "Bob"
        assert saved_meeting.action_items[0].deadline == "2026-10-15"
        assert len(saved_meeting.participants) >= 2

        # ----------------------------------------------------------------------
        # STAGE 2: Index into Knowledge Repository Vector Store (Milestone 3)
        # ----------------------------------------------------------------------
        idx_res = index_meeting(
            meeting_id=meeting_id,
            db_path=db_file,
            vector_store=vstore,
            embedding_svc=embedder,
        )
        assert idx_res.success
        assert idx_res.chunks_indexed > 0
        assert vstore.count_for_meeting(meeting_id) == idx_res.chunks_indexed

        # ----------------------------------------------------------------------
        # STAGE 3: Semantic Search Retrieval (Milestone 3 Task 4)
        # ----------------------------------------------------------------------
        search_svc = SemanticSearchService(embedding_svc=embedder, vector_store=vstore)
        search_req = SearchRequest(query="PostgreSQL database migration", top_k=3)
        search_results = search_svc.search(search_req)

        assert len(search_results) > 0
        top_res = search_results[0]
        assert top_res.meeting_id == meeting_id
        assert top_res.relevance_score > 0.0
        assert top_res.search_latency_ms > 0.0

        # ----------------------------------------------------------------------
        # STAGE 4: RAG Grounded Question Answering (Milestone 3 Task 5)
        # ----------------------------------------------------------------------
        rag_svc = RAGService.__new__(RAGService)
        rag_svc._search_svc = search_svc
        rag_svc._model = "test-model"
        rag_svc._prompt_template = "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"

        mock_answer_text = (
            "Bob is responsible for finalizing the PostgreSQL schema migration scripts "
            "by 2026-10-15 [Meeting: mtg_e2e_001]."
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text=mock_answer_text)
        rag_svc._client = mock_client

        rag_resp = rag_svc.answer(RAGRequest(question="What is the deadline for database migration scripts?"))
        assert isinstance(rag_resp, RAGResponse)
        assert "2026-10-15" in rag_resp.answer
        assert meeting_id in rag_resp.meeting_ids
        assert len(rag_resp.sources) > 0

        # ----------------------------------------------------------------------
        # STAGE 5: Unsupported Question (No hallucination fallback)
        # ----------------------------------------------------------------------
        empty_search_svc = MagicMock()
        empty_search_svc.search.return_value = []
        unsupported_rag_svc = RAGService.__new__(RAGService)
        unsupported_rag_svc._search_svc = empty_search_svc
        unsupported_rag_svc._model = "test-model"

        unsupported_resp = unsupported_rag_svc.answer(
            RAGRequest(question="What is the quantum computing architecture?")
        )
        assert "could not find relevant information" in unsupported_resp.answer.lower()
        assert unsupported_resp.meeting_ids == []

    def test_e2e_api_workflow_with_authentication(self, e2e_env):
        """Tests the complete workflow via FastAPI REST endpoints with authentication."""
        db_file = e2e_env["db_file"]
        vstore = e2e_env["vstore"]
        embedder = e2e_env["embedder"]
        repo = e2e_env["repo"]

        # Insert a sample meeting in DB and vector store
        meeting_id = "mtg_api_e2e"
        parsed = MeetingIntelligence(
            summary="API End-to-end meeting summary.",
            key_points=["Point 1", "Point 2"],
            decisions=["Decision Alpha"],
            action_items=[
                ActionItem(
                    task="Deploy API to production",
                    assignee="DevOps",
                    deadline="Tomorrow",
                    priority="High",
                    status="In Progress",
                )
            ],
            participants=["DevOps Lead"],
        )
        save_meeting("Alice: Deploy API tomorrow.", parsed, meeting_id=meeting_id, db_path=db_file)
        index_meeting(meeting_id=meeting_id, db_path=db_file, vector_store=vstore, embedding_svc=embedder)

        search_svc = SemanticSearchService(embedding_svc=embedder, vector_store=vstore)

        rag_svc = RAGService.__new__(RAGService)
        rag_svc._search_svc = search_svc
        rag_svc._model = "test-model"
        rag_svc._prompt_template = "{context}\n{question}"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text="The deployment is scheduled for Tomorrow by DevOps [Meeting: mtg_api_e2e]."
        )
        rag_svc._client = mock_client

        from milestone3.api import get_repository, get_search_service, get_rag_service

        app.dependency_overrides[get_repository] = lambda: repo
        app.dependency_overrides[get_search_service] = lambda: search_svc
        app.dependency_overrides[get_rag_service] = lambda: rag_svc

        token = "test-auth-secret-key"
        try:
            with patch.dict(os.environ, {"API_AUTH_TOKEN": token}):
                with TestClient(app) as client:
                    # 1. Unauthenticated -> 401
                    resp = client.get("/meetings")
                    assert resp.status_code == 401

                    # 2. Authenticated GET /meetings -> 200
                    auth_headers = {"X-API-Key": token}
                    resp = client.get("/meetings", headers=auth_headers)
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["total"] >= 1
                    assert any(m["id"] == meeting_id for m in data["meetings"])

                    # 3. Authenticated GET /meetings/{id} -> 200
                    resp = client.get(f"/meetings/{meeting_id}", headers=auth_headers)
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["id"] == meeting_id
                    assert data["action_items"][0]["assignee"] == "DevOps"

                    # 4. Authenticated POST /search -> 200
                    resp = client.post(
                        "/search",
                        headers=auth_headers,
                        json={"query": "Deploy API", "top_k": 2},
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["total_results"] > 0
                    assert data["results"][0]["meeting_id"] == meeting_id

                    # 5. Authenticated POST /ask -> 200
                    resp = client.post(
                        "/ask",
                        headers=auth_headers,
                        json={"question": "When is the deployment?"},
                    )
                    assert resp.status_code == 200
                    data = resp.json()
                    assert "Tomorrow" in data["answer"]
                    assert meeting_id in data["meeting_ids"]
        finally:
            app.dependency_overrides.clear()
