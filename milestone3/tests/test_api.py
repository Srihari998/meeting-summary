"""
Tests for milestone3.api — REST API Layer (Task 6).
Tests endpoints /meetings, /meetings/{id}, /search, /ask, /health, /index and authentication.
Uses FastAPI TestClient with mocked services and SQLite database.
"""

import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from milestone3.api import app, get_repository, get_search_service, get_rag_service
from milestone3.schemas import SearchResult, RAGResponse


@pytest.fixture
def client():
    """Returns a TestClient with clean environment."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("API_AUTH_TOKEN", None)
        os.environ.pop("API_KEY", None)
        with TestClient(app) as tc:
            yield tc


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    meeting = MagicMock()
    meeting.id = "mtg-101"
    meeting.summary = "Executive summary of backend planning."
    meeting.transcript = "Alice: We will migrate database to PostgreSQL."
    meeting.key_points = ["DB migration", "Performance tuning"]
    meeting.decisions = ["Approved PostgreSQL migration"]
    meeting.created_at = None
    
    ai = MagicMock()
    ai.id = 1
    ai.task = "Complete migration schema"
    ai.assignee = "Bob"
    ai.deadline = "2026-09-30"
    ai.priority = "High"
    ai.status = "In Progress"
    meeting.action_items = [ai]

    p = MagicMock()
    p.id = 1
    p.name = "Alice"
    p.is_unknown_assignee = False
    meeting.participants = [p]

    repo.get_all_meetings.return_value = [meeting]
    repo.get_meeting_by_id.side_effect = lambda mid: meeting if mid == "mtg-101" else None
    repo.count_meetings.return_value = 1
    return repo


@pytest.fixture
def mock_search_svc():
    svc = MagicMock()
    svc.search.return_value = [
        SearchResult(
            meeting_id="mtg-101",
            relevance_score=0.92,
            source_type="transcript",
            content="Alice: We will migrate database to PostgreSQL.",
            search_latency_ms=45.2,
        )
    ]
    return svc


@pytest.fixture
def mock_rag_svc():
    svc = MagicMock()
    svc.answer.return_value = RAGResponse(
        answer="The team approved the PostgreSQL migration.",
        meeting_ids=["mtg-101"],
        sources=[
            SearchResult(
                meeting_id="mtg-101",
                relevance_score=0.92,
                source_type="decision",
                content="Approved PostgreSQL migration",
                search_latency_ms=45.2,
            )
        ],
        latency_ms=120.5,
    )
    return svc


class TestHealthEndpoint:
    def test_health_check_success(self, client, mock_repo):
        app.dependency_overrides[get_repository] = lambda: mock_repo
        try:
            with patch("milestone3.api.VectorStoreService") as mock_vstore:
                mock_vstore.return_value.count.return_value = 5
                resp = client.get("/health")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "healthy"
                assert data["meetings_in_db"] == 1
                assert data["chunks_indexed"] == 5
        finally:
            app.dependency_overrides.clear()


class TestMeetingsEndpoints:
    def test_get_all_meetings(self, client, mock_repo):
        app.dependency_overrides[get_repository] = lambda: mock_repo
        try:
            resp = client.get("/meetings")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 1
            assert len(data["meetings"]) == 1
            assert data["meetings"][0]["id"] == "mtg-101"
            assert data["meetings"][0]["action_items_count"] == 1
            assert data["meetings"][0]["participants_count"] == 1
        finally:
            app.dependency_overrides.clear()

    def test_get_meeting_by_id_found(self, client, mock_repo):
        app.dependency_overrides[get_repository] = lambda: mock_repo
        try:
            resp = client.get("/meetings/mtg-101")
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == "mtg-101"
            assert "PostgreSQL" in data["transcript"]
            assert len(data["action_items"]) == 1
            assert data["action_items"][0]["assignee"] == "Bob"
            assert len(data["participants"]) == 1
        finally:
            app.dependency_overrides.clear()

    def test_get_meeting_by_id_not_found(self, client, mock_repo):
        app.dependency_overrides[get_repository] = lambda: mock_repo
        try:
            resp = client.get("/meetings/nonexistent-id")
            assert resp.status_code == 404
            assert "not found" in resp.json()["detail"].lower()
        finally:
            app.dependency_overrides.clear()


class TestSearchEndpoint:
    def test_search_success(self, client, mock_search_svc):
        app.dependency_overrides[get_search_service] = lambda: mock_search_svc
        try:
            resp = client.post("/search", json={"query": "database migration", "top_k": 3})
            assert resp.status_code == 200
            data = resp.json()
            assert data["query"] == "database migration"
            assert data["total_results"] == 1
            assert data["results"][0]["meeting_id"] == "mtg-101"
            assert data["results"][0]["relevance_score"] == 0.92
        finally:
            app.dependency_overrides.clear()

    def test_search_empty_query_rejected(self, client):
        resp = client.post("/search", json={"query": "", "top_k": 3})
        assert resp.status_code == 422

    def test_search_empty_results(self, client):
        empty_svc = MagicMock()
        empty_svc.search.return_value = []
        app.dependency_overrides[get_search_service] = lambda: empty_svc
        try:
            resp = client.post("/search", json={"query": "mars mission"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_results"] == 0
            assert data["results"] == []
        finally:
            app.dependency_overrides.clear()


class TestAskEndpoint:
    def test_ask_rag_success(self, client, mock_rag_svc):
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_svc
        try:
            resp = client.post("/ask", json={"question": "What database was chosen?"})
            assert resp.status_code == 200
            data = resp.json()
            assert "PostgreSQL" in data["answer"]
            assert data["meeting_ids"] == ["mtg-101"]
            assert len(data["sources"]) == 1
        finally:
            app.dependency_overrides.clear()

    def test_ask_empty_question_rejected(self, client):
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422


class TestAuthentication:
    def test_auth_enforced_when_configured(self, mock_repo):
        app.dependency_overrides[get_repository] = lambda: mock_repo
        try:
            with patch.dict(os.environ, {"API_AUTH_TOKEN": "secret-token-123"}):
                with TestClient(app) as tc:
                    # 1. No token -> 401
                    resp = tc.get("/meetings")
                    assert resp.status_code == 401

                    # 2. Invalid token -> 401
                    resp = tc.get("/meetings", headers={"X-API-Key": "wrong-token"})
                    assert resp.status_code == 401

                    # 3. Valid X-API-Key -> 200
                    resp = tc.get("/meetings", headers={"X-API-Key": "secret-token-123"})
                    assert resp.status_code == 200

                    # 4. Valid Bearer Token -> 200
                    resp = tc.get("/meetings", headers={"Authorization": "Bearer secret-token-123"})
                    assert resp.status_code == 200
        finally:
            app.dependency_overrides.clear()
