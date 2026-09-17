"""
Tests for milestone3.schemas — Pydantic data contract validation.
"""

import pytest
from pydantic import ValidationError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from milestone3.schemas import (
    RAGRequest,
    RAGResponse,
    SearchRequest,
    SearchResult,
)


class TestSearchRequest:
    def test_valid_minimal(self):
        req = SearchRequest(query="budget decisions")
        assert req.query == "budget decisions"
        assert req.top_k == 5
        assert req.source_type is None

    def test_valid_with_all_fields(self):
        req = SearchRequest(query="action items", top_k=10, source_type="action_item")
        assert req.top_k == 10
        assert req.source_type == "action_item"

    def test_valid_source_types(self):
        for st in ["transcript", "summary", "decision", "action_item"]:
            req = SearchRequest(query="test", source_type=st)
            assert req.source_type == st

    def test_invalid_source_type(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", source_type="unknown_type")

    def test_top_k_minimum(self):
        req = SearchRequest(query="test", top_k=1)
        assert req.top_k == 1

    def test_top_k_maximum(self):
        req = SearchRequest(query="test", top_k=50)
        assert req.top_k == 50

    def test_top_k_out_of_range(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=0)
        with pytest.raises(ValidationError):
            SearchRequest(query="test", top_k=51)

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="")


class TestSearchResult:
    def test_valid(self):
        result = SearchResult(
            meeting_id="abc-123",
            relevance_score=0.87,
            source_type="transcript",
            content="We agreed to ship by Friday.",
            search_latency_ms=150.5,
        )
        assert result.meeting_id == "abc-123"
        assert result.relevance_score == pytest.approx(0.87)
        assert result.source_type == "transcript"
        assert result.search_latency_ms == pytest.approx(150.5)

    def test_relevance_score_bounds(self):
        # Edge values
        SearchResult(meeting_id="x", relevance_score=0.0, source_type="summary", content="text")
        SearchResult(meeting_id="x", relevance_score=1.0, source_type="summary", content="text")

    def test_relevance_score_out_of_bounds(self):
        with pytest.raises(ValidationError):
            SearchResult(meeting_id="x", relevance_score=1.1, source_type="summary", content="text")
        with pytest.raises(ValidationError):
            SearchResult(meeting_id="x", relevance_score=-0.1, source_type="summary", content="text")

    def test_default_latency(self):
        result = SearchResult(meeting_id="x", relevance_score=0.5, source_type="decision", content="ok")
        assert result.search_latency_ms == 0.0


class TestRAGRequest:
    def test_valid_minimal(self):
        req = RAGRequest(question="Who is responsible for the backend?")
        assert req.question == "Who is responsible for the backend?"
        assert req.top_k == 5

    def test_valid_custom_top_k(self):
        req = RAGRequest(question="When is the deadline?", top_k=3)
        assert req.top_k == 3

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            RAGRequest(question="")

    def test_top_k_out_of_range(self):
        with pytest.raises(ValidationError):
            RAGRequest(question="test", top_k=0)
        with pytest.raises(ValidationError):
            RAGRequest(question="test", top_k=21)


class TestRAGResponse:
    def test_valid_full(self):
        sources = [
            SearchResult(
                meeting_id="m1",
                relevance_score=0.9,
                source_type="decision",
                content="Ship by Friday.",
            )
        ]
        resp = RAGResponse(
            answer="The team decided to ship by Friday.",
            meeting_ids=["m1"],
            sources=sources,
            latency_ms=1200.0,
        )
        assert resp.answer == "The team decided to ship by Friday."
        assert resp.meeting_ids == ["m1"]
        assert len(resp.sources) == 1
        assert resp.latency_ms == pytest.approx(1200.0)

    def test_defaults(self):
        resp = RAGResponse(answer="No information found.")
        assert resp.meeting_ids == []
        assert resp.sources == []
        assert resp.latency_ms == 0.0
