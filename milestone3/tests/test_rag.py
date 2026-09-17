"""
Tests for milestone3.rag_service — RAG pipeline (retrieve → ground → answer).
Mocks SemanticSearchService and Gemini LLM — no real API calls.
"""

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from milestone3.schemas import RAGRequest, RAGResponse, SearchResult
from milestone3.rag_service import RAGService, RAGError


def _make_search_result(meeting_id="mtg1", source_type="transcript", content="Test content", score=0.85):
    return SearchResult(
        meeting_id=meeting_id,
        relevance_score=score,
        source_type=source_type,
        content=content,
        search_latency_ms=100.0,
    )


def _make_mock_search_svc(results: List[SearchResult] = None):
    if results is None:
        results = [_make_search_result()]
    svc = MagicMock()
    svc.search.return_value = results
    return svc


def _make_rag_service(search_results=None, llm_answer="The answer is X."):
    """Creates a RAGService with mocked search and mocked LLM."""
    search_svc = _make_mock_search_svc(search_results)

    # Build RAGService bypassing __init__ to inject mocks
    svc = RAGService.__new__(RAGService)
    svc._search_svc = search_svc
    svc._model = "gemini-3.6-flash"

    # Mock LLM client
    mock_response = MagicMock()
    mock_response.text = llm_answer
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    svc._client = mock_client

    # Set prompt template directly
    svc._prompt_template = (
        "MEETING EXCERPTS:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    )

    return svc


class TestRAGServiceAnswer:
    def test_returns_rag_response(self):
        svc = _make_rag_service()
        resp = svc.answer(RAGRequest(question="Who owns the API task?"))
        assert isinstance(resp, RAGResponse)

    def test_answer_is_non_empty(self):
        svc = _make_rag_service(llm_answer="John owns the API task.")
        resp = svc.answer(RAGRequest(question="Who owns the API task?"))
        assert resp.answer == "John owns the API task."

    def test_meeting_ids_populated(self):
        results = [
            _make_search_result(meeting_id="mtg-A"),
            _make_search_result(meeting_id="mtg-B"),
            _make_search_result(meeting_id="mtg-A"),  # duplicate — should dedupe
        ]
        svc = _make_rag_service(search_results=results)
        resp = svc.answer(RAGRequest(question="test question"))
        assert set(resp.meeting_ids) == {"mtg-A", "mtg-B"}

    def test_sources_populated(self):
        results = [_make_search_result(), _make_search_result(meeting_id="mtg2")]
        svc = _make_rag_service(search_results=results)
        resp = svc.answer(RAGRequest(question="test"))
        assert len(resp.sources) == 2

    def test_latency_ms_is_positive(self):
        svc = _make_rag_service()
        resp = svc.answer(RAGRequest(question="test"))
        assert resp.latency_ms > 0

    def test_no_sources_returns_no_information_response(self):
        svc = _make_rag_service(search_results=[])
        resp = svc.answer(RAGRequest(question="nonexistent topic"))
        assert "could not find" in resp.answer.lower()
        assert resp.meeting_ids == []
        assert resp.sources == []

    def test_llm_called_with_context_and_question(self):
        """The LLM prompt must include both context from search results and the user question."""
        svc = _make_rag_service(
            search_results=[_make_search_result(content="Budget is 100k.")],
            llm_answer="The budget is 100k.",
        )
        svc.answer(RAGRequest(question="What is the budget?"))
        call_args = svc._client.models.generate_content.call_args
        prompt_sent = call_args[1]["contents"]
        assert "Budget is 100k." in prompt_sent
        assert "What is the budget?" in prompt_sent

    def test_llm_retry_on_failure(self):
        """Should retry on LLM failure and succeed on the third attempt."""
        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("Transient error")
            resp = MagicMock()
            resp.text = "Retry succeeded."
            return resp

        svc = _make_rag_service()
        svc._client.models.generate_content.side_effect = side_effect

        with patch("milestone3.rag_service.time.sleep"):
            resp = svc.answer(RAGRequest(question="test"))

        assert call_count["n"] == 3
        assert resp.answer == "Retry succeeded."

    def test_raises_rag_error_after_all_retries(self):
        svc = _make_rag_service()
        svc._client.models.generate_content.side_effect = RuntimeError("Persistent failure")

        with patch("milestone3.rag_service.time.sleep"):
            with pytest.raises(RAGError, match="RAG LLM call failed"):
                svc.answer(RAGRequest(question="test"))
