"""
Tests for milestone3.semantic_search_service — end-to-end search pipeline.
Mocks EmbeddingService and VectorStoreService — no real API or DB calls.
"""

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from milestone3.schemas import SearchRequest, SearchResult
from milestone3.semantic_search_service import SemanticSearchService


def _make_mock_embedder(embedding=None):
    if embedding is None:
        embedding = [0.1] * 768
    svc = MagicMock()
    svc.embed_text.return_value = embedding
    return svc


def _make_mock_vstore(raw_results=None, count=5):
    vstore = MagicMock()
    vstore.count.return_value = count
    if raw_results is None:
        raw_results = [
            {
                "id": "mtg1__transcript__0",
                "document": "We discussed the Q3 budget.",
                "metadata": {"meeting_id": "mtg1", "source_type": "transcript", "content_hash": "h1", "chunk_index": 0},
                "distance": 0.1,
                "relevance_score": 0.9,
            },
            {
                "id": "mtg2__decision__0",
                "document": "Decided to launch in December.",
                "metadata": {"meeting_id": "mtg2", "source_type": "decision", "content_hash": "h2", "chunk_index": 0},
                "distance": 0.3,
                "relevance_score": 0.7,
            },
        ]
    vstore.similarity_search.return_value = raw_results
    return vstore


class TestSemanticSearchService:
    def _make_svc(self, raw_results=None, count=5):
        return SemanticSearchService(
            embedding_svc=_make_mock_embedder(),
            vector_store=_make_mock_vstore(raw_results=raw_results, count=count),
        )

    def test_returns_list_of_search_results(self):
        svc = self._make_svc()
        results = svc.search(SearchRequest(query="budget discussion"))
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_results_have_required_fields(self):
        svc = self._make_svc()
        results = svc.search(SearchRequest(query="budget discussion"))
        assert len(results) > 0
        for r in results:
            assert r.meeting_id is not None
            assert r.source_type is not None
            assert r.content is not None
            assert 0.0 <= r.relevance_score <= 1.0
            assert r.search_latency_ms >= 0.0

    def test_sorted_by_relevance_descending(self):
        svc = self._make_svc()
        results = svc.search(SearchRequest(query="launch decision"))
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_vector_store_returns_empty_list(self):
        svc = SemanticSearchService(
            embedding_svc=_make_mock_embedder(),
            vector_store=_make_mock_vstore(raw_results=[], count=0),
        )
        results = svc.search(SearchRequest(query="anything"))
        assert results == []

    def test_source_type_filter_passed_to_vstore(self):
        mock_vstore = _make_mock_vstore()
        svc = SemanticSearchService(
            embedding_svc=_make_mock_embedder(),
            vector_store=mock_vstore,
        )
        svc.search(SearchRequest(query="decisions", source_type="decision"))
        call_kwargs = mock_vstore.similarity_search.call_args[1]
        assert call_kwargs.get("source_type_filter") == "decision"

    def test_top_k_passed_to_vstore(self):
        mock_vstore = _make_mock_vstore()
        svc = SemanticSearchService(
            embedding_svc=_make_mock_embedder(),
            vector_store=mock_vstore,
        )
        svc.search(SearchRequest(query="test", top_k=10))
        call_kwargs = mock_vstore.similarity_search.call_args[1]
        assert call_kwargs.get("top_k") == 10

    def test_search_latency_ms_is_measured(self):
        svc = self._make_svc()
        results = svc.search(SearchRequest(query="budget"))
        # All results should have the same latency (measured once for the whole call)
        assert all(r.search_latency_ms > 0 for r in results)

    def test_embed_text_called_with_query(self):
        mock_embedder = _make_mock_embedder()
        svc = SemanticSearchService(
            embedding_svc=mock_embedder,
            vector_store=_make_mock_vstore(),
        )
        svc.search(SearchRequest(query="specific query text"))
        mock_embedder.embed_text.assert_called_once_with("specific query text")
