"""
Tests for milestone3.embedding_service — Gemini embedding API wrapper.
All tests mock the Gemini API client — no real API calls are made.
"""

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from milestone3.embedding_service import EmbeddingError, EmbeddingService


def _make_mock_embedding(dim: int = 768) -> List[float]:
    """Returns a fake embedding vector of the given dimension."""
    return [0.1] * dim


def _make_mock_client(embedding: List[float] = None):
    """
    Creates a mock genai.Client where embed_content returns the given embedding.
    """
    if embedding is None:
        embedding = _make_mock_embedding()

    mock_emb_obj = MagicMock()
    mock_emb_obj.values = embedding

    mock_response = MagicMock()
    mock_response.embeddings = [mock_emb_obj]

    mock_models = MagicMock()
    mock_models.embed_content.return_value = mock_response

    mock_client = MagicMock()
    mock_client.models = mock_models

    return mock_client


class TestEmbeddingServiceInit:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with patch("milestone3.embedding_service.os.environ.get", return_value=None):
            with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
                EmbeddingService(api_key=None)

    def test_initializes_with_explicit_key(self):
        with patch("google.genai.Client") as mock_cls:
            svc = EmbeddingService(api_key="fake-key")
            mock_cls.assert_called_once_with(api_key="fake-key")


class TestEmbedText:
    def _make_service(self, embedding=None):
        svc = EmbeddingService.__new__(EmbeddingService)
        svc._model = "test-model"
        svc._client = _make_mock_client(embedding)
        return svc

    def test_returns_list_of_floats(self):
        svc = self._make_service([0.5, 0.3, 0.2])
        result = svc.embed_text("hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_correct_embedding_values(self):
        embedding = [0.1, 0.2, 0.3]
        svc = self._make_service(embedding)
        result = svc.embed_text("test text")
        assert result == pytest.approx(embedding)

    def test_empty_text_raises_value_error(self):
        svc = self._make_service()
        with pytest.raises(ValueError, match="empty"):
            svc.embed_text("")

    def test_whitespace_only_raises_value_error(self):
        svc = self._make_service()
        with pytest.raises(ValueError):
            svc.embed_text("   \n  ")

    def test_retry_on_api_failure(self):
        """Service should retry on failure and succeed on third attempt."""
        call_count = {"n": 0}
        mock_emb_obj = MagicMock()
        mock_emb_obj.values = [0.5, 0.5]

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("Transient API error")
            resp = MagicMock()
            resp.embeddings = [mock_emb_obj]
            return resp

        svc = EmbeddingService.__new__(EmbeddingService)
        svc._model = "test-model"
        svc._client = MagicMock()
        svc._client.models.embed_content.side_effect = side_effect

        with patch("milestone3.embedding_service.time.sleep"):
            result = svc.embed_text("test text")

        assert call_count["n"] == 3
        assert result == pytest.approx([0.5, 0.5])

    def test_raises_after_all_retries_exhausted(self):
        svc = EmbeddingService.__new__(EmbeddingService)
        svc._model = "test-model"
        svc._client = MagicMock()
        svc._client.models.embed_content.side_effect = RuntimeError("Persistent failure")

        with patch("milestone3.embedding_service.time.sleep"):
            with pytest.raises(EmbeddingError, match="Embedding API failed"):
                svc.embed_text("test text")


class TestEmbedBatch:
    def _make_service(self, per_call_embedding=None):
        """Makes a service where each embed_content call returns embeddings for all texts in the call."""
        if per_call_embedding is None:
            per_call_embedding = [0.1, 0.2, 0.3]

        def embed_content(model, contents):
            n = len(contents) if isinstance(contents, list) else 1
            embeddings = []
            for _ in range(n):
                e = MagicMock()
                e.values = per_call_embedding
                embeddings.append(e)
            resp = MagicMock()
            resp.embeddings = embeddings
            return resp

        svc = EmbeddingService.__new__(EmbeddingService)
        svc._model = "test-model"
        svc._client = MagicMock()
        svc._client.models.embed_content.side_effect = embed_content
        return svc

    def test_returns_same_length_as_input(self):
        svc = self._make_service()
        texts = ["text one", "text two", "text three"]
        results = svc.embed_batch(texts)
        assert len(results) == len(texts)

    def test_all_results_are_embedding_vectors(self):
        svc = self._make_service([0.5, 0.6])
        results = svc.embed_batch(["a", "b", "c"])
        for r in results:
            assert r is not None
            assert isinstance(r, list)
            assert all(isinstance(x, float) for x in r)

    def test_skip_hashes_causes_none_result(self):
        import hashlib
        text = "already indexed text"
        h = hashlib.sha256(text.encode()).hexdigest()

        svc = self._make_service()
        results = svc.embed_batch([text], skip_hashes={h})
        # Skipped → position is None
        assert results[0] is None

    def test_empty_text_in_batch_is_none(self):
        svc = self._make_service()
        results = svc.embed_batch(["valid text", "", "another text"])
        # Empty string at index 1 is skipped → None
        assert results[1] is None
        assert results[0] is not None
        assert results[2] is not None
