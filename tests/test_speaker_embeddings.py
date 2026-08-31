"""
tests/test_speaker_embeddings.py
Pytest test suite for speaker embedding extraction, cosine similarity, and clustering.
"""

from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from speaker_embeddings import (
    SpeakerEmbeddingExtractor,
    compute_similarity,
    cluster_embeddings,
    format_speaker_id,
)


class TestCosineSimilarity:
    def test_identical_vectors_similarity_is_one(self):
        v = np.random.randn(256).astype(np.float32)
        v /= np.linalg.norm(v)
        sim = compute_similarity(v, v)
        assert pytest.approx(sim, abs=1e-5) == 1.0

    def test_orthogonal_vectors_similarity_is_zero(self):
        v1 = np.zeros(256, dtype=np.float32)
        v2 = np.zeros(256, dtype=np.float32)
        v1[0] = 1.0
        v2[1] = 1.0
        sim = compute_similarity(v1, v2)
        assert pytest.approx(sim, abs=1e-5) == 0.0

    def test_opposite_vectors_similarity_is_negative_one(self):
        v = np.random.randn(256).astype(np.float32)
        v /= np.linalg.norm(v)
        sim = compute_similarity(v, -v)
        assert pytest.approx(sim, abs=1e-5) == -1.0

    def test_zero_vector_similarity_is_zero(self):
        v1 = np.zeros(256, dtype=np.float32)
        v2 = np.random.randn(256).astype(np.float32)
        assert compute_similarity(v1, v2) == 0.0


class TestEmbeddingExtractor:
    def test_empty_audio_returns_zeros(self):
        ext = SpeakerEmbeddingExtractor()
        emb = ext.embed_utterance(np.array([], dtype=np.float32))
        assert len(emb) == 256
        assert np.all(emb == 0)

    def test_short_audio_returns_zeros(self):
        ext = SpeakerEmbeddingExtractor()
        # Less than 100ms (1600 samples at 16kHz)
        tiny_pcm = np.random.randn(500).astype(np.float32)
        emb = ext.embed_utterance(tiny_pcm, sample_rate=16000)
        assert len(emb) == 256
        assert np.all(emb == 0)

    def test_extractor_with_mocked_model(self):
        ext = SpeakerEmbeddingExtractor()
        mock_encoder = MagicMock()
        mock_vec = np.ones(256, dtype=np.float32)
        mock_encoder.embed_utterance.return_value = mock_vec
        ext._encoder = mock_encoder

        pcm = np.random.randn(16000).astype(np.float32)
        emb = ext.embed_utterance(pcm)
        assert len(emb) == 256
        assert pytest.approx(np.linalg.norm(emb), abs=1e-5) == 1.0


class TestClustering:
    def test_empty_embeddings_returns_empty_list(self):
        assert cluster_embeddings([]) == []

    def test_single_embedding_returns_single_cluster(self):
        emb = np.random.randn(256).astype(np.float32)
        labels = cluster_embeddings([emb])
        assert labels == [0]

    def test_one_speaker_multiple_segments(self):
        """All segments belonging to the same voice should get cluster ID 0."""
        base_vec = np.random.randn(256).astype(np.float32)
        base_vec /= np.linalg.norm(base_vec)
        # 5 slightly perturbed versions of same speaker
        embs = [base_vec + np.random.randn(256) * 0.02 for _ in range(5)]
        labels = cluster_embeddings(embs, threshold=0.70)
        assert labels == [0, 0, 0, 0, 0]

    def test_two_distinct_speakers(self):
        """Two distinct orthogonal voices should be assigned cluster 0 and 1."""
        spk1 = np.zeros(256, dtype=np.float32); spk1[0:100] = 1.0; spk1 /= np.linalg.norm(spk1)
        spk2 = np.zeros(256, dtype=np.float32); spk2[100:200] = 1.0; spk2 /= np.linalg.norm(spk2)

        embs = [spk1, spk1, spk2, spk2, spk1]
        labels = cluster_embeddings(embs, threshold=0.70)
        assert labels == [0, 0, 1, 1, 0]

    def test_four_distinct_speakers(self):
        """Four distinct orthogonal speakers must get cluster IDs 0, 1, 2, 3 in order."""
        s1 = np.zeros(256, dtype=np.float32); s1[0:50] = 1.0; s1 /= np.linalg.norm(s1)
        s2 = np.zeros(256, dtype=np.float32); s2[50:100] = 1.0; s2 /= np.linalg.norm(s2)
        s3 = np.zeros(256, dtype=np.float32); s3[100:150] = 1.0; s3 /= np.linalg.norm(s3)
        s4 = np.zeros(256, dtype=np.float32); s4[150:200] = 1.0; s4 /= np.linalg.norm(s4)

        embs = [s1, s2, s3, s4]
        labels = cluster_embeddings(embs, threshold=0.70)
        assert labels == [0, 1, 2, 3]

    def test_same_speaker_recurring_later(self):
        """A B A C B A pattern must preserve speaker consistency."""
        s1 = np.zeros(256, dtype=np.float32); s1[0:50] = 1.0; s1 /= np.linalg.norm(s1)
        s2 = np.zeros(256, dtype=np.float32); s2[50:100] = 1.0; s2 /= np.linalg.norm(s2)
        s3 = np.zeros(256, dtype=np.float32); s3[100:150] = 1.0; s3 /= np.linalg.norm(s3)

        embs = [s1, s2, s1, s3, s2, s1]
        labels = cluster_embeddings(embs, threshold=0.70)
        assert labels == [0, 1, 0, 2, 1, 0]

    def test_explicit_speaker_count_constraint(self):
        """When num_speakers=2 is passed, exactly 2 clusters must be created."""
        s1 = np.zeros(256, dtype=np.float32); s1[0:50] = 1.0; s1 /= np.linalg.norm(s1)
        s2 = np.zeros(256, dtype=np.float32); s2[50:100] = 1.0; s2 /= np.linalg.norm(s2)

        embs = [s1, s1, s2, s2]
        labels = cluster_embeddings(embs, num_speakers=2)
        assert len(set(labels)) == 2

    def test_format_speaker_id(self):
        assert format_speaker_id(0) == "Speaker 1"
        assert format_speaker_id(1) == "Speaker 2"
        assert format_speaker_id(3) == "Speaker 4"
