"""
speaker_embeddings.py
Speaker Embedding Extraction & Clustering Engine

Uses a pretrained deep speaker embedding model (VoiceEncoder d-vectors, 256-dimensional)
to extract numerical voice embeddings and cluster them into consistent anonymous speaker IDs:
'Speaker 1', 'Speaker 2', etc.

100% offline, local, privacy-preserving, zero external API or Hugging Face token required.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ── Module-level Lazy Model Cache ─────────────────────────────────────────────
_VOICE_ENCODER = None


def get_voice_encoder(device: str | None = None) -> Any:
    """Lazy-load and cache the pretrained VoiceEncoder (ResNet d-vector model)."""
    global _VOICE_ENCODER
    if _VOICE_ENCODER is None:
        try:
            import torch
            from resemblyzer import VoiceEncoder

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading pretrained VoiceEncoder on %s...", device)
            _VOICE_ENCODER = VoiceEncoder(device=device)
        except Exception as exc:
            logger.warning("Could not initialize resemblyzer VoiceEncoder: %s", exc)
            raise RuntimeError(
                f"Speaker embedding model failed to initialize: {exc}"
            ) from exc
    return _VOICE_ENCODER


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D speaker embedding vectors in [-1, 1]."""
    e1 = np.asarray(emb1, dtype=np.float32)
    e2 = np.asarray(emb2, dtype=np.float32)
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(e1, e2) / (norm1 * norm2))


class SpeakerEmbeddingExtractor:
    """Extracts numerical 256-dimensional speaker embeddings from audio waveforms."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self._encoder = None

    @property
    def encoder(self) -> Any:
        if self._encoder is None:
            self._encoder = get_voice_encoder(self.device)
        return self._encoder

    def embed_utterance(self, wav_pcm: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract a 256-d unit-normalized speaker embedding vector from a 1D float32 numpy audio array.
        Audio should be 16kHz mono.
        """
        if wav_pcm is None or len(wav_pcm) == 0:
            return np.zeros(256, dtype=np.float32)

        wav = np.asarray(wav_pcm, dtype=np.float32)
        # Rescale if audio is integer PCM
        if np.max(np.abs(wav)) > 1.0:
            wav = wav / 32768.0

        if len(wav) < int(sample_rate * 0.1):  # under 100ms
            return np.zeros(256, dtype=np.float32)

        try:
            emb = self.encoder.embed_utterance(wav)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb.astype(np.float32)
        except Exception as exc:
            logger.warning("embed_utterance error: %s", exc)
            return np.zeros(256, dtype=np.float32)


def cluster_embeddings(
    embeddings: Sequence[np.ndarray],
    threshold: float = 0.70,
    num_speakers: int | None = None,
    min_speakers: int = 1,
    max_speakers: int = 10,
) -> list[int]:
    """
    Cluster a sequence of speaker embedding vectors into anonymous cluster IDs [0, 1, 2...].

    Strategy:
    - If num_speakers is specified (>=1): uses Agglomerative Clustering with cosine distance.
    - If num_speakers is None/Auto: uses distance threshold clustering with cosine metric.
    - Preserves temporal cluster discovery order (first discovered voice is cluster 0, second is 1, etc.).

    Returns:
        list of integer cluster IDs corresponding to each input embedding.
    """
    if not embeddings:
        return []

    embs = [np.asarray(e, dtype=np.float32) for e in embeddings]
    n_samples = len(embs)
    if n_samples == 1:
        return [0]

    # Normalize embeddings to unit vectors
    X = np.stack(embs)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    # If all embeddings are near-zero
    if np.all(np.abs(X) < 1e-6):
        return [0] * n_samples

    # If exact speaker count is requested
    if num_speakers is not None and num_speakers >= 1:
        k = min(num_speakers, n_samples)
        if k == 1:
            return [0] * n_samples
        try:
            from sklearn.cluster import AgglomerativeClustering

            clustering = AgglomerativeClustering(
                n_clusters=k,
                metric="cosine",
                linkage="average",
            )
            raw_labels = clustering.fit_predict(X).tolist()
            return _remap_labels_chronologically(raw_labels)
        except Exception as exc:
            logger.warning("AgglomerativeClustering failed: %s, falling back to sequential", exc)

    # Automatic speaker count discovery via distance threshold (distance = 1 - similarity)
    distance_threshold = 1.0 - threshold
    try:
        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        raw_labels = clustering.fit_predict(X).tolist()
        return _remap_labels_chronologically(raw_labels)
    except Exception:
        # Fallback to online greedy sequential clustering
        return _sequential_cluster(X, threshold=threshold)


def _sequential_cluster(X: np.ndarray, threshold: float = 0.70) -> list[int]:
    """Online greedy clustering comparing each embedding to existing speaker centroids."""
    centroids: list[np.ndarray] = []
    labels: list[int] = []

    for vec in X:
        if not centroids:
            centroids.append(vec.copy())
            labels.append(0)
            continue

        # Compute cosine similarity with each centroid
        sims = [compute_similarity(vec, c) for c in centroids]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= threshold:
            labels.append(best_idx)
            # Update centroid with exponential moving average
            centroids[best_idx] = 0.8 * centroids[best_idx] + 0.2 * vec
            norm = np.linalg.norm(centroids[best_idx])
            if norm > 0:
                centroids[best_idx] /= norm
        else:
            new_id = len(centroids)
            centroids.append(vec.copy())
            labels.append(new_id)

    return labels


def _remap_labels_chronologically(raw_labels: list[int]) -> list[int]:
    """
    Ensure cluster IDs appear in order of appearance in the audio timeline:
    First speaker encountered is 0 ('Speaker 1'), next distinct speaker is 1 ('Speaker 2'), etc.
    """
    mapping: dict[int, int] = {}
    next_id = 0
    ordered: list[int] = []
    for lbl in raw_labels:
        if lbl not in mapping:
            mapping[lbl] = next_id
            next_id += 1
        ordered.append(mapping[lbl])
    return ordered


def format_speaker_id(cluster_id: int) -> str:
    """Format integer cluster ID 0, 1, 2 into user-facing anonymous label 'Speaker 1', 'Speaker 2', ..."""
    return f"Speaker {cluster_id + 1}"
