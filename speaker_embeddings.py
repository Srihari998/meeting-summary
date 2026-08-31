"""
speaker_embeddings.py
Speaker Embedding Extraction & Clustering Engine

Uses a pretrained deep speaker embedding model (VoiceEncoder d-vectors, 256-dimensional)
to extract numerical voice embeddings and cluster them into consistent anonymous speaker IDs:
'Speaker 1', 'Speaker 2', etc.

Includes:
- Cosine similarity computation
- Agglomerative clustering with cosine metric
- Post-clustering centroid merging for recurring voices
- Satellite cluster pruning for short interjections, laughter, and noise bursts
- Strict chronological speaker ID assignment

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
    threshold: float = 0.80,
    num_speakers: int | None = None,
    min_speakers: int = 1,
    max_speakers: int = 10,
    merge_similarity: float = 0.87,
    min_cluster_samples: int = 4,
) -> list[int]:
    """
    Cluster a sequence of speaker embedding vectors into anonymous cluster IDs [0, 1, 2...].

    Features:
    1. Agglomerative Clustering with cosine distance.
    2. Centroid Merging: merges clusters whose centroids have similarity >= merge_similarity.
    3. Satellite Cluster Pruning: reassigns isolated single-fragment/noise clusters (< min_cluster_samples)
       to the nearest dominant cluster centroid.
    4. Chronological Mapping: speaker IDs appear in order of first speech.

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

    # ── Automatic Speaker Discovery (Auto Mode) ──────────────────────────────
    distance_threshold = max(0.05, 1.0 - threshold)
    try:
        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        raw_labels = clustering.fit_predict(X).tolist()
    except Exception:
        raw_labels = _sequential_cluster(X, threshold=threshold)

    # ── Stage 2: Iterative Centroid Merging for Recurring Voices ─────────────
    raw_labels = _merge_similar_centroids(X, raw_labels, merge_similarity=merge_similarity)

    # ── Stage 3: Satellite Cluster Pruning for Short Interjections/Noise ─────
    final_labels = _prune_satellite_clusters(
        X, raw_labels, min_cluster_samples=min_cluster_samples
    )

    # ── Stage 4: Chronological Relabeling ────────────────────────────────────
    return _remap_labels_chronologically(final_labels)


def _merge_similar_centroids(
    X: np.ndarray, labels: list[int], merge_similarity: float = 0.87
) -> list[int]:
    """Iteratively merge cluster centroids whose cosine similarity exceeds merge_similarity."""
    labels_arr = np.array(labels)
    unique_lbls = sorted(list(set(labels)))
    if len(unique_lbls) <= 1:
        return labels

    centroids: dict[int, np.ndarray] = {}
    for lbl in unique_lbls:
        mask = labels_arr == lbl
        c = np.mean(X[mask], axis=0)
        norm = np.linalg.norm(c)
        centroids[lbl] = c / norm if norm > 0 else c

    merged_map = {lbl: lbl for lbl in unique_lbls}
    changed = True
    while changed:
        changed = False
        active = sorted(list(set(merged_map.values())))
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                l1 = active[i]
                l2 = active[j]
                sim = compute_similarity(centroids[l1], centroids[l2])
                if sim >= merge_similarity:
                    for k, v in merged_map.items():
                        if v == l2:
                            merged_map[k] = l1
                    # Update merged centroid
                    mask = np.array([merged_map[orig] == l1 for orig in labels])
                    c = np.mean(X[mask], axis=0)
                    norm = np.linalg.norm(c)
                    centroids[l1] = c / norm if norm > 0 else c
                    changed = True
                    break
            if changed:
                break

    return [merged_map[l] for l in labels]


def _prune_satellite_clusters(
    X: np.ndarray, labels: list[int], min_cluster_samples: int = 4
) -> list[int]:
    """
    Reassign tiny satellite clusters (e.g. laughter bursts or single interjections < min_cluster_samples)
    to the nearest dominant speaker centroid.
    """
    labels_arr = np.array(labels)
    unique_lbls = set(labels)
    if len(unique_lbls) <= 1:
        return labels

    counts = {lbl: labels.count(lbl) for lbl in unique_lbls}
    major_lbls = [lbl for lbl, cnt in counts.items() if cnt >= min_cluster_samples]

    if not major_lbls:
        # If no cluster meets the threshold, keep the largest cluster as major
        major_lbls = [max(counts, key=counts.get)]

    # Compute centroids for major clusters
    major_centroids: dict[int, np.ndarray] = {}
    for ml in major_lbls:
        mask = labels_arr == ml
        c = np.mean(X[mask], axis=0)
        norm = np.linalg.norm(c)
        major_centroids[ml] = c / norm if norm > 0 else c

    pruned_labels: list[int] = []
    for lbl, vec in zip(labels, X):
        if lbl in major_lbls:
            pruned_labels.append(lbl)
        else:
            # Reassign to closest major centroid
            best_ml = max(
                major_lbls, key=lambda ml: compute_similarity(vec, major_centroids[ml])
            )
            pruned_labels.append(best_ml)

    return pruned_labels


def _sequential_cluster(X: np.ndarray, threshold: float = 0.80) -> list[int]:
    """Online greedy clustering comparing each embedding to existing speaker centroids."""
    centroids: list[np.ndarray] = []
    labels: list[int] = []

    for vec in X:
        if not centroids:
            centroids.append(vec.copy())
            labels.append(0)
            continue

        sims = [compute_similarity(vec, c) for c in centroids]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= threshold:
            labels.append(best_idx)
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
