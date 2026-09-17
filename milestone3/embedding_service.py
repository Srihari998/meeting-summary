"""
Embedding Service — Milestone 3
Wraps the Gemini text-embedding API to produce dense vector representations.

Reuses the existing GEMINI_API_KEY environment variable — no new credentials needed.
Supports single-text embedding and batched embedding with idempotency via content hash checking.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Set

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Gemini embedding model — supports up to 2048 tokens input per text
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-exp-03-07")

# Maximum texts per batch call (Gemini API limit)
_MAX_BATCH_SIZE = 100

# Retry config (mirrors milestone2 llm_service pattern)
_MAX_RETRIES = 3
_RETRY_DELAYS = (1.0, 2.0, 4.0)


class EmbeddingError(Exception):
    """Raised when an embedding API call fails after all retries."""
    pass


class EmbeddingService:
    """
    Generates text embeddings using the Gemini API.

    Args:
        api_key:    Gemini API key. Reads GEMINI_API_KEY env var if not provided.
        model:      Embedding model name. Defaults to EMBEDDING_MODEL constant.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = EMBEDDING_MODEL,
    ) -> None:
        self._model = model
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise EmbeddingError(
                "GEMINI_API_KEY environment variable is not set. "
                "Configure it before using EmbeddingService."
            )
        try:
            from google import genai
            self._client = genai.Client(api_key=key)
        except ImportError as exc:
            raise EmbeddingError(
                "google-genai package is required. Run: pip install google-genai"
            ) from exc

    def embed_text(self, text: str) -> List[float]:
        """
        Generates a dense embedding vector for a single text string.

        Args:
            text: The text to embed. Must be non-empty.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            EmbeddingError: If the API call fails after all retries.
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.models.embed_content(
                    model=self._model,
                    contents=text,
                )
                # The SDK returns an EmbedContentResponse with .embeddings list
                embeddings = response.embeddings
                if not embeddings:
                    raise EmbeddingError("Gemini embedding API returned empty embeddings list.")
                return list(embeddings[0].values)
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_DELAYS[attempt - 1]
                    logger.warning(
                        "[EmbeddingService] Attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt, _MAX_RETRIES, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "[EmbeddingService] All %d attempts failed. Last error: %s",
                        _MAX_RETRIES, exc,
                    )

        raise EmbeddingError(
            f"Embedding API failed after {_MAX_RETRIES} attempts. Last error: {last_exc}"
        ) from last_exc

    def embed_batch(
        self,
        texts: List[str],
        skip_hashes: Optional[Set[str]] = None,
    ) -> List[Optional[List[float]]]:
        """
        Generates embeddings for a list of texts.

        Processes texts in batches of up to _MAX_BATCH_SIZE.
        If skip_hashes is provided, texts whose SHA-256 hash is in the set
        are skipped (idempotency — already indexed).

        Args:
            texts:       List of text strings to embed.
            skip_hashes: Optional set of content hashes to skip.

        Returns:
            List of embedding vectors (same length as texts).
            Positions corresponding to skipped texts contain None.
        """
        import hashlib

        results: List[Optional[List[float]]] = [None] * len(texts)
        to_embed_indices: List[int] = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            if skip_hashes is not None:
                h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if h in skip_hashes:
                    logger.debug("[EmbeddingService] Skipping already-indexed chunk (hash=%s)", h[:12])
                    continue
            to_embed_indices.append(i)

        # Process in batches
        for batch_start in range(0, len(to_embed_indices), _MAX_BATCH_SIZE):
            batch_indices = to_embed_indices[batch_start: batch_start + _MAX_BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_indices]

            last_exc: Optional[Exception] = None
            batch_embeddings: Optional[List[List[float]]] = None

            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    response = self._client.models.embed_content(
                        model=self._model,
                        contents=batch_texts,
                    )
                    batch_embeddings = [list(e.values) for e in response.embeddings]
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < _MAX_RETRIES:
                        delay = _RETRY_DELAYS[attempt - 1]
                        logger.warning(
                            "[EmbeddingService] Batch attempt %d/%d failed: %s. Retrying in %.1fs...",
                            attempt, _MAX_RETRIES, exc, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "[EmbeddingService] Batch failed after %d attempts: %s",
                            _MAX_RETRIES, exc,
                        )

            if batch_embeddings is None:
                raise EmbeddingError(
                    f"Batch embedding failed after {_MAX_RETRIES} attempts. Last error: {last_exc}"
                ) from last_exc

            for local_idx, global_idx in enumerate(batch_indices):
                if local_idx < len(batch_embeddings):
                    results[global_idx] = batch_embeddings[local_idx]

        return results
