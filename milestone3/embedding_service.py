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

# Gemini embedding model — supports text embedding
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# Maximum texts per batch call (Gemini API limit)
_MAX_BATCH_SIZE = 100

# Retry config (mirrors milestone2 llm_service pattern)
_MAX_RETRIES = 3
_RETRY_DELAYS = (1.0, 2.0, 4.0)


def get_all_api_keys() -> List[str]:
    """
    Returns an ordered list of all configured Gemini API keys for failover resilience.
    """
    keys: List[str] = []

    # 1. Comma-separated list
    raw_keys = os.environ.get("GEMINI_API_KEYS", "")
    if raw_keys:
        for k in raw_keys.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    # 2. Individual numbered keys
    for var_name in ["GEMINI_API_KEY", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]:
        k = (os.environ.get(var_name) or "").strip()
        if k and k not in keys:
            keys.append(k)

    return keys


class EmbeddingError(Exception):
    """Raised when an embedding API call fails or inputs are invalid."""
    pass


class EmbeddingService:
    """
    Service responsible for producing dense vector representations of text.

    Supports automatic multi-key failover across all configured Gemini API keys.

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
        self._api_keys = [api_key] if api_key else get_all_api_keys()
        if not self._api_keys:
            raise EmbeddingError(
                "GEMINI_API_KEY environment variable is not set. "
                "Configure it before using EmbeddingService."
            )
        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_keys[0])
        except ImportError as exc:
            raise EmbeddingError(
                "google-genai package is required. Run: pip install google-genai"
            ) from exc

    def embed_text(self, text: str) -> List[float]:
        """
        Generates a dense embedding vector for a single text string with multi-key failover.

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

        # If a client is already set / injected (e.g. In unit tests), use it directly
        if hasattr(self, "_client") and self._client is not None:
            last_exc: Optional[Exception] = None
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    response = self._client.models.embed_content(
                        model=self._model,
                        contents=text,
                    )
                    embeddings = response.embeddings
                    if not embeddings:
                        raise EmbeddingError("Gemini embedding API returned empty embeddings list.")
                    return list(embeddings[0].values)
                except Exception as exc:
                    last_exc = exc
                    if attempt < _MAX_RETRIES:
                        delay = _RETRY_DELAYS[attempt - 1]
                        logger.warning(
                            f"[Embedding Retry {attempt}/{_MAX_RETRIES}] Error: {exc}. Retrying in {delay}s..."
                        )
                        time.sleep(delay)
            raise EmbeddingError(
                f"Embedding API failed after {_MAX_RETRIES} retries. Last error: {last_exc}"
            ) from last_exc

        from google import genai

        api_keys = getattr(self, "_api_keys", None) or get_all_api_keys()
        last_exc = None
        for key_idx, key in enumerate(api_keys, start=1):
            try:
                active_client = genai.Client(api_key=key)
                for attempt in range(1, _MAX_RETRIES + 1):
                    try:
                        response = active_client.models.embed_content(
                            model=self._model,
                            contents=text,
                        )
                        embeddings = response.embeddings
                        if not embeddings:
                            raise EmbeddingError("Gemini embedding API returned empty embeddings list.")
                        return list(embeddings[0].values)
                    except Exception as exc:
                        last_exc = exc
                        if attempt < _MAX_RETRIES:
                            delay = _RETRY_DELAYS[attempt - 1]
                            logger.warning(
                                f"[Embedding Retry {attempt}/{_MAX_RETRIES} on Key {key_idx}/{len(api_keys)}] "
                                f"Error: {exc}. Retrying in {delay}s..."
                            )
                            time.sleep(delay)
                        else:
                            logger.warning(
                                f"[Embedding Key Failover] Key {key_idx}/{len(api_keys)} exhausted. "
                                f"Failing over to next available key..."
                            )
            except Exception as key_err:
                last_exc = key_err
                continue

        raise EmbeddingError(
            f"Embedding API failed across all {len(api_keys)} API keys. Last error: {last_exc}"
        ) from last_exc

    def embed_batch(
        self,
        texts: List[str],
        skip_hashes: Optional[Set[str]] = None,
    ) -> List[Optional[List[float]]]:
        """
        Generates embeddings for a list of texts with automatic multi-key failover.

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
        from google import genai

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

        if not to_embed_indices:
            return results

        # If a client is already set / injected (e.g. In unit tests), use it directly
        if hasattr(self, "_client") and self._client is not None:
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
                            time.sleep(delay)

                if batch_embeddings is None:
                    raise EmbeddingError(
                        f"Batch embedding failed after {_MAX_RETRIES} attempts. Last error: {last_exc}"
                    ) from last_exc

                for local_idx, global_idx in enumerate(batch_indices):
                    if local_idx < len(batch_embeddings):
                        results[global_idx] = batch_embeddings[local_idx]

            return results

        # Process in batches across multi-key failover pool
        api_keys = getattr(self, "_api_keys", None) or get_all_api_keys()

        for batch_start in range(0, len(to_embed_indices), _MAX_BATCH_SIZE):
            batch_indices = to_embed_indices[batch_start: batch_start + _MAX_BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_indices]

            last_exc = None
            batch_embeddings = None

            for key_idx, key in enumerate(api_keys, start=1):
                try:
                    active_client = genai.Client(api_key=key)
                    for attempt in range(1, _MAX_RETRIES + 1):
                        try:
                            response = active_client.models.embed_content(
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
                                    f"[Embedding Batch Retry {attempt}/{_MAX_RETRIES} on Key {key_idx}/{len(api_keys)}] "
                                    f"Error: {exc}. Retrying in {delay}s..."
                                )
                                time.sleep(delay)
                    if batch_embeddings is not None:
                        break
                except Exception as key_err:
                    last_exc = key_err
                    continue

            if batch_embeddings is None:
                raise EmbeddingError(
                    f"Batch embedding failed across all {len(api_keys)} API keys. Last error: {last_exc}"
                ) from last_exc

            for local_idx, global_idx in enumerate(batch_indices):
                if local_idx < len(batch_embeddings):
                    results[global_idx] = batch_embeddings[local_idx]

        return results
