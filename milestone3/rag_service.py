"""
RAG Service — Milestone 3
Grounded LLM question answering over retrieved meeting context.

Flow:
  1. Semantic search (SemanticSearchService) → top-k meeting chunks
  2. Build context string with traceability tags [meeting_id | source_type]
  3. Call Gemini LLM with anti-hallucination RAG prompt
  4. Return RAGResponse with answer, meeting_ids, sources, latency

The RAG prompt explicitly prohibits fabrication and requires citing meeting IDs.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from milestone3.schemas import RAGRequest, RAGResponse, SearchRequest, SearchResult
    from milestone3.semantic_search_service import SemanticSearchService
except ImportError:
    from schemas import RAGRequest, RAGResponse, SearchRequest, SearchResult  # type: ignore
    from semantic_search_service import SemanticSearchService  # type: ignore

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Retry config (mirrors Milestone 2 llm_service pattern)
_MAX_RETRIES = 4
_RETRY_DELAYS = (1.5, 3.0, 5.0, 8.0)

# Maximum characters per context chunk shown in the RAG prompt
_MAX_CONTEXT_CHARS_PER_CHUNK = 800


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


class RAGError(Exception):
    """Raised when RAG processing or LLM generation fails."""
    pass


class RAGService:
    """
    RAG Question-Answering Service.
    Retrieves semantically relevant meeting excerpts and generates grounded answers.
    Supports automatic multi-key failover across all configured Gemini API keys.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        search_svc: Optional[SemanticSearchService] = None,
    ) -> None:
        self._model = model or os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        self._search_svc = search_svc or SemanticSearchService()
        self._api_keys = [api_key] if api_key else get_all_api_keys()

        if not self._api_keys:
            raise RAGError(
                "GEMINI_API_KEY environment variable is not set. "
                "Configure it before using RAGService."
            )
        # Verify the google-genai package is available at construction time.
        try:
            from google import genai as _genai  # noqa: F401
        except ImportError as exc:
            raise RAGError(
                "google-genai package is required. Run: pip install google-genai"
            ) from exc
        # NOTE: self._client is intentionally NOT set here.
        # Production calls use the multi-key failover loop in _call_llm.
        # Unit tests inject _client via RAGService.__new__() + svc._client = mock.

        # Load and cache the RAG prompt template
        prompt_path = _PROMPTS_DIR / "rag_prompt.txt"
        if not prompt_path.exists():
            raise RAGError(f"RAG prompt template not found at: {prompt_path}")
        self._prompt_template = prompt_path.read_text(encoding="utf-8")

    def _build_context(self, sources: List[SearchResult]) -> str:
        """
        Builds a formatted context string from retrieved search results.
        Each chunk is labelled with its meeting_id and source_type for citation.
        """
        lines: List[str] = []
        for i, result in enumerate(sources, start=1):
            # Truncate very long chunks to keep prompt manageable
            content = result.content
            if len(content) > _MAX_CONTEXT_CHARS_PER_CHUNK:
                content = content[:_MAX_CONTEXT_CHARS_PER_CHUNK] + "..."

            lines.append(
                f"[Excerpt {i} | Meeting: {result.meeting_id} | Type: {result.source_type}]\n"
                f"{content}"
            )
        return "\n\n---\n\n".join(lines)

    def _call_llm(self, prompt: str) -> str:
        """Calls Gemini with automatic multi-key failover and retry logic."""
        import time as _time
        from google import genai
        from google.genai import types

        config = types.GenerateContentConfig(
            max_output_tokens=2048,
            temperature=0.1,  # Low temperature for factual grounded answers
        )

        # If a client is already set / injected (e.g. In unit tests), use it directly
        if hasattr(self, "_client") and self._client is not None:
            last_exc: Optional[Exception] = None
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    response = self._client.models.generate_content(
                        model=self._model,
                        contents=prompt,
                        config=config,
                    )
                    if not response.text:
                        raise RAGError("Gemini returned an empty response.")
                    return response.text
                except Exception as exc:
                    last_exc = exc
                    if attempt < _MAX_RETRIES:
                        delay = _RETRY_DELAYS[attempt - 1]
                        logger.warning(
                            f"[RAG Retry {attempt}/{_MAX_RETRIES}] Error: {exc}. Retrying in {delay:.1f}s..."
                        )
                        _time.sleep(delay)
            raise RAGError(
                f"RAG LLM call failed after {_MAX_RETRIES} attempts. Last error: {last_exc}"
            ) from last_exc

        api_keys = getattr(self, "_api_keys", None) or get_all_api_keys()
        last_exc = None

        for key_idx, key in enumerate(api_keys, start=1):
            try:
                active_client = genai.Client(api_key=key)
                for attempt in range(1, _MAX_RETRIES + 1):
                    try:
                        response = active_client.models.generate_content(
                            model=self._model,
                            contents=prompt,
                            config=config,
                        )
                        if not response.text:
                            raise RAGError("Gemini returned an empty response.")
                        return response.text
                    except Exception as exc:
                        last_exc = exc
                        if attempt < _MAX_RETRIES:
                            delay = _RETRY_DELAYS[attempt - 1]
                            logger.warning(
                                f"[RAG Retry {attempt}/{_MAX_RETRIES} on Key {key_idx}/{len(api_keys)}] "
                                f"Error: {exc}. Retrying in {delay:.1f}s..."
                            )
                            _time.sleep(delay)
                        else:
                            logger.warning(
                                f"[RAG Key Failover] Key {key_idx}/{len(api_keys)} exhausted. "
                                f"Failing over to next available API key..."
                            )
            except Exception as key_err:
                last_exc = key_err
                continue

        raise RAGError(
            f"RAG LLM call failed across all {len(api_keys)} API keys. Last error: {last_exc}"
        ) from last_exc

    def answer(self, request: RAGRequest) -> RAGResponse:
        """
        Answers a natural-language question using retrieved meeting context.

        Args:
            request: RAGRequest with question text and top_k.

        Returns:
            RAGResponse with grounded answer, meeting_ids, sources, and latency_ms.

        Raises:
            RAGError: If the LLM call fails after all retries.
        """
        t_start = time.perf_counter()

        # 1. Retrieve relevant context
        search_request = SearchRequest(query=request.question, top_k=request.top_k)
        sources: List[SearchResult] = self._search_svc.search(search_request)

        # 2. If no context found, return a grounded "no information" response
        if not sources:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return RAGResponse(
                answer="I could not find relevant information in the stored meetings.",
                meeting_ids=[],
                sources=[],
                latency_ms=elapsed_ms,
            )

        # 3. Build context string
        context = self._build_context(sources)

        # 4. Format RAG prompt
        prompt = self._prompt_template.replace("{context}", context).replace(
            "{question}", request.question
        )

        # 5. Call LLM
        answer_text = self._call_llm(prompt)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # 6. Extract unique meeting IDs from sources (maintain relevance order)
        seen: set = set()
        meeting_ids: List[str] = []
        for src in sources:
            if src.meeting_id not in seen:
                seen.add(src.meeting_id)
                meeting_ids.append(src.meeting_id)

        logger.info(
            "[RAGService] Question answered in %.1f ms using %d sources from %d meeting(s).",
            elapsed_ms, len(sources), len(meeting_ids),
        )

        return RAGResponse(
            answer=answer_text.strip(),
            meeting_ids=meeting_ids,
            sources=sources,
            latency_ms=elapsed_ms,
        )
