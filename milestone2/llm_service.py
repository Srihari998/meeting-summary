import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

try:
    from schemas import JSONValidationError, MeetingIntelligence, validate_and_parse
except ImportError:
    from .schemas import JSONValidationError, MeetingIntelligence, validate_and_parse

# Configure logger
logger = logging.getLogger(__name__)

# Pinned Model & Generation Constants
# Note: gemini-2.5-flash was decommissioned. Using gemini-3.6-flash as default.
DEFAULT_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
DEFAULT_MAX_OUTPUT_TOKENS: int = 4096
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_CHUNK_MAX_TOKENS: int = 6000
MAX_INPUT_TOKENS: int = 150_000
MIN_INPUT_WORDS: int = 3

API_MAX_RETRIES: int = 3
API_RETRY_DELAYS: tuple = (1.0, 2.0, 4.0)
JSON_FIX_MAX_RETRIES: int = 2

PROMPTS_DIR: Path = Path(__file__).resolve().parent / "prompts"


# Custom Exceptions
class LLMExtractionError(Exception):
    """Base exception for all LLM service extraction failures."""
    pass


class InvalidInputError(LLMExtractionError):
    """Raised when the input transcript is empty, near-empty, or exceeds max length."""
    pass


class LLMAPIError(LLMExtractionError):
    """Raised when the Gemini API encounters persistent network, timeout, or server errors."""
    pass


class JSONFixError(LLMExtractionError):
    """Raised when output fails schema validation even after all JSON-fix retries are exhausted."""
    pass


def get_all_api_keys() -> List[str]:
    """
    Returns an ordered list of all configured Gemini API keys for failover resilience.
    Checks GEMINI_API_KEYS (comma-separated), and numbered keys (GEMINI_API_KEY, GEMINI_API_KEY_2, etc.).
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


def get_genai_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Initializes and returns a Google GenAI client instance.
    Reads first available key from get_all_api_keys() if not provided explicitly.
    """
    if api_key:
        return genai.Client(api_key=api_key)
    
    keys = get_all_api_keys()
    if not keys:
        raise LLMAPIError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please configure GEMINI_API_KEY before using the LLM service."
        )
    return genai.Client(api_key=keys[0])


def load_prompt(template_name: str, **kwargs) -> str:
    """
    Loads a prompt template from the prompts/ directory and replaces {key} placeholders.
    """
    prompt_path = PROMPTS_DIR / template_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Find expected template placeholders of the form {variable_name}
    placeholders = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", template))

    for p in placeholders:
        if p not in kwargs:
            raise ValueError(f"Missing required template parameter '{p}' for {template_name}")

    result = template
    for key, val in kwargs.items():
        result = result.replace(f"{{{key}}}", str(val))

    return result


def count_tokens(text: str, client: Optional[genai.Client] = None, model: str = DEFAULT_MODEL) -> int:
    """
    Counts tokens for the provided text. Uses ~1.3 tokens per word estimation.
    """
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def chunk_transcript(
    transcript: str,
    max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
) -> List[str]:
    """
    Splits a transcript into semantically coherent chunks respecting max_tokens.
    Prefers splitting at double newlines, then single newlines, then sentence boundaries.

    Args:
        transcript: Full raw meeting transcript.
        max_tokens: Maximum token budget per chunk.

    Returns:
        List of transcript text chunks.
    """
    if not transcript or not transcript.strip():
        return []

    if count_tokens(transcript) <= max_tokens:
        return [transcript.strip()]

    # Split at natural conversation turn / paragraph boundaries first
    if "\n\n" in transcript:
        units = [u.strip() for u in transcript.split("\n\n") if u.strip()]
    elif "\n" in transcript:
        units = [u.strip() for u in transcript.split("\n") if u.strip()]
    else:
        # Sentence boundary split
        units = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]

    chunks: List[str] = []
    current_chunk_parts: List[str] = []

    for unit in units:
        candidate_chunk = "\n\n".join(current_chunk_parts + [unit]) if current_chunk_parts else unit
        if count_tokens(candidate_chunk) <= max_tokens:
            current_chunk_parts.append(unit)
        else:
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
            current_chunk_parts = [unit]

    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    return chunks


def call_gemini_with_retry(
    prompt: str,
    client: Optional[genai.Client] = None,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """
    Executes a Gemini API call with automatic multi-key failover and exponential backoff retries.

    If an API key fails (due to quota, rate-limit, 503, etc.), it automatically switches
    to the next configured API key in the failover pool.

    Args:
        prompt: The input prompt string.
        client: Optional genai.Client instance.
        model: Model identifier.
        max_output_tokens: Explicit token budget for response.
        temperature: Sampling temperature.

    Returns:
        Generated text response from the model.

    Raises:
        LLMAPIError: If all keys and retries are exhausted.
    """
    config = types.GenerateContentConfig(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    last_exception: Optional[Exception] = None

    # If an explicit client was passed in, use it directly with retries
    if client is not None:
        for attempt in range(1, API_MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                if not response.text:
                    raise LLMAPIError("Gemini API returned an empty response body.")
                return response.text
            except Exception as exc:
                last_exception = exc
                if attempt < API_MAX_RETRIES:
                    delay = API_RETRY_DELAYS[min(attempt - 1, len(API_RETRY_DELAYS) - 1)]
                    time.sleep(delay)
        raise LLMAPIError(
            f"Gemini API request failed after {API_MAX_RETRIES} attempts. Last error: {last_exception}"
        ) from last_exception

    api_keys = get_all_api_keys()

    # Multi-Key Failover Loop
    for key_idx, key in enumerate(api_keys, start=1):
        try:
            active_client = genai.Client(api_key=key)
            for attempt in range(1, API_MAX_RETRIES + 1):
                try:
                    response = active_client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    if not response.text:
                        raise LLMAPIError("Gemini API returned an empty response body.")
                    return response.text

                except Exception as exc:
                    last_exception = exc
                    if attempt < API_MAX_RETRIES:
                        delay = API_RETRY_DELAYS[min(attempt - 1, len(API_RETRY_DELAYS) - 1)]
                        logger.warning(
                            f"[API Retry {attempt}/{API_MAX_RETRIES} on Key {key_idx}/{len(api_keys)}] "
                            f"Error: {exc}. Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.warning(
                            f"[API Key Failover] Key {key_idx}/{len(api_keys)} ({key[:12]}...) exhausted. "
                            f"Failing over to next available API key..."
                        )
        except Exception as key_err:
            last_exception = key_err
            continue

    raise LLMAPIError(
        f"Gemini API request failed across all {len(api_keys)} configured API keys. "
        f"Last error: {last_exception}"
    ) from last_exception


def summarize_chunk(
    chunk: str,
    client: Optional[genai.Client] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Summarizes an individual transcript chunk using Gemini.
    """
    prompt = load_prompt("chunk_summary_prompt.txt", chunk=chunk)
    return call_gemini_with_retry(prompt=prompt, client=client, model=model)


def merge_chunk_summaries(
    summaries: List[str],
    client: Optional[genai.Client] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Merges multiple chunk summaries into a single comprehensive transcript-equivalent text.
    """
    formatted_summaries = "\n\n---\n\n".join(
        f"[Section {i+1}]:\n{s}" for i, s in enumerate(summaries)
    )
    prompt = load_prompt("merge_chunks_prompt.txt", summaries=formatted_summaries)
    return call_gemini_with_retry(prompt=prompt, client=client, model=model)


def extract_meeting_intelligence(
    transcript: str,
    client: Optional[genai.Client] = None,
    model: str = DEFAULT_MODEL,
    max_chunk_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
) -> MeetingIntelligence:
    """
    Extracts structured meeting intelligence from a raw transcript string.

    Workflow:
    1. Validates input transcript (rejects empty/near-empty or oversized input).
    2. If long, applies map-reduce chunking, partial summarization, and merging.
    3. Calls LLM with meeting_intelligence_prompt.
    4. Validates output schema with validate_and_parse().
    5. If validation fails, retries up to 2 times with fix_json_prompt before raising.
    6. API failures are separately handled with 3 exponential backoff retries.

    Args:
        transcript: Raw transcript text string.
        client: Optional genai.Client (useful for dependency injection / mocking).
        model: Gemini model identifier (defaults to pinned DEFAULT_MODEL).
        max_chunk_tokens: Threshold token count triggering chunking.

    Returns:
        Validated MeetingIntelligence Pydantic instance.

    Raises:
        InvalidInputError: If transcript is empty, near-empty, or exceeds max length.
        JSONFixError: If schema validation fails after all fix retries.
        LLMAPIError: If API calls fail after all exponential backoff retries.
        LLMExtractionError: General extraction failure.
    """
    # 1. Input Validation
    if not transcript or not transcript.strip():
        raise InvalidInputError("Transcript is empty or contains only whitespace.")

    words = transcript.strip().split()
    if len(words) < MIN_INPUT_WORDS:
        raise InvalidInputError(
            f"Transcript is too short ({len(words)} words). "
            f"A minimum of {MIN_INPUT_WORDS} words is required."
        )

    estimated_tokens = count_tokens(transcript)
    if estimated_tokens > MAX_INPUT_TOKENS:
        raise InvalidInputError(
            f"Transcript exceeds maximum allowed input size: {estimated_tokens} estimated tokens "
            f"(limit: {MAX_INPUT_TOKENS})."
        )

    # 2. Chunking & Merge Pass (if transcript is long)
    working_transcript = transcript
    if estimated_tokens > max_chunk_tokens:
        logger.info(
            f"Transcript estimated at {estimated_tokens} tokens (> {max_chunk_tokens}). "
            "Chunking and summarizing before extraction."
        )
        chunks = chunk_transcript(transcript, max_tokens=max_chunk_tokens)
        chunk_summaries = [summarize_chunk(c, client=client, model=model) for c in chunks]
        working_transcript = merge_chunk_summaries(chunk_summaries, client=client, model=model)

    # 3. Initial Extraction Call
    prompt = load_prompt("meeting_intelligence_prompt.txt", transcript=working_transcript)
    raw_output = call_gemini_with_retry(prompt=prompt, client=client, model=model)

    # 4. Parsing & Validation with JSON-Fix Retries
    try:
        return validate_and_parse(raw_output)
    except JSONValidationError as initial_err:
        current_output = raw_output
        current_err = initial_err

        for retry in range(1, JSON_FIX_MAX_RETRIES + 1):
            logger.warning(
                f"[JSON-Fix Retry {retry}/{JSON_FIX_MAX_RETRIES}] Validation failed: {current_err}. "
                "Prompting LLM with fix template..."
            )
            fix_prompt = load_prompt(
                "fix_json_prompt.txt",
                raw_output=current_output,
                error_message=str(current_err),
            )
            try:
                fixed_output = call_gemini_with_retry(
                    prompt=fix_prompt,
                    client=client,
                    model=model,
                )
                return validate_and_parse(fixed_output)
            except JSONValidationError as retry_err:
                current_err = retry_err
                current_output = fixed_output
            except LLMAPIError:
                raise

        logger.error(
            f"[JSON-Fix Exhausted] Failed to produce valid JSON schema after {JSON_FIX_MAX_RETRIES} retries."
        )
        raise JSONFixError(
            f"Failed to extract valid meeting intelligence after {JSON_FIX_MAX_RETRIES} JSON-fix retries. "
            f"Last validation error:\n{current_err}"
        ) from current_err