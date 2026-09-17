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


def get_genai_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Initializes and returns a Google GenAI client instance.
    Reads GEMINI_API_KEY from environment if not provided explicitly.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMAPIError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please configure GEMINI_API_KEY before using the LLM service."
        )
    return genai.Client(api_key=key)


def load_prompt(template_name: str, **kwargs) -> str:
    """
    Loads a prompt template from the prompts/ directory and replaces {key} placeholders.

    Args:
        template_name: Filename of the template (e.g. 'meeting_intelligence_prompt.txt').
        **kwargs: Variables to substitute into the template string.

    Returns:
        Formatted prompt string.

    Raises:
        FileNotFoundError: If the template file does not exist.
        ValueError: If a required template placeholder is not provided in kwargs.
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


def count_tokens(text: str) -> int:
    """
    Estimates the number of tokens for a given text string.
    Uses ~4 characters per token heuristic, adjusted for word density.
    """
    if not text:
        return 0
    char_count_tokens = len(text) / 4.0
    word_count_tokens = len(text.split()) * 1.3
    return int(max(char_count_tokens, word_count_tokens, 1))


def chunk_transcript(text: str, max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS) -> List[str]:
    """
    Splits long transcripts into chunks on natural boundaries (speaker turns,
    paragraphs, or sentence boundaries) without cutting mid-sentence.

    Args:
        text: The full transcript text.
        max_tokens: Maximum allowed estimated tokens per chunk.

    Returns:
        List of text chunks, each within the max_tokens limit.
    """
    if count_tokens(text) <= max_tokens:
        return [text.strip()]

    # Split on natural boundaries: speaker turns (e.g. "Alice:", "[Speaker 1]") or paragraph breaks
    paragraphs = re.split(r"(?:\r?\n){2,}|(?=\n[A-Za-z0-9 _-]+:)", text)
    
    units: List[str] = []
    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue
        
        # If a single paragraph is too large, break it down by sentences
        if count_tokens(para_clean) > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para_clean)
            for sentence in sentences:
                sent_clean = sentence.strip()
                if not sent_clean:
                    continue
                # If a single sentence is absurdly long, break by words
                if count_tokens(sent_clean) > max_tokens:
                    words = sent_clean.split()
                    current_word_chunk: List[str] = []
                    for word in words:
                        current_word_chunk.append(word)
                        if count_tokens(" ".join(current_word_chunk)) >= max_tokens:
                            units.append(" ".join(current_word_chunk))
                            current_word_chunk = []
                    if current_word_chunk:
                        units.append(" ".join(current_word_chunk))
                else:
                    units.append(sent_clean)
        else:
            units.append(para_clean)

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
    Executes a Gemini API call with exponential backoff retries on transient API/network errors.

    Retries up to API_MAX_RETRIES times with backoff delays (1s, 2s, 4s).

    Args:
        prompt: The input prompt string.
        client: Optional genai.Client instance (created if None).
        model: Model identifier.
        max_output_tokens: Explicit token budget for response.
        temperature: Sampling temperature.

    Returns:
        Generated text response from the model.

    Raises:
        LLMAPIError: If all retries are exhausted.
    """
    active_client = client or get_genai_client()
    config = types.GenerateContentConfig(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    last_exception: Optional[Exception] = None

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
                delay = API_RETRY_DELAYS[attempt - 1] if (attempt - 1) < len(API_RETRY_DELAYS) else 4.0
                logger.warning(
                    f"[API Retry {attempt}/{API_MAX_RETRIES}] Gemini API call failed with error: {exc}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"[API Exhausted] Gemini API call failed on final attempt {attempt}/{API_MAX_RETRIES}: {exc}"
                )

    raise LLMAPIError(
        f"Gemini API request failed after {API_MAX_RETRIES} attempts. Last error: {last_exception}"
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