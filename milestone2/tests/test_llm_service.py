import json
import pytest
from unittest.mock import MagicMock, patch, call

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import (
    ActionItem,
    JSONValidationError,
    MeetingIntelligence,
    validate_and_parse,
)
from llm_service import (
    DEFAULT_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    API_MAX_RETRIES,
    API_RETRY_DELAYS,
    JSON_FIX_MAX_RETRIES,
    InvalidInputError,
    JSONFixError,
    LLMAPIError,
    LLMExtractionError,
    call_gemini_with_retry,
    chunk_transcript,
    count_tokens,
    extract_meeting_intelligence,
    load_prompt,
    merge_chunk_summaries,
    summarize_chunk,
)

VALID_PAYLOAD = {
    "summary": "Engineering sync discussing milestone progress.",
    "key_points": [
        "LLM service is modularized.",
        "Database layer uses SQLite."
    ],
    "decisions": [
        "Use SQLite for local persistence."
    ],
    "action_items": [
        {
            "task": "Implement unit tests",
            "assignee": "Charlie",
            "deadline": "2026-10-01",
            "priority": "High",
            "status": "Not Started"
        }
    ],
    "participants": ["Charlie", "Diana"]
}


@pytest.fixture
def mock_genai_client():
    return MagicMock()


def test_valid_json_parses_correctly(mock_genai_client):
    """Test that valid JSON from Gemini parses into MeetingIntelligence."""
    mock_response = MagicMock()
    mock_response.text = json.dumps(VALID_PAYLOAD)
    mock_genai_client.models.generate_content.return_value = mock_response

    transcript = "Charlie: We are discussing milestone 2. Diana: Agreed."
    result = extract_meeting_intelligence(transcript, client=mock_genai_client)

    assert isinstance(result, MeetingIntelligence)
    assert result.summary == VALID_PAYLOAD["summary"]
    assert len(result.action_items) == 1
    assert result.action_items[0].task == "Implement unit tests"


def test_json_fix_retry_success(mock_genai_client):
    """Test that malformed JSON triggers fix-retry and succeeds on second attempt."""
    bad_resp = MagicMock(text="INVALID JSON CONTENT")
    good_resp = MagicMock(text=json.dumps(VALID_PAYLOAD))
    mock_genai_client.models.generate_content.side_effect = [bad_resp, good_resp]

    transcript = "Charlie: We need to complete task 2. Diana: Yes."
    result = extract_meeting_intelligence(transcript, client=mock_genai_client)

    assert isinstance(result, MeetingIntelligence)
    assert result.summary == VALID_PAYLOAD["summary"]
    assert mock_genai_client.models.generate_content.call_count == 2


def test_empty_input_rejected_before_api_call(mock_genai_client):
    """Test that empty or near-empty input is rejected before calling Gemini API."""
    with pytest.raises(InvalidInputError):
        extract_meeting_intelligence("", client=mock_genai_client)

    with pytest.raises(InvalidInputError):
        extract_meeting_intelligence("One two", client=mock_genai_client)

    mock_genai_client.models.generate_content.assert_not_called()


@patch("time.sleep", return_value=None)
def test_api_timeout_triggers_backoff_and_exhaustion(mock_sleep, mock_genai_client):
    """Test that persistent API timeout triggers backoff retries and raises LLMAPIError."""
    mock_genai_client.models.generate_content.side_effect = TimeoutError("API Timeout")

    with pytest.raises(LLMAPIError) as exc_info:
        call_gemini_with_retry("prompt", client=mock_genai_client)

    assert "after 3 attempts" in str(exc_info.value)
    assert mock_genai_client.models.generate_content.call_count == 3
    assert mock_sleep.call_args_list == [call(1.0), call(2.0)]


def test_chunk_transcript_natural_boundaries():
    """Test that chunk_transcript does not split in the middle of sentences."""
    sentences = [
        "Alice: We are discussing milestone 2 deliverables.",
        "Bob: Action item extraction is prioritized.",
        "Charlie: Participant mapping is verified."
    ]
    long_text = "\n\n".join(sentences * 20)
    chunks = chunk_transcript(long_text, max_tokens=100)

    assert len(chunks) > 1
    for chunk in chunks:
        stripped = chunk.strip()
        assert stripped[-1] in ".!?" or stripped.endswith("verified.")