"""
Pipeline Module
The single public entry point for Milestone 2: Summarization & Action Extraction.

Usage:
    from milestone2.pipeline import process_meeting
    result = process_meeting(transcript_text)
"""

import logging
from typing import Optional
from google import genai

try:
    from action_items import normalize_action_items
    from db import save_meeting
    from llm_service import InvalidInputError, extract_meeting_intelligence
    from participants import reconcile_participants_and_assignees
    from schemas import MeetingIntelligence
except ImportError:
    from .action_items import normalize_action_items
    from .db import save_meeting
    from .llm_service import InvalidInputError, extract_meeting_intelligence
    from .participants import reconcile_participants_and_assignees
    from .schemas import MeetingIntelligence

logger = logging.getLogger(__name__)
MIN_INPUT_WORDS = 3


def process_meeting(
    transcript: str,
    meeting_id: Optional[str] = None,
    db_path: Optional[str] = None,
    client: Optional[genai.Client] = None,
) -> MeetingIntelligence:
    """
    End-to-end meeting processing pipeline:
    1. Validates input transcript (rejects empty or near-empty inputs).
    2. Calls LLM extraction service with dual retry resilience.
    3. Normalizes and deduplicates participants, flagging unknown assignees.
    4. Normalizes action items (priority default 'Medium', status 'Not Started').
    5. Persists meeting, summary, action items, and participants to SQLite database.
    6. Returns structured MeetingIntelligence record.

    Args:
        transcript: Raw meeting transcript string.
        meeting_id: Optional custom identifier for the meeting record.
        db_path: Optional path to SQLite database file.
        client: Optional genai.Client instance (useful for testing/mocking).

    Returns:
        MeetingIntelligence: Structured, validated meeting data.

    Raises:
        InvalidInputError: If transcript is empty or near-empty.
        LLMExtractionError: If LLM extraction or schema parsing fails.
    """
    # 1. Validate Input
    if not transcript or not transcript.strip():
        raise InvalidInputError("Transcript is empty or contains only whitespace.")

    words = transcript.strip().split()
    if len(words) < MIN_INPUT_WORDS:
        raise InvalidInputError(
            f"Transcript is too short ({len(words)} words). Minimum {MIN_INPUT_WORDS} words required."
        )

    # 2. Extract Meeting Intelligence via LLM Service
    logger.info("Extracting structured intelligence from transcript...")
    raw_intelligence = extract_meeting_intelligence(transcript, client=client)

    # 3. Normalize & Deduplicate Participants + Flag Unknown Assignees
    normalized_action_items = normalize_action_items(raw_intelligence.action_items)
    canonical_participants, participant_records = reconcile_participants_and_assignees(
        raw_intelligence.participants,
        normalized_action_items,
    )

    cleaned_intelligence = MeetingIntelligence(
        summary=raw_intelligence.summary.strip(),
        key_points=raw_intelligence.key_points,
        decisions=raw_intelligence.decisions,
        action_items=normalized_action_items,
        participants=canonical_participants,
    )

    # 4. Persist to Database
    logger.info("Persisting meeting intelligence to database...")
    saved_id = save_meeting(
        transcript=transcript,
        parsed=cleaned_intelligence,
        meeting_id=meeting_id,
        db_path=db_path,
        participant_records=participant_records,
    )
    logger.info(f"Meeting record successfully saved with ID: {saved_id}")

    # 5. Return Structured Result
    return cleaned_intelligence