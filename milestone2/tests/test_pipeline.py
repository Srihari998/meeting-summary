import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import process_meeting
from schemas import MeetingIntelligence
from db import get_session
from models import Meeting, ActionItemRecord, ParticipantRecord

SAMPLE_TRANSCRIPT = """
Alice: Welcome everyone. Today we are locking in the Milestone 2 deliverables.
Bob: I will implement the database models and persistence layer by Friday.
Charlie: I will coordinate the participant mapping and action item validation.
Alice: Priority for database is High, action items is Medium. Let us review on Monday.
"""

MOCK_LLM_OUTPUT = {
    "summary": "Team aligned on Milestone 2 deliverables and assignments.",
    "key_points": [
        "Milestone 2 deliverables locked in.",
        "Database models and persistence targeted for Friday."
    ],
    "decisions": [
        "Review progress next Monday."
    ],
    "action_items": [
        {
            "task": "Implement database models and persistence layer",
            "assignee": "Bob",
            "deadline": "Friday",
            "priority": "High",
            "status": "Not Started"
        },
        {
            "task": "Coordinate participant mapping and action item validation",
            "assignee": "Charlie",
            "deadline": None,
            "priority": "Medium",
            "status": "Not Started"
        },
        {
            "task": "Prepare security audit report",
            "assignee": "Eve",
            "deadline": "Next week",
            "priority": "Low",
            "status": "Not Started"
        }
    ],
    "participants": ["Alice", "Bob", "charlie"]
}


@pytest.fixture
def temp_sqlite_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield db_path
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


def test_process_meeting_end_to_end(temp_sqlite_db):
    """
    End-to-end integration test for process_meeting():
    - Mocks the LLM generate_content call.
    - Runs full pipeline (validation -> LLM -> normalization -> DB save).
    - Verifies returned MeetingIntelligence.
    - Queries the SQLite database to verify all rows landed correctly.
    """
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(MOCK_LLM_OUTPUT)
    mock_client.models.generate_content.return_value = mock_resp

    saved_id, result = process_meeting(
        transcript=SAMPLE_TRANSCRIPT,
        meeting_id="test-meeting-101",
        db_path=temp_sqlite_db,
        client=mock_client
    )

    # 0. Verify returned ID matches the one we passed in (Issue #4 regression)
    assert saved_id == "test-meeting-101", (
        f"process_meeting must return the exact saved meeting ID. Got: {saved_id}"
    )

    # 1. Verify structured result
    assert isinstance(result, MeetingIntelligence)
    assert result.summary == MOCK_LLM_OUTPUT["summary"]
    assert len(result.action_items) == 3
    # Charlie should be capitalized
    assert "Charlie" in result.participants
    # Eve should be added as unknown assignee
    assert "Eve" in result.participants

    # 2. Verify Database Persistence
    session = get_session(temp_sqlite_db)
    try:
        meeting_row = session.query(Meeting).filter_by(id="test-meeting-101").first()
        assert meeting_row is not None
        assert meeting_row.transcript == SAMPLE_TRANSCRIPT
        assert meeting_row.summary == MOCK_LLM_OUTPUT["summary"]
        assert len(meeting_row.key_points) == 2
        assert len(meeting_row.decisions) == 1

        # Check action items in DB
        action_rows = session.query(ActionItemRecord).filter_by(meeting_id="test-meeting-101").all()
        assert len(action_rows) == 3
        tasks = {r.task for r in action_rows}
        assert "Implement database models and persistence layer" in tasks

        # Check participants in DB
        participant_rows = session.query(ParticipantRecord).filter_by(meeting_id="test-meeting-101").all()
        assert len(participant_rows) == 4 # Alice, Bob, Charlie, Eve
        p_dict = {p.name: p.is_unknown_assignee for p in participant_rows}
        assert p_dict["Alice"] is False
        assert p_dict["Bob"] is False
        assert p_dict["Charlie"] is False
        assert p_dict["Eve"] is True

    finally:
        session.close()


def test_process_meeting_returns_exact_saved_id(temp_sqlite_db):
    """
    Regression test for Issue #4: process_meeting must return the exact meeting_id
    that was persisted to the database, enabling downstream consumers (e.g. the
    vector store indexer) to use it directly without an additional DB lookup.
    """
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(MOCK_LLM_OUTPUT)
    mock_client.models.generate_content.return_value = mock_resp

    custom_id = "regression-issue-4-id"
    saved_id, intel = process_meeting(
        transcript=SAMPLE_TRANSCRIPT,
        meeting_id=custom_id,
        db_path=temp_sqlite_db,
        client=mock_client,
    )

    assert saved_id == custom_id, (
        f"Returned ID '{saved_id}' does not match the persisted ID '{custom_id}'"
    )
    assert isinstance(intel, MeetingIntelligence)

    # Verify the row actually exists under that exact ID
    session = get_session(temp_sqlite_db)
    try:
        row = session.query(Meeting).filter_by(id=custom_id).first()
        assert row is not None, f"No row found in DB for id={custom_id}"
    finally:
        session.close()