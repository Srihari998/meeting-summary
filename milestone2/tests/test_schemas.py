import json
import pytest
from pydantic import ValidationError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import (
    ActionItem,
    JSONValidationError,
    MeetingIntelligence,
    validate_and_parse,
)

VALID_PAYLOAD = {
    "summary": "Team aligned on sprint goals.",
    "key_points": ["Point 1", "Point 2"],
    "decisions": ["Decision 1"],
    "action_items": [
        {
            "task": "Write test cases",
            "assignee": "Alice",
            "deadline": "2026-09-30",
            "priority": "High",
            "status": "Not Started"
        }
    ],
    "participants": ["Alice", "Bob"]
}


def test_valid_schema_instantiation():
    """Tests creating MeetingIntelligence with valid fields."""
    intel = MeetingIntelligence.model_validate(VALID_PAYLOAD)
    assert intel.summary == "Team aligned on sprint goals."
    assert len(intel.action_items) == 1
    assert intel.action_items[0].priority == "High"


def test_extra_fields_forbidden_on_action_item():
    """Tests that extra fields on ActionItem are forbidden."""
    bad_item = {
        "task": "Task description",
        "assignee": "Bob",
        "deadline": None,
        "priority": "Medium",
        "status": "Not Started",
        "unexpected_field": "disallowed"
    }
    with pytest.raises(ValidationError):
        ActionItem.model_validate(bad_item)


def test_extra_fields_forbidden_on_meeting_intelligence():
    """Tests that extra fields on MeetingIntelligence are forbidden."""
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["extra_meta"] = "not permitted"
    with pytest.raises(ValidationError):
        MeetingIntelligence.model_validate(bad_payload)


def test_invalid_priority_enum_rejected():
    """Tests that an invalid priority value (e.g., 'Critical') is rejected."""
    bad_payload = json.loads(json.dumps(VALID_PAYLOAD))
    bad_payload["action_items"][0]["priority"] = "Critical"
    with pytest.raises(JSONValidationError) as exc_info:
        validate_and_parse(json.dumps(bad_payload))
    assert "priority" in str(exc_info.value)
    assert "Critical" in str(exc_info.value)


def test_validate_and_parse_missing_summary():
    """Tests that missing required summary field is rejected."""
    bad_payload = VALID_PAYLOAD.copy()
    del bad_payload["summary"]
    with pytest.raises(JSONValidationError) as exc_info:
        validate_and_parse(json.dumps(bad_payload))
    assert "summary" in str(exc_info.value)