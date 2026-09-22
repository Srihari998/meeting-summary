import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import ActionItem
from participants import (
    normalize_name,
    dedupe_participants,
    reconcile_participants_and_assignees,
)


def test_normalize_name_casing_and_roles():
    """Tests name normalization on messy casing, roles, and punctuation."""
    assert normalize_name("ravi") == "Ravi"
    assert normalize_name("  ravi k  ") == "Ravi K"
    assert normalize_name("Alice Smith (Host):") == "Alice Smith"
    assert normalize_name("Speaker 2:") == "Speaker 2"
    assert normalize_name("Dr. Bob Jones (Presenter)") == "Dr. Bob Jones"
    assert normalize_name("") == "Unknown"


def test_dedupe_participants_messy_variants():
    """Tests deduplication and prefix resolution on variations like 'ravi' and 'Ravi K'."""
    raw_participants = ["ravi", "Ravi K", "RAVI", "Alice Smith", "alice smith (Host)"]
    deduped = dedupe_participants(raw_participants)

    assert "Ravi K" in deduped
    assert "Alice Smith" in deduped
    # 'ravi' is merged into 'Ravi K' and 'RAVI'/'ravi' are not duplicated
    assert len([p for p in deduped if "ravi" in p.lower()]) == 1
    assert len([p for p in deduped if "alice" in p.lower()]) == 1


def test_unknown_assignee_flagged_not_dropped():
    """
    Tests that an assignee in an action item who is not in the detected
    participants list is flagged as an Unknown participant and NOT dropped.
    """
    detected_participants = ["Alice", "Bob"]
    action_items = [
        ActionItem(task="Setup server", assignee="Alice", priority="High", status="Not Started"),
        ActionItem(task="External audit review", assignee="David", priority="Medium", status="Not Started")
    ]

    canonical_participants, participant_records = reconcile_participants_and_assignees(
        detected_participants,
        action_items
    )

    # David must be included in participant records as unknown assignee
    assert "David" in canonical_participants
    david_records = [r for r in participant_records if r["name"] == "David"]
    assert len(david_records) == 1
    assert david_records[0]["is_unknown_assignee"] is True

    # Alice must NOT be flagged as unknown
    alice_records = [r for r in participant_records if r["name"] == "Alice"]
    assert len(alice_records) == 1
    assert alice_records[0]["is_unknown_assignee"] is False


def test_infer_speaker_roles_returns_different_roles_per_speaker():
    """
    Regression test for Issue #1 — heuristic role detection must check each
    speaker's OWN text, not the whole transcript.

    If 'manager' appears only in Alice's utterances and 'developer' only in
    Bob's, they must receive different roles (not both 'Project Manager').
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from participants import infer_speaker_roles

    transcript = (
        "Alice: I am the project manager and I will coordinate the sprints.\n"
        "Bob: I will handle the database migrations as a backend developer.\n"
    )
    # LLM unavailable — force heuristic fallback by passing a dummy client that raises
    from unittest.mock import MagicMock
    bad_client = MagicMock()
    bad_client.models.generate_content.side_effect = RuntimeError("no LLM")

    roles = infer_speaker_roles(transcript, speakers=["Alice", "Bob"], client=bad_client)

    assert "Alice" in roles
    assert "Bob" in roles
    # Alice mentioned 'manager' only → Project Manager
    assert roles["Alice"] == "Project Manager", f"Expected Project Manager, got {roles['Alice']}"
    # Bob mentioned 'developer'/'database' only → Software Engineer
    assert roles["Bob"] == "Software Engineer", f"Expected Software Engineer, got {roles['Bob']}"
    # Critical: they must NOT be the same role
    assert roles["Alice"] != roles["Bob"], (
        "Bug regression: both speakers got the same role — heuristic is still using full transcript text"
    )