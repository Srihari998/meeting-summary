"""
Live Smoke Test against real Gemini API.
Skipped by default unless RUN_LIVE_TESTS environment variable is set.

To run this test:
  $env:RUN_LIVE_TESTS="1"; $env:GEMINI_API_KEY="your-key-here"; python -m pytest milestone2/tests/test_live_smoke.py -v -s
"""

import json
import os
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_service import extract_meeting_intelligence
from schemas import MeetingIntelligence

SAMPLE_LIVE_TRANSCRIPT = """
Alice: Welcome everyone to our sprint retrospective. Let's do a quick sync on the deployment status.
Bob: The database migration scripts for PostgreSQL are ready, but we need load testing before production.
Charlie: I will run the load testing benchmark suite on staging by Thursday 5 PM.
Alice: Sounds good. Let's approve the staging release for Wednesday morning and target Friday for the production rollout.
Bob: I will also prepare the rollback plan document by Wednesday afternoon.
Alice: Great, let's reconvene on Friday morning to review test results.
"""


@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_TESTS"),
    reason="Live Gemini API test skipped by default. Set RUN_LIVE_TESTS=1 and GEMINI_API_KEY to run."
)
def test_live_gemini_extraction_smoke():
    """
    Executes a real (non-mocked) call to the Gemini API using gemini-2.5-flash.
    Prints the resulting structured MeetingIntelligence to stdout for visual verification.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.fail("GEMINI_API_KEY environment variable is required to run live smoke tests.")

    print("\n" + "=" * 70)
    print("RUNNING LIVE GEMINI API SMOKE TEST (gemini-2.5-flash)")
    print("=" * 70)

    result = extract_meeting_intelligence(SAMPLE_LIVE_TRANSCRIPT)

    print("\n[EXECUTIVE SUMMARY]:")
    print(result.summary)

    print("\n[KEY POINTS]:")
    for pt in result.key_points:
        print(f"  • {pt}")

    print("\n[DECISIONS]:")
    for dec in result.decisions:
        print(f"  ✓ {dec}")

    print("\n[PARTICIPANTS]:")
    print(f"  {', '.join(result.participants)}")

    print("\n[ACTION ITEMS]:")
    for item in result.action_items:
        print(f"  - [{item.priority}] {item.task}")
        print(f"      Assignee: {item.assignee} | Deadline: {item.deadline} | Status: {item.status}")

    print("=" * 70 + "\n")

    # Schema & Content Assertions
    assert isinstance(result, MeetingIntelligence)
    assert len(result.summary.strip()) > 0
    assert len(result.key_points) > 0
    assert len(result.decisions) > 0
    assert len(result.action_items) >= 1
    assert any("load" in item.task.lower() or "rollback" in item.task.lower() for item in result.action_items)