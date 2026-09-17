"""
Milestone 2 Pipeline Demo & Streamlit Integration Helper

Demonstrates how to run `process_meeting()` on a transcript (from a file or
default string) and format the results for terminal display or a UI.

Usage from CLI:
    # Run with default sample transcript
    python milestone2/examples/run_pipeline_demo.py

    # Run with a custom transcript file
    python milestone2/examples/run_pipeline_demo.py path/to/transcript.txt

Usage from your Milestone 1 Streamlit app:
    from milestone2.pipeline import process_meeting
    result = process_meeting(transcribed_text)
    st.write(result.summary)
"""

import os
import sys
from pathlib import Path

# Add project root / milestone2 to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from milestone2.pipeline import process_meeting

DEFAULT_SAMPLE_TRANSCRIPT = """
Alice (Product Lead): Good morning everyone. Let's do our weekly milestone check-in.
Bob (Backend Engineer): I have completed the database schema and SQLAlchemy ORM models. Next, I need to wire up the SQLite migrations by Wednesday.
Charlie (Frontend Lead): The React dashboard components are ready. I will connect them to the backend API endpoints by Thursday afternoon.
Alice: Excellent. Let's set Friday as the feature-freeze deadline so QA can run regression tests.
Bob: Sounds great. I will also coordinate with David on the external security audit next week.
Alice: Thanks all, let's reconvene on Friday for the final demo.
"""


def format_action_items_table(action_items) -> str:
    """Formats a list of ActionItem objects into an aligned ASCII table."""
    if not action_items:
        return "No action items extracted."

    headers = ["Task", "Assignee", "Deadline", "Priority", "Status"]
    rows = [
        [
            item.task,
            item.assignee,
            str(item.deadline or "N/A"),
            item.priority,
            item.status,
        ]
        for item in action_items
    ]

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    # Format table lines
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_lines = [
        " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row))
        for row in rows
    ]

    return f"{header_line}\n{separator_line}\n" + "\n".join(row_lines)


def main():
    # 1. Load transcript text from CLI argument or default sample
    if len(sys.argv) > 1:
        transcript_path = Path(sys.argv[1])
        if not transcript_path.exists():
            print(f"Error: File not found at '{transcript_path}'")
            sys.exit(1)
        transcript_text = transcript_path.read_text(encoding="utf-8")
        print(f"Loaded transcript from: {transcript_path}")
    else:
        transcript_text = DEFAULT_SAMPLE_TRANSCRIPT
        print("Using default sample transcript.")

    print("\nProcessing transcript with Milestone 2 pipeline...")

    # 2. Run the end-to-end pipeline
    result = process_meeting(transcript_text)

    # 3. Print formatted output
    print("\n" + "=" * 80)
    print("MEETING INTELLIGENCE RESULTS")
    print("=" * 80)

    print("\n[EXECUTIVE SUMMARY]:")
    print(result.summary)

    print("\n[KEY POINTS]:")
    for kp in result.key_points:
        print(f"  • {kp}")

    print("\n[DECISIONS]:")
    for dec in result.decisions:
        print(f"  ✓ {dec}")

    print("\n[PARTICIPANTS]:")
    print(f"  {', '.join(result.participants)}")

    print("\n[ACTION ITEMS TABLE]:")
    print(format_action_items_table(result.action_items))

    print("\n" + "=" * 80)
    print("Record saved to database (meeting_intelligence.db). Pipeline complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()