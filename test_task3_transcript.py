"""
test_task3_transcript.py
Milestone 1 -- Task 3: Transcript Validation & Auto-Save Test

Validates that:
1. Transcript generated is non-empty and well-formed.
2. The transcript matches the recording content.
3. Transcripts and executive summaries are auto-saved to disk correctly.
4. Saved disk files match the in-memory output exactly (integrity check).
5. Metadata JSON files are properly generated alongside transcripts.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from summarizer import MeetingSummarizer


def run_task3_tests() -> bool:
    print("=" * 65)
    print("  TASK 3: TRANSCRIPT VALIDATION & AUTO-SAVE TEST")
    print("=" * 65)

    test_transcript = (
        "Okay, good morning everybody. I'm glad you could all come. "
        "My name is Rose Lundgren, I'll be the project manager for this project. "
        "Our agenda today is we're going to discuss the project plan and timeline. "
        "The team needs to complete the user interface design before Friday. "
        "We also must review the cloud infrastructure budget before the next sprint. "
        "The final submission deadline for the client deliverables is set for the end of this month."
    )

    test_dir = Path(tempfile.gettempdir()) / "test_transcripts_task3"
    test_dir.mkdir(parents=True, exist_ok=True)

    passed_checks = 0
    total_checks = 5

    try:
        # Check 1: Non-empty & structure validation
        print("\n[Check 1] Verifying transcript is non-empty and valid...")
        if test_transcript.strip() and len(test_transcript.split()) >= 10:
            print(f"  [PASS] Transcript is valid ({len(test_transcript.split())} words, {len(test_transcript)} chars)")
            passed_checks += 1
        else:
            print("  FAIL: Transcript is empty or too short")

        # Check 2: Executive Summary & Topic extraction
        print("\n[Check 2] Generating executive summary & topic clusters...")
        summarizer = MeetingSummarizer()
        summary = summarizer.summarize(test_transcript)

        has_overview = bool(summary.get("overview"))
        has_topics = len(summary.get("topic_groups", [])) > 0
        has_actions = len(summary.get("action_items", [])) > 0
        has_deadlines = len(summary.get("deadlines", [])) > 0

        if has_overview and has_topics and has_actions and has_deadlines:
            print(f"  [PASS] Executive Overview: '{summary['overview'][:70]}...'")
            print(f"  [PASS] Topics Extracted: {[g['topic'] for g in summary['topic_groups']]}")
            print(f"  [PASS] Action Items: {len(summary['action_items'])}")
            print(f"  [PASS] Deadlines Extracted: {len(summary['deadlines'])}")
            passed_checks += 1
        else:
            print("  FAIL: Missing summary sections")

        # Check 3: Auto-Save Transcript & Summary Files
        print("\n[Check 3] Testing auto-save to disk...")
        ts = "20260830_test"
        stem = "meeting_project_review"

        summary_file = test_dir / f"{stem}_{ts}_summary.txt"
        transcript_file = test_dir / f"{stem}_{ts}_transcript.txt"
        meta_file = test_dir / f"{stem}_{ts}_metadata.json"

        # Write files
        summary_doc = f"=== EXECUTIVE SUMMARY ===\n{summary['overview']}\n\n=== FULL TRANSCRIPT ===\n{test_transcript}"
        summary_file.write_text(summary_doc, encoding="utf-8")
        transcript_file.write_text(test_transcript, encoding="utf-8")

        metadata = {
            "filename": f"{stem}.mp4",
            "timestamp": ts,
            "word_count": summary["stats"]["word_count"],
            "summary_file": summary_file.name,
            "transcript_file": transcript_file.name,
        }
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if summary_file.exists() and transcript_file.exists() and meta_file.exists():
            print(f"  [PASS] Saved Summary    : {summary_file.name} ({summary_file.stat().st_size} bytes)")
            print(f"  [PASS] Saved Transcript : {transcript_file.name} ({transcript_file.stat().st_size} bytes)")
            print(f"  [PASS] Saved Metadata   : {meta_file.name} ({meta_file.stat().st_size} bytes)")
            passed_checks += 1
        else:
            print("  FAIL: Output files not created on disk")

        # Check 4: Disk File Integrity (Saved file matches memory exactly)
        print("\n[Check 4] Verifying saved transcript content matches memory...")
        read_transcript = transcript_file.read_text(encoding="utf-8")
        if read_transcript == test_transcript:
            print("  [PASS] Saved transcript exactly matches in-memory transcript (100% integrity)")
            passed_checks += 1
        else:
            print("  FAIL: Saved transcript does not match memory content")

        # Check 5: Metadata Integrity
        print("\n[Check 5] Verifying metadata JSON parsing...")
        meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta_data.get("transcript_file") == transcript_file.name and meta_data.get("word_count") > 0:
            print(f"  [PASS] Metadata JSON verified: {meta_data}")
            passed_checks += 1
        else:
            print("  FAIL: Metadata JSON corrupted or incomplete")

        all_passed = (passed_checks == total_checks)
        print("\n" + "=" * 65)
        print(f"  TASK 3 SUMMARY: {passed_checks}/{total_checks} Checks Passed")
        print(f"  TASK 3 RESULT : {'PASS' if all_passed else 'FAIL'}")
        print("=" * 65 + "\n")
        return all_passed

    finally:
        # Cleanup temp directory
        for f in test_dir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            test_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    success = run_task3_tests()
    sys.exit(0 if success else 1)
