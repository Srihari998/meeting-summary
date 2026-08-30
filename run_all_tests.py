"""
run_all_tests.py
Milestone 1 Master Test Suite Runner

Runs all verification and testing scripts mapped to Milestone 1 tasks:
- Task 1: Complete Whisper Transcription Workflow
- Task 2: File Upload Validation & Negative Tests
- Task 3: Transcript Validation & Auto-Save Integrity
- Task 5: Accuracy Benchmarking & Word Error Rate (WER) Evaluation
"""

import sys
import subprocess
import os

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


TEST_SCRIPTS = [
    ("Task 1 - Whisper Transcription Workflow", "test_task1_workflow.py"),
    ("Task 2 - File Upload Validation Suite",    "test_task2_validation.py"),
    ("Task 3 - Transcript Validation & Saving",  "test_task3_transcript.py"),
    ("Task 5 - Accuracy & WER Benchmarking",     "test_task5_accuracy.py"),
]


def run_suite() -> bool:
    print("\n" + "=" * 70)
    print("      MILESTONE 1: COMPLETE AUTOMATED TEST SUITE RUNNER")
    print("=" * 70 + "\n")

    results = []

    for name, script in TEST_SCRIPTS:
        script_path = os.path.join(os.path.dirname(__file__), script)
        print(f"▶ Running {name} ({script})...")

        res = subprocess.run([sys.executable, script_path], capture_output=False)
        passed = (res.returncode == 0)
        results.append((name, script, passed))

    print("\n" + "=" * 70)
    print("                    MILESTONE 1 SCORECARD")
    print("=" * 70)

    for name, script, passed in results:
        status_str = "[ PASS ]" if passed else "[ FAIL ]"
        print(f"  {status_str}  {name} ({script})")

    all_passed = all(p for _, _, p in results)
    print("=" * 70)
    print(f"  OVERALL RESULT: {'ALL MILESTONE 1 TASKS VERIFIED (100% PASS)' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_suite()
    sys.exit(0 if success else 1)
