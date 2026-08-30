"""
accuracy_test.py
Task 5 -- Accuracy Testing

Tests transcription accuracy by comparing Whisper output against a
known reference transcript using Word Error Rate (WER).

Target: >= 90% accuracy (WER <= 10%)

Usage
-----
# Single file test (you provide the reference text):
python accuracy_test.py <audio_file> "<reference transcript>"

# Batch test using a JSON test-suite file:
python accuracy_test.py --batch tests.json

tests.json format:
[
  {"file": "sample1.wav", "reference": "hello this is a test"},
  {"file": "sample2.mp3", "reference": "the quick brown fox jumps"}
]
"""

from __future__ import annotations
import sys
import os
import json
import argparse
import traceback

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from audio_processor import AudioProcessor
from transcriber import Transcriber

# ── WER calculation ────────────────────────────────────────────────────────────

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) without external libraries.

    WER = (Substitutions + Deletions + Insertions) / len(reference_words)

    Uses dynamic programming (edit distance on word sequences).
    """
    ref_words  = reference.lower().split()
    hyp_words  = hypothesis.lower().split()

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return 0.0 if m == 0 else 1.0

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[n][m] / n


def accuracy_from_wer(wer: float) -> float:
    return max(0.0, 1.0 - wer)


def find_diff_words(reference: str, hypothesis: str) -> list[str]:
    """Return words in reference that are missing from hypothesis."""
    ref_set = set(reference.lower().split())
    hyp_set = set(hypothesis.lower().split())
    return sorted(ref_set - hyp_set)


# ── Single file test ───────────────────────────────────────────────────────────

def run_single_test(
    audio_path: str,
    reference: str,
    model_name: str = "base",
    verbose: bool = True,
) -> dict:
    """
    Transcribe one file and compare against reference.

    Returns a results dict with: file, wer, accuracy, passed,
    hypothesis, reference, missing_words.
    """
    result = {
        "file":          audio_path,
        "reference":     reference,
        "hypothesis":    "",
        "wer":           1.0,
        "accuracy":      0.0,
        "passed":        False,
        "missing_words": [],
        "error":         None,
    }

    wav_path = None
    try:
        # Step 1: convert audio
        processor = AudioProcessor()
        wav_path = processor.process(audio_path)

        # Step 2: transcribe
        transcriber = Transcriber(model_name=model_name)
        transcript  = transcriber.transcribe(wav_path)
        hypothesis  = transcript.get("text", "").strip()

        # Step 3: calculate WER
        wer      = calculate_wer(reference, hypothesis)
        accuracy = accuracy_from_wer(wer)

        result.update({
            "hypothesis":    hypothesis,
            "wer":           round(wer, 4),
            "accuracy":      round(accuracy, 4),
            "passed":        accuracy >= 0.90,
            "missing_words": find_diff_words(reference, hypothesis),
        })

    except Exception as exc:
        result["error"] = str(exc)
        if verbose:
            traceback.print_exc()
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

    return result


# ── Report printer ─────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def print_report(results: list[dict]) -> None:
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Transcription Accuracy Report{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    passed_count = 0

    for i, r in enumerate(results, 1):
        filename = os.path.basename(r["file"])
        status   = f"{GREEN}PASS{RESET}" if r["passed"] else f"{RED}FAIL{RESET}"
        print(f"  Test {i}: {CYAN}{filename}{RESET}  [{status}]")

        if r["error"]:
            print(f"    {RED}ERROR: {r['error']}{RESET}")
            continue

        acc_pct = r['accuracy'] * 100
        wer_pct = r['wer'] * 100
        color   = GREEN if r["passed"] else RED

        print(f"    Accuracy : {color}{acc_pct:.1f}%{RESET}  (WER: {wer_pct:.1f}%)")
        print(f"    Reference: {r['reference'][:80]}")
        print(f"    Got      : {r['hypothesis'][:80]}")

        if r["missing_words"]:
            print(f"    {YELLOW}Missing words: {', '.join(r['missing_words'][:10])}{RESET}")

        if r["passed"]:
            passed_count += 1
        print()

    # Summary
    total  = len(results)
    failed = total - passed_count

    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Results  : {passed_count}/{total} passed")
    print(f"  Pass rate: {(passed_count/total)*100:.0f}%" if total else "  No tests run.")

    if failed == 0:
        print(f"  {GREEN}All tests passed the >= 90% accuracy threshold.{RESET}")
    else:
        print(f"  {RED}{failed} test(s) below 90% accuracy threshold.{RESET}")
        print(f"  {YELLOW}Tip: try a larger Whisper model (--model small or medium).{RESET}")

    print(f"{BOLD}{'='*60}{RESET}\n")


# ── Save JSON report ───────────────────────────────────────────────────────────

def save_json_report(results: list[dict], path: str = "accuracy_report.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Report saved to: {path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Whisper transcription accuracy."
    )
    parser.add_argument("audio", nargs="?", help="Path to audio/video file")
    parser.add_argument("reference", nargs="?", help="Reference transcript text")
    parser.add_argument(
        "--batch", metavar="JSON_FILE",
        help="Path to a JSON test-suite file for batch testing"
    )
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (default: base)"
    )
    parser.add_argument(
        "--save-report", action="store_true",
        help="Save results as accuracy_report.json"
    )
    args = parser.parse_args()

    # ── Batch mode ────────────────────────────────────────────────────
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"{RED}Batch file not found: {args.batch}{RESET}")
            sys.exit(1)
        with open(args.batch, encoding="utf-8") as f:
            test_cases = json.load(f)

        print(f"\nRunning {len(test_cases)} test(s) with Whisper '{args.model}' model...")
        results = []
        for tc in test_cases:
            print(f"  Testing: {tc['file']}")
            r = run_single_test(tc["file"], tc["reference"], model_name=args.model)
            results.append(r)

    # ── Single file mode ──────────────────────────────────────────────
    elif args.audio and args.reference:
        print(f"\nRunning single test with Whisper '{args.model}' model...")
        results = [run_single_test(args.audio, args.reference, model_name=args.model)]

    else:
        parser.print_help()
        print(
            f"\n{YELLOW}Examples:{RESET}\n"
            f"  python accuracy_test.py sample.wav \"hello this is a test\"\n"
            f"  python accuracy_test.py --batch tests.json --model small\n"
            f"  python accuracy_test.py sample.wav \"reference text\" --save-report\n"
        )
        sys.exit(0)

    print_report(results)

    if args.save_report:
        save_json_report(results)

    # Exit with non-zero if any test failed
    all_passed = all(r["passed"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
