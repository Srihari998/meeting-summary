"""
run_evaluation.py
Milestone 1 Task 5 — Batch Accuracy Evaluation Script

Processes all recordings in evaluation/recordings/ that have a matching
reference transcript in evaluation/references/ and writes results to
evaluation/accuracy_results.csv.

Usage:
    python run_evaluation.py [--model base]
"""

from __future__ import annotations
import argparse
import csv
import datetime
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from accuracy import calculate_wer_and_metrics
from audio_processor import AudioProcessor
from transcriber import Transcriber
from validator import FileValidator, SUPPORTED_EXTENSIONS

EVAL_DIR = Path(__file__).parent / "evaluation"
RECORDINGS_DIR = EVAL_DIR / "recordings"
REFERENCES_DIR = EVAL_DIR / "references"
RESULTS_CSV = EVAL_DIR / "accuracy_results.csv"

CSV_FIELDS = [
    "recording", "model", "duration_sec", "ref_words", "gen_words",
    "substitutions", "deletions", "insertions", "wer", "accuracy_pct",
    "pass_fail", "notes"
]


def get_audio_duration(wav_path: str) -> float:
    import wave
    try:
        with wave.open(wav_path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def find_recording_pairs() -> list[tuple[Path, Path]]:
    """Find all matching recording/reference pairs."""
    pairs = []
    if not RECORDINGS_DIR.exists():
        return pairs
    for rec in RECORDINGS_DIR.iterdir():
        ext = rec.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS or rec.stem.startswith("."):
            continue
        ref = REFERENCES_DIR / f"{rec.stem}.txt"
        if ref.exists():
            pairs.append((rec, ref))
    return sorted(pairs)


def evaluate_recording(
    recording: Path,
    reference_path: Path,
    model_name: str,
) -> dict:
    processor = AudioProcessor()
    transcriber = Transcriber(model_name=model_name)
    tmp_wav = None

    try:
        tmp_wav = processor.process(str(recording))
        duration = get_audio_duration(tmp_wav)
        result = transcriber.transcribe(tmp_wav)
        hypothesis = result.get("text", "").strip()
        reference = reference_path.read_text(encoding="utf-8").strip()

        metrics = calculate_wer_and_metrics(reference, hypothesis)
        return {
            "recording": recording.name,
            "model": model_name,
            "duration_sec": round(duration, 1),
            "ref_words": metrics["ref_words"],
            "gen_words": metrics["hyp_words"],
            "substitutions": metrics["substitutions"],
            "deletions": metrics["deletions"],
            "insertions": metrics["insertions"],
            "wer": metrics["wer"],
            "accuracy_pct": metrics["accuracy_pct"],
            "pass_fail": "PASS" if metrics["passed"] else "FAIL",
            "notes": "",
        }
    except Exception as exc:
        return {
            "recording": recording.name,
            "model": model_name,
            "duration_sec": 0,
            "ref_words": 0, "gen_words": 0,
            "substitutions": 0, "deletions": 0, "insertions": 0,
            "wer": 1.0, "accuracy_pct": 0.0,
            "pass_fail": "ERROR",
            "notes": str(exc),
        }
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Milestone 1 — Batch Accuracy Evaluation")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"])
    args = parser.parse_args()

    pairs = find_recording_pairs()

    if not pairs:
        print("=" * 60)
        print("  ACCURACY EVALUATION — No Test Pairs Found")
        print("=" * 60)
        print(
            "\nNo recording/reference pairs were found in:\n"
            f"  Recordings : {RECORDINGS_DIR}\n"
            f"  References : {REFERENCES_DIR}\n\n"
            "To run evaluation:\n"
            "  1. Place audio/video files in evaluation/recordings/\n"
            "  2. Place matching .txt reference files in evaluation/references/\n"
            "  3. Run: python run_evaluation.py\n"
        )
        print("STATUS: FRAMEWORK READY — NOT YET TESTED")
        return

    print("=" * 65)
    print(f"  MILESTONE 1 TASK 5 — ACCURACY EVALUATION  (model={args.model})")
    print("=" * 65)

    results = []
    for rec, ref in pairs:
        print(f"\n  Processing: {rec.name}")
        row = evaluate_recording(rec, ref, args.model)
        results.append(row)
        status = row["pass_fail"]
        print(f"    Accuracy: {row['accuracy_pct']}% | WER: {row['wer'] * 100:.1f}% | {status}")

    # Write CSV
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    passed = sum(1 for r in results if r["pass_fail"] == "PASS")
    avg_acc = sum(r["accuracy_pct"] for r in results) / len(results)

    print("\n" + "=" * 65)
    print(f"  Recordings tested : {len(results)}")
    print(f"  Passed (>=90%)    : {passed}/{len(results)}")
    print(f"  Average accuracy  : {avg_acc:.1f}%")
    print(f"  Results saved to  : {RESULTS_CSV}")
    print("=" * 65)


if __name__ == "__main__":
    main()
