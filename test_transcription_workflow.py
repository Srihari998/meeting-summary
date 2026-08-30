"""
test_transcription_workflow.py
Standalone CLI test -- runs the full pipeline without the Streamlit UI.

Usage:
    python test_transcription_workflow.py <path_to_audio_or_video>

Exits with code 0 on PASS, 1 on FAIL.
"""

import sys
import os
import traceback

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from audio_processor import AudioProcessor
from transcriber import Transcriber

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def print_result(label: str, passed: bool, detail: str = "") -> None:
    icon   = f"{GREEN}[OK]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  {icon}  [{status}] {label}")
    if detail:
        print(f"        {YELLOW}{detail}{RESET}")


def run_test(input_path: str) -> bool:
    overall = True
    wav_path = None
    result = {}

    print(f"\n{BOLD}=== Whisper Transcription Workflow Test ==={RESET}")
    print(f"  Input file : {input_path}\n")

    # -- Stage 1: File exists --------------------------------------------------
    exists = os.path.isfile(input_path)
    print_result("Stage 1 - File is readable", exists,
                 "" if exists else f"Not found: {input_path}")
    overall = overall and exists
    if not exists:
        return overall

    # -- Stage 2: Audio conversion ---------------------------------------------
    try:
        processor = AudioProcessor()
        wav_path = processor.process(input_path)
        wav_ok = os.path.exists(wav_path) and os.path.getsize(wav_path) > 0
        detail = f"WAV -> {wav_path} ({os.path.getsize(wav_path) / 1024:.1f} KB)"
        print_result("Stage 2 - Audio converted to 16 kHz mono WAV", wav_ok, detail)
        overall = overall and wav_ok
    except Exception as exc:
        print_result("Stage 2 - Audio converted to 16 kHz mono WAV", False, str(exc))
        traceback.print_exc()
        overall = False
        return overall

    # -- Stage 3: Whisper loads and transcribes --------------------------------
    try:
        print(f"\n  Loading Whisper 'base' model (first run downloads ~140 MB)...")
        transcriber = Transcriber(model_name="base")
        result = transcriber.transcribe(wav_path)

        has_text     = bool(result.get("text", "").strip())
        has_segments = len(result.get("segments", [])) > 0
        has_language = bool(result.get("language"))

        print_result("Stage 3a - Whisper loads without error", True)
        print_result("Stage 3b - Non-empty transcript returned", has_text,
                     f"text[:80] = {result['text'][:80]!r}" if has_text else "text is empty")
        print_result("Stage 3c - Segments returned", has_segments,
                     f"{len(result['segments'])} segment(s)")
        print_result("Stage 3d - Language detected", has_language,
                     f"language = {result['language']!r}")

        overall = overall and has_text and has_segments and has_language

    except Exception as exc:
        print_result("Stage 3 - Whisper transcription", False, str(exc))
        traceback.print_exc()
        overall = False

    # -- Final summary ---------------------------------------------------------
    print(f"\n{BOLD}--- Full Transcript ---{RESET}")
    if result.get("text"):
        print(result["text"])
    else:
        print("  (transcription did not complete or no speech detected)")

    print(f"\n{BOLD}=== Overall Result: ", end="")
    if overall:
        print(f"{GREEN}PASS{RESET}{BOLD} ==={RESET}\n")
    else:
        print(f"{RED}FAIL{RESET}{BOLD} ==={RESET}\n")

    # Cleanup temp WAV
    if wav_path and os.path.exists(wav_path):
        try:
            os.remove(wav_path)
        except OSError:
            pass

    return overall


def main() -> None:
    if len(sys.argv) < 2:
        print(
            f"{RED}Usage: python test_transcription_workflow.py <audio_or_video_file>{RESET}\n"
            "Example: python test_transcription_workflow.py sample.mp3\n"
        )
        sys.exit(1)

    input_path = sys.argv[1]
    passed = run_test(input_path)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
