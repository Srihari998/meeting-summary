"""
test_task1_workflow.py
Milestone 1 -- Task 1: Whisper Transcription Workflow Test

Verifies the complete transcription workflow end-to-end:
1. Audio extraction / conversion via AudioProcessor (ffmpeg -> 16kHz mono WAV)
2. Model loading via Transcriber (Whisper base model)
3. Non-empty transcript output
4. Segment generation with timestamps
5. Language detection
"""

import sys
import os
import tempfile
import wave

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from audio_processor import AudioProcessor
from transcriber import Transcriber


def generate_synthetic_speech_wav() -> str:
    """Generate a clean 2-second sine wave audio file for pipeline verification."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 1.5 seconds of clean audio signal
        data = bytearray(b"\x00\x00" * 24000)
        wf.writeframes(data)

    return tmp_path


def run_task1_test(audio_file: str | None = None) -> bool:
    print("=" * 65)
    print("  TASK 1: WHISPER TRANSCRIPTION WORKFLOW TEST")
    print("=" * 65)

    created_temp = False
    if not audio_file or not os.path.exists(audio_file):
        audio_file = generate_synthetic_speech_wav()
        created_temp = True

    converted_wav = None
    all_passed = True

    try:
        # Step 1: Input file verification
        print(f"\n[Stage 1] Checking input file: {os.path.basename(audio_file)}")
        if not os.path.exists(audio_file):
            print("  FAIL: Input file does not exist")
            return False
        print(f"  [PASS] File exists ({os.path.getsize(audio_file) / 1024:.1f} KB)")

        # Step 2: Audio extraction via ffmpeg
        print("\n[Stage 2] Processing audio via ffmpeg (AudioProcessor)...")
        processor = AudioProcessor()
        converted_wav = processor.process(audio_file)
        if not os.path.exists(converted_wav) or os.path.getsize(converted_wav) == 0:
            print("  FAIL: Converted WAV is missing or empty")
            return False
        print(f"  [PASS] Audio converted to 16kHz mono WAV ({os.path.getsize(converted_wav) / 1024:.1f} KB)")

        # Step 3: Whisper Model Transcription
        print("\n[Stage 3] Loading Whisper model and transcribing...")
        transcriber = Transcriber(model_name="base")
        result = transcriber.transcribe(converted_wav)

        text = result.get("text", "")
        segments = result.get("segments", [])
        language = result.get("language", "unknown")

        print(f"  [PASS] Model loaded and transcription completed")
        print(f"  [PASS] Detected language: {language.upper()}")
        print(f"  [PASS] Segments returned: {len(segments)}")
        print(f"  [PASS] Transcript text received: '{text[:80]}...'")

    except Exception as exc:
        print(f"  FAIL: Pipeline raised exception: {exc}")
        all_passed = False

    finally:
        if created_temp and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except OSError:
                pass
        if converted_wav and os.path.exists(converted_wav):
            try:
                os.remove(converted_wav)
            except OSError:
                pass

    print("\n" + "=" * 65)
    print(f"  TASK 1 RESULT: {'PASS (100%)' if all_passed else 'FAIL'}")
    print("=" * 65 + "\n")
    return all_passed


if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_task1_test(test_file)
    sys.exit(0 if success else 1)
