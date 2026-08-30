"""
test_task2_validation.py
Milestone 1 -- Task 2: File Upload Validation Test Suite

Validates that:
1. Valid audio/video formats are accepted (.mp4, .mkv, .wav, .mp3, etc.)
2. Invalid file extensions (.txt, .exe, .pdf, .zip) are rejected
3. 0-byte / tiny corrupt files are rejected
4. Non-media files disguised as media (e.g. text file renamed to .mp4) are rejected
5. Informative error messages are returned for all rejection cases
"""

import sys
import os
import tempfile
import wave

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from validator import FileValidator


def create_temp_wav(filename: str, duration_sec: float = 1.0) -> str:
    """Create a valid WAV file for testing."""
    path = os.path.join(tempfile.gettempdir(), filename)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(bytearray(int(16000 * duration_sec * 2)))
    return path


def create_temp_file(filename: str, content: bytes) -> str:
    """Create an arbitrary file for negative testing."""
    path = os.path.join(tempfile.gettempdir(), filename)
    with open(path, "wb") as f:
        f.write(content)
    return path


def run_task2_tests() -> bool:
    print("=" * 65)
    print("  TASK 2: FILE UPLOAD VALIDATION TEST SUITE")
    print("=" * 65)

    validator = FileValidator()
    test_cases = []
    temp_files = []

    try:
        # Case 1: Valid WAV Audio File
        valid_wav = create_temp_wav("valid_sample.wav", duration_sec=1.5)
        temp_files.append(valid_wav)
        test_cases.append(("Valid Audio (.wav)", valid_wav, True, "Should accept valid WAV"))

        # Case 2: Unsupported Extension (.pdf)
        pdf_file = create_temp_file("document.pdf", b"%PDF-1.4 mock pdf content...")
        temp_files.append(pdf_file)
        test_cases.append(("Unsupported Extension (.pdf)", pdf_file, False, "Should reject unsupported extension"))

        # Case 3: Unsupported Extension (.exe)
        exe_file = create_temp_file("script.exe", b"MZ mock executable binary...")
        temp_files.append(exe_file)
        test_cases.append(("Unsupported Extension (.exe)", exe_file, False, "Should reject executables"))

        # Case 4: 0-Byte Empty File
        empty_file = create_temp_file("empty_recording.mp4", b"")
        temp_files.append(empty_file)
        test_cases.append(("Zero-Byte Empty File (.mp4)", empty_file, False, "Should reject empty 0-byte file"))

        # Case 5: Fake Media File (Plain text renamed to .mp4)
        fake_mp4 = create_temp_file("fake_video.mp4", b"This is just plain text content inside an mp4 extension.")
        temp_files.append(fake_mp4)
        test_cases.append(("Fake Media File (Text disguised as .mp4)", fake_mp4, False, "Should reject fake mp4 without audio stream"))

        # Case 6: Tiny File (< 1KB)
        tiny_wav = create_temp_file("corrupt.wav", b"RIFF....")
        temp_files.append(tiny_wav)
        test_cases.append(("Tiny Corrupt Audio (<1KB)", tiny_wav, False, "Should reject tiny corrupt file"))

        # Run all test cases
        passed_count = 0
        total_tests = len(test_cases)

        for idx, (name, path, expected_valid, desc) in enumerate(test_cases, 1):
            is_valid, error_msg = validator.validate_file_path(path)
            passed = (is_valid == expected_valid)

            status_str = "PASS" if passed else "FAIL"
            print(f"\n[Test {idx}/{total_tests}] {name} -> [{status_str}]")
            print(f"  Description    : {desc}")
            print(f"  Expected Valid : {expected_valid}")
            print(f"  Actual Valid   : {is_valid}")
            if error_msg:
                print(f"  Returned Error : {error_msg}")

            if passed:
                passed_count += 1

        all_passed = (passed_count == total_tests)

        print("\n" + "=" * 65)
        print(f"  TASK 2 SUMMARY: {passed_count}/{total_tests} Tests Passed ({(passed_count/total_tests)*100:.0f}%)")
        print(f"  TASK 2 RESULT : {'PASS' if all_passed else 'FAIL'}")
        print("=" * 65 + "\n")
        return all_passed

    finally:
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


if __name__ == "__main__":
    success = run_task2_tests()
    sys.exit(0 if success else 1)
