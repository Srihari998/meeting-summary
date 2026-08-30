"""
tests/test_validator.py
Pytest test suite for Task 2 — File Upload Validation

Tests:
  - Valid audio and video formats (including uppercase extensions)
  - Unsupported/invalid extensions
  - Empty and too-small files
  - Oversized files
  - Files without audio streams
  - Renamed non-media files (fake media)
  - ffprobe unavailable behavior
  - Invalid/malformed ffprobe JSON output
"""

from __future__ import annotations
import json
import os
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from validator import FileValidator, ValidationError, SUPPORTED_EXTENSIONS


# ── Helper Factories ──────────────────────────────────────────────────────────

def make_wav(path: str, duration_sec: float = 1.0) -> str:
    """Create a real WAV file with audible PCM content."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(bytes([0x00, 0x10] * int(16000 * duration_sec)))
    return path


def make_file(path: str, content: bytes = b"X" * 2048) -> str:
    with open(path, "wb") as f:
        f.write(content)
    return path


def temp_path(suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def validator():
    return FileValidator()


@pytest.fixture
def valid_wav(tmp_path):
    p = str(tmp_path / "meeting.wav")
    make_wav(p)
    return p


# ── Supported Format Tests ─────────────────────────────────────────────────────

class TestSupportedExtensions:
    def test_supported_audio_extensions_present(self):
        for ext in [".mp3", ".wav", ".m4a", ".ogg", ".flac"]:
            assert ext in SUPPORTED_EXTENSIONS

    def test_supported_video_extensions_present(self):
        for ext in [".mp4", ".mkv", ".mov", ".avi", ".webm"]:
            assert ext in SUPPORTED_EXTENSIONS


class TestValidExtensions:
    """validator._check_extension must accept all valid lowercase extensions."""

    @pytest.mark.parametrize("filename", [
        "meeting.mp4", "meeting.mkv", "meeting.mov",
        "meeting.avi", "meeting.webm",
        "audio.mp3", "audio.wav", "audio.m4a",
        "audio.ogg", "audio.flac",
    ])
    def test_valid_lowercase(self, validator, filename):
        # Should not raise
        validator._check_extension(filename)

    @pytest.mark.parametrize("filename", [
        "MEETING.MP3", "AUDIO.WAV", "Recording.M4A",
        "Meeting.MP4", "Record.MKV",
    ])
    def test_valid_uppercase(self, validator, filename):
        """Uppercase extensions must be accepted."""
        validator._check_extension(filename)


class TestInvalidExtensions:
    @pytest.mark.parametrize("filename", [
        "doc.txt", "image.jpg", "report.pdf",
        "archive.zip", "program.exe", "data.csv",
    ])
    def test_invalid_extension_rejected(self, validator, filename):
        with pytest.raises(ValidationError, match="Unsupported format"):
            validator._check_extension(filename)

    def test_no_extension_rejected(self, validator):
        with pytest.raises(ValidationError):
            validator._check_extension("noextension")


# ── Size Validation Tests ──────────────────────────────────────────────────────

class TestFileSize:
    def test_too_small_rejected(self, validator):
        with pytest.raises(ValidationError, match="too small"):
            validator._check_size(0, "empty.wav")

    def test_exactly_min_size_accepted(self, validator):
        # Should NOT raise
        validator._check_size(1024, "ok.wav")

    def test_too_large_rejected(self, validator):
        # 501 MB
        with pytest.raises(ValidationError, match="too large"):
            validator._check_size(501 * 1024 * 1024, "huge.mp4")

    def test_exactly_max_size_accepted(self, validator):
        validator._check_size(500 * 1024 * 1024, "maxsize.mp4")


# ── ffprobe Stream Tests ───────────────────────────────────────────────────────

class TestStreamValidation:
    def test_valid_wav_passes(self, validator, valid_wav):
        is_valid, err = validator.validate_file_path(valid_wav)
        assert is_valid, f"Expected valid WAV to pass, got error: {err}"

    def test_zero_byte_file_rejected(self, validator, tmp_path):
        p = str(tmp_path / "empty.wav")
        open(p, "wb").close()
        is_valid, err = validator.validate_file_path(p)
        assert not is_valid
        assert "too small" in err.lower() or "empty" in err.lower()

    def test_renamed_text_file_rejected(self, validator, tmp_path):
        """A text file renamed to .mp4 must be rejected by ffprobe."""
        p = str(tmp_path / "fake.mp4")
        make_file(p, b"This is just plain text, not a media file." * 100)
        is_valid, err = validator.validate_file_path(p)
        assert not is_valid

    def test_corrupted_wav_rejected(self, validator, tmp_path):
        """Garbage bytes with .wav extension must be rejected."""
        p = str(tmp_path / "corrupt.wav")
        make_file(p, b"\x00\xFF\xAB\xCD" * 1024)
        is_valid, err = validator.validate_file_path(p)
        assert not is_valid


class TestFfprobeEdgeCases:
    def test_ffprobe_missing_raises_validation_error(self, validator, tmp_path):
        """When ffprobe is not on PATH, validation must FAIL with a clear message."""
        p = str(tmp_path / "test.wav")
        make_wav(p)
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(ValidationError, match="ffprobe"):
                validator._check_stream_path(p)

    def test_invalid_json_from_ffprobe_rejected(self, validator, tmp_path):
        """If ffprobe returns invalid JSON, the file must be rejected."""
        p = str(tmp_path / "test.wav")
        make_wav(p)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b"NOT VALID JSON {"
        mock_result.stderr = b""
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(ValidationError, match="Unable to validate"):
                validator._check_stream_path(p)

    def test_empty_streams_from_ffprobe_rejected(self, validator, tmp_path):
        """If ffprobe returns empty streams list, the file must be rejected."""
        p = str(tmp_path / "test.wav")
        make_wav(p)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"streams": []}).encode()
        mock_result.stderr = b""
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(ValidationError, match="Unable to validate"):
                validator._check_stream_path(p)

    def test_video_without_audio_stream_rejected(self, validator, tmp_path):
        """A video file with only a video stream (no audio) must be rejected."""
        p = str(tmp_path / "silent_video.mp4")
        make_file(p, b"X" * 2048)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "streams": [{"codec_type": "video"}]
        }).encode()
        mock_result.stderr = b""
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(ValidationError, match="audio stream"):
                validator._check_stream_path(p)

    def test_ffprobe_nonzero_exit_rejected(self, validator, tmp_path):
        """If ffprobe exits with non-zero code, file must be rejected."""
        p = str(tmp_path / "test.wav")
        make_wav(p)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = b""
        mock_result.stderr = b"error"
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(ValidationError, match="Unable to validate"):
                validator._check_stream_path(p)

    def test_audio_stream_present_accepted(self, validator, tmp_path):
        """If ffprobe confirms an audio stream, validation must pass."""
        p = str(tmp_path / "test.wav")
        make_wav(p)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "streams": [{"codec_type": "audio"}]
        }).encode()
        mock_result.stderr = b""
        with patch("subprocess.run", return_value=mock_result):
            # Should NOT raise
            validator._check_stream_path(p)
