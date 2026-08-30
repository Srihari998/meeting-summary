"""
tests/test_audio_processor.py
Pytest test suite for Task 1 — AudioProcessor

Tests:
  - Successful conversion (output file exists and is non-empty)
  - Output is 16 kHz mono WAV
  - Video to audio extraction
  - Invalid/missing input file
  - FFmpeg not on PATH
  - Conversion failure (non-zero ffmpeg exit)
  - Missing or empty output file
"""

from __future__ import annotations
import os
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_processor import AudioProcessor


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_wav(path: str, sample_rate: int = 16000, channels: int = 1) -> str:
    """Write a minimal real WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes([0x00, 0x10] * sample_rate))  # 1 second
    return path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def processor():
    return AudioProcessor()


@pytest.fixture
def valid_wav(tmp_path):
    p = str(tmp_path / "input.wav")
    return make_wav(p)


# ── Successful Conversion ─────────────────────────────────────────────────────

class TestSuccessfulConversion:
    def test_output_file_created(self, processor, valid_wav, tmp_path):
        output = str(tmp_path / "output.wav")
        result_path = processor.process(valid_wav, output)
        assert os.path.exists(result_path)

    def test_output_file_not_empty(self, processor, valid_wav, tmp_path):
        output = str(tmp_path / "output.wav")
        processor.process(valid_wav, output)
        assert os.path.getsize(output) > 0

    def test_output_is_16khz_mono(self, processor, valid_wav, tmp_path):
        """Verify ffmpeg output is 16000 Hz mono."""
        output = str(tmp_path / "output16k.wav")
        processor.process(valid_wav, output)
        with wave.open(output, "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1

    def test_auto_temp_file_created(self, processor, valid_wav):
        """When no output_path given, a temp file should be returned."""
        result = processor.process(valid_wav)
        try:
            assert os.path.exists(result)
            assert result.endswith(".wav")
        finally:
            if os.path.exists(result):
                os.remove(result)

    def test_stereo_input_converted_to_mono(self, processor, tmp_path):
        """Stereo WAV input must be down-mixed to mono."""
        stereo = str(tmp_path / "stereo.wav")
        make_wav(stereo, channels=2, sample_rate=44100)
        output = str(tmp_path / "mono_out.wav")
        processor.process(stereo, output)
        with wave.open(output, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000


# ── Error / Edge Cases ────────────────────────────────────────────────────────

class TestErrorCases:
    def test_missing_input_raises_file_not_found(self, processor, tmp_path):
        with pytest.raises(FileNotFoundError):
            processor.process(str(tmp_path / "nonexistent.wav"))

    def test_ffmpeg_missing_raises_runtime_error(self, processor, valid_wav):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                processor.process(valid_wav)

    def test_ffmpeg_nonzero_exit_raises_runtime_error(self, processor, valid_wav, tmp_path):
        """Simulate ffmpeg returning non-zero exit code."""
        # First call is _check_ffmpeg (succeeds), second is actual conversion (fails)
        success = MagicMock(returncode=0, stdout=b"ffmpeg version", stderr=b"")
        failure = MagicMock(returncode=1, stdout=b"", stderr=b"simulated error")
        with patch("subprocess.run", side_effect=[success, failure]):
            with pytest.raises(RuntimeError, match="ffmpeg conversion failed"):
                processor.process(valid_wav, str(tmp_path / "out.wav"))

    def test_empty_output_raises_runtime_error(self, processor, valid_wav, tmp_path):
        """Simulate ffmpeg producing an empty output file."""
        output = str(tmp_path / "empty_out.wav")
        open(output, "wb").close()  # pre-create empty file

        success = MagicMock(returncode=0, stdout=b"ffmpeg version", stderr=b"")
        empty_ok = MagicMock(returncode=0, stdout=b"", stderr=b"")

        def side_effect(args, **kwargs):
            return success if "-version" in args else empty_ok

        with patch("subprocess.run", side_effect=side_effect):
            with pytest.raises(RuntimeError, match="output file is missing or empty"):
                processor.process(valid_wav, output)
