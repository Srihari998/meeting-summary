"""
audio_processor.py
Extracts and converts audio from any video/audio file to a
16 kHz mono WAV suitable for Whisper.
"""

import os
import subprocess
import tempfile
from pathlib import Path


class AudioProcessor:
    """Handles audio extraction and conversion via ffmpeg."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

    def _check_ffmpeg(self) -> None:
        """Raise RuntimeError if ffmpeg is not on PATH."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                raise RuntimeError("ffmpeg returned a non-zero exit code.")
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install it and make sure it is on your PATH.\n"
                "  Windows : winget install ffmpeg\n"
                "  macOS   : brew install ffmpeg\n"
                "  Linux   : sudo apt install ffmpeg"
            )

    def process(self, input_path: str, output_path: str | None = None) -> str:
        """
        Convert *input_path* (any audio/video) to a 16 kHz mono WAV.

        Parameters
        ----------
        input_path  : path to the source file
        output_path : optional destination WAV path.
                      If None a temp file is created and its path returned.

        Returns
        -------
        str : absolute path to the produced WAV file.
        """
        self._check_ffmpeg()

        input_path = str(Path(input_path).resolve())
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            # Create a named temp file that survives after close
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, prefix="whisper_audio_"
            )
            output_path = tmp.name
            tmp.close()

        output_path = str(Path(output_path).resolve())

        cmd = [
            "ffmpeg",
            "-y",                        # overwrite without asking
            "-i", input_path,            # input file
            "-vn",                       # drop video stream
            "-acodec", "pcm_s16le",      # 16-bit PCM
            "-ar", str(self.sample_rate),# sample rate
            "-ac", str(self.channels),   # channels (1 = mono)
            output_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            error_msg = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ffmpeg conversion failed (exit {result.returncode}):\n{error_msg}"
            )

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(
                f"ffmpeg ran without error but output file is missing or empty: {output_path}"
            )

        return output_path
