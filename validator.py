"""
validator.py
Task 2 -- File Upload Validation

Validates uploaded audio/video files before entering the processing pipeline:
- Validates file extension against allowed audio/video formats (case-insensitive).
- Checks file size boundaries (min 1KB, max 500MB).
- Uses ffprobe to strictly verify container integrity and presence of an audio stream.
- Rejects non-media files disguised with media extensions.
- Fails explicitly with helpful guidance if ffprobe is not installed on PATH.
"""

from __future__ import annotations
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO


# ── Supported Formats ─────────────────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

MAX_FILE_SIZE_MB = 500
MIN_FILE_SIZE_BYTES = 1_024  # 1 KB


class ValidationError(Exception):
    """Raised when an uploaded file fails validation checks."""
    pass


class FileValidator:
    """
    Performs comprehensive pre-transcription validation for media files.
    """

    def __init__(
        self,
        max_size_mb: int = MAX_FILE_SIZE_MB,
        min_size_bytes: int = MIN_FILE_SIZE_BYTES,
    ) -> None:
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.min_size_bytes = min_size_bytes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_streamlit_upload(self, uploaded_file) -> tuple[bool, str]:
        """
        Validate a Streamlit UploadedFile object.
        Memory-efficient: streams file in chunks rather than loading all at once.

        Returns
        -------
        (True, "")          if valid
        (False, error_msg)  if invalid
        """
        try:
            self._check_extension(uploaded_file.name)
            self._check_size(uploaded_file.size, uploaded_file.name)
            self._check_stream_via_ffprobe(uploaded_file)
            return True, ""
        except ValidationError as exc:
            return False, str(exc)

    def validate_file_path(self, path: str) -> tuple[bool, str]:
        """
        Validate a media file located at a filesystem path.

        Returns
        -------
        (True, "")          if valid
        (False, error_msg)  if invalid
        """
        try:
            p = Path(path)
            if not p.exists():
                raise ValidationError(f"File not found: {path}")
            self._check_extension(p.name)
            self._check_size(p.stat().st_size, p.name)
            self._check_stream_path(str(p))
            return True, ""
        except ValidationError as exc:
            return False, str(exc)

    # ------------------------------------------------------------------
    # Internal Validation Steps
    # ------------------------------------------------------------------

    def _check_extension(self, filename: str) -> None:
        """Verify extension is in the supported whitelist (case-insensitive)."""
        ext = Path(filename).suffix.lower()
        if not ext:
            raise ValidationError(
                f"'{filename}' has no file extension. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported format '{ext}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    def _check_size(self, size_bytes: int, filename: str) -> None:
        """Verify file meets minimum and maximum size boundaries."""
        if size_bytes < self.min_size_bytes:
            raise ValidationError(
                f"'{filename}' is too small ({size_bytes} bytes). "
                "The file may be empty or corrupt."
            )
        if size_bytes > self.max_size_bytes:
            size_mb = size_bytes / (1024 * 1024)
            max_mb = self.max_size_bytes // (1024 * 1024)
            raise ValidationError(
                f"'{filename}' is too large ({size_mb:.1f} MB). "
                f"Maximum allowed file size is {max_mb} MB."
            )

    def _check_stream_via_ffprobe(self, uploaded_file) -> None:
        """Write stream chunk-by-chunk to temp file and validate with ffprobe."""
        suffix = Path(uploaded_file.name).suffix or ".tmp"
        tmp_path = None
        try:
            uploaded_file.seek(0)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="validate_"
            ) as tmp:
                while chunk := uploaded_file.read(1024 * 1024):
                    tmp.write(chunk)
                tmp_path = tmp.name

            # Reset file pointer for subsequent application usage
            uploaded_file.seek(0)
            self._check_stream_path(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _check_stream_path(path: str) -> None:
        """
        Run ffprobe to strictly inspect media streams.
        - Fails if ffprobe is missing.
        - Fails if ffprobe reports exit error or invalid JSON structure.
        - Fails if no audio stream is present.
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_streams",
                    path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise ValidationError(
                "ffprobe is not available. Please install FFmpeg and ensure "
                "both 'ffmpeg' and 'ffprobe' are added to your system PATH."
            )

        if result.returncode != 0:
            raise ValidationError(
                "Unable to validate the media file. The file may be corrupted or unsupported."
            )

        try:
            raw_json = result.stdout.decode("utf-8", errors="replace")
            info = json.loads(raw_json)
            streams = info.get("streams")
            if not isinstance(streams, list) or len(streams) == 0:
                raise ValidationError(
                    "Unable to validate the media file. The file may be corrupted or unsupported."
                )

            codec_types = [s.get("codec_type", "") for s in streams if isinstance(s, dict)]
            if "audio" not in codec_types:
                raise ValidationError(
                    "The file does not contain any audio stream. "
                    "Please upload a file with audio (e.g., an actual meeting recording, not a silent video)."
                )
        except (json.JSONDecodeError, AttributeError):
            raise ValidationError(
                "Unable to validate the media file. The file may be corrupted or unsupported."
            )
