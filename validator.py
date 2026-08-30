"""
validator.py
Task 2 -- File Upload Validation

Validates uploaded audio/video files:
- Checks file extension is supported
- Checks file size (not empty, not too large)
- Uses ffprobe to confirm the file contains a valid audio/video stream
- Returns clear, user-friendly error messages
"""

from __future__ import annotations
import subprocess
import json
import tempfile
import os
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm",   # video
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",   # audio
}

MAX_FILE_SIZE_MB = 500          # reject files larger than 500 MB
MIN_FILE_SIZE_BYTES = 1_024     # reject files smaller than 1 KB (likely corrupt)


class ValidationError(Exception):
    """Raised when a file fails validation."""
    pass


class FileValidator:
    """
    Validates an uploaded file before it enters the transcription pipeline.

    Usage
    -----
    validator = FileValidator()
    is_valid, error_msg = validator.validate_streamlit_upload(uploaded_file)
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
        Validate a file already on disk (used by the CLI test script).

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
    # Private checks
    # ------------------------------------------------------------------

    def _check_extension(self, filename: str) -> None:
        ext = Path(filename).suffix.lower()
        if not ext:
            raise ValidationError(
                f"'{filename}' has no file extension. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported format '{ext}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    def _check_size(self, size_bytes: int, filename: str) -> None:
        if size_bytes < self.min_size_bytes:
            raise ValidationError(
                f"'{filename}' is too small ({size_bytes} bytes). "
                "The file may be empty or corrupt."
            )
        if size_bytes > self.max_size_bytes:
            size_mb = size_bytes / (1024 * 1024)
            raise ValidationError(
                f"'{filename}' is too large ({size_mb:.1f} MB). "
                f"Maximum allowed size is {self.max_size_bytes // (1024*1024)} MB."
            )

    def _check_stream_via_ffprobe(self, uploaded_file) -> None:
        """Write upload to a temp file, then run ffprobe on it."""
        suffix = Path(uploaded_file.name).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="validate_"
        ) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            self._check_stream_path(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _check_stream_path(path: str) -> None:
        """
        Run ffprobe to verify the file contains at least one audio stream.
        Raises ValidationError if ffprobe cannot find an audio stream.
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
            # ffprobe not available — skip deep validation, pass on extension only
            return

        if result.returncode != 0:
            raise ValidationError(
                "The file could not be read by ffprobe. "
                "It may be corrupt or not a valid audio/video file."
            )

        try:
            info = json.loads(result.stdout.decode("utf-8", errors="replace"))
            streams = info.get("streams", [])
            codec_types = [s.get("codec_type", "") for s in streams]
            if "audio" not in codec_types:
                raise ValidationError(
                    "The file does not contain any audio stream. "
                    "Please upload a file with audio (e.g., an actual meeting recording, "
                    "not a silent video)."
                )
        except (json.JSONDecodeError, KeyError):
            # Could not parse ffprobe output — assume valid
            pass
