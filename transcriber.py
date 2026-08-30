"""
transcriber.py
Wraps openai-whisper: loads a model and transcribes a WAV file,
returning text, segments, and detected language.
"""

from __future__ import annotations
from typing import Any

import whisper


class Transcriber:
    """
    Load a Whisper model once and reuse it for multiple transcriptions.

    Parameters
    ----------
    model_name : Whisper model size — "tiny", "base", "small", "medium", "large".
                 Defaults to "base" (good balance of speed and accuracy).
    device     : "cpu" or "cuda". None lets Whisper pick automatically.
    """

    VALID_MODELS = ("tiny", "base", "small", "medium", "large")

    def __init__(self, model_name: str = "base", device: str | None = None):
        if model_name not in self.VALID_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from: {self.VALID_MODELS}"
            )
        self.model_name = model_name
        self.device = device
        self._model: Any = None  # lazy-loaded

    def _load_model(self) -> None:
        """Download / load model weights (runs once)."""
        if self._model is None:
            kwargs: dict[str, Any] = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = whisper.load_model(self.model_name, **kwargs)

    def transcribe(self, wav_path: str) -> dict[str, Any]:
        """
        Transcribe a WAV file.

        Parameters
        ----------
        wav_path : path to a 16 kHz mono WAV file.

        Returns
        -------
        dict with keys:
            text     (str)  : full transcript
            segments (list) : per-segment dicts with start/end/text
            language (str)  : detected language code (e.g. "en")
        """
        self._load_model()

        result = self._model.transcribe(
            wav_path,
            fp16=False,   # safer on CPU; no-op on GPU
            verbose=False,
        )

        return {
            "text": result.get("text", "").strip(),
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown"),
        }
