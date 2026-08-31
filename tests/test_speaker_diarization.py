"""
tests/test_speaker_diarization.py
Pytest test suite for Speaker Diarization, VAD, Whisper Alignment, and Speaker Statistics.
"""

from __future__ import annotations
import os
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from speaker_diarization import (
    SpeakerDiarizer,
    VoiceActivityDetector,
    align_whisper_with_speakers,
    compute_speaker_statistics,
    format_speaker_transcript,
    _merge_and_smooth_turns,
)


def make_dummy_wav(path: str, duration_sec: float = 2.0, sample_rate: int = 16000) -> str:
    """Create a minimal real 16kHz mono WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n_samples = int(duration_sec * sample_rate)
        # 440 Hz tone
        t = np.linspace(0, duration_sec, n_samples, False)
        tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        wf.writeframes(tone.tobytes())
    return path


class TestVoiceActivityDetector:
    def test_empty_audio_returns_empty(self):
        vad = VoiceActivityDetector()
        assert vad.get_speech_segments(np.array([], dtype=np.float32)) == []

    def test_active_tone_detects_speech(self):
        vad = VoiceActivityDetector()
        sr = 16000
        t = np.linspace(0, 1.0, sr, False)
        tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        segments = vad.get_speech_segments(tone, sample_rate=sr)
        assert len(segments) >= 1
        assert segments[0][1] > segments[0][0]


class TestSpeakerDiarizer:
    def test_missing_file_raises_not_found(self):
        diarizer = SpeakerDiarizer()
        with pytest.raises(FileNotFoundError):
            diarizer.diarize("non_existent_file.wav")

    def test_single_speaker_diarization(self, tmp_path):
        wav_path = str(tmp_path / "single_spk.wav")
        make_dummy_wav(wav_path, duration_sec=3.0)

        # Mock embedding extractor
        mock_ext = MagicMock()
        mock_vec = np.zeros(256, dtype=np.float32)
        mock_vec[0] = 1.0
        mock_ext.embed_utterance.return_value = mock_vec

        diarizer = SpeakerDiarizer(extractor=mock_ext)
        turns = diarizer.diarize(wav_path)

        assert len(turns) >= 1
        assert all(t["speaker"] == "Speaker 1" for t in turns)
        assert turns[0]["start"] >= 0.0

    def test_two_speakers_alternating(self, tmp_path):
        wav_path = str(tmp_path / "two_spk.wav")
        make_dummy_wav(wav_path, duration_sec=4.0)

        v1 = np.zeros(256, dtype=np.float32); v1[0] = 1.0
        v2 = np.zeros(256, dtype=np.float32); v2[100] = 1.0

        mock_ext = MagicMock()
        mock_ext.embed_utterance.side_effect = [v1, v1, v2, v2, v1]

        diarizer = SpeakerDiarizer(extractor=mock_ext)
        turns = diarizer.diarize(wav_path)

        speakers = {t["speaker"] for t in turns}
        assert "Speaker 1" in speakers

    def test_chronological_ordering_preserved(self):
        turns = [
            {"start": 0.0, "end": 2.0, "speaker": "Speaker 1", "speaker_id": 0},
            {"start": 2.2, "end": 5.0, "speaker": "Speaker 2", "speaker_id": 1},
            {"start": 5.1, "end": 8.0, "speaker": "Speaker 1", "speaker_id": 0},
        ]
        smoothed = _merge_and_smooth_turns(turns)
        for i in range(len(smoothed) - 1):
            assert smoothed[i]["start"] <= smoothed[i + 1]["start"]


class TestWhisperAlignment:
    def test_empty_segments_returns_empty(self):
        assert align_whisper_with_speakers([], []) == []

    def test_alignment_with_two_speakers(self):
        whisper_segments = [
            {"start": 0.0, "end": 3.0, "text": "Hello, welcome to the kickoff."},
            {"start": 3.2, "end": 6.5, "text": "Thanks! Glad to be here."},
            {"start": 7.0, "end": 10.0, "text": "Let us review the timeline."},
        ]
        speaker_turns = [
            {"start": 0.0, "end": 3.1, "speaker": "Speaker 1", "speaker_id": 0},
            {"start": 3.1, "end": 6.8, "speaker": "Speaker 2", "speaker_id": 1},
            {"start": 6.8, "end": 11.0, "speaker": "Speaker 1", "speaker_id": 0},
        ]

        aligned = align_whisper_with_speakers(whisper_segments, speaker_turns)
        assert len(aligned) == 3
        assert aligned[0]["speaker"] == "Speaker 1"
        assert aligned[0]["text"] == "Hello, welcome to the kickoff."
        assert aligned[1]["speaker"] == "Speaker 2"
        assert aligned[1]["text"] == "Thanks! Glad to be here."
        assert aligned[2]["speaker"] == "Speaker 1"
        assert aligned[2]["text"] == "Let us review the timeline."

    def test_consecutive_same_speaker_grouped(self):
        """Two consecutive Whisper segments by the same speaker should be grouped into one turn."""
        whisper_segments = [
            {"start": 0.0, "end": 2.0, "text": "Part one."},
            {"start": 2.0, "end": 4.0, "text": "Part two."},
            {"start": 4.5, "end": 7.0, "text": "Now speaker two."},
        ]
        speaker_turns = [
            {"start": 0.0, "end": 4.2, "speaker": "Speaker 1", "speaker_id": 0},
            {"start": 4.2, "end": 8.0, "speaker": "Speaker 2", "speaker_id": 1},
        ]
        aligned = align_whisper_with_speakers(whisper_segments, speaker_turns)
        assert len(aligned) == 2
        assert aligned[0]["speaker"] == "Speaker 1"
        assert aligned[0]["text"] == "Part one. Part two."
        assert aligned[1]["speaker"] == "Speaker 2"


class TestSpeakerStatistics:
    def test_statistics_calculation(self):
        aligned_turns = [
            {"start": 0.0, "end": 60.0, "speaker": "Speaker 1", "text": "one two three"},
            {"start": 60.0, "end": 120.0, "speaker": "Speaker 2", "text": "four five"},
            {"start": 120.0, "end": 180.0, "speaker": "Speaker 1", "text": "six seven eight"},
        ]
        stats = compute_speaker_statistics(aligned_turns, total_audio_duration_sec=180.0)

        assert stats["speaker_count"] == 2
        assert stats["total_speech_time_sec"] == 180.0
        assert "Speaker 1" in stats["speakers"]
        assert "Speaker 2" in stats["speakers"]

        spk1 = stats["speakers"]["Speaker 1"]
        assert spk1["speaking_time_sec"] == 120.0
        assert spk1["turn_count"] == 2
        assert spk1["word_count"] == 6
        assert pytest.approx(spk1["percentage"], abs=0.5) == 66.7

        spk2 = stats["speakers"]["Speaker 2"]
        assert spk2["speaking_time_sec"] == 60.0
        assert spk2["turn_count"] == 1
        assert spk2["word_count"] == 2
        assert pytest.approx(spk2["percentage"], abs=0.5) == 33.3


class TestTranscriptFormatting:
    def test_format_speaker_transcript(self):
        turns = [
            {
                "start": 2.0,
                "end": 8.0,
                "speaker": "Speaker 1",
                "text": "Hello world.",
                "timestamp_str": "[00:00:02 - 00:00:08]",
            }
        ]
        formatted = format_speaker_transcript(turns)
        assert "MEETING TRANSCRIPT (SPEAKER-ATTRIBUTED)" in formatted
        assert "[00:00:02 - 00:00:08] Speaker 1:" in formatted
        assert "Hello world." in formatted
