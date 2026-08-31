"""
speaker_diarization.py
Speaker Diarization & Whisper Alignment Module

Pipeline:
  16kHz WAV
  → Voice Activity Detection (VAD)
  → Windowed Voice Segments (2.0s window, 1.0s step)
  → Speaker Embeddings (256-d d-vectors via ResNet VoiceEncoder)
  → Speaker Clustering (Agglomerative + Centroid Merging + Satellite Pruning)
  → Chronological Turn Smoothing & Micro-Turn Absorption
  → OpenAI Whisper Segment Alignment
  → Structured Speaker Transcript & Analytics Cards

Supports optional pyannote pipeline if Hugging Face token is provided,
with full automatic fallback to the local offline embedding engine.
"""

from __future__ import annotations

import logging
import os
import wave
from typing import Any

import numpy as np

from speaker_embeddings import (
    SpeakerEmbeddingExtractor,
    cluster_embeddings,
    format_speaker_id,
)

logger = logging.getLogger(__name__)


def _format_seconds_to_timestamp(seconds: float) -> str:
    """Format float seconds to [HH:MM:SS] format."""
    total_sec = max(0, int(round(seconds)))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to 'Xm Ys' or 'Xs'."""
    sec = max(0, int(round(seconds)))
    mins = sec // 60
    rem_sec = sec % 60
    if mins > 0:
        return f"{mins}m {rem_sec:02d}s"
    return f"{rem_sec}s"


class VoiceActivityDetector:
    """Detects active speech frames and removes silence."""

    def __init__(self, mode: int = 2) -> None:
        self.mode = mode
        self._vad = None
        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(mode)
        except Exception:
            self._vad = None

    def get_speech_segments(
        self,
        wav_pcm: np.ndarray,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        padding_duration_ms: int = 300,
    ) -> list[tuple[float, float]]:
        """
        Identify contiguous speech intervals (start_sec, end_sec) in 16kHz mono PCM.
        """
        if wav_pcm is None or len(wav_pcm) == 0:
            return []

        # Convert to 16-bit integer PCM for webrtcvad
        wav_float = np.asarray(wav_pcm, dtype=np.float32)
        if np.max(np.abs(wav_float)) <= 1.0:
            wav_int16 = (wav_float * 32767).astype(np.int16)
        else:
            wav_int16 = wav_float.astype(np.int16)

        frame_bytes = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        raw_bytes = wav_int16.tobytes()
        n_frames = len(raw_bytes) // frame_bytes

        if n_frames == 0:
            return [(0.0, len(wav_float) / sample_rate)]

        is_speech_list: list[bool] = []
        for i in range(n_frames):
            chunk = raw_bytes[i * frame_bytes : (i + 1) * frame_bytes]
            if self._vad is not None:
                try:
                    is_speech_list.append(self._vad.is_speech(chunk, sample_rate))
                except Exception:
                    # Energy fallback
                    chunk_arr = np.frombuffer(chunk, dtype=np.int16)
                    rms = np.sqrt(np.mean(chunk_arr.astype(np.float32) ** 2))
                    is_speech_list.append(rms > 200)
            else:
                chunk_arr = np.frombuffer(chunk, dtype=np.int16)
                rms = np.sqrt(np.mean(chunk_arr.astype(np.float32) ** 2))
                is_speech_list.append(rms > 200)

        # Smooth and group active frames
        step_sec = frame_duration_ms / 1000.0
        padding_frames = max(1, int(padding_duration_ms / frame_duration_ms))

        segments: list[tuple[float, float]] = []
        in_speech = False
        start_frame = 0
        silence_count = 0

        for idx, is_spk in enumerate(is_speech_list):
            if is_spk:
                if not in_speech:
                    in_speech = True
                    start_frame = max(0, idx - padding_frames)
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= padding_frames:
                        in_speech = False
                        end_frame = idx
                        start_sec = start_frame * step_sec
                        end_sec = end_frame * step_sec
                        if end_sec - start_sec >= 0.25:  # at least 250ms
                            segments.append((start_sec, end_sec))

        if in_speech:
            segments.append((start_frame * step_sec, n_frames * step_sec))

        if not segments:
            total_dur = len(wav_float) / sample_rate
            if total_dur > 0:
                segments = [(0.0, total_dur)]

        return segments


class SpeakerDiarizer:
    """Performs speaker diarization on 16kHz audio."""

    def __init__(
        self,
        extractor: SpeakerEmbeddingExtractor | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.extractor = extractor or SpeakerEmbeddingExtractor()
        self.hf_token = hf_token
        self.vad = VoiceActivityDetector(mode=2)

    def diarize(
        self,
        wav_path: str,
        num_speakers: int | None = None,
        min_speakers: int = 1,
        max_speakers: int = 10,
        window_size_sec: float = 2.0,
        window_step_sec: float = 1.0,
    ) -> list[dict[str, Any]]:
        """
        Run speaker diarization on a 16kHz mono WAV file.

        Returns:
            list of chronologically sorted speaker turn dicts:
            [
              {"start": 0.0, "end": 4.5, "speaker": "Speaker 1", "speaker_id": 0},
              {"start": 4.5, "end": 9.2, "speaker": "Speaker 2", "speaker_id": 1},
              ...
            ]
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        # Try pyannote pipeline if user provided HF token
        if self.hf_token and self.hf_token.strip():
            try:
                pyannote_res = self._try_pyannote(wav_path, num_speakers)
                if pyannote_res:
                    logger.info("Successfully ran pyannote.audio pipeline")
                    return pyannote_res
            except Exception as exc:
                logger.warning(
                    "pyannote diarization failed (%s), falling back to local embedding engine", exc
                )

        # ── Local Pretrained Embedding Engine ────────────────────────────────
        return self._diarize_local(
            wav_path=wav_path,
            num_speakers=num_speakers,
            window_size_sec=window_size_sec,
            window_step_sec=window_step_sec,
        )

    def _diarize_local(
        self,
        wav_path: str,
        num_speakers: int | None = None,
        window_size_sec: float = 2.0,
        window_step_sec: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Diarize using local VAD + VoiceEncoder embeddings + Multi-stage Agglomerative Clustering."""
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)

        if n_frames == 0:
            return []

        audio_pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        total_duration = len(audio_pcm) / sr

        # 1. Voice Activity Detection
        speech_intervals = self.vad.get_speech_segments(audio_pcm, sample_rate=sr, padding_duration_ms=300)
        if not speech_intervals:
            speech_intervals = [(0.0, total_duration)]

        # 2. Extract sliding-window segments across speech intervals
        window_samples = int(window_size_sec * sr)
        step_samples = int(window_step_sec * sr)

        window_intervals: list[tuple[float, float]] = []
        embeddings: list[np.ndarray] = []

        for start_sec, end_sec in speech_intervals:
            start_idx = int(start_sec * sr)
            end_idx = int(end_sec * sr)
            seg_len = end_idx - start_idx

            if seg_len < int(0.5 * sr):  # skip tiny clicks (<500ms)
                continue

            if seg_len <= window_samples:
                clip = audio_pcm[start_idx:end_idx]
                emb = self.extractor.embed_utterance(clip, sample_rate=sr)
                if np.linalg.norm(emb) > 0:
                    embeddings.append(emb)
                    window_intervals.append((start_sec, end_sec))
            else:
                # Slide window
                for w_start in range(start_idx, end_idx - window_samples // 2, step_samples):
                    w_end = min(w_start + window_samples, end_idx)
                    clip = audio_pcm[w_start:w_end]
                    emb = self.extractor.embed_utterance(clip, sample_rate=sr)
                    if np.linalg.norm(emb) > 0:
                        embeddings.append(emb)
                        window_intervals.append((w_start / sr, w_end / sr))

        if not embeddings or not window_intervals:
            # Fallback single speaker
            return [{
                "start": 0.0,
                "end": round(total_duration, 2),
                "speaker": "Speaker 1",
                "speaker_id": 0,
            }]

        # 3. Enhanced Multi-stage Clustering
        cluster_ids = cluster_embeddings(
            embeddings,
            threshold=0.80,
            num_speakers=num_speakers,
            merge_similarity=0.88,
            min_cluster_samples=3,
        )

        # 4. Merge consecutive windows with the same speaker label
        raw_turns: list[dict[str, Any]] = []
        for (w_start, w_end), cid in zip(window_intervals, cluster_ids):
            raw_turns.append({
                "start": w_start,
                "end": w_end,
                "speaker": format_speaker_id(cid),
                "speaker_id": cid,
            })

        return _merge_and_smooth_turns(raw_turns, max_gap_sec=0.8)

    def _try_pyannote(self, wav_path: str, num_speakers: int | None) -> list[dict[str, Any]] | None:
        """Attempt pyannote.audio pipeline if HF token is valid."""
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token,
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        diarization = pipeline(
            wav_path,
            num_speakers=num_speakers,
        )

        turns: list[dict[str, Any]] = []
        spk_map: dict[str, str] = {}
        spk_id_map: dict[str, int] = {}
        next_id = 0

        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if speaker_label not in spk_map:
                spk_map[speaker_label] = f"Speaker {next_id + 1}"
                spk_id_map[speaker_label] = next_id
                next_id += 1

            turns.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": spk_map[speaker_label],
                "speaker_id": spk_id_map[speaker_label],
            })

        return _merge_and_smooth_turns(turns, max_gap_sec=0.8)


def _merge_and_smooth_turns(
    turns: list[dict[str, Any]], max_gap_sec: float = 0.8
) -> list[dict[str, Any]]:
    """Merge contiguous turns of the same speaker and absorb sub-second micro-turns."""
    if not turns:
        return []

    # Sort strictly by start time
    sorted_turns = sorted(turns, key=lambda x: (x["start"], x["end"]))
    merged: list[dict[str, Any]] = []

    current = dict(sorted_turns[0])

    for nxt in sorted_turns[1:]:
        if (
            nxt["speaker_id"] == current["speaker_id"]
            and nxt["start"] <= current["end"] + max_gap_sec
        ):
            # Extend current turn
            current["end"] = max(current["end"], nxt["end"])
        else:
            merged.append({
                "start": round(current["start"], 2),
                "end": round(current["end"], 2),
                "speaker": current["speaker"],
                "speaker_id": current["speaker_id"],
            })
            current = dict(nxt)

    merged.append({
        "start": round(current["start"], 2),
        "end": round(current["end"], 2),
        "speaker": current["speaker"],
        "speaker_id": current["speaker_id"],
    })

    # Second pass: absorb isolated micro-turns (< 0.6s) into surrounding turns if flanked by same speaker
    if len(merged) >= 3:
        smoothed: list[dict[str, Any]] = [merged[0]]
        i = 1
        while i < len(merged) - 1:
            prev_t = smoothed[-1]
            curr_t = merged[i]
            next_t = merged[i + 1]
            curr_dur = curr_t["end"] - curr_t["start"]

            # If isolated micro-turn between two identical speaker turns
            if curr_dur < 0.6 and prev_t["speaker_id"] == next_t["speaker_id"]:
                # Absorb into prev_t
                prev_t["end"] = next_t["end"]
                i += 2  # skip curr and next (next is absorbed)
            else:
                smoothed.append(curr_t)
                i += 1

        if i < len(merged):
            smoothed.append(merged[i])
        return smoothed

    return merged


def align_whisper_with_speakers(
    whisper_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Align Whisper timestamped segments with diarization speaker turns.
    Assigns each Whisper segment to the dominant speaker in that time window.
    Consecutive Whisper segments from the same speaker are grouped into unified turns.

    Returns:
        list of speaker-attributed turns:
        [
          {
            "start": 2.1,
            "end": 8.5,
            "speaker": "Speaker 1",
            "speaker_id": 0,
            "text": "Good morning everyone and welcome.",
            "timestamp_str": "[00:00:02 - 00:00:08]"
          },
          ...
        ]
    """
    if not whisper_segments:
        return []

    if not speaker_turns:
        full_text = " ".join(s.get("text", "").strip() for s in whisper_segments)
        start = whisper_segments[0].get("start", 0.0)
        end = whisper_segments[-1].get("end", 0.0)
        return [{
            "start": start,
            "end": end,
            "speaker": "Speaker 1",
            "speaker_id": 0,
            "text": full_text,
            "timestamp_str": f"[{_format_seconds_to_timestamp(start)} - {_format_seconds_to_timestamp(end)}]",
        }]

    aligned_items: list[dict[str, Any]] = []

    for w_seg in whisper_segments:
        w_start = float(w_seg.get("start", 0.0))
        w_end = float(w_seg.get("end", w_start + 0.1))
        w_text = w_seg.get("text", "").strip()
        if not w_text:
            continue

        # Find overlapping speaker turns
        overlap_scores: dict[int, float] = {}
        for spk_turn in speaker_turns:
            s_start = spk_turn["start"]
            s_end = spk_turn["end"]
            overlap_start = max(w_start, s_start)
            overlap_end = min(w_end, s_end)
            if overlap_end > overlap_start:
                overlap_dur = overlap_end - overlap_start
                cid = spk_turn["speaker_id"]
                overlap_scores[cid] = overlap_scores.get(cid, 0.0) + overlap_dur

        if overlap_scores:
            best_cid = max(overlap_scores, key=overlap_scores.get)
        else:
            distances = [
                (min(abs(w_start - st["end"]), abs(w_end - st["start"])), st["speaker_id"])
                for st in speaker_turns
            ]
            best_cid = min(distances, key=lambda x: x[0])[1]

        aligned_items.append({
            "start": w_start,
            "end": w_end,
            "speaker": format_speaker_id(best_cid),
            "speaker_id": best_cid,
            "text": w_text,
        })

    if not aligned_items:
        return []

    # Group adjacent segments with the same speaker
    grouped: list[dict[str, Any]] = []
    curr = dict(aligned_items[0])
    curr_texts = [curr["text"]]

    for nxt in aligned_items[1:]:
        if nxt["speaker_id"] == curr["speaker_id"]:
            curr["end"] = max(curr["end"], nxt["end"])
            curr_texts.append(nxt["text"])
        else:
            curr["text"] = " ".join(curr_texts)
            curr["timestamp_str"] = (
                f"[{_format_seconds_to_timestamp(curr['start'])} - {_format_seconds_to_timestamp(curr['end'])}]"
            )
            grouped.append(curr)
            curr = dict(nxt)
            curr_texts = [curr["text"]]

    curr["text"] = " ".join(curr_texts)
    curr["timestamp_str"] = (
        f"[{_format_seconds_to_timestamp(curr['start'])} - {_format_seconds_to_timestamp(curr['end'])}]"
    )
    grouped.append(curr)

    return grouped


def compute_speaker_statistics(
    aligned_turns: list[dict[str, Any]],
    total_audio_duration_sec: float | None = None,
) -> dict[str, Any]:
    """
    Calculate per-speaker speaking time, percentage of total speech, and turn counts.
    """
    if not aligned_turns:
        return {
            "speaker_count": 0,
            "total_speech_time_sec": 0.0,
            "total_speech_time_formatted": "0s",
            "speakers": {},
        }

    speakers_data: dict[str, dict[str, Any]] = {}
    total_speech_sec = 0.0

    for turn in aligned_turns:
        spk = turn["speaker"]
        dur = max(0.0, turn["end"] - turn["start"])
        words = len(turn.get("text", "").split())
        total_speech_sec += dur

        if spk not in speakers_data:
            speakers_data[spk] = {
                "speaking_time_sec": 0.0,
                "turn_count": 0,
                "word_count": 0,
            }
        speakers_data[spk]["speaking_time_sec"] += dur
        speakers_data[spk]["turn_count"] += 1
        speakers_data[spk]["word_count"] += words

    benchmark_total = (
        total_speech_sec if total_speech_sec > 0 else (total_audio_duration_sec or 1.0)
    )

    formatted_speakers: dict[str, dict[str, Any]] = {}
    for spk, data in speakers_data.items():
        dur_sec = round(data["speaking_time_sec"], 1)
        pct = (
            round((dur_sec / benchmark_total) * 100.0, 1)
            if benchmark_total > 0
            else 0.0
        )
        formatted_speakers[spk] = {
            "speaking_time_sec": dur_sec,
            "speaking_time_formatted": _format_duration(dur_sec),
            "percentage": pct,
            "turn_count": data["turn_count"],
            "word_count": data["word_count"],
        }

    return {
        "speaker_count": len(formatted_speakers),
        "total_speech_time_sec": round(total_speech_sec, 1),
        "total_speech_time_formatted": _format_duration(total_speech_sec),
        "speakers": formatted_speakers,
    }


def format_speaker_transcript(aligned_turns: list[dict[str, Any]]) -> str:
    """Format aligned turns into clean text transcript with speaker headers and timestamps."""
    lines = [
        "=" * 60,
        "          MEETING TRANSCRIPT (SPEAKER-ATTRIBUTED)",
        "=" * 60,
        "",
    ]
    for turn in aligned_turns:
        ts_str = turn.get(
            "timestamp_str",
            f"[{_format_seconds_to_timestamp(turn['start'])} - {_format_seconds_to_timestamp(turn['end'])}]",
        )
        lines.append(f"{ts_str} {turn['speaker']}:")
        lines.append(f"  {turn['text']}")
        lines.append("")

    return "\n".join(lines)
