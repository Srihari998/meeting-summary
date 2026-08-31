"""
app.py
Meeting Transcription, Speaker Diarization & Executive Summary Tool

Pipeline:
  Upload
  → Validate
  → Extract Audio (16kHz mono WAV via FFmpeg)
  → VAD & Voice Activity Detection
  → Speaker Embeddings & Voice Clustering (Anonymous: Speaker 1, 2, ...)
  → OpenAI Whisper Transcription
  → Speaker-Segment Alignment
  → Transcript Validation
  → Topic Classification & Executive Summary
  → Speaker Analytics & Multi-Format Export
"""

from __future__ import annotations

import datetime
import html
import json
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Any

import streamlit as st

from accuracy import calculate_wer_and_metrics
from audio_processor import AudioProcessor
from speaker_diarization import (
    SpeakerDiarizer,
    align_whisper_with_speakers,
    compute_speaker_statistics,
    format_speaker_transcript,
)
from summarizer import MeetingSummarizer
from transcriber import Transcriber
from validator import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, FileValidator

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Summarizer & Speaker Diarization",
    page_icon="🎙️",
    layout="wide",
)

# ── Transcripts Directory ─────────────────────────────────────────────────────
TRANSCRIPT_DIR = Path(__file__).parent / "transcripts"
TRANSCRIPT_DIR.mkdir(exist_ok=True)


# ── Model Caches (survive re-runs) ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_transcriber(model_name: str) -> Transcriber:
    """Cached Whisper model."""
    return Transcriber(model_name=model_name)


@st.cache_resource(show_spinner=False)
def get_diarizer(hf_token: str | None = None) -> SpeakerDiarizer:
    """Cached Speaker Diarizer (d-vector voice encoder + VAD)."""
    return SpeakerDiarizer(hf_token=hf_token)


# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stFileUploader"] {
        border: 2px dashed #2563EB; border-radius: 12px;
        padding: 24px 16px; background: #F8FAFC;
        transition: border-color .2s ease, background .2s ease;
        text-align: center;
    }
    [data-testid="stFileUploader"]:hover { border-color: #1D4ED8; background: #EFF6FF; }
    [data-testid="stFileUploader"] label { font-size:1rem; font-weight:600; color:#1E293B; }
    [data-testid="stFileUploader"] button {
        background-color:#2563EB !important; color:white !important;
        border-radius:8px !important; border:none !important;
        padding:8px 20px !important; font-weight:600 !important;
    }
    div.stButton > button[kind="primary"] {
        width:100%; padding:12px; font-size:1.05rem;
        border-radius:10px; font-weight:700; background-color:#2563EB;
    }
    .overview-card {
        background:#F0FDF4; border-left:4px solid #16A34A;
        border-radius:0 10px 10px 0; padding:16px 20px;
        margin-bottom:20px; color:#14532D; font-size:1.05rem; line-height:1.6;
    }
    .topic-card {
        background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px;
        padding:16px; margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,.05);
    }
    .topic-header { font-size:1.05rem; font-weight:700; color:#1E3A8A; margin-bottom:8px; }
    .action-card {
        background:#F8FAFC; border-left:3px solid #3B82F6;
        border-radius:0 8px 8px 0; padding:10px 14px;
        margin-bottom:8px; font-size:.95rem; color:#1E293B;
    }
    .deadline-card {
        background:#FFFBEB; border-left:3px solid #D97706;
        border-radius:0 8px 8px 0; padding:10px 14px;
        margin-bottom:8px; font-size:.95rem; color:#92400E;
    }
    .speaker-turn-card {
        background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px;
        padding:12px 16px; margin-bottom:10px; border-left:4px solid #6366F1;
    }
    .speaker-badge-1 { background:#EEF2FF; color:#4338CA; padding:2px 8px; border-radius:6px; font-weight:700; font-size:.85rem; }
    .speaker-badge-2 { background:#FDF2F8; color:#BE185D; padding:2px 8px; border-radius:6px; font-weight:700; font-size:.85rem; }
    .speaker-badge-3 { background:#F0FDF4; color:#15803D; padding:2px 8px; border-radius:6px; font-weight:700; font-size:.85rem; }
    .speaker-badge-4 { background:#FFFBEB; color:#B45309; padding:2px 8px; border-radius:6px; font-weight:700; font-size:.85rem; }
    .speaker-badge-def { background:#F1F5F9; color:#475569; padding:2px 8px; border-radius:6px; font-weight:700; font-size:.85rem; }
    .badge-valid   { background:#DCFCE7; color:#15803D; padding:3px 10px;
                     border-radius:12px; font-weight:600; font-size:.8rem; }
    .badge-invalid { background:#FEE2E2; color:#B91C1C; padding:3px 10px;
                     border-radius:12px; font-weight:600; font-size:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

model_name: str = st.sidebar.selectbox(
    "Whisper Model",
    options=["tiny", "base", "small", "medium", "large"],
    index=1,
    help="Larger models are more accurate but slower.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Speaker Diarization")
enable_diarization: bool = st.sidebar.checkbox(
    "Enable Speaker Diarization",
    value=True,
    help="Identifies different voices and assigns anonymous labels (Speaker 1, Speaker 2...).",
)

expected_speakers_opt: str = "Auto"
hf_token_input: str = ""

if enable_diarization:
    speaker_choice = st.sidebar.selectbox(
        "Expected Speakers",
        options=["Auto", "1", "2", "3", "4", "5", "6", "7", "8"],
        index=0,
        help="Choose 'Auto' to automatically discover distinct voices.",
    )
    expected_speakers_opt = speaker_choice

    with st.sidebar.expander("🔑 Hugging Face Token (Optional)"):
        st.markdown(
            "<small>Optional: For pyannote.audio pipeline. Local d-vector embedding engine runs by default without tokens.</small>",
            unsafe_allow_html=True,
        )
        hf_token_input = st.text_input(
            "HF Token",
            type="password",
            placeholder="hf_...",
            help="Your Hugging Face user access token (optional).",
        )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Accuracy Testing (Task 5)")
reference_text: str = st.sidebar.text_area(
    "Reference Transcript (Optional)",
    placeholder="Paste the known transcript here to calculate WER and accuracy…",
    height=100,
)

st.sidebar.markdown("---")
auto_save: bool = st.sidebar.checkbox("Auto-save transcript & summary", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Supported Formats:**\n"
    "- 🎬 Video: `MP4` `MKV` `MOV` `AVI` `WebM`\n"
    "- 🎵 Audio: `MP3` `WAV` `M4A` `OGG` `FLAC`\n\n"
    "**Limit:** 500 MB\n\n"
    "_FFmpeg must be installed on the system._"
)

# ── Main Header ───────────────────────────────────────────────────────────────
st.title("🎙️ Meeting Summarizer & Speaker Diarization")
st.markdown(
    "Upload any meeting recording to generate an **AI speaker-labeled transcript**, "
    "structured executive summary, and speaking-time analytics."
)

# ── Upload Area ───────────────────────────────────────────────────────────────
SUPPORTED = sorted(
    ext.lstrip(".") for ext in (AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)
)
uploaded_file = st.file_uploader(
    label="Upload meeting recording",
    type=SUPPORTED,
    label_visibility="collapsed",
    help="Supports all common video and audio formats up to 500 MB.",
)

# ── File Uploaded ─────────────────────────────────────────────────────────────
if uploaded_file is not None:
    st.markdown("---")

    file_ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
    col_preview, col_info = st.columns([2, 1])
    with col_preview:
        if f".{file_ext}" in VIDEO_EXTENSIONS:
            st.video(uploaded_file)
        else:
            st.audio(uploaded_file)

    validator = FileValidator()
    is_valid, val_error = validator.validate_streamlit_upload(uploaded_file)

    with col_info:
        size_mb = uploaded_file.size / (1024 * 1024)
        badge = (
            "<span class='badge-valid'>✔ Valid File</span>"
            if is_valid else
            "<span class='badge-invalid'>✘ Invalid File</span>"
        )
        diar_badge = "✅ Enabled" if enable_diarization else "Disabled"
        st.markdown(
            f"**📁 File:** `{html.escape(uploaded_file.name)}`  \n"
            f"**Size:** {size_mb:.2f} MB  \n"
            f"**Model:** `{model_name}`  \n"
            f"**Diarization:** `{diar_badge}`  \n"
            f"**Status:** {badge}",
            unsafe_allow_html=True,
        )

    if not is_valid:
        st.error(f"❌ **Validation Failed:** {val_error}")
        st.stop()

    # ── Transcribe Button ─────────────────────────────────────────────────────
    if st.button("🎙️ Transcribe & Generate Summary", type="primary"):
        tmp_input: str | None = None
        tmp_wav: str | None = None

        try:
            # ── Stage 1: Save upload to temp file (chunked) ──────────────────
            suffix = Path(uploaded_file.name).suffix or ".tmp"
            uploaded_file.seek(0)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="upload_"
            ) as f:
                while chunk := uploaded_file.read(1024 * 1024):
                    f.write(chunk)
                tmp_input = f.name

            st.markdown("#### ⚙️ Processing Pipeline")

            # ── Stage 2: File Validation ──────────────────────────────────────
            with st.spinner("🔍 Validating file…"):
                if not os.path.exists(tmp_input) or os.path.getsize(tmp_input) == 0:
                    raise RuntimeError("Uploaded file could not be saved for processing.")
            st.success("✅ File validation completed")

            # ── Stage 3: Audio Processing ─────────────────────────────────────
            with st.spinner("🔊 Processing audio via ffmpeg (converting to 16 kHz mono WAV)…"):
                processor = AudioProcessor()
                tmp_wav = processor.process(tmp_input)

            # Get duration
            audio_duration_sec = 0.0
            try:
                with wave.open(tmp_wav, "rb") as wf:
                    audio_duration_sec = wf.getnframes() / wf.getframerate()
            except Exception:
                pass

            st.success(f"✅ Audio processing completed ({audio_duration_sec:.1f}s duration)")

            # ── Stage 4: Speaker Diarization (if enabled) ─────────────────────
            speaker_turns: list[dict[str, Any]] = []
            num_speakers_detected: int = 1

            if enable_diarization:
                with st.spinner("👥 Detecting speakers & voice clusters (VAD + 256-d embeddings)..."):
                    try:
                        n_spk = None if expected_speakers_opt == "Auto" else int(expected_speakers_opt)
                        diarizer = get_diarizer(hf_token=hf_token_input if hf_token_input else None)
                        speaker_turns = diarizer.diarize(tmp_wav, num_speakers=n_spk)
                        unique_spks = set(t["speaker"] for t in speaker_turns)
                        num_speakers_detected = len(unique_spks)
                        st.success(f"✅ Speaker diarization completed — **{num_speakers_detected} speakers detected**")
                    except Exception as exc:
                        logger.warning("Diarization failed: %s, falling back to single speaker", exc)
                        st.warning(f"⚠️ Speaker diarization encountered an issue: {exc}. Continuing with standard transcription.")
                        speaker_turns = []

            # ── Stage 5: Whisper Transcription ────────────────────────────────
            with st.spinner(f"🤖 Transcribing with Whisper ({model_name} model)…"):
                transcriber = get_transcriber(model_name)
                result = transcriber.transcribe(tmp_wav)

            raw_text: str = result.get("text", "").strip()
            segments: list = result.get("segments", [])
            language: str = result.get("language", "unknown")
            st.success("✅ Whisper transcription completed")

            # ── Stage 6: Transcript Validation ───────────────────────────────
            with st.spinner("🔎 Validating transcript…"):
                is_meaningful = bool(raw_text) and len(raw_text.split()) >= 3

            if not is_meaningful:
                st.error(
                    "❌ **Transcript Validation Failed:** No meaningful speech was detected. "
                    "Please check that the audio track contains spoken content and try again."
                )
                logger.warning("Empty transcript for file: %s", uploaded_file.name)
                st.stop()

            st.success(
                f"✅ Transcript validation completed "
                f"({len(raw_text.split())} words, language: {language.upper()})"
            )

            # ── Stage 7: Speaker-Segment Alignment ────────────────────────────
            aligned_turns: list[dict[str, Any]] = []
            speaker_stats: dict[str, Any] = {}
            speaker_transcript_doc: str = ""

            if enable_diarization and speaker_turns:
                with st.spinner("🔗 Aligning Whisper segments with speaker turns…"):
                    aligned_turns = align_whisper_with_speakers(segments, speaker_turns)
                    speaker_stats = compute_speaker_statistics(aligned_turns, audio_duration_sec)
                    speaker_transcript_doc = format_speaker_transcript(aligned_turns)
                st.success("✅ Speaker alignment completed")

            # ── Stage 8: Summary Generation ───────────────────────────────────
            with st.spinner("✨ Analyzing topics & generating executive summary…"):
                summarizer = MeetingSummarizer()
                summary: dict[str, Any] = summarizer.summarize(raw_text)
            st.success("✅ Executive summary generated")

            # ── Display Results ───────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📋 Executive Meeting Summary & Transcript")

            # Overview Card
            overview_html = html.escape(summary["overview"])
            st.markdown(
                f"""<div class="overview-card">
                    <strong>🎯 Main Objective &amp; Overview:</strong><br>
                    {overview_html}
                </div>""",
                unsafe_allow_html=True,
            )

            # Tab setup
            tab_titles = [
                "👥 Speaker Transcript" if enable_diarization and aligned_turns else "📝 Transcript",
                "📑 Topics Discussed",
                "✅ Action Items",
                "⏰ Deadlines & Milestones",
                "📊 Stats & Speakers",
                "📝 Raw Transcript",
            ]
            tabs = st.tabs(tab_titles)

            # Tab 1: Speaker Transcript
            with tabs[0]:
                if enable_diarization and aligned_turns:
                    st.markdown(f"**🗣️ Conversation Flow ({len(aligned_turns)} speaker turns):**")
                    for turn in aligned_turns:
                        spk = turn["speaker"]
                        spk_id = turn.get("speaker_id", 0)
                        badge_class = f"speaker-badge-{(spk_id % 4) + 1}"
                        ts_str = turn.get("timestamp_str", f"[{turn['start']:.1f}s - {turn['end']:.1f}s]")
                        text_esc = html.escape(turn["text"])

                        st.markdown(
                            f"""<div class="speaker-turn-card">
                                <span class="{badge_class}">{html.escape(spk)}</span>
                                <small style="color:#64748B; margin-left:8px;">{html.escape(ts_str)}</small>
                                <div style="margin-top:6px; color:#1E293B; font-size:.98rem;">{text_esc}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.text_area("Transcript", value=raw_text, height=350, label_visibility="collapsed")

            # Tab 2: Topics
            with tabs[1]:
                if summary["topic_groups"]:
                    for group in summary["topic_groups"]:
                        topic_esc = html.escape(group["topic"])
                        points_html = "".join(
                            f"<li style='margin-bottom:6px;'>{html.escape(p)}</li>"
                            for p in group["points"]
                        )
                        st.markdown(
                            f"""<div class="topic-card">
                                <div class="topic-header">🔹 {topic_esc}</div>
                                <ul style="margin:0;padding-left:20px;color:#334155;">
                                    {points_html}
                                </ul>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No specific topic clusters detected. See the Full Transcript tab.")

            # Tab 3: Action Items
            with tabs[2]:
                if summary["action_items"]:
                    for item in summary["action_items"]:
                        st.markdown(
                            f"<div class='action-card'>☑️ {html.escape(item)}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No explicit action items were assigned in this recording.")

            # Tab 4: Deadlines
            with tabs[3]:
                if summary["deadlines"]:
                    for dl in summary["deadlines"]:
                        st.markdown(
                            f"<div class='deadline-card'>⏰ {html.escape(dl)}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No explicit deadlines or dates were mentioned.")

            # Tab 5: Stats & Speakers
            with tabs[4]:
                s = summary["stats"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Word Count", s["word_count"])
                c2.metric("Sentences", s["sentence_count"])
                c3.metric("Speaking Time", s["speaking_time"])
                c4.metric("Language", language.upper())

                if enable_diarization and speaker_stats.get("speakers"):
                    st.markdown("---")
                    st.markdown(f"#### 👥 Speaker Analytics ({speaker_stats['speaker_count']} Speakers Detected)")
                    spk_cols = st.columns(min(4, max(1, speaker_stats["speaker_count"])))
                    for idx, (spk_name, s_data) in enumerate(speaker_stats["speakers"].items()):
                        col_idx = idx % len(spk_cols)
                        with spk_cols[col_idx]:
                            st.metric(
                                label=spk_name,
                                value=f"{s_data['percentage']}%",
                                delta=f"{s_data['speaking_time_formatted']} ({s_data['turn_count']} turns)",
                                delta_color="off",
                            )

                if reference_text.strip():
                    st.markdown("---")
                    st.markdown("#### 🎯 Accuracy Evaluation (Task 5)")
                    metrics = calculate_wer_and_metrics(reference_text.strip(), raw_text)

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Reference Words", metrics["ref_words"])
                    mc2.metric("Generated Words", metrics["hyp_words"])
                    mc3.metric("WER", f"{metrics['wer'] * 100:.2f}%")

                    mc4, mc5, mc6 = st.columns(3)
                    mc4.metric("Substitutions", metrics["substitutions"])
                    mc5.metric("Deletions", metrics["deletions"])
                    mc6.metric("Insertions", metrics["insertions"])

                    acc = metrics["accuracy_pct"]
                    if metrics["passed"]:
                        st.success(f"**Accuracy:** {acc:.2f}%  |  **Required:** ≥90%  |  **Status:** ✅ PASS")
                    else:
                        st.error(f"**Accuracy:** {acc:.2f}%  |  **Required:** ≥90%  |  **Status:** ❌ FAIL")

            # Tab 6: Raw Transcript & Timestamps
            with tabs[5]:
                st.text_area("Complete Raw Transcript", value=raw_text, height=250, label_visibility="collapsed")
                if segments:
                    with st.expander("🕐 Timestamped Raw Segments"):
                        for seg in segments:
                            st.markdown(
                                f"**[{seg.get('start', 0):.1f}s → {seg.get('end', 0):.1f}s]** "
                                f"{html.escape(seg.get('text', '').strip())}"
                            )

            # ── Prepare Saved Documents ───────────────────────────────────────
            stem = Path(uploaded_file.name).stem
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            summary_lines = [
                "=" * 60,
                "          EXECUTIVE MEETING SUMMARY",
                "=" * 60,
                f"File : {uploaded_file.name}",
                f"Date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"Model: {model_name}",
                f"Diarization: {'Enabled (' + str(num_speakers_detected) + ' speakers)' if enable_diarization else 'Disabled'}",
                "",
                "MAIN OBJECTIVE & OVERVIEW:",
                f"  {summary['overview']}",
                "",
                "TOPICS DISCUSSED:",
            ]
            for group in summary["topic_groups"]:
                summary_lines.append(f"\n  [{group['topic'].upper()}]")
                for p in group["points"]:
                    summary_lines.append(f"     • {p}")
            if summary["action_items"]:
                summary_lines.append("\nACTION ITEMS & DELIVERABLES:")
                for item in summary["action_items"]:
                    summary_lines.append(f"  ☑ {item}")
            if summary["deadlines"]:
                summary_lines.append("\nDEADLINES & KEY MILESTONES:")
                for dl in summary["deadlines"]:
                    summary_lines.append(f"  ⏰ {dl}")
            summary_doc = "\n".join(summary_lines)

            # ── Auto-save ─────────────────────────────────────────────────────
            if auto_save:
                transcript_path = TRANSCRIPT_DIR / f"{stem}_{ts}_transcript.txt"
                summary_path = TRANSCRIPT_DIR / f"{stem}_{ts}_summary.txt"
                transcript_path.write_text(raw_text, encoding="utf-8")
                summary_path.write_text(summary_doc, encoding="utf-8")

                saved_notes = [f"`transcripts/{transcript_path.name}`", f"`transcripts/{summary_path.name}`"]

                if enable_diarization and speaker_transcript_doc:
                    spk_txt_path = TRANSCRIPT_DIR / f"{stem}_{ts}_speaker_transcript.txt"
                    spk_meta_path = TRANSCRIPT_DIR / f"{stem}_{ts}_speaker_metadata.json"
                    spk_txt_path.write_text(speaker_transcript_doc, encoding="utf-8")
                    spk_meta_path.write_text(
                        json.dumps({
                            "filename": uploaded_file.name,
                            "timestamp": ts,
                            "model": model_name,
                            "speaker_count": speaker_stats.get("speaker_count", 1),
                            "speakers": speaker_stats.get("speakers", {}),
                            "total_speech_time_sec": speaker_stats.get("total_speech_time_sec", 0.0),
                        }, indent=2),
                        encoding="utf-8",
                    )
                    saved_notes.append(f"`transcripts/{spk_txt_path.name}`")

                st.info("💾 Auto-saved → " + ", ".join(saved_notes))

            # ── Downloads ─────────────────────────────────────────────────────
            st.markdown("---")
            dl_cols = st.columns(3 if enable_diarization and speaker_transcript_doc else 2)
            with dl_cols[0]:
                st.download_button(
                    label="⬇️ Raw Transcript (.txt)",
                    data=raw_text,
                    file_name=f"{stem}_raw_transcript.txt",
                    mime="text/plain",
                )
            with dl_cols[1]:
                st.download_button(
                    label="⬇️ Executive Summary (.txt)",
                    data=summary_doc,
                    file_name=f"{stem}_summary.txt",
                    mime="text/plain",
                )
            if enable_diarization and speaker_transcript_doc:
                with dl_cols[2]:
                    st.download_button(
                        label="⬇️ Speaker-Attributed Transcript (.txt)",
                        data=speaker_transcript_doc,
                        file_name=f"{stem}_speaker_transcript.txt",
                        mime="text/plain",
                    )

        except RuntimeError as exc:
            logger.error("Pipeline RuntimeError: %s", exc, exc_info=True)
            st.error(f"❌ **Processing Error:** {exc}")

        except FileNotFoundError as exc:
            logger.error("File not found: %s", exc, exc_info=True)
            st.error(f"❌ **File Error:** {exc}")

        except Exception as exc:
            logger.exception("Unexpected error: %s", exc)
            st.error("❌ An unexpected error occurred. Please check that FFmpeg is installed and try again.")

        finally:
            for p in [tmp_input, tmp_wav]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
