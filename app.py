"""
app.py
Meeting Transcription, Speaker Diarization & Executive Meeting Intelligence Tool
Integrating Milestone 1 (OpenAI Whisper + Speaker Diarization) with
Milestone 2 (LLM Summarization, Decision Tracking & Action Item Extraction).

Pipeline:
  Upload
  → Validate
  → Extract Audio (16kHz mono WAV via FFmpeg)
  → VAD & Voice Activity Detection
  → Speaker Embeddings & Voice Clustering (Anonymous: Speaker 1, 2, ...)
  → OpenAI Whisper Transcription
  → Speaker-Segment Alignment
  → Transcript Validation
  → Milestone 2 Meeting Intelligence (Gemini LLM Extraction: Summary, Key Points, Decisions, Participants, Action Items)
  → Speaker Analytics & Multi-Format Export
"""

from __future__ import annotations

import datetime
import html
import json
import logging
import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Optional

import streamlit as st

# ── Ensure Milestone 2 Project Root is in sys.path ────────────────────────────
PROJECT_ROOT = Path(r"C:\Users\Dell\.gemini\antigravity\scam caller\project-ai")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASK1_ROOT = Path(r"C:\Users\Dell\project-ai\task-1")
if str(TASK1_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK1_ROOT))

# Milestone 1 Imports
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

# Milestone 2 Imports
try:
    from milestone2.pipeline import process_meeting
    from milestone2.llm_service import LLMExtractionError, InvalidInputError
    from milestone2.schemas import MeetingIntelligence, ActionItem
    MILESTONE2_AVAILABLE = True
except Exception as _m2_err:
    MILESTONE2_AVAILABLE = False
    LLMExtractionError = Exception
    InvalidInputError = ValueError
    MeetingIntelligence = None
    ActionItem = None

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Intelligence & Speaker Diarization",
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
    .decision-card {
        background:#EFF6FF; border-left:4px solid #2563EB;
        border-radius:0 10px 10px 0; padding:14px 18px;
        margin-bottom:12px; color:#1E3A8A; font-size:1rem;
    }
    .topic-card {
        background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px;
        padding:16px 20px; margin-bottom:14px; box-shadow:0 1px 3px rgba(0,0,0,0.05);
    }
    .topic-header { font-weight:700; color:#1E293B; font-size:1rem; margin-bottom:8px; }
    .action-card {
        background:#F8FAFC; border-left:4px solid #3B82F6;
        border-radius:0 8px 8px 0; padding:12px 16px;
        margin-bottom:10px; color:#1E293B; font-size:.95rem;
    }
    .participant-pill {
        display: inline-block; background: #E0E7FF; color: #3730A3;
        font-weight: 600; font-size: 0.88rem; padding: 4px 12px;
        border-radius: 9999px; margin-right: 8px; margin-bottom: 8px;
    }
    .speaker-turn-card {
        background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px;
        padding:12px 16px; margin-bottom:10px;
    }
    .speaker-badge-1 { background:#DBEAFE; color:#1E40AF; font-weight:700; padding:2px 8px; border-radius:6px; font-size:0.85rem; }
    .speaker-badge-2 { background:#D1FAE5; color:#065F46; font-weight:700; padding:2px 8px; border-radius:6px; font-size:0.85rem; }
    .speaker-badge-3 { background:#FEF3C7; color:#92400E; font-weight:700; padding:2px 8px; border-radius:6px; font-size:0.85rem; }
    .speaker-badge-4 { background:#FCE7F3; color:#9D174D; font-weight:700; padding:2px 8px; border-radius:6px; font-size:0.85rem; }
    .priority-high { background:#FEE2E2; color:#991B1B; font-weight:700; padding:2px 8px; border-radius:6px; font-size:0.82rem; }
    .priority-medium { background:#FEF3C7; color:#92400E; font-weight:700; padding:2px 8px; border-radius:6px; font-size:0.82rem; }
    .priority-low { background:#E0E7FF; color:#3730A3; font-weight:700; padding:2px 8px; border-radius:6px; font-size:0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microphone.png", width=64)
    st.title("Settings & Engine")

    st.markdown("### 🤖 Model Selection")
    model_name = st.selectbox(
        "Whisper ASR Model",
        options=["base", "tiny", "small", "medium", "large"],
        index=0,
        help="base: balanced speed and accuracy. small/medium: higher accuracy.",
    )

    st.markdown("### 👥 Speaker Diarization")
    enable_diarization = st.checkbox(
        "Enable Speaker Diarization",
        value=True,
        help="Detects distinct voices and assigns consistent anonymous speaker IDs (Speaker 1, Speaker 2...).",
    )

    expected_speakers_opt = "Auto"
    hf_token = ""
    if enable_diarization:
        expected_speakers_opt = st.selectbox(
            "Expected Speaker Count",
            options=["Auto", "1", "2", "3", "4", "5", "6", "7", "8"],
            index=0,
            help="Auto: discovers speaker count automatically. Or constrain to a known number.",
        )
        with st.expander("⚙️ Advanced Diarization (PyAnnote)"):
            hf_token = st.text_input(
                "Hugging Face Token (Optional)",
                type="password",
                help="Optional: Enter HF token if you wish to use the pyannote.audio pipeline instead of the built-in local offline engine.",
            )

    st.markdown("### 🧠 AI Intelligence (Milestone 2)")
    enable_milestone2 = st.checkbox(
        "Enable Milestone 2 LLM Extraction",
        value=MILESTONE2_AVAILABLE,
        disabled=not MILESTONE2_AVAILABLE,
        help="Uses Google Gemini to extract executive summary, key points, decisions, participants, and action items table.",
    )
    if not MILESTONE2_AVAILABLE:
        st.caption("⚠️ Milestone 2 package not detected. Using Milestone 1 heuristic summarizer.")

    st.markdown("### 💾 Export & Auto-save")
    auto_save = st.checkbox("Auto-save outputs to `transcripts/`", value=True)

    st.markdown("---")
    st.caption("🎙️ **Milestone 1 + 2 Integrated Pipeline**")
    st.caption("✅ OpenAI Whisper ASR | ✅ ResNet Voice Clustering | ✅ Gemini LLM Intelligence")

# ── Main Header ───────────────────────────────────────────────────────────────
st.title("🎙️ Meeting Summarizer & Action Item Extraction")
st.markdown(
    "Upload meeting video or audio to generate **speaker-attributed transcripts**, "
    "**executive summaries**, **key decisions**, and a structured **Action Items Table**."
)

all_supported = sorted(list(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS))
uploaded_file = st.file_uploader(
    "Drop meeting recording here or click to browse",
    type=all_supported,
    help=f"Supported formats: {', '.join(all_supported).upper()}",
)

# Optional reference transcript for live accuracy testing
with st.expander("🎯 Accuracy Evaluation / Ground-Truth Reference (Optional)"):
    reference_text = st.text_area(
        "Paste Ground-Truth Reference Transcript",
        placeholder="Paste official reference transcript here to compute real-time WER, accuracy, substitutions, deletions, and insertions…",
        height=100,
    )

start_button = st.button("🚀 Process Recording", type="primary", disabled=uploaded_file is None)

# ── Pipeline Execution ────────────────────────────────────────────────────────
if start_button and uploaded_file is not None:
    tmp_input: str | None = None
    tmp_wav: str | None = None

    with st.container():
        try:
            # ── Stage 1: File Validation ──────────────────────────────────────
            with st.spinner("🔍 Validating file format and integrity…"):
                validator = FileValidator()
                is_valid, err_msg = validator.validate(uploaded_file)
                if not is_valid:
                    st.error(f"❌ **Validation Failed:** {err_msg}")
                    st.stop()

            st.success("✅ File validation passed")

            # ── Stage 2: Save to Temp File ────────────────────────────────────
            suffix = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f_in:
                f_in.write(uploaded_file.getvalue())
                tmp_input = f_in.name

            # ── Stage 3: Audio Extraction & 16kHz WAV Conversion ─────────────
            with st.spinner("🔊 Converting audio to 16 kHz Mono WAV…"):
                processor = AudioProcessor()
                tmp_wav = processor.process(tmp_input)

            # Get duration
            with wave.open(tmp_wav, "rb") as wf:
                audio_duration_sec = wf.getnframes() / wf.getframerate()

            st.success(f"✅ Audio converted: {audio_duration_sec:.1f}s ({audio_duration_sec/60:.1f} min)")

            # ── Stage 4: Speaker Diarization (Optional) ───────────────────────
            speaker_turns: list[dict[str, Any]] = []
            num_speakers_detected: int = 1

            if enable_diarization:
                with st.spinner("👥 Analyzing voices & clustering speaker embeddings…"):
                    try:
                        diarizer = get_diarizer(hf_token=hf_token.strip() if hf_token else None)
                        n_spk = int(expected_speakers_opt) if expected_speakers_opt != "Auto" else None
                        speaker_turns = diarizer.diarize(tmp_wav, num_speakers=n_spk)
                        detected_set = set(t["speaker"] for t in speaker_turns)
                        num_speakers_detected = max(1, len(detected_set))
                        st.success(f"✅ Speaker diarization complete ({num_speakers_detected} speakers detected)")
                    except Exception as exc:
                        logger.warning("Diarization failed: %s, continuing with standard transcription", exc)
                        st.warning(f"⚠️ Speaker diarization note: {exc}")
                        speaker_turns = []

            # ── Stage 5: Full-Stream Whisper Transcription ────────────────────
            with st.spinner(f"🤖 Transcribing speech with Whisper ({model_name} model)…"):
                transcriber = get_transcriber(model_name)
                result = transcriber.transcribe(tmp_wav)

            raw_text: str = result.get("text", "").strip()
            segments: list = result.get("segments", [])
            language: str = result.get("language", "unknown")
            st.success("✅ Whisper transcription completed")

            # ── Stage 6: Transcript Validation ───────────────────────────────
            with st.spinner("🔎 Validating transcript content…"):
                is_meaningful = bool(raw_text) and len(raw_text.split()) >= 3

            if not is_meaningful:
                st.error(
                    "❌ **Transcript Validation Failed:** No meaningful speech was detected. "
                    "Please verify that the audio contains spoken words and try again."
                )
                st.stop()

            st.success(f"✅ Transcript validated ({len(raw_text.split())} words, language: {language.upper()})")

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

            # ── Stage 8: Milestone 2 Meeting Intelligence & LLM Extraction ───
            meeting_intel: Optional[MeetingIntelligence] = None
            m1_summary: dict[str, Any] = {}

            if enable_milestone2 and MILESTONE2_AVAILABLE:
                with st.spinner("✨ Extracting Meeting Intelligence via Gemini (Milestone 2)…"):
                    try:
                        meeting_intel = process_meeting(raw_text)
                        st.success("✅ Milestone 2 Meeting Intelligence extracted successfully")
                    except InvalidInputError as exc:
                        logger.warning("Milestone 2 InvalidInputError: %s", exc)
                        st.warning(f"⚠️ Milestone 2 Input Notice: {exc}")
                    except LLMExtractionError as exc:
                        logger.warning("Milestone 2 LLMExtractionError: %s", exc)
                        st.warning(f"⚠️ Milestone 2 LLM Extraction Notice: {exc}")
                    except Exception as exc:
                        logger.error("Milestone 2 error: %s", exc, exc_info=True)
                        st.warning(f"⚠️ Milestone 2 processing encountered an issue: {exc}")

            # Heuristic Milestone 1 fallback if needed
            summarizer = MeetingSummarizer()
            m1_summary = summarizer.summarize(raw_text)

            # ── Display Results ───────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📋 Meeting Intelligence Dashboard")

            # Overview Card
            summary_text = meeting_intel.summary if meeting_intel else m1_summary["overview"]
            overview_html = html.escape(summary_text)
            engine_badge = "Gemini LLM (Milestone 2)" if meeting_intel else "Heuristic Classifier (Milestone 1)"
            
            st.markdown(
                f"""<div class="overview-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <strong>🎯 Executive Summary:</strong>
                        <span style="font-size:0.8rem; background:#DCFCE7; color:#166534; padding:2px 8px; border-radius:6px; font-weight:600;">{engine_badge}</span>
                    </div>
                    {overview_html}
                </div>""",
                unsafe_allow_html=True,
            )

            # Participants Pills
            participants_list = meeting_intel.participants if (meeting_intel and meeting_intel.participants) else []
            if participants_list:
                pills_html = "".join(f"<span class='participant-pill'>👤 {html.escape(p)}</span>" for p in participants_list)
                st.markdown(f"**👥 Meeting Participants:** {pills_html}", unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

            # Tab setup
            tab_titles = [
                "👥 Speaker Transcript" if enable_diarization and aligned_turns else "📝 Transcript",
                "✅ Action Items",
                "📑 Key Discussion Points",
                "🤝 Agreed Decisions",
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

            # Tab 2: Action Items Table (Milestone 2)
            with tabs[1]:
                if meeting_intel and meeting_intel.action_items:
                    st.markdown(f"#### ✅ Extracted Action Items ({len(meeting_intel.action_items)} Tasks)")
                    
                    # Render structured interactive table
                    table_rows = []
                    for item in meeting_intel.action_items:
                        table_rows.append({
                            "Task Description": item.task,
                            "Assignee": item.assignee,
                            "Deadline": item.deadline or "—",
                            "Priority": item.priority,
                            "Status": item.status,
                        })
                    st.dataframe(table_rows, use_container_width=True, hide_index=True)

                    # Card view
                    st.markdown("##### 📌 Detailed Task Cards")
                    for item in meeting_intel.action_items:
                        p_class = f"priority-{item.priority.lower()}"
                        st.markdown(
                            f"""<div class="action-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <strong>☑️ {html.escape(item.task)}</strong>
                                    <span class="{p_class}">{html.escape(item.priority)}</span>
                                </div>
                                <div style="margin-top:4px; font-size:0.88rem; color:#475569;">
                                    👤 <strong>Assignee:</strong> {html.escape(item.assignee)} &nbsp;|&nbsp; 
                                    ⏰ <strong>Deadline:</strong> {html.escape(item.deadline or 'Not specified')} &nbsp;|&nbsp; 
                                    🔄 <strong>Status:</strong> {html.escape(item.status)}
                                </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                elif m1_summary.get("action_items"):
                    st.markdown("#### ✅ Action Items (Heuristic)")
                    for item in m1_summary["action_items"]:
                        st.markdown(f"<div class='action-card'>☑️ {html.escape(item)}</div>", unsafe_allow_html=True)
                else:
                    st.info("No explicit action items detected in this recording.")

            # Tab 3: Key Discussion Points
            with tabs[2]:
                if meeting_intel and meeting_intel.key_points:
                    st.markdown("#### 📑 Key Discussion Points")
                    for pt in meeting_intel.key_points:
                        st.markdown(f"• {pt}")
                elif m1_summary.get("topic_groups"):
                    for group in m1_summary["topic_groups"]:
                        topic_esc = html.escape(group["topic"])
                        points_html = "".join(f"<li style='margin-bottom:6px;'>{html.escape(p)}</li>" for p in group["points"])
                        st.markdown(
                            f"""<div class="topic-card">
                                <div class="topic-header">🔹 {topic_esc}</div>
                                <ul style="margin:0;padding-left:20px;color:#334155;">{points_html}</ul>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No key discussion points identified.")

            # Tab 4: Agreed Decisions
            with tabs[3]:
                if meeting_intel and meeting_intel.decisions:
                    st.markdown("#### 🤝 Agreed Decisions & Consensus")
                    for dec in meeting_intel.decisions:
                        st.markdown(
                            f"""<div class="decision-card">
                                <strong>✔️ Decision:</strong> {html.escape(dec)}
                            </div>""",
                            unsafe_allow_html=True,
                        )
                elif m1_summary.get("deadlines"):
                    st.markdown("#### ⏰ Key Deadlines & Milestones")
                    for dl in m1_summary["deadlines"]:
                        st.markdown(f"<div class='decision-card'>⏰ {html.escape(dl)}</div>", unsafe_allow_html=True)
                else:
                    st.info("No formal decisions or agreements detected.")

            # Tab 5: Stats & Speakers
            with tabs[4]:
                s = m1_summary["stats"]
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
                    st.markdown("#### 🎯 Accuracy Evaluation (Milestone 1 Task 5)")
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
                "          EXECUTIVE MEETING INTELLIGENCE REPORT",
                "=" * 60,
                f"File : {uploaded_file.name}",
                f"Date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"ASR Model : Whisper {model_name}",
                f"Intelligence Engine : {'Milestone 2 (Gemini LLM)' if meeting_intel else 'Milestone 1 (Heuristic)'}",
                f"Diarization : {'Enabled (' + str(num_speakers_detected) + ' speakers)' if enable_diarization else 'Disabled'}",
                "",
                "EXECUTIVE SUMMARY:",
                f"  {summary_text}",
            ]

            if participants_list:
                summary_lines.extend(["", "PARTICIPANTS:", "  " + ", ".join(participants_list)])

            if meeting_intel and meeting_intel.key_points:
                summary_lines.extend(["", "KEY DISCUSSION POINTS:"])
                for pt in meeting_intel.key_points:
                    summary_lines.append(f"  • {pt}")

            if meeting_intel and meeting_intel.decisions:
                summary_lines.extend(["", "AGREED DECISIONS:"])
                for dec in meeting_intel.decisions:
                    summary_lines.append(f"  ✔️ {dec}")

            if meeting_intel and meeting_intel.action_items:
                summary_lines.extend(["", "ACTION ITEMS & DELIVERABLES:"])
                for item in meeting_intel.action_items:
                    dl = f" (Deadline: {item.deadline})" if item.deadline else ""
                    summary_lines.append(f"  ☑ [{item.priority}] {item.task} — Assignee: {item.assignee}{dl} [Status: {item.status}]")

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
                            "participants": participants_list,
                            "action_items_count": len(meeting_intel.action_items) if meeting_intel else 0,
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
                    file_name=f"{stem}_executive_summary.txt",
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
            st.error(f"❌ An unexpected error occurred: {exc}")

        finally:
            for p in [tmp_input, tmp_wav]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
