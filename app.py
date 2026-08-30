"""
app.py
Meeting Transcription & Executive Summary Tool — Milestone 1

Pipeline:
  Upload → Validate → Extract Audio (ffmpeg) → Whisper → Validate Transcript → Summarize → Display
"""

from __future__ import annotations

import datetime
import html
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from accuracy import calculate_wer_and_metrics
from audio_processor import AudioProcessor
from summarizer import MeetingSummarizer
from transcriber import Transcriber
from validator import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, FileValidator

# ── Logging (debug info written to log, not exposed to user) ──────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Summarizer & Transcription",
    page_icon="🎙️",
    layout="wide",
)

# ── Transcripts Directory ─────────────────────────────────────────────────────
TRANSCRIPT_DIR = Path(__file__).parent / "transcripts"
TRANSCRIPT_DIR.mkdir(exist_ok=True)

# ── Model Cache (loads weights once; reloads only when model name changes) ────
@st.cache_resource(show_spinner=False)
def get_transcriber(model_name: str) -> Transcriber:
    """Cached Whisper model — survives re-runs for the same model choice."""
    return Transcriber(model_name=model_name)


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
    .badge-valid   { background:#DCFCE7; color:#15803D; padding:3px 10px;
                     border-radius:12px; font-weight:600; font-size:.8rem; }
    .badge-invalid { background:#FEE2E2; color:#B91C1C; padding:3px 10px;
                     border-radius:12px; font-weight:600; font-size:.8rem; }
    .pipeline-step { margin-bottom: 4px; }
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
st.sidebar.markdown("### 🎯 Accuracy Testing (Task 5)")
reference_text: str = st.sidebar.text_area(
    "Reference Transcript (Optional)",
    placeholder="Paste the known transcript here to calculate WER and accuracy…",
    height=110,
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
st.title("🎙️ Meeting Summarizer & Transcription")
st.markdown(
    "Upload any meeting recording to generate a clean, structured transcript and executive summary."
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

    # Decide preview type: video files get st.video, audio gets st.audio
    file_ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
    col_preview, col_info = st.columns([2, 1])
    with col_preview:
        if f".{file_ext}" in VIDEO_EXTENSIONS:
            st.video(uploaded_file)
        else:
            st.audio(uploaded_file)

    # Inline validation (extension + size; ffprobe runs after button press)
    validator = FileValidator()
    is_valid, val_error = validator.validate_streamlit_upload(uploaded_file)

    with col_info:
        size_mb = uploaded_file.size / (1024 * 1024)
        badge = (
            "<span class='badge-valid'>✔ Valid File</span>"
            if is_valid else
            "<span class='badge-invalid'>✘ Invalid File</span>"
        )
        st.markdown(
            f"**📁 File:** `{html.escape(uploaded_file.name)}`  \n"
            f"**Size:** {size_mb:.2f} MB  \n"
            f"**Model:** `{model_name}`  \n"
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
            # ── Stage 1: Save upload to temp file ────────────────────────────
            suffix = Path(uploaded_file.name).suffix or ".tmp"
            uploaded_file.seek(0)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="upload_"
            ) as f:
                # Stream in chunks to avoid loading entire file into memory twice
                while chunk := uploaded_file.read(1024 * 1024):
                    f.write(chunk)
                tmp_input = f.name

            st.markdown("#### ⚙️ Processing Pipeline")

            # ── Stage 2: File Validation ──────────────────────────────────────
            with st.spinner("🔍 Validating file…"):
                # We already validated above; confirm temp file is readable
                if not os.path.exists(tmp_input) or os.path.getsize(tmp_input) == 0:
                    raise RuntimeError("Uploaded file could not be saved for processing.")
            st.success("✅ File validation completed")

            # ── Stage 3: Audio Processing ─────────────────────────────────────
            with st.spinner("🔊 Processing audio via ffmpeg (converting to 16 kHz mono WAV)…"):
                processor = AudioProcessor()
                tmp_wav = processor.process(tmp_input)
            st.success("✅ Audio processing completed")

            # ── Stage 4: Whisper Transcription ────────────────────────────────
            with st.spinner(f"🤖 Transcribing with Whisper ({model_name} model)…"):
                transcriber = get_transcriber(model_name)
                result = transcriber.transcribe(tmp_wav)

            raw_text: str = result.get("text", "").strip()
            segments: list = result.get("segments", [])
            language: str = result.get("language", "unknown")
            st.success("✅ Whisper transcription completed")

            # ── Stage 5: Transcript Validation ───────────────────────────────
            with st.spinner("🔎 Validating transcript…"):
                is_meaningful = bool(raw_text) and len(raw_text.split()) >= 3

            if not is_meaningful:
                st.error(
                    "❌ **Transcript Validation Failed:** No meaningful speech was detected. "
                    "Please check that the audio track contains spoken content and try again."
                )
                logger.warning("Empty/trivial transcript returned for file: %s", uploaded_file.name)
                st.stop()

            st.success(
                f"✅ Transcript validation completed "
                f"({len(raw_text.split())} words detected, language: {language.upper()})"
            )

            # ── Stage 6: Summary Generation ───────────────────────────────────
            with st.spinner("✨ Analyzing topics & generating executive summary…"):
                summarizer = MeetingSummarizer()
                summary: dict[str, Any] = summarizer.summarize(raw_text)
            st.success("✅ Summary generation completed")

            # ── Display Results ───────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📋 Executive Meeting Summary")

            # Overview card — content escaped before insertion
            overview_html = html.escape(summary["overview"])
            st.markdown(
                f"""<div class="overview-card">
                    <strong>🎯 Main Objective &amp; Overview:</strong><br>
                    {overview_html}
                </div>""",
                unsafe_allow_html=True,
            )

            tab_topics, tab_actions, tab_deadlines, tab_transcript, tab_stats = st.tabs([
                "📑 Topics Discussed",
                "✅ Action Items",
                "⏰ Deadlines & Milestones",
                "📝 Full Transcript",
                "📊 Stats & Accuracy",
            ])

            # Tab 1 — Topics
            with tab_topics:
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

            # Tab 2 — Action Items
            with tab_actions:
                if summary["action_items"]:
                    for item in summary["action_items"]:
                        st.markdown(
                            f"<div class='action-card'>☑️ {html.escape(item)}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No explicit action items were assigned in this recording.")

            # Tab 3 — Deadlines
            with tab_deadlines:
                if summary["deadlines"]:
                    for dl in summary["deadlines"]:
                        st.markdown(
                            f"<div class='deadline-card'>⏰ {html.escape(dl)}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No explicit deadlines or dates were mentioned.")

            # Tab 4 — Full Transcript
            with tab_transcript:
                st.text_area(
                    "Complete Transcript",
                    value=raw_text,
                    height=300,
                    label_visibility="collapsed",
                )
                if segments:
                    with st.expander("🕐 Timestamped Segments"):
                        for seg in segments:
                            st.markdown(
                                f"**[{seg.get('start', 0):.1f}s → {seg.get('end', 0):.1f}s]** "
                                f"{html.escape(seg.get('text', '').strip())}"
                            )

            # Tab 5 — Stats & Accuracy
            with tab_stats:
                s = summary["stats"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Word Count", s["word_count"])
                c2.metric("Sentences", s["sentence_count"])
                c3.metric("Speaking Time", s["speaking_time"])
                c4.metric("Language", language.upper())

                if reference_text.strip():
                    st.markdown("---")
                    st.markdown("#### 🎯 Accuracy Evaluation (Task 5)")
                    metrics = calculate_wer_and_metrics(reference_text.strip(), raw_text)

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Reference Words", metrics["ref_words"])
                    mc2.metric("Generated Words", metrics["hyp_words"])
                    mc3.metric("WER", f"{metrics['wer'] * 100:.1f}%")

                    mc4, mc5, mc6 = st.columns(3)
                    mc4.metric("Substitutions", metrics["substitutions"])
                    mc5.metric("Deletions", metrics["deletions"])
                    mc6.metric("Insertions", metrics["insertions"])

                    acc = metrics["accuracy_pct"]
                    if metrics["passed"]:
                        st.success(
                            f"**Accuracy:** {acc:.1f}%  |  "
                            f"**Required:** ≥90%  |  **Status:** ✅ PASS"
                        )
                    else:
                        st.error(
                            f"**Accuracy:** {acc:.1f}%  |  "
                            f"**Required:** ≥90%  |  **Status:** ❌ FAIL  "
                            f"— Consider using a larger Whisper model."
                        )

            # ── Auto-save ─────────────────────────────────────────────────────
            stem = Path(uploaded_file.name).stem
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            summary_lines = [
                "=" * 60,
                "          EXECUTIVE MEETING SUMMARY",
                "=" * 60,
                f"File : {uploaded_file.name}",
                f"Date : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"Model: {model_name}",
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

            if auto_save:
                transcript_path = TRANSCRIPT_DIR / f"{stem}_{ts}_transcript.txt"
                summary_path = TRANSCRIPT_DIR / f"{stem}_{ts}_summary.txt"
                meta_path = TRANSCRIPT_DIR / f"{stem}_{ts}_metadata.json"

                transcript_path.write_text(raw_text, encoding="utf-8")
                summary_path.write_text(summary_doc, encoding="utf-8")
                meta_path.write_text(
                    json.dumps(
                        {
                            "filename": uploaded_file.name,
                            "timestamp": ts,
                            "model": model_name,
                            "language": language,
                            "word_count": s["word_count"],
                            "sentence_count": s["sentence_count"],
                            "speaking_time": s["speaking_time"],
                            "transcript_file": transcript_path.name,
                            "summary_file": summary_path.name,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                st.info(
                    f"💾 Auto-saved → `transcripts/{transcript_path.name}` "
                    f"and `transcripts/{summary_path.name}`"
                )

            # ── Downloads ─────────────────────────────────────────────────────
            st.markdown("---")
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="⬇️ Download Raw Transcript (.txt)",
                    data=raw_text,
                    file_name=f"{stem}_transcript.txt",
                    mime="text/plain",
                )
            with dl_col2:
                st.download_button(
                    label="⬇️ Download Executive Summary (.txt)",
                    data=summary_doc,
                    file_name=f"{stem}_summary.txt",
                    mime="text/plain",
                )

        except RuntimeError as exc:
            logger.error("Pipeline RuntimeError: %s", exc, exc_info=True)
            st.error(f"❌ **Processing Error:** {exc}")

        except FileNotFoundError as exc:
            logger.error("File not found: %s", exc, exc_info=True)
            st.error(f"❌ **File Error:** {exc}")

        except OSError as exc:
            logger.error("OS/IO error: %s", exc, exc_info=True)
            st.error(f"❌ **File I/O Error:** {exc}")

        except Exception as exc:
            logger.exception("Unexpected error during processing: %s", exc)
            st.error(
                "❌ An unexpected error occurred during processing. "
                "Please check that FFmpeg is installed and the file is a valid meeting recording."
            )

        finally:
            for p in [tmp_input, tmp_wav]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
