"""
app.py
Meeting Transcription & Executive Summary Tool

Pipeline:
1. Upload Video/Audio (MP4, MKV, MOV, AVI, MP3, WAV, M4A, etc.)
2. File Validation (Stream check, size check, format check)
3. Audio Processing via ffmpeg -> 16kHz mono WAV
4. Whisper Speech-to-Text Transcription (Cached model)
5. Clean Context-Aware Summary:
   - Executive Overview
   - Discussion Topics & Points
   - Action Items & Deliverables
   - Deadlines & Key Milestones
"""

import os
import json
import tempfile
import datetime
from pathlib import Path

import streamlit as st

from audio_processor import AudioProcessor
from transcriber import Transcriber
from validator import FileValidator
from summarizer import MeetingSummarizer

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Summarizer & Transcription",
    page_icon="🎙️",
    layout="wide",
)

# ── Saved Transcripts Directory ────────────────────────────────────────────────
TRANSCRIPT_DIR = Path(__file__).parent / "transcripts"
TRANSCRIPT_DIR.mkdir(exist_ok=True)

# ── Model Caching (Prevents reloading weights on every run) ────────────────────
@st.cache_resource
def get_transcriber(model_name: str) -> Transcriber:
    """Cache the loaded Whisper model instance across runs."""
    return Transcriber(model_name=model_name)

# ── Modern, Clean Styling ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Drop zone container */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2563EB;
        border-radius: 12px;
        padding: 24px 16px;
        background: #F8FAFC;
        transition: border-color .2s ease, background .2s ease;
        text-align: center;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #1D4ED8;
        background: #EFF6FF;
        cursor: pointer;
    }
    [data-testid="stFileUploader"] label { font-size:1rem; font-weight:600; color:#1E293B; }
    [data-testid="stFileUploader"] button {
        background-color:#2563EB !important; color:white !important;
        border-radius:8px !important; border:none !important;
        padding:8px 20px !important; font-weight:600 !important;
    }
    [data-testid="stFileUploader"] button:hover { background-color:#1D4ED8 !important; }

    /* Transcribe button */
    div.stButton > button[kind="primary"] {
        width:100%; padding:12px; font-size:1.05rem;
        border-radius:10px; font-weight:700; background-color:#2563EB;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color:#1D4ED8;
    }

    /* Executive overview callout */
    .overview-card {
        background: #F0FDF4;
        border-left: 4px solid #16A34A;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        margin-bottom: 20px;
        color: #14532D;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* Topic card */
    .topic-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .topic-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 8px;
    }

    /* Action item card */
    .action-card {
        background: #F8FAFC;
        border-left: 3px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.95rem;
        color: #1E293B;
    }

    /* Deadline card */
    .deadline-card {
        background: #FFFBEB;
        border-left: 3px solid #D97706;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.95rem;
        color: #92400E;
    }

    /* Badge */
    .badge-valid { background:#DCFCE7; color:#15803D; padding:3px 10px; border-radius:12px; font-weight:600; font-size:.8rem; }
    .badge-invalid { background:#FEE2E2; color:#B91C1C; padding:3px 10px; border-radius:12px; font-weight:600; font-size:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar Settings ───────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Whisper Model",
    options=["tiny", "base", "small", "medium", "large"],
    index=1,  # default: base
    help="Larger models are more accurate for domain/technical terms, but take longer.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Accuracy Testing (Task 5)")
reference_text = st.sidebar.text_area(
    "Reference Transcript (Optional)",
    placeholder="Paste expected transcript to calculate Word Error Rate (WER)...",
    height=90,
)

st.sidebar.markdown("---")
auto_save = st.sidebar.checkbox("Auto-save transcript & summary", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Supported Formats:**\n"
    "- 🎬 Video: `MP4`, `MKV`, `MOV`, `AVI`, `WebM`\n"
    "- 🎵 Audio: `MP3`, `WAV`, `M4A`, `OGG`, `FLAC`"
)

# ── Main Header ────────────────────────────────────────────────────────────────
st.title("🎙️ Meeting Summarizer & Transcription")
st.markdown("Upload any meeting video or audio to automatically generate a clean, structured summary.")
st.markdown("")

# ── Upload Zone ────────────────────────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center; font-size:1.05rem; color:#475569; margin-bottom:4px;'>"
    "⬇️ <strong>Drag &amp; drop</strong> meeting recording here, or click <em>Browse files</em>"
    "</p>",
    unsafe_allow_html=True,
)

SUPPORTED = ["mp4", "mkv", "mov", "avi", "webm", "mp3", "wav", "m4a", "ogg", "flac"]
uploaded_file = st.file_uploader(
    label="Upload meeting recording",
    type=SUPPORTED,
    label_visibility="collapsed",
    help="Supports all common video and audio formats up to 500 MB.",
)

# ── WER Helper (Uses jiwer for efficient industry-standard calculation) ────────
def calculate_wer(ref: str, hyp: str) -> float:
    """Calculate Word Error Rate (WER) using jiwer with graceful fallback."""
    try:
        import jiwer
        return float(jiwer.wer(ref, hyp))
    except Exception:
        r, h = ref.lower().split(), hyp.lower().split()
        n, m = len(r), len(h)
        if n == 0:
            return 0.0 if m == 0 else 1.0
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i][j] = dp[i - 1][j - 1] if r[i - 1] == h[j - 1] else 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[n][m] / n


# ── File Uploaded & Processing Flow ────────────────────────────────────────────
if uploaded_file is not None:
    st.markdown("---")

    # Step 1: Validation
    validator = FileValidator()
    is_valid, val_error = validator.validate_streamlit_upload(uploaded_file)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.audio(uploaded_file)
    with col_b:
        size_mb = uploaded_file.size / (1024 * 1024)
        badge = (
            "<span class='badge-valid'>✔ Valid File</span>"
            if is_valid else
            "<span class='badge-invalid'>✘ Invalid File</span>"
        )
        st.markdown(
            f"**📁 File:** `{uploaded_file.name}`  \n"
            f"**Size:** {size_mb:.2f} MB  \n"
            f"**Model:** `{model_name}`  \n"
            f"**Status:** {badge}",
            unsafe_allow_html=True,
        )

    if not is_valid:
        st.error(f"❌ **Validation Failed:** {val_error}")
        st.stop()

    if st.button("🔄 Generate Meeting Summary", type="primary"):
        tmp_input = None
        tmp_wav = None
        has_speech = False
        try:
            # 1. Save uploaded file to temp
            suffix = Path(uploaded_file.name).suffix or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="upload_") as f:
                f.write(uploaded_file.getvalue())
                tmp_input = f.name

            # 2. Audio extraction via ffmpeg
            with st.spinner("🔊 Extracting & converting audio via ffmpeg..."):
                processor = AudioProcessor()
                tmp_wav = processor.process(tmp_input)

            # 3. Whisper Speech-to-Text (using cached model)
            with st.spinner(f"🤖 Transcribing with Whisper ({model_name} model)..."):
                transcriber = get_transcriber(model_name)
                result = transcriber.transcribe(tmp_wav)

            transcript_text = result.get("text", "").strip()
            segments = result.get("segments", [])
            language = result.get("language", "en")
            has_speech = bool(transcript_text)

            if not has_speech:
                st.error("❌ No speech detected in the recording. Please check the audio track or recording volume.")

            if has_speech:
                # 4. Context-Aware Meeting Summarization
                with st.spinner("✨ Analyzing discussion topics & generating executive summary..."):
                    summarizer = MeetingSummarizer()
                    summary = summarizer.summarize(transcript_text)

                # 5. Display Structured Summary
                st.markdown("---")
                st.subheader("📋 Executive Meeting Summary")

                # ── Overview Card ──────────────────────────────────────────────
                st.markdown(
                    f"""
                    <div class="overview-card">
                        <strong>🎯 Main Objective & Overview:</strong><br>
                        {summary["overview"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Summary Tabs ───────────────────────────────────────────────
                tab_topics, tab_actions, tab_deadlines, tab_transcript, tab_stats = st.tabs([
                    "📑 Topics Discussed",
                    "✅ Action Items",
                    "⏰ Deadlines & Milestones",
                    "📝 Full Transcript",
                    "📊 Stats",
                ])

                # Tab 1: Topics Discussed
                with tab_topics:
                    if summary["topic_groups"]:
                        for group in summary["topic_groups"]:
                            points_html = "".join(f"<li style='margin-bottom:6px;'>{p}</li>" for p in group["points"])
                            st.markdown(
                                f"""
                                <div class="topic-card">
                                <div class="topic-header">🔹 {group['topic']}</div>
                                <ul style="margin:0; padding-left:20px; color:#334155;">
                                    {points_html}
                                </ul>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("No specific topic clusters detected. View Full Transcript for raw discussion.")

                # Tab 2: Action Items
                with tab_actions:
                    if summary["action_items"]:
                        for item in summary["action_items"]:
                            st.markdown(f"<div class='action-card'>☑️ {item}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No specific action items or tasks were explicitly assigned.")

                # Tab 3: Deadlines & Milestones
                with tab_deadlines:
                    if summary["deadlines"]:
                        for dl in summary["deadlines"]:
                            st.markdown(f"<div class='deadline-card'>⏰ {dl}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No explicit deadlines or dates were mentioned in this recording.")

                # Tab 4: Full Transcript with Timestamps
                with tab_transcript:
                    st.text_area("Complete Transcript", value=transcript_text, height=260, label_visibility="collapsed")
                    if segments:
                        with st.expander("🕐 Timestamped Segments"):
                            for s in segments:
                                st.markdown(f"**[{s.get('start', 0):.1f}s → {s.get('end', 0):.1f}s]** {s.get('text', '').strip()}")

                # Tab 5: Stats & Accuracy
                with tab_stats:
                    s = summary["stats"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Word Count", s["word_count"])
                    c2.metric("Sentences", s["sentence_count"])
                    c3.metric("Speaking Time", s["speaking_time"])
                    c4.metric("Language", language.upper())

                    if reference_text.strip():
                        wer = calculate_wer(reference_text.strip(), transcript_text)
                        acc = max(0.0, 1.0 - wer) * 100
                        if acc >= 90:
                            st.success(f"🎯 **Accuracy:** {acc:.1f}% (WER: {wer*100:.1f}%) — Meets the ≥90% threshold.")
                        else:
                            st.warning(f"🎯 **Accuracy:** {acc:.1f}% (WER: {wer*100:.1f}%) — Try `small` or `medium` model.")

                # ── Build Download Document ────────────────────────────────────
                doc_lines = [
                    "============================================================",
                    "               EXECUTIVE MEETING SUMMARY",
                    "============================================================",
                    f"File: {uploaded_file.name}",
                    f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "",
                    "🎯 MAIN OBJECTIVE & OVERVIEW:",
                    f"  {summary['overview']}",
                    "",
                    "📑 TOPICS DISCUSSED:",
                ]
                for group in summary["topic_groups"]:
                    doc_lines.append(f"\n  🔹 {group['topic'].upper()}:")
                    for p in group["points"]:
                        doc_lines.append(f"     • {p}")

                if summary["action_items"]:
                    doc_lines.append("\n✅ ACTION ITEMS & DELIVERABLES:")
                    for item in summary["action_items"]:
                        doc_lines.append(f"  ☑ {item}")

                if summary["deadlines"]:
                    doc_lines.append("\n⏰ DEADLINES & KEY MILESTONES:")
                    for dl in summary["deadlines"]:
                        doc_lines.append(f"  ⏰ {dl}")

                doc_lines.extend([
                    "",
                    "============================================================",
                    "                    FULL TRANSCRIPT",
                    "============================================================",
                    transcript_text,
                ])
                summary_doc = "\n".join(doc_lines)

                # Auto-save both summary, raw transcript, and metadata
                if auto_save:
                    stem = Path(uploaded_file.name).stem
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # 1. Summary Document
                    summary_path = TRANSCRIPT_DIR / f"{stem}_{ts}_summary.txt"
                    summary_path.write_text(summary_doc, encoding="utf-8")

                    # 2. Raw Transcript
                    raw_transcript_path = TRANSCRIPT_DIR / f"{stem}_{ts}_transcript.txt"
                    raw_transcript_path.write_text(transcript_text, encoding="utf-8")

                    # 3. Structured JSON Metadata
                    meta_path = TRANSCRIPT_DIR / f"{stem}_{ts}_metadata.json"
                    metadata = {
                        "filename": uploaded_file.name,
                        "timestamp": ts,
                        "model": model_name,
                        "language": language,
                        "word_count": s["word_count"],
                        "sentence_count": s["sentence_count"],
                        "speaking_time": s["speaking_time"],
                        "summary_file": summary_path.name,
                        "transcript_file": raw_transcript_path.name,
                    }
                    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

                # ── Download Section ───────────────────────────────────────────
                st.markdown("---")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        label="⬇️ Download Executive Summary (.txt)",
                        data=summary_doc,
                        file_name=f"{Path(uploaded_file.name).stem}_summary.txt",
                        mime="text/plain",
                    )
                with col_d2:
                    st.download_button(
                        label="⬇️ Download Raw Transcript (.txt)",
                        data=transcript_text,
                        file_name=f"{Path(uploaded_file.name).stem}_raw_transcript.txt",
                        mime="text/plain",
                    )

        except Exception as exc:
            st.error(f"❌ Error during processing: {exc}")
            import traceback
            with st.expander("🔍 Error details"):
                st.code(traceback.format_exc())

        finally:
            for p in [tmp_input, tmp_wav]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
