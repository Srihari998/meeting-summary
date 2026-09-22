"""
app.py — IntelliMeet: Intelligent Meeting Intelligence & Speaker Diarization Platform
Integrating Whisper ASR, PyTorch ResNet Diarization, Gemini LLM Intelligence,
and ChromaDB Vector Knowledge Repository with RAG.
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

# ── Ensure Project Root in sys.path ───────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

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
    from milestone2.participants import infer_speaker_roles
    MILESTONE2_AVAILABLE = True
except Exception as _m2_err:
    MILESTONE2_AVAILABLE = False
    LLMExtractionError = Exception
    InvalidInputError = ValueError
    MeetingIntelligence = None
    ActionItem = None
    infer_speaker_roles = None

# Milestone 3 Imports
try:
    from milestone3.indexing_pipeline import index_meeting, index_all_meetings, IndexingResult
    from milestone3.semantic_search_service import SemanticSearchService
    from milestone3.rag_service import RAGService
    from milestone3.schemas import SearchRequest, RAGRequest
    from milestone3.vector_store_service import VectorStoreService
    from milestone3.meeting_repository import MeetingRepository
    MILESTONE3_AVAILABLE = True
except Exception as _m3_err:
    MILESTONE3_AVAILABLE = False
    _m3_err_msg = str(_m3_err)
    MeetingRepository = None

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IntelliMeet — Turn conversations into intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
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
    /* Brand Header */
    .brand-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .brand-sparkle {
        color: #6366F1;
        font-size: 1.8rem;
        line-height: 1;
    }
    .brand-tagline {
        color: #64748B;
        font-size: 0.88rem;
        font-weight: 400;
        margin-bottom: 20px;
    }
    .nav-header {
        font-size: 0.72rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 1.2px;
        margin-top: 18px;
        margin-bottom: 8px;
        text-transform: uppercase;
    }

    /* Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #6366F1; border-radius: 12px;
        padding: 24px 16px; background: #F8FAFC;
        transition: border-color .2s ease, background .2s ease;
        text-align: center;
    }
    [data-testid="stFileUploader"]:hover { border-color: #4F46E5; background: #EEF2FF; }
    [data-testid="stFileUploader"] label { font-size:1rem; font-weight:600; color:#1E293B; }
    [data-testid="stFileUploader"] button {
        background-color:#6366F1 !important; color:white !important;
        border-radius:8px !important; border:none !important;
        padding:8px 20px !important; font-weight:600 !important;
    }
    div.stButton > button[kind="primary"] {
        width:100%; padding:12px; font-size:1.05rem;
        border-radius:10px; font-weight:700; background-color:#6366F1;
    }
    /* Suggested Questions Buttons */
    div[data-testid="column"] button[kind="secondary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 5px rgba(99, 102, 241, 0.25) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    div[data-testid="column"] button[kind="secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.4) !important;
        color: #FFFFFF !important;
    }
    .overview-card {
        background:#F0FDF4; border-left:4px solid #16A34A;
        border-radius:0 10px 10px 0; padding:16px 20px;
        margin-bottom:20px; color:#14532D; font-size:1.05rem; line-height:1.6;
    }
    .decision-card {
        background:#EFF6FF; border-left:4px solid #3B82F6;
        border-radius:0 10px 10px 0; padding:14px 18px;
        margin-bottom:12px; color:#1E3A8A; font-size:1rem;
    }
    .topic-card {
        background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px;
        padding:16px 20px; margin-bottom:14px; box-shadow:0 1px 3px rgba(0,0,0,0.05);
    }
    .topic-header { font-weight:700; color:#1E293B; font-size:1rem; margin-bottom:8px; }
    .action-card {
        background:#F8FAFC; border-left:4px solid #6366F1;
        border-radius:0 8px 8px 0; padding:12px 16px;
        margin-bottom:10px; color:#1E293B; font-size:.95rem;
    }
    .participant-pill {
        display: inline-block; background: #EEF2FF; color: #4338CA;
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

# ── Sidebar Navigation & Controls ─────────────────────────────────────────────
with st.sidebar:
    # IntelliMeet Branding Header
    st.markdown(
        """
        <div style="padding: 6px 0 10px 0;">
            <div class="brand-title">
                <span class="brand-sparkle">✦</span>
                <span>IntelliMeet</span>
            </div>
            <div class="brand-tagline">
                Turn conversations into intelligence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # WORKSPACE Navigation Section
    st.markdown('<div class="nav-header">WORKSPACE</div>', unsafe_allow_html=True)
    nav_selection = st.radio(
        "Workspace Navigation",
        options=[
            "Command Center",
            "Meetings",
            "Intelligence",
            "Action Hub",
            "People",
            "Ask IntelliMeet",
            "Validation",
        ],
        index=0,
        label_visibility="collapsed",
    )

    # SYSTEM Section
    st.markdown('<div class="nav-header">SYSTEM</div>', unsafe_allow_html=True)
    
    with st.expander("⚙️ Engine & Model Configuration", expanded=False):
        model_name = st.selectbox(
            "Whisper ASR Model",
            options=["base", "tiny", "small", "medium", "large"],
            index=0,
            help="base: balanced speed and accuracy. small/medium: higher accuracy.",
        )

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
            )
            hf_token = st.text_input(
                "Hugging Face Token (Optional)",
                type="password",
                help="Optional PyAnnote pipeline token.",
            )

        enable_milestone2 = st.checkbox(
            "Enable AI Intelligence (LLM)",
            value=MILESTONE2_AVAILABLE,
            disabled=not MILESTONE2_AVAILABLE,
            help="Uses Google Gemini for executive summary, key points, decisions, roles, and action items.",
        )

        auto_save = st.checkbox("Auto-save outputs to `transcripts/`", value=True)

    # Knowledge Base Quick Status & Index Button
    if MILESTONE3_AVAILABLE:
        try:
            _vstore_sb = VectorStoreService()
            _total_vecs = _vstore_sb.count()
        except Exception:
            _total_vecs = 0
        st.caption(f"📚 **{_total_vecs} knowledge chunks** indexed")

        if st.button("📥 Index All Past Meetings", help="Embeds all historical meetings into ChromaDB vector store."):
            with st.spinner("Indexing meetings into knowledge base…"):
                try:
                    _idx_res = index_all_meetings()
                    if _idx_res.chunks_indexed > 0:
                        st.success(f"✅ Indexed {_idx_res.chunks_indexed} new chunks from {len(_idx_res.meeting_ids_processed)} meeting(s).")
                    elif _idx_res.chunks_skipped > 0:
                        st.info(f"ℹ️ All {_idx_res.chunks_skipped} chunks already up to date.")
                    if _idx_res.errors:
                        st.warning(f"⚠️ {_idx_res.errors[0]}")
                except Exception as _iexc:
                    st.error(f"❌ Indexing error: {_iexc}")

    st.markdown("---")
    st.caption("✦ **IntelliMeet Platform** | Multi-Speaker Diarization, LLM Intelligence & RAG")


# Helper function to render meeting intelligence results
def render_intelligence_dashboard(data: dict[str, Any], show_uploader_note: bool = False):
    meeting_intel: Optional[MeetingIntelligence] = data.get("meeting_intel")
    m1_summary: dict[str, Any] = data.get("m1_summary", {})
    speaker_roles: dict[str, str] = data.get("speaker_roles", {})
    aligned_turns: list[dict[str, Any]] = data.get("aligned_turns", [])
    raw_text: str = data.get("raw_text", "")
    segments: list = data.get("segments", [])
    speaker_stats: dict[str, Any] = data.get("speaker_stats", {})
    language: str = data.get("language", "en")
    enable_diarization: bool = data.get("enable_diarization", True)
    reference_text: str = data.get("reference_text", "")

    summary_text = meeting_intel.summary if meeting_intel else m1_summary.get("overview", "")
    overview_html = html.escape(summary_text)
    engine_badge = "Gemini AI Engine" if meeting_intel else "Fast Heuristic Engine"

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

    # Participants with Roles Pills
    distinct_speakers = [t["speaker"] for t in aligned_turns] if aligned_turns else []
    participants_list = meeting_intel.participants if (meeting_intel and meeting_intel.participants) else distinct_speakers
    if participants_list:
        pills_html_list = []
        for p in participants_list:
            p_role = speaker_roles.get(p, speaker_roles.get(p.strip(":"), "Team Member / Coworker"))
            pills_html_list.append(
                f"<span class='participant-pill'>👤 <strong>{html.escape(p)}</strong> <span style='opacity:0.85; font-size:0.82rem;'>({html.escape(p_role)})</span></span>"
            )
        st.markdown(f"**👥 Meeting Participants & Roles:** " + "".join(pills_html_list), unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

    # Tabs setup
    tab_titles = [
        "👥 Speaker Transcript" if enable_diarization and aligned_turns else "📝 Transcript",
        "✅ Action Items",
        "📑 Key Discussion Points",
        "🤝 Agreed Decisions",
        "📊 Stats & Speakers",
        "📝 Raw Transcript",
        "🔍 Semantic Search",
        "💬 Ask IntelliMeet",
    ]
    tabs = st.tabs(tab_titles)

    # Tab 1: Speaker Transcript
    with tabs[0]:
        if enable_diarization and aligned_turns:
            st.markdown(f"**🗣️ Conversation Flow with Inferred Roles ({len(aligned_turns)} turns):**")
            for turn in aligned_turns:
                spk = turn["speaker"]
                spk_id = turn.get("speaker_id", 0)
                spk_role = speaker_roles.get(spk, speaker_roles.get(f"Speaker {spk_id + 1}", "Team Member / Coworker"))
                badge_class = f"speaker-badge-{(spk_id % 4) + 1}"
                ts_str = turn.get("timestamp_str", f"[{turn['start']:.1f}s - {turn['end']:.1f}s]")
                text_esc = html.escape(turn["text"])

                st.markdown(
                    f"""<div class="speaker-turn-card">
                        <span class="{badge_class}">{html.escape(spk)}</span>
                        <span style="background:#F1F5F9; color:#334155; font-weight:600; padding:2px 8px; border-radius:6px; font-size:0.82rem; margin-left:6px;">💼 {html.escape(spk_role)}</span>
                        <small style="color:#64748B; margin-left:8px;">{html.escape(ts_str)}</small>
                        <div style="margin-top:6px; color:#1E293B; font-size:.98rem;">{text_esc}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.text_area("Transcript", value=raw_text, height=350, label_visibility="collapsed")

    # Tab 2: Action Items Table (Responsive & Word-Wrapped)
    with tabs[1]:
        if meeting_intel and meeting_intel.action_items:
            st.markdown(f"#### ✅ Extracted Action Items ({len(meeting_intel.action_items)} Tasks)")
            table_html_rows = []
            for idx, item in enumerate(meeting_intel.action_items, start=1):
                a_role = speaker_roles.get(item.assignee, "")
                role_tag = f"<br><small style='color:#64748B;'>({html.escape(a_role)})</small>" if a_role and a_role != "Team Member / Coworker" else ""
                p_class = f"priority-{item.priority.lower()}"
                table_html_rows.append(
                    f"""<tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 12px 14px; font-weight:600; color:#0F172A; line-height:1.5; word-break:break-word;">{idx}. {html.escape(item.task)}</td>
                        <td style="padding: 12px 14px; color:#1E293B; word-break:break-word;">👤 {html.escape(item.assignee)}{role_tag}</td>
                        <td style="padding: 12px 14px; color:#475569; word-break:break-word;">⏰ {html.escape(item.deadline or '—')}</td>
                        <td style="padding: 12px 14px;"><span class="{p_class}">{html.escape(item.priority)}</span></td>
                        <td style="padding: 12px 14px; color:#334155; font-size:0.88rem;">{html.escape(item.status)}</td>
                    </tr>"""
                )

            st.markdown(
                f"""<div style="overflow-x:auto; width:100%; margin-bottom:24px; border:1px solid #CBD5E1; border-radius:10px; background:#FFFFFF;">
                    <table style="width:100%; border-collapse:collapse; font-size:0.95rem; text-align:left;">
                        <thead>
                            <tr style="background:#F8FAFC; border-bottom:2px solid #CBD5E1; color:#1E293B; font-weight:700;">
                                <th style="padding:12px 14px; width:45%;">Task Description</th>
                                <th style="padding:12px 14px; width:22%;">Assignee</th>
                                <th style="padding:12px 14px; width:15%;">Deadline</th>
                                <th style="padding:12px 14px; width:10%;">Priority</th>
                                <th style="padding:12px 14px; width:8%;">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(table_html_rows)}
                        </tbody>
                    </table>
                </div>""",
                unsafe_allow_html=True,
            )
        elif m1_summary.get("action_items"):
            st.markdown("#### ✅ Action Items (Heuristic)")
            for item in m1_summary["action_items"]:
                st.markdown(f"<div class='action-card'>☑️ {html.escape(item)}</div>", unsafe_allow_html=True)
        else:
            st.info("No explicit action items detected.")

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
        s = m1_summary.get("stats", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Word Count", s.get("word_count", len(raw_text.split())))
        c2.metric("Sentences", s.get("sentence_count", 0))
        c3.metric("Speaking Time", s.get("speaking_time", "0:00"))
        c4.metric("Language", language.upper())

        if enable_diarization and speaker_stats.get("speakers"):
            st.markdown("---")
            st.markdown(f"#### 👥 Speaker Analytics & Roles ({speaker_stats['speaker_count']} Speakers Detected)")
            spk_cols = st.columns(min(4, max(1, speaker_stats["speaker_count"])))
            for idx, (spk_name, s_data) in enumerate(speaker_stats["speakers"].items()):
                col_idx = idx % len(spk_cols)
                r = speaker_roles.get(spk_name, "")
                label_str = f"{spk_name} ({r})" if r else spk_name
                with spk_cols[col_idx]:
                    st.metric(
                        label=label_str,
                        value=f"{s_data['percentage']}%",
                        delta=f"{s_data['speaking_time_formatted']} ({s_data['turn_count']} turns)",
                        delta_color="off",
                    )

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

    # Tab 7: Semantic Search
    with tabs[6]:
        if not MILESTONE3_AVAILABLE:
            st.warning("⚠️ Semantic search requires ChromaDB.")
        else:
            render_search_interface()

    # Tab 8: Ask MEETIQ
    with tabs[7]:
        if not MILESTONE3_AVAILABLE:
            st.warning("⚠️ RAG requires ChromaDB and Gemini.")
        else:
            render_rag_interface()


def render_search_interface():
    st.markdown("#### 🔍 Search Meeting Knowledge Base")
    st.caption("Search across all indexed meetings using semantic understanding.")
    _sc1, _sc2 = st.columns([4, 1])
    with _sc1:
        _q = st.text_input("Search query", placeholder="e.g. 'budget decisions' or 'API integration deadline'", label_visibility="collapsed", key="search_query_input")
    with _sc2:
        _top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, key="search_top_k_input")

    _filter = st.selectbox("Filter by content type", options=["All", "transcript", "summary", "decision", "action_item"], index=0, key="search_filter_input")

    if st.button("🔍 Execute Search", key="btn_exec_search"):
        if not _q.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching knowledge base…"):
                try:
                    svc = SemanticSearchService()
                    results = svc.search(SearchRequest(query=_q, top_k=int(_top_k), source_type=None if _filter == "All" else _filter))
                    if not results:
                        st.info("No matching records found.")
                    else:
                        latency = results[0].search_latency_ms if results else 0
                        status_color = "#16A34A" if latency < 3000 else "#DC2626"
                        st.markdown(f"**{len(results)} result(s)** — <span style='color:{status_color}; font-weight:600;'>⏱ {latency:.0f} ms</span>", unsafe_allow_html=True)
                        for rank_i, _sr in enumerate(results, 1):
                            score_pct = int(_sr.relevance_score * 100)
                            score_color = "#16A34A" if _sr.relevance_score > 0.7 else "#D97706" if _sr.relevance_score > 0.4 else "#6B7280"
                            _badge_map = {
                                "transcript": ("📄", "#DBEAFE", "#1E40AF"),
                                "summary": ("📋", "#D1FAE5", "#065F46"),
                                "decision": ("✔️", "#EDE9FE", "#4C1D95"),
                                "action_item": ("☑️", "#FEF3C7", "#92400E"),
                            }
                            _icon, _bg, _fg = _badge_map.get(_sr.source_type, ("📄", "#F1F5F9", "#334155"))
                            st.markdown(
                                f"""<div style="border:1px solid #E2E8F0; border-radius:10px; padding:14px 18px; margin-bottom:12px; background:#FFFFFF;">
                                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                        <div>
                                            <span style="font-weight:700; color:#1E293B;">#{rank_i}</span>
                                            <span style="background:{_bg}; color:{_fg}; font-size:0.82rem; font-weight:600; padding:2px 8px; border-radius:6px; margin-left:8px;">{_icon} {html.escape(_sr.source_type)}</span>
                                        </div>
                                        <div>
                                            <span style="color:{score_color}; font-weight:700; font-size:0.9rem;">{score_pct}% match</span>
                                            <span style="color:#94A3B8; font-size:0.8rem; margin-left:10px;">Meeting: {html.escape(_sr.meeting_id[:16])}…</span>
                                        </div>
                                    </div>
                                    <div style="color:#334155; font-size:0.95rem; line-height:1.55;">{html.escape(_sr.content[:600])}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                except Exception as ex:
                    st.error(f"❌ Search error: {ex}")


def render_rag_interface():
    # 1. Suggested Questions Section Header
    st.markdown("##### 💡 **Suggested Questions (Click to test):**")
    
    suggested_questions = [
        "What did we decide about the mobile application?",
        "What tasks were assigned to Priya?",
        "What was the status of API integration?",
        "Which meetings discussed the project launch?",
        "What deadlines were discussed?",
        "Who was responsible for UI testing?",
    ]
    
    if "rag_query_text" not in st.session_state:
        st.session_state["rag_query_text"] = "what are the tasks for Q4 Mobile Application Launch Roadmap"
    
    # Render 2 rows of 3 columns with purple gradient buttons
    for row_idx in range(0, len(suggested_questions), 3):
        cols = st.columns(3)
        for col_idx, q_text in enumerate(suggested_questions[row_idx:row_idx+3]):
            with cols[col_idx]:
                if st.button(f"💭 {q_text}", key=f"sugg_q_{row_idx + col_idx}", use_container_width=True):
                    st.session_state["rag_query_text"] = q_text
                    st.session_state["trigger_rag_search"] = True
                    st.rerun()

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.markdown("**Your Question:**")
    
    user_query = st.text_input(
        "Your Question",
        value=st.session_state.get("rag_query_text", ""),
        placeholder="e.g. what are the tasks for Q4 Mobile Application Launch Roadmap",
        label_visibility="collapsed",
        key="rag_input_field",
    )
    
    # Keep session state in sync
    if user_query != st.session_state.get("rag_query_text", ""):
        st.session_state["rag_query_text"] = user_query
    
    # Action Row
    col_btn1, col_btn2, col_note = st.columns([1.5, 1.2, 5])
    with col_btn1:
        search_clicked = st.button("🔮 Search with AI", key="btn_search_ai", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("🧹 Clear", key="btn_clear_ai", use_container_width=True)
    with col_note:
        st.markdown("<div style='color: #64748B; font-size: 0.86rem; padding-top: 8px;'>Answers are validated and synthesized strictly against your SQLite meeting repository.</div>", unsafe_allow_html=True)
        
    if clear_clicked:
        st.session_state["rag_query_text"] = ""
        st.session_state["last_rag_resp"] = None
        st.session_state["trigger_rag_search"] = False
        st.rerun()
        
    should_search = search_clicked or st.session_state.get("trigger_rag_search", False)
    
    if should_search:
        st.session_state["trigger_rag_search"] = False
        active_q = st.session_state.get("rag_query_text", "").strip()
        if not active_q:
            st.warning("Please enter a question or click a suggested question above.")
        else:
            with st.spinner("Retrieving meeting context & synthesizing grounded answer…"):
                try:
                    rag_svc = RAGService()
                    resp = rag_svc.answer(RAGRequest(question=active_q, top_k=5))
                    st.session_state["last_rag_resp"] = resp
                except Exception as ex:
                    st.error(f"❌ Answer generation error: {ex}")
                    st.session_state["last_rag_resp"] = None

    # Render Answer Card & Sources
    last_resp = st.session_state.get("last_rag_resp")
    if last_resp:
        latency_sec = last_resp.latency_ms / 1000.0
        
        # Emerald Green Answer Card matching screenshot
        st.markdown(
            f"""<div style="background: #F0FDF4; border: 2px solid #10B981; border-radius: 12px; padding: 18px 22px; margin-top: 20px; margin-bottom: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-weight: 700; color: #065F46; font-size: 1.08rem; display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem;">💡</span>
                        <span>AI Synthesized Answer</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="background: #DCFCE7; color: #166534; font-size: 0.82rem; font-weight: 600; padding: 4px 10px; border-radius: 6px; border: 1px solid #BBF7D0;">
                            ✅ Grounded in Repository
                        </span>
                        <span style="background: #EFF6FF; color: #1E40AF; font-size: 0.82rem; font-weight: 600; padding: 4px 10px; border-radius: 6px; border: 1px solid #DBEAFE;">
                            ⚡ {latency_sec:.2f}s
                        </span>
                    </div>
                </div>
                <div style="color: #1E293B; font-size: 0.98rem; line-height: 1.65;">
                    {html.escape(last_resp.answer)}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        
        # Source Meeting Records Section
        if last_resp.sources:
            st.markdown(
                """<div style="font-weight: 700; font-size: 1.15rem; color: #1E293B; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.2rem;">📁</span>
                    <span>Source Meeting Records</span>
                </div>""",
                unsafe_allow_html=True,
            )
            
            _badge_map = {
                "transcript": ("📄", "#DBEAFE", "#1E40AF"),
                "summary": ("📋", "#D1FAE5", "#065F46"),
                "decision": ("✔️", "#EDE9FE", "#4C1D95"),
                "action_item": ("☑️", "#FEF3C7", "#92400E"),
            }
            
            for idx, src in enumerate(last_resp.sources, 1):
                icon, bg, fg = _badge_map.get(src.source_type, ("📄", "#F1F5F9", "#334155"))
                score_pct = int(src.relevance_score * 100)
                st.markdown(
                    f"""<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-weight: 700; color: #0F172A; font-size: 0.92rem;">Meeting ID: <code>{html.escape(src.meeting_id)}</code></span>
                                <span style="background: {bg}; color: {fg}; font-size: 0.78rem; font-weight: 600; padding: 2px 8px; border-radius: 6px;">
                                    {icon} {html.escape(src.source_type)}
                                </span>
                            </div>
                            <span style="color: #16A34A; font-weight: 700; font-size: 0.88rem;">{score_pct}% match</span>
                        </div>
                        <div style="color: #334155; font-size: 0.92rem; line-height: 1.55;">
                            {html.escape(src.content)}
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# VIEW ROUTING BASED ON SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

# 1. COMMAND CENTER (Process Recording + Active Dashboard)
if nav_selection == "Command Center":
    st.markdown("## ✦ Command Center")
    st.markdown("Upload meeting audio or video to run automated **transcription**, **speaker diarization**, **AI summarization**, and **action item extraction**.")

    all_supported = sorted(list(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS))
    uploaded_file = st.file_uploader(
        "Drop meeting recording here or click to browse",
        type=all_supported,
        help=f"Supported formats: {', '.join(all_supported).upper()}",
    )

    with st.expander("🎯 Accuracy Benchmark / Ground-Truth Reference (Optional)"):
        reference_text = st.text_area(
            "Paste Ground-Truth Reference Transcript",
            placeholder="Paste reference text here to compute real-time WER, accuracy, substitutions, deletions, and insertions…",
            height=80,
        )

    start_button = st.button("🚀 Process Recording", type="primary", disabled=uploaded_file is None)

    if start_button and uploaded_file is not None:
        tmp_input: str | None = None
        tmp_wav: str | None = None

        try:
            # Stage 1: Validation
            with st.spinner("🔍 Validating file format and integrity…"):
                validator = FileValidator()
                is_valid, err_msg = validator.validate(uploaded_file)
                if not is_valid:
                    st.error(f"❌ **Validation Failed:** {err_msg}")
                    st.stop()

            # Stage 2: Save to Temp
            suffix = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f_in:
                f_in.write(uploaded_file.getvalue())
                tmp_input = f_in.name

            # Stage 3: Audio Conversion
            with st.spinner("🔊 Converting audio to 16 kHz Mono WAV…"):
                processor = AudioProcessor()
                tmp_wav = processor.process(tmp_input)

            with wave.open(tmp_wav, "rb") as wf:
                audio_duration_sec = wf.getnframes() / wf.getframerate()

            st.success(f"✅ Audio processed: {audio_duration_sec:.1f}s ({audio_duration_sec/60:.1f} min)")

            # Stage 4: Diarization
            speaker_turns = []
            num_speakers_detected = 1
            if enable_diarization:
                with st.spinner("👥 Analyzing voices & clustering speaker embeddings…"):
                    try:
                        diarizer = get_diarizer(hf_token=hf_token.strip() if hf_token else None)
                        n_spk = int(expected_speakers_opt) if expected_speakers_opt != "Auto" else None
                        speaker_turns = diarizer.diarize(tmp_wav, num_speakers=n_spk)
                        detected_set = set(t["speaker"] for t in speaker_turns)
                        num_speakers_detected = max(1, len(detected_set))
                        st.success(f"✅ Diarization complete ({num_speakers_detected} speakers detected)")
                    except Exception as exc:
                        logger.warning("Diarization note: %s", exc)
                        st.warning(f"⚠️ Diarization note: {exc}")

            # Stage 5: Whisper
            with st.spinner(f"🤖 Transcribing speech with Whisper ({model_name} model)…"):
                transcriber = get_transcriber(model_name)
                result = transcriber.transcribe(tmp_wav)

            raw_text = result.get("text", "").strip()
            segments = result.get("segments", [])
            language = result.get("language", "unknown")
            st.success("✅ Whisper transcription completed")

            if not raw_text or len(raw_text.split()) < 3:
                st.error("❌ No meaningful speech detected in recording.")
                st.stop()

            # Stage 6: Speaker Alignment
            aligned_turns = []
            speaker_stats = {}
            speaker_transcript_doc = ""
            if enable_diarization and speaker_turns:
                with st.spinner("🔗 Aligning Whisper segments with speaker turns…"):
                    aligned_turns = align_whisper_with_speakers(segments, speaker_turns)
                    speaker_stats = compute_speaker_statistics(aligned_turns, audio_duration_sec)
                    speaker_transcript_doc = format_speaker_transcript(aligned_turns)

            # Stage 7: LLM Intelligence
            meeting_intel = None
            saved_meeting_id = None
            if enable_milestone2 and MILESTONE2_AVAILABLE:
                with st.spinner("✨ Extracting Meeting Intelligence & Action Items via Gemini…"):
                    try:
                        saved_meeting_id, meeting_intel = process_meeting(raw_text)
                        st.success("✅ Intelligence & Action Items extracted successfully")
                    except Exception as exc:
                        logger.warning("LLM extraction note: %s", exc)
                        st.warning(f"⚠️ LLM extraction note: {exc}")

            summarizer = MeetingSummarizer()
            m1_summary = summarizer.summarize(raw_text)

            # Stage 8: Role Inference
            distinct_speakers = [t["speaker"] for t in aligned_turns] if aligned_turns else []
            if meeting_intel and meeting_intel.participants:
                for p in meeting_intel.participants:
                    if p not in distinct_speakers:
                        distinct_speakers.append(p)

            speaker_roles = {}
            if (distinct_speakers or raw_text) and infer_speaker_roles:
                with st.spinner("💼 Inferring participant roles & designations…"):
                    try:
                        speaker_roles = infer_speaker_roles(raw_text, distinct_speakers)
                    except Exception as r_err:
                        logger.warning("Role inference note: %s", r_err)

            # Store in session state
            st.session_state["active_meeting_data"] = {
                "meeting_intel": meeting_intel,
                "m1_summary": m1_summary,
                "speaker_roles": speaker_roles,
                "aligned_turns": aligned_turns,
                "raw_text": raw_text,
                "segments": segments,
                "speaker_stats": speaker_stats,
                "language": language,
                "enable_diarization": enable_diarization,
                "reference_text": reference_text,
            }

            # Auto-index into vector store using the exact saved meeting ID
            if MILESTONE3_AVAILABLE and meeting_intel and saved_meeting_id:
                try:
                    idx_res = index_meeting(saved_meeting_id)
                    if idx_res.chunks_indexed > 0:
                        st.toast(f"🔍 Indexed {idx_res.chunks_indexed} chunks into knowledge base", icon="✅")
                except Exception as a_err:
                    logger.warning("Auto-indexing notice: %s", a_err)


            # Auto-save
            if auto_save:
                stem = Path(uploaded_file.name).stem
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                txt_p = TRANSCRIPT_DIR / f"{stem}_{ts}_transcript.txt"
                txt_p.write_text(raw_text, encoding="utf-8")
                st.info(f"💾 Transcript saved to `transcripts/{txt_p.name}`")

        finally:
            for p in [tmp_input, tmp_wav]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    # Render dashboard if active meeting data exists
    if "active_meeting_data" in st.session_state:
        st.markdown("---")
        st.subheader("📋 Meeting Intelligence Dashboard")
        render_intelligence_dashboard(st.session_state["active_meeting_data"])

# 2. MEETINGS (Browse & Search Repository)
elif nav_selection == "Meetings":
    st.markdown("## 📁 Meeting Knowledge Repository")
    st.markdown("Browse, inspect, and search across all historical meetings stored in the database.")

    if not MILESTONE3_AVAILABLE or not MeetingRepository:
        st.warning("⚠️ Meeting repository requires Milestone 3 modules.")
    else:
        repo = MeetingRepository()
        all_meetings = repo.get_all_meetings()

        if not all_meetings:
            st.info("No meetings currently saved in the repository. Process a recording in Command Center first!")
        else:
            col_sel, col_stat = st.columns([3, 1])
            with col_stat:
                st.metric("Total Meetings", len(all_meetings))

            with col_sel:
                meeting_options = {
                    f"{m.created_at.strftime('%Y-%m-%d %H:%M')} — {m.summary[:60]}… (ID: {m.id[:8]}…)": m.id
                    for m in all_meetings
                }
                selected_label = st.selectbox("Select a meeting to view details:", list(meeting_options.keys()))
                selected_mid = meeting_options[selected_label]

            selected_meeting = repo.get_meeting_by_id(selected_mid)
            if selected_meeting:
                st.markdown("---")
                st.markdown(f"### 📋 Meeting Details (`{selected_meeting.id}`)")
                st.caption(f"Recorded on: {selected_meeting.created_at.strftime('%Y-%m-%d %H:%M UTC')}")

                # Summary Card
                st.markdown(
                    f"""<div class="overview-card">
                        <strong>🎯 Executive Summary:</strong><br>
                        {html.escape(selected_meeting.summary)}
                    </div>""",
                    unsafe_allow_html=True,
                )

                m_tabs = st.tabs(["✅ Action Items", "📑 Decisions & Key Points", "📝 Transcript", "👥 Participants"])
                with m_tabs[0]:
                    if selected_meeting.action_items:
                        for ai in selected_meeting.action_items:
                            p_class = f"priority-{ai.priority.lower()}"
                            st.markdown(
                                f"""<div class="action-card">
                                    <strong>☑️ {html.escape(ai.task)}</strong> — <span class="{p_class}">{ai.priority}</span><br>
                                    <small>👤 Assignee: {html.escape(ai.assignee)} | ⏰ Deadline: {html.escape(ai.deadline or '—')} | 🔄 Status: {html.escape(ai.status)}</small>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No action items recorded.")

                with m_tabs[1]:
                    if selected_meeting.decisions:
                        st.markdown("**Agreed Decisions:**")
                        for d in selected_meeting.decisions:
                            st.markdown(f"<div class='decision-card'>✔️ {html.escape(d)}</div>", unsafe_allow_html=True)
                    if selected_meeting.key_points:
                        st.markdown("**Key Discussion Points:**")
                        for kp in selected_meeting.key_points:
                            st.markdown(f"• {kp}")

                with m_tabs[2]:
                    st.text_area("Transcript", value=selected_meeting.transcript, height=300)

                with m_tabs[3]:
                    if selected_meeting.participants:
                        p_pills = "".join(f"<span class='participant-pill'>👤 {html.escape(p.name)}</span>" for p in selected_meeting.participants)
                        st.markdown(p_pills, unsafe_allow_html=True)
                    else:
                        st.info("No participants recorded.")

        st.markdown("---")
        render_search_interface()

# 3. INTELLIGENCE (Executive Summary & Consensus)
elif nav_selection == "Intelligence":
    st.markdown("## 💡 Executive Meeting Intelligence")
    st.markdown("High-level executive overview, key discussion points, and agreed consensus.")

    if "active_meeting_data" in st.session_state:
        data = st.session_state["active_meeting_data"]
        meeting_intel: Optional[MeetingIntelligence] = data.get("meeting_intel")
        m1_summary = data.get("m1_summary", {})
        summary_text = meeting_intel.summary if meeting_intel else m1_summary.get("overview", "No summary available.")

        st.markdown(
            f"""<div class="overview-card">
                <strong>🎯 Active Executive Summary:</strong><br>
                {html.escape(summary_text)}
            </div>""",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📑 Key Discussion Points")
            if meeting_intel and meeting_intel.key_points:
                for pt in meeting_intel.key_points:
                    st.markdown(f"• {pt}")
            else:
                st.info("No key points available.")

        with col2:
            st.markdown("#### 🤝 Agreed Decisions & Consensus")
            if meeting_intel and meeting_intel.decisions:
                for d in meeting_intel.decisions:
                    st.markdown(f"<div class='decision-card'>✔️ {html.escape(d)}</div>", unsafe_allow_html=True)
            else:
                st.info("No decisions recorded.")
    else:
        st.info("Process a recording in **Command Center** to see live intelligence, or browse saved records in **Meetings**.")

# 4. ACTION HUB (Tasks & Deliverables)
elif nav_selection == "Action Hub":
    st.markdown("## ✅ Action Hub & Deliverables Tracker")
    st.markdown("Consolidated action items, assignees, deadlines, and execution status.")

    if "active_meeting_data" in st.session_state:
        data = st.session_state["active_meeting_data"]
        meeting_intel: Optional[MeetingIntelligence] = data.get("meeting_intel")
        speaker_roles = data.get("speaker_roles", {})

        if meeting_intel and meeting_intel.action_items:
            st.markdown(f"### 📋 Action Items Table ({len(meeting_intel.action_items)} Tasks)")
            table_html_rows = []
            for idx, item in enumerate(meeting_intel.action_items, start=1):
                a_role = speaker_roles.get(item.assignee, "")
                role_tag = f"<br><small style='color:#64748B;'>({html.escape(a_role)})</small>" if a_role and a_role != "Team Member / Coworker" else ""
                p_class = f"priority-{item.priority.lower()}"
                table_html_rows.append(
                    f"""<tr style="border-bottom: 1px solid #E2E8F0;">
                        <td style="padding: 12px 14px; font-weight:600; color:#0F172A; line-height:1.5; word-break:break-word;">{idx}. {html.escape(item.task)}</td>
                        <td style="padding: 12px 14px; color:#1E293B; word-break:break-word;">👤 {html.escape(item.assignee)}{role_tag}</td>
                        <td style="padding: 12px 14px; color:#475569; word-break:break-word;">⏰ {html.escape(item.deadline or '—')}</td>
                        <td style="padding: 12px 14px;"><span class="{p_class}">{html.escape(item.priority)}</span></td>
                        <td style="padding: 12px 14px; color:#334155; font-size:0.88rem;">{html.escape(item.status)}</td>
                    </tr>"""
                )

            st.markdown(
                f"""<div style="overflow-x:auto; width:100%; margin-bottom:24px; border:1px solid #CBD5E1; border-radius:10px; background:#FFFFFF;">
                    <table style="width:100%; border-collapse:collapse; font-size:0.95rem; text-align:left;">
                        <thead>
                            <tr style="background:#F8FAFC; border-bottom:2px solid #CBD5E1; color:#1E293B; font-weight:700;">
                                <th style="padding:12px 14px; width:45%;">Task Description</th>
                                <th style="padding:12px 14px; width:22%;">Assignee</th>
                                <th style="padding:12px 14px; width:15%;">Deadline</th>
                                <th style="padding:12px 14px; width:10%;">Priority</th>
                                <th style="padding:12px 14px; width:8%;">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(table_html_rows)}
                        </tbody>
                    </table>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("No action items extracted for the current recording.")
    else:
        st.info("No active meeting loaded. Process a recording in **Command Center** or view past tasks in **Meetings**.")

# 5. PEOPLE (Participants & Roles)
elif nav_selection == "People":
    st.markdown("## 👥 Participants & Inferred Roles")
    st.markdown("Participant directory, inferred professional designations, and speaker voice metrics.")

    if "active_meeting_data" in st.session_state:
        data = st.session_state["active_meeting_data"]
        meeting_intel = data.get("meeting_intel")
        speaker_roles = data.get("speaker_roles", {})
        speaker_stats = data.get("speaker_stats", {})
        aligned_turns = data.get("aligned_turns", [])

        distinct_speakers = [t["speaker"] for t in aligned_turns] if aligned_turns else []
        participants_list = meeting_intel.participants if (meeting_intel and meeting_intel.participants) else distinct_speakers

        if participants_list:
            st.markdown("#### 💼 Identified Team Members")
            cols = st.columns(min(3, max(1, len(participants_list))))
            for idx, p in enumerate(participants_list):
                col_i = idx % len(cols)
                role = speaker_roles.get(p, speaker_roles.get(p.strip(":"), "Team Member / Coworker"))
                with cols[col_i]:
                    st.markdown(
                        f"""<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:16px; margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                            <div style="font-weight:700; font-size:1.05rem; color:#1E293B;">👤 {html.escape(p)}</div>
                            <div style="color:#6366F1; font-weight:600; font-size:0.9rem; margin-top:4px;">💼 {html.escape(role)}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        if speaker_stats.get("speakers"):
            st.markdown("---")
            st.markdown("#### 📊 Voice Time & Speaking Distribution")
            for spk_name, s_data in speaker_stats["speakers"].items():
                st.write(f"**{spk_name}**: {s_data['percentage']}% ({s_data['speaking_time_formatted']} across {s_data['turn_count']} turns)")
                st.progress(s_data["percentage"] / 100.0)
    else:
        st.info("Process a meeting in **Command Center** to see participant role breakdowns and voice analytics.")

# 6. ASK INTELLIMEET (RAG Q&A)
elif nav_selection == "Ask IntelliMeet":
    st.markdown("## ✦ Ask IntelliMeet")
    st.markdown("Ask natural-language questions across your entire meeting knowledge repository. Responses are strictly grounded in stored recordings.")
    if not MILESTONE3_AVAILABLE:
        st.warning("⚠️ Ask IntelliMeet requires ChromaDB and Gemini configuration.")
    else:
        render_rag_interface()

# 7. VALIDATION (Quality & Benchmarking)
elif nav_selection == "Validation":
    st.markdown("## 🧪 Quality & Ground-Truth Benchmarking")
    st.markdown("Word Error Rate (WER) accuracy benchmark, file format verification, and system diagnostic status.")

    if "active_meeting_data" in st.session_state:
        data = st.session_state["active_meeting_data"]
        raw_text = data.get("raw_text", "")
        ref_text = data.get("reference_text", "")

        if ref_text.strip():
            metrics = calculate_wer_and_metrics(ref_text.strip(), raw_text)
            c1, c2, c3 = st.columns(3)
            c1.metric("Reference Words", metrics["ref_words"])
            c2.metric("Generated Words", metrics["hyp_words"])
            c3.metric("WER", f"{metrics['wer'] * 100:.2f}%")

            c4, c5, c6 = st.columns(3)
            c4.metric("Substitutions", metrics["substitutions"])
            c5.metric("Deletions", metrics["deletions"])
            c6.metric("Insertions", metrics["insertions"])

            acc = metrics["accuracy_pct"]
            if metrics["passed"]:
                st.success(f"**Accuracy:** {acc:.2f}%  |  **Required:** ≥90%  |  **Status:** ✅ PASS")
            else:
                st.error(f"**Accuracy:** {acc:.2f}%  |  **Required:** ≥90%  |  **Status:** ❌ FAIL")
        else:
            st.info("To test accuracy, paste a ground-truth reference text in the **Command Center** before processing.")
    else:
        st.info("No active recording data. Process a recording in **Command Center** with a reference transcript to view real-time accuracy benchmarks.")

    st.markdown("---")
    st.markdown("#### 🩺 System Diagnostics")
    d1, d2, d3 = st.columns(3)
    d1.metric("Whisper Engine", "Operational ✅")
    d2.metric("Gemini Intelligence", "Operational ✅" if MILESTONE2_AVAILABLE else "Disabled ⚠️")
    d3.metric("ChromaDB Vector Store", "Operational ✅" if MILESTONE3_AVAILABLE else "Disabled ⚠️")
