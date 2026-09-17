"""
Summarizer Module
Contains prompt-side logic, summarization guidelines, and validation for meeting summary,
key points, and decisions. In Milestone 2, summarization extraction is consolidated
into the single `meeting_intelligence_prompt.txt` call for maximum efficiency
and context coherence.
"""

from typing import List, Tuple
try:
    from schemas import MeetingIntelligence
except ImportError:
    from .schemas import MeetingIntelligence


# Summarization guidelines shaping LLM extraction prompt instructions
SUMMARIZATION_GUIDELINES: dict = {
    "summary": (
        "Write a concise 2-4 sentence executive overview of the meeting. "
        "Focus on the primary purpose, core outcomes, and strategic alignment. "
        "Maintain an objective, professional tone and omit conversational filler or pleasantries."
    ),
    "key_points": (
        "Extract 2-6 distinct, high-signal discussion topics. "
        "Each point must capture a substantive update, problem debated, or technical finding."
    ),
    "decisions": (
        "Identify all definitive consensus points, architecture choices, policy agreements, "
        "or timeline approvals. Exclude speculative ideas or open questions that were not resolved."
    ),
    "exclusions": (
        "Exclude small talk, greetings, administrative audio-check banter, "
        "and non-actionable tangents."
    )
}


def get_summarization_prompt_instructions() -> str:
    """
    Returns formatted summarization guidelines to be embedded into the extraction prompt.
    """
    return (
        f"SUMMARIZATION GUIDELINES:\n"
        f"- Summary: {SUMMARIZATION_GUIDELINES['summary']}\n"
        f"- Key Points: {SUMMARIZATION_GUIDELINES['key_points']}\n"
        f"- Decisions: {SUMMARIZATION_GUIDELINES['decisions']}\n"
        f"- What to Exclude: {SUMMARIZATION_GUIDELINES['exclusions']}\n"
    )


def validate_summary_components(intelligence: MeetingIntelligence) -> Tuple[bool, List[str]]:
    """
    Validates that the summarization section contains non-empty content
    and consistent bullet points.
    """
    issues: List[str] = []
    if not intelligence.summary or not intelligence.summary.strip():
        issues.append("Summary text is empty.")
    if not intelligence.key_points:
        issues.append("No key points were identified.")
    
    return len(issues) == 0, issues