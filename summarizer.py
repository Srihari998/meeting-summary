"""
summarizer.py
Clean, Context-Aware Meeting Summarizer & Topic Classifier

Features:
- Filters out small talk, casual banter, audio checks, and pleasantries.
- Identifies main meeting objective / project goal.
- Groups key discussion points by specific Topic headings.
- Extracts actionable tasks and explicit deadlines.
"""

from __future__ import annotations
import re
import math
from collections import Counter, defaultdict
from typing import Any


# ── Noise & Small Talk Filter ───────────────────────────────────────────────────
SMALL_TALK_PATTERNS = [
    r"\b(how are you|how('s| is) it going|how was your (weekend|day|trip|lunch))\b",
    r"\b(had lunch|have you eaten|did you eat|get coffee|grab a bite)\b",
    r"\b(can you hear me|am i audible|hear you loud and clear|you're muted|unmute)\b",
    r"\b(can you see my screen|sharing my screen|see the slides|see the powerpoint)\b",
    r"\b(good (morning|afternoon|evening)|nice to see you|thanks for joining|glad you could come)\b",
    r"\b(testing (one|1) (two|2)|mic check|audio check|sound check)\b",
    r"\b(see you later|bye everyone|have a good (one|day|weekend)|take care)\b",
    r"\b(give me a second|hold on a second|wait a minute|let me pull up)\b",
]
SMALL_TALK_RE = re.compile("|".join(SMALL_TALK_PATTERNS), re.IGNORECASE)

# ── Stop Words ──────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","shall","can",
    "this","that","these","those","i","we","you","he","she","it","they",
    "my","our","your","his","her","its","their","what","which","who","when",
    "where","how","why","not","no","yes","so","if","as","by","from","up",
    "about","into","through","during","also","just","then","than","more",
    "very","all","any","both","each","few","most","other","some","such",
    "only","own","same","too","s","t","now","here","there","well","right",
    "good","okay","ok","um","uh","like","know","think","get","got","let",
    "make","go","see","said","say","tell","talk","mean","thing","things",
    "going","want","need","one","two","three","four","five","actually","really",
}

# ── Action & Task Patterns ─────────────────────────────────────────────────────
ACTION_RE = re.compile(
    r"\b(will|should|need to|needs to|must|have to|going to|plan to|"
    r"responsible for|assigned to|action item|follow up|deliver|prepare|"
    r"submit|complete|finish|review|update|provide|create|develop|"
    r"schedule|coordinate|organize|test|implement|send|share|confirm)\b",
    re.IGNORECASE,
)

# ── Deadline Patterns ───────────────────────────────────────────────────────────
DEADLINE_RE = re.compile(
    r"\b(before\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"tomorrow|next week|end of\s+\w+|eod|eow|the deadline|\d{1,2}\w*)|"
    r"by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"tomorrow|next week|end of\s+\w+|eod|eow|\d{1,2}[\/\-]\d{1,2}|\d{1,2}\s+(am|pm))|"
    r"due\s+(on|by|before|in)\s+[\w\s]+|deadline\s+(is|of|set for)\s+[\w\s]+|"
    r"within\s+\d+\s+(days?|weeks?|hours?))\b",
    re.IGNORECASE,
)

# ── Predefined Topic Keywords (Domain Classification) ──────────────────────────
TOPIC_DICTIONARY = {
    "Project Planning & Roadmap": [
        "project", "plan", "roadmap", "kickoff", "milestone", "phase", "agenda",
        "scope", "objective", "timeline", "schedule", "deliverable", "goals"
    ],
    "Design & User Interface": [
        "design", "ui", "ux", "interface", "wireframe", "prototype", "layout",
        "look", "feel", "colors", "buttons", "screen", "frontend", "style"
    ],
    "Budget & Financials": [
        "budget", "cost", "price", "expense", "financial", "fund", "money",
        "dollar", "allocation", "estimate", "pricing", "resources"
    ],
    "Technical Architecture & Development": [
        "code", "database", "backend", "api", "server", "architecture", "system",
        "module", "data", "software", "hardware", "tool", "integration", "infrastructure"
    ],
    "Testing, Quality & Review": [
        "test", "testing", "qa", "review", "feedback", "bug", "issue", "verify",
        "validation", "performance", "check", "assessment", "evaluation"
    ],
    "Marketing & Client Requirements": [
        "market", "marketing", "client", "customer", "user", "requirement",
        "sales", "presentation", "demo", "release", "launch", "competitor"
    ],
}


class MeetingSummarizer:
    """
    Produces clean executive summaries categorized by discussion topics.
    """

    def __init__(self, min_sent_words: int = 4) -> None:
        self.min_sent_words = min_sent_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, transcript: str) -> dict[str, Any]:
        if not transcript or not transcript.strip():
            return self._empty()

        # Step 1: Split sentences
        raw_sentences = self._split_sentences(transcript)

        # Step 2: Filter noise and small talk
        meaningful_sentences = [
            s for s in raw_sentences
            if len(s.split()) >= self.min_sent_words and not self._is_small_talk(s)
        ]

        if not meaningful_sentences:
            return self._empty()

        # Step 3: Extract structured components
        overview = self._build_executive_overview(meaningful_sentences)
        topic_groups = self._group_by_topics(meaningful_sentences)
        action_items = self._extract_action_items(meaningful_sentences)
        deadlines = self._extract_deadlines(meaningful_sentences)
        stats = self._compute_stats(transcript, meaningful_sentences)

        return {
            "overview": overview,
            "topic_groups": topic_groups,
            "action_items": action_items,
            "deadlines": deadlines,
            "stats": stats,
        }

    # ------------------------------------------------------------------
    # Small Talk Filter
    # ------------------------------------------------------------------

    @staticmethod
    def _is_small_talk(sentence: str) -> bool:
        s = sentence.lower().strip()
        # Direct regex match
        if SMALL_TALK_RE.search(s):
            return True
        # Common ultra-short pleasantries
        if s in {"hi", "hello", "good morning", "okay", "alright", "yeah", "sure", "thanks", "thank you"}:
            return True
        return False

    # ------------------------------------------------------------------
    # Executive Overview Builder
    # ------------------------------------------------------------------

    def _build_executive_overview(self, sentences: list[str]) -> str:
        # Score sentences based on substantive keyword density
        scores = self._score_sentences(sentences)
        if not scores:
            return "The meeting discussion was transcribed successfully."

        # Get top 2-3 most informative substantive sentences
        ranked_indices = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(ranked_indices[:3])
        top_points = [self._clean(sentences[i]) for i in top_indices]

        return " ".join(top_points)

    # ------------------------------------------------------------------
    # Topic Grouping Engine
    # ------------------------------------------------------------------

    def _group_by_topics(self, sentences: list[str]) -> list[dict[str, Any]]:
        """
        Groups sentences by matching keywords into distinct topic themes.
        Returns a list of dicts: [{"topic": "...", "points": ["...", "..."]}]
        """
        groups: dict[str, list[str]] = defaultdict(list)
        unclassified: list[str] = []

        for sent in sentences:
            words = set(re.findall(r"\b[a-zA-Z]+\b", sent.lower()))
            matched_topic = None
            max_matches = 0

            for topic, keywords in TOPIC_DICTIONARY.items():
                match_count = len(words.intersection(set(keywords)))
                if match_count > max_matches:
                    max_matches = match_count
                    matched_topic = topic

            if matched_topic and max_matches >= 1:
                cleaned = self._clean(sent)
                if cleaned not in groups[matched_topic]:
                    groups[matched_topic].append(cleaned)
            else:
                cleaned = self._clean(sent)
                if cleaned not in unclassified:
                    unclassified.append(cleaned)

        results: list[dict[str, Any]] = []

        # Add classified topics (limit points per topic to top 3 for crispness)
        for topic, points in groups.items():
            if points:
                results.append({
                    "topic": topic,
                    "points": points[:3]
                })

        # If there are general points not captured by standard dictionary
        if unclassified and len(results) < 2:
            results.append({
                "topic": "General Discussion & Collaboration",
                "points": unclassified[:3]
            })

        return results

    # ------------------------------------------------------------------
    # Action Items & Deadlines Extraction
    # ------------------------------------------------------------------

    def _extract_action_items(self, sentences: list[str]) -> list[str]:
        items: list[str] = []
        seen: set[str] = set()

        for s in sentences:
            if ACTION_RE.search(s):
                c = self._clean(s)
                if c not in seen and len(c.split()) >= 4:
                    seen.add(c)
                    items.append(c)

        return items[:8]

    def _extract_deadlines(self, sentences: list[str]) -> list[str]:
        deadlines: list[str] = []
        seen: set[str] = set()

        for s in sentences:
            if DEADLINE_RE.search(s):
                c = self._clean(s)
                if c not in seen:
                    seen.add(c)
                    deadlines.append(c)

        return deadlines[:6]

    # ------------------------------------------------------------------
    # Scoring & Utilities
    # ------------------------------------------------------------------

    def _score_sentences(self, sentences: list[str]) -> list[float]:
        all_words: list[str] = []
        sent_words: list[list[str]] = []

        for s in sentences:
            wl = [
                w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", s)
                if w.lower() not in STOP_WORDS and len(w) > 2
            ]
            sent_words.append(wl)
            all_words.extend(wl)

        if not all_words:
            return [0.0] * len(sentences)

        tf = Counter(all_words)
        max_tf = max(tf.values()) or 1
        N = len(sentences)
        df = Counter()
        for wl in sent_words:
            df.update(set(wl))

        scores: list[float] = []
        for wl in sent_words:
            if not wl:
                scores.append(0.0)
                continue
            sc = sum((tf[w] / max_tf) * math.log((N + 1) / (df[w] + 1)) for w in wl)
            scores.append(sc / (len(wl) ** 0.3))

        return scores

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        raw = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _clean(s: str) -> str:
        s = s.strip()
        if s and s[-1] not in ".!?":
            s += "."
        return s[0].upper() + s[1:] if s else ""

    @staticmethod
    def _compute_stats(text: str, sentences: list[str]) -> dict[str, Any]:
        words = text.split()
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "reading_time": f"{max(1, math.ceil(len(words) / 180))} min",
            "speaking_time": f"{max(1, math.ceil(len(words) / 130))} min",
        }

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "overview": "No clear substantive points could be extracted.",
            "topic_groups": [],
            "action_items": [],
            "deadlines": [],
            "stats": {"word_count": 0, "sentence_count": 0, "reading_time": "0 min", "speaking_time": "0 min"},
        }
