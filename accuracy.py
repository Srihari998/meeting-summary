"""
accuracy.py
Task 5 -- Word Error Rate (WER) & Transcription Accuracy Engine

Standardized evaluation utilities:
- Text normalization (lowercasing, punctuation stripping, whitespace normalization)
- Industry-standard WER alignment via jiwer
- Word-level accuracy derived from WER: Accuracy = max(0, 1 - WER) * 100%
- Detailed substitution, deletion, and insertion metrics
"""

from __future__ import annotations
import re
from typing import Any

try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False


def normalize_transcript(text: str) -> str:
    """
    Standard text normalization for fair speech-to-text evaluation:
    1. Convert to lowercase
    2. Remove punctuation marks (periods, commas, question marks, quotes, etc.)
    3. Normalize all whitespace sequences to a single space
    4. Strip leading/trailing whitespace
    """
    if not text:
        return ""
    # Lowercase
    normalized = text.lower()
    # Remove punctuation
    normalized = re.sub(r"[^\w\s]", "", normalized)
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def calculate_wer_and_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    """
    Compute normalized WER and full error decomposition (Substitutions, Deletions, Insertions).

    Returns
    -------
    dict with keys:
        ref_words     (int)   : Word count in normalized reference
        hyp_words     (int)   : Word count in normalized hypothesis
        substitutions (int)   : Substituted words
        deletions     (int)   : Deleted/missing words
        insertions    (int)   : Inserted extra words
        wer           (float) : Word Error Rate (0.0 to 1.0+)
        accuracy_pct  (float) : Accuracy % = max(0, 1 - WER) * 100
        passed        (bool)  : True if accuracy >= 90.0%
    """
    norm_ref = normalize_transcript(reference)
    norm_hyp = normalize_transcript(hypothesis)

    ref_words = norm_ref.split() if norm_ref else []
    hyp_words = norm_hyp.split() if norm_hyp else []

    ref_count = len(ref_words)
    hyp_count = len(hyp_words)

    # Edge cases
    if ref_count == 0:
        if hyp_count == 0:
            return {
                "ref_words": 0, "hyp_words": 0,
                "substitutions": 0, "deletions": 0, "insertions": 0,
                "wer": 0.0, "accuracy_pct": 100.0, "passed": True
            }
        return {
            "ref_words": 0, "hyp_words": hyp_count,
            "substitutions": 0, "deletions": 0, "insertions": hyp_count,
            "wer": 1.0, "accuracy_pct": 0.0, "passed": False
        }

    if HAS_JIWER:
        out = jiwer.process_words(norm_ref, norm_hyp)
        substitutions = out.substitutions
        deletions = out.deletions
        insertions = out.insertions
        wer = float(out.wer)
    else:
        # Dynamic programming Levenshtein alignment on words
        n, m = ref_count, hyp_count
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        wer = float(dp[n][m] / n)
        substitutions = 0
        deletions = 0
        insertions = 0

    accuracy_pct = max(0.0, (1.0 - wer) * 100.0)
    passed = accuracy_pct >= 90.0

    return {
        "ref_words": ref_count,
        "hyp_words": hyp_count,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "wer": round(wer, 4),
        "accuracy_pct": round(accuracy_pct, 2),
        "passed": passed,
    }
