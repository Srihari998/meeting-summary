"""
tests/test_accuracy.py
Pytest test suite for Task 5 — WER Accuracy Engine

Tests:
  - Identical transcripts → 100% accuracy
  - Substitution errors
  - Deletion errors
  - Insertion errors
  - Punctuation differences (should not penalize)
  - Capitalization differences (should not penalize)
  - Whitespace differences (should not penalize)
  - Empty reference
  - Empty hypothesis
  - normalize_transcript function
"""

from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from accuracy import normalize_transcript, calculate_wer_and_metrics


# ── normalize_transcript ──────────────────────────────────────────────────────

class TestNormalization:
    def test_lowercase(self):
        assert normalize_transcript("Hello World") == "hello world"

    def test_punctuation_removed(self):
        result = normalize_transcript("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_extra_whitespace_collapsed(self):
        result = normalize_transcript("hello   world\t\nnow")
        assert "  " not in result
        assert result == "hello world now"

    def test_leading_trailing_stripped(self):
        result = normalize_transcript("  hello world  ")
        assert result == "hello world"

    def test_empty_string(self):
        assert normalize_transcript("") == ""

    def test_quotes_removed(self):
        result = normalize_transcript('"hello" \'world\'')
        assert '"' not in result
        assert "'" not in result


# ── WER Accuracy Calculation ──────────────────────────────────────────────────

class TestWerCalculation:
    def test_identical_transcripts_100_percent(self):
        ref = "the meeting will start at nine am"
        hyp = "the meeting will start at nine am"
        result = calculate_wer_and_metrics(ref, hyp)
        assert result["wer"] == 0.0
        assert result["accuracy_pct"] == 100.0
        assert result["passed"] is True

    def test_single_substitution(self):
        ref = "the project starts on Monday"
        hyp = "the project starts on Tuesday"
        result = calculate_wer_and_metrics(ref, hyp)
        assert result["wer"] > 0.0
        assert result["accuracy_pct"] < 100.0

    def test_single_deletion(self):
        ref = "please submit the report before Friday"
        hyp = "please submit report before Friday"
        result = calculate_wer_and_metrics(ref, hyp)
        assert result["wer"] > 0.0

    def test_single_insertion(self):
        ref = "the team reviewed the budget"
        hyp = "the team carefully reviewed the budget"
        result = calculate_wer_and_metrics(ref, hyp)
        assert result["wer"] > 0.0

    def test_punctuation_difference_no_penalty(self):
        """Punctuation must not inflate WER."""
        ref = "Hello, world. How are you?"
        hyp = "Hello world How are you"
        result = calculate_wer_and_metrics(ref, hyp)
        assert result["wer"] == 0.0

    def test_capitalization_difference_no_penalty(self):
        """Case differences must not inflate WER."""
        ref = "The Project Manager Confirmed the Budget."
        hyp = "the project manager confirmed the budget"
        result = calculate_wer_and_metrics(ref, hyp)
        assert result["wer"] == 0.0

    def test_whitespace_difference_no_penalty(self):
        """Extra spaces/newlines must not inflate WER."""
        ref = "confirm  the  plan"
        hyp = "confirm the plan"
        result = calculate_wer_and_metrics(ref, hyp)
        assert result["wer"] == 0.0

    def test_empty_reference(self):
        """Empty reference with non-empty hypothesis → wer=1.0."""
        result = calculate_wer_and_metrics("", "some words")
        assert result["wer"] == 1.0
        assert result["passed"] is False

    def test_both_empty(self):
        """Both empty → 0 WER, 100% accuracy."""
        result = calculate_wer_and_metrics("", "")
        assert result["wer"] == 0.0
        assert result["accuracy_pct"] == 100.0
        assert result["passed"] is True

    def test_empty_hypothesis(self):
        """Empty hypothesis with non-empty reference → high WER."""
        result = calculate_wer_and_metrics("hello world meeting", "")
        assert result["wer"] > 0.0
        assert result["passed"] is False

    def test_pass_threshold_at_90(self):
        result = calculate_wer_and_metrics("a b c d e f g h i j", "a b c d e f g h i j")
        assert result["passed"] is True

    def test_fail_threshold_below_90(self):
        """Significantly wrong transcript should fail the 90% threshold."""
        ref = "the quick brown fox jumps over the lazy dog"
        hyp = "completely different words here with no match at all"
        result = calculate_wer_and_metrics(ref, hyp)
        assert result["passed"] is False

    def test_result_keys_present(self):
        result = calculate_wer_and_metrics("hello world", "hello world")
        expected_keys = {
            "ref_words", "hyp_words", "substitutions",
            "deletions", "insertions", "wer", "accuracy_pct", "passed"
        }
        assert expected_keys.issubset(result.keys())

    def test_ref_word_count(self):
        result = calculate_wer_and_metrics("one two three four five", "one two three four five")
        assert result["ref_words"] == 5

    def test_hyp_word_count(self):
        result = calculate_wer_and_metrics("one two three", "one two three four")
        assert result["hyp_words"] == 4
