# 🎙️ Meeting Summarizer & Transcription

An AI-powered meeting transcription and summarization application built with **OpenAI Whisper**, **FFmpeg**, and **Streamlit**.

Uploads any meeting video or audio file, transcribes the speech using Whisper, filters out small talk, and generates a structured executive summary organized by discussion topics, action items, and deadlines.

> **Casual chatter is filtered from the generated summary while the complete raw transcript is preserved.**

---

## ✨ Features

- 🎬 **Universal Audio & Video Support** — `MP4`, `MKV`, `MOV`, `AVI`, `WebM`, `MP3`, `WAV`, `M4A`, `OGG`, `FLAC` (case-insensitive).
- 🔊 **Automatic Audio Extraction** — FFmpeg converts any input to 16 kHz mono WAV for optimal Whisper performance.
- 🤖 **Whisper Speech-to-Text** — Powered by OpenAI Whisper (`tiny` → `large`). Model is cached for the session using `@st.cache_resource` — no redundant reloading.
- 🧹 **Small-Talk Filtering** — Keyword-based classification removes pleasantries and mic checks from the generated summary. The raw transcript is always preserved in full.
- 📋 **Structured Executive Summary**:
  - 🎯 **Main Objective & Overview** — derived from highest-scoring sentences.
  - 📑 **Topics Discussed** — keyword-based topic classification into categories (Planning, Design, Budget, Technical, Testing, Marketing).
  - ✅ **Action Items** — sentences containing task-assignment patterns.
  - ⏰ **Deadlines & Milestones** — sentences containing explicit deadline patterns.
- 📊 **Validation & Accuracy Benchmarking** — WER calculation with punctuation/case/whitespace normalization via `jiwer`. Detailed substitution/deletion/insertion breakdown.
- 💾 **Separate Auto-save** — saves `_transcript.txt` and `_summary.txt` separately to `transcripts/`.

---

## 📁 Project Structure

```text
├── app.py                     # Streamlit web application
├── audio_processor.py         # FFmpeg audio extraction & 16kHz conversion
├── transcriber.py             # OpenAI Whisper model wrapper (lazy-loads model)
├── summarizer.py              # Keyword-based topic classifier & executive summary builder
├── validator.py               # File upload validation (extension, size, ffprobe stream check)
├── accuracy.py                # WER normalization and calculation engine (uses jiwer)
├── requirements.txt           # Python dependencies
│
├── tests/
│   ├── test_validator.py      # Validator test suite (68 tests)
│   ├── test_audio_processor.py
│   └── test_accuracy.py
│
├── evaluation/
│   ├── README.md              # Evaluation instructions
│   ├── accuracy_results.csv   # Real measured results (populated by run_evaluation.py)
│   ├── recordings/            # Place test recordings here
│   └── references/            # Place matching reference transcripts here
│
├── run_evaluation.py          # Batch accuracy evaluation script
├── README.md                  # This file
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Prerequisites

**Python 3.10+** and **FFmpeg** installed on your system:

| Platform | Install Command |
| :------- | :-------------- |
| Windows  | `winget install ffmpeg` |
| macOS    | `brew install ffmpeg` |
| Linux    | `sudo apt install ffmpeg` |

Verify: `ffmpeg -version` and `ffprobe -version` must both work.

### 2. Install Python Dependencies

```bash
git clone https://github.com/Srihari998/meeting-summary.git
cd meeting-summary
pip install -r requirements.txt
```

### 3. Run the Web Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧪 Running the Test Suite

```bash
python -m pytest tests/ -v
```

---

## 📊 Milestone 1 Validation

### Task 1 — Whisper Transcription

**Pipeline:**

```
Upload → Validate → FFmpeg (16 kHz mono WAV) → Whisper → Validate Transcript → Display
```

- `AudioProcessor.process()` uses `ffmpeg -vn -acodec pcm_s16le -ar 16000 -ac 1` to produce a clean 16 kHz mono WAV.
- `Transcriber.transcribe()` wraps `whisper.load_model()` with lazy loading (`_model` is `None` until first call).
- The Streamlit app caches the `Transcriber` instance with `@st.cache_resource(show_spinner=False)` keyed by model name — switching from `base` to `small` loads the correct model; repeated runs with the same model reuse it.
- The complete pipeline status is shown step-by-step (✅ or ❌) in the UI.

**Status: IMPLEMENTED**

---

### Task 2 — File Validation

**Supported formats:**

| Type   | Extensions                                     |
| :----- | :--------------------------------------------- |
| Video  | `mp4`, `mkv`, `mov`, `avi`, `webm` (+ uppercase) |
| Audio  | `mp3`, `wav`, `m4a`, `ogg`, `flac` (+ uppercase) |

**Rejected cases:**

| Condition                          | Error                                                                 |
| :--------------------------------- | :-------------------------------------------------------------------- |
| Unsupported extension (`.txt`, `.pdf`, `.jpg`, …) | "Unsupported format '…'. Supported formats: …" |
| Empty file or below 1 KB           | "… is too small (N bytes). The file may be empty or corrupt."        |
| File larger than 500 MB            | "… is too large (N MB). Maximum allowed is 500 MB."                  |
| Non-media renamed as `.mp4`        | ffprobe returns non-zero or empty streams → rejected                  |
| Video with no audio stream         | "The file does not contain any audio stream."                         |
| Corrupted/unreadable media         | "Unable to validate the media file. The file may be corrupted or unsupported." |
| **ffprobe missing**                | **"ffprobe is not available. Please install FFmpeg…"** ← fails explicitly |
| **Invalid ffprobe JSON output**    | **"Unable to validate the media file…"** ← rejected, not silently passed |

> **Important fix:** Previous version silently passed validation when ffprobe was missing or returned invalid JSON. Both cases now fail with clear, actionable messages.

**Status: IMPLEMENTED**

---

### Task 3 — Transcript Validation

After Whisper returns:

1. Checks `transcript_text` is not empty.
2. Checks it is not whitespace-only (`.strip()`).
3. Checks it contains at least 3 words (meaningful content threshold).
4. If invalid: shows ❌ error, does **not** proceed to summary generation, does **not** save.
5. If valid: shows ✅ with word count and detected language.

Auto-save (when enabled) writes:
- `transcripts/<name>_<timestamp>_transcript.txt` — raw transcript only
- `transcripts/<name>_<timestamp>_summary.txt` — executive summary only
- `transcripts/<name>_<timestamp>_metadata.json` — structured metadata

**Status: IMPLEMENTED**

---

### Task 4 — Streamlit Interface

| Element                          | Implementation                                          |
| :------------------------------- | :------------------------------------------------------ |
| File upload (drag & drop)        | `st.file_uploader()` with supported type list           |
| Video preview                    | `st.video()` for `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm` |
| Audio preview                    | `st.audio()` for `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac` |
| Transcription button             | `🎙️ Transcribe & Generate Summary`                     |
| Step-by-step processing status   | ✅/🔄/❌ per pipeline stage (validation → audio → whisper → transcript check → summary) |
| Transcript display               | Full text area + timestamped segments expander          |
| Summary display                  | Overview card, Topics tab, Actions tab, Deadlines tab   |
| Statistics display               | Word count, sentences, speaking time, language          |
| Download buttons                 | Separate raw transcript + summary download              |

**Status: IMPLEMENTED**

---

### Task 5 — Accuracy Testing

**WER Formula:**

```
Accuracy = max(0, 1 − WER) × 100%
```

**Normalization before comparison** (both reference and hypothesis):
1. Lowercase
2. Remove all punctuation
3. Normalize whitespace
4. Strip leading/trailing whitespace

**Tools:** `jiwer` (industry-standard WER alignment). Falls back to dynamic programming if unavailable.

**Detailed output in the Stats tab (when reference provided):**

| Metric        | Description                          |
| :------------ | :----------------------------------- |
| Reference Words | Word count after normalization      |
| Generated Words | Word count after normalization      |
| WER             | Word Error Rate                     |
| Substitutions   | Words changed                       |
| Deletions       | Words missing from hypothesis       |
| Insertions      | Extra words in hypothesis           |
| Accuracy        | `max(0, 1 − WER) × 100%`           |
| Status          | ✅ PASS (≥90%) or ❌ FAIL (<90%)   |

**Batch evaluation:** Run `python run_evaluation.py` to process all recordings in `evaluation/recordings/` against matching reference transcripts in `evaluation/references/`.

**Real measured results — ES2002a.wav (AMI Corpus, 21-minute kickoff meeting):**

| Recording | Model | Duration | Ref Words | Gen Words | Subs | Dels | Ins | WER | Accuracy | Pass/Fail |
|-----------|-------|----------|-----------|-----------|------|------|-----|-----|----------|-----------|
| ES2002a.wav | base | 21.2 min | 2,600 | 2,416 | 400 | 417 | 233 | 40.4% | 59.6% | FAIL |

**Reference:** Official AMI manual annotation v1.6.2 (multi-speaker, word-level XML).

**Analysis:** The `base` model achieves **59.6% accuracy** on this multi-speaker meeting. The 90% threshold is not met with `base`. Use `small` or `medium` Whisper model for higher accuracy on multi-speaker recordings.

**Status: TESTED WITH REAL DATA — base model: 59.6% accuracy**

---

## ⚙️ Automated Tests

```bash
python -m pytest tests/ -v
```

| Test File                 | Tests | Covers                                                           |
| :------------------------ | :---: | :--------------------------------------------------------------- |
| `tests/test_validator.py` |  36   | Extension validation, size checks, ffprobe edge cases, fake media |
| `tests/test_audio_processor.py` | 9 | Conversion output, 16kHz/mono, ffmpeg errors, missing input |
| `tests/test_accuracy.py`  |  23   | Normalization, identical/substitution/deletion/insertion, punctuation/case/whitespace tolerance |
| **Total**                 | **68** | **68/68 PASS** |

> Whisper models are **not downloaded** during tests. `AudioProcessor` and `Transcriber` are tested with mocking where model loading would be required.

---

## 📄 License

MIT License
