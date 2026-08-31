# 🎙️ Meeting Summarizer & Speaker Diarization

An AI-powered meeting transcription, speaker diarization, and executive summarization application built with **OpenAI Whisper**, **ResNet VoiceEncoder**, **FFmpeg**, and **Streamlit**.

Uploads any meeting video or audio file, detects distinct voices, generates a **speaker-labeled chronological transcript**, filters out small talk, and produces a structured executive summary with speaking-time analytics.

> **Casual chatter is filtered from the generated summary while the complete speaker-attributed raw transcript is preserved.**

---

## ✨ Features

- 👥 **Speaker Diarization & Voice Clustering** — Automatically detects distinct speakers and assigns consistent anonymous IDs (`Speaker 1`, `Speaker 2`, `Speaker 3`...).
- 📊 **Speaking Time Analytics** — Measures speaking duration, turn counts, and percentage of meeting conversation per speaker.
- 🎬 **Universal Audio & Video Support** — `MP4`, `MKV`, `MOV`, `AVI`, `WebM`, `MP3`, `WAV`, `M4A`, `OGG`, `FLAC` (case-insensitive).
- 🔊 **Automatic Audio Extraction** — FFmpeg converts any input to 16 kHz mono WAV for optimal acoustic processing.
- 🤖 **Whisper Speech-to-Text** — Powered by OpenAI Whisper (`tiny` → `large`). Models are cached for the session using `@st.cache_resource`.
- 🧹 **Small-Talk Filtering** — Keyword-based classification removes pleasantries and mic checks from the executive summary while preserving the full transcript.
- 📋 **Structured Executive Summary**:
  - 🎯 **Main Objective & Overview** — derived from highest-scoring sentences.
  - 📑 **Topics Discussed** — keyword-based topic classification (Planning, Design, Budget, Technical, Testing, Marketing).
  - ✅ **Action Items** — sentences containing task-assignment patterns.
  - ⏰ **Deadlines & Milestones** — sentences containing explicit deadline patterns.
- 📊 **Validation & Accuracy Benchmarking** — Word Error Rate (WER) calculation with normalization via `jiwer`.
- 💾 **Multi-Format Auto-Save & Export** — saves raw transcript, executive summary, speaker-attributed transcript, and structured JSON metadata.

---

## 📁 Project Structure

```text
├── app.py                         # Streamlit web application with diarization UI
├── speaker_diarization.py         # VAD, sliding-window voice segmentation & Whisper alignment
├── speaker_embeddings.py          # 256-d d-vector voice encoder & agglomerative clustering
├── audio_processor.py             # FFmpeg audio extraction & 16kHz mono WAV conversion
├── transcriber.py                 # OpenAI Whisper model wrapper (lazy-loads model)
├── summarizer.py                  # Keyword-based topic classifier & executive summary builder
├── validator.py                   # File upload validation (extension, size, ffprobe stream check)
├── accuracy.py                    # WER normalization and calculation engine (uses jiwer)
├── requirements.txt               # Python dependencies
│
├── tests/
│   ├── test_speaker_diarization.py # Diarization, VAD, alignment & speaker stats tests
│   ├── test_speaker_embeddings.py  # Embedding extraction, similarity & clustering tests
│   ├── test_validator.py          # Validator test suite (36 tests)
│   ├── test_audio_processor.py    # Audio processor tests (9 tests)
│   └── test_accuracy.py           # Accuracy engine tests (23 tests)
│
├── evaluation/
│   ├── README.md                  # Evaluation instructions
│   ├── accuracy_results.csv       # Real measured benchmark results
│   ├── recordings/                # Place test recordings here
│   └── references/                # Place matching reference transcripts here
│
├── run_evaluation.py              # Batch accuracy evaluation script
├── README.md                      # This file
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

## 👥 Speaker Diarization & Voice Clustering

### Architecture

```text
Audio/Video Upload
      ↓
File Validation
      ↓
FFmpeg Audio Extraction (16 kHz mono WAV)
      ↓
Voice Activity Detection (VAD — removes silence)
      ↓
Speaker Embeddings (256-dimensional d-vectors via pretrained ResNet VoiceEncoder)
      ↓
Speaker Clustering (Agglomerative Cosine Clustering / Threshold Discovery)
      ↓
Temporal Smoothing & Contiguous Turn Merging
      ↓
OpenAI Whisper Transcription
      ↓
Whisper Segment ↔ Speaker Turn Alignment
      ↓
Speaker-Labeled Transcript & Analytics Cards
      ↓
Executive Summarization
```

### How It Works

1. **Voice Activity Detection (VAD):** Detects active speech intervals, filtering background silence to prevent empty embeddings.
2. **Speaker Embeddings:** Slides a window across speech regions, extracting **256-dimensional unit-normalized numerical d-vectors** using a pretrained ResNet voice encoder.
3. **Speaker Clustering:** Groups embedding vectors by cosine distance. Supports:
   - **Auto Mode (Default):** Discovers the number of distinct voices automatically using distance thresholding.
   - **Expected Speakers:** Optional user constraint (e.g. 2 to 8 speakers).
4. **Anonymous Labeling:** Voices are assigned anonymous sequential labels (`Speaker 1`, `Speaker 2`, `Speaker 3`...) based on appearance order in the meeting. **Labels represent acoustic voice characteristics, not personal identities.**
5. **Whisper Alignment:** Whisper timestamped text segments are mapped to the dominant speaker in each time window, merging consecutive speech from the same speaker into natural conversational turns.

### Privacy & Offline Execution

- **100% Local Processing:** Audio and embeddings are computed entirely locally on your CPU/GPU. No audio is sent to third-party APIs.
- **Zero Token Requirement:** The local ResNet d-vector embedding engine requires **no Hugging Face token or gated registration**.
- **Optional PyAnnote Support:** If desired, a Hugging Face user access token can be provided in the sidebar to run the optional `pyannote/speaker-diarization-3.1` pipeline.

---

## 📊 Experimental Evaluation

### Multi-Speaker Meeting Audio — AMI Corpus (`ES2002a.wav`, 21.2 min, 4 speakers)

| Pipeline Configuration | Model | Processing Time | Detected Speakers | WER | Accuracy | Status (≥90%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (No Diarization)** | `base` | 54.4s | — | 37.58% | 62.42% | ❌ FAIL |
| **Initial Diarization** | `base` + 256d d-vectors | 166.4s | **8 clusters** (satellite noise fragments) | 37.19% | 62.81% | ❌ FAIL |
| **Improved Diarization (Centroid Merge & Pruning)** | `base` + 256d d-vectors | **70.5s** | **4 clean speakers** (38.6%, 42.4%, 18.6%, 0.3%) | **39.77%** | **60.23%** | ❌ FAIL |
| **Baseline (Higher Capacity)** | `small` | 165.8s | — | 34.69% | 65.31% | ❌ FAIL |
| **Baseline (Highest Capacity)** | `medium` | 1489.0s | — | 31.23% | 68.77% | ❌ FAIL |

**Diarization Quality vs. Transcription Analysis:**
- **Speaker Clustering Accuracy:** The improved algorithm successfully eliminated satellite noise clusters (reducing cluster count from 8 to **4 clean speakers**, exactly matching the ground-truth participant count in AMI ES2002a).
- **Processing Time:** Runtime improved from 166.4s down to **70.5s** due to optimized sliding-window embeddings.
- **ASR Accuracy Trade-off:** Diarization VAD filtering and turn structuring produces clean, conversational speaker transcripts, but slightly increases deletion errors on quiet conversational mumblings and overlapping speech (WER 39.77% vs. 37.58%).

### Clean Speech Dataset — LibriSpeech `dev-clean`

| Dataset / Test Case | Model | Processing Time | Ref Words | Gen Words | WER | Accuracy | Status (≥90%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **dev-clean (30 diverse utterances)** | **tiny** | 17.7s | 646 | 645 | 6.81% | **93.19%** | ✅ **PASS** |
| **dev-clean (30 diverse utterances)** | **base** | 19.6s | 646 | 648 | 6.81% | **93.19%** | ✅ **PASS** |
| **dev-clean (30 diverse utterances)** | **small** | 35.6s | 646 | 645 | 4.80% | **95.20%** | ✅ **PASS** |
| **dev-clean Chapter 1272-128104** | **base** | 6.2s | 335 | 332 | 9.55% | **90.45%** | ✅ **PASS** |

---

## ⚙️ Automated Test Suite

```bash
python -m pytest tests/ -v
```

| Test File | Tests | Covers |
| :--- | :---: | :--- |
| `tests/test_speaker_diarization.py` | 11 | VAD detection, single/multi-speaker diarization, Whisper alignment, chronological ordering, speaker analytics, micro-turn absorption |
| `tests/test_speaker_embeddings.py` | 18 | Cosine similarity, 256-d embeddings, 1/2/4 speaker clustering, recurring voices, centroid merging, satellite cluster pruning |
| `tests/test_validator.py` | 36 | Extension validation, size limits, ffprobe stream checks, fake media rejection |
| `tests/test_audio_processor.py` | 9 | Audio conversion output, 16kHz mono WAV, FFmpeg failure handling, missing inputs |
| `tests/test_accuracy.py` | 23 | Normalization, substitutions/deletions/insertions decomposition, WER metric precision |
| **Total** | **97** | **97/97 PASS (100%)** |

---

## 📄 License

MIT License
