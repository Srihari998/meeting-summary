# 🎙️ Meeting Transcription & Executive Summary Tool

An AI-powered meeting transcription and summarization application built with **OpenAI Whisper**, **FFmpeg**, and **Streamlit**.

It automatically extracts audio from any uploaded video or audio meeting recording, transcribes speech with high accuracy, filters out small talk/chatter, and generates a structured executive summary organized by discussion topics, action items, and deadlines.

---

## ✨ Features

- 🎬 **Universal Audio & Video Support**: Supports `MP4`, `MKV`, `MOV`, `AVI`, `WebM`, `MP3`, `WAV`, `M4A`, `OGG`, and `FLAC`.
- 🔊 **Automatic Audio Extraction**: Uses FFmpeg to convert video and audio files into 16kHz mono WAV for optimal transcription.
- 🤖 **Whisper Speech-to-Text with Resource Caching**: Uses `@st.cache_resource` for fast inference without reloading model weights on each run.
- 🧹 **Small-Talk & Noise Filtering**: Automatically detects and strips casual chatter, pleasantries (*"did you have lunch"*, *"can you hear me"*), and mic checks.
- 📋 **Structured Executive Summary**:
  - 🎯 **Main Objective & Overview**: Core purpose and summary of the meeting.
  - 📑 **Topics Discussed**: Discussion points organized under clear topic categories.
  - ✅ **Action Items**: Explicitly assigned tasks and deliverables.
  - ⏰ **Deadlines & Milestones**: Extracted dates, days, and submission deadlines.
- 📊 **Validation & Accuracy Benchmarking**: Built-in file stream validation and Word Error Rate (WER) accuracy calculation via `jiwer`.
- 💾 **Export & Auto-save**: Auto-saves transcripts (`_transcript.txt`), executive summaries (`_summary.txt`), and structured metadata JSON.

---

## 📁 Project Structure

```text
├── app.py                     # Streamlit web application UI (with model caching & jiwer)
├── audio_processor.py         # FFmpeg audio conversion & extraction
├── transcriber.py             # OpenAI Whisper model wrapper
├── summarizer.py              # Topic clustering & executive summary generator
├── validator.py               # File upload & audio stream validation
├── requirements.txt           # Python package dependencies (streamlit, openai-whisper, jiwer)
│
├── run_all_tests.py           # Master Milestone 1 test runner
├── test_task1_workflow.py     # Task 1: Complete Whisper transcription workflow test
├── test_task2_validation.py   # Task 2: File upload validation & negative test suite
├── test_task3_transcript.py   # Task 3: Transcript validation & auto-save integrity test
├── test_task5_accuracy.py     # Task 5: Accuracy benchmark & batch WER evaluation
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+**
- **FFmpeg** installed and added to your system `PATH`:
  - **Windows**: `winget install ffmpeg`
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg`

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Srihari998/meeting-summary.git
cd meeting-summary
pip install -r requirements.txt
```

### 3. Running the Web Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Milestone 1 Test Suite & Verification Evidence

All Milestone 1 tasks come with dedicated automated test scripts:

| Task | Test Script | Description |
| :--- | :--- | :--- |
| **Task 1** | `python test_task1_workflow.py` | Tests end-to-end ffmpeg conversion $\rightarrow$ Whisper transcription $\rightarrow$ segments $\rightarrow$ language detection. |
| **Task 2** | `python test_task2_validation.py` | Verifies rejection of 0-byte files, invalid extensions (`.pdf`, `.exe`), fake disguised media, and verifies error messages. |
| **Task 3** | `python test_task3_transcript.py` | Verifies non-empty transcript generation, auto-saving of `_summary.txt` + `_transcript.txt` + `_metadata.json`, and verifies disk file integrity. |
| **Task 5** | `python test_task5_accuracy.py` | Evaluates Word Error Rate (WER) using `jiwer` and verifies $\ge 90\%$ accuracy threshold. |

### Run All Milestone 1 Tests at Once:
```bash
python run_all_tests.py
```

---

## 📄 License
MIT License
