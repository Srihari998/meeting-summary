# 🎙️ Meeting Transcription & Executive Summary Tool

An AI-powered meeting transcription and summarization application built with **OpenAI Whisper**, **FFmpeg**, and **Streamlit**.

It automatically extracts audio from any uploaded video or audio meeting recording, transcribes speech with high accuracy, filters out small talk/chatter, and generates a structured executive summary organized by discussion topics, action items, and deadlines.

---

## ✨ Features

- 🎬 **Universal Audio & Video Support**: Supports `MP4`, `MKV`, `MOV`, `AVI`, `WebM`, `MP3`, `WAV`, `M4A`, `OGG`, and `FLAC`.
- 🔊 **Automatic Audio Extraction**: Uses FFmpeg to convert video and audio files into 16kHz mono WAV for optimal transcription.
- 🤖 **Whisper Speech-to-Text**: Powered by OpenAI's Whisper model (supports `tiny`, `base`, `small`, `medium`, `large`).
- 🧹 **Small-Talk & Noise Filtering**: Automatically detects and strips casual chatter, pleasantries (*"did you have lunch"*, *"can you hear me"*), and mic checks.
- 📋 **Structured Executive Summary**:
  - 🎯 **Main Objective & Overview**: Core purpose and summary of the meeting.
  - 📑 **Topics Discussed**: Discussion points organized under clear topic categories.
  - ✅ **Action Items**: Explicitly assigned tasks and deliverables.
  - ⏰ **Deadlines & Milestones**: Extracted dates, days, and submission deadlines.
- 📊 **Validation & Accuracy Check**: Built-in file stream validation and Word Error Rate (WER) accuracy calculation against reference transcripts.
- 💾 **Export & Auto-save**: Auto-saves transcripts with metadata and provides single-click download for both **Executive Summary** and **Raw Transcript**.

---

## 📁 Project Structure

```text
task-1/
├── app.py                         # Streamlit web application UI
├── audio_processor.py             # FFmpeg audio conversion & extraction
├── transcriber.py                 # OpenAI Whisper model wrapper
├── summarizer.py                  # Topic clustering & executive summary generator
├── validator.py                   # File upload & audio stream validation
├── accuracy_test.py               # CLI tool for WER accuracy benchmarking
├── test_transcription_workflow.py # CLI end-to-end pipeline test script
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
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

## 🧪 CLI Testing

You can also run tests without opening the browser:

```bash
# Test the complete workflow on a sample file:
python test_transcription_workflow.py sample_recording.wav

# Run accuracy testing with a reference transcript:
python accuracy_test.py sample_recording.wav "expected transcript text"
```

---

## 📄 License
MIT License
