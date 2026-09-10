# 🎙️ End-to-End System Architecture

## 📌 1. Overview & Core Mission
The **Meeting Intelligence & Speaker Diarization Platform** is an end-to-end AI system that converts raw multi-speaker video and audio recordings into structured, actionable business intelligence. 

It seamlessly combines:
- **Audio Preprocessing & Validation** (FFmpeg & ffprobe container stream analysis)
- **Speech-to-Text ASR** (OpenAI Whisper continuous stream transcription)
- **Speaker Diarization & Voice Clustering** (WebRTC VAD + 256-d ResNet d-vectors)
- **Speaker-to-Transcript Alignment & Role Inference** (Temporal overlap matching & NLP job designation extraction)
- **Generative AI Meeting Intelligence** (Google Gemini LLM for executive summaries, key decisions, and action items with Pydantic validation)
- **Interactive UI Dashboard** (Streamlit with speaker metrics, responsive action item tables, and multi-format export)

---

## 🏗️ 2. Architectural Pipeline Diagram

```
                       ┌────────────────────────────────────────┐
                       │   Raw Audio / Video Upload (.mp4/.wav) │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  Stage 1: Validation & Audio Extract   │
                       │  • FileValidator (Container & streams) │
                       │  • FFmpeg: 16 kHz Mono WAV conversion  │
                       └───────────────────┬────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │                                           │
                     ▼                                           ▼
       ┌───────────────────────────┐               ┌───────────────────────────┐
       │ Stage 2A: Continuous ASR  │               │ Stage 2B: Voice Clustering│
       │ • OpenAI Whisper Engine   │               │ • WebRTC VAD (Voice Detect│
       │ • Full-stream timestamps  │               │ • 256-d ResNet d-vectors  │
       │ • Sub-word language model │               │ • Agglomerative Cluster   │
       └─────────────┬─────────────┘               │ • Centroid Merging (≥0.88)│
                     │                             └─────────────┬─────────────┘
                     └─────────────────────┬─────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  Stage 3: Time-Overlap Alignment       │
                       │  • Align Whisper segments with turns   │
                       │  • Micro-turn smoothing (<0.8s filter) │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  Stage 4: Role & Position Inference    │
                       │  • Detects designations from dialog:   │
                       │    (Project Manager, Designer, Lead...)│
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  Stage 5: GenAI Intelligence (Gemini)  │
                       │  • Executive Summary synthesis         │
                       │  • Key Discussion Points & Decisions   │
                       │  • Action Items Table (Task/Assignee)  │
                       │  • Pydantic validation + Auto-Retry    │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  Stage 6: UI Dashboard & Export Layer  │
                       │  • Streamlit Web Application           │
                       │  • Speaker Metrics (% talk time)       │
                       │  • Auto-save TXT & JSON reports        │
                       └────────────────────────────────────────┘
```

---

## 🧩 3. Component Details

### 1. Preprocessing & Container Validation
- **Files:** `validator.py`, `audio_processor.py`
- Enforces strict file validation (format whitelist, 1KB - 500MB boundary, `ffprobe` stream checks).
- Converts audio/video into standardized 16 kHz mono WAV via FFmpeg.

### 2. Speech Transcription Engine
- **Files:** `transcriber.py`
- Transcribes the continuous 16kHz audio stream using OpenAI Whisper (`tiny`, `base`, `small`, `medium`).
- Avoids boundary truncation and repetition loops by keeping the stream unbroken.

### 3. Speaker Diarization & Voice Clustering
- **Files:** `speaker_embeddings.py`, `speaker_diarization.py`
- Uses `webrtcvad` for speech activity detection.
- Extracts 256-dimensional unit-normalized voice embeddings using a local ResNet `VoiceEncoder`.
- Clusters voice vectors with Agglomerative Clustering and centroid merging ($\ge 0.88$ similarity) without requiring paid tokens.

### 4. Speaker Alignment & Role Inference
- **Files:** `speaker_diarization.py`, `milestone2/participants.py`
- Maps Whisper timestamped segments to diarization turns.
- Infers professional job titles and roles (`Project Manager`, `Industrial Designer`, `Team Leader`, `Software Engineer`) from conversation context.

### 5. Generative AI Intelligence & Structured Extraction
- **Files:** `milestone2/pipeline.py`, `milestone2/llm_service.py`, `milestone2/schemas.py`
- Uses Google Gemini (`gemini-3.6-flash`) with structured JSON schema mode and Pydantic validation.
- Extracts Executive Summary, Key Points, Agreed Decisions, and structured Action Items with priorities and deadlines.

### 6. Interactive UI Dashboard
- **Files:** `app.py`
- Streamlit application displaying conversation flow, speaker analytics, and responsive, word-wrapped Action Items tables.

---

## 🧪 4. Testing & Verification
- **Milestone 1 Test Suite:** 99/99 PASS (100%) in `tests/`
- **Milestone 2 Test Suite:** 14/14 PASS (100%) in `milestone2/tests/`
- **LibriSpeech Benchmark Accuracy:** 93.19% – 95.20% (PASS $\ge 90\%$)
