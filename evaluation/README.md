# Evaluation Framework — Milestone 1 Accuracy Testing (Task 5)

This directory contains the structured evaluation framework for Milestone 1 Task 5.

---

## Directory Structure

```
evaluation/
├── README.md              # This file
├── accuracy_results.csv   # Results table (updated after each run)
├── recordings/            # Place test audio/video files here
│   └── recording_01.wav   # Example filename convention
└── references/            # Place reference transcript text files here
    └── recording_01.txt   # Must match recording filename (without extension)
```

---

## File Naming Convention

Recording files and reference transcripts must share the same base name:

| Recording                         | Reference Transcript             |
| --------------------------------- | -------------------------------- |
| `recordings/recording_01.wav`     | `references/recording_01.txt`    |
| `recordings/meeting_sprint.mp4`   | `references/meeting_sprint.txt`  |

---

## How to Add a Recording

1. Place your audio/video file in `evaluation/recordings/`.
2. Create a matching plain-text `.txt` file in `evaluation/references/` with the verbatim expected speech.
3. Run the evaluation script from the project root:
   ```bash
   python run_evaluation.py
   ```
4. Results are written to `evaluation/accuracy_results.csv`.

---

## Accuracy Metric

The project uses **Word Error Rate (WER)** normalized for punctuation, case, and whitespace:

```
Accuracy = max(0, 1 − WER) × 100%
```

The milestone requirement is: **Accuracy ≥ 90%**

---

## Current Status

> **No reference recordings have been added yet.**

The evaluation framework (directory structure, CSV schema, and run script) is fully implemented.
Results will appear in `accuracy_results.csv` once recordings and references are placed in the
respective folders and `run_evaluation.py` is executed.

The table below will be populated with real measured results only — no fabricated numbers.

| Recording | Model | Duration | Ref Words | Gen Words | Subs | Dels | Ins | WER | Accuracy | Pass/Fail |
|-----------|-------|----------|-----------|-----------|------|------|-----|-----|----------|-----------|
| _Not yet tested_ | — | — | — | — | — | — | — | — | — | — |

---

## Evaluation Conditions (Planned)

When recordings are available, evaluations should cover:

1. **Clear speech** — single speaker, quiet environment
2. **Multiple speakers** — two or more distinct speakers
3. **Background noise** — noise present during recording
4. **Technical/domain vocabulary** — acronyms, technical terms
5. **Longer meeting** — recordings ≥ 10 minutes

Each condition will be recorded separately in `accuracy_results.csv`.
