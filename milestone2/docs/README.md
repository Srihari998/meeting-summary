# Milestone 2: Summarization & Action Extraction

This module is the complete, self-contained **Milestone 2** service for the meeting transcript summarization pipeline. It ingests a raw transcript string, performs LLM-powered extraction with Google Gemini (`gemini-2.5-flash`), applies participant normalization/deduplication, reconciles unknown action item assignees, validates schemas with Pydantic, and persists relational records into SQLite.

---

## 1. How to Connect Your Own Milestone 1

Each team member can plug their Milestone 1 Whisper transcription output directly into Milestone 2 in just **2–3 lines of code**:

```python
from milestone2.pipeline import process_meeting

# 1. Obtain transcript from your Milestone 1 Whisper transcription
transcript_text = "Alice: Let's finalize the backend architecture. Bob: I will set up PostgreSQL by Friday."

# 2. Run the complete Milestone 2 intelligence & persistence pipeline
result = process_meeting(transcript_text)

# 3. Use the typed, validated result
print("Summary:", result.summary)
print("Action Items:", result.action_items)
print("Participants:", result.participants)
```

### Streamlit Integration Example
```python
import streamlit as st
from milestone2.pipeline import process_meeting

st.title("Meeting Intelligence Dashboard")
transcript_input = st.text_area("Paste Whisper Transcript:")

if st.button("Process Meeting") and transcript_input:
    with st.spinner("Extracting summary & action items..."):
        data = process_meeting(transcript_input)
    
    st.subheader("Executive Summary")
    st.write(data.summary)
    
    st.subheader("Key Discussion Points")
    for pt in data.key_points:
        st.markdown(f"- {pt}")
        
    st.subheader("Decisions")
    for dec in data.decisions:
        st.markdown(f"✓ {dec}")
        
    st.subheader("Action Items")
    st.table([{
        "Task": it.task,
        "Assignee": it.assignee,
        "Deadline": it.deadline or "N/A",
        "Priority": it.priority,
        "Status": it.status
    } for it in data.action_items])
```

---

## 2. Public Pipeline Entry Point

### `process_meeting()` Signature
```python
def process_meeting(
    transcript: str,
    meeting_id: Optional[str] = None,
    db_path: Optional[str] = None,
    client: Optional[genai.Client] = None,
) -> MeetingIntelligence:
```

### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `transcript` | `str` | Raw meeting transcript text (from Whisper or test source). |
| `meeting_id` | `Optional[str]` | Optional custom UUID or string identifier for the meeting. If omitted, a UUID is automatically generated. |
| `db_path` | `Optional[str]` | Optional SQLite database filepath (defaults to `meeting_intelligence.db`). |
| `client` | `Optional[genai.Client]` | Optional `google-genai` client instance (useful for mocking/testing). |

### Return Value
Returns a validated, typed `MeetingIntelligence` object.

---

## 3. Data Schema & JSON Output

All models are defined with strict Pydantic rules where **extra fields are strictly forbidden** (`extra="forbid"`).

### Pydantic Models
```python
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict

class ActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    assignee: str
    deadline: Optional[str] = None
    priority: Literal["High", "Medium", "Low"] = "Medium"
    status: str = "Not Started"

class MeetingIntelligence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    key_points: List[str]
    decisions: List[str]
    action_items: List[ActionItem]
    participants: List[str]
```

### JSON Structure
```json
{
  "summary": "Team aligned on Q4 deliverables and database migration timeline.",
  "key_points": [
    "Milestone 2 pipeline is fully integrated.",
    "SQLite schema configured with cascading foreign keys."
  ],
  "decisions": [
    "Local persistence uses SQLite with SQLAlchemy."
  ],
  "action_items": [
    {
      "task": "Set up database models and migration scripts",
      "assignee": "Bob",
      "deadline": "2026-10-15",
      "priority": "High",
      "status": "Not Started"
    }
  ],
  "participants": [
    "Alice",
    "Bob",
    "Charlie"
  ]
}
```

---

## 4. Database Schema (SQLite / SQLAlchemy)

The database automatically manages tables and foreign keys via SQLAlchemy:

### Tables
1. **`meetings`**:
   - `id` (`VARCHAR(64)`, Primary Key)
   - `transcript` (`TEXT`, NOT NULL)
   - `summary` (`TEXT`, NOT NULL)
   - `key_points_json` (`TEXT`, JSON-encoded list)
   - `decisions_json` (`TEXT`, JSON-encoded list)
   - `created_at` (`DATETIME`, default UTC timestamp)

2. **`action_items`**:
   - `id` (`INTEGER`, Primary Key, Autoincrement)
   - `meeting_id` (`VARCHAR(64)`, Foreign Key -> `meetings.id`, ON DELETE CASCADE)
   - `task` (`TEXT`, NOT NULL)
   - `assignee` (`VARCHAR(255)`, NOT NULL, default `"Unassigned"`)
   - `deadline` (`VARCHAR(100)`, NULLABLE)
   - `priority` (`VARCHAR(20)`, default `"Medium"`)
   - `status` (`VARCHAR(50)`, default `"Not Started"`)

3. **`participants`**:
   - `id` (`INTEGER`, Primary Key, Autoincrement)
   - `meeting_id` (`VARCHAR(64)`, Foreign Key -> `meetings.id`, ON DELETE CASCADE)
   - `name` (`VARCHAR(255)`, NOT NULL)
   - `is_unknown_assignee` (`BOOLEAN`, default `False`)

---

## 5. Dual Retry Architecture

Milestone 2 cleanly isolates transient infrastructure failures from schema/syntax errors:

```
[process_meeting]
       │
       ▼
[Input Validation] ──(Empty/Short)──► Raises InvalidInputError
       │
       ▼
[LLM Call] ──(Network / 5xx / 429)──► [API Backoff Loop: 3x (1s, 2s, 4s)] ──(Exhausted)──► LLMAPIError
       │
       ▼
[validate_and_parse] ──(JSON/Schema Fail)──► [JSON-Fix Loop: 2x via fix_json_prompt] ──(Exhausted)──► JSONFixError
       │
       ▼
[Participant & Assignee Reconciliation]
       │
       ▼
[SQLite DB Persistence (save_meeting)]
       │
       ▼
[Return MeetingIntelligence]
```

### 1. API Network / Rate Limit Retries
- Retries up to **3 times** with exponential backoff delays (`1.0s`, `2.0s`, `4.0s`).
- Logs every attempt: `[API Retry {attempt}/3] ...`
- Raises `LLMAPIError` if all 3 attempts fail.

### 2. JSON Schema Fix Retries
- If the model returns malformed JSON or violates schema constraints (missing fields, wrong enums), the parser captures the exact field error.
- Prompts the LLM with `prompts/fix_json_prompt.txt` up to **2 times**, feeding back the faulty output and exact error.
- Logs every attempt: `[JSON-Fix Retry {attempt}/2] ...`
- Raises `JSONFixError` if all fix retries are exhausted.

---

## 6. Participant Normalization & Unknown Assignee Reconciliation

- **Name Cleaning**: Standardizes casing (`"ravi"` -> `"Ravi"`), strips noise like `"(Host)"` or `"Speaker 1:"`.
- **Deduplication**: Unifies case-insensitive duplicates and matches prefixes (e.g. merging `"Ravi"` into `"Ravi K"` if unambiguous).
- **Unknown Assignees**: If an action item names an assignee (e.g. `"David"`) who was not in the detected participants list, `"David"` is **not dropped** — David is preserved in the action item and registered in the `participants` database table with `is_unknown_assignee = True`.

---

## 7. Running Tests & Live API Smoke Testing

### Running Mocked Unit & Integration Tests (Fast, Offline)
```powershell
python -m pytest milestone2/tests -v
```

### Running Live (Non-Mocked) Gemini API Smoke Test
To test against the real Gemini API (`gemini-2.5-flash`), set `RUN_LIVE_TESTS=1` and your `GEMINI_API_KEY`:

**PowerShell (Windows):**
```powershell
$env:RUN_LIVE_TESTS="1"; $env:GEMINI_API_KEY="your-real-gemini-api-key"; python -m pytest milestone2/tests/test_live_smoke.py -v -s
```

**Bash / Linux / macOS:**
```bash
RUN_LIVE_TESTS=1 GEMINI_API_KEY="your-real-gemini-api-key" python -m pytest milestone2/tests/test_live_smoke.py -v -s
```

### Running CLI Pipeline Demo
```powershell
python milestone2/examples/run_pipeline_demo.py
```