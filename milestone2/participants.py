"""
Participants Module
Handles normalization, deduplication, and assignee reconciliation:
- normalize_name(): cleans speaker titles, roles, casing, and whitespace.
- dedupe_participants(): removes case-insensitive duplicates and unifies partial names.
- reconcile_participants(): flags assignees not detected in participant list as 'Unknown'
  without dropping the action items.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
try:
    from schemas import ActionItem
except ImportError:
    from .schemas import ActionItem


# Common noise patterns in transcripts
ROLE_PATTERN = re.compile(r"\s*\((?:Host|Co-host|Presenter|Speaker|Guest|Admin|Moderator)\)", re.IGNORECASE)
TITLE_PREFIX_PATTERN = re.compile(r"^(?:Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\s+", re.IGNORECASE)
SPEAKER_LABEL_PATTERN = re.compile(r"^Speaker\s+(\d+|[A-Z]):?", re.IGNORECASE)


def normalize_name(name: str) -> str:
    """
    Normalizes a participant name:
    - Strips surrounding whitespace and punctuation (e.g. trailing colons, quotes).
    - Removes role suffixes (e.g. '(Host)', '(Presenter)').
    - Standardizes title capitalization (e.g. 'ravi' -> 'Ravi').
    """
    if not name or not name.strip():
        return "Unknown"

    cleaned = name.strip().strip(":\"'-,")
    # Remove role badges like (Host)
    cleaned = ROLE_PATTERN.sub("", cleaned).strip()

    # Handle speaker label like "Speaker 1:" -> "Speaker 1"
    speaker_match = SPEAKER_LABEL_PATTERN.match(cleaned)
    if speaker_match:
        return f"Speaker {speaker_match.group(1)}"

    # Title-case clean words (preserve initials like "Ravi K" or "J. Doe")
    words = cleaned.split()
    normalized_words = []
    for w in words:
        if len(w) == 1 or (len(w) == 2 and w.endswith(".")):
            normalized_words.append(w.upper())
        elif w.isupper() and len(w) <= 3:
            normalized_words.append(w)
        else:
            normalized_words.append(w.capitalize())

    return " ".join(normalized_words) if normalized_words else "Unknown"


def dedupe_participants(participants: List[str]) -> List[str]:
    """
    Deduplicates a list of participant names:
    - Normalizes all names.
    - Resolves case-insensitive collisions (e.g. 'ravi' vs 'Ravi').
    - If a single-word name is an exact prefix of a full name in the list
      (e.g., 'Ravi' and 'Ravi K'), prefers the more specific 'Ravi K'.
    """
    if not participants:
        return []

    normalized_names: List[str] = []
    seen_lower: Set[str] = set()

    for p in participants:
        norm = normalize_name(p)
        if norm.lower() not in seen_lower and norm != "Unknown":
            seen_lower.add(norm.lower())
            normalized_names.append(norm)

    # Secondary deduplication: merge 'First' with 'First Last' if unambiguous
    final_names: List[str] = []
    for name in normalized_names:
        # Check if this name is a single-word prefix of a longer multi-word name
        name_parts = name.split()
        if len(name_parts) == 1:
            has_longer_match = any(
                other != name and other.split()[0].lower() == name.lower()
                for other in normalized_names
            )
            if has_longer_match:
                continue
        final_names.append(name)

    return final_names if final_names else normalized_names


def reconcile_participants_and_assignees(
    participants: List[str],
    action_items: List[ActionItem]
) -> Tuple[List[str], List[dict]]:
    """
    Cross-references action item assignees with the detected participants list.
    
    Rules:
    - Normalizes and dedupes all participants.
    - If an action item assignee is named but NOT found in the participants list,
      flags the participant as 'Unknown' (e.g. adds to participant records with
      an Unknown flag) without dropping the action item.
    
    Returns:
        Tuple of:
        - List of canonical participant names (including flagged unknown assignees)
        - List of participant metadata dicts with {"name": str, "is_unknown_assignee": bool}
    """
    canonical_participants = dedupe_participants(participants)
    participant_lower_set = {p.lower() for p in canonical_participants}

    participant_records: List[dict] = [
        {"name": p, "is_unknown_assignee": False} for p in canonical_participants
    ]

    for item in action_items:
        assignee_raw = item.assignee.strip()
        if not assignee_raw or assignee_raw.lower() in {"unassigned", "none", "tbd", "unknown"}:
            continue

        norm_assignee = normalize_name(assignee_raw)
        norm_assignee_lower = norm_assignee.lower()

        # Check if assignee matches any known participant
        matches = any(
            p_lower == norm_assignee_lower or p_lower.startswith(norm_assignee_lower)
            for p_lower in participant_lower_set
        )

        if not matches:
            # Flag as Unknown assignee without dropping
            flagged_name = f"{norm_assignee} (Unknown)"
            if norm_assignee_lower not in participant_lower_set:
                canonical_participants.append(norm_assignee)
                participant_lower_set.add(norm_assignee_lower)
                participant_records.append({
                    "name": norm_assignee,
                    "is_unknown_assignee": True
                })

    return canonical_participants, participant_records


def infer_speaker_roles(
    transcript: str,
    speakers: List[str] = None,
    client: Any = None,
) -> dict[str, str]:
    """
    Infers the professional position, job title, or role (e.g. Project Manager,
    Team Leader, Industrial Designer, Engineer, Marketing Lead, Coworker)
    for each speaker/participant from the script text.

    Uses Gemini LLM when available, with a fast heuristic fallback.

    Returns:
        dict[str, str]: Mapping from speaker label/name to inferred role.
    """
    if not transcript or not transcript.strip():
        return {}

    detected_roles: dict[str, str] = {}
    candidate_speakers = speakers or []

    # 1. Attempt LLM-based role inference
    try:
        from google import genai
        import os
        import json
        from dotenv import load_dotenv
        load_dotenv()

        active_client = client or genai.Client()
        prompt = (
            "Analyze the following meeting transcript. For each detected speaker, person, or identifier "
            "(e.g. Speaker 1, Speaker 2, Sarah, David), infer their professional role, job title, or position "
            "(e.g. 'Project Manager', 'Team Leader', 'Industrial Designer', 'Frontend Developer', 'Marketing Lead', 'Coworker / Contributor') "
            "based strictly on contextual evidence, duties, and conversational cues in the script.\n\n"
            "Return JSON mapping each speaker/name to their specific role:\n"
            'Example: {"Speaker 1": "Project Manager", "Speaker 2": "Industrial Designer", "Speaker 3": "Marketing Lead"}\n\n'
            f"TRANSCRIPT:\n{transcript[:10000]}"
        )

        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        response = active_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        if response and response.text:
            cleaned = response.text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            parsed = json.loads(cleaned.strip())
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        pass

    # 2. Heuristic fallback based on context keywords
    text_lower = transcript.lower()
    for spk in candidate_speakers:
        spk_lower = spk.lower()
        if "manager" in text_lower or "pm" in text_lower:
            detected_roles[spk] = "Project Manager"
        elif "lead" in text_lower or "leader" in text_lower:
            detected_roles[spk] = "Team Leader"
        elif "design" in text_lower or "casing" in text_lower or "cad" in text_lower:
            detected_roles[spk] = "Product Designer"
        elif "developer" in text_lower or "database" in text_lower or "frontend" in text_lower:
            detected_roles[spk] = "Software Engineer"
        elif "marketing" in text_lower or "sales" in text_lower:
            detected_roles[spk] = "Marketing Specialist"
        else:
            detected_roles[spk] = "Team Member / Coworker"

    return detected_roles