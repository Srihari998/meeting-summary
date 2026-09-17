"""
Database Service Layer
Handles SQLite engine initialization, sessions, and persistence functions
for MeetingIntelligence objects.
"""

import json
import os
import uuid
from typing import List, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

try:
    from models import ActionItemRecord, Base, Meeting, ParticipantRecord
    from schemas import MeetingIntelligence
    from participants import reconcile_participants_and_assignees
except ImportError:
    from .models import ActionItemRecord, Base, Meeting, ParticipantRecord
    from .schemas import MeetingIntelligence
    from .participants import reconcile_participants_and_assignees

DEFAULT_DB_FILE = "meeting_intelligence.db"


def get_db_url(db_path: Optional[str] = None) -> str:
    """Constructs the SQLite connection URL."""
    target_path = db_path or os.environ.get("MEETING_DB_PATH", DEFAULT_DB_FILE)
    return f"sqlite:///{target_path}"


def get_engine(db_path: Optional[str] = None):
    """Returns a SQLAlchemy engine configured for SQLite."""
    return create_engine(get_db_url(db_path), echo=False)


def get_session(db_path: Optional[str] = None) -> Session:
    """Returns a new database session."""
    engine = get_engine(db_path)
    session_factory = sessionmaker(bind=engine)
    return session_factory()


def init_db(db_path: Optional[str] = None):
    """Initializes the database schema (creates tables if they do not exist)."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)


def save_meeting(
    transcript: str,
    parsed: MeetingIntelligence,
    meeting_id: Optional[str] = None,
    db_path: Optional[str] = None,
    participant_records: Optional[List[dict]] = None,
) -> str:
    """
    Persists a transcript and its parsed MeetingIntelligence data to SQLite.

    Args:
        transcript: Raw transcript string.
        parsed: Validated MeetingIntelligence Pydantic object.
        meeting_id: Optional custom identifier. If None, a UUID is generated.
        db_path: Optional path to SQLite file.
        participant_records: Optional precomputed list of participant metadata dicts.

    Returns:
        The meeting_id of the saved record.
    """
    init_db(db_path)
    session = get_session(db_path)

    active_id = meeting_id or str(uuid.uuid4())

    try:
        meeting_row = Meeting(
            id=active_id,
            transcript=transcript,
            summary=parsed.summary,
            key_points_json=json.dumps(parsed.key_points),
            decisions_json=json.dumps(parsed.decisions),
        )
        session.add(meeting_row)

        # Use precomputed participant records or reconcile
        if participant_records is None:
            _, participant_records = reconcile_participants_and_assignees(
                parsed.participants,
                parsed.action_items,
            )

        for p_info in participant_records:
            p_row = ParticipantRecord(
                meeting_id=active_id,
                name=p_info["name"],
                is_unknown_assignee=p_info.get("is_unknown_assignee", False),
            )
            session.add(p_row)

        for item in parsed.action_items:
            item_row = ActionItemRecord(
                meeting_id=active_id,
                task=item.task,
                assignee=item.assignee,
                deadline=item.deadline,
                priority=item.priority,
                status=item.status,
            )
            session.add(item_row)

        session.commit()
        return active_id

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()