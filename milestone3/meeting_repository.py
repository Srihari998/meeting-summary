"""
Meeting Repository — Milestone 3
Read-only access layer over the existing Milestone 2 SQLite database.
Wraps milestone2.db session utilities and ORM models.

The relational DB is the single source of truth; this module never writes to it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

# Re-use Milestone 2 DB infrastructure (no new tables, no migrations)
try:
    from milestone2.db import get_session
    from milestone2.models import ActionItemRecord, Meeting, ParticipantRecord
except ImportError:
    from db import get_session  # type: ignore
    from models import ActionItemRecord, Meeting, ParticipantRecord  # type: ignore

logger = logging.getLogger(__name__)


from sqlalchemy.orm import joinedload

class MeetingRepository:
    """
    Provides structured read access to all stored meetings.

    Args:
        db_path: Optional path to the SQLite database file.
                 Defaults to the value resolved by milestone2.db.get_db_url().
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path

    def _session(self):
        return get_session(self._db_path)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_all_meetings(self) -> List[Meeting]:
        """
        Returns all stored Meeting rows, ordered by creation date descending.

        Eagerly loads action_items and participants so the returned objects
        remain fully usable after session close.
        """
        session = self._session()
        try:
            return (
                session.query(Meeting)
                .options(
                    joinedload(Meeting.action_items),
                    joinedload(Meeting.participants),
                )
                .order_by(Meeting.created_at.desc())
                .all()
            )
        finally:
            session.close()

    def get_meeting_by_id(self, meeting_id: str) -> Optional[Meeting]:
        """
        Returns a single Meeting row by its primary key, or None if not found.

        Eagerly loads action_items and participants.
        """
        session = self._session()
        try:
            return (
                session.query(Meeting)
                .options(
                    joinedload(Meeting.action_items),
                    joinedload(Meeting.participants),
                )
                .filter(Meeting.id == meeting_id)
                .first()
            )
        finally:
            session.close()

    def get_meetings_since(self, after: datetime) -> List[Meeting]:
        """
        Returns all meetings created after the given datetime, most recent first.
        """
        session = self._session()
        try:
            return (
                session.query(Meeting)
                .filter(Meeting.created_at > after)
                .order_by(Meeting.created_at.desc())
                .all()
            )
        finally:
            session.close()

    def count_meetings(self) -> int:
        """Returns the total number of stored meetings."""
        session = self._session()
        try:
            return session.query(Meeting).count()
        finally:
            session.close()

    def list_meeting_ids(self) -> List[str]:
        """Returns a list of all meeting IDs in the database."""
        session = self._session()
        try:
            rows = session.query(Meeting.id).all()
            return [row[0] for row in rows]
        finally:
            session.close()

    def get_action_items_for_meeting(self, meeting_id: str) -> List[ActionItemRecord]:
        """Returns all action items for a given meeting."""
        session = self._session()
        try:
            return (
                session.query(ActionItemRecord)
                .filter(ActionItemRecord.meeting_id == meeting_id)
                .all()
            )
        finally:
            session.close()

    def get_participants_for_meeting(self, meeting_id: str) -> List[ParticipantRecord]:
        """Returns all participants for a given meeting."""
        session = self._session()
        try:
            return (
                session.query(ParticipantRecord)
                .filter(ParticipantRecord.meeting_id == meeting_id)
                .all()
            )
        finally:
            session.close()
