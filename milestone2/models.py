"""
Database Models (SQLAlchemy)
Defines relational schema for storing meeting summaries, action items,
and participants in SQLite.
"""

from datetime import datetime, timezone
import json
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Meeting(Base):
    """Represents a meeting record and its generated summary intelligence."""
    __tablename__ = "meetings"

    id = Column(String(64), primary_key=True, index=True)
    transcript = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    key_points_json = Column(Text, nullable=True)
    decisions_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    action_items = relationship("ActionItemRecord", back_populates="meeting", cascade="all, delete-orphan")
    participants = relationship("ParticipantRecord", back_populates="meeting", cascade="all, delete-orphan")

    @property
    def key_points(self):
        return json.loads(self.key_points_json) if self.key_points_json else []

    @key_points.setter
    def key_points(self, value):
        self.key_points_json = json.dumps(value) if value else "[]"

    @property
    def decisions(self):
        return json.loads(self.decisions_json) if self.decisions_json else []

    @decisions.setter
    def decisions(self, value):
        self.decisions_json = json.dumps(value) if value else "[]"


class ActionItemRecord(Base):
    """Represents an action item linked to a specific meeting."""
    __tablename__ = "action_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id = Column(String(64), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True)
    task = Column(Text, nullable=False)
    assignee = Column(String(255), nullable=False, default="Unassigned")
    deadline = Column(String(100), nullable=True)
    priority = Column(String(20), nullable=False, default="Medium")
    status = Column(String(50), nullable=False, default="Not Started")

    meeting = relationship("Meeting", back_populates="action_items")


class ParticipantRecord(Base):
    """Represents a meeting participant linked to a specific meeting."""
    __tablename__ = "participants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id = Column(String(64), ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    is_unknown_assignee = Column(Boolean, default=False, nullable=False)

    meeting = relationship("Meeting", back_populates="participants")