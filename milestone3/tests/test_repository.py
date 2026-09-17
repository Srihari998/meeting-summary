"""
Tests for milestone3.meeting_repository — read-only DB access layer.
Uses an in-memory SQLite database with real ORM models.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from milestone2.models import ActionItemRecord, Base, Meeting, ParticipantRecord
from milestone3.meeting_repository import MeetingRepository


def _make_in_memory_db():
    """Creates an in-memory SQLite engine with the Milestone 2 schema."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


def _seed_meetings(engine, count: int = 3) -> List[str]:
    """Seeds the database with `count` test meetings. Returns list of meeting IDs."""
    Session = sessionmaker(bind=engine)
    session = Session()
    ids = []
    for i in range(count):
        mtg_id = f"test-meeting-{i}"
        m = Meeting(
            id=mtg_id,
            transcript=f"Transcript content for meeting {i}.",
            summary=f"Summary for meeting {i}.",
            key_points_json=json.dumps([f"Point A{i}", f"Point B{i}"]),
            decisions_json=json.dumps([f"Decision {i}"]),
            created_at=datetime.now(timezone.utc),
        )
        session.add(m)
        session.add(ParticipantRecord(meeting_id=mtg_id, name=f"Speaker {i}"))
        session.add(ActionItemRecord(
            meeting_id=mtg_id,
            task=f"Task {i}",
            assignee=f"Speaker {i}",
            deadline=f"2025-01-{i+10:02d}",
            priority="High",
            status="Not Started",
        ))
        ids.append(mtg_id)
    session.commit()
    session.close()
    return ids


class TestMeetingRepository:
    @pytest.fixture
    def repo(self, monkeypatch):
        """Returns a MeetingRepository backed by an in-memory SQLite DB."""
        engine = _make_in_memory_db()
        ids = _seed_meetings(engine, count=3)

        # Patch get_session to use our in-memory engine
        Session = sessionmaker(bind=engine)

        def _patched_session(db_path=None):
            return Session()

        import milestone3.meeting_repository as repo_module
        monkeypatch.setattr(repo_module, "get_session", _patched_session)

        return MeetingRepository(), ids

    def test_count_meetings(self, repo):
        repository, ids = repo
        assert repository.count_meetings() == 3

    def test_get_all_meetings_returns_all(self, repo):
        repository, ids = repo
        meetings = repository.get_all_meetings()
        assert len(meetings) == 3

    def test_get_all_meetings_are_meeting_objects(self, repo):
        repository, ids = repo
        meetings = repository.get_all_meetings()
        for m in meetings:
            assert isinstance(m, Meeting)
            assert m.transcript is not None
            assert m.summary is not None

    def test_get_meeting_by_id_found(self, repo):
        repository, ids = repo
        m = repository.get_meeting_by_id(ids[0])
        assert m is not None
        assert m.id == ids[0]

    def test_get_meeting_by_id_not_found(self, repo):
        repository, ids = repo
        m = repository.get_meeting_by_id("nonexistent-id")
        assert m is None

    def test_list_meeting_ids(self, repo):
        repository, ids = repo
        result_ids = repository.list_meeting_ids()
        assert set(result_ids) == set(ids)

    def test_count_on_empty_db(self, monkeypatch):
        engine = _make_in_memory_db()
        Session = sessionmaker(bind=engine)

        def _patched_session(db_path=None):
            return Session()

        import milestone3.meeting_repository as repo_module
        monkeypatch.setattr(repo_module, "get_session", _patched_session)

        repo = MeetingRepository()
        assert repo.count_meetings() == 0

    def test_get_all_meetings_empty_db(self, monkeypatch):
        engine = _make_in_memory_db()
        Session = sessionmaker(bind=engine)

        def _patched_session(db_path=None):
            return Session()

        import milestone3.meeting_repository as repo_module
        monkeypatch.setattr(repo_module, "get_session", _patched_session)

        repo = MeetingRepository()
        assert repo.get_all_meetings() == []
