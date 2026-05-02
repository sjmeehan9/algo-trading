"""Unit tests for trading session persistence."""

from __future__ import annotations

from pathlib import Path

import pytest
from algotrading.src.trading.session import SessionPersistence, SessionState


@pytest.mark.asyncio
async def test_session_persistence_saves_loads_and_deletes_state(
    tmp_path: Path,
) -> None:
    """SessionPersistence should round-trip state and remove files."""

    persistence = SessionPersistence(tmp_path / "sessions")
    state = SessionState(session_id="session-1", positions={"AAPL": 4.0})

    await persistence.save_state("session-1", state)
    loaded = await persistence.load_state("session-1")

    assert loaded is not None
    assert loaded.get_position("AAPL") == 4.0

    sessions = await persistence.load_all_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "session-1"

    await persistence.delete_state("session-1")
    assert await persistence.load_state("session-1") is None


@pytest.mark.asyncio
async def test_session_persistence_saves_complete_session_snapshot(
    tmp_path: Path,
) -> None:
    """Complete session snapshots should retain config and status metadata."""

    persistence = SessionPersistence(tmp_path / "sessions")
    await persistence.save_session(
        "session-2",
        {
            "config": {"model_id": "model-1"},
            "status": "running",
            "state": SessionState(session_id="session-2").to_dict(),
        },
    )

    loaded = await persistence.load_session("session-2")

    assert loaded is not None
    assert loaded["status"] == "running"
    assert loaded["config"] == {"model_id": "model-1"}
    assert "saved_at" in loaded
