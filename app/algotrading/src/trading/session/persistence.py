"""Filesystem persistence for trading session state."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from algotrading.src.trading.session.exceptions import SessionPersistenceError
from algotrading.src.trading.session.state import SessionState


class SessionPersistence:
    """Persist trading session snapshots as atomic JSON files."""

    def __init__(self, storage_path: str | Path | None = None) -> None:
        """Initialize persistence under the configured storage path."""

        self.storage_path = Path(storage_path or "data/sessions")
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def save_state(self, session_id: str, state: SessionState) -> None:
        """Save a state-only snapshot for compatibility callers."""

        await self.save_session(
            session_id,
            {
                "session_id": session_id,
                "state": state.to_dict(),
                "saved_at": datetime.now(tz=UTC).isoformat(),
            },
        )

    async def save_session(
        self,
        session_id: str,
        payload: Mapping[str, object],
    ) -> None:
        """Persist one complete session snapshot atomically."""

        normalized_id = _normalize_session_id(session_id)
        file_path = self._file_path(normalized_id)
        serializable = dict(payload)
        serializable["session_id"] = normalized_id
        serializable["saved_at"] = datetime.now(tz=UTC).isoformat()
        await asyncio.to_thread(_write_json_atomic, file_path, serializable)

    async def load_state(self, session_id: str) -> SessionState | None:
        """Load a state-only snapshot when present."""

        payload = await self.load_session(session_id)
        if payload is None:
            return None
        state_payload = payload.get("state")
        if not isinstance(state_payload, dict):
            return None
        return SessionState.from_dict(state_payload)

    async def load_session(self, session_id: str) -> dict[str, object] | None:
        """Load a complete session snapshot by ID."""

        file_path = self._file_path(_normalize_session_id(session_id))
        if not file_path.exists():
            return None
        return await asyncio.to_thread(_read_json, file_path)

    async def load_all_sessions(self) -> list[dict[str, object]]:
        """Load every persisted session snapshot from disk."""

        file_paths = sorted(self.storage_path.glob("*.json"))
        sessions: list[dict[str, object]] = []
        for file_path in file_paths:
            sessions.append(await asyncio.to_thread(_read_json, file_path))
        return sessions

    async def delete_state(self, session_id: str) -> None:
        """Delete a persisted session snapshot if it exists."""

        file_path = self._file_path(_normalize_session_id(session_id))
        try:
            await asyncio.to_thread(file_path.unlink, missing_ok=True)
        except OSError as exc:
            raise SessionPersistenceError(
                f"Could not delete persisted session '{session_id}': {exc}"
            ) from exc

    def _file_path(self, session_id: str) -> Path:
        """Return the JSON file path for a normalized session ID."""

        return self.storage_path / f"{session_id}.json"


def _normalize_session_id(session_id: str) -> str:
    normalized = session_id.strip()
    if not normalized:
        raise ValueError("session_id must be non-empty")
    if "/" in normalized or "\\" in normalized:
        raise ValueError("session_id cannot contain path separators")
    return normalized


def _write_json_atomic(file_path: Path, payload: Mapping[str, object]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(file_path)
    except (OSError, TypeError, ValueError) as exc:
        raise SessionPersistenceError(
            f"Could not persist session snapshot at '{file_path}': {exc}"
        ) from exc


def _read_json(file_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionPersistenceError(
            f"Could not load session snapshot at '{file_path}': {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SessionPersistenceError(
            f"Session snapshot at '{file_path}' must contain a JSON object"
        )
    return payload


__all__ = ["SessionPersistence"]
