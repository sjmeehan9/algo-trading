"""Persistence backends for model generation tracking."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from threading import RLock
from typing import Any

from algotrading.src.models.tracking.generation import Generation


class GenerationStorage(ABC):
    """Abstract generation persistence interface."""

    @abstractmethod
    def save(self, generation: Generation) -> None:
        """Persist a generation record."""

    @abstractmethod
    def load(self, generation_id: str) -> Generation | None:
        """Load a generation by ID."""

    @abstractmethod
    def load_all_for_model(self, model_id: str) -> list[Generation]:
        """Load all generations for a specific model."""

    @abstractmethod
    def delete(self, generation_id: str) -> bool:
        """Delete a generation by ID and return deletion status."""

    @abstractmethod
    def query(self, filters: dict[str, object]) -> list[Generation]:
        """Query generations using filter criteria."""


class JsonFileStorage(GenerationStorage):
    """JSON file-based generation storage with a global index."""

    INDEX_FILENAME = "index.json"

    def __init__(self, base_path: str) -> None:
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self._base_path / self.INDEX_FILENAME
        self._lock = RLock()
        self._ensure_index()

    def save(self, generation: Generation) -> None:
        """Persist a generation JSON file and update index metadata."""

        with self._lock:
            model_dir = self._base_path / generation.model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            file_path = model_dir / f"{generation.generation_id}.json"

            with file_path.open("w", encoding="utf-8") as file_handle:
                json.dump(generation.to_dict(), file_handle, indent=2, sort_keys=True)

            index = self._read_index()
            generation_index = index.setdefault("generation_index", {})
            model_index = index.setdefault("model_index", {})

            generation_index[generation.generation_id] = {
                "model_id": generation.model_id,
                "path": str(file_path.relative_to(self._base_path)),
                "generation_number": generation.generation_number,
                "created_at": generation.to_dict()["created_at"],
                "status": generation.status,
                "tags": list(generation.tags),
                "parent_generation_id": generation.parent_generation_id,
            }

            model_index.setdefault(generation.model_id, [])
            if generation.generation_id not in model_index[generation.model_id]:
                model_index[generation.model_id].append(generation.generation_id)

            self._write_index(index)

    def load(self, generation_id: str) -> Generation | None:
        """Load a generation by ID from index and JSON payload."""

        with self._lock:
            index = self._read_index()
            entry = index.get("generation_index", {}).get(generation_id)
            if entry is None:
                return None

            file_path = self._base_path / entry["path"]
            if not file_path.exists():
                return None

            with file_path.open("r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
            return Generation.from_dict(payload)

    def load_all_for_model(self, model_id: str) -> list[Generation]:
        """Load all generations for a model, ordered by generation number."""

        with self._lock:
            index = self._read_index()
            generation_ids = list(index.get("model_index", {}).get(model_id, []))

        generations: list[Generation] = []
        for generation_id in generation_ids:
            generation = self.load(generation_id)
            if generation is not None:
                generations.append(generation)

        generations.sort(key=lambda item: item.generation_number)
        return generations

    def delete(self, generation_id: str) -> bool:
        """Delete a generation file and clean up index references."""

        with self._lock:
            index = self._read_index()
            generation_index = index.get("generation_index", {})
            entry = generation_index.get(generation_id)
            if entry is None:
                return False

            file_path = self._base_path / entry["path"]
            if file_path.exists():
                file_path.unlink()

            model_id = entry["model_id"]
            del generation_index[generation_id]
            model_generations = index.get("model_index", {}).get(model_id, [])
            index["model_index"][model_id] = [
                item for item in model_generations if item != generation_id
            ]
            if not index["model_index"][model_id]:
                del index["model_index"][model_id]

            self._write_index(index)
            return True

    def query(self, filters: dict[str, object]) -> list[Generation]:
        """Query generation records by field equality and tag inclusion."""

        with self._lock:
            index = self._read_index()
            generation_ids = list(index.get("generation_index", {}).keys())

        matches: list[Generation] = []
        for generation_id in generation_ids:
            generation = self.load(generation_id)
            if generation is None:
                continue
            if self._matches(generation, filters):
                matches.append(generation)

        matches.sort(key=lambda item: (item.model_id, item.generation_number))
        return matches

    def _ensure_index(self) -> None:
        """Create index file if absent."""

        if self._index_path.exists():
            return
        self._write_index({"generation_index": {}, "model_index": {}})

    def _read_index(self) -> dict[str, Any]:
        """Read and parse index JSON file."""

        with self._index_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        payload.setdefault("generation_index", {})
        payload.setdefault("model_index", {})
        return payload

    def _write_index(self, payload: dict[str, Any]) -> None:
        """Write index JSON atomically."""

        with self._index_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, sort_keys=True)

    @staticmethod
    def _matches(generation: Generation, filters: dict[str, object]) -> bool:
        """Evaluate whether generation satisfies filter constraints."""

        for key, expected in filters.items():
            if key == "tag":
                if not isinstance(expected, str) or expected not in generation.tags:
                    return False
                continue
            actual = getattr(generation, key, None)
            if actual != expected:
                return False
        return True


class SqliteStorage(GenerationStorage):
    """SQLite-backed generation persistence optimized for query flexibility."""

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._init_schema()

    def save(self, generation: Generation) -> None:
        """Upsert a generation row in SQLite."""

        payload = json.dumps(generation.to_dict(), sort_keys=True)
        tags = json.dumps(generation.tags)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO generations (
                    generation_id,
                    model_id,
                    generation_number,
                    parent_generation_id,
                    created_at,
                    status,
                    tags,
                    payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(generation_id) DO UPDATE SET
                    model_id=excluded.model_id,
                    generation_number=excluded.generation_number,
                    parent_generation_id=excluded.parent_generation_id,
                    created_at=excluded.created_at,
                    status=excluded.status,
                    tags=excluded.tags,
                    payload=excluded.payload
                """,
                (
                    generation.generation_id,
                    generation.model_id,
                    generation.generation_number,
                    generation.parent_generation_id,
                    generation.to_dict()["created_at"],
                    generation.status,
                    tags,
                    payload,
                ),
            )
            connection.commit()

    def load(self, generation_id: str) -> Generation | None:
        """Load a generation record by ID."""

        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM generations WHERE generation_id = ?",
                (generation_id,),
            ).fetchone()
        if row is None:
            return None
        return Generation.from_dict(json.loads(row[0]))

    def load_all_for_model(self, model_id: str) -> list[Generation]:
        """Load all generations for a model ordered by generation number."""

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload
                FROM generations
                WHERE model_id = ?
                ORDER BY generation_number ASC
                """,
                (model_id,),
            ).fetchall()
        return [Generation.from_dict(json.loads(row[0])) for row in rows]

    def delete(self, generation_id: str) -> bool:
        """Delete a generation row and return whether a row was removed."""

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM generations WHERE generation_id = ?",
                (generation_id,),
            )
            connection.commit()
            return cursor.rowcount > 0

    def query(self, filters: dict[str, object]) -> list[Generation]:
        """Query generations by indexed fields and optional tag filter."""

        query = "SELECT payload FROM generations"
        clauses: list[str] = []
        args: list[object] = []

        if "model_id" in filters:
            clauses.append("model_id = ?")
            args.append(filters["model_id"])
        if "status" in filters:
            clauses.append("status = ?")
            args.append(filters["status"])
        if "parent_generation_id" in filters:
            clauses.append("parent_generation_id = ?")
            args.append(filters["parent_generation_id"])

        if clauses:
            query = f"{query} WHERE {' AND '.join(clauses)}"

        query = f"{query} ORDER BY model_id ASC, generation_number ASC"

        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(args)).fetchall()

        generations = [Generation.from_dict(json.loads(row[0])) for row in rows]

        if "tag" in filters:
            tag_value = filters["tag"]
            if isinstance(tag_value, str):
                generations = [item for item in generations if tag_value in item.tags]
            else:
                generations = []

        return generations

    def _init_schema(self) -> None:
        """Initialize SQLite schema for generation persistence."""

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generations (
                    generation_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    generation_number INTEGER NOT NULL,
                    parent_generation_id TEXT,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_generations_model
                ON generations (model_id, generation_number)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_generations_status
                ON generations (status)
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection with row factory for dictionary access."""

        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection


__all__ = [
    "GenerationStorage",
    "JsonFileStorage",
    "SqliteStorage",
]
