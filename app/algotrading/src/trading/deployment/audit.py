"""Audit logging for model deployment attempts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("deployment.audit")


@dataclass(slots=True)
class DeploymentAttempt:
    """Immutable record of a deployment validation attempt.

    Args:
        timestamp: UTC timestamp when the attempt was recorded.
        model_id: Model identifier supplied by the caller.
        user_id: Optional user or client identifier.
        mode: Deployment mode such as ``paper`` or ``live``.
        result: Attempt result, usually ``approved`` or ``rejected``.
        reason: Rejection reason or additional approval context.
        model_type: Resolved model type when available.
        generation_id: Optional model generation identifier.
        metadata: Additional structured context safe for logs.
    """

    timestamp: datetime
    model_id: str
    user_id: str | None
    mode: str
    result: str
    reason: str | None
    model_type: str | None
    generation_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize this attempt to JSON-safe primitives."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "user_id": self.user_id,
            "mode": self.mode,
            "result": self.result,
            "reason": self.reason,
            "model_type": self.model_type,
            "generation_id": self.generation_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DeploymentAttempt:
        """Deserialize one audit attempt from persisted JSON data."""

        timestamp = datetime.fromisoformat(str(payload["timestamp"]))
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        metadata = payload.get("metadata")
        return cls(
            timestamp=timestamp,
            model_id=str(payload["model_id"]),
            user_id=(
                str(payload["user_id"]) if payload.get("user_id") is not None else None
            ),
            mode=str(payload["mode"]),
            result=str(payload["result"]),
            reason=(str(payload["reason"]) if payload.get("reason") else None),
            model_type=(
                str(payload["model_type"])
                if payload.get("model_type") is not None
                else None
            ),
            generation_id=(
                str(payload["generation_id"])
                if payload.get("generation_id") is not None
                else None
            ),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )


class DeploymentAuditLog:
    """Append-only JSONL audit log for deployment attempts."""

    def __init__(self, log_path: str | Path | None = None) -> None:
        """Initialize an audit log.

        Args:
            log_path: Optional JSONL path. Defaults to
                ``logs/deployment_audit.jsonl`` relative to the process cwd.
        """

        self.log_path = (
            Path(log_path)
            if log_path is not None
            else Path("logs/deployment_audit.jsonl")
        )

    def log_attempt(
        self,
        *,
        model_id: str,
        user_id: str | None = None,
        mode: str = "paper",
        result: str = "rejected",
        reason: str | None = None,
        model_type: str | None = None,
        generation_id: str | None = None,
        **metadata: object,
    ) -> DeploymentAttempt:
        """Record one deployment attempt to structured logs and JSONL storage."""

        attempt = DeploymentAttempt(
            timestamp=datetime.now(tz=UTC),
            model_id=model_id,
            user_id=user_id,
            mode=mode,
            result=result,
            reason=reason,
            model_type=model_type,
            generation_id=generation_id,
            metadata=dict(metadata),
        )
        payload = attempt.to_dict()

        if result == "approved":
            logger.info("Deployment approved", extra={"deployment_attempt": payload})
        else:
            logger.warning(
                "Deployment rejected: %s",
                reason or "unspecified reason",
                extra={"deployment_attempt": payload},
            )

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(payload, sort_keys=True) + "\n")

        return attempt

    def get_attempts(
        self,
        *,
        model_id: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[DeploymentAttempt]:
        """Return persisted attempts filtered by model and date bounds."""

        if not self.log_path.exists():
            return []

        attempts: list[DeploymentAttempt] = []
        with self.log_path.open("r", encoding="utf-8") as audit_file:
            for line in audit_file:
                stripped = line.strip()
                if not stripped:
                    continue
                attempt = DeploymentAttempt.from_dict(json.loads(stripped))
                if model_id is not None and attempt.model_id != model_id:
                    continue
                if start_date is not None and attempt.timestamp < _aware(start_date):
                    continue
                if end_date is not None and attempt.timestamp > _aware(end_date):
                    continue
                attempts.append(attempt)
        return attempts


def _aware(timestamp: datetime) -> datetime:
    """Return a timezone-aware UTC timestamp for comparisons."""

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp


__all__ = ["DeploymentAttempt", "DeploymentAuditLog"]
