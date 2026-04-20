"""Utility helpers shared by concrete news provider sources."""

from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from threading import Lock

_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class RateLimiter:
    """Simple thread-safe rate limiter using fixed minimum request spacing."""

    def __init__(self, calls_per_minute: int) -> None:
        """Initialize limiter with allowed request volume per minute."""

        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be greater than 0")
        self._calls_per_minute = calls_per_minute
        self._min_interval_seconds = 60.0 / float(calls_per_minute)
        self._lock = Lock()
        self._last_call_monotonic: float | None = None

    def wait(self) -> None:
        """Block the caller until the next request is allowed."""

        with self._lock:
            now = time.monotonic()
            if self._last_call_monotonic is None:
                self._last_call_monotonic = now
                return

            elapsed = now - self._last_call_monotonic
            remaining = self._min_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
                now = time.monotonic()

            self._last_call_monotonic = now


def ensure_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime from naive or aware input."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_benzinga_datetime(value: str) -> datetime:
    """Parse Benzinga datetime strings into UTC-aware datetimes."""

    candidate = value.strip()
    if not candidate:
        raise ValueError("Benzinga datetime value cannot be empty")

    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"

    parse_attempts = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
    )

    try:
        return ensure_utc(datetime.fromisoformat(candidate))
    except ValueError:
        pass

    for fmt in parse_attempts:
        try:
            return ensure_utc(datetime.strptime(candidate, fmt))
        except ValueError:
            continue

    try:
        return ensure_utc(parsedate_to_datetime(candidate))
    except (ValueError, TypeError):
        pass

    raise ValueError(f"Unsupported Benzinga datetime format: {value}")


def generate_stable_id(url: str, timestamp_value: str) -> str:
    """Generate deterministic article IDs from URL and timestamp string."""

    payload = f"{url.strip()}|{timestamp_value.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def resolve_env_placeholder(value: str) -> str:
    """Resolve ${ENV_VAR} placeholders, returning the original value otherwise."""

    text = value.strip()
    match = _ENV_PATTERN.match(text)
    if match is None:
        return text

    env_name = match.group(1)
    resolved = os.getenv(env_name)
    if resolved is None or not resolved.strip():
        raise ValueError(f"Environment variable '{env_name}' is not set")
    return resolved.strip()


__all__ = [
    "RateLimiter",
    "ensure_utc",
    "parse_benzinga_datetime",
    "generate_stable_id",
    "resolve_env_placeholder",
]
