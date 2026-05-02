"""Exceptions for live trading session management."""

from __future__ import annotations


class TradingSessionError(Exception):
    """Base exception for trading session failures."""


class InvalidSessionStateError(TradingSessionError):
    """Raised when a lifecycle operation is invalid for a session state."""


class SessionConfigurationError(TradingSessionError):
    """Raised when a session configuration cannot be used safely."""


class SessionNotFoundError(TradingSessionError):
    """Raised when a requested trading session does not exist."""

    def __init__(self, session_id: str) -> None:
        """Initialize the error with the missing session identifier."""

        super().__init__(f"Trading session '{session_id}' was not found")
        self.session_id = session_id


class SessionPersistenceError(TradingSessionError):
    """Raised when persisted session state cannot be read or written."""
