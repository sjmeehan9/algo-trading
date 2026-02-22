"""Exceptions for model trainer abstractions.

This module defines the canonical exception hierarchy used by trainer
implementations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class TrainerError(Exception):
    """Base exception for all trainer lifecycle failures.

    Args:
        message: Human-readable failure message.
        trainer_name: Optional trainer implementation identifier.
    """

    message: str
    trainer_name: str | None = None

    def __post_init__(self) -> None:
        super().__init__(self.__str__())

    def __str__(self) -> str:
        """Return a readable trainer error representation."""

        if self.trainer_name:
            return f"[{self.trainer_name}] {self.message}"
        return self.message


@dataclass(kw_only=True)
class ModelNotTrainedError(TrainerError):
    """Raised when an operation requires a trained model but none is ready."""


@dataclass(kw_only=True)
class ModelLoadError(TrainerError):
    """Raised when loading a persisted model fails.

    Args:
        message: Human-readable failure message.
        filepath: Path to the model file that failed to load.
        reason: Optional low-level error reason.
        trainer_name: Optional trainer implementation identifier.
    """

    filepath: str
    reason: str | None = None

    def __str__(self) -> str:
        """Return a readable model-load error representation."""

        detail = f", reason={self.reason}" if self.reason else ""
        context = f"filepath={self.filepath}{detail}"
        if self.trainer_name:
            return f"[{self.trainer_name}] {self.message} ({context})"
        return f"{self.message} ({context})"


@dataclass(kw_only=True)
class TrainingError(TrainerError):
    """Raised when model training fails.

    Args:
        message: Human-readable failure message.
        stage: Optional stage where failure occurred.
        reason: Optional low-level error reason.
        trainer_name: Optional trainer implementation identifier.
    """

    stage: str | None = None
    reason: str | None = None

    def __str__(self) -> str:
        """Return a readable training error representation."""

        details: list[str] = []
        if self.stage:
            details.append(f"stage={self.stage}")
        if self.reason:
            details.append(f"reason={self.reason}")
        suffix = f" ({', '.join(details)})" if details else ""
        if self.trainer_name:
            return f"[{self.trainer_name}] {self.message}{suffix}"
        return f"{self.message}{suffix}"


@dataclass(kw_only=True)
class InvalidEnvironmentError(TrainerError):
    """Raised when an incompatible gymnasium environment is provided.

    Args:
        message: Human-readable failure message.
        env_type: Optional provided environment type name.
        reason: Optional low-level reason.
        trainer_name: Optional trainer implementation identifier.
    """

    env_type: str | None = None
    reason: str | None = None

    def __str__(self) -> str:
        """Return a readable invalid-environment error representation."""

        details: list[str] = []
        if self.env_type:
            details.append(f"env_type={self.env_type}")
        if self.reason:
            details.append(f"reason={self.reason}")
        suffix = f" ({', '.join(details)})" if details else ""
        if self.trainer_name:
            return f"[{self.trainer_name}] {self.message}{suffix}"
        return f"{self.message}{suffix}"


__all__ = [
    "TrainerError",
    "ModelNotTrainedError",
    "ModelLoadError",
    "TrainingError",
    "InvalidEnvironmentError",
]
