"""Exceptions for supporting model registry operations."""

from __future__ import annotations


class RegistryError(Exception):
    """Base exception for supporting model registry errors."""


class ModelNotFoundError(RegistryError):
    """Raised when a model ID cannot be found in the registry."""


class ModelAlreadyExistsError(RegistryError):
    """Raised when attempting to register a duplicate model ID."""


class InvalidDependencyError(RegistryError):
    """Raised when a supporting model depends on another supporting model signal."""


class ModelStateError(RegistryError):
    """Raised when an invalid model state transition is requested."""


__all__ = [
    "RegistryError",
    "ModelNotFoundError",
    "ModelAlreadyExistsError",
    "InvalidDependencyError",
    "ModelStateError",
]
