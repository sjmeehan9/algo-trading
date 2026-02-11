"""Configuration schema validation utilities."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator, ValidationError


@dataclass(frozen=True)
class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    message: str
    errors: tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.errors:
            return self.message
        details = "\n".join(self.errors)
        return f"{self.message}\n{details}"


def validate_pipeline_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a pipeline configuration dictionary.

    Args:
        config: Pipeline configuration data.

    Returns:
        Tuple containing validation success flag and list of error messages.
    """

    schema = _load_schema("pipeline_schema.json")
    return _validate_with_schema(config, schema)


def validate_runtime_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a runtime configuration dictionary.

    YAML parsers may deserialise date-like strings (e.g. ``2024-10-21``) as
    ``datetime.date`` objects.  A shallow coercion step converts them back to
    ISO-format strings so the JSON-Schema string checks pass transparently.

    Args:
        config: Runtime configuration data.

    Returns:
        Tuple containing validation success flag and list of error messages.
    """

    coerced = _coerce_yaml_dates(config)
    schema = _load_schema("config_schema.json")
    return _validate_with_schema(coerced, schema)


def _coerce_yaml_dates(data: Any) -> Any:
    """Recursively convert ``datetime.date`` values to ISO-format strings.

    PyYAML deserialises bare dates (e.g. ``2024-10-21``) as
    ``datetime.date`` objects.  This helper normalises them so JSON-Schema
    ``"type": "string"`` checks work correctly.

    Args:
        data: Arbitrary nested data loaded from YAML.

    Returns:
        A copy of *data* with all ``datetime.date`` leaves replaced by strings.
    """

    if isinstance(data, dict):
        return {key: _coerce_yaml_dates(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_coerce_yaml_dates(item) for item in data]
    if isinstance(data, datetime.date) and not isinstance(data, datetime.datetime):
        return data.isoformat()
    return data


def _load_schema(schema_name: str) -> dict[str, Any]:
    with resources.files("algotrading.src.config.schemas").joinpath(schema_name).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def _validate_with_schema(
    payload: dict[str, Any], schema: dict[str, Any]
) -> tuple[bool, list[str]]:
    validator = Draft202012Validator(schema)
    errors = [
        format_validation_error(error) for error in validator.iter_errors(payload)
    ]
    if errors:
        return False, errors
    return True, []


def format_validation_error(error: ValidationError) -> str:
    """Convert jsonschema validation error to a readable message.

    Args:
        error: Validation error from jsonschema.

    Returns:
        Human-readable error message.
    """

    field_path = _format_error_path(error)

    if error.validator == "required":
        missing_field = _extract_missing_field(error)
        path = field_path
        if missing_field:
            path = f"{field_path}.{missing_field}" if field_path else missing_field
        return f"{path}: Required field is missing"

    if error.validator == "enum":
        expected = error.validator_value
        actual = error.instance
        return f"{field_path}: Expected one of {expected}, got {actual!r}"

    if error.validator == "type":
        expected = error.validator_value
        actual_type = type(error.instance).__name__
        return f"{field_path}: Expected type {expected}, got {actual_type}"

    if error.validator == "pattern":
        return f"{field_path}: Value does not match expected pattern"

    if error.validator == "minimum":
        return f"{field_path}: Value must be >= {error.validator_value}"

    if error.validator == "maximum":
        return f"{field_path}: Value must be <= {error.validator_value}"

    if error.validator == "minItems":
        return f"{field_path}: Must contain at least {error.validator_value} items"

    if error.validator == "maxItems":
        return f"{field_path}: Must contain at most {error.validator_value} items"

    if error.validator == "minLength":
        return f"{field_path}: Must not be empty"

    return f"{field_path}: {error.message}"


def _format_error_path(error: ValidationError) -> str:
    path = ""
    for part in error.absolute_path:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            if path:
                path += "."
            path += str(part)
    return path


def _extract_missing_field(error: ValidationError) -> str | None:
    if not error.message:
        return None
    segments = error.message.split("'")
    if len(segments) >= 2:
        return segments[1]
    return None
