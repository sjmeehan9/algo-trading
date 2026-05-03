"""Configuration schema validation utilities."""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass
from importlib import resources
from typing import Any

from algotrading.src.config.settings import Settings
from jsonschema import Draft202012Validator, ValidationError

logger = logging.getLogger(__name__)


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


def validate_configuration(settings: Settings) -> tuple[bool, list[str]]:
    """Validate runtime settings for application startup.

    Args:
        settings: Resolved application settings.

    Returns:
        Tuple containing validation success and actionable error messages.
    """

    errors: list[str] = []

    if not settings.api_key.strip():
        errors.append("ALGOTRADING_API_KEY is required")

    has_ib_config = bool(settings.ib_host.strip() and settings.ib_port > 0)
    has_alpaca_config = bool(settings.alpaca_api_key and settings.alpaca_secret_key)
    if not has_ib_config and not has_alpaca_config:
        errors.append(
            "At least one broker must be configured: Interactive Brokers host/port "
            "or Alpaca API credentials"
        )

    if settings.default_broker == "alpaca" and not has_alpaca_config:
        errors.append(
            "Alpaca API credentials are required when ALGOTRADING_DEFAULT_BROKER=alpaca"
        )

    if settings.is_production():
        if settings.debug:
            errors.append("Debug mode must be disabled in production")
        if len(settings.api_key) < 32:
            errors.append(
                "ALGOTRADING_API_KEY must be at least 32 characters in production"
            )
        if any("localhost" in origin for origin in settings.cors_origins):
            logger.warning("Configuration warning: CORS allows localhost in production")

    warnings: list[str] = []
    if not settings.openai_api_key:
        warnings.append(
            "OpenAI API key not set - LLM hyperparameter optimization uses fallback mode"
        )
    if not settings.benzinga_api_key and not settings.alphavantage_api_key:
        warnings.append(
            "No news provider API keys set - live news sentiment features are disabled"
        )
    if settings.news_provider == settings.news_fallback_provider:
        warnings.append("News provider fallback matches primary provider")

    for warning in warnings:
        logger.warning("Configuration warning: %s", warning)

    return len(errors) == 0, errors


def mask_secret(value: str | None, visible_chars: int = 4) -> str:
    """Mask a secret value for logs or diagnostics.

    Args:
        value: Secret value to mask.
        visible_chars: Number of leading characters to retain for operator
            recognition.

    Returns:
        Masked secret string.
    """

    if not value:
        return "****"
    if visible_chars <= 0 or len(value) <= visible_chars:
        return "****"
    return f"{value[:visible_chars]}{'*' * (len(value) - visible_chars)}"


def log_configuration(settings: Settings) -> None:
    """Log sanitized configuration values for startup diagnostics.

    Args:
        settings: Resolved application settings.
    """

    logger.info(
        "Configuration loaded",
        extra={"configuration": configuration_snapshot(settings)},
    )


def configuration_snapshot(settings: Settings) -> dict[str, object]:
    """Return a sanitized configuration snapshot suitable for logs.

    Args:
        settings: Resolved application settings.

    Returns:
        Dictionary containing non-sensitive values and masked secret markers.
    """

    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "debug": settings.debug,
        "log_level": settings.log_level,
        "api_host": settings.api_host,
        "api_port": settings.api_port,
        "api_key": mask_secret(settings.api_key),
        "cors_origins": settings.cors_origins,
        "default_broker": settings.default_broker,
        "ib_host": settings.ib_host,
        "ib_port": settings.ib_port,
        "ib_client_id": settings.ib_client_id,
        "alpaca_api_key": mask_secret(settings.alpaca_api_key),
        "alpaca_secret_key": mask_secret(settings.alpaca_secret_key),
        "alpaca_paper": settings.alpaca_paper,
        "alpaca_data_feed": settings.alpaca_data_feed,
        "benzinga_api_key": mask_secret(settings.benzinga_api_key),
        "alphavantage_api_key": mask_secret(settings.alphavantage_api_key),
        "news_provider": settings.news_provider,
        "news_fallback_provider": settings.news_fallback_provider,
        "openai_api_key": mask_secret(settings.openai_api_key),
        "openai_model": settings.openai_model,
        "openai_timeout_seconds": settings.openai_timeout_seconds,
        "data_path": settings.data_path,
        "model_path": settings.model_path,
        "log_path": settings.log_path,
        "max_position_pct": settings.max_position_pct,
        "min_confidence": settings.min_confidence,
    }


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
