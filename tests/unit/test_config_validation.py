"""Tests for configuration schema validation."""

from __future__ import annotations

import datetime
import json
from copy import deepcopy
from pathlib import Path

import yaml
from algotrading.src.config.validation import (
    ConfigValidationError,
    format_validation_error,
    validate_pipeline_config,
    validate_runtime_config,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


# ---------------------------------------------------------------------------
# Pipeline schema – valid & invalid
# ---------------------------------------------------------------------------


def test_validate_pipeline_config_valid(
    mock_pipeline_config: dict[str, object],
) -> None:
    """Valid pipeline configs should pass schema validation."""

    is_valid, errors = validate_pipeline_config(mock_pipeline_config)

    assert is_valid
    assert errors == []


def test_validate_pipeline_config_missing_field(
    mock_pipeline_config: dict[str, object],
) -> None:
    """Missing required pipeline fields should fail validation."""

    config = deepcopy(mock_pipeline_config)
    config.pop("pipeline", None)

    is_valid, errors = validate_pipeline_config(config)

    assert not is_valid
    assert any("pipeline" in error for error in errors)


def test_validate_pipeline_config_invalid_enum(
    mock_pipeline_config: dict[str, object],
) -> None:
    """Invalid enum values should be reported clearly."""

    config = deepcopy(mock_pipeline_config)
    pipeline = config["pipeline"]
    pipeline["pipeline_type"] = "invalid"

    is_valid, errors = validate_pipeline_config(config)

    assert not is_valid
    assert any("pipeline.pipeline_type" in error for error in errors)


def test_validate_pipeline_config_invalid_type(
    mock_pipeline_config: dict[str, object],
) -> None:
    """Wrong value types should be reported with expected and actual types."""

    config = deepcopy(mock_pipeline_config)
    config["pipeline"]["trading_config"]["balance_multiplier"] = "not_a_number"

    is_valid, errors = validate_pipeline_config(config)

    assert not is_valid
    assert any("balance_multiplier" in e and "type" in e.lower() for e in errors)


def test_validate_pipeline_config_out_of_range(
    mock_pipeline_config: dict[str, object],
) -> None:
    """Numeric values outside allowed range should fail."""

    config = deepcopy(mock_pipeline_config)
    config["pipeline"]["trading_config"]["balance_multiplier"] = 5.0

    is_valid, errors = validate_pipeline_config(config)

    assert not is_valid
    assert any("balance_multiplier" in e for e in errors)


def test_validate_pipeline_config_empty_columns(
    mock_pipeline_config: dict[str, object],
) -> None:
    """Empty required arrays should fail with minItems error."""

    config = deepcopy(mock_pipeline_config)
    config["pipeline"]["historical_data_config"]["columns"] = []

    is_valid, errors = validate_pipeline_config(config)

    assert not is_valid
    assert any("columns" in e for e in errors)


# ---------------------------------------------------------------------------
# Runtime schema – valid & invalid
# ---------------------------------------------------------------------------


def test_validate_runtime_config_valid(mock_config: dict[str, object]) -> None:
    """Valid runtime configs should pass schema validation."""

    is_valid, errors = validate_runtime_config(mock_config)

    assert is_valid
    assert errors == []


def test_validate_runtime_config_missing_required(
    mock_config: dict[str, object],
) -> None:
    """Missing required runtime fields should fail validation."""

    config = deepcopy(mock_config)
    config.pop("ip_address", None)

    is_valid, errors = validate_runtime_config(config)

    assert not is_valid
    assert any("ip_address" in error for error in errors)


def test_validate_runtime_config_task2_requirements(
    mock_config: dict[str, object],
) -> None:
    """Task 2 requires training-specific fields."""

    config = deepcopy(mock_config)
    config["task_selection"] = "task2"
    config.pop("training_date_list", None)

    is_valid, errors = validate_runtime_config(config)

    assert not is_valid
    assert any("training_date_list" in error for error in errors)


def test_validate_runtime_config_yaml_dates(
    mock_config: dict[str, object],
) -> None:
    """datetime.date values produced by PyYAML should be coerced to strings."""

    config = deepcopy(mock_config)
    config["date_list"] = [datetime.date(2024, 10, 21)]
    config["task_selection"] = "task1"

    is_valid, errors = validate_runtime_config(config)

    assert is_valid
    assert errors == []


# ---------------------------------------------------------------------------
# Error message formatting
# ---------------------------------------------------------------------------


def test_format_error_required_field() -> None:
    """Required-field errors should include path and field name."""

    config: dict[str, object] = {}
    _, errors = validate_pipeline_config(config)

    assert any("pipeline" in e and "Required field is missing" in e for e in errors)


def test_format_error_enum_value(mock_pipeline_config: dict[str, object]) -> None:
    """Enum errors should show expected values and actual value."""

    config = deepcopy(mock_pipeline_config)
    config["pipeline"]["model"]["model_type"] = "xgboost"

    _, errors = validate_pipeline_config(config)

    matching = [e for e in errors if "model_type" in e]
    assert len(matching) == 1
    assert "Expected one of" in matching[0]
    assert "'xgboost'" in matching[0]


def test_format_error_type_mismatch(mock_pipeline_config: dict[str, object]) -> None:
    """Type errors should show expected type and actual type."""

    config = deepcopy(mock_pipeline_config)
    config["pipeline"]["filename"] = 12345

    _, errors = validate_pipeline_config(config)

    matching = [e for e in errors if "filename" in e]
    assert len(matching) == 1
    assert "Expected type string" in matching[0]
    assert "int" in matching[0]


def test_format_error_minimum_value(mock_pipeline_config: dict[str, object]) -> None:
    """Minimum constraint errors should include the boundary value."""

    config = deepcopy(mock_pipeline_config)
    config["pipeline"]["model"]["model_config"]["n_steps"] = 0

    _, errors = validate_pipeline_config(config)

    matching = [e for e in errors if "n_steps" in e]
    assert len(matching) >= 1


# ---------------------------------------------------------------------------
# ConfigValidationError
# ---------------------------------------------------------------------------


def test_config_validation_error_str_no_errors() -> None:
    """ConfigValidationError with no detail errors shows only the message."""

    exc = ConfigValidationError(message="bad config")
    assert str(exc) == "bad config"


def test_config_validation_error_str_with_errors() -> None:
    """ConfigValidationError with detail errors joins them with newlines."""

    exc = ConfigValidationError(
        message="validation failed",
        errors=("field1: missing", "field2: wrong type"),
    )
    text = str(exc)
    assert "validation failed" in text
    assert "field1: missing" in text
    assert "field2: wrong type" in text


# ---------------------------------------------------------------------------
# Integration – real config files
# ---------------------------------------------------------------------------


def test_validate_real_pipeline_0001() -> None:
    """Existing pipeline_0001.json must pass schema validation."""

    pipeline_path = SCRIPTS_DIR / "pipeline_settings" / "pipeline_0001.json"
    with open(pipeline_path, encoding="utf-8") as f:
        config = json.load(f)

    is_valid, errors = validate_pipeline_config(config)

    assert is_valid, f"pipeline_0001.json failed: {errors}"


def test_validate_real_pipeline_0002() -> None:
    """Existing pipeline_0002.json must pass schema validation."""

    pipeline_path = SCRIPTS_DIR / "pipeline_settings" / "pipeline_0002.json"
    with open(pipeline_path, encoding="utf-8") as f:
        config = json.load(f)

    is_valid, errors = validate_pipeline_config(config)

    assert is_valid, f"pipeline_0002.json failed: {errors}"


def test_validate_real_config_yml() -> None:
    """Existing config.yml must pass schema validation (YAML dates coerced)."""

    config_path = SCRIPTS_DIR / "config.yml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    is_valid, errors = validate_runtime_config(config)

    assert is_valid, f"config.yml failed: {errors}"
