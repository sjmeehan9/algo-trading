"""Integration tests for configuration loader error handling."""

from __future__ import annotations

from pathlib import Path

import pytest
from algotrading.src.exceptions import ConfigurationError
from algotrading.src.load_config import config_loader, pipeline_loader


@pytest.mark.integration
def test_config_loader_missing_file_raises_configuration_error(tmp_path: Path) -> None:
    """Missing YAML file should raise ConfigurationError with chained cause."""

    missing_path = tmp_path / "does_not_exist.yml"

    with pytest.raises(ConfigurationError) as exc_info:
        config_loader(str(missing_path))

    assert "Configuration file not found" in str(exc_info.value)
    assert exc_info.value.config_file == str(missing_path)
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


@pytest.mark.integration
def test_pipeline_loader_invalid_json_raises_configuration_error(
    tmp_path: Path,
) -> None:
    """Invalid JSON should raise ConfigurationError with parse details."""

    bad_pipeline = tmp_path / "bad_pipeline.json"
    bad_pipeline.write_text("{ bad json", encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        pipeline_loader(str(bad_pipeline))

    assert "Failed to parse JSON pipeline" in str(exc_info.value)
    assert exc_info.value.config_file == str(bad_pipeline)


@pytest.mark.integration
def test_config_loader_validate_false_skips_runtime_validation(
    tmp_path: Path,
) -> None:
    """Script-specific YAML configs should load without runtime validation.

    Regression test for a bug where config_loader() unconditionally applied
    runtime schema validation to all YAML files, causing script-specific
    configs (historical_data.yml, live_streaming.yml, pipeline_types.yml) to
    fail with 'Required field is missing' errors.
    """

    script_config = tmp_path / "historical_data.yml"
    script_config.write_text(
        "bar_columns:\n  bar_date: 'date'\nstep_size:\n  5 secs:\n    barSize: 5\n",
        encoding="utf-8",
    )

    result = config_loader(str(script_config), validate=False)

    assert isinstance(result, dict)
    assert "bar_columns" in result
    assert result["bar_columns"]["bar_date"] == "date"


@pytest.mark.integration
def test_config_loader_validate_true_rejects_invalid_runtime_config(
    tmp_path: Path,
) -> None:
    """Runtime configs should still be validated by default."""

    bad_runtime = tmp_path / "config.yml"
    bad_runtime.write_text("bar_columns:\n  bar_date: 'date'\n", encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        config_loader(str(bad_runtime))

    assert "Configuration validation failed" in str(exc_info.value)
