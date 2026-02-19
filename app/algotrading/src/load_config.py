import json
import logging

import yaml
from algotrading.src.config.validation import (
    ConfigValidationError,
    validate_pipeline_config,
    validate_runtime_config,
)
from algotrading.src.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# Load a YAML config file
def config_loader(filepath, *, validate: bool = True) -> dict:
    """Load and optionally validate a YAML configuration file.

    Args:
        filepath: Path to the YAML configuration file.
        validate: When True (default), validate against the runtime config
            schema.  Pass False for script-specific configs that use a
            different structure (e.g. historical_data.yml, live_streaming.yml).

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigurationError: If the file is missing, unparseable, or fails
            validation.
    """
    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
            if validate:
                is_valid, errors = validate_runtime_config(config)
                if not is_valid:
                    raise ConfigValidationError(
                        f"Config validation failed for {filepath}", tuple(errors)
                    )
            return config
    except FileNotFoundError as exc:
        logger.error("Configuration file not found", extra={"filepath": filepath})
        raise ConfigurationError(
            "Configuration file not found",
            config_file=filepath,
            context={"error": str(exc)},
        ) from exc
    except yaml.YAMLError as exc:
        logger.error("Failed to parse YAML configuration", extra={"filepath": filepath})
        raise ConfigurationError(
            f"Failed to parse YAML configuration: {filepath}",
            config_file=filepath,
            context={"error": str(exc)},
        ) from exc
    except ConfigValidationError as exc:
        logger.error("Configuration validation failed", extra={"filepath": filepath})
        raise ConfigurationError(
            f"Configuration validation failed: {filepath}",
            config_file=filepath,
            context={"error": str(exc)},
        ) from exc


# Load a JSON pipeline file
def pipeline_loader(filepath) -> dict:
    try:
        with open(filepath, "r") as f:
            pipeline = json.load(f)
            is_valid, errors = validate_pipeline_config(pipeline)
            if not is_valid:
                raise ConfigValidationError(
                    f"Pipeline validation failed for {filepath}", tuple(errors)
                )
            return pipeline
    except FileNotFoundError as exc:
        logger.error("Pipeline file not found", extra={"filepath": filepath})
        raise ConfigurationError(
            "Pipeline file not found",
            config_file=filepath,
            context={"error": str(exc)},
        ) from exc
    except json.decoder.JSONDecodeError as exc:
        logger.error("Failed to parse JSON pipeline", extra={"filepath": filepath})
        raise ConfigurationError(
            f"Failed to parse JSON pipeline: {filepath}",
            config_file=filepath,
            context={"error": str(exc)},
        ) from exc
    except ConfigValidationError as exc:
        logger.error("Pipeline validation failed", extra={"filepath": filepath})
        raise ConfigurationError(
            f"Pipeline validation failed: {filepath}",
            config_file=filepath,
            context={"error": str(exc)},
        ) from exc
