"""Broker registry and runtime broker selection utilities."""

from __future__ import annotations

import inspect
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import ClassVar

import yaml
from algotrading.src.broker.adapter import BrokerAdapter
from algotrading.src.broker.adapters.alpaca import AlpacaAdapter
from algotrading.src.broker.exceptions import (
    BrokerConfigurationError,
    BrokerConnectionError,
    NoBrokerAvailableError,
)
from algotrading.src.broker.ib_adapter import InteractiveBrokersAdapter

logger = logging.getLogger(__name__)

_ENV_REFERENCE_PATTERN = re.compile(
    r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::(?P<default>[^}]*))?\}$"
)


@dataclass(frozen=True, slots=True)
class BrokerConfig:
    """Validated broker registry configuration.

    Args:
        primary_broker: Broker name attempted first.
        fallback_brokers: Broker names attempted if the primary fails.
        broker_configs: Adapter constructor configuration by broker name.
        connection_params: Adapter connection parameters by broker name.
    """

    primary_broker: str
    fallback_brokers: tuple[str, ...] = ()
    broker_configs: dict[str, dict[str, object]] = field(default_factory=dict)
    connection_params: dict[str, dict[str, object]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "BrokerConfig":
        """Build a broker config from a YAML-style mapping.

        Args:
            payload: Mapping loaded from broker configuration YAML.

        Returns:
            Normalized broker configuration.

        Raises:
            BrokerConfigurationError: If required fields are missing or invalid.
        """

        primary_broker = _normalise_broker_name(payload.get("primary_broker"))
        if not primary_broker:
            raise BrokerConfigurationError(
                message="Broker configuration requires a primary_broker value"
            )

        return cls(
            primary_broker=primary_broker,
            fallback_brokers=_normalise_broker_sequence(
                payload.get("fallback_brokers", ())
            ),
            broker_configs=_normalise_nested_mapping(payload.get("broker_configs")),
            connection_params=_normalise_nested_mapping(
                payload.get("connection_params")
            ),
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "BrokerConfig":
        """Load broker configuration from a YAML file.

        Args:
            config_path: Path to ``brokers.yml``.

        Returns:
            Parsed and normalized broker configuration.

        Raises:
            BrokerConfigurationError: If the file is missing or invalid.
        """

        resolved_path = Path(config_path)
        try:
            with resolved_path.open("r", encoding="utf-8") as file_handle:
                payload = yaml.safe_load(file_handle) or {}
        except FileNotFoundError as exc:
            raise BrokerConfigurationError(
                message="Broker configuration file not found",
                config_path=str(resolved_path),
                reason=str(exc),
            ) from exc
        except yaml.YAMLError as exc:
            raise BrokerConfigurationError(
                message="Broker configuration YAML is invalid",
                config_path=str(resolved_path),
                reason=str(exc),
            ) from exc
        except OSError as exc:
            raise BrokerConfigurationError(
                message="Broker configuration file could not be read",
                config_path=str(resolved_path),
                reason=str(exc),
            ) from exc

        if not isinstance(payload, Mapping):
            raise BrokerConfigurationError(
                message="Broker configuration root must be a mapping",
                config_path=str(resolved_path),
            )

        try:
            return cls.from_mapping(payload)
        except BrokerConfigurationError as exc:
            exc.config_path = str(resolved_path)
            raise

    def candidate_brokers(self) -> tuple[str, ...]:
        """Return primary and fallback broker names in de-duplicated order."""

        seen: set[str] = set()
        candidates: list[str] = []
        for broker_name in (self.primary_broker, *self.fallback_brokers):
            if broker_name and broker_name not in seen:
                seen.add(broker_name)
                candidates.append(broker_name)
        return tuple(candidates)

    def adapter_config(self, broker_name: str) -> dict[str, object]:
        """Return a copy of constructor configuration for one broker."""

        return dict(self.broker_configs.get(_normalise_broker_name(broker_name), {}))

    def connection_config(self, broker_name: str) -> dict[str, object]:
        """Return a copy of connection parameters for one broker."""

        return dict(self.connection_params.get(_normalise_broker_name(broker_name), {}))


class BrokerRegistry:
    """Singleton registry for broker adapters and runtime broker selection."""

    _instance: ClassVar["BrokerRegistry | None"] = None
    DEFAULT_CONFIG_PATH: ClassVar[Path] = (
        Path(__file__).resolve().parents[2] / "config" / "brokers.yml"
    )

    def __new__(
        cls,
        config_path: str | Path | None = None,
        include_builtins: bool = True,
    ) -> "BrokerRegistry":
        """Return the process-wide registry instance."""

        del config_path, include_builtins
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        config_path: str | Path | None = None,
        include_builtins: bool = True,
    ) -> None:
        """Initialize the singleton registry once.

        Args:
            config_path: Optional path to broker configuration YAML.
            include_builtins: Whether to register Interactive Brokers and Alpaca.
        """

        if getattr(self, "_initialized", False):
            if config_path is not None:
                self._config_path = Path(config_path)
            return
        self._initialize(config_path=config_path, include_builtins=include_builtins)

    @classmethod
    def isolated(
        cls,
        *,
        config_path: str | Path | None = None,
        include_builtins: bool = True,
    ) -> "BrokerRegistry":
        """Create a non-singleton registry for scoped services and tests."""

        instance = object.__new__(cls)
        instance._initialize(config_path=config_path, include_builtins=include_builtins)
        return instance

    def register(self, name: str, adapter_class: type[BrokerAdapter]) -> None:
        """Register a broker adapter class by name.

        Args:
            name: Registry key used in configuration.
            adapter_class: Concrete ``BrokerAdapter`` implementation class.

        Raises:
            TypeError: If ``adapter_class`` does not implement ``BrokerAdapter``.
            ValueError: If ``name`` is empty.
        """

        broker_name = _normalise_broker_name(name)
        if not broker_name:
            raise ValueError("Broker adapter name must be non-empty")
        if not issubclass(adapter_class, BrokerAdapter):
            raise TypeError(f"{adapter_class} must implement BrokerAdapter")

        with self._lock:
            self._adapters[broker_name] = adapter_class
        logger.info("Registered broker adapter: %s", broker_name)

    def list_available(self) -> list[str]:
        """List registered broker adapter names."""

        with self._lock:
            return list(self._adapters.keys())

    def get_broker(
        self,
        config: BrokerConfig | Mapping[str, object] | None = None,
        *,
        force_reconnect: bool = False,
    ) -> BrokerAdapter:
        """Return a connected broker adapter using primary/fallback config.

        Args:
            config: Optional config object or raw config mapping. When omitted,
                ``brokers.yml`` is loaded from the application config directory.
            force_reconnect: When True, reconnect even if the active primary is
                already connected.

        Returns:
            Connected broker adapter instance.

        Raises:
            NoBrokerAvailableError: If every configured broker fails.
        """

        broker_config = self._coerce_config(config)
        candidates = broker_config.candidate_brokers()
        failure_reasons: list[str] = []

        for broker_name in candidates:
            if not self._is_registered(broker_name):
                failure_reasons.append(f"{broker_name}: adapter is not registered")
                continue

            with self._lock:
                active_matches_candidate = (
                    broker_name == self._active_broker_name
                    and self._active_broker is not None
                    and self._active_broker.is_connected()
                )
                if active_matches_candidate and not force_reconnect:
                    return self._active_broker

            try:
                adapter = self._connect_named_broker(broker_name, broker_config)
            except Exception as exc:
                failure_reasons.append(f"{broker_name}: {exc}")
                logger.warning("Broker connection attempt failed: %s", broker_name)
                continue

            if broker_name != broker_config.primary_broker:
                logger.warning("Using fallback broker: %s", broker_name)
            else:
                logger.info("Connected to primary broker: %s", broker_name)
            return adapter

        raise NoBrokerAvailableError(
            message="No broker available for connection",
            attempted_brokers=candidates,
            failure_reasons=tuple(failure_reasons),
        )

    def get_active_broker(self) -> BrokerAdapter | None:
        """Return the currently active broker without reconnecting."""

        with self._lock:
            return self._active_broker

    def get_active_broker_name(self) -> str | None:
        """Return the registry name of the currently active broker."""

        with self._lock:
            return self._active_broker_name

    def switch_broker(
        self,
        broker_name: str,
        config: BrokerConfig | Mapping[str, object] | None = None,
        connection_params: Mapping[str, object] | None = None,
    ) -> BrokerAdapter:
        """Switch the active broker at runtime.

        Args:
            broker_name: Registered broker name to activate.
            config: Adapter constructor configuration or a full ``BrokerConfig``.
            connection_params: Optional connection parameters for ``connect()``.

        Returns:
            Newly connected active broker adapter.

        Raises:
            ValueError: If ``broker_name`` is not registered.
            BrokerConnectionError: If the new broker cannot connect.
        """

        normalised_name = _normalise_broker_name(broker_name)
        if not self._is_registered(normalised_name):
            raise ValueError(f"Unknown broker: {broker_name}")

        adapter_config, connect_config = self._switch_configs(
            normalised_name, config, connection_params
        )

        with self._lock:
            previous = self._active_broker
            self._active_broker = None
            self._active_broker_name = None

        if previous is not None:
            previous.disconnect()

        adapter = self._create_adapter(normalised_name, adapter_config)
        self._connect_adapter(normalised_name, adapter, connect_config)
        with self._lock:
            self._active_broker = adapter
            self._active_broker_name = normalised_name
        logger.info("Switched to broker: %s", normalised_name)
        return adapter

    def load_config(self) -> BrokerConfig:
        """Load broker configuration from the registry config path."""

        return BrokerConfig.from_yaml(self._config_path)

    def _initialize(
        self,
        *,
        config_path: str | Path | None,
        include_builtins: bool,
    ) -> None:
        self._lock = RLock()
        self._adapters: dict[str, type[BrokerAdapter]] = {}
        self._active_broker: BrokerAdapter | None = None
        self._active_broker_name: str | None = None
        self._config_path = (
            Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        )
        self._initialized = True
        if include_builtins:
            self._register_builtin_adapters()

    def _register_builtin_adapters(self) -> None:
        self.register("interactive_brokers", InteractiveBrokersAdapter)
        self.register("alpaca", AlpacaAdapter)

    def _coerce_config(
        self, config: BrokerConfig | Mapping[str, object] | None
    ) -> BrokerConfig:
        if config is None:
            return self.load_config()
        if isinstance(config, BrokerConfig):
            return config
        return BrokerConfig.from_mapping(config)

    def _is_registered(self, broker_name: str) -> bool:
        with self._lock:
            return broker_name in self._adapters

    def _connect_named_broker(
        self,
        broker_name: str,
        broker_config: BrokerConfig,
    ) -> BrokerAdapter:
        adapter = self._create_adapter(
            broker_name, broker_config.adapter_config(broker_name)
        )
        self._connect_adapter(
            broker_name,
            adapter,
            broker_config.connection_config(broker_name),
        )

        with self._lock:
            previous = self._active_broker
            self._active_broker = adapter
            self._active_broker_name = broker_name

        if previous is not None and previous is not adapter:
            previous.disconnect()
        return adapter

    def _create_adapter(
        self,
        broker_name: str,
        adapter_config: Mapping[str, object],
    ) -> BrokerAdapter:
        with self._lock:
            adapter_class = self._adapters[broker_name]

        try:
            if _constructor_accepts_config(adapter_class):
                return adapter_class(dict(adapter_config))
            return adapter_class()
        except Exception as exc:
            raise BrokerConnectionError(
                message=f"Failed to initialize broker adapter '{broker_name}'",
                broker_name=broker_name,
                host=broker_name,
                port=0,
                reason=str(exc),
            ) from exc

    def _connect_adapter(
        self,
        broker_name: str,
        adapter: BrokerAdapter,
        connection_params: Mapping[str, object],
    ) -> None:
        connect_kwargs = _filter_connect_kwargs(adapter, connection_params)
        try:
            result = adapter.connect(**connect_kwargs)
        except BrokerConnectionError:
            raise
        except Exception as exc:
            raise _broker_connection_error(
                broker_name, connection_params, str(exc)
            ) from exc

        if result is False:
            raise _broker_connection_error(
                broker_name, connection_params, "connect() returned False"
            )
        if not adapter.is_connected():
            raise _broker_connection_error(
                broker_name, connection_params, "adapter did not report connected state"
            )

    def _switch_configs(
        self,
        broker_name: str,
        config: BrokerConfig | Mapping[str, object] | None,
        connection_params: Mapping[str, object] | None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        if isinstance(config, BrokerConfig):
            return config.adapter_config(broker_name), config.connection_config(
                broker_name
            )
        if config is not None and "primary_broker" in config:
            broker_config = BrokerConfig.from_mapping(config)
            return broker_config.adapter_config(
                broker_name
            ), broker_config.connection_config(broker_name)
        adapter_config = dict(config or {})
        connect_config = dict(connection_params or {})
        return adapter_config, connect_config


def _normalise_broker_name(value: object) -> str:
    return str(value or "").strip().lower()


def _normalise_broker_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        normalised = _normalise_broker_name(value)
        return (normalised,) if normalised else ()
    if not isinstance(value, (list, tuple)):
        raise BrokerConfigurationError(
            message="fallback_brokers must be a string or list of strings",
            reason=f"got {type(value).__name__}",
        )
    brokers = [_normalise_broker_name(item) for item in value]
    return tuple(broker_name for broker_name in brokers if broker_name)


def _normalise_nested_mapping(value: object) -> dict[str, dict[str, object]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BrokerConfigurationError(
            message="Broker configuration sections must be mappings",
            reason=f"got {type(value).__name__}",
        )

    normalised: dict[str, dict[str, object]] = {}
    for raw_broker_name, raw_config in value.items():
        broker_name = _normalise_broker_name(raw_broker_name)
        if not broker_name:
            continue
        if raw_config is None:
            normalised[broker_name] = {}
            continue
        if not isinstance(raw_config, Mapping):
            raise BrokerConfigurationError(
                message="Broker-specific configuration must be a mapping",
                broker_name=broker_name,
                reason=f"got {type(raw_config).__name__}",
            )
        normalised[broker_name] = {
            str(config_key): _resolve_env_reference(config_value)
            for config_key, config_value in raw_config.items()
        }
    return normalised


def _resolve_env_reference(value: object) -> object:
    if isinstance(value, str):
        match = _ENV_REFERENCE_PATTERN.fullmatch(value.strip())
        if match is None:
            return value
        env_name = match.group("name")
        default_value = match.group("default")
        return os.getenv(env_name, default_value or "")
    if isinstance(value, Mapping):
        return {
            str(nested_key): _resolve_env_reference(nested_value)
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_resolve_env_reference(item) for item in value]
    return value


def _constructor_accepts_config(adapter_class: type[BrokerAdapter]) -> bool:
    signature = inspect.signature(adapter_class)
    parameters = list(signature.parameters.values())
    if any(
        parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
        for parameter in parameters
    ):
        return True
    if "config" in signature.parameters:
        return True
    return bool(parameters) and parameters[0].default is not inspect.Parameter.empty


def _filter_connect_kwargs(
    adapter: BrokerAdapter, connection_params: Mapping[str, object]
) -> dict[str, object]:
    signature = inspect.signature(adapter.connect)
    if any(
        parameter.kind == parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(connection_params)
    return {
        key: value
        for key, value in connection_params.items()
        if key in signature.parameters
    }


def _broker_connection_error(
    broker_name: str,
    connection_params: Mapping[str, object],
    reason: str,
) -> BrokerConnectionError:
    host = str(connection_params.get("host") or broker_name)
    raw_port = connection_params.get("port") or 0
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        port = 0
    return BrokerConnectionError(
        message=f"Failed to connect broker '{broker_name}'",
        broker_name=broker_name,
        host=host,
        port=port,
        reason=reason,
    )


__all__ = ["BrokerConfig", "BrokerRegistry"]
