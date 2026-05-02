"""Unit tests for broker registry configuration and health checks."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from algotrading.src.broker import (
    AccountInfo,
    BarData,
    BrokerAdapter,
    BrokerConfig,
    BrokerConfigurationError,
    BrokerConnectionError,
    BrokerHealth,
    BrokerRegistry,
    ContractSpec,
    NoBrokerAvailableError,
    OrderSpec,
    OrderStatus,
    PositionInfo,
)


class _RegistryTestAdapter(BrokerAdapter):
    """Configurable adapter used to test registry orchestration."""

    instances: list["_RegistryTestAdapter"] = []

    def __init__(self, config: dict[str, object] | None = None) -> None:
        self.config = dict(config or {})
        self.connected = False
        self.disconnected = False
        self.connect_calls: list[tuple[str, int, int]] = []
        self.account_calls = 0
        self.instances.append(self)

    def connect(self, host: str = "", port: int = 0, client_id: int = 0) -> None:
        self.connect_calls.append((host, port, client_id))
        if self.config.get("fail_connect"):
            raise BrokerConnectionError(
                message="configured failure",
                broker_name="registry_test",
                host=host,
                port=port,
                reason="fail_connect",
            )
        self.connected = True
        self.disconnected = False

    def disconnect(self) -> None:
        self.connected = False
        self.disconnected = True

    def is_connected(self) -> bool:
        return self.connected

    @property
    def account_id(self) -> str:
        return "TEST-ACCOUNT"

    def request_historical_data(
        self,
        contract: ContractSpec,
        end_datetime: datetime,
        duration: str,
        bar_size: str,
        data_type: str = "TRADES",
    ) -> list[BarData]:
        del contract, end_datetime, duration, bar_size, data_type
        return []

    def subscribe_realtime_data(
        self,
        contract: ContractSpec,
        bar_size: int,
        data_type: str,
        callback,
    ) -> int:
        del contract, bar_size, data_type, callback
        return 1

    def unsubscribe_realtime_data(self, subscription_id: int) -> None:
        del subscription_id

    def place_order(self, contract: ContractSpec, order: OrderSpec) -> str:
        del contract, order
        return "order-1"

    def cancel_order(self, order_id: str) -> None:
        del order_id

    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus(
            order_id=order_id,
            status="SUBMITTED",
            filled_quantity=0.0,
            remaining_quantity=1.0,
            average_fill_price=None,
            last_update=datetime.now(tz=UTC),
        )

    def register_order_callback(self, callback) -> None:
        del callback

    def get_account_info(self) -> AccountInfo:
        self.account_calls += 1
        return AccountInfo(
            account_id="TEST-ACCOUNT",
            cash_balance=1000.0,
            buying_power=2000.0,
            currency="USD",
        )

    def get_positions(self) -> list[PositionInfo]:
        return []

    def subscribe_account_updates(self, callback) -> None:
        callback(self.get_account_info())

    def subscribe_position_updates(self, callback) -> None:
        del callback

    def get_next_order_id(self) -> str:
        return "order-1"


class _NotBroker:
    """Non-adapter class used to validate registration guards."""


def _reset_test_adapter() -> None:
    _RegistryTestAdapter.instances.clear()


def _registry_config(
    primary: str = "primary",
    fallback: tuple[str, ...] = (),
    broker_configs: dict[str, dict[str, object]] | None = None,
    connection_params: dict[str, dict[str, object]] | None = None,
) -> BrokerConfig:
    return BrokerConfig(
        primary_broker=primary,
        fallback_brokers=fallback,
        broker_configs=broker_configs or {},
        connection_params=connection_params or {},
    )


def test_registry_registers_builtin_adapters() -> None:
    """Isolated registries register IB and Alpaca adapters by default."""

    registry = BrokerRegistry.isolated()

    assert registry.list_available() == ["interactive_brokers", "alpaca"]


def test_register_accepts_adapters_and_rejects_invalid_classes() -> None:
    """Custom registrations must implement the broker adapter interface."""

    registry = BrokerRegistry.isolated(include_builtins=False)

    registry.register("primary", _RegistryTestAdapter)

    assert registry.list_available() == ["primary"]
    with pytest.raises(TypeError):
        registry.register("invalid", _NotBroker)
    with pytest.raises(ValueError):
        registry.register("", _RegistryTestAdapter)


def test_broker_config_loads_yaml_and_resolves_env_refs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Broker YAML is normalized and environment placeholders are resolved."""

    monkeypatch.setenv("TEST_BROKER_TOKEN", "secret-token")
    config_path = tmp_path / "brokers.yml"
    config_path.write_text(
        """
primary_broker: PRIMARY
fallback_brokers:
  - fallback
  - primary
broker_configs:
  PRIMARY:
    token: ${TEST_BROKER_TOKEN}
connection_params:
  PRIMARY:
    host: localhost
    port: 4001
    client_id: 7
""",
        encoding="utf-8",
    )

    config = BrokerConfig.from_yaml(config_path)

    assert config.primary_broker == "primary"
    assert config.candidate_brokers() == ("primary", "fallback")
    assert config.adapter_config("primary")["token"] == "secret-token"
    assert config.connection_config("primary")["port"] == 4001


def test_broker_config_rejects_invalid_payloads(tmp_path) -> None:
    """Invalid broker YAML fails with typed configuration errors."""

    config_path = tmp_path / "brokers.yml"
    config_path.write_text("fallback_brokers: 5\n", encoding="utf-8")

    with pytest.raises(BrokerConfigurationError):
        BrokerConfig.from_yaml(config_path)


def test_get_broker_returns_connected_primary_adapter() -> None:
    """Primary broker is instantiated, connected, and marked active."""

    _reset_test_adapter()
    registry = BrokerRegistry.isolated(include_builtins=False)
    registry.register("primary", _RegistryTestAdapter)

    adapter = registry.get_broker(
        _registry_config(
            broker_configs={"primary": {"region": "test"}},
            connection_params={
                "primary": {"host": "localhost", "port": 4001, "client_id": 9}
            },
        )
    )

    assert adapter is registry.get_active_broker()
    assert registry.get_active_broker_name() == "primary"
    assert isinstance(adapter, _RegistryTestAdapter)
    assert adapter.config == {"region": "test"}
    assert adapter.connect_calls == [("localhost", 4001, 9)]


def test_get_broker_falls_back_when_primary_connection_fails() -> None:
    """Fallback broker is used when the primary raises during connect."""

    _reset_test_adapter()
    registry = BrokerRegistry.isolated(include_builtins=False)
    registry.register("primary", _RegistryTestAdapter)
    registry.register("fallback", _RegistryTestAdapter)

    adapter = registry.get_broker(
        _registry_config(
            primary="primary",
            fallback=("fallback",),
            broker_configs={"primary": {"fail_connect": True}},
            connection_params={
                "primary": {"host": "primary", "port": 1, "client_id": 1},
                "fallback": {"host": "fallback", "port": 2, "client_id": 2},
            },
        )
    )

    assert isinstance(adapter, _RegistryTestAdapter)
    assert registry.get_active_broker_name() == "fallback"
    assert _RegistryTestAdapter.instances[0].is_connected() is False
    assert adapter.connect_calls == [("fallback", 2, 2)]


def test_get_broker_raises_when_no_configured_broker_can_connect() -> None:
    """Registry reports all attempted brokers when every option fails."""

    _reset_test_adapter()
    registry = BrokerRegistry.isolated(include_builtins=False)
    registry.register("primary", _RegistryTestAdapter)

    with pytest.raises(NoBrokerAvailableError) as exc_info:
        registry.get_broker(
            _registry_config(
                broker_configs={"primary": {"fail_connect": True}},
                connection_params={
                    "primary": {"host": "primary", "port": 1, "client_id": 1}
                },
            )
        )

    assert exc_info.value.attempted_brokers == ("primary",)
    assert "primary" in str(exc_info.value)


def test_switch_broker_disconnects_previous_and_connects_new() -> None:
    """Runtime switching replaces the active adapter without restarting."""

    _reset_test_adapter()
    registry = BrokerRegistry.isolated(include_builtins=False)
    registry.register("primary", _RegistryTestAdapter)
    registry.register("secondary", _RegistryTestAdapter)

    first = registry.get_broker(_registry_config(primary="primary"))
    second = registry.switch_broker(
        "secondary",
        connection_params={"host": "secondary", "port": 3, "client_id": 3},
    )

    assert isinstance(first, _RegistryTestAdapter)
    assert first.disconnected is True
    assert second is registry.get_active_broker()
    assert registry.get_active_broker_name() == "secondary"
    assert isinstance(second, _RegistryTestAdapter)
    assert second.connect_calls == [("secondary", 3, 3)]


def test_broker_health_caches_account_probe() -> None:
    """Health checks avoid repeated account probes until forced."""

    adapter = _RegistryTestAdapter()
    adapter.connect()
    health = BrokerHealth(adapter, cache_ttl_seconds=30)

    assert health.check() is True
    assert health.check() is True
    assert adapter.account_calls == 1
    assert health.check(force=True) is True
    assert adapter.account_calls == 2
    assert health.get_status()["healthy"] is True


def test_broker_health_reports_disconnected_adapter() -> None:
    """Disconnected adapters are unhealthy without account probing."""

    adapter = _RegistryTestAdapter()
    health = BrokerHealth(adapter)

    assert health.check() is False
    assert adapter.account_calls == 0
    assert health.get_status()["message"] == "broker adapter is disconnected"
