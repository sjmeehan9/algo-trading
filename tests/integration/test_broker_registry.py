"""Integration-style tests for config-driven broker registry flows."""

from __future__ import annotations

from datetime import UTC, datetime

from algotrading.src.broker import (
    AccountInfo,
    BarData,
    BrokerAdapter,
    BrokerConfig,
    BrokerConnectionError,
    BrokerHealth,
    BrokerRegistry,
    ContractSpec,
    OrderSpec,
    OrderStatus,
    PositionInfo,
)


class _ConfigDrivenAdapter(BrokerAdapter):
    """Broker adapter used for YAML-driven registry integration checks."""

    instances: list["_ConfigDrivenAdapter"] = []

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
                broker_name="config_adapter",
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
        return "CONFIG-ACCOUNT"

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
            account_id="CONFIG-ACCOUNT",
            cash_balance=5000.0,
            buying_power=10000.0,
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


def _reset_adapter_instances() -> None:
    _ConfigDrivenAdapter.instances.clear()


def test_registry_connects_primary_from_yaml_config(tmp_path) -> None:
    """Registry loads broker config from YAML and connects the primary broker."""

    _reset_adapter_instances()
    config_path = tmp_path / "brokers.yml"
    config_path.write_text(
        """
primary_broker: mock_primary
broker_configs:
  mock_primary:
    mode: paper
connection_params:
  mock_primary:
    host: localhost
    port: 4001
    client_id: 17
""",
        encoding="utf-8",
    )
    registry = BrokerRegistry.isolated(config_path=config_path, include_builtins=False)
    registry.register("mock_primary", _ConfigDrivenAdapter)

    adapter = registry.get_broker()

    assert isinstance(adapter, _ConfigDrivenAdapter)
    assert adapter.config == {"mode": "paper"}
    assert adapter.connect_calls == [("localhost", 4001, 17)]
    assert registry.get_active_broker_name() == "mock_primary"


def test_registry_falls_back_from_yaml_when_primary_fails(tmp_path) -> None:
    """Config-driven fallback connects a secondary broker after primary failure."""

    _reset_adapter_instances()
    config_path = tmp_path / "brokers.yml"
    config_path.write_text(
        """
primary_broker: mock_primary
fallback_brokers:
  - mock_fallback
broker_configs:
  mock_primary:
    fail_connect: true
  mock_fallback:
    mode: paper
connection_params:
  mock_primary:
    host: primary
    port: 4001
    client_id: 17
  mock_fallback:
    host: fallback
    port: 4002
    client_id: 18
""",
        encoding="utf-8",
    )
    registry = BrokerRegistry.isolated(config_path=config_path, include_builtins=False)
    registry.register("mock_primary", _ConfigDrivenAdapter)
    registry.register("mock_fallback", _ConfigDrivenAdapter)

    adapter = registry.get_broker()

    assert isinstance(adapter, _ConfigDrivenAdapter)
    assert registry.get_active_broker_name() == "mock_fallback"
    assert adapter.config == {"mode": "paper"}
    assert adapter.connect_calls == [("fallback", 4002, 18)]


def test_registry_runtime_switch_uses_full_broker_config() -> None:
    """Runtime switching can use a full config object for adapter and connect data."""

    _reset_adapter_instances()
    registry = BrokerRegistry.isolated(include_builtins=False)
    registry.register("mock_primary", _ConfigDrivenAdapter)
    registry.register("mock_secondary", _ConfigDrivenAdapter)
    config = BrokerConfig(
        primary_broker="mock_primary",
        broker_configs={
            "mock_primary": {"mode": "paper"},
            "mock_secondary": {"mode": "live"},
        },
        connection_params={
            "mock_primary": {"host": "primary", "port": 4001, "client_id": 17},
            "mock_secondary": {"host": "secondary", "port": 4002, "client_id": 18},
        },
    )

    first = registry.get_broker(config)
    second = registry.switch_broker("mock_secondary", config=config)

    assert isinstance(first, _ConfigDrivenAdapter)
    assert first.disconnected is True
    assert isinstance(second, _ConfigDrivenAdapter)
    assert second.config == {"mode": "live"}
    assert second.connect_calls == [("secondary", 4002, 18)]
    assert registry.get_active_broker_name() == "mock_secondary"


def test_registry_health_check_caches_yaml_connected_broker(tmp_path) -> None:
    """Connected brokers can be wrapped with cached health checks."""

    _reset_adapter_instances()
    config_path = tmp_path / "brokers.yml"
    config_path.write_text(
        """
primary_broker: mock_primary
connection_params:
  mock_primary:
    host: localhost
    port: 4001
    client_id: 17
""",
        encoding="utf-8",
    )
    registry = BrokerRegistry.isolated(config_path=config_path, include_builtins=False)
    registry.register("mock_primary", _ConfigDrivenAdapter)
    adapter = registry.get_broker()
    health = BrokerHealth(adapter, cache_ttl_seconds=30)

    assert health.check() is True
    assert health.check() is True
    assert isinstance(adapter, _ConfigDrivenAdapter)
    assert adapter.account_calls == 1
