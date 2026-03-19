"""Unit tests for BrokerDataSource and conversion helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from algotrading.src.broker import (
    AccountInfo,
    BarData,
    BrokerAdapter,
    ContractSpec,
    InstrumentType,
    OrderSpec,
    OrderStatus,
    PositionInfo,
)
from algotrading.src.data_pipeline.sources.broker_source import (
    BrokerDataSource,
    bar_to_record,
    contract_spec_from_config,
)
from algotrading.src.data_pipeline.sources.exceptions import (
    DataSourceConnectionError,
)
from algotrading.src.data_pipeline.types import DataFrequency


class _MockBrokerAdapter(BrokerAdapter):
    """Minimal adapter test double for BrokerDataSource behavior."""

    def __init__(self) -> None:
        self.connected = False
        self.connect_calls: list[tuple[str, int, int]] = []
        self.disconnect_calls = 0
        self.request_calls: list[dict[str, object]] = []
        self.subscriptions: dict[int, object] = {}
        self.unsubscribed: list[int] = []
        self._next_sub_id = 1

    def connect(self, host: str, port: int, client_id: int) -> None:
        self.connect_calls.append((host, port, client_id))
        self.connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    @property
    def account_id(self) -> str:
        return "DU_TEST"

    def request_historical_data(
        self,
        contract: ContractSpec,
        end_datetime: datetime,
        duration: str,
        bar_size: str,
        data_type: str = "TRADES",
    ) -> list[BarData]:
        self.request_calls.append(
            {
                "contract": contract,
                "end_datetime": end_datetime,
                "duration": duration,
                "bar_size": bar_size,
                "data_type": data_type,
            }
        )
        return [
            BarData(
                timestamp=datetime(2026, 2, 24, 14, 0, 0, tzinfo=UTC),
                open=100.0,
                high=101.0,
                low=99.5,
                close=100.5,
                volume=1000,
                vwap=100.25,
                trade_count=20,
            ),
            BarData(
                timestamp=datetime(2026, 2, 24, 14, 0, 5, tzinfo=UTC),
                open=100.5,
                high=101.2,
                low=100.1,
                close=100.8,
                volume=1200,
                vwap=100.6,
                trade_count=24,
            ),
        ]

    def subscribe_realtime_data(
        self,
        contract: ContractSpec,
        bar_size: int,
        data_type: str,
        callback,
    ) -> int:
        sub_id = self._next_sub_id
        self._next_sub_id += 1
        self.subscriptions[sub_id] = callback
        callback(
            BarData(
                timestamp=datetime(2026, 2, 24, 14, 1, 0, tzinfo=UTC),
                open=101.0,
                high=101.4,
                low=100.9,
                close=101.2,
                volume=900,
                vwap=101.1,
                trade_count=17,
            )
        )
        return sub_id

    def unsubscribe_realtime_data(self, subscription_id: int) -> None:
        self.unsubscribed.append(subscription_id)
        self.subscriptions.pop(subscription_id, None)

    def place_order(self, contract: ContractSpec, order: OrderSpec) -> str:
        del contract, order
        return "1"

    def cancel_order(self, order_id: str) -> None:
        del order_id

    def get_order_status(self, order_id: str) -> OrderStatus:
        del order_id
        return OrderStatus(
            order_id="1",
            status="SUBMITTED",
            filled_quantity=0.0,
            remaining_quantity=1.0,
            average_fill_price=None,
            last_update=datetime(2026, 2, 24, 14, 0, tzinfo=UTC),
        )

    def register_order_callback(self, callback) -> None:
        del callback

    def get_account_info(self) -> AccountInfo:
        return AccountInfo(
            account_id="DU_TEST",
            cash_balance=10000.0,
            buying_power=40000.0,
            currency="USD",
        )

    def get_positions(self) -> list[PositionInfo]:
        return []

    def subscribe_account_updates(self, callback) -> None:
        del callback

    def subscribe_position_updates(self, callback) -> None:
        del callback

    def get_next_order_id(self) -> str:
        return "2"


def test_bar_to_record_maps_broker_fields() -> None:
    """bar_to_record maps BarData values into DataRecord payload."""

    bar = BarData(
        timestamp=datetime(2026, 2, 24, 14, 0, 0, tzinfo=UTC),
        open=100.0,
        high=101.0,
        low=99.5,
        close=100.5,
        volume=1_000,
        vwap=100.2,
        trade_count=30,
    )

    record = bar_to_record(
        bar=bar,
        symbol="AMD",
        source_id="broker_Mock",
        frequency=DataFrequency.SECOND_5,
    )

    assert record.symbol == "AMD"
    assert record.payload["open"] == 100.0
    assert record.payload["vwap"] == 100.2
    assert record.payload["trade_count"] == 30
    assert record.frequency == DataFrequency.SECOND_5


def test_contract_spec_from_config_supports_pipeline_contract_fields() -> None:
    """contract_spec_from_config accepts existing pipeline naming conventions."""

    contract = contract_spec_from_config(
        {
            "symbol": "AMD",
            "secType": "STK",
            "exchange": "SMART",
            "currency": "USD",
            "primaryExchange": "NASDAQ",
        }
    )

    assert contract.symbol == "AMD"
    assert contract.instrument_type == InstrumentType.STOCK
    assert contract.primary_exchange == "NASDAQ"


def test_connect_requires_connection_details_when_adapter_disconnected() -> None:
    """connect raises when adapter is disconnected and no connection params exist."""

    adapter = _MockBrokerAdapter()
    source = BrokerDataSource(adapter=adapter)

    with pytest.raises(DataSourceConnectionError):
        source.connect()


def test_connect_uses_adapter_with_configured_params() -> None:
    """connect forwards host, port, client_id to the adapter."""

    adapter = _MockBrokerAdapter()
    source = BrokerDataSource(
        adapter=adapter,
        host="127.0.0.1",
        port=7497,
        client_id=11,
    )

    source.connect()

    assert adapter.connect_calls == [("127.0.0.1", 7497, 11)]
    assert source.is_connected


def test_fetch_batch_calls_adapter_and_returns_batch() -> None:
    """fetch_batch delegates to request_historical_data and converts bars."""

    adapter = _MockBrokerAdapter()
    adapter.connected = True
    source = BrokerDataSource(adapter=adapter)

    batch = source.fetch_batch(
        symbol="AMD",
        start=datetime(2026, 2, 24, 14, 0, 0, tzinfo=UTC),
        end=datetime(2026, 2, 24, 14, 0, 10, tzinfo=UTC),
    )

    assert len(batch.records) == 2
    assert batch.symbol == "AMD"
    assert adapter.request_calls[0]["duration"] == "10 S"
    assert batch.records[0].payload["close"] == 100.5


def test_fetch_stream_subscribes_yields_and_unsubscribes() -> None:
    """fetch_stream yields converted records and unsubscribes on close."""

    adapter = _MockBrokerAdapter()
    adapter.connected = True
    source = BrokerDataSource(adapter=adapter)

    stream = source.fetch_stream(symbol="AMD")
    record = next(stream)

    assert record.symbol == "AMD"
    assert record.payload["close"] == 101.2

    stream.close()
    assert adapter.unsubscribed == [1]
