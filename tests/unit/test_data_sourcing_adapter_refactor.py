"""Unit tests for adapter-based data sourcing classes.

These tests verify that `PastData` and `LiveData` delegate broker operations to
`BrokerAdapter` while preserving their public workflow behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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
from algotrading.src.data_sourcing.live_streaming import LiveData
from algotrading.src.data_sourcing.save_historical import PastData


class _MockBrokerAdapter(BrokerAdapter):
    def __init__(self) -> None:
        self.connected = False
        self.connect_calls: list[tuple[str, int, int]] = []
        self.disconnect_calls = 0
        self.historical_calls: list[tuple[ContractSpec, datetime, str, str, str]] = []
        self.subscribe_calls: list[tuple[ContractSpec, int, str]] = []
        self.unsubscribe_calls: list[int] = []
        self.callback = None
        self.subscription_id = 123
        self.bars_to_return: list[BarData] = []

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
        return "DU123"

    def request_historical_data(
        self,
        contract: ContractSpec,
        end_datetime: datetime,
        duration: str,
        bar_size: str,
        data_type: str = "TRADES",
    ) -> list[BarData]:
        self.historical_calls.append(
            (contract, end_datetime, duration, bar_size, data_type)
        )
        return list(self.bars_to_return)

    def subscribe_realtime_data(
        self,
        contract: ContractSpec,
        bar_size: int,
        data_type: str,
        callback,
    ) -> int:
        self.subscribe_calls.append((contract, bar_size, data_type))
        self.callback = callback
        return self.subscription_id

    def unsubscribe_realtime_data(self, subscription_id: int) -> None:
        self.unsubscribe_calls.append(subscription_id)

    def place_order(self, contract: ContractSpec, order: OrderSpec) -> str:
        del contract, order
        return "1"

    def cancel_order(self, order_id: str) -> None:
        del order_id

    def get_order_status(self, order_id: str) -> OrderStatus:
        del order_id
        raise NotImplementedError

    def register_order_callback(self, callback) -> None:
        del callback

    def get_account_info(self) -> AccountInfo:
        raise NotImplementedError

    def get_positions(self) -> list[PositionInfo]:
        return []

    def subscribe_account_updates(self, callback) -> None:
        del callback

    def subscribe_position_updates(self, callback) -> None:
        del callback

    def get_next_order_id(self) -> str:
        return "1"


def _sample_bar(ts: datetime) -> BarData:
    return BarData(
        timestamp=ts,
        open=100.0,
        high=101.0,
        low=99.5,
        close=100.5,
        volume=1000,
        vwap=100.2,
        trade_count=10,
    )


def test_pastdata_delegates_connect_request_and_disconnect(
    mock_config: dict, mock_pipeline_config: dict, tmp_path: Path
) -> None:
    """Verify PastData delegates connect, request, and disconnect to the adapter."""
    config = dict(mock_config)
    config["date_list"] = [datetime(2024, 2, 5)]
    config["data_path"] = str(tmp_path)
    pipeline = dict(mock_pipeline_config)

    adapter = _MockBrokerAdapter()
    app = PastData(config, pipeline, adapter=adapter)
    app.step_size = {
        "date_hour_max": 16,
        "date_minute_max": 0,
        "date_second_max": 0,
        "loops_required": 1,
        "increment_size": 5,
        "durationString": "10 S",
        "durationNum": 10,
        "barSize": 5,
    }
    app._finalize_date_file = lambda _: None
    adapter.bars_to_return = [_sample_bar(datetime(2024, 2, 5, 9, 30, tzinfo=UTC))]
    app.date_list = []

    app.connect("127.0.0.1", 7497, 7)
    app.run()
    app.stop()

    assert adapter.connect_calls == [("127.0.0.1", 7497, 7)]
    assert len(adapter.historical_calls) == 1
    assert adapter.disconnect_calls == 1


def test_livedata_delegates_historical_and_realtime(
    mock_config: dict, mock_pipeline_config: dict, monkeypatch
) -> None:
    """Verify LiveData delegates warmup fetch and realtime subscription to the adapter."""
    config = dict(mock_config)
    pipeline = dict(mock_pipeline_config)

    adapter = _MockBrokerAdapter()
    app = LiveData(config, pipeline, adapter=adapter)
    app.connectDates = lambda _: True

    queued: list[dict | list[dict]] = []
    monkeypatch.setattr(app.queue, "put", lambda payload: queued.append(payload))

    adapter.bars_to_return = [
        _sample_bar(datetime(2024, 2, 5, 9, 30, 0, tzinfo=UTC)),
        _sample_bar(datetime(2024, 2, 5, 9, 30, 5, tzinfo=UTC)),
    ]

    app.connect("127.0.0.1", 7497, 8)
    app.run()

    assert len(adapter.historical_calls) == 1
    assert len(adapter.subscribe_calls) == 1

    assert adapter.callback is not None
    adapter.callback(_sample_bar(datetime(2024, 2, 5, 9, 30, 10, tzinfo=UTC)))
    assert queued

    app.stop()
    assert adapter.unsubscribe_calls == [adapter.subscription_id]
    assert adapter.disconnect_calls == 1
