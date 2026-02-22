"""Integration tests for broker adapter behavior (real IB + offline mocks)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest
from algotrading.src.broker import (
    BarData,
    ContractSpec,
    InstrumentType,
    InteractiveBrokersAdapter,
)
from algotrading.src.data_sourcing.live_streaming import LiveData
from algotrading.src.data_sourcing.save_historical import PastData

from tests.mocks import MockBrokerAdapter


def _amd_contract() -> ContractSpec:
    return ContractSpec(
        symbol="AMD",
        instrument_type=InstrumentType.STOCK,
        exchange="SMART",
        currency="USD",
        primary_exchange="NASDAQ",
    )


def _bar_at(ts: datetime, close: float) -> BarData:
    return BarData(
        timestamp=ts,
        open=close - 0.2,
        high=close + 0.3,
        low=close - 0.4,
        close=close,
        volume=1_000,
        vwap=close,
        trade_count=10,
    )


@pytest.mark.requires_ib
def test_ib_adapter_connection(confirm_ib_gateway: dict) -> None:
    """Connect/disconnect succeeds for live IB adapter."""

    adapter = InteractiveBrokersAdapter()
    conn = confirm_ib_gateway
    try:
        adapter.connect(conn["host"], conn["port"], conn["client_id"] + 20)
        assert adapter.is_connected() is True
    finally:
        adapter.disconnect()


@pytest.mark.requires_ib
def test_ib_adapter_historical_data(confirm_ib_gateway: dict) -> None:
    """Historical data request through live IB adapter returns valid bars."""

    adapter = InteractiveBrokersAdapter()
    conn = confirm_ib_gateway
    try:
        adapter.connect(conn["host"], conn["port"], conn["client_id"] + 21)
        bars = adapter.request_historical_data(
            contract=_amd_contract(),
            end_datetime=datetime.now(tz=UTC),
            duration="1800 S",
            bar_size="10 secs",
            data_type="TRADES",
        )
        assert bars
        assert bars[-1].close > 0
    finally:
        adapter.disconnect()


@pytest.mark.requires_ib
def test_ib_adapter_realtime_subscribe(confirm_ib_gateway: dict) -> None:
    """Realtime subscription returns at least one callback or fallback bar."""

    adapter = InteractiveBrokersAdapter()
    conn = confirm_ib_gateway
    event = threading.Event()

    def _on_bar(bar: BarData) -> None:
        del bar
        event.set()

    subscription_id = -1
    try:
        adapter.connect(conn["host"], conn["port"], conn["client_id"] + 22)
        subscription_id = adapter.subscribe_realtime_data(
            contract=_amd_contract(),
            bar_size=5,
            data_type="TRADES",
            callback=_on_bar,
        )

        if not event.wait(timeout=20):
            bars = adapter.request_historical_data(
                contract=_amd_contract(),
                end_datetime=datetime.now(tz=UTC),
                duration="120 S",
                bar_size="1 min",
                data_type="TRADES",
            )
            assert bars
            assert bars[-1].close > 0
    finally:
        if subscription_id != -1:
            adapter.unsubscribe_realtime_data(subscription_id)
        adapter.disconnect()


def test_mock_adapter_historical_data() -> None:
    """Mock adapter returns exactly configured historical bars."""

    adapter = MockBrokerAdapter()
    bars = [
        _bar_at(datetime(2024, 2, 5, 9, 30, tzinfo=UTC), 100.0),
        _bar_at(datetime(2024, 2, 5, 9, 30, 5, tzinfo=UTC), 100.2),
    ]
    adapter.set_historical_data(bars)

    result = adapter.request_historical_data(
        _amd_contract(),
        datetime.now(tz=UTC),
        "10 S",
        "5 secs",
        "TRADES",
    )
    assert len(result) == 2
    assert result[1].close == 100.2


def test_pastdata_with_mock_adapter(
    integration_config: dict,
    integration_pipeline_rl: dict,
) -> None:
    """PastData should consume broker operations via injected mock adapter."""

    config = dict(integration_config)
    config["task_selection"] = "task1"
    config["date_list"] = [datetime(2024, 2, 5)]

    adapter = MockBrokerAdapter()
    adapter.set_historical_data(
        [_bar_at(datetime(2024, 2, 5, 9, 30, 5, tzinfo=UTC), 100.5)]
    )

    app = PastData(config, integration_pipeline_rl, adapter=adapter)
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
    app.date_list = []

    app.connect("127.0.0.1", 7497, 201)
    app.run()

    assert app.isConnected() is True
    assert len(adapter.historical_calls) == 1


def test_livedata_with_mock_adapter(
    integration_config: dict,
    integration_pipeline_rl: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LiveData warmup + realtime flow should work through mock adapter."""

    config = dict(integration_config)
    config["task_selection"] = "task3"

    adapter = MockBrokerAdapter()
    adapter.set_historical_data(
        [
            _bar_at(datetime(2024, 2, 5, 9, 30, 0, tzinfo=UTC), 100.0),
            _bar_at(datetime(2024, 2, 5, 9, 30, 5, tzinfo=UTC), 100.1),
        ]
    )

    class _QueueStub:
        def __init__(self, config: dict, pipeline: dict) -> None:
            del config, pipeline
            self.rows: list[dict | list[dict]] = []

        def put(self, payload: dict | list[dict]) -> None:
            self.rows.append(payload)

    monkeypatch.setattr(
        "algotrading.src.data_sourcing.live_streaming.StreamQueue",
        _QueueStub,
    )

    app = LiveData(config, integration_pipeline_rl, adapter=adapter)
    app.connectDates = lambda _: True

    app.connect("127.0.0.1", 7497, 202)
    app.run()

    adapter.emit_realtime_bar(
        _bar_at(datetime(2024, 2, 5, 9, 30, 10, tzinfo=UTC), 100.2)
    )

    assert app.queue.rows
    app.stop()
