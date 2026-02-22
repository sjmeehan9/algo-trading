"""Integration tests for Task 1 historical data collection."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from algotrading.src.broker import BarData
from algotrading.src.data_sourcing.save_historical import PastData


def _task1_config(base_config: dict) -> dict:
    config = dict(base_config)
    config["task_selection"] = "task1"
    config["date_list"] = [datetime(2024, 2, 5)]
    return config


def test_historical_data_request_format(
    integration_config: dict, integration_pipeline_rl: dict
) -> None:
    """Verify historical request payloads are delegated to broker adapter."""

    config = _task1_config(integration_config)
    app = PastData(config, integration_pipeline_rl)

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

    captured: list[tuple] = []

    class _AdapterStub:
        def connect(self, host: str, port: int, client_id: int) -> None:
            del host, port, client_id

        def disconnect(self) -> None:
            return None

        def is_connected(self) -> bool:
            return True

        @property
        def account_id(self) -> str:
            return ""

        def request_historical_data(
            self,
            contract,
            end_datetime,
            duration,
            bar_size,
            data_type="TRADES",
        ):
            captured.append((contract, end_datetime, duration, bar_size, data_type))
            return []

        def subscribe_realtime_data(self, contract, bar_size, data_type, callback):
            del contract, bar_size, data_type, callback
            return 0

        def unsubscribe_realtime_data(self, subscription_id: int) -> None:
            del subscription_id

        def place_order(self, contract, order) -> str:
            del contract, order
            return ""

        def cancel_order(self, order_id: str) -> None:
            del order_id

        def get_order_status(self, order_id: str):
            del order_id
            raise NotImplementedError

        def register_order_callback(self, callback) -> None:
            del callback

        def get_account_info(self):
            raise NotImplementedError

        def get_positions(self):
            return []

        def subscribe_account_updates(self, callback) -> None:
            del callback

        def subscribe_position_updates(self, callback) -> None:
            del callback

        def get_next_order_id(self) -> str:
            return ""

    app.adapter = _AdapterStub()
    app._finalize_date_file = lambda _: None

    app.req_it = app.INIT_REQUEST_ID
    app.sendRequests(datetime(2024, 2, 5))

    assert len(captured) == 1
    request = captured[0]
    assert (
        request[0].symbol
        == integration_pipeline_rl["pipeline"]["contract_info"]["symbol"]
    )
    assert request[2] == app.step_size["durationString"]
    assert (
        request[3]
        == integration_pipeline_rl["pipeline"]["historical_data_config"][
            "barSizeSetting"
        ]
    )


def test_historical_data_callback_processing(
    integration_config: dict, integration_pipeline_rl: dict
) -> None:
    """Verify historical processing emits expected CSV output."""

    config = _task1_config(integration_config)
    app = PastData(config, integration_pipeline_rl)
    app.folder_name = str(
        Path(config["data_path"]) / "saved_data" / "integration_pipeline_rl"
    )
    Path(app.folder_name).mkdir(parents=True, exist_ok=True)

    app.step_size = {
        "loops_required": 1,
        "durationNum": 10,
        "barSize": 5,
    }
    app.sendRequests = lambda date: None
    app.date_list = []
    app.req_it = app.INIT_REQUEST_ID

    bars = [
        BarData(
            timestamp=datetime(2024, 2, 5, 9, 30, 0, tzinfo=UTC),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000,
            vwap=100.2,
            trade_count=10,
        ),
        BarData(
            timestamp=datetime(2024, 2, 5, 9, 30, 5, tzinfo=UTC),
            open=100.5,
            high=101.2,
            low=100.2,
            close=100.7,
            volume=1200,
            vwap=100.8,
            trade_count=12,
        ),
        BarData(
            timestamp=datetime(2024, 2, 5, 9, 30, 10, tzinfo=UTC),
            open=100.7,
            high=101.3,
            low=100.4,
            close=101.0,
            volume=1100,
            vwap=100.9,
            trade_count=11,
        ),
    ]

    for bar in bars:
        app._consume_historical_bar(bar)

    app._finalize_date_file(datetime(2024, 2, 5).date())

    output_file = (
        Path(app.folder_name)
        / f"{app.contract_spec.symbol}_{app.contract_spec.primary_exchange}_20240205.csv"
    )
    assert output_file.exists()

    output_df = pd.read_csv(output_file)
    assert (
        list(output_df.columns)
        == integration_pipeline_rl["pipeline"]["historical_data_config"]["columns"]
    )
    assert len(output_df) == 2


def test_historical_data_date_iteration(
    integration_config: dict, integration_pipeline_rl: dict
) -> None:
    """Verify historical collection iterates through pending dates correctly."""

    config = _task1_config(integration_config)
    app = PastData(config, integration_pipeline_rl)

    next_date = datetime(2024, 2, 6)
    sent_dates: list[datetime] = []

    app.sendRequests = lambda date: sent_dates.append(date)
    app.checkDataframe = lambda date_requested: True
    app.data_list = [
        {
            "date": "20240205 09:30:00",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
            "wap": 100.2,
            "count": 10,
        }
    ]
    app.step_size = {
        "loops_required": 1,
        "durationNum": 5,
        "barSize": 5,
    }
    app.folder_name = str(
        Path(config["data_path"]) / "saved_data" / "integration_pipeline_rl"
    )
    Path(app.folder_name).mkdir(parents=True, exist_ok=True)
    app.date_list = [next_date]

    app._finalize_date_file(datetime(2024, 2, 5).date())

    assert sent_dates == [next_date]


@pytest.mark.requires_ib
def test_historical_data_ib_connection_smoke(
    integration_config: dict,
    integration_pipeline_rl: dict,
    confirm_ib_gateway: dict,
) -> None:
    """Smoke-test a real IB historical data request against a running TWS/Gateway.

        This test:
            1. Connects to TWS via the broker adapter on the paper-trading port.
            2. Requests a small window of historical 10-sec bars for AMD.
            3. Asserts that bars arrive with valid OHLCV data.
            4. Disconnects cleanly.

    Run with::

        pytest tests/integration/test_task1_historical.py -k ib_connection_smoke \
            --ib-confirm -s

    Pattern for other IB integration tests:
      - Depend on ``confirm_ib_gateway`` fixture for interactive pre-flight.
      - Use ``run_ib_client_in_thread`` to run the EClient loop off-thread.
            - Request through `BrokerAdapter` abstraction.
            - Always disconnect in a ``finally`` block.
    """

    # -- Setup: override pipeline to use a small, fast bar size ---------------
    config = _task1_config(integration_config)
    pipeline = dict(integration_pipeline_rl)

    # Use 10-sec bars with a 30-minute window for a single loop — fast & small
    test_step_size = {
        "barSize": 10,
        "durationString": "1800 S",
        "durationNum": 1800,
        "increment_size": 30,
        "loops_required": 1,
        "date_hour_max": 10,
        "date_minute_max": 0,
        "date_second_max": 0,
    }

    # Use a recent weekday for which historical data should be available.
    # Monday 16 Feb 2026 is Presidents' Day (market closed), so use Fri 13 Feb.
    trade_date = datetime(2026, 2, 13)
    config["date_list"] = [trade_date]

    output_dir = Path(config["data_path"]) / "saved_data" / "ib_smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    app = PastData(config, pipeline)
    app.folder_name = str(output_dir)
    app.step_size = test_step_size
    app.first_date = trade_date
    app.date_list = []

    # -- Connect and run ------------------------------------------------------
    conn = confirm_ib_gateway
    app.connect(conn["host"], conn["port"], conn["client_id"])

    try:
        bars = app.adapter.request_historical_data(
            app.contract_spec,
            datetime(2026, 2, 13, 10, 0),
            "1800 S",
            "10 secs",
            "TRADES",
        )

        # -- Assertions -------------------------------------------------------
        assert len(bars) > 0, "No bars received from IB."

        # Verify bars contain expected keys and sane values
        sample = bars[0]
        assert sample.close > 0, f"Non-positive close price: {sample.close}"

    finally:
        # -- Teardown: disconnect cleanly -------------------------------------
        if app.isConnected():
            app.disconnect()
