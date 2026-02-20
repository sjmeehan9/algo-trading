"""Integration tests for Task 1 historical data collection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from algotrading.src.data_sourcing.save_historical import PastData


def _task1_config(base_config: dict) -> dict:
    config = dict(base_config)
    config["task_selection"] = "task1"
    config["date_list"] = [datetime(2024, 2, 5)]
    return config


def test_historical_data_request_format(
    integration_config: dict, integration_pipeline_rl: dict
) -> None:
    """Verify historical request payloads are built correctly for IB API calls."""

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
    app.reqHistoricalData = lambda *args: captured.append(args)

    app.req_it = app.INIT_REQUEST_ID
    app.sendRequests(datetime(2024, 2, 5))

    assert len(captured) == 1
    request = captured[0]
    assert request[0] == app.INIT_REQUEST_ID
    assert (
        request[1].symbol
        == integration_pipeline_rl["pipeline"]["contract_info"]["symbol"]
    )
    assert request[3] == app.step_size["durationString"]
    assert (
        request[4]
        == integration_pipeline_rl["pipeline"]["historical_data_config"][
            "barSizeSetting"
        ]
    )


def test_historical_data_callback_processing(
    integration_config: dict, integration_pipeline_rl: dict
) -> None:
    """Verify callback processing emits expected CSV output."""

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
    app.cancelHistoricalData = lambda req_id: None
    app.sendRequests = lambda date: None
    app.date_list = []
    app.req_it = app.INIT_REQUEST_ID

    bars = [
        SimpleNamespace(
            date="20240205 09:30:00",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000,
            wap=100.2,
            barCount=10,
        ),
        SimpleNamespace(
            date="20240205 09:30:05",
            open=100.5,
            high=101.2,
            low=100.2,
            close=100.7,
            volume=1200,
            wap=100.8,
            barCount=12,
        ),
        SimpleNamespace(
            date="20240205 09:30:10",
            open=100.7,
            high=101.3,
            low=100.4,
            close=101.0,
            volume=1100,
            wap=100.9,
            barCount=11,
        ),
    ]

    for bar in bars:
        app.historicalData(app.req_it, bar)

    app.historicalDataEnd(app.req_it, "", "20240205 16:00:00")

    output_file = (
        Path(app.folder_name)
        / f"{app.contract.symbol}_{app.contract.primaryExchange}_20240205.csv"
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

    app.cancelHistoricalData = lambda req_id: None
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

    app.historicalDataEnd(app.INIT_REQUEST_ID, "", "20240205 16:00:00")

    assert sent_dates == [next_date]


@pytest.mark.requires_ib
def test_historical_data_ib_connection_smoke(
    integration_config: dict,
    integration_pipeline_rl: dict,
    confirm_ib_gateway: dict,
) -> None:
    """Smoke-test a real IB historical data request against a running TWS/Gateway.

    This test:
      1. Connects to TWS via the IB API on the paper-trading port.
      2. Requests a small window of historical 10-sec bars for AMD.
      3. Asserts that bars arrive with valid OHLCV data.
      4. Disconnects cleanly.

    The test bypasses ``PastData.historicalDataEnd`` row-count validation
    because the exact number of bars returned by IB varies with market
    conditions.  Instead it asserts directly on the bars received via the
    ``historicalData`` callback.

    Run with::

        pytest tests/integration/test_task1_historical.py -k ib_connection_smoke \
            --ib-confirm -s

    Pattern for other IB integration tests:
      - Depend on ``confirm_ib_gateway`` fixture for interactive pre-flight.
      - Use ``run_ib_client_in_thread`` to run the EClient loop off-thread.
      - Intercept EWrapper callbacks with ``threading.Event`` for coordination.
      - Always disconnect in a ``finally`` block.
    """

    import threading

    from tests.integration.conftest import run_ib_client_in_thread

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

    # -- Signals for inter-thread coordination --------------------------------
    connected_event = threading.Event()
    data_end_event = threading.Event()
    received_bars: list[dict] = []
    request_errors: list[str] = []

    # IB system status codes (reqId == -1) that are informational, not errors.
    _IB_INFO_CODES = {2104, 2106, 2107, 2108, 2119, 2158}

    # Intercept callbacks to coordinate with the test thread
    _original_next_valid_id = app.nextValidId
    _original_historical_data = app.historicalData
    _original_error = app.error

    def _on_next_valid_id(order_id: int) -> None:
        _original_next_valid_id(order_id)
        connected_event.set()

    def _on_historical_data(req_id: int, bar: object) -> None:
        _original_historical_data(req_id, bar)
        received_bars.append({"date": bar.date, "close": bar.close})

    def _on_historical_data_end(req_id: int, start: str, end: str) -> None:
        # Bypass PastData.historicalDataEnd entirely — its row-count validation
        # is too strict for a smoke test where exact bar counts can vary.
        # We only need to know data delivery finished so we can assert on bars.
        app.cancelHistoricalData(req_id)
        data_end_event.set()

    def _on_error(
        req_id: int, code: int, msg: str, advanced: str = ""
    ) -> None:
        _original_error(req_id, code, msg, advanced)
        # Only capture request-specific errors, not system status messages.
        if req_id != -1 or code not in _IB_INFO_CODES:
            if code >= 2000 and code not in _IB_INFO_CODES:
                request_errors.append(f"code={code} msg={msg}")

    app.nextValidId = _on_next_valid_id
    app.historicalData = _on_historical_data
    app.historicalDataEnd = _on_historical_data_end
    app.error = _on_error

    # -- Connect and run ------------------------------------------------------
    conn = confirm_ib_gateway
    app.connect(conn["host"], conn["port"], conn["client_id"])
    api_thread = run_ib_client_in_thread(app)

    try:
        assert connected_event.wait(timeout=10), (
            "Timed out waiting for TWS connection — is TWS running on "
            f"{conn['host']}:{conn['port']}?"
        )

        # Wait for the historical data cycle to complete
        assert data_end_event.wait(timeout=30), (
            "Timed out waiting for historical data — check TWS market data "
            "subscriptions and that the requested date has available data."
        )

        # -- Assertions -------------------------------------------------------
        assert len(received_bars) > 0, "No bars received from IB."

        # Verify bars contain expected keys and sane values
        sample = received_bars[0]
        assert "date" in sample, "Bar missing 'date' field."
        assert "close" in sample, "Bar missing 'close' field."
        assert sample["close"] > 0, f"Non-positive close price: {sample['close']}"

        # No request-specific errors should have occurred
        assert len(request_errors) == 0, (
            f"IB API errors during smoke test: {request_errors}"
        )

    finally:
        # -- Teardown: disconnect cleanly -------------------------------------
        if app.isConnected():
            app.disconnect()
        api_thread.join(timeout=5)
