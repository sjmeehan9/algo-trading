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
    integration_config: dict, integration_pipeline_rl: dict
) -> None:
    """Optional IB smoke test for manual broker verification environments."""

    pytest.skip("Requires local TWS/Gateway setup; run manually when IB is available.")
