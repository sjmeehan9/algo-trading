"""Integration tests for Task 3 fake streaming and trading flow."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from algotrading.src.data_sourcing.stream_faker import StreamFaker
from algotrading.src.trading.order import OrderManager
from algotrading.src.trading.tools import TradingTools
from algotrading.src.trading.trading_data import TradingStream


def _task3_config(base_config: dict) -> dict:
    config = dict(base_config)
    config["task_selection"] = "task3"
    config["stream_data"] = "fake"
    config["data_mode"] = "live"
    return config


def test_stream_faker_data_flow(
    integration_config: dict,
    integration_pipeline_rl: dict,
    rl_sample_data_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify StreamFaker reads fixture data and queues processed rows."""

    del rl_sample_data_file

    config = _task3_config(integration_config)

    class _QueueStub:
        def __init__(self, cfg: dict, pipeline: dict) -> None:
            del cfg, pipeline
            self.rows: list[dict] = []

        def put(self, row: dict) -> None:
            self.rows.append(row)

    monkeypatch.setattr(
        "algotrading.src.data_sourcing.stream_faker.StreamQueue", _QueueStub
    )
    monkeypatch.setattr(
        StreamFaker,
        "process_time",
        lambda self, value: int(pd.Timestamp(value).timestamp()),
    )

    faker = StreamFaker(config, integration_pipeline_rl)
    captured_rows = faker.queue.rows

    faker.run()

    assert captured_rows
    assert len(captured_rows) == len(faker.final_dataframe)
    first_row = captured_rows[0]
    assert isinstance(first_row["volume"], Decimal)
    assert isinstance(first_row["wap"], Decimal)


@dataclass
class _PayloadStub:
    update_state_data: bool = False
    temp_action_int: int = 0
    previous_pos: str = "NONE"
    last_price: float = 100.0
    active_pos: str = "NONE"
    action_int: int = 0
    action_dict: dict = None

    def __post_init__(self) -> None:
        if self.action_dict is None:
            self.action_dict = {
                "NONE": 0,
                "BUY": 1,
                "SELL": 2,
                0: "NONE",
                1: "BUY",
                2: "SELL",
            }


def test_trading_session_with_fake_data(
    integration_config: dict,
    integration_pipeline_rl: dict,
    rl_sample_data_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run fake streaming mode end-to-end through TradingStream."""

    del rl_sample_data_file

    config = _task3_config(integration_config)

    class _TradingStub:
        def __init__(self, config: dict, pipeline: dict) -> None:
            del config, pipeline
            self.payload = _PayloadStub()
            self.confirm_count = 0

        def confirmTrades(self) -> None:
            self.confirm_count += 1

        def tradingAlgorithm(self, state: dict, state_df: pd.DataFrame) -> None:
            del state, state_df

    monkeypatch.setattr(
        "algotrading.src.data_sourcing.state_builder.Trading", _TradingStub
    )

    def _fake_send_data(self: StreamFaker) -> None:
        self.queue.put(
            {
                "date": 1707143400,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
                "wap": 100.2,
                "count": 10,
            }
        )

    monkeypatch.setattr(StreamFaker, "send_data", _fake_send_data)

    stream = TradingStream(config, integration_pipeline_rl, config["stream_data"])
    stream.start()

    assert stream.mode == "fake"


def test_order_manager_integration(
    integration_config: dict, integration_pipeline_rl: dict
) -> None:
    """Verify OrderManager builds order specs and order objects correctly."""

    manager = OrderManager(integration_config, integration_pipeline_rl)
    state_df = pd.DataFrame({"close": [100.0, 101.0]})

    price = manager.priceAction(state_df)
    spec = manager.calcOrderSpec(
        balance=10_000,
        units=0,
        action="BUY",
        activePos="NONE",
        price=price,
    )
    order, active_pos = manager.buildOrder(spec[0], int(spec[1]))

    assert spec[0] == "BUY"
    assert order.action == "BUY"
    assert active_pos == "BUY_PEND"


def test_stop_take_profit_integration(integration_pipeline_rl: dict) -> None:
    """Verify TradingTools stop/take logic integrates with state values."""

    tools = TradingTools(integration_pipeline_rl)

    neutral_state = {
        "current_position": np.array([0]),
        "trade_change": np.array([0.0]),
    }
    hold_action = tools.stop_take(1, neutral_state)
    assert hold_action == 1

    loss_state = {
        "current_position": np.array([1]),
        "trade_change": np.array([-0.2]),
    }
    close_action = tools.stop_take(1, loss_state)
    assert close_action == 2
