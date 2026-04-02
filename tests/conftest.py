"""Shared pytest fixtures for algo-trading tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.fixtures.mock_data import generate_ohlcv_data


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom CLI options for the test suite."""

    parser.addoption(
        "--ib-confirm",
        action="store_true",
        default=False,
        help="Run tests marked 'requires_ib' with interactive TWS confirmation.",
    )
    parser.addoption(
        "--news-api-confirm",
        action="store_true",
        default=False,
        help=(
            "Run tests marked 'requires_news_api' with interactive provider "
            "confirmation."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip ``requires_ib`` tests unless ``--ib-confirm`` is passed."""

    if not config.getoption("--ib-confirm"):
        skip_ib = pytest.mark.skip(
            reason="IB tests require --ib-confirm flag and a running TWS/Gateway."
        )
        for item in items:
            if "requires_ib" in item.keywords:
                item.add_marker(skip_ib)

    if not config.getoption("--news-api-confirm"):
        skip_news = pytest.mark.skip(
            reason=(
                "News API tests require --news-api-confirm flag and configured "
                "provider credentials."
            )
        )
        for item in items:
            if "requires_news_api" in item.keywords:
                item.add_marker(skip_news)


class MockBrokerConnection:
    """Lightweight broker connection stub for tests."""

    def __init__(self, market_data: pd.DataFrame) -> None:
        self._market_data = market_data
        self.is_connected = False
        self.request_log: list[dict[str, Any]] = []

    def connect(self, host: str, port: int, client_id: int) -> bool:
        """Simulate a broker connection handshake."""

        self.is_connected = True
        self.request_log.append(
            {"action": "connect", "host": host, "port": port, "client_id": client_id}
        )
        return self.is_connected

    def disconnect(self) -> None:
        """Simulate closing a broker connection."""

        self.is_connected = False
        self.request_log.append({"action": "disconnect"})

    def request_historical_data(
        self, symbol: str, limit: int | None = None
    ) -> pd.DataFrame:
        """Return a slice of mock data for the requested symbol."""

        self.request_log.append(
            {"action": "historical", "symbol": symbol, "limit": limit}
        )
        if limit is None:
            return self._market_data.copy()
        return self._market_data.head(limit).copy()

    def place_order(self, symbol: str, action: str, quantity: int) -> dict[str, Any]:
        """Simulate placing an order and return a confirmation payload."""

        self.request_log.append(
            {
                "action": "order",
                "symbol": symbol,
                "order_action": action,
                "qty": quantity,
            }
        )
        return {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "status": "filled",
        }


@pytest.fixture(scope="session")
def mock_pipeline_config() -> dict[str, Any]:
    """Return a valid pipeline configuration dictionary."""

    return {
        "pipeline": {
            "filename": "pipeline_0001",
            "description": "Basic RL model for AMD NASDAQ stock",
            "pipeline_type": "rl",
            "client_id": {"historical": 0, "live": 1, "trading": 2},
            "timezone": "US/Eastern",
            "contract_info": {
                "symbol": "AMD",
                "secType": "STK",
                "exchange": "SMART",
                "currency": "USD",
                "primaryExchange": "NASDAQ",
            },
            "trading_config": {
                "order_type": "MKT",
                "price_key": "close",
                "balance_multiplier": 0.9,
                "stop_take": {
                    "enabled": True,
                    "takeover_mode": False,
                    "stop_loss_limit": -0.15,
                    "take_profit_rolling": 0.30,
                    "take_profit_floor": 0.15,
                    "profit_key": "trade_change",
                    "position_key": "current_position",
                },
            },
            "historical_data_config": {
                "columns": [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "wap",
                    "count",
                ],
                "barSizeSetting": "5 secs",
                "whatToShow": "TRADES",
            },
            "live_data_config": {
                "columns": [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "wap",
                    "count",
                ],
                "barSizeSetting": 5,
                "whatToShow": "TRADES",
                "runtime": 60,
                "fakerColumnTypes": {"volume": "int", "wap": "float"},
            },
            "env_config": {"env_name": "trading_env"},
            "state_data_config": {
                "columns": {
                    "date": [True, False],
                    "open": [False, True],
                    "high": [False, True],
                    "low": [False, True],
                    "close": [False, True],
                    "volume": [False, True],
                    "count": [False, True],
                },
                "file_trim": 0.038,
                "past_events": 60,
                "scaler": "MinMaxScaler",
            },
            "strategy": {
                "strategy_name": "profit_metrics",
                "strategy_wrapper_filename": "strategy_logic",
                "strategy_wrapper_path": "/custom_functions/",
            },
            "model": {
                "model_type": "ppo",
                "model_policy": "MultiInputPolicy",
                "model_reward": "profit_seeker",
                "reward_wrapper_filename": "reward",
                "reward_wrapper_path": "/custom_functions/",
                "file_extension": ".zip",
                "replay_buffer_extension": "_replay_buffer.pkl",
                "model_config": {
                    "learning_rate": 0.0001,
                    "n_steps": 4096,
                    "batch_size": 32,
                    "n_epochs": 20,
                    "gamma": 0.99,
                    "gae_lambda": 0.9,
                    "clip_range": 0.1,
                    "clip_range_vf": None,
                    "ent_coef": 0.1,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "target_kl": None,
                    "seed": 42,
                    "verbose": 1,
                    "normalize_advantage": True,
                    "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
                },
            },
        }
    }


@pytest.fixture(scope="session")
def mock_config() -> dict[str, Any]:
    """Return a valid runtime configuration dictionary."""

    return {
        "ip_address": "127.0.0.1",
        "port": 7497,
        "account_number": "DU1234567",
        "pipeline": "pipeline_0002",
        "log_path": "logs/",
        "data_path": "data/",
        "task_selection": "task4",
        "data_mode": "historical",
        "date_list": ["2024-10-21", "2024-10-22"],
        "input_model": "ppo_x",
        "save_to_file": "ppo_x",
        "training_date_list": ["2024-10-21", "2024-10-22"],
        "stream_data": "fake",
        "trading_model": "ppo_x",
        "backtest_model": "ppo_x",
        "backtest_date_list": ["2024-10-21", "2024-10-22", "2024-11-22"],
    }


@pytest.fixture(scope="session")
def mock_market_data() -> pd.DataFrame:
    """Generate realistic mock market data for tests."""

    return generate_ohlcv_data(
        num_bars=1000,
        frequency="5s",
        start_date=datetime(2024, 1, 2, 9, 30),
        base_price=150.0,
    )


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test file operations."""

    return tmp_path


@pytest.fixture()
def mock_ib_connection(mock_market_data: pd.DataFrame) -> MockBrokerConnection:
    """Provide a mock broker connection with seeded market data."""

    return MockBrokerConnection(mock_market_data)
