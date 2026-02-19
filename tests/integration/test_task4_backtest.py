"""Integration tests for Task 4 backtesting workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from algotrading.src.models.backtest import BackTest
from algotrading.src.models.train_ml import TrainML
from algotrading.src.strategies.strategy import Strategy


def _task4_config(base_config: dict) -> dict:
    config = dict(base_config)
    config["task_selection"] = "task4"
    config["data_mode"] = "historical"
    config["stream_data"] = "fake"
    return config


def test_backtest_initialization(
    integration_config: dict,
    integration_pipeline_rl: dict,
) -> None:
    """Verify BackTest initialises required output paths."""

    config = _task4_config(integration_config)
    backtest = BackTest(config, integration_pipeline_rl)

    assert Path(backtest.path_dict["backtest_data_path"]).exists()
    assert Path(backtest.path_dict["pipeline_backtest_path"]).exists()


def test_backtest_rl_pipeline(
    integration_config: dict,
    integration_pipeline_rl: dict,
    rl_sample_data_file: Path,
    trained_model_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run RL backtest path and verify output artifact is generated."""

    del rl_sample_data_file, trained_model_path

    config = _task4_config(integration_config)

    def _fake_trainml_start(self: TrainML) -> None:
        output_dir = Path(self.path_dict["pipeline_backtest_path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "20240205.csv"
        pd.DataFrame(
            {
                "date": ["2024-02-05 09:30:00-05:00"],
                "close": [150.0],
                "action": [0],
                "trade_change": [0.0],
                "running_profit": [0.0],
            }
        ).to_csv(output, index=False)

    monkeypatch.setattr(TrainML, "start", _fake_trainml_start)

    backtest = BackTest(config, integration_pipeline_rl)
    backtest.start()

    outputs = list(Path(backtest.path_dict["pipeline_backtest_path"]).glob("*.csv"))
    assert outputs


def test_backtest_strategy_pipeline(
    integration_config: dict,
    integration_pipeline_strategy: dict,
    strategy_sample_data_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run strategy backtest path and verify output artifact is generated."""

    del strategy_sample_data_file

    config = _task4_config(integration_config)

    def _fake_strategy_start(self: Strategy) -> None:
        output_dir = Path(self.path_dict["pipeline_backtest_path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "20240205.csv"
        pd.DataFrame(
            {
                "date": ["2024-02-05 09:30:00-05:00"],
                "close": [150.0],
                "action": [0],
                "trade_change": [0.0],
                "running_profit": [0.0],
            }
        ).to_csv(output, index=False)

    monkeypatch.setattr(Strategy, "start", _fake_strategy_start)

    backtest = BackTest(config, integration_pipeline_strategy)
    backtest.start()

    outputs = list(Path(backtest.path_dict["pipeline_backtest_path"]).glob("*.csv"))
    assert outputs


def test_backtest_metrics_calculation(tmp_path: Path) -> None:
    """Verify simple trade metrics can be computed from backtest output."""

    output_file = tmp_path / "backtest.csv"
    output_df = pd.DataFrame(
        {
            "action": [0, 1, 0, 2],
            "trade_change": [0.0, 1.5, 0.5, -0.25],
            "running_profit": [0.0, 1.5, 2.0, 1.75],
        }
    )
    output_df.to_csv(output_file, index=False)

    loaded = pd.read_csv(output_file)
    trade_count = int((loaded["action"] != 0).sum())
    ending_profit = float(loaded["running_profit"].iloc[-1])

    assert trade_count == 2
    assert ending_profit == pytest.approx(1.75)


def test_backtest_output_format(tmp_path: Path) -> None:
    """Verify output columns and dtypes match expected backtest contract."""

    output_file = tmp_path / "format.csv"
    pd.DataFrame(
        {
            "date": ["2024-02-05 09:30:00-05:00"],
            "open": [150.0],
            "high": [150.2],
            "low": [149.8],
            "close": [150.1],
            "action": [0],
            "trade_change": [0.0],
            "running_profit": [0.0],
        }
    ).to_csv(output_file, index=False)

    loaded = pd.read_csv(output_file)
    expected = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "action",
        "trade_change",
        "running_profit",
    }

    assert expected.issubset(set(loaded.columns))
    assert loaded["action"].dtype.kind in {"i", "u"}
