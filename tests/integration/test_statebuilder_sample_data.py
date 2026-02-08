"""Integration test for StateBuilder with sample data fixtures."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pandas as pd

from tests.fixtures.data_validation import load_sample_data


def _ensure_package(name: str, path: Path) -> None:
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules.setdefault(name, package)


def _load_state_builder() -> type:
    repo_root = Path(__file__).resolve().parents[2]
    app_root = repo_root / "app"
    algo_root = app_root / "algotrading"
    src_root = algo_root / "src"

    _ensure_package("app", app_root)
    _ensure_package("app.algotrading", algo_root)
    _ensure_package("app.algotrading.src", src_root)
    _ensure_package("app.algotrading.src.data_sourcing", src_root / "data_sourcing")
    _ensure_package("app.algotrading.src.data_processing", src_root / "data_processing")
    _ensure_package("app.algotrading.src.trading", src_root / "trading")

    trading_module = ModuleType("app.algotrading.src.trading.trading")

    class Trading:  # noqa: D401 - minimal stub for import safety
        def __init__(self, config: dict, pipeline: dict) -> None:
            self.payload = None

    trading_module.Trading = Trading
    sys.modules.setdefault("app.algotrading.src.trading.trading", trading_module)

    from app.algotrading.src.data_sourcing.state_builder import StateBuilder

    return StateBuilder


@dataclass
class _CustomLogicStub:
    """Minimal custom logic implementation for StateBuilder tests."""

    def initialise_variables(self) -> dict[str, float]:
        return {"current_position": 0.0, "trade_change": 0.0}

    def step(
        self,
        action: int,
        state_df: pd.DataFrame,
        custom_variable_dict: dict[str, float],
        terminated: bool,
    ) -> dict[str, float]:
        return custom_variable_dict


def _write_sample_data(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def test_statebuilder_loads_sample_data(tmp_path: Path) -> None:
    """Ensure StateBuilder can load sample fixture data."""

    StateBuilder = _load_state_builder()
    df = load_sample_data("sample_normal_day")
    first_date = pd.to_datetime(df["date"].iloc[0])

    data_root = tmp_path / "data"
    pipeline_name = "test_pipeline"
    pipeline_data_dir = data_root / "saved_data" / pipeline_name
    filename = f"AMD_NASDAQ_{first_date:%Y%m%d}.csv"
    _write_sample_data(pipeline_data_dir / filename, df)

    config = {
        "data_path": f"{data_root}/",
        "task_selection": "task2",
        "data_mode": "historical",
        "training_date_list": [],
        "backtest_date_list": [],
    }

    pipeline = {
        "pipeline": {
            "filename": pipeline_name,
            "contract_info": {
                "symbol": "AMD",
                "primaryExchange": "NASDAQ",
            },
            "state_data_config": {
                "columns": {
                    "date": [True, False],
                    "open": [False, True],
                    "high": [False, True],
                    "low": [False, True],
                    "close": [False, True],
                    "volume": [False, True],
                    "wap": [False, True],
                    "count": [False, True],
                },
                "file_trim": 0.01,
                "past_events": 30,
                "scaler": "MinMaxScaler",
            },
        }
    }

    state_builder = StateBuilder(config, pipeline, _CustomLogicStub())
    state_builder.read_data(evaluate=False)
    state_builder.initialise_state()

    assert not state_builder.final_dataframe.empty
    assert (
        state_builder.state["close"].size
        == pipeline["pipeline"]["state_data_config"]["past_events"]
    )
