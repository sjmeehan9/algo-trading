"""Integration-specific pytest fixtures for end-to-end task coverage."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from algotrading.src.models.train_ml import TrainML
from algotrading.src.models.train_rl import TrainRL

from tests.fixtures.data_validation import load_sample_data

# ---------------------------------------------------------------------------
# IB Gateway / TWS interactive confirmation fixtures
# ---------------------------------------------------------------------------

_IB_CONFIRMATION_PROMPT = (
    "\n"
    "=" * 60 + "\n"
    "  IB INTEGRATION TEST — PRE-FLIGHT CHECK\n"
    "=" * 60 + "\n"
    "  Please confirm the following before continuing:\n"
    "\n"
    "  1. TWS or IB Gateway is running on 127.0.0.1:7497\n"
    "  2. API connections are enabled (Edit > Global Config > API)\n"
    "  3. You have market data subscriptions for the test symbols\n"
    "  4. The account is in paper-trading mode\n"
    "\n"
    "  Type 'yes' to proceed or 'no' to skip: "
)


@pytest.fixture(scope="session")
def confirm_ib_gateway() -> dict[str, Any]:
    """Prompt the developer once per session to confirm TWS is available.

    Returns:
        Connection parameter dict with ``host``, ``port``, and ``client_id``.

    Raises:
        pytest.skip: If the developer declines or stdin is unavailable.
    """

    try:
        answer = input(_IB_CONFIRMATION_PROMPT).strip().lower()
    except EOFError:
        pytest.skip("Non-interactive environment — cannot confirm TWS availability.")

    if answer != "yes":
        pytest.skip("Developer declined IB confirmation prompt.")

    return {"host": "127.0.0.1", "port": 7497, "client_id": 100}


def run_ib_client_in_thread(app: Any) -> threading.Thread:
    """Start an EClient event-loop in a daemon thread.

    Args:
        app: An EClient/EWrapper instance that has already called ``connect``.

    Returns:
        The running daemon thread (stops when ``app.disconnect()`` is called).
    """

    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    return thread


@pytest.fixture(scope="session")
def temp_output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a session-scoped output directory for integration tests."""

    return tmp_path_factory.mktemp("integration_outputs")


@pytest.fixture(scope="session")
def sample_data_path() -> Path:
    """Return path to integration sample market data fixture file."""

    return (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "data"
        / "sample_normal_day.csv"
    )


def _load_pipeline_fixture(filename: str) -> dict[str, Any]:
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures" / "configs"
    with open(fixtures_dir / filename, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _base_config(temp_output_dir: Path) -> dict[str, Any]:
    test_date = datetime(2024, 2, 5)
    return {
        "ip_address": "127.0.0.1",
        "port": 7497,
        "account_number": "DU1234567",
        "pipeline": "test_pipeline",
        "log_path": str(temp_output_dir / "logs") + "/",
        "data_path": str(temp_output_dir / "data") + "/",
        "task_selection": "task2",
        "data_mode": "historical",
        "date_list": [test_date],
        "input_model": "integration_input",
        "save_to_file": "integration_model",
        "training_date_list": [test_date],
        "stream_data": "fake",
        "trading_model": "integration_model",
        "backtest_model": "integration_model",
        "backtest_date_list": [test_date],
    }


@pytest.fixture()
def integration_config(temp_output_dir: Path) -> dict[str, Any]:
    """Return an isolated runtime config for integration tests."""

    return _base_config(temp_output_dir)


@pytest.fixture()
def integration_pipeline_rl() -> dict[str, Any]:
    """Return RL pipeline config tuned for fast integration tests."""

    pipeline = _load_pipeline_fixture("test_pipeline_rl.json")
    pipeline = deepcopy(pipeline)
    pipeline["pipeline"]["filename"] = "integration_pipeline_rl"
    pipeline["pipeline"]["state_data_config"]["file_trim"] = 0.02
    pipeline["pipeline"]["state_data_config"]["past_events"] = 20
    pipeline["pipeline"]["model"]["model_config"]["n_steps"] = 64
    pipeline["pipeline"]["model"]["model_config"]["batch_size"] = 32
    pipeline["pipeline"]["model"]["model_config"]["verbose"] = 0
    pipeline["pipeline"]["model"]["reward_wrapper_path"] = "/custom_functions/"
    pipeline["pipeline"]["strategy"]["strategy_wrapper_path"] = "/custom_functions/"
    return pipeline


@pytest.fixture()
def integration_pipeline_strategy() -> dict[str, Any]:
    """Return strategy pipeline config tuned for fast integration tests."""

    pipeline = _load_pipeline_fixture("test_pipeline_strategy.json")
    pipeline = deepcopy(pipeline)
    pipeline["pipeline"]["filename"] = "integration_pipeline_strategy"
    pipeline["pipeline"]["state_data_config"]["file_trim"] = 0.02
    pipeline["pipeline"]["state_data_config"]["past_events"] = 20
    pipeline["pipeline"]["strategy"]["strategy_wrapper_path"] = "/custom_functions/"
    pipeline["pipeline"]["model"]["reward_wrapper_path"] = "/custom_functions/"
    return pipeline


def _write_pipeline_sample_data(
    config: dict[str, Any], pipeline: dict[str, Any], rows: int = 260
) -> Path:
    data_root = (
        Path(config["data_path"]) / "saved_data" / pipeline["pipeline"]["filename"]
    )
    data_root.mkdir(parents=True, exist_ok=True)

    sample_df = load_sample_data("sample_normal_day").head(rows).copy()
    date_for_file = pd.to_datetime(sample_df["date"].iloc[0]).to_pydatetime()

    contract = pipeline["pipeline"]["contract_info"]
    filename = (
        f"{contract['symbol']}_{contract['primaryExchange']}_"
        f"{date_for_file.year}{date_for_file.month:02d}{date_for_file.day:02d}.csv"
    )
    file_path = data_root / filename
    sample_df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture()
def rl_sample_data_file(
    integration_config: dict[str, Any], integration_pipeline_rl: dict[str, Any]
) -> Path:
    """Create sample historical CSV for RL integration paths."""

    return _write_pipeline_sample_data(integration_config, integration_pipeline_rl)


@pytest.fixture()
def strategy_sample_data_file(
    integration_config: dict[str, Any], integration_pipeline_strategy: dict[str, Any]
) -> Path:
    """Create sample historical CSV for strategy integration paths."""

    return _write_pipeline_sample_data(
        integration_config, integration_pipeline_strategy
    )


@pytest.fixture()
def trained_model_path(
    integration_config: dict[str, Any],
    integration_pipeline_rl: dict[str, Any],
    rl_sample_data_file: Path,
) -> Path:
    """Run a minimal training flow and return created model path."""

    del rl_sample_data_file

    model_dir = (
        Path(integration_config["data_path"])
        / "models"
        / integration_pipeline_rl["pipeline"]["filename"]
    )
    existing_model = model_dir / f"{integration_config['save_to_file']}.zip"
    if existing_model.exists():
        existing_model.unlink()

    original_train_ppo = TrainRL.train_ppo
    original_env_factory = TrainRL.env_factory

    def _fake_env_factory(self: TrainRL, env_name: str) -> object:
        del env_name
        return object()

    def _fake_train_ppo(self: TrainRL) -> None:
        model_file = Path(self.path_dict["model_filepath"] + ".zip")
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_bytes(b"integration-model")

    TrainRL.env_factory = _fake_env_factory
    TrainRL.train_ppo = _fake_train_ppo
    try:
        trainer = TrainML(integration_config, integration_pipeline_rl)
        trainer.start()
        return Path(trainer.path_dict["model_filepath"] + ".zip")
    finally:
        TrainRL.train_ppo = original_train_ppo
        TrainRL.env_factory = original_env_factory
