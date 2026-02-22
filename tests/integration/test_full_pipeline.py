"""Full pipeline integration tests through adapter and trainer abstractions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from algotrading.src.broker import (
    BarData,
    ContractSpec,
    InstrumentType,
    InteractiveBrokersAdapter,
)
from algotrading.src.data_sourcing.save_historical import PastData
from algotrading.src.data_sourcing.stream_faker import StreamFaker
from algotrading.src.models.backtest import BackTest
from algotrading.src.models.train_ml import TrainML
from algotrading.src.models.train_rl import TrainRL
from algotrading.src.models.trainer_factory import create_rl_trainer
from algotrading.src.trainers import (
    SB3Algorithm,
    StableBaselines3Trainer,
    TrainingConfig,
)
from gymnasium import Env, spaces

from tests.mocks import MockBrokerAdapter, MockRLTrainer


class _TinyEnv(Env):
    """Minimal deterministic environment for trainer output comparison."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "price": spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(4,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = spaces.Discrete(3)
        self._step = 0

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        del options
        self._step = 0
        return {"price": np.zeros(4, dtype=np.float32)}, {}

    def step(self, action: int):  # type: ignore[override]
        self._step += 1
        obs = {"price": np.full(4, self._step * 0.1, dtype=np.float32)}
        reward = 1.0 - abs(float(action - 1)) * 0.1
        terminated = self._step >= 12
        truncated = False
        info = {"step": self._step}
        return obs, reward, terminated, truncated, info


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
        open=close - 0.3,
        high=close + 0.4,
        low=close - 0.5,
        close=close,
        volume=1_000,
        vwap=close,
        trade_count=10,
    )


def test_task1_with_mock_adapter(
    integration_config: dict,
    integration_pipeline_rl: dict,
) -> None:
    """Task 1 should produce historical CSV output through MockBrokerAdapter."""

    config = dict(integration_config)
    config["task_selection"] = "task1"
    config["date_list"] = [datetime(2024, 2, 5)]

    adapter = MockBrokerAdapter()
    adapter.set_historical_data(
        [
            _bar_at(datetime(2024, 2, 5, 9, 30, 0, tzinfo=UTC), 100.0),
            _bar_at(datetime(2024, 2, 5, 9, 30, 5, tzinfo=UTC), 100.2),
            _bar_at(datetime(2024, 2, 5, 9, 30, 10, tzinfo=UTC), 100.4),
        ]
    )

    app = PastData(config, integration_pipeline_rl, adapter=adapter)
    app.SLEEP_DURATION = 0
    app.LOAD_DURATION = 0
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

    app.connect("127.0.0.1", 7497, 301)
    app.run()

    output_file = (
        Path(app.folder_name)
        / f"{app.contract_spec.symbol}_{app.contract_spec.primary_exchange}_20240205.csv"
    )
    assert output_file.exists()
    output_df = pd.read_csv(output_file)
    assert not output_df.empty


def test_task2_with_mock_trainer(
    integration_config: dict,
    integration_pipeline_rl: dict,
    rl_sample_data_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 2 should complete training orchestration with MockRLTrainer."""

    del rl_sample_data_file
    config = dict(integration_config)
    config["task_selection"] = "task2"

    trainer = MockRLTrainer()

    def _fake_env_factory(self: TrainRL, env_name: str) -> object:
        del env_name
        return object()

    monkeypatch.setattr(TrainRL, "env_factory", _fake_env_factory)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    train_ml = TrainML(config, integration_pipeline_rl, trainer=trainer)
    train_ml.start()

    assert Path(train_ml.path_dict["model_filepath"]).exists()
    assert trainer.train_calls == 1


def test_task3_fake_streaming_with_mock(
    integration_config: dict,
    integration_pipeline_rl: dict,
    rl_sample_data_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 3 fake streaming should process fixture rows deterministically."""

    del rl_sample_data_file
    config = dict(integration_config)
    config["task_selection"] = "task3"
    config["stream_data"] = "fake"
    config["data_mode"] = "live"

    sent: list[dict] = []

    class _QueueStub:
        def __init__(self, cfg: dict, pipeline: dict) -> None:
            del cfg, pipeline

        def put(self, row: dict) -> None:
            sent.append(row)

    monkeypatch.setattr(
        "algotrading.src.data_sourcing.stream_faker.StreamQueue", _QueueStub
    )
    monkeypatch.setattr(
        StreamFaker,
        "process_time",
        lambda self, value: int(pd.Timestamp(value).timestamp()),
    )

    faker = StreamFaker(config, integration_pipeline_rl)
    faker.run()
    assert sent


def test_task4_with_mock_trainer(
    integration_config: dict,
    integration_pipeline_rl: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 4 should produce evaluation output via injected mock trainer path."""

    config = dict(integration_config)
    config["task_selection"] = "task4"
    config["data_mode"] = "historical"
    config["stream_data"] = "fake"

    trainer = MockRLTrainer()
    model_dir = (
        Path(config["data_path"])
        / "models"
        / integration_pipeline_rl["pipeline"]["filename"]
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"{config['backtest_model']}.zip").write_bytes(b"mock")

    def _fake_trainml_start(self: TrainML) -> None:
        assert self.trainer is trainer
        output_dir = Path(self.path_dict["pipeline_backtest_path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "date": ["2024-02-05 09:30:00-05:00"],
                "close": [150.0],
                "action": [0],
                "trade_change": [0.0],
                "running_profit": [0.0],
            }
        ).to_csv(output_dir / "20240205.csv", index=False)

    monkeypatch.setattr(TrainML, "start", _fake_trainml_start)

    backtest = BackTest(config, integration_pipeline_rl, trainer=trainer)
    backtest.start()
    outputs = list(Path(backtest.path_dict["pipeline_backtest_path"]).glob("*.csv"))
    assert outputs


@pytest.mark.requires_ib
def test_task1_output_comparison(
    integration_config: dict,
    integration_pipeline_rl: dict,
    confirm_ib_gateway: dict,
) -> None:
    """Task 1 adapter path should produce data equivalent to direct adapter request."""

    config = dict(integration_config)
    config["task_selection"] = "task1"
    config["date_list"] = [datetime(2026, 2, 13)]

    adapter = InteractiveBrokersAdapter()
    conn = confirm_ib_gateway

    app = PastData(config, integration_pipeline_rl, adapter=adapter)
    try:
        app.connect(conn["host"], conn["port"], conn["client_id"] + 40)

        direct_bars = adapter.request_historical_data(
            contract=_amd_contract(),
            end_datetime=datetime(2026, 2, 13, 10, 0, tzinfo=UTC),
            duration="1800 S",
            bar_size="10 secs",
            data_type="TRADES",
        )

        task_bars = app.adapter.request_historical_data(
            contract=app.contract_spec,
            end_datetime=datetime(2026, 2, 13, 10, 0, tzinfo=UTC),
            duration="1800 S",
            bar_size="10 secs",
            data_type="TRADES",
        )

        assert direct_bars
        assert task_bars
        assert len(task_bars) == len(direct_bars)
        assert task_bars[-1].close == pytest.approx(direct_bars[-1].close, rel=1e-9)
    finally:
        app.disconnect()


@pytest.mark.slow
def test_task2_output_comparison() -> None:
    """Trainer-factory output should be equivalent to direct SB3 trainer behavior."""

    pipeline = {
        "pipeline": {
            "model": {
                "model_type": "ppo",
                "model_policy": "MultiInputPolicy",
            }
        }
    }

    env = _TinyEnv()
    config = TrainingConfig(
        total_timesteps=96,
        n_steps=32,
        batch_size=32,
        custom_params={"verbose": 0},
    )

    via_factory = create_rl_trainer(pipeline)
    via_factory.create_model(env, config)
    result_factory = via_factory.train(config)

    direct = StableBaselines3Trainer(
        algorithm=SB3Algorithm.PPO,
        policy="MultiInputPolicy",
    )
    direct.create_model(_TinyEnv(), config)
    result_direct = direct.train(config)

    assert via_factory.model_type == direct.model_type
    assert result_factory.timesteps_trained == result_direct.timesteps_trained == 96
