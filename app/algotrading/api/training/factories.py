"""Factories that adapt API model configs into training runtime inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import pandas as pd
from algotrading.api.schemas.data_sources import (
    TrainingDataRequest,
    normalize_training_data_request,
)
from algotrading.api.schemas.models import ModelConfigResponse, ModelType
from algotrading.api.services.data_acquisition_service import DataAcquisitionService
from algotrading.api.services.model_service import ModelService
from algotrading.api.workers.training_worker import DatasetFactory, EnvironmentFactory
from algotrading.src.broker.models import InstrumentType
from algotrading.src.data_pipeline.storage import StoredMarketData
from algotrading.src.data_sourcing.state_builder import StateBuilder
from algotrading.src.envs.trading_env import TradingEnv
from algotrading.src.models.registry import ModelState
from algotrading.src.reward_functions.reward import reward_factory


class TrainingFactoryError(ValueError):
    """Raised when API training inputs cannot build a runtime object."""


@dataclass(slots=True)
class BuiltTradingEnvironment:
    """A constructed `TradingEnv` plus the metadata needed to replay it.

    The training path only needs the environment, but the backtesting replay
    engine also needs the normalized, single-pass market frame and the
    observation window so it can map each environment step back to the
    underlying market bar (timestamp and close price) used for fills.
    """

    env: TradingEnv
    market_frame: pd.DataFrame
    observation_window: int
    symbol: str
    request: TrainingDataRequest


_DEFAULT_STATE_COLUMNS: dict[str, tuple[bool, bool]] = {
    "date": (True, False),
    "open": (False, True),
    "high": (False, True),
    "low": (False, True),
    "close": (False, True),
    "volume": (False, True),
    "count": (False, True),
}

_DEFAULT_STOP_TAKE: dict[str, object] = {
    "enabled": False,
    "takeover_mode": False,
    "stop_loss_limit": -0.15,
    "take_profit_rolling": 0.30,
    "take_profit_floor": 0.15,
    "profit_key": "trade_change",
    "position_key": "current_position",
}

_INSTRUMENT_TO_LEGACY_SEC_TYPE: dict[InstrumentType, str] = {
    InstrumentType.STOCK: "STK",
    InstrumentType.OPTION: "OPT",
    InstrumentType.FUTURE: "FUT",
    InstrumentType.CRYPTO: "CRYPTO",
    InstrumentType.INDEX: "IND",
}


def build_environment_factory(
    *,
    data_service: DataAcquisitionService,
    model_service: ModelService,
    project_root: Path | None = None,
) -> EnvironmentFactory:
    """Create the default environment factory used by API RL training jobs.

    The returned factory builds a market-data :class:`TradingEnv` for both core
    RL models (``core_rl``) and supporting RL signal models (``supporting_rl``).
    Supporting RL models train on the same single-symbol market environment but
    do not require a configured reward function or supporting model inputs of
    their own, so the factory supplies safe defaults where the supporting model
    config omits them.
    """

    root = project_root or Path(__file__).resolve().parents[4]

    def _factory(model: ModelConfigResponse, data_config: dict[str, Any]) -> TradingEnv:
        if model.model_type == ModelType.SUPPORTING_RL:
            return build_supporting_rl_environment(
                model=model,
                data_config=data_config,
                data_service=data_service,
                model_service=model_service,
                project_root=root,
            )
        return build_core_rl_environment(
            model=model,
            data_config=data_config,
            data_service=data_service,
            model_service=model_service,
            project_root=root,
        )

    return _factory


def build_dataset_factory(
    *,
    data_service: DataAcquisitionService,
    model_service: ModelService,
    project_root: Path | None = None,
) -> DatasetFactory:
    """Create the default dataset factory used by API ML training jobs."""

    root = project_root or Path(__file__).resolve().parents[4]

    def _factory(
        model: ModelConfigResponse,
        data_config: dict[str, Any],
    ) -> tuple[Any, Any]:
        return build_news_sentiment_dataset(
            model=model,
            data_config=data_config,
            data_service=data_service,
            model_service=model_service,
            project_root=root,
        )

    return _factory


def build_core_rl_environment(
    *,
    model: ModelConfigResponse,
    data_config: Mapping[str, Any],
    data_service: DataAcquisitionService,
    model_service: ModelService,
    project_root: Path | None = None,
) -> TradingEnv:
    """Build a `TradingEnv` for one core RL API training job."""

    return build_trading_environment(
        model=model,
        data_config=data_config,
        data_service=data_service,
        model_service=model_service,
        project_root=project_root,
        single_pass=False,
    ).env


# Default reward used for supporting RL signal models that do not configure a
# reward function of their own. Supporting RL models learn an action policy over
# the same market-data environment as core RL, so the standard profit-seeking
# reward is a safe, fully-implemented default.
_SUPPORTING_RL_DEFAULT_REWARD = "profit_seeker"


def build_supporting_rl_environment(
    *,
    model: ModelConfigResponse,
    data_config: Mapping[str, Any],
    data_service: DataAcquisitionService,
    model_service: ModelService,
    project_root: Path | None = None,
) -> TradingEnv:
    """Build a market-data `TradingEnv` for one supporting RL training job.

    Supporting RL models are signal providers trained on a single-symbol market
    environment. They reuse the core RL environment construction path, but the
    supporting model registry does not persist a ``reward_function`` or a rich
    ``environment_config`` the way core models do. This builder injects safe,
    fully-implemented defaults (a profit-seeking reward and a default
    observation window) when the supporting model omits them, so a supporting RL
    job can train and produce a reloadable SB3 artifact through the default API
    runtime.

    Args:
        model: Supporting RL model whose training_data_config drives the env.
        data_config: Job overrides merged onto the model defaults.
        data_service: Phase 7 acquisition service used to source/load bars.
        model_service: Model service (unused for supporting RL, kept for parity).
        project_root: Optional override for the project data root.

    Returns:
        A constructed market-data :class:`TradingEnv`.

    Raises:
        TrainingFactoryError: If the model is not supporting RL or the data
            window is too short for the configured observation window.
    """

    if model.model_type != ModelType.SUPPORTING_RL:
        raise TrainingFactoryError(
            "Supporting RL environment construction only supports supporting_rl "
            "models"
        )

    resolved = _resolve_supporting_rl_model(model)
    return build_trading_environment(
        model=resolved,
        data_config=data_config,
        data_service=data_service,
        model_service=model_service,
        project_root=project_root,
        single_pass=False,
    ).env


def _resolve_supporting_rl_model(
    model: ModelConfigResponse,
) -> ModelConfigResponse:
    """Return a model copy with core-RL-style env defaults for supporting RL.

    The supporting registry stores the model as ``supporting_rl`` with an empty
    environment config and no reward function. To reuse the shared market-data
    environment builder (which requires a reward function and core RL type), this
    produces a derived ``core_rl`` view with defaults filled in. The derived view
    is only used to construct the training environment; the persisted supporting
    model and its lifecycle state are unchanged.
    """

    environment_config = dict(model.environment_config or {})
    training_config = _mapping(model.training_data_config)
    config_environment = _mapping(training_config.get("environment_config"))
    if config_environment:
        environment_config = {**config_environment, **environment_config}

    reward_function = (
        model.reward_function
        or str(training_config.get("reward_function") or "").strip()
        or _SUPPORTING_RL_DEFAULT_REWARD
    )

    return model.model_copy(
        update={
            "model_type": ModelType.CORE_RL,
            "reward_function": reward_function,
            "environment_config": environment_config,
            "supporting_model_ids": [],
        }
    )


def build_trading_environment(
    *,
    model: ModelConfigResponse,
    data_config: Mapping[str, Any] | None = None,
    data_service: DataAcquisitionService,
    model_service: ModelService,
    project_root: Path | None = None,
    single_pass: bool = False,
    request: TrainingDataRequest | None = None,
) -> BuiltTradingEnvironment:
    """Build a `TradingEnv` from sourced market data for training or replay.

    This is the shared environment-construction seam used by both Component
    7.4 training and Component 7.5 backtesting. Training repeats the market
    frame enough times to satisfy the requested timestep budget. Backtesting
    requests ``single_pass=True`` so the environment walks each sourced bar
    exactly once, which is what a deterministic historical replay requires.

    Args:
        model: Core RL model whose saved data/environment config drives the env.
        data_config: Job/backtest overrides merged onto the model defaults.
            Ignored when an explicit ``request`` is supplied.
        data_service: Phase 7 acquisition service used to source/load bars.
        model_service: Model service used to validate supporting-model readiness.
        project_root: Optional override for the project data root.
        single_pass: When true, build a single-episode environment that visits
            each sourced bar once (backtest replay). When false, repeat the
            frame to cover ``total_timesteps`` (training).
        request: Optional pre-built canonical data request. Backtesting passes
            the already-merged request (with overridden dates/symbols) so the
            environment uses the exact same contract as data acquisition rather
            than re-normalizing a serialized request.

    Returns:
        The constructed environment plus the normalized market frame,
        observation window, resolved symbol, and canonical data request.

    Raises:
        TrainingFactoryError: If the model is not core RL, the data window is
            too short, the reward function is unsupported, or supporting model
            inputs are required but not ready.
    """

    if model.model_type != ModelType.CORE_RL:
        raise TrainingFactoryError(
            "Core RL environment construction only supports core_rl models"
        )

    _validate_supporting_inputs_ready(model=model, model_service=model_service)

    if request is None:
        request = normalize_training_data_request(
            model.training_data_config,
            data_config or {},
        )
    _validate_single_symbol(request)

    data_service.ensure_market_data(request)
    if request.news_source.enabled:
        data_service.ensure_news_data(request, model=model)

    stored_market = data_service.store.find_market_data(request)
    symbol = request.symbols[0]
    frame = _market_dataframe(stored_market[symbol])

    environment_config = _mapping(model.environment_config)
    observation_window = _observation_window(environment_config)
    if len(frame) <= observation_window:
        raise TrainingFactoryError(
            "Market data window is too short for core RL "
            f"{'replay' if single_pass else 'training'}: "
            f"{len(frame)} rows are available, but observation_window "
            f"requires more than {observation_window} rows"
        )

    base_episode_length = len(frame) - observation_window
    if single_pass:
        episodes = 1
    else:
        requested_timesteps = _positive_int_or_none(
            (data_config or {}).get("total_timesteps")
        )
        episodes = _episode_count(
            requested_timesteps=requested_timesteps,
            episode_length=base_episode_length,
        )
    runtime_frame = _repeat_market_frame(frame, episodes)

    root = project_root or Path(__file__).resolve().parents[4]
    pipeline = _pipeline_config(
        model=model,
        request=request,
        environment_config=environment_config,
        observation_window=observation_window,
    )
    runtime_config = _runtime_config(project_root=root, model=model)
    reward = reward_factory(model.reward_function or "", runtime_config, pipeline)
    if reward is None:
        raise TrainingFactoryError(
            f"Unsupported reward_function '{model.reward_function}' for core RL training"
        )

    state_builder = StateBuilder(runtime_config, pipeline, reward)
    state_builder.load_dataframe(
        runtime_frame,
        episode_length=base_episode_length,
        total_timesteps=base_episode_length * episodes,
    )
    state_builder.initialise_state()
    return BuiltTradingEnvironment(
        env=TradingEnv(state_builder),
        market_frame=frame.reset_index(drop=True).copy(),
        observation_window=observation_window,
        symbol=symbol,
        request=request,
    )


def build_news_sentiment_dataset(
    *,
    model: ModelConfigResponse,
    data_config: Mapping[str, Any],
    data_service: DataAcquisitionService,
    model_service: ModelService,
    project_root: Path | None = None,
) -> tuple[Any, Any]:
    """Build a labeled text dataset for supporting sentiment training."""

    del model_service
    if model.model_type != ModelType.SUPPORTING_ML:
        raise TrainingFactoryError(
            "Dataset construction only supports supporting_ml models"
        )

    dataset_path = _dataset_path_or_none(
        data_config=data_config,
        project_root=project_root,
    )
    if dataset_path is not None:
        frame = pd.read_csv(dataset_path)
        return _dataset_from_labeled_frame(
            frame=frame,
            data_config=data_config,
            source_description=f"Labeled dataset {dataset_path}",
        )

    request = normalize_training_data_request(model.training_data_config, data_config)
    acquisition = data_service.ensure_news_data(request, model=model)
    loaded_news = _load_news_data_for_acquisition(
        data_service=data_service,
        request=request,
        provider_values={report.provider for report in acquisition.reports},
    )
    return _dataset_from_news_records(loaded_news)


def _dataset_from_labeled_frame(
    *,
    frame: pd.DataFrame,
    data_config: Mapping[str, Any],
    source_description: str,
) -> tuple[Any, Any]:
    """Build a sentiment dataset from a labeled dataframe."""

    target_column = _target_column(data_config, frame)
    feature_columns = _feature_columns(data_config, frame, target_column)

    if frame.empty:
        raise TrainingFactoryError(f"{source_description} is empty")
    if not feature_columns:
        raise TrainingFactoryError(
            "Labeled dataset must include at least one feature column"
        )

    features = frame[feature_columns].astype(str).agg(" ".join, axis=1).to_numpy()
    targets = frame[target_column].to_numpy()
    return features, targets


def _load_news_data_for_acquisition(
    *,
    data_service: DataAcquisitionService,
    request: TrainingDataRequest,
    provider_values: set[str],
) -> list[object]:
    """Load canonical news records for the providers used by acquisition."""

    candidate_providers = {
        str(provider).strip().lower()
        for provider in provider_values
        if str(provider).strip()
    }
    if request.news_source.provider.value != "none":
        candidate_providers.add(request.news_source.provider.value)

    records: list[object] = []
    errors: list[str] = []
    for provider in sorted(candidate_providers):
        provider_request = request.model_copy(
            update={
                "news_source": request.news_source.model_copy(
                    update={"provider": provider}
                )
            }
        )
        try:
            loaded = data_service.store.find_news_data(provider_request)
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
            continue
        for stored in loaded.values():
            records.extend(stored.records)

    if records:
        return records

    detail = "; ".join(errors) if errors else "no news acquisition reports returned"
    raise TrainingFactoryError(
        "supporting_ml news-derived training could not load canonical news records: "
        f"{detail}"
    )


def _dataset_from_news_records(records: list[object]) -> tuple[Any, Any]:
    """Build a sentiment dataset from canonical news records with labels."""

    rows: list[dict[str, object]] = []
    missing_labels = 0
    for record in records:
        sentiment_score = getattr(record, "sentiment_score", None)
        if sentiment_score is None:
            missing_labels += 1
            continue
        headline = str(getattr(record, "headline", "")).strip()
        body_value = getattr(record, "body", None)
        body = str(body_value).strip() if body_value is not None else ""
        text = " ".join(part for part in (headline, body) if part)
        if not text:
            continue
        rows.append({"text": text, "sentiment_score": float(sentiment_score)})

    if not rows:
        raise TrainingFactoryError(
            "supporting_ml news-derived training requires provider news records "
            "with sentiment_score labels; supply labeled_dataset_path, use a "
            "provider that returns sentiment labels, or activate/load a pretrained "
            f"supporting model instead. Loaded {len(records)} records, "
            f"{missing_labels} without labels."
        )

    frame = pd.DataFrame(rows)
    return _dataset_from_labeled_frame(
        frame=frame,
        data_config={},
        source_description="News-derived sentiment dataset",
    )


def _validate_supporting_inputs_ready(
    *,
    model: ModelConfigResponse,
    model_service: ModelService,
) -> None:
    if not model.supporting_model_ids:
        return

    for model_id in model.supporting_model_ids:
        entry = model_service.supporting_registry.get(model_id)
        if entry is None:
            raise TrainingFactoryError(
                f"supporting_model_id '{model_id}' does not exist"
            )
        if entry.state != ModelState.READY:
            raise TrainingFactoryError(
                f"supporting_model_id '{model_id}' is not READY "
                f"(current state: {entry.state.value})"
            )

    raise TrainingFactoryError(
        "Core RL API training with supporting_model_ids requires cached signal "
        "features, which are not available in this training factory yet"
    )


def _validate_single_symbol(request: TrainingDataRequest) -> None:
    if len(request.symbols) == 1:
        return
    raise TrainingFactoryError(
        "Core RL API training currently supports exactly one market symbol per job"
    )


def _market_dataframe(stored: StoredMarketData) -> pd.DataFrame:
    frame = stored.batch.to_dataframe()
    if frame.empty:
        raise TrainingFactoryError(
            f"No market rows were loaded for symbol '{stored.symbol}'"
        )

    frame = frame.rename(
        columns={
            "timestamp": "date",
            "vwap": "wap",
            "trade_count": "count",
        }
    )
    if "wap" not in frame.columns:
        frame["wap"] = frame["close"]
    if "count" not in frame.columns:
        frame["count"] = 0

    required = ["date", "open", "high", "low", "close", "volume", "wap", "count"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise TrainingFactoryError(
            "Canonical market data is missing required environment columns: "
            + ", ".join(missing)
        )

    frame = frame[required].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume", "wap", "count"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["date", *numeric_columns])
    if frame.empty:
        raise TrainingFactoryError(
            f"Market rows for symbol '{stored.symbol}' are empty after normalization"
        )

    return frame.sort_values("date").reset_index(drop=True)


def _repeat_market_frame(frame: pd.DataFrame, episodes: int) -> pd.DataFrame:
    if episodes <= 1:
        return frame.reset_index(drop=True).copy()
    return pd.concat([frame.copy() for _ in range(episodes)], ignore_index=True)


def _episode_count(*, requested_timesteps: int | None, episode_length: int) -> int:
    if episode_length < 1:
        raise TrainingFactoryError("episode_length must be positive")
    if requested_timesteps is None:
        return 1
    return max(1, ceil(requested_timesteps / episode_length))


def _pipeline_config(
    *,
    model: ModelConfigResponse,
    request: TrainingDataRequest,
    environment_config: Mapping[str, Any],
    observation_window: int,
) -> dict[str, object]:
    state_config = _state_data_config(environment_config, observation_window)
    return {
        "pipeline": {
            "filename": f"api_{model.model_id}",
            "description": model.description or model.name,
            "pipeline_type": "rl",
            "client_id": {"historical": 0, "live": 1, "trading": 2},
            "timezone": "UTC",
            "contract_info": _contract_info(request),
            "trading_config": _trading_config(environment_config),
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
                "barSizeSetting": request.market_source.bar_size,
                "whatToShow": request.market_source.broker_data_type,
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
                "barSizeSetting": request.market_source.bar_size,
                "whatToShow": request.market_source.broker_data_type,
                "runtime": 60,
                "fakerColumnTypes": {"volume": "int", "wap": "float"},
            },
            "env_config": {"env_name": "trading_env"},
            "state_data_config": state_config,
            "strategy": {
                "strategy_name": "profit_metrics",
                "strategy_wrapper_filename": "strategy_logic",
                "strategy_wrapper_path": "/custom_functions/",
            },
            "model": {
                "model_type": model.algorithm.lower(),
                "model_policy": str(
                    model.hyperparameters.get("model_policy")
                    or model.hyperparameters.get("policy")
                    or "MultiInputPolicy"
                ),
                "model_reward": model.reward_function,
                "reward_wrapper_filename": "reward",
                "reward_wrapper_path": "/custom_functions/",
                "file_extension": ".zip",
                "replay_buffer_extension": "_replay_buffer.pkl",
                "model_config": dict(model.hyperparameters),
            },
        }
    }


def _contract_info(request: TrainingDataRequest) -> dict[str, object]:
    instrument_type = request.market_source.instrument_type
    return {
        "symbol": request.symbols[0],
        "secType": _INSTRUMENT_TO_LEGACY_SEC_TYPE.get(
            instrument_type,
            instrument_type.value,
        ),
        "exchange": request.market_source.exchange,
        "currency": request.market_source.currency,
        "primaryExchange": request.market_source.primary_exchange
        or request.market_source.exchange,
    }


def _trading_config(environment_config: Mapping[str, Any]) -> dict[str, object]:
    raw_trading = _mapping(environment_config.get("trading_config"))
    raw_stop_take = {
        **_mapping(raw_trading.get("stop_take")),
        **_mapping(environment_config.get("stop_take")),
    }
    stop_take = {**_DEFAULT_STOP_TAKE, **raw_stop_take}
    return {
        "order_type": str(raw_trading.get("order_type") or "MKT"),
        "price_key": str(raw_trading.get("price_key") or "close"),
        "balance_multiplier": float(raw_trading.get("balance_multiplier") or 0.9),
        "stop_take": stop_take,
    }


def _state_data_config(
    environment_config: Mapping[str, Any],
    observation_window: int,
) -> dict[str, object]:
    raw_state = _mapping(environment_config.get("state_data_config"))
    raw_columns = (
        environment_config.get("state_columns")
        or environment_config.get("columns")
        or raw_state.get("columns")
    )
    columns = _state_columns(raw_columns)
    scaler = str(
        raw_state.get("scaler") or environment_config.get("scaler") or "MinMaxScaler"
    )
    return {
        "columns": columns,
        "file_trim": float(raw_state.get("file_trim") or 0.0),
        "past_events": observation_window,
        "scaler": scaler,
    }


def _state_columns(raw_columns: object) -> dict[str, tuple[bool, bool]]:
    if raw_columns is None:
        return dict(_DEFAULT_STATE_COLUMNS)
    if isinstance(raw_columns, list):
        columns = dict(_DEFAULT_STATE_COLUMNS)
        selected = {
            str(column).strip() for column in raw_columns if str(column).strip()
        }
        return {
            key: value
            for key, value in columns.items()
            if key in selected or key == "date"
        }
    if not isinstance(raw_columns, Mapping):
        raise TrainingFactoryError(
            "environment state columns must be a mapping or list"
        )

    columns: dict[str, tuple[bool, bool]] = {}
    for raw_name, raw_value in raw_columns.items():
        name = str(raw_name).strip()
        if not name:
            continue
        if isinstance(raw_value, (list, tuple)) and len(raw_value) >= 2:
            columns[name] = (bool(raw_value[0]), bool(raw_value[1]))
        elif isinstance(raw_value, Mapping):
            columns[name] = (
                bool(raw_value.get("is_key", False)),
                bool(raw_value.get("scale", raw_value.get("should_scale", True))),
            )
        else:
            columns[name] = (False, bool(raw_value))

    if "date" not in columns:
        columns = {"date": (True, False), **columns}
    return columns or dict(_DEFAULT_STATE_COLUMNS)


def _observation_window(environment_config: Mapping[str, Any]) -> int:
    raw_state = _mapping(environment_config.get("state_data_config"))
    candidates = (
        environment_config.get("observation_window"),
        environment_config.get("past_events"),
        environment_config.get("window_size"),
        raw_state.get("past_events"),
        raw_state.get("window_size"),
        60,
    )
    for candidate in candidates:
        value = _positive_int_or_none(candidate)
        if value is not None:
            return value
    raise TrainingFactoryError("observation_window must be a positive integer")


def _runtime_config(
    *,
    project_root: Path,
    model: ModelConfigResponse,
) -> dict[str, object]:
    return {
        "task_selection": "task2",
        "data_mode": "historical",
        "data_path": str(project_root / "data"),
        "training_date_list": [],
        "backtest_date_list": [],
        "input_model": "",
        "save_to_file": model.model_id,
    }


def _dataset_path_or_none(
    *,
    data_config: Mapping[str, Any],
    project_root: Path | None,
) -> Path | None:
    raw_path = (
        data_config.get("labeled_dataset_path")
        or data_config.get("dataset_path")
        or data_config.get("training_dataset_path")
    )
    if raw_path is None:
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute() and project_root is not None:
        path = project_root / path
    if not path.exists() or not path.is_file():
        raise TrainingFactoryError(f"Labeled dataset file does not exist: {path}")
    return path


def _target_column(data_config: Mapping[str, Any], frame: pd.DataFrame) -> str:
    configured = data_config.get("target_column") or data_config.get("label_column")
    candidates = [str(configured)] if configured is not None else []
    candidates.extend(["label", "target", "sentiment_score"])
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise TrainingFactoryError(
        "Labeled dataset must include target_column, label, target, or sentiment_score"
    )


def _feature_columns(
    data_config: Mapping[str, Any],
    frame: pd.DataFrame,
    target_column: str,
) -> list[str]:
    configured = data_config.get("feature_columns")
    if isinstance(configured, list):
        columns = [str(column) for column in configured if str(column) in frame.columns]
        if columns:
            return columns
    preferred = ["headline", "body", "text"]
    columns = [column for column in preferred if column in frame.columns]
    if columns:
        return columns
    return [column for column in frame.columns if column != target_column]


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _positive_int_or_none(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
