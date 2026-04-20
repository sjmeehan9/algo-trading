"""Generation tracking domain models for supporting and core model training."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _isoformat(dt: datetime) -> str:
    """Serialize datetime to ISO-8601 with timezone awareness."""

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def _from_isoformat(value: str) -> datetime:
    """Parse ISO-8601 datetime and normalize to timezone-aware values."""

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


@dataclass(slots=True)
class TrainingMetrics:
    """Training metrics collected from a model training run."""

    final_reward: float
    mean_reward: float
    std_reward: float
    episodes_completed: int
    timesteps_trained: int
    training_time_seconds: float
    loss_history: list[float] | None = None
    custom_metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a JSON-serializable dictionary."""

        return {
            "final_reward": self.final_reward,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "episodes_completed": self.episodes_completed,
            "timesteps_trained": self.timesteps_trained,
            "training_time_seconds": self.training_time_seconds,
            "loss_history": self.loss_history,
            "custom_metrics": self.custom_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingMetrics:
        """Create training metrics from a dictionary payload."""

        return cls(
            final_reward=float(data["final_reward"]),
            mean_reward=float(data["mean_reward"]),
            std_reward=float(data["std_reward"]),
            episodes_completed=int(data["episodes_completed"]),
            timesteps_trained=int(data["timesteps_trained"]),
            training_time_seconds=float(data["training_time_seconds"]),
            loss_history=(
                [float(value) for value in data["loss_history"]]
                if data.get("loss_history") is not None
                else None
            ),
            custom_metrics=(
                {
                    str(key): float(value)
                    for key, value in data["custom_metrics"].items()
                }
                if data.get("custom_metrics") is not None
                else None
            ),
        )


@dataclass(slots=True)
class EvaluationMetrics:
    """Evaluation metrics collected from backtesting/evaluation runs."""

    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    total_return: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    num_trades: int | None = None
    custom_metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation metrics to a JSON-serializable dictionary."""

        return {
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "num_trades": self.num_trades,
            "custom_metrics": self.custom_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationMetrics:
        """Create evaluation metrics from a dictionary payload."""

        return cls(
            sharpe_ratio=(
                float(data["sharpe_ratio"])
                if data.get("sharpe_ratio") is not None
                else None
            ),
            max_drawdown=(
                float(data["max_drawdown"])
                if data.get("max_drawdown") is not None
                else None
            ),
            total_return=(
                float(data["total_return"])
                if data.get("total_return") is not None
                else None
            ),
            win_rate=(
                float(data["win_rate"]) if data.get("win_rate") is not None else None
            ),
            profit_factor=(
                float(data["profit_factor"])
                if data.get("profit_factor") is not None
                else None
            ),
            num_trades=(
                int(data["num_trades"]) if data.get("num_trades") is not None else None
            ),
            custom_metrics=(
                {
                    str(key): float(value)
                    for key, value in data["custom_metrics"].items()
                }
                if data.get("custom_metrics") is not None
                else None
            ),
        )


@dataclass(slots=True)
class Generation:
    """Represents one model training generation."""

    generation_id: str
    model_id: str
    generation_number: int
    parent_generation_id: str | None
    created_at: datetime
    status: str
    hyperparameters: dict[str, Any]
    training_metrics: TrainingMetrics | None = None
    evaluation_metrics: EvaluationMetrics | None = None
    model_path: str | None = None
    notes: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this generation into a dictionary for persistence."""

        return {
            "generation_id": self.generation_id,
            "model_id": self.model_id,
            "generation_number": self.generation_number,
            "parent_generation_id": self.parent_generation_id,
            "created_at": _isoformat(self.created_at),
            "status": self.status,
            "hyperparameters": self.hyperparameters,
            "training_metrics": (
                self.training_metrics.to_dict()
                if self.training_metrics is not None
                else None
            ),
            "evaluation_metrics": (
                self.evaluation_metrics.to_dict()
                if self.evaluation_metrics is not None
                else None
            ),
            "model_path": self.model_path,
            "notes": self.notes,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Generation:
        """Deserialize a generation record from dictionary payload."""

        return cls(
            generation_id=str(data["generation_id"]),
            model_id=str(data["model_id"]),
            generation_number=int(data["generation_number"]),
            parent_generation_id=data.get("parent_generation_id"),
            created_at=_from_isoformat(str(data["created_at"])),
            status=str(data["status"]),
            hyperparameters=dict(data.get("hyperparameters") or {}),
            training_metrics=(
                TrainingMetrics.from_dict(data["training_metrics"])
                if data.get("training_metrics") is not None
                else None
            ),
            evaluation_metrics=(
                EvaluationMetrics.from_dict(data["evaluation_metrics"])
                if data.get("evaluation_metrics") is not None
                else None
            ),
            model_path=data.get("model_path"),
            notes=data.get("notes"),
            tags=list(data.get("tags") or []),
        )

    def metric_value(self, metric: str) -> float | None:
        """Get a metric value by name across training/evaluation/custom metric namespaces."""

        if self.training_metrics is not None:
            if hasattr(self.training_metrics, metric):
                value = getattr(self.training_metrics, metric)
                if value is not None:
                    return float(value)
            if (
                self.training_metrics.custom_metrics is not None
                and metric in self.training_metrics.custom_metrics
            ):
                return float(self.training_metrics.custom_metrics[metric])

        if self.evaluation_metrics is not None:
            if hasattr(self.evaluation_metrics, metric):
                value = getattr(self.evaluation_metrics, metric)
                if value is not None:
                    return float(value)
            if (
                self.evaluation_metrics.custom_metrics is not None
                and metric in self.evaluation_metrics.custom_metrics
            ):
                return float(self.evaluation_metrics.custom_metrics[metric])

        return None

    def is_better_than(self, other: Generation, metric: str) -> bool:
        """Compare this generation against another generation using one metric."""

        mine = self.metric_value(metric)
        theirs = other.metric_value(metric)
        if mine is None or theirs is None:
            return False
        return mine > theirs


__all__ = [
    "TrainingMetrics",
    "EvaluationMetrics",
    "Generation",
]
