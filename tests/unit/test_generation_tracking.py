"""Unit tests for model generation tracking system."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from algotrading.src.models.tracking import (
    EvaluationMetrics,
    Generation,
    GenerationTracker,
    JsonFileStorage,
    SqliteStorage,
    TrainingMetrics,
)


def _training_metrics(final_reward: float = 1.0) -> TrainingMetrics:
    return TrainingMetrics(
        final_reward=final_reward,
        mean_reward=0.6,
        std_reward=0.2,
        episodes_completed=8,
        timesteps_trained=1500,
        training_time_seconds=18.5,
        loss_history=[1.1, 0.9, 0.5],
        custom_metrics={"stability": 0.82},
    )


def _evaluation_metrics(sharpe_ratio: float = 0.9) -> EvaluationMetrics:
    return EvaluationMetrics(
        sharpe_ratio=sharpe_ratio,
        max_drawdown=0.18,
        total_return=0.34,
        win_rate=0.57,
        profit_factor=1.25,
        num_trades=44,
        custom_metrics={"hit_rate": 0.61},
    )


def test_generation_serialization_round_trip() -> None:
    """Generation and nested metrics round-trip through dictionary serialization."""

    generation = Generation(
        generation_id="gen-001",
        model_id="core-rl",
        generation_number=1,
        parent_generation_id=None,
        created_at=datetime(2026, 3, 22, 12, 0, 0, tzinfo=UTC),
        status="completed",
        hyperparameters={"learning_rate": 0.0003, "gamma": 0.99},
        training_metrics=_training_metrics(),
        evaluation_metrics=_evaluation_metrics(),
        model_path="models/core-rl/gen-001.zip",
        notes="baseline",
        tags=["baseline", "paper"],
    )

    rebuilt = Generation.from_dict(generation.to_dict())

    assert rebuilt == generation
    assert rebuilt.metric_value("final_reward") == 1.0
    assert rebuilt.metric_value("sharpe_ratio") == 0.9
    assert rebuilt.metric_value("hit_rate") == 0.61


def test_generation_tracker_lifecycle_methods(tmp_path) -> None:
    """Tracker lifecycle supports start, complete, evaluate, and fail transitions."""

    storage = JsonFileStorage(str(tmp_path / "tracking"))
    tracker = GenerationTracker(storage)

    generation = tracker.start_generation(
        model_id="core-rl",
        hyperparameters={"learning_rate": 0.0001},
    )
    assert generation.status == "training"
    assert generation.generation_number == 1

    tracker.complete_generation(
        generation.generation_id,
        training_metrics=_training_metrics(final_reward=1.4),
        model_path="models/core-rl/gen-1.zip",
    )

    completed = tracker.get_generation(generation.generation_id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.training_metrics is not None
    assert completed.training_metrics.final_reward == 1.4

    tracker.add_evaluation(
        generation.generation_id, _evaluation_metrics(sharpe_ratio=1.2)
    )

    evaluated = tracker.get_generation(generation.generation_id)
    assert evaluated is not None
    assert evaluated.status == "evaluated"
    assert evaluated.evaluation_metrics is not None
    assert evaluated.evaluation_metrics.sharpe_ratio == 1.2

    failed = tracker.start_generation(
        model_id="core-rl",
        hyperparameters={"learning_rate": 0.0002},
        parent_generation_id=generation.generation_id,
    )
    tracker.fail_generation(failed.generation_id, "nan detected")

    failed_loaded = tracker.get_generation(failed.generation_id)
    assert failed_loaded is not None
    assert failed_loaded.status == "failed"
    assert failed_loaded.notes == "nan detected"


def test_json_file_storage_persistence_and_query(tmp_path) -> None:
    """JSON storage saves, loads, queries, and deletes generation records."""

    storage = JsonFileStorage(str(tmp_path / "json-store"))

    generation = Generation(
        generation_id="gen-json-1",
        model_id="news-sentiment",
        generation_number=1,
        parent_generation_id=None,
        created_at=datetime.now(UTC),
        status="completed",
        hyperparameters={"alpha": 0.5},
        training_metrics=_training_metrics(),
        evaluation_metrics=None,
        model_path=None,
        notes=None,
        tags=["baseline"],
    )
    storage.save(generation)

    loaded = storage.load("gen-json-1")
    assert loaded is not None
    assert loaded.model_id == "news-sentiment"

    queried = storage.query({"model_id": "news-sentiment", "tag": "baseline"})
    assert len(queried) == 1
    assert queried[0].generation_id == "gen-json-1"

    assert storage.delete("gen-json-1") is True
    assert storage.load("gen-json-1") is None


def test_sqlite_storage_round_trip(tmp_path) -> None:
    """SQLite storage supports save/load/query/delete workflow."""

    storage = SqliteStorage(str(tmp_path / "tracking.db"))
    generation = Generation(
        generation_id="gen-sql-1",
        model_id="core-rl",
        generation_number=1,
        parent_generation_id=None,
        created_at=datetime.now(UTC),
        status="training",
        hyperparameters={"batch_size": 32},
        tags=["exp"],
    )

    storage.save(generation)

    loaded = storage.load("gen-sql-1")
    assert loaded is not None
    assert loaded.status == "training"

    queried = storage.query({"model_id": "core-rl", "tag": "exp"})
    assert len(queried) == 1

    assert storage.delete("gen-sql-1") is True
    assert storage.load("gen-sql-1") is None


def test_generation_comparison_and_summary(tmp_path) -> None:
    """Comparison computes per-metric deltas and readable summary output."""

    tracker = GenerationTracker(JsonFileStorage(str(tmp_path / "tracking")))

    gen_a = tracker.start_generation("core-rl", {"lr": 0.0003})
    tracker.complete_generation(gen_a.generation_id, _training_metrics(1.0), "a.zip")
    tracker.add_evaluation(gen_a.generation_id, _evaluation_metrics(sharpe_ratio=0.8))

    gen_b = tracker.start_generation(
        "core-rl",
        {"lr": 0.0002},
        parent_generation_id=gen_a.generation_id,
    )
    tracker.complete_generation(gen_b.generation_id, _training_metrics(1.5), "b.zip")
    tracker.add_evaluation(gen_b.generation_id, _evaluation_metrics(sharpe_ratio=1.1))

    comparison = tracker.compare_generations(gen_a.generation_id, gen_b.generation_id)

    assert comparison.metric_comparisons["final_reward"] == (1.0, 1.5, 0.5)
    sharpe = comparison.metric_comparisons["sharpe_ratio"]
    assert sharpe[0] == 0.8
    assert sharpe[1] == 1.1
    assert sharpe[2] == pytest.approx(0.3)
    assert "Improved:" in comparison.improvement_summary


def test_lineage_tracking_returns_ancestors_and_descendants(tmp_path) -> None:
    """Tracker returns deterministic ancestor and descendant chains."""

    tracker = GenerationTracker(JsonFileStorage(str(tmp_path / "tracking")))

    root = tracker.start_generation("core-rl", {"lr": 0.001})
    child = tracker.start_generation(
        "core-rl",
        {"lr": 0.0005},
        parent_generation_id=root.generation_id,
    )
    grandchild = tracker.start_generation(
        "core-rl",
        {"lr": 0.0003},
        parent_generation_id=child.generation_id,
    )
    tracker.start_generation(
        "core-rl", {"lr": 0.0007}, parent_generation_id=root.generation_id
    )

    lineage = tracker.get_lineage(grandchild.generation_id)
    assert [item.generation_id for item in lineage] == [
        root.generation_id,
        child.generation_id,
        grandchild.generation_id,
    ]

    descendants = tracker.get_descendants(root.generation_id)
    assert len(descendants) == 3
    assert all(item.generation_number > 1 for item in descendants)


def test_best_generation_query_and_prune_policy(tmp_path) -> None:
    """Best-generation query and pruning keep strong and recent runs."""

    tracker = GenerationTracker(JsonFileStorage(str(tmp_path / "tracking")))

    generation_ids: list[str] = []
    for index, reward in enumerate([1.0, 1.4, 0.9, 1.8, 1.2], start=1):
        generation = tracker.start_generation("core-rl", {"index": index})
        tracker.complete_generation(
            generation.generation_id,
            _training_metrics(final_reward=reward),
            f"{index}.zip",
        )
        generation_ids.append(generation.generation_id)

    best = tracker.get_best_generation("core-rl", "final_reward")
    assert best is not None
    assert best.training_metrics is not None
    assert best.training_metrics.final_reward == 1.8

    removed = tracker.prune_generations("core-rl", keep_best=1, keep_recent=2)
    assert removed == 3

    remaining = tracker.get_generations_for_model("core-rl")
    assert len(remaining) == 2
    remaining_ids = {item.generation_id for item in remaining}
    assert generation_ids[-1] in remaining_ids
    assert best.generation_id in remaining_ids
