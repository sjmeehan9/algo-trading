"""Integration tests for end-to-end generation tracking workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from algotrading.src.models.tracking import (
    EvaluationMetrics,
    GenerationTracker,
    JsonFileStorage,
    TrainingMetrics,
)


def _metrics(reward: float) -> TrainingMetrics:
    return TrainingMetrics(
        final_reward=reward,
        mean_reward=reward * 0.8,
        std_reward=0.2,
        episodes_completed=10,
        timesteps_trained=2000,
        training_time_seconds=20.0,
        loss_history=[1.0, 0.8, 0.6],
        custom_metrics={"stability": 0.7 + (reward * 0.05)},
    )


def test_full_training_generation_workflow_and_restart_persistence(tmp_path) -> None:
    """Generation tracking persists complete lifecycle state across tracker restarts."""

    base_path = tmp_path / "generation_history"

    tracker_a = GenerationTracker(JsonFileStorage(str(base_path)))

    gen_1 = tracker_a.start_generation(
        model_id="core-rl",
        hyperparameters={"learning_rate": 0.0003, "gamma": 0.99},
    )
    tracker_a.complete_generation(gen_1.generation_id, _metrics(1.1), "model-gen-1.zip")
    tracker_a.add_evaluation(
        gen_1.generation_id,
        EvaluationMetrics(
            sharpe_ratio=0.8,
            max_drawdown=0.24,
            total_return=0.14,
            win_rate=0.52,
            profit_factor=1.18,
            num_trades=30,
        ),
    )

    gen_2 = tracker_a.start_generation(
        model_id="core-rl",
        hyperparameters={"learning_rate": 0.0002, "gamma": 0.995},
        parent_generation_id=gen_1.generation_id,
    )
    tracker_a.complete_generation(gen_2.generation_id, _metrics(1.5), "model-gen-2.zip")
    tracker_a.add_evaluation(
        gen_2.generation_id,
        EvaluationMetrics(
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            total_return=0.23,
            win_rate=0.59,
            profit_factor=1.33,
            num_trades=42,
        ),
    )

    export_path = tmp_path / "exports" / "core-rl-history.json"
    tracker_a.export_history("core-rl", str(export_path))

    tracker_b = GenerationTracker(JsonFileStorage(str(base_path)))

    all_generations = tracker_b.get_generations_for_model("core-rl")
    assert len(all_generations) == 2
    assert [item.generation_number for item in all_generations] == [1, 2]

    lineage = tracker_b.get_lineage(gen_2.generation_id)
    assert [item.generation_id for item in lineage] == [
        gen_1.generation_id,
        gen_2.generation_id,
    ]

    comparison = tracker_b.compare_generations(gen_1.generation_id, gen_2.generation_id)
    final_reward = comparison.metric_comparisons["final_reward"]
    assert final_reward[0] == 1.1
    assert final_reward[1] == 1.5
    assert final_reward[2] == pytest.approx(0.4)

    sharpe_ratio = comparison.metric_comparisons["sharpe_ratio"]
    assert sharpe_ratio[0] == 0.8
    assert sharpe_ratio[1] == 1.2
    assert sharpe_ratio[2] == pytest.approx(0.4)

    best = tracker_b.get_best_generation("core-rl", "sharpe_ratio")
    assert best is not None
    assert best.generation_id == gen_2.generation_id

    assert export_path.exists()
    payload = json.loads(Path(export_path).read_text(encoding="utf-8"))
    assert payload["model_id"] == "core-rl"
    assert len(payload["generations"]) == 2
