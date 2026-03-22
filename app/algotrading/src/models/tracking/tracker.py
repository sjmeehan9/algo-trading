"""Service layer for model generation tracking and comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from uuid import uuid4

from algotrading.src.models.tracking.generation import (
    EvaluationMetrics,
    Generation,
    TrainingMetrics,
)
from algotrading.src.models.tracking.storage import GenerationStorage


@dataclass(slots=True)
class GenerationComparison:
    """Comparison output for two generation records."""

    gen_a: Generation
    gen_b: Generation
    metric_comparisons: dict[str, tuple[float, float, float]]
    improvement_summary: str


class GenerationTracker:
    """Tracks model generation lifecycle and enables historical analysis."""

    DEFAULT_COMPARISON_METRICS = ["final_reward", "sharpe_ratio", "max_drawdown"]
    LOWER_IS_BETTER_METRICS = {"max_drawdown"}

    def __init__(self, storage: GenerationStorage) -> None:
        self._storage = storage
        self._lock = RLock()

    def start_generation(
        self,
        model_id: str,
        hyperparameters: dict[str, object],
        parent_generation_id: str | None = None,
    ) -> Generation:
        """Create and persist a new in-progress generation record."""

        with self._lock:
            generation = Generation(
                generation_id=str(uuid4()),
                model_id=model_id,
                generation_number=self._get_next_generation_number(model_id),
                parent_generation_id=parent_generation_id,
                created_at=datetime.now(UTC),
                status="training",
                hyperparameters=dict(hyperparameters),
            )
            self._storage.save(generation)
            return generation

    def complete_generation(
        self,
        generation_id: str,
        training_metrics: TrainingMetrics,
        model_path: str,
    ) -> None:
        """Mark a generation as completed with final training metrics."""

        generation = self._require_generation(generation_id)
        generation.training_metrics = training_metrics
        generation.model_path = model_path
        generation.status = "completed"
        self._storage.save(generation)

    def fail_generation(self, generation_id: str, error: str) -> None:
        """Mark a generation as failed and append the error message to notes."""

        generation = self._require_generation(generation_id)
        generation.status = "failed"
        if generation.notes:
            generation.notes = f"{generation.notes}\n{error}"
        else:
            generation.notes = error
        self._storage.save(generation)

    def add_evaluation(
        self,
        generation_id: str,
        eval_metrics: EvaluationMetrics,
    ) -> None:
        """Attach evaluation metrics and promote generation to evaluated status."""

        generation = self._require_generation(generation_id)
        generation.evaluation_metrics = eval_metrics
        generation.status = "evaluated"
        self._storage.save(generation)

    def get_generation(self, generation_id: str) -> Generation | None:
        """Fetch one generation by ID."""

        return self._storage.load(generation_id)

    def get_generations_for_model(self, model_id: str) -> list[Generation]:
        """Fetch all generations for a model in generation order."""

        return self._storage.load_all_for_model(model_id)

    def get_latest_generation(self, model_id: str) -> Generation | None:
        """Get the newest generation for a model."""

        generations = self._storage.load_all_for_model(model_id)
        if not generations:
            return None
        return max(generations, key=lambda item: item.generation_number)

    def get_best_generation(
        self,
        model_id: str,
        metric: str,
        higher_is_better: bool = True,
    ) -> Generation | None:
        """Get the best generation by a chosen metric."""

        generations = self._storage.load_all_for_model(model_id)
        scored = [
            (generation, self._get_metric_value(generation, metric))
            for generation in generations
        ]
        scored = [
            (generation, value) for generation, value in scored if value is not None
        ]
        if not scored:
            return None

        reverse = higher_is_better
        scored.sort(
            key=lambda item: (
                float(item[1]),
                item[0].generation_number,
            ),
            reverse=reverse,
        )
        return scored[0][0]

    def compare_generations(
        self,
        gen_a_id: str,
        gen_b_id: str,
        metrics: list[str] | None = None,
    ) -> GenerationComparison:
        """Compare two generations across training and evaluation metrics."""

        gen_a = self._require_generation(gen_a_id)
        gen_b = self._require_generation(gen_b_id)

        comparisons: dict[str, tuple[float, float, float]] = {}
        for metric in metrics or self.DEFAULT_COMPARISON_METRICS:
            a_val = self._get_metric_value(gen_a, metric)
            b_val = self._get_metric_value(gen_b, metric)
            if a_val is None or b_val is None:
                continue
            comparisons[metric] = (a_val, b_val, b_val - a_val)

        return GenerationComparison(
            gen_a=gen_a,
            gen_b=gen_b,
            metric_comparisons=comparisons,
            improvement_summary=self._generate_summary(comparisons),
        )

    def get_lineage(self, generation_id: str) -> list[Generation]:
        """Return ancestor chain from root generation to the given generation."""

        lineage: list[Generation] = []
        current = self._require_generation(generation_id)
        lineage.append(current)

        while current.parent_generation_id is not None:
            parent = self._storage.load(current.parent_generation_id)
            if parent is None:
                break
            lineage.append(parent)
            current = parent

        lineage.reverse()
        return lineage

    def get_descendants(self, generation_id: str) -> list[Generation]:
        """Return all descendants of a generation within the same model lineage."""

        root = self._require_generation(generation_id)
        all_generations = self._storage.load_all_for_model(root.model_id)

        by_parent: dict[str, list[Generation]] = {}
        for generation in all_generations:
            if generation.parent_generation_id is None:
                continue
            by_parent.setdefault(generation.parent_generation_id, []).append(generation)

        descendants: list[Generation] = []
        stack = list(by_parent.get(root.generation_id, []))
        while stack:
            child = stack.pop()
            descendants.append(child)
            stack.extend(by_parent.get(child.generation_id, []))

        descendants.sort(key=lambda item: item.generation_number)
        return descendants

    def prune_generations(
        self,
        model_id: str,
        keep_best: int = 5,
        keep_recent: int = 10,
    ) -> int:
        """Prune older generations while preserving recent and top-performing runs."""

        with self._lock:
            generations = self._storage.load_all_for_model(model_id)
            if not generations:
                return 0

            generations.sort(key=lambda item: item.generation_number)
            keep_ids = {
                generation.generation_id
                for generation in generations[-max(keep_recent, 0) :]
            }

            best_candidates = [
                generation
                for generation in generations
                if generation.status in {"completed", "evaluated"}
            ]
            best_candidates.sort(key=self._composite_score, reverse=True)
            for generation in best_candidates[: max(keep_best, 0)]:
                keep_ids.add(generation.generation_id)

            deleted = 0
            for generation in generations:
                if generation.generation_id in keep_ids:
                    continue
                if self._storage.delete(generation.generation_id):
                    deleted += 1
            return deleted

    def export_history(self, model_id: str, filepath: str) -> None:
        """Export full generation history for a model to a JSON file."""

        generations = self._storage.load_all_for_model(model_id)
        export_payload = {
            "model_id": model_id,
            "exported_at": datetime.now(UTC).isoformat(),
            "generations": [generation.to_dict() for generation in generations],
        }

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(export_payload, file_handle, indent=2, sort_keys=True)

    def _get_next_generation_number(self, model_id: str) -> int:
        """Compute next generation number for a model ID."""

        generations = self._storage.load_all_for_model(model_id)
        if not generations:
            return 1
        return max(item.generation_number for item in generations) + 1

    def _get_metric_value(self, generation: Generation, metric: str) -> float | None:
        """Resolve a metric value from generation metric containers."""

        return generation.metric_value(metric)

    def _generate_summary(
        self,
        comparisons: dict[str, tuple[float, float, float]],
    ) -> str:
        """Build a concise text summary for metric comparison outcomes."""

        if not comparisons:
            return "No overlapping metrics available for comparison."

        improved: list[str] = []
        regressed: list[str] = []
        unchanged: list[str] = []

        for metric, (_a_val, _b_val, diff) in comparisons.items():
            lower_is_better = metric in self.LOWER_IS_BETTER_METRICS
            if diff == 0:
                unchanged.append(metric)
                continue
            if (diff > 0 and not lower_is_better) or (diff < 0 and lower_is_better):
                improved.append(metric)
            else:
                regressed.append(metric)

        parts: list[str] = []
        if improved:
            parts.append(f"Improved: {', '.join(sorted(improved))}")
        if regressed:
            parts.append(f"Regressed: {', '.join(sorted(regressed))}")
        if unchanged:
            parts.append(f"Unchanged: {', '.join(sorted(unchanged))}")
        return " | ".join(parts)

    def _require_generation(self, generation_id: str) -> Generation:
        """Fetch generation and raise a clear error if not found."""

        generation = self._storage.load(generation_id)
        if generation is None:
            raise ValueError(f"Generation not found: {generation_id}")
        return generation

    def _composite_score(
        self, generation: Generation
    ) -> tuple[float, float, float, float, float, int]:
        """Compute deterministic sort key for selecting best generations."""

        final_reward = self._get_metric_value(generation, "final_reward") or float(
            "-inf"
        )
        sharpe_ratio = self._get_metric_value(generation, "sharpe_ratio") or float(
            "-inf"
        )
        total_return = self._get_metric_value(generation, "total_return") or float(
            "-inf"
        )
        win_rate = self._get_metric_value(generation, "win_rate") or float("-inf")
        profit_factor = self._get_metric_value(generation, "profit_factor") or float(
            "-inf"
        )

        max_drawdown = self._get_metric_value(generation, "max_drawdown")
        drawdown_score = float("-inf") if max_drawdown is None else -max_drawdown

        return (
            final_reward,
            sharpe_ratio,
            total_return,
            win_rate,
            profit_factor + drawdown_score,
            generation.generation_number,
        )


__all__ = [
    "GenerationComparison",
    "GenerationTracker",
]
