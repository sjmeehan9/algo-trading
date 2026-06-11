"""LLM-assisted hyperparameter optimization service."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from threading import RLock
from typing import TypeAlias
from uuid import uuid4

from algotrading.api.config import APIConfig
from algotrading.api.schemas.models import ModelConfigResponse, ModelConfigUpdate
from algotrading.api.schemas.optimizer import (
    AppliedSuggestionResponse,
    ApplySuggestionRequest,
    HyperparameterPrimitive,
    HyperparameterSuggestion,
    OptimizationResult,
    OptimizationSource,
    OptimizerAnalyzeRequest,
    SuggestionConfidence,
    SuggestionOutcomeStatus,
)
from algotrading.api.services.model_service import ModelNotFoundError, ModelService
from algotrading.api.services.openai_client import (
    ChatCompletionClient,
    OpenAIClient,
    OpenAIClientError,
    OptimizerUnavailableError,
)
from algotrading.src.models.tracking import Generation
from fastapi import Request

logger = logging.getLogger(__name__)

JsonObject: TypeAlias = dict[str, object]
NonNullPrimitive: TypeAlias = int | float | str | bool
_NUMERIC_TYPES: tuple[type[object], ...] = (int, float)


class OptimizerServiceError(Exception):
    """Base exception for optimizer service failures."""


class OptimizerValidationError(OptimizerServiceError):
    """Raised when an optimizer request is invalid."""


class OptimizerSuggestionNotFoundError(OptimizerServiceError):
    """Raised when a stored suggestion cannot be found."""


@dataclass(slots=True)
class ParsedOptimizationResponse:
    """Normalized content parsed from an LLM optimizer response."""

    summary: str
    suggestions: list[HyperparameterSuggestion]
    priority_changes: list[str]


class LLMOptimizer:
    """Analyze model history and generate hyperparameter suggestions."""

    JSON_INSTRUCTIONS = {
        "summary": "Concise analysis of the training/evaluation trend.",
        "suggestions": [
            {
                "parameter": "hyperparameter name, for example learning_rate",
                "current_value": "current primitive value or null",
                "suggested_value": "new primitive value",
                "rationale": "why this change follows from the metrics",
                "confidence": "low, medium, or high",
                "expected_impact": "expected training/evaluation impact",
            }
        ],
        "priority_changes": ["highest priority changes in order"],
    }

    def __init__(
        self,
        model_service: ModelService,
        openai_client: ChatCompletionClient | None = None,
        suggestions_path: str | Path | None = None,
    ) -> None:
        """Initialize optimizer dependencies.

        Args:
            model_service: Model service used to read and update model configs.
            openai_client: Optional OpenAI-compatible completion client.
            suggestions_path: Optional JSON persistence path for optimizer results.
        """

        self.model_service = model_service
        self.generation_tracker = model_service.generation_tracker
        self.openai_client = openai_client
        self._suggestions_path = Path(suggestions_path) if suggestions_path else None
        self._results: dict[str, OptimizationResult] = {}
        self._lock = RLock()
        self._load_results()

    def analyze(self, request: OptimizerAnalyzeRequest) -> OptimizationResult:
        """Analyze model history and persist a new optimizer result."""

        model = self.model_service.get_model(request.model_id)
        generations = self._recent_generations(
            model_id=model.model_id,
            limit=request.max_generations,
        )
        prompt = self.build_prompt(
            model=model,
            generations=generations,
            include_backtest_metrics=request.include_backtest_metrics,
        )

        raw_response: str | None = None
        source = OptimizationSource.OPENAI
        try:
            if self.openai_client is None:
                raise OptimizerUnavailableError("OpenAI client is not configured.")
            raw_response = self.openai_client.complete_json(prompt)
            parsed = self.parse_response(
                raw_response=raw_response,
                model=model,
            )
        except OpenAIClientError as exc:
            logger.info(
                "Using local optimizer fallback for model_id=%s reason=%s",
                model.model_id,
                type(exc).__name__,
            )
            source = OptimizationSource.FALLBACK
            parsed = self._fallback_response(model=model, generations=generations)

        result = OptimizationResult(
            result_id=f"optimizer-{uuid4().hex[:12]}",
            model_id=model.model_id,
            created_at=datetime.now(tz=UTC),
            analyzed_generation_ids=[item.generation_id for item in generations],
            suggestions=parsed.suggestions,
            priority_changes=parsed.priority_changes,
            summary=parsed.summary,
            source=source,
            raw_response=raw_response,
        )

        with self._lock:
            self._results[result.result_id] = result
            self._persist_results_locked()
        return result

    def build_prompt(
        self,
        model: ModelConfigResponse,
        generations: list[Generation],
        include_backtest_metrics: bool,
    ) -> str:
        """Build the LLM prompt containing model and generation context."""

        payload = {
            "task": "Suggest hyperparameter changes for the next training generation.",
            "model": model.model_dump(mode="json"),
            "history": [
                self._generation_context(
                    generation,
                    include_backtest_metrics=include_backtest_metrics,
                )
                for generation in generations
            ],
            "metric_trends": self._metric_trends(generations),
            "response_schema": self.JSON_INSTRUCTIONS,
            "constraints": [
                "Suggest only hyperparameter changes, not model architecture rewrites.",
                "Use primitive JSON values for suggested_value.",
                "Prefer small, explainable changes that can be evaluated in the next generation.",
                "Do not recommend parameters that are absent unless they are standard for the model algorithm.",
            ],
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def parse_response(
        self,
        raw_response: str,
        model: ModelConfigResponse,
    ) -> ParsedOptimizationResponse:
        """Parse a raw LLM response into normalized optimizer content."""

        response_text = raw_response.strip()
        payload = self._extract_json_payload(response_text)
        if payload is None:
            return ParsedOptimizationResponse(
                summary=response_text[:800]
                or "Optimizer response did not contain JSON.",
                suggestions=[],
                priority_changes=[],
            )

        current_hyperparameters = dict(model.hyperparameters)
        suggestions = self._parse_suggestions(payload, current_hyperparameters)
        summary = self._string_field(
            payload,
            ("summary", "analysis", "overview"),
            default="Optimizer analysis completed.",
        )
        priority_changes = self._string_list(payload.get("priority_changes"))

        return ParsedOptimizationResponse(
            summary=summary,
            suggestions=suggestions,
            priority_changes=priority_changes,
        )

    def get_latest_result(self, model_id: str) -> OptimizationResult | None:
        """Return the newest optimizer result for a model, if one exists."""

        self.model_service.get_model(model_id)
        with self._lock:
            matching = [
                result
                for result in self._results.values()
                if result.model_id == model_id
            ]
        if not matching:
            return None
        latest = max(matching, key=lambda item: item.created_at)
        return self._refresh_outcomes(latest)

    def apply_suggestion(
        self,
        request: ApplySuggestionRequest,
    ) -> AppliedSuggestionResponse:
        """Apply one stored suggestion to the target model hyperparameters."""

        model = self.model_service.get_model(request.model_id)
        result = self._find_result_for_suggestion(
            model_id=model.model_id,
            suggestion_id=request.suggestion_id,
            result_id=request.result_id,
        )
        suggestion_index = next(
            (
                index
                for index, suggestion in enumerate(result.suggestions)
                if suggestion.suggestion_id == request.suggestion_id
            ),
            None,
        )
        if suggestion_index is None:
            raise OptimizerSuggestionNotFoundError(
                f"Suggestion '{request.suggestion_id}' not found"
            )

        suggestion = result.suggestions[suggestion_index]
        if suggestion.applied:
            raise OptimizerValidationError(
                f"Suggestion '{request.suggestion_id}' has already been applied"
            )

        parameter = self._normalize_parameter(suggestion.parameter)
        suggested_value = self._require_non_null_primitive(suggestion.suggested_value)
        hyperparameters = dict(model.hyperparameters)
        hyperparameters[parameter] = suggested_value
        updated_model = self.model_service.update_model(
            model.model_id,
            ModelConfigUpdate(hyperparameters=hyperparameters),
        )

        applied = suggestion.model_copy(
            update={
                "applied": True,
                "applied_at": datetime.now(tz=UTC),
                "outcome_status": SuggestionOutcomeStatus.PENDING_NEXT_GENERATION,
                "outcome_notes": "Applied to model configuration; waiting for the next generation outcome.",
            }
        )
        updated_suggestions = list(result.suggestions)
        updated_suggestions[suggestion_index] = applied
        updated_result = result.model_copy(update={"suggestions": updated_suggestions})

        with self._lock:
            self._results[updated_result.result_id] = updated_result
            self._persist_results_locked()

        return AppliedSuggestionResponse(
            suggestion=applied,
            updated_model=updated_model,
            optimization_result=updated_result,
        )

    def _parse_suggestions(
        self,
        payload: JsonObject,
        current_hyperparameters: dict[str, HyperparameterPrimitive],
    ) -> list[HyperparameterSuggestion]:
        """Parse suggestion dictionaries from a response payload."""

        raw_suggestions = payload.get("suggestions")
        if not isinstance(raw_suggestions, list):
            return []

        suggestions: list[HyperparameterSuggestion] = []
        for raw_item in raw_suggestions:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            parameter = self._string_field(item, ("parameter", "name"), default="")
            try:
                parameter = self._normalize_parameter(parameter)
            except OptimizerValidationError:
                continue
            if not parameter:
                continue
            suggested_value = self._coerce_primitive(item.get("suggested_value"))
            if suggested_value is None:
                continue
            current_value = self._coerce_primitive(
                item.get("current_value", current_hyperparameters.get(parameter))
            )
            suggestions.append(
                HyperparameterSuggestion(
                    suggestion_id=f"suggestion-{uuid4().hex[:12]}",
                    parameter=parameter,
                    current_value=current_value,
                    suggested_value=suggested_value,
                    rationale=self._string_field(
                        item,
                        ("rationale", "reason", "explanation"),
                        default="Suggested by optimizer analysis.",
                    ),
                    confidence=self._coerce_confidence(item.get("confidence")),
                    expected_impact=self._string_field(
                        item,
                        ("expected_impact", "impact"),
                        default="Expected to improve next-generation training stability.",
                    ),
                )
            )
        return suggestions

    def _fallback_response(
        self,
        model: ModelConfigResponse,
        generations: list[Generation],
    ) -> ParsedOptimizationResponse:
        """Create deterministic local suggestions when OpenAI is unavailable."""

        suggestions: list[HyperparameterSuggestion] = []
        hyperparameters = dict(model.hyperparameters)
        latest = generations[0] if generations else None
        trend = self._metric_trends(generations)
        reward_delta = _numeric_or_none(trend.get("final_reward_delta"))
        drawdown = None
        if latest is not None:
            drawdown = latest.metric_value("max_drawdown")

        learning_rate = hyperparameters.get("learning_rate")
        if _is_numeric(learning_rate) and (
            (reward_delta is not None and reward_delta < 0)
            or (drawdown is not None and drawdown > 0.1)
        ):
            suggested = max(float(learning_rate) * 0.5, 0.000001)
            suggestions.append(
                self._fallback_suggestion(
                    parameter="learning_rate",
                    current_value=learning_rate,
                    suggested_value=suggested,
                    rationale=(
                        "Recent metrics indicate instability or weaker reward momentum; "
                        "a smaller learning rate can reduce update noise."
                    ),
                    expected_impact="More stable policy updates with lower drawdown risk.",
                    confidence=SuggestionConfidence.MEDIUM,
                )
            )

        gamma = hyperparameters.get("gamma")
        if _is_numeric(gamma) and float(gamma) < 0.995:
            suggested_gamma = min(float(gamma) + 0.01, 0.999)
            suggestions.append(
                self._fallback_suggestion(
                    parameter="gamma",
                    current_value=gamma,
                    suggested_value=suggested_gamma,
                    rationale="Trading rewards often depend on longer-horizon outcomes than a single step.",
                    expected_impact="Encourages the policy to value longer-term portfolio performance.",
                    confidence=SuggestionConfidence.LOW,
                )
            )

        n_steps = hyperparameters.get("n_steps")
        if model.algorithm.lower() == "ppo" and _is_numeric(n_steps):
            suggested_steps = int(max(float(n_steps) * 1.25, float(n_steps) + 1))
            suggestions.append(
                self._fallback_suggestion(
                    parameter="n_steps",
                    current_value=n_steps,
                    suggested_value=suggested_steps,
                    rationale="Longer rollouts can smooth reward estimates for noisy market episodes.",
                    expected_impact="Improves PPO batch signal quality for the next generation.",
                    confidence=SuggestionConfidence.LOW,
                )
            )

        if not suggestions:
            summary = (
                "OpenAI analysis is unavailable, and the local heuristic did not find an "
                "obvious safe hyperparameter adjustment from the current history."
            )
        else:
            summary = (
                "OpenAI analysis is unavailable; local heuristics generated conservative "
                "hyperparameter suggestions from recent training and evaluation metrics."
            )

        return ParsedOptimizationResponse(
            summary=summary,
            suggestions=suggestions[:3],
            priority_changes=[suggestion.parameter for suggestion in suggestions[:3]],
        )

    def _fallback_suggestion(
        self,
        *,
        parameter: str,
        current_value: HyperparameterPrimitive,
        suggested_value: NonNullPrimitive,
        rationale: str,
        expected_impact: str,
        confidence: SuggestionConfidence,
    ) -> HyperparameterSuggestion:
        """Build a local fallback suggestion model."""

        return HyperparameterSuggestion(
            suggestion_id=f"suggestion-{uuid4().hex[:12]}",
            parameter=parameter,
            current_value=current_value,
            suggested_value=suggested_value,
            rationale=rationale,
            confidence=confidence,
            expected_impact=expected_impact,
        )

    def _recent_generations(self, model_id: str, limit: int) -> list[Generation]:
        """Return newest generations first for a model."""

        generations = self.generation_tracker.get_generations_for_model(model_id)
        terminal = [
            generation
            for generation in generations
            if generation.status in {"completed", "evaluated", "failed"}
        ]
        terminal.sort(key=lambda item: item.generation_number, reverse=True)
        return terminal[:limit]

    def _generation_context(
        self,
        generation: Generation,
        include_backtest_metrics: bool,
    ) -> JsonObject:
        """Convert one generation into JSON prompt context."""

        training_metrics = (
            generation.training_metrics.to_dict()
            if generation.training_metrics is not None
            else None
        )
        evaluation_metrics = (
            generation.evaluation_metrics.to_dict()
            if include_backtest_metrics and generation.evaluation_metrics is not None
            else None
        )
        return {
            "generation_id": generation.generation_id,
            "generation_number": generation.generation_number,
            "status": generation.status,
            "created_at": generation.created_at.isoformat(),
            "parent_generation_id": generation.parent_generation_id,
            "hyperparameters": dict(generation.hyperparameters),
            "training_metrics": training_metrics,
            "evaluation_metrics": evaluation_metrics,
            "notes": generation.notes,
            "tags": list(generation.tags),
        }

    def _metric_trends(self, generations: list[Generation]) -> JsonObject:
        """Summarize metric movement from oldest to newest generation."""

        if len(generations) < 2:
            return {}
        ordered = sorted(generations, key=lambda item: item.generation_number)
        oldest = ordered[0]
        newest = ordered[-1]
        trends: JsonObject = {}
        for metric in (
            "final_reward",
            "mean_reward",
            "sharpe_ratio",
            "total_return",
            "max_drawdown",
            "win_rate",
            "profit_factor",
        ):
            old_value = oldest.metric_value(metric)
            new_value = newest.metric_value(metric)
            if old_value is None or new_value is None:
                continue
            trends[f"{metric}_delta"] = new_value - old_value
        return trends

    def _find_result_for_suggestion(
        self,
        *,
        model_id: str,
        suggestion_id: str,
        result_id: str | None,
    ) -> OptimizationResult:
        """Find a result containing a suggestion for the requested model."""

        with self._lock:
            if result_id is not None:
                result = self._results.get(result_id)
                if result is not None and result.model_id == model_id:
                    return result
                raise OptimizerSuggestionNotFoundError(
                    f"Optimizer result '{result_id}' not found for model '{model_id}'"
                )

            candidates = [
                result
                for result in self._results.values()
                if result.model_id == model_id
                and any(
                    suggestion.suggestion_id == suggestion_id
                    for suggestion in result.suggestions
                )
            ]

        if not candidates:
            raise OptimizerSuggestionNotFoundError(
                f"Suggestion '{suggestion_id}' not found for model '{model_id}'"
            )
        return max(candidates, key=lambda item: item.created_at)

    def _refresh_outcomes(self, result: OptimizationResult) -> OptimizationResult:
        """Update applied suggestion outcomes when matching generations exist."""

        generations = self.generation_tracker.get_generations_for_model(result.model_id)
        updated_suggestions: list[HyperparameterSuggestion] = []
        changed = False
        for suggestion in result.suggestions:
            if (
                not suggestion.applied
                or suggestion.applied_at is None
                or suggestion.outcome_status
                != SuggestionOutcomeStatus.PENDING_NEXT_GENERATION
            ):
                updated_suggestions.append(suggestion)
                continue

            parameter = self._normalize_parameter(suggestion.parameter)
            matches = [
                generation
                for generation in generations
                if generation.created_at >= suggestion.applied_at
                and generation.status in {"completed", "evaluated"}
                and generation.hyperparameters.get(parameter)
                == suggestion.suggested_value
            ]
            matches.sort(key=lambda item: item.generation_number)
            if not matches:
                updated_suggestions.append(suggestion)
                continue

            generation = matches[0]
            metrics = []
            for metric in ("final_reward", "sharpe_ratio", "total_return"):
                value = generation.metric_value(metric)
                if value is not None:
                    metrics.append(f"{metric}={value:.4g}")
            note = f"Observed in generation {generation.generation_number} ({generation.status})."
            if metrics:
                note = f"{note} {'; '.join(metrics)}."
            updated_suggestions.append(
                suggestion.model_copy(
                    update={
                        "outcome_status": SuggestionOutcomeStatus.EVALUATED,
                        "outcome_notes": note,
                    }
                )
            )
            changed = True

        if not changed:
            return result

        updated = result.model_copy(update={"suggestions": updated_suggestions})
        with self._lock:
            self._results[updated.result_id] = updated
            self._persist_results_locked()
        return updated

    def _load_results(self) -> None:
        """Load persisted optimizer results if available."""

        if self._suggestions_path is None or not self._suggestions_path.exists():
            return
        try:
            payload = json.loads(self._suggestions_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(
                "Could not parse optimizer suggestions file %s", self._suggestions_path
            )
            return

        loaded: dict[str, OptimizationResult] = {}
        for item in payload.get("results", []):
            try:
                result = OptimizationResult.model_validate(item)
            except Exception:
                logger.exception("Skipping invalid persisted optimizer result entry")
                continue
            loaded[result.result_id] = result

        with self._lock:
            self._results = loaded

    def _persist_results_locked(self) -> None:
        """Persist optimizer results while holding the service lock."""

        if self._suggestions_path is None:
            return
        self._suggestions_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now(tz=UTC).isoformat(),
            "results": [
                result.model_dump(mode="json") for result in self._results.values()
            ],
        }
        self._suggestions_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _extract_json_payload(self, response_text: str) -> JsonObject | None:
        """Extract a JSON object from raw response text."""

        candidates = [response_text]
        if "```" in response_text:
            parts = response_text.split("```")
            for part in parts:
                stripped = part.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                if stripped.startswith("{"):
                    candidates.append(stripped)

        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(response_text[start : end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return dict(parsed)
        return None

    def _string_field(
        self,
        payload: JsonObject,
        keys: tuple[str, ...],
        default: str,
    ) -> str:
        """Return the first non-empty string-like field in a payload."""

        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return default

    def _string_list(self, value: object) -> list[str]:
        """Normalize arbitrary values into a list of non-empty strings."""

        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _coerce_primitive(self, value: object) -> HyperparameterPrimitive:
        """Convert a JSON value into an allowed hyperparameter primitive."""

        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    def _require_non_null_primitive(
        self,
        value: HyperparameterPrimitive,
    ) -> NonNullPrimitive:
        """Validate that a suggested value can be stored in model hyperparameters."""

        if value is None:
            raise OptimizerValidationError("suggested_value cannot be null")
        return value

    def _coerce_confidence(self, value: object) -> SuggestionConfidence:
        """Normalize string or numeric confidence values."""

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {item.value for item in SuggestionConfidence}:
                return SuggestionConfidence(normalized)
        if _is_numeric(value):
            numeric = float(value)
            if numeric >= 0.75:
                return SuggestionConfidence.HIGH
            if numeric >= 0.45:
                return SuggestionConfidence.MEDIUM
            return SuggestionConfidence.LOW
        return SuggestionConfidence.MEDIUM

    def _normalize_parameter(self, parameter: str) -> str:
        """Normalize model hyperparameter field names."""

        normalized = parameter.strip()
        if normalized.startswith("hyperparameters."):
            normalized = normalized.split(".", 1)[1]
        if "." in normalized:
            raise OptimizerValidationError(
                f"Only direct hyperparameter suggestions are supported: {parameter}"
            )
        return normalized


def _is_numeric(value: object) -> bool:
    """Return True when value is finite numeric and not boolean."""

    return (
        isinstance(value, _NUMERIC_TYPES)
        and not isinstance(value, bool)
        and isfinite(float(value))
    )


def _numeric_or_none(value: object) -> float | None:
    """Convert numeric values to float or return None."""

    if not _is_numeric(value):
        return None
    return float(value)


def create_default_optimizer_service(
    *,
    model_service: ModelService,
    config: APIConfig,
    project_root: Path | None = None,
    openai_client: ChatCompletionClient | None = None,
) -> LLMOptimizer:
    """Build an optimizer service with filesystem-backed persistence."""

    root = project_root or Path(__file__).resolve().parents[4]
    data_dir = root / "data" / "api"
    resolved_client = openai_client
    if resolved_client is None and config.openai_api_key:
        try:
            resolved_client = OpenAIClient(
                api_key=config.openai_api_key,
                model=config.openai_model,
                timeout_seconds=config.openai_timeout_seconds,
            )
        except OptimizerUnavailableError:
            logger.warning(
                "OpenAI optimizer client is unavailable; fallback mode is enabled."
            )

    return LLMOptimizer(
        model_service=model_service,
        openai_client=resolved_client,
        suggestions_path=data_dir / "hyperparameter_suggestions.json",
    )


def get_optimizer_service(request: Request) -> LLMOptimizer:
    """FastAPI dependency resolver for the shared optimizer service."""

    from algotrading.api.services.app_state import get_or_create_state

    def _build() -> LLMOptimizer:
        from algotrading.api.services.model_service import get_model_service

        model_service = get_model_service(request)
        config = request.app.state.api_config
        return create_default_optimizer_service(
            model_service=model_service,
            config=config,
        )

    return get_or_create_state(request, "optimizer_service", _build)


__all__ = [
    "LLMOptimizer",
    "OptimizerServiceError",
    "OptimizerSuggestionNotFoundError",
    "OptimizerValidationError",
    "ParsedOptimizationResponse",
    "create_default_optimizer_service",
    "get_optimizer_service",
]
