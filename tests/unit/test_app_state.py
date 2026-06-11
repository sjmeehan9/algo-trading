"""Unit tests for thread-safe lazy app.state service initialization."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from algotrading.api.services.app_state import get_or_create_state


def _fake_request() -> SimpleNamespace:
    """Build a request-like object exposing a fresh ``app.state``."""

    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


def test_returns_cached_instance_without_factory_call() -> None:
    """An existing app.state attribute is returned without invoking the factory."""

    request = _fake_request()
    sentinel = object()
    request.app.state.model_service = sentinel

    def factory() -> object:
        raise AssertionError("factory must not be called for cached state")

    assert get_or_create_state(request, "model_service", factory) is sentinel


def test_concurrent_getters_create_exactly_one_instance() -> None:
    """Racing threads must share a single created service instance.

    Regression test for UAT error-005: FastAPI runs sync dependency resolvers
    in a threadpool, and the unguarded check-create-cache pattern let two
    threads build duplicate ModelService instances. The duplicate captured by
    TrainingService had a registry snapshot predating later model
    registrations, producing "Model not found" for a registered model.
    """

    request = _fake_request()
    factory_calls: list[int] = []

    def slow_factory() -> object:
        factory_calls.append(1)
        # Widen the race window: without locking, every waiting thread
        # observes None and builds its own instance.
        time.sleep(0.05)
        return object()

    thread_count = 16
    barrier = threading.Barrier(thread_count)

    def resolve() -> object:
        barrier.wait()
        return get_or_create_state(request, "model_service", slow_factory)

    with ThreadPoolExecutor(max_workers=thread_count) as pool:
        results = [future.result() for future in [pool.submit(resolve) for _ in range(thread_count)]]

    assert len(factory_calls) == 1
    assert all(result is results[0] for result in results)
    assert request.app.state.model_service is results[0]


def test_nested_composition_shares_the_dependency_instance() -> None:
    """A composite factory resolving its dependency re-enters the lock safely

    and captures the same instance that later direct resolutions return,
    mirroring get_training_service -> get_model_service composition.
    """

    request = _fake_request()

    def model_factory() -> object:
        return object()

    def training_factory() -> SimpleNamespace:
        model_service = get_or_create_state(request, "model_service", model_factory)
        return SimpleNamespace(model_service=model_service)

    def resolve_training() -> SimpleNamespace:
        return get_or_create_state(request, "training_service", training_factory)

    def resolve_model() -> object:
        return get_or_create_state(request, "model_service", model_factory)

    with ThreadPoolExecutor(max_workers=8) as pool:
        training_futures = [pool.submit(resolve_training) for _ in range(4)]
        model_futures = [pool.submit(resolve_model) for _ in range(4)]
        training_results = [future.result() for future in training_futures]
        model_results = [future.result() for future in model_futures]

    shared_model = request.app.state.model_service
    assert all(result is shared_model for result in model_results)
    assert all(result.model_service is shared_model for result in training_results)
    assert request.app.state.training_service.model_service is shared_model
