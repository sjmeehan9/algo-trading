"""Thread-safe lazy initialization for services cached on ``app.state``.

FastAPI executes synchronous dependency resolvers in a threadpool, so two
concurrent requests can race through an unguarded check-create-cache getter
and build duplicate service instances. The duplicate that loses the
``app.state`` slot can survive captured inside a composite service (for
example a ``TrainingService`` holding a stale ``ModelService`` whose registry
snapshot predates later registrations), producing split-brain state within a
single process.

``get_or_create_state`` serializes creation behind a process-wide re-entrant
lock so exactly one instance is ever created per attribute. The lock is
re-entrant because service factories compose other getters on the same thread
(training -> model, deployment -> model + backtest, and so on).
"""

from __future__ import annotations

import threading
from typing import Callable, TypeVar

from fastapi import Request

T = TypeVar("T")

_APP_STATE_LOCK = threading.RLock()


def get_or_create_state(
    request: Request,
    attribute: str,
    factory: Callable[[], T],
) -> T:
    """Return the ``app.state`` attribute, creating it atomically when absent.

    Args:
        request: The current request (or any object exposing ``app.state``).
        attribute: Name of the ``app.state`` attribute to resolve.
        factory: Zero-argument callable building the service. Invoked at most
            once per attribute per process; it may safely call other getters
            that use this helper.

    Returns:
        The cached or newly created service instance.
    """

    service = getattr(request.app.state, attribute, None)
    if service is not None:
        return service

    with _APP_STATE_LOCK:
        service = getattr(request.app.state, attribute, None)
        if service is None:
            service = factory()
            setattr(request.app.state, attribute, service)
        return service


__all__ = ["get_or_create_state"]
