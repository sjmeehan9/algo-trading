"""Unit tests for the API WebSocket manager."""

from __future__ import annotations

import asyncio

from algotrading.api.websocket import WebSocketManager


class FakeWebSocket:
    """Minimal async WebSocket stub for manager unit tests."""

    def __init__(self) -> None:
        self.accepted = False
        self.sent_messages: list[dict] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, message: dict) -> None:
        self.sent_messages.append(message)


def test_connect_and_disconnect_client() -> None:
    """Manager should track active clients and clear subscriptions on disconnect."""

    manager = WebSocketManager()
    ws = FakeWebSocket()

    asyncio.run(manager.connect(ws, "client-1"))
    manager.subscribe("client-1", "training:model-1")

    assert ws.accepted is True
    assert "client-1" in manager.active_connections

    manager.disconnect("client-1")

    assert "client-1" not in manager.active_connections


def test_send_to_specific_client() -> None:
    """Manager should send directed messages to the specified client only."""

    manager = WebSocketManager()
    ws_one = FakeWebSocket()
    ws_two = FakeWebSocket()

    asyncio.run(manager.connect(ws_one, "client-1"))
    asyncio.run(manager.connect(ws_two, "client-2"))

    asyncio.run(manager.send_to_client("client-1", {"type": "hello"}))

    assert ws_one.sent_messages == [{"type": "hello"}]
    assert ws_two.sent_messages == []


def test_broadcast_to_topic() -> None:
    """Topic broadcast should only reach subscribed clients."""

    manager = WebSocketManager()
    ws_one = FakeWebSocket()
    ws_two = FakeWebSocket()

    asyncio.run(manager.connect(ws_one, "client-1"))
    asyncio.run(manager.connect(ws_two, "client-2"))

    manager.subscribe("client-1", "training:model-1")
    manager.subscribe("client-2", "training:model-2")

    asyncio.run(
        manager.broadcast_to_topic(
            "training:model-1",
            {"type": "progress", "data": {"percent": 10}},
        )
    )

    assert ws_one.sent_messages == [
        {
            "topic": "training:model-1",
            "data": {"type": "progress", "data": {"percent": 10}},
        }
    ]
    assert ws_two.sent_messages == []
