"""Connection manager for API WebSocket clients."""

from __future__ import annotations

import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Track active WebSocket clients and topic subscriptions."""

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}
        self._topic_subscribers: dict[str, set[str]] = {}
        self._client_topics: dict[str, set[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept and register an incoming WebSocket connection."""

        await websocket.accept()
        self.active_connections[client_id] = websocket
        self._client_topics.setdefault(client_id, set())

    def disconnect(self, client_id: str) -> None:
        """Remove a disconnected client and clear its subscriptions."""

        self.active_connections.pop(client_id, None)

        topics = self._client_topics.pop(client_id, set())
        for topic in topics:
            subscribers = self._topic_subscribers.get(topic)
            if not subscribers:
                continue
            subscribers.discard(client_id)
            if not subscribers:
                self._topic_subscribers.pop(topic, None)

    async def send_to_client(self, client_id: str, message: dict) -> None:
        """Send a message to a specific connected client."""

        websocket = self.active_connections.get(client_id)
        if websocket is None:
            return

        try:
            await websocket.send_json(message)
        except Exception:
            logger.warning(
                "Failed sending message to client_id=%s; disconnecting client.",
                client_id,
            )
            self.disconnect(client_id)

    async def broadcast(self, message: dict) -> None:
        """Send a message to all connected clients."""

        for client_id in list(self.active_connections):
            await self.send_to_client(client_id, message)

    async def broadcast_to_topic(self, topic: str, message: dict) -> None:
        """Send a message to clients subscribed to a topic."""

        subscribers = self._topic_subscribers.get(topic, set())
        for client_id in list(subscribers):
            await self.send_to_client(client_id, {"topic": topic, "data": message})

    def subscribe(self, client_id: str, topic: str) -> None:
        """Subscribe a client to a topic channel."""

        self._topic_subscribers.setdefault(topic, set()).add(client_id)
        self._client_topics.setdefault(client_id, set()).add(topic)

    def unsubscribe(self, client_id: str, topic: str) -> None:
        """Unsubscribe a client from a topic channel."""

        subscribers = self._topic_subscribers.get(topic)
        if subscribers is not None:
            subscribers.discard(client_id)
            if not subscribers:
                self._topic_subscribers.pop(topic, None)

        client_topics = self._client_topics.get(client_id)
        if client_topics is not None:
            client_topics.discard(topic)
