"""Trading operations for the Alpaca adapter."""

from __future__ import annotations

from dataclasses import dataclass

from algotrading.src.broker.adapters.alpaca.auth import AlpacaAuth
from algotrading.src.broker.adapters.alpaca.models import (
    alpaca_account_to_account_info,
    alpaca_order_to_order_status,
    alpaca_position_to_position_info,
    contract_to_alpaca_symbol,
)
from algotrading.src.broker.models import (
    AccountInfo,
    ContractSpec,
    OrderSide,
    OrderSpec,
    OrderStatus,
    OrderType,
    PositionInfo,
)


@dataclass(frozen=True)
class AlpacaOrderPayload:
    """SDK-free order payload used when ``alpaca-py`` is unavailable in tests."""

    symbol: str
    qty: float
    side: str
    time_in_force: str
    order_type: str
    limit_price: float | None = None
    stop_price: float | None = None


class AlpacaTradingClient:
    """Wrapper around Alpaca's trading SDK client."""

    def __init__(self, auth: AlpacaAuth, trading_client: object | None = None) -> None:
        """Initialize a trading wrapper with optional injected SDK client."""

        self.auth = auth
        self._client = trading_client

    @property
    def client(self) -> object:
        """Return the underlying SDK trading client, creating it lazily."""

        if self._client is None:
            self._client = self.auth.get_trading_client()
        return self._client

    def verify_connection(self) -> AccountInfo:
        """Verify trading connectivity by fetching the account snapshot."""

        return self.get_account_info()

    def submit_order(self, contract: ContractSpec, order: OrderSpec) -> OrderStatus:
        """Submit an order and return the normalized initial status."""

        request = build_order_request(contract, order)
        response = self.client.submit_order(request)
        return alpaca_order_to_order_status(response, original_order=order)

    def cancel_order(self, order_id: str) -> None:
        """Cancel an Alpaca order by UUID."""

        self.client.cancel_order_by_id(order_id)

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Fetch and normalize an Alpaca order status by UUID."""

        response = self.client.get_order_by_id(order_id)
        return alpaca_order_to_order_status(response)

    def get_account_info(self) -> AccountInfo:
        """Fetch and normalize account information."""

        return alpaca_account_to_account_info(self.client.get_account())

    def get_positions(self) -> list[PositionInfo]:
        """Fetch and normalize all open positions."""

        return [
            alpaca_position_to_position_info(position)
            for position in self.client.get_all_positions()
        ]


def build_order_request(contract: ContractSpec, order: OrderSpec) -> object:
    """Build an Alpaca SDK order request from canonical order models."""

    symbol = contract_to_alpaca_symbol(contract)
    side = _coerce_order_side(order.side)
    time_in_force = _coerce_time_in_force(order.time_in_force)

    try:
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )
    except ImportError:
        return AlpacaOrderPayload(
            symbol=symbol,
            qty=order.quantity,
            side=str(_enum_value(side)),
            time_in_force=str(_enum_value(time_in_force)),
            order_type=_alpaca_order_type_value(order.order_type),
            limit_price=order.limit_price,
            stop_price=order.stop_price,
        )

    common = {
        "symbol": symbol,
        "qty": order.quantity,
        "side": side,
        "time_in_force": time_in_force,
    }
    if order.order_type == OrderType.MARKET:
        return MarketOrderRequest(**common)
    if order.order_type == OrderType.LIMIT:
        return LimitOrderRequest(**common, limit_price=order.limit_price)
    if order.order_type == OrderType.STOP:
        return StopOrderRequest(**common, stop_price=order.stop_price)
    return StopLimitOrderRequest(
        **common,
        limit_price=order.limit_price,
        stop_price=order.stop_price,
    )


def _coerce_order_side(side: OrderSide) -> object:
    try:
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide
    except ImportError:
        return side.value.lower()
    return AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL


def _coerce_time_in_force(value: str) -> object:
    normalized = value.strip().upper()
    try:
        from alpaca.trading.enums import TimeInForce
    except ImportError:
        return normalized.lower()
    if hasattr(TimeInForce, normalized):
        return getattr(TimeInForce, normalized)
    return normalized.lower()


def _alpaca_order_type_value(order_type: OrderType) -> str:
    mapping = {
        OrderType.MARKET: "market",
        OrderType.LIMIT: "limit",
        OrderType.STOP: "stop",
        OrderType.STOP_LIMIT: "stop_limit",
    }
    return mapping[order_type]


def _enum_value(value: object) -> object:
    if hasattr(value, "value"):
        return getattr(value, "value")
    return value


__all__ = [
    "AlpacaOrderPayload",
    "AlpacaTradingClient",
    "build_order_request",
]
