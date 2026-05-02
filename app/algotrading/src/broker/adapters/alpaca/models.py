"""Conversion helpers between Alpaca SDK payloads and broker models."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Mapping

from algotrading.src.broker.models import (
    AccountInfo,
    BarData,
    ContractSpec,
    InstrumentType,
    OrderSpec,
    OrderStatus,
    PositionInfo,
)

_MISSING = object()


def alpaca_bar_to_bar_data(bar: object) -> BarData:
    """Convert an Alpaca bar payload into canonical ``BarData``."""

    timestamp = _parse_timestamp(_read_field(bar, "timestamp", "t"))
    trade_count = _optional_int(_read_field(bar, "trade_count", "n", default=None))
    return BarData(
        timestamp=timestamp,
        open=float(_read_field(bar, "open", "o")),
        high=float(_read_field(bar, "high", "h")),
        low=float(_read_field(bar, "low", "l")),
        close=float(_read_field(bar, "close", "c")),
        volume=int(float(_read_field(bar, "volume", "v", default=0))),
        vwap=_optional_float(_read_field(bar, "vwap", "vw", default=None)),
        trade_count=trade_count,
    )


def contract_to_alpaca_symbol(contract: ContractSpec) -> str:
    """Return the Alpaca symbol representation for a contract."""

    symbol = contract.symbol.strip().upper()
    if contract.instrument_type in {InstrumentType.STOCK, InstrumentType.INDEX}:
        return symbol
    if contract.instrument_type == InstrumentType.CRYPTO:
        if "/" in symbol:
            return symbol
        return f"{symbol}/{contract.currency.strip().upper() or 'USD'}"
    if contract.instrument_type == InstrumentType.OPTION:
        return _contract_to_occ_option_symbol(contract)
    if contract.instrument_type == InstrumentType.FUTURE:
        raise ValueError("Alpaca adapter does not support futures contracts")
    raise ValueError(f"Unsupported Alpaca contract type: {contract.instrument_type}")


def alpaca_account_to_account_info(account: object) -> AccountInfo:
    """Convert an Alpaca account payload into ``AccountInfo``."""

    account_id = str(
        _read_field(account, "account_number", "id", "account_id", default="ALPACA")
    )
    return AccountInfo(
        account_id=account_id,
        cash_balance=float(_read_field(account, "cash", default=0.0)),
        buying_power=float(_read_field(account, "buying_power", default=0.0)),
        currency=str(_read_field(account, "currency", default="USD")),
    )


def alpaca_position_to_position_info(position: object) -> PositionInfo:
    """Convert an Alpaca position payload into ``PositionInfo``."""

    symbol = str(_read_field(position, "symbol"))
    asset_class = str(_read_field(position, "asset_class", default="us_equity"))
    instrument_type = (
        InstrumentType.CRYPTO
        if asset_class.lower() == "crypto"
        else InstrumentType.STOCK
    )
    contract = ContractSpec(
        symbol=symbol,
        instrument_type=instrument_type,
        exchange="PAXOS" if instrument_type == InstrumentType.CRYPTO else "SMART",
        currency="USD",
    )
    return PositionInfo(
        contract=contract,
        quantity=float(_read_field(position, "qty", "quantity", default=0.0)),
        average_cost=float(_read_field(position, "avg_entry_price", default=0.0)),
        market_value=_optional_float(
            _read_field(position, "market_value", default=None)
        ),
        unrealized_pnl=_optional_float(
            _read_field(position, "unrealized_pl", "unrealized_pnl", default=None)
        ),
    )


def alpaca_order_to_order_status(
    order: object,
    original_order: OrderSpec | None = None,
) -> OrderStatus:
    """Convert an Alpaca order payload into canonical ``OrderStatus``."""

    order_id = str(_read_field(order, "id", "order_id"))
    raw_status = str(_read_field(order, "status", default="new"))
    quantity = _optional_float(_read_field(order, "qty", "quantity", default=None))
    filled = _optional_float(_read_field(order, "filled_qty", default=None))
    remaining = _optional_float(_read_field(order, "remaining_qty", default=None))

    if quantity is None and original_order is not None:
        quantity = float(original_order.quantity)
    if filled is None:
        filled = 0.0
    if remaining is None:
        remaining = max((quantity or filled) - filled, 0.0)

    avg_fill_price = _optional_float(
        _read_field(order, "filled_avg_price", "avg_fill_price", default=None)
    )
    last_update = _parse_timestamp(
        _read_field(order, "updated_at", "submitted_at", "created_at", default=None)
    )
    return OrderStatus(
        order_id=order_id,
        status=_normalize_order_status(raw_status),
        filled_quantity=filled,
        remaining_quantity=remaining,
        average_fill_price=avg_fill_price,
        last_update=last_update,
    )


def _contract_to_occ_option_symbol(contract: ContractSpec) -> str:
    if not contract.expiry or contract.strike is None or not contract.right:
        raise ValueError("Option contracts require expiry, strike, and right")
    expiry = contract.expiry.strip()
    if len(expiry) == 8:
        expiry = expiry[2:]
    if len(expiry) != 6 or not expiry.isdigit():
        raise ValueError("Option expiry must be YYYYMMDD or YYMMDD")
    right = contract.right.strip().upper()
    if right not in {"C", "P"}:
        raise ValueError("Option right must be C or P")
    strike = int(Decimal(str(contract.strike)) * Decimal("1000"))
    return f"{contract.symbol.strip().upper()}{expiry}{right}{strike:08d}"


def _normalize_order_status(raw_status: str) -> str:
    normalized = raw_status.strip().upper().replace("-", "_")
    status_map = {
        "ACCEPTED": "SUBMITTED",
        "ACCEPTED_FOR_BIDDING": "SUBMITTED",
        "CALCULATED": "SUBMITTED",
        "DONE_FOR_DAY": "SUBMITTED",
        "EXPIRED": "CANCELLED",
        "FILLED": "FILLED",
        "NEW": "SUBMITTED",
        "PARTIALLY_FILLED": "PARTIAL",
        "PENDING_CANCEL": "SUBMITTED",
        "PENDING_NEW": "PENDING",
        "PENDING_REPLACE": "SUBMITTED",
        "REJECTED": "REJECTED",
        "REPLACED": "SUBMITTED",
        "STOPPED": "SUBMITTED",
        "SUSPENDED": "REJECTED",
        "CANCELED": "CANCELLED",
        "CANCELLED": "CANCELLED",
    }
    return status_map.get(normalized, "SUBMITTED")


def _parse_timestamp(value: object) -> datetime:
    if value is None:
        return datetime.now(tz=UTC)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if hasattr(value, "to_pydatetime"):
        return _parse_timestamp(value.to_pydatetime())
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=UTC)

    raw = str(value).strip()
    if raw.isdigit():
        return datetime.fromtimestamp(int(raw), tz=UTC)
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _read_field(
    payload: Mapping[str, object] | object,
    *keys: str,
    default: object = _MISSING,
) -> object:
    if isinstance(payload, Mapping):
        for key in keys:
            if key in payload:
                return payload[key]
        if default is not _MISSING:
            return default
        raise KeyError(f"missing required field(s): {', '.join(keys)}")

    for key in keys:
        if hasattr(payload, key):
            return getattr(payload, key)
    if default is not _MISSING:
        return default
    raise KeyError(f"missing required attribute(s): {', '.join(keys)}")


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, InvalidOperation) as exc:
        raise ValueError(f"cannot convert {value!r} to float") from exc


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


__all__ = [
    "alpaca_account_to_account_info",
    "alpaca_bar_to_bar_data",
    "alpaca_order_to_order_status",
    "alpaca_position_to_position_info",
    "contract_to_alpaca_symbol",
]
