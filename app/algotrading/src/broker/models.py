"""Broker-agnostic trading data models.

These models define the canonical data structures exchanged between the
application and broker adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Mapping, TypeAlias
from zoneinfo import ZoneInfo


class InstrumentType(str, Enum):
    """Supported instrument classes across broker adapters."""

    STOCK = "STOCK"
    OPTION = "OPTION"
    FUTURE = "FUTURE"
    CRYPTO = "CRYPTO"
    INDEX = "INDEX"


class OrderSide(str, Enum):
    """Supported order sides."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Supported order execution types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


ContractLike: TypeAlias = Mapping[str, object] | object
OrderLike: TypeAlias = Mapping[str, object] | object
BarLike: TypeAlias = Mapping[str, object] | object
_MISSING = object()


@dataclass(frozen=True)
class ContractSpec:
    """Broker-agnostic contract definition.

    Args:
        symbol: Ticker or instrument symbol.
        instrument_type: Instrument classification.
        exchange: Routing exchange or venue.
        currency: Settlement currency.
        primary_exchange: Primary listing exchange, when applicable.
        expiry: Expiry date/contract month (options/futures).
        strike: Strike price for options.
        right: Option right, ``"C"`` for call or ``"P"`` for put.
        multiplier: Contract multiplier.
    """

    symbol: str
    instrument_type: InstrumentType
    exchange: str
    currency: str
    primary_exchange: str | None = None
    expiry: str | None = None
    strike: float | None = None
    right: str | None = None
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        """Validate contract invariants."""

        if not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not self.exchange.strip():
            raise ValueError("exchange must be non-empty")
        if not self.currency.strip():
            raise ValueError("currency must be non-empty")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be greater than zero")

        if self.instrument_type == InstrumentType.OPTION:
            if not self.expiry:
                raise ValueError("expiry is required for option contracts")
            if self.strike is None or self.strike <= 0:
                raise ValueError("strike must be greater than zero for options")
            if self.right not in {"C", "P"}:
                raise ValueError("right must be 'C' or 'P' for options")
        elif self.instrument_type == InstrumentType.FUTURE:
            if not self.expiry:
                raise ValueError("expiry is required for futures contracts")
            if self.right is not None or self.strike is not None:
                raise ValueError("futures contracts cannot define option strike/right")
        else:
            if (
                self.right is not None
                or self.strike is not None
                or self.expiry is not None
            ):
                raise ValueError(
                    "expiry/strike/right are only supported for option and future contracts"
                )

    def to_dict(self) -> dict[str, object]:
        """Serialize contract to a JSON-compatible dictionary."""

        payload = asdict(self)
        payload["instrument_type"] = self.instrument_type.value
        return payload


@dataclass(frozen=True)
class BarData:
    """OHLCV bar representation normalized across brokers.

    Args:
        timestamp: Bar timestamp. Must be timezone-aware.
        open: Open price.
        high: High price.
        low: Low price.
        close: Close price.
        volume: Traded volume.
        vwap: Optional volume-weighted average price.
        trade_count: Optional trade count for the interval.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    trade_count: int | None = None

    def __post_init__(self) -> None:
        """Validate OHLCV data consistency."""

        _ensure_timezone_aware(self.timestamp)
        if self.volume < 0:
            raise ValueError("volume must be non-negative")
        if self.trade_count is not None and self.trade_count < 0:
            raise ValueError("trade_count must be non-negative")
        if self.low > self.high:
            raise ValueError("low cannot be greater than high")
        if not (self.low <= self.open <= self.high):
            raise ValueError("open must be between low and high")
        if not (self.low <= self.close <= self.high):
            raise ValueError("close must be between low and high")
        if self.vwap is not None and self.vwap < 0:
            raise ValueError("vwap must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Serialize bar to a JSON-compatible dictionary."""

        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass
class OrderSpec:
    """Broker-agnostic order request.

    Args:
        side: Buy or sell side.
        quantity: Quantity to trade.
        order_type: Order type.
        limit_price: Limit price for limit orders.
        stop_price: Stop trigger for stop orders.
        time_in_force: Time in force (for example, ``DAY`` or ``GTC``).
    """

    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "DAY"

    def __post_init__(self) -> None:
        """Validate order invariants."""

        if self.quantity <= 0:
            raise ValueError("quantity must be greater than zero")
        if not self.time_in_force.strip():
            raise ValueError("time_in_force must be non-empty")

        if self.order_type == OrderType.MARKET:
            if self.limit_price is not None or self.stop_price is not None:
                raise ValueError("market orders cannot define limit or stop prices")
            return

        if self.order_type == OrderType.LIMIT:
            if self.limit_price is None or self.limit_price <= 0:
                raise ValueError("limit orders require limit_price > 0")
            if self.stop_price is not None:
                raise ValueError("limit orders cannot define stop_price")
            return

        if self.order_type == OrderType.STOP:
            if self.stop_price is None or self.stop_price <= 0:
                raise ValueError("stop orders require stop_price > 0")
            if self.limit_price is not None:
                raise ValueError("stop orders cannot define limit_price")
            return

        if self.limit_price is None or self.limit_price <= 0:
            raise ValueError("stop limit orders require limit_price > 0")
        if self.stop_price is None or self.stop_price <= 0:
            raise ValueError("stop limit orders require stop_price > 0")

    def to_dict(self) -> dict[str, object]:
        """Serialize order request to a JSON-compatible dictionary."""

        return {
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
        }


@dataclass(frozen=True)
class OrderStatus:
    """Current order state and fill progress.

    Args:
        order_id: Broker-assigned order identifier.
        status: Lifecycle status.
        filled_quantity: Filled quantity.
        remaining_quantity: Remaining quantity.
        average_fill_price: Average executed fill price.
        last_update: Last broker update timestamp.
    """

    order_id: str
    status: str
    filled_quantity: float
    remaining_quantity: float
    average_fill_price: float | None
    last_update: datetime

    def __post_init__(self) -> None:
        """Validate order status payload."""

        if not self.order_id.strip():
            raise ValueError("order_id must be non-empty")
        valid_statuses = {
            "PENDING",
            "SUBMITTED",
            "PARTIAL",
            "FILLED",
            "CANCELLED",
            "REJECTED",
        }
        if self.status not in valid_statuses:
            raise ValueError(f"unsupported order status: {self.status}")
        if self.filled_quantity < 0 or self.remaining_quantity < 0:
            raise ValueError(
                "filled_quantity and remaining_quantity must be non-negative"
            )
        if self.average_fill_price is not None and self.average_fill_price < 0:
            raise ValueError("average_fill_price must be non-negative")
        _ensure_timezone_aware(self.last_update)

    def to_dict(self) -> dict[str, object]:
        """Serialize order status to a JSON-compatible dictionary."""

        payload = asdict(self)
        payload["last_update"] = self.last_update.isoformat()
        return payload


@dataclass(frozen=True)
class AccountInfo:
    """Snapshot of account-level financial information.

    Args:
        account_id: Account identifier.
        cash_balance: Available cash balance.
        buying_power: Buying power available.
        currency: Account base currency.
    """

    account_id: str
    cash_balance: float
    buying_power: float
    currency: str

    def __post_init__(self) -> None:
        """Validate account payload."""

        if not self.account_id.strip():
            raise ValueError("account_id must be non-empty")
        if self.cash_balance < 0:
            raise ValueError("cash_balance must be non-negative")
        if self.buying_power < 0:
            raise ValueError("buying_power must be non-negative")
        if not self.currency.strip():
            raise ValueError("currency must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Serialize account info to a JSON-compatible dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class PositionInfo:
    """Position-level holdings information.

    Args:
        contract: Position contract.
        quantity: Position quantity (negative for short).
        average_cost: Average open cost.
        market_value: Current mark-to-market position value.
        unrealized_pnl: Unrealized profit and loss.
    """

    contract: ContractSpec
    quantity: float
    average_cost: float
    market_value: float | None = None
    unrealized_pnl: float | None = None

    def __post_init__(self) -> None:
        """Validate position payload."""

        if self.average_cost < 0:
            raise ValueError("average_cost must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Serialize position info to a JSON-compatible dictionary."""

        return {
            "contract": self.contract.to_dict(),
            "quantity": self.quantity,
            "average_cost": self.average_cost,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
        }


_INSTRUMENT_TYPE_TO_IB = {
    InstrumentType.STOCK: "STK",
    InstrumentType.OPTION: "OPT",
    InstrumentType.FUTURE: "FUT",
    InstrumentType.CRYPTO: "CRYPTO",
    InstrumentType.INDEX: "IND",
}

_IB_TO_INSTRUMENT_TYPE = {
    "STK": InstrumentType.STOCK,
    "OPT": InstrumentType.OPTION,
    "FUT": InstrumentType.FUTURE,
    "CRYPTO": InstrumentType.CRYPTO,
    "IND": InstrumentType.INDEX,
    "INDEX": InstrumentType.INDEX,
}

_ORDER_TYPE_TO_IB = {
    OrderType.MARKET: "MKT",
    OrderType.LIMIT: "LMT",
    OrderType.STOP: "STP",
    OrderType.STOP_LIMIT: "STP LMT",
}


def contract_spec_to_ib_fields(spec: ContractSpec) -> dict[str, object]:
    """Convert a `ContractSpec` into an IB-compatible field dictionary.

    Args:
        spec: Contract specification.

    Returns:
        IB contract field mapping.
    """

    payload: dict[str, object] = {
        "symbol": spec.symbol,
        "secType": _INSTRUMENT_TYPE_TO_IB[spec.instrument_type],
        "exchange": spec.exchange,
        "currency": spec.currency,
        "multiplier": str(spec.multiplier),
    }
    if spec.primary_exchange is not None:
        payload["primaryExchange"] = spec.primary_exchange
    if spec.expiry is not None:
        payload["lastTradeDateOrContractMonth"] = spec.expiry
    if spec.strike is not None:
        payload["strike"] = spec.strike
    if spec.right is not None:
        payload["right"] = spec.right
    return payload


def contract_spec_from_ib_contract(contract: ContractLike) -> ContractSpec:
    """Convert an IB contract object or mapping into `ContractSpec`.

    Args:
        contract: IB contract object or dict-like representation.

    Returns:
        Broker-agnostic contract specification.
    """

    sec_type = str(_read_field(contract, "secType", "sectype")).upper()
    if sec_type not in _IB_TO_INSTRUMENT_TYPE:
        raise ValueError(f"unsupported IB secType: {sec_type}")

    expiry = _optional_str(
        _read_field(contract, "lastTradeDateOrContractMonth", default=None)
    )
    strike_raw = _read_field(contract, "strike", default=None)
    strike = float(strike_raw) if strike_raw not in (None, "") else None
    if sec_type != "OPT" and strike == 0.0:
        strike = None
    right = _optional_str(_read_field(contract, "right", default=None))
    multiplier_raw = _read_field(contract, "multiplier", default=None)
    multiplier = float(multiplier_raw) if multiplier_raw not in (None, "") else 1.0

    return ContractSpec(
        symbol=str(_read_field(contract, "symbol")),
        instrument_type=_IB_TO_INSTRUMENT_TYPE[sec_type],
        exchange=str(_read_field(contract, "exchange")),
        currency=str(_read_field(contract, "currency")),
        primary_exchange=_optional_str(
            _read_field(contract, "primaryExchange", default=None)
        ),
        expiry=expiry,
        strike=strike,
        right=right,
        multiplier=multiplier,
    )


def order_spec_to_ib_fields(spec: OrderSpec) -> dict[str, object]:
    """Convert an `OrderSpec` into an IB-compatible order field dictionary.

    Args:
        spec: Broker-agnostic order specification.

    Returns:
        IB order field mapping.
    """

    payload: dict[str, object] = {
        "action": spec.side.value,
        "totalQuantity": spec.quantity,
        "orderType": _ORDER_TYPE_TO_IB[spec.order_type],
        "tif": spec.time_in_force,
    }
    if spec.limit_price is not None:
        payload["lmtPrice"] = spec.limit_price
    if spec.stop_price is not None:
        payload["auxPrice"] = spec.stop_price
    return payload


def order_status_from_ib_fields(status_payload: OrderLike) -> OrderStatus:
    """Convert IB order status payload into `OrderStatus`.

    Args:
        status_payload: IB order status object or mapping.

    Returns:
        Normalized order status model.
    """

    raw_status = str(_read_field(status_payload, "status")).upper()
    normalized_status = _normalize_order_status(raw_status)
    filled_raw = _read_field(status_payload, "filled", "filledQuantity")
    remaining_raw = _read_field(status_payload, "remaining", "remainingQuantity")
    avg_fill_raw = _read_field(status_payload, "avgFillPrice", "averageFillPrice")
    timestamp_raw = _read_field(status_payload, "lastUpdate", "timestamp")
    last_update = (
        parse_ib_timestamp(timestamp_raw)
        if timestamp_raw is not None
        else datetime.now(tz=UTC)
    )

    return OrderStatus(
        order_id=str(_read_field(status_payload, "orderId", "order_id")),
        status=normalized_status,
        filled_quantity=float(filled_raw or 0.0),
        remaining_quantity=float(remaining_raw or 0.0),
        average_fill_price=(
            float(avg_fill_raw) if avg_fill_raw not in (None, "") else None
        ),
        last_update=last_update,
    )


def bar_data_from_ib_bar(bar: BarLike, timestamp: datetime | None = None) -> BarData:
    """Convert an IB bar payload into `BarData`.

    Args:
        bar: IB bar object or mapping.
        timestamp: Explicit timestamp override.

    Returns:
        Normalized market bar data.
    """

    bar_timestamp = timestamp or parse_ib_timestamp(_read_field(bar, "date", "time"))

    trade_count_raw = _read_field(bar, "barCount", "trade_count")
    return BarData(
        timestamp=bar_timestamp,
        open=float(_read_field(bar, "open")),
        high=float(_read_field(bar, "high")),
        low=float(_read_field(bar, "low")),
        close=float(_read_field(bar, "close")),
        volume=int(float(_read_field(bar, "volume", default=0))),
        vwap=_optional_float(_read_field(bar, "wap", "vwap")),
        trade_count=(
            int(float(trade_count_raw)) if trade_count_raw not in (None, "") else None
        ),
    )


def parse_ib_timestamp(value: object) -> datetime:
    """Parse a timestamp value returned by IB into a timezone-aware datetime.

    Args:
        value: Datetime, epoch integer/string, or IB formatted string.

    Returns:
        Parsed timezone-aware datetime in UTC when timezone info is absent.
    """

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    if value is None:
        raise ValueError("timestamp value cannot be None")

    if isinstance(value, int):
        return datetime.fromtimestamp(value, tz=UTC)

    raw = str(value).strip()
    if raw.isdigit():
        return datetime.fromtimestamp(int(raw), tz=UTC)

    if " " in raw:
        datetime_part, tz_name = raw.rsplit(" ", maxsplit=1)
        if "/" in tz_name:
            for candidate in (
                "%Y%m%d %H:%M:%S",
                "%Y%m%d  %H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    parsed = datetime.strptime(datetime_part, candidate)
                    return parsed.replace(tzinfo=ZoneInfo(tz_name))
                except ValueError:
                    continue

    for candidate in ("%Y%m%d %H:%M:%S", "%Y%m%d  %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(raw, candidate)
            return parsed.replace(tzinfo=UTC)
        except ValueError:
            continue

    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"unsupported IB timestamp format: {value!r}") from exc

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _normalize_order_status(raw_status: str) -> str:
    status_map = {
        "PENDINGSUBMIT": "PENDING",
        "PRESUBMITTED": "SUBMITTED",
        "SUBMITTED": "SUBMITTED",
        "PARTIALLYFILLED": "PARTIAL",
        "FILLED": "FILLED",
        "CANCELLED": "CANCELLED",
        "INACTIVE": "REJECTED",
        "API CANCELLED": "CANCELLED",
        "REJECTED": "REJECTED",
    }
    if raw_status in status_map:
        return status_map[raw_status]
    if raw_status == "PARTIAL":
        return "PARTIAL"
    raise ValueError(f"unsupported IB order status value: {raw_status}")


def _ensure_timezone_aware(value: datetime) -> None:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("timestamp must be timezone-aware")


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


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


__all__ = [
    "AccountInfo",
    "BarData",
    "BarLike",
    "ContractLike",
    "ContractSpec",
    "InstrumentType",
    "OrderLike",
    "OrderSide",
    "OrderSpec",
    "OrderStatus",
    "OrderType",
    "PositionInfo",
    "bar_data_from_ib_bar",
    "contract_spec_from_ib_contract",
    "contract_spec_to_ib_fields",
    "order_spec_to_ib_fields",
    "order_status_from_ib_fields",
    "parse_ib_timestamp",
]
