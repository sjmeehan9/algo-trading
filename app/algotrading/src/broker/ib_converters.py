"""Interactive Brokers conversion helpers.

This module translates between broker-agnostic models and IB API objects.
"""

from __future__ import annotations

from datetime import UTC, datetime

from algotrading.src.broker.models import (
    BarData,
    ContractSpec,
    OrderSpec,
    bar_data_from_ib_bar,
    contract_spec_from_ib_contract,
    contract_spec_to_ib_fields,
    order_spec_to_ib_fields,
)
from ibapi.contract import Contract
from ibapi.order import Order


def contract_to_ib(spec: ContractSpec) -> Contract:
    """Convert a broker-agnostic contract into an IB `Contract` object.

    Args:
        spec: Broker-agnostic contract definition.

    Returns:
        Populated IB contract.
    """

    payload = contract_spec_to_ib_fields(spec)
    contract = Contract()
    for key, value in payload.items():
        setattr(contract, key, value)
    return contract


def ib_to_contract(contract: Contract) -> ContractSpec:
    """Convert an IB `Contract` object into a broker-agnostic contract.

    Args:
        contract: IB contract object.

    Returns:
        Broker-agnostic contract definition.
    """

    return contract_spec_from_ib_contract(contract)


def order_to_ib(spec: OrderSpec, order_id: int) -> Order:
    """Convert a broker-agnostic order spec into an IB `Order` object.

    Args:
        spec: Broker-agnostic order specification.
        order_id: IB order identifier.

    Returns:
        Populated IB order object.
    """

    payload = order_spec_to_ib_fields(spec)
    order = Order()
    for key, value in payload.items():
        setattr(order, key, value)

    order.orderId = order_id
    order.eTradeOnly = False
    order.firmQuoteOnly = False
    return order


def bar_from_ib(bar: object, timestamp: datetime | None = None) -> BarData:
    """Convert an IB bar payload into `BarData`.

    Args:
        bar: IB bar object or mapping-like payload.
        timestamp: Explicit timestamp override. If absent, parse from IB payload.

    Returns:
        Normalized broker-agnostic bar data.
    """

    effective_timestamp = timestamp
    if effective_timestamp is not None and effective_timestamp.tzinfo is None:
        effective_timestamp = effective_timestamp.replace(tzinfo=UTC)
    return bar_data_from_ib_bar(bar=bar, timestamp=effective_timestamp)


__all__ = [
    "bar_from_ib",
    "contract_to_ib",
    "ib_to_contract",
    "order_to_ib",
]
