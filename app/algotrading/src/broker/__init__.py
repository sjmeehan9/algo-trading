"""Broker abstraction models and utilities."""

from algotrading.src.broker.adapter import (
    AccountCallback,
    BrokerAdapter,
    DataCallback,
    OrderCallback,
    PositionCallback,
)
from algotrading.src.broker.exceptions import (
    BrokerConnectionError,
    BrokerDataError,
    BrokerError,
    BrokerOrderError,
    BrokerTimeoutError,
)
from algotrading.src.broker.models import (
    AccountInfo,
    BarData,
    BarLike,
    ContractLike,
    ContractSpec,
    InstrumentType,
    OrderLike,
    OrderSide,
    OrderSpec,
    OrderStatus,
    OrderType,
    PositionInfo,
    bar_data_from_ib_bar,
    contract_spec_from_ib_contract,
    contract_spec_to_ib_fields,
    order_spec_to_ib_fields,
    order_status_from_ib_fields,
    parse_ib_timestamp,
)

ContractList = list[ContractSpec]
BarSeries = list[BarData]
PositionList = list[PositionInfo]

__all__ = [
    "AccountCallback",
    "AccountInfo",
    "BarData",
    "BarLike",
    "BarSeries",
    "BrokerAdapter",
    "BrokerConnectionError",
    "BrokerDataError",
    "BrokerError",
    "BrokerOrderError",
    "BrokerTimeoutError",
    "ContractLike",
    "ContractList",
    "ContractSpec",
    "DataCallback",
    "InstrumentType",
    "OrderCallback",
    "OrderLike",
    "OrderSide",
    "OrderSpec",
    "OrderStatus",
    "OrderType",
    "PositionCallback",
    "PositionInfo",
    "PositionList",
    "bar_data_from_ib_bar",
    "contract_spec_from_ib_contract",
    "contract_spec_to_ib_fields",
    "order_spec_to_ib_fields",
    "order_status_from_ib_fields",
    "parse_ib_timestamp",
]
