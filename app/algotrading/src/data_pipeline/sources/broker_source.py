"""Broker-backed data source implementation for market bar retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from queue import Empty, Queue
from typing import Iterator

from algotrading.src.broker import (
    BarData,
    BrokerAdapter,
    ContractSpec,
    InstrumentType,
)
from algotrading.src.data_pipeline.sources.base import DataSource
from algotrading.src.data_pipeline.sources.exceptions import (
    DataSourceConnectionError,
    DataSourceError,
    DataValidationError,
)
from algotrading.src.data_pipeline.types import (
    DataBatch,
    DataFrequency,
    DataRecord,
    DataType,
    SourceMetadata,
)


@dataclass(frozen=True, slots=True)
class _ContractDefaults:
    """Default contract attributes used for symbol-only requests."""

    instrument_type: InstrumentType = InstrumentType.STOCK
    exchange: str = "SMART"
    currency: str = "USD"
    primary_exchange: str | None = None


def contract_spec_from_config(config: dict[str, object]) -> ContractSpec:
    """Build a ``ContractSpec`` from a contract-style config dictionary.

    Args:
        config: Contract config payload supporting either pipeline field names
            (for example, ``secType``, ``primaryExchange``) or canonical names
            (for example, ``instrument_type``, ``primary_exchange``).

    Returns:
        A validated broker-agnostic ``ContractSpec``.
    """

    symbol = str(config.get("symbol", "")).strip()
    if not symbol:
        raise ValueError("Contract config must include non-empty 'symbol'")

    raw_type = str(
        config.get("instrument_type")
        or config.get("secType")
        or InstrumentType.STOCK.value
    ).upper()
    type_map = {
        "STK": InstrumentType.STOCK,
        "STOCK": InstrumentType.STOCK,
        "OPT": InstrumentType.OPTION,
        "OPTION": InstrumentType.OPTION,
        "FUT": InstrumentType.FUTURE,
        "FUTURE": InstrumentType.FUTURE,
        "CRYPTO": InstrumentType.CRYPTO,
        "CRYPTOCURRENCY": InstrumentType.CRYPTO,
        "IND": InstrumentType.INDEX,
        "INDEX": InstrumentType.INDEX,
    }
    instrument_type = type_map.get(raw_type)
    if instrument_type is None:
        raise ValueError(f"Unsupported instrument type in config: {raw_type}")

    exchange = str(config.get("exchange") or "SMART").strip()
    currency = str(config.get("currency") or "USD").strip()
    primary_exchange_value = config.get("primary_exchange") or config.get(
        "primaryExchange"
    )
    primary_exchange = (
        str(primary_exchange_value).strip() if primary_exchange_value else None
    )

    return ContractSpec(
        symbol=symbol,
        instrument_type=instrument_type,
        exchange=exchange,
        currency=currency,
        primary_exchange=primary_exchange,
    )


def bar_to_record(
    bar: BarData,
    symbol: str,
    source_id: str,
    frequency: DataFrequency,
) -> DataRecord:
    """Convert broker ``BarData`` into pipeline ``DataRecord``.

    Args:
        bar: Broker bar payload.
        symbol: Bar symbol.
        source_id: Source identifier to stamp in the record.
        frequency: Pipeline frequency for this bar stream.

    Returns:
        Immutable pipeline record with normalized market payload fields.
    """

    return DataRecord(
        timestamp=bar.timestamp,
        data_type=DataType.MARKET_BAR,
        symbol=symbol,
        payload={
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "vwap": bar.vwap,
            "trade_count": bar.trade_count,
        },
        source_id=source_id,
        frequency=frequency,
    )


class BrokerDataSource(DataSource):
    """Data source that retrieves market bars through a ``BrokerAdapter``.

    Args:
        adapter: Broker adapter implementation.
        default_symbol: Optional default symbol used by stream callers.
        host: Optional host used when establishing adapter connection.
        port: Optional port used when establishing adapter connection.
        client_id: Optional client id used when establishing adapter connection.
        historical_bar_size: Broker bar-size string for historical requests.
        realtime_bar_size_seconds: Realtime bar interval in seconds.
        broker_data_type: Broker-native data type (for example, ``TRADES``).
        contract_defaults: Optional default contract attributes.
        historical_frequency: Frequency assigned to historical records.
        stream_frequency: Frequency assigned to streamed records.
        stream_queue_timeout_seconds: Queue timeout while awaiting streamed bars.
    """

    def __init__(
        self,
        adapter: BrokerAdapter,
        default_symbol: str | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        historical_bar_size: str = "5 secs",
        realtime_bar_size_seconds: int = 5,
        broker_data_type: str = "TRADES",
        contract_defaults: _ContractDefaults | None = None,
        historical_frequency: DataFrequency = DataFrequency.SECOND_5,
        stream_frequency: DataFrequency = DataFrequency.SECOND_5,
        stream_queue_timeout_seconds: float = 1.0,
    ) -> None:
        self._adapter = adapter
        self._default_symbol = default_symbol
        self._host = host
        self._port = port
        self._client_id = client_id
        self._historical_bar_size = historical_bar_size
        self._realtime_bar_size_seconds = realtime_bar_size_seconds
        self._broker_data_type = broker_data_type
        self._contract_defaults = contract_defaults or _ContractDefaults()
        self._historical_frequency = historical_frequency
        self._stream_frequency = stream_frequency
        self._stream_queue_timeout_seconds = stream_queue_timeout_seconds

    @property
    def source_id(self) -> str:
        """Return stable source identifier derived from the adapter type."""

        return f"broker_{self._adapter.__class__.__name__}"

    @property
    def metadata(self) -> SourceMetadata:
        """Return broker source capabilities and runtime configuration."""

        return SourceMetadata(
            source_id=self.source_id,
            source_type="broker",
            supported_types=[DataType.MARKET_BAR],
            supported_frequencies=[
                DataFrequency.SECOND_1,
                DataFrequency.SECOND_5,
                DataFrequency.SECOND_10,
                DataFrequency.SECOND_30,
                DataFrequency.MINUTE_1,
                DataFrequency.MINUTE_5,
                DataFrequency.MINUTE_15,
                DataFrequency.HOUR_1,
                DataFrequency.DAY_1,
            ],
            config={
                "historical_bar_size": self._historical_bar_size,
                "realtime_bar_size_seconds": self._realtime_bar_size_seconds,
                "broker_data_type": self._broker_data_type,
                "default_symbol": self._default_symbol,
            },
        )

    @property
    def is_connected(self) -> bool:
        """Return current adapter connection state."""

        return self._adapter.is_connected()

    def connect(self) -> None:
        """Connect the adapter if it is not already connected."""

        if self._adapter.is_connected():
            return

        if self._host is None or self._port is None or self._client_id is None:
            raise DataSourceConnectionError(
                "BrokerDataSource requires host, port, and client_id to connect "
                "when adapter is not already connected"
            )

        try:
            self._adapter.connect(self._host, self._port, self._client_id)
        except Exception as exc:
            raise DataSourceConnectionError(
                f"Failed to connect broker adapter: {exc}"
            ) from exc

    def disconnect(self) -> None:
        """Disconnect the adapter and release broker resources."""

        if not self._adapter.is_connected():
            return

        try:
            self._adapter.disconnect()
        except Exception as exc:
            raise DataSourceConnectionError(
                f"Failed to disconnect broker adapter: {exc}"
            ) from exc

    def fetch_batch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_type: DataType = DataType.MARKET_BAR,
    ) -> DataBatch:
        """Fetch historical bars and convert them to a pipeline ``DataBatch``."""

        self._ensure_connected()
        if data_type != DataType.MARKET_BAR:
            raise DataValidationError(
                "BrokerDataSource currently supports DataType.MARKET_BAR only"
            )
        if start > end:
            raise DataValidationError("start must be less than or equal to end")

        contract = self._contract_for_symbol(symbol)
        duration = self._duration_string(start=start, end=end)

        try:
            bars = self._adapter.request_historical_data(
                contract=contract,
                end_datetime=end,
                duration=duration,
                bar_size=self._historical_bar_size,
                data_type=self._broker_data_type,
            )
        except Exception as exc:
            raise DataSourceError(f"Failed historical data request: {exc}") from exc

        records = [
            bar_to_record(
                bar=bar,
                symbol=symbol,
                source_id=self.source_id,
                frequency=self._historical_frequency,
            )
            for bar in bars
            if start <= bar.timestamp <= end
        ]

        records.sort(key=lambda record: record.timestamp)
        if not records:
            return DataBatch(
                records=[],
                start_time=start,
                end_time=end,
                data_type=DataType.MARKET_BAR,
                symbol=symbol,
            )

        return DataBatch(
            records=records,
            start_time=records[0].timestamp,
            end_time=records[-1].timestamp,
            data_type=DataType.MARKET_BAR,
            symbol=symbol,
        )

    def fetch_stream(
        self,
        symbol: str,
        data_type: DataType = DataType.MARKET_BAR,
    ) -> Iterator[DataRecord]:
        """Subscribe to real-time bars and yield records as they arrive."""

        self._ensure_connected()
        if data_type != DataType.MARKET_BAR:
            raise DataValidationError(
                "BrokerDataSource currently supports DataType.MARKET_BAR only"
            )

        resolved_symbol = symbol or self._default_symbol
        if not resolved_symbol:
            raise DataValidationError(
                "symbol must be provided when no default_symbol is configured"
            )

        contract = self._contract_for_symbol(resolved_symbol)
        stream_queue: Queue[BarData] = Queue()

        def _on_bar(bar: BarData) -> None:
            stream_queue.put(bar)

        try:
            subscription_id = self._adapter.subscribe_realtime_data(
                contract=contract,
                bar_size=self._realtime_bar_size_seconds,
                data_type=self._broker_data_type,
                callback=_on_bar,
            )
        except Exception as exc:
            raise DataSourceError(
                f"Failed realtime subscription request: {exc}"
            ) from exc

        def _record_iterator() -> Iterator[DataRecord]:
            try:
                while True:
                    try:
                        bar = stream_queue.get(
                            timeout=self._stream_queue_timeout_seconds
                        )
                    except Empty:
                        if not self._adapter.is_connected():
                            break
                        continue

                    yield bar_to_record(
                        bar=bar,
                        symbol=resolved_symbol,
                        source_id=self.source_id,
                        frequency=self._stream_frequency,
                    )
            finally:
                try:
                    self._adapter.unsubscribe_realtime_data(subscription_id)
                except Exception as exc:
                    raise DataSourceError(
                        f"Failed to unsubscribe realtime feed: {exc}"
                    ) from exc

        return _record_iterator()

    def get_available_symbols(self) -> list[str]:
        """Return broker symbols known to this source configuration."""

        if self._default_symbol:
            return [self._default_symbol]
        return []

    def get_available_dates(self, symbol: str) -> list[date]:
        """Return available dates for a broker source (on-demand only)."""

        del symbol
        return []

    def _ensure_connected(self) -> None:
        if not self._adapter.is_connected():
            raise DataSourceConnectionError(
                "BrokerDataSource is not connected. Call connect() first."
            )

    def _contract_for_symbol(self, symbol: str) -> ContractSpec:
        cleaned_symbol = symbol.strip()
        if not cleaned_symbol:
            raise DataValidationError("symbol must be non-empty")

        return ContractSpec(
            symbol=cleaned_symbol,
            instrument_type=self._contract_defaults.instrument_type,
            exchange=self._contract_defaults.exchange,
            currency=self._contract_defaults.currency,
            primary_exchange=self._contract_defaults.primary_exchange,
        )

    def _duration_string(self, start: datetime, end: datetime) -> str:
        start_utc = (
            start.replace(tzinfo=UTC) if start.tzinfo is None else start.astimezone(UTC)
        )
        end_utc = end.replace(tzinfo=UTC) if end.tzinfo is None else end.astimezone(UTC)
        delta_seconds = int((end_utc - start_utc).total_seconds())
        bounded_seconds = max(1, delta_seconds)
        return f"{bounded_seconds} S"


__all__ = [
    "BrokerDataSource",
    "bar_to_record",
    "contract_spec_from_config",
]
