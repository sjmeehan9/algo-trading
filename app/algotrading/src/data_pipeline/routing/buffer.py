"""Multi-frequency buffering primitives for timestamp-aligned data access."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock, RLock

from algotrading.src.data_pipeline.types import DataFrequency, DataRecord, DataType


@dataclass(frozen=True, slots=True)
class ChannelConfig:
    """Configuration for a single buffer channel.

    Args:
        data_type: Data category accepted by the channel.
        frequency: Frequency accepted by the channel.
        max_size: Maximum number of records retained.
        ttl_seconds: Optional retention horizon in seconds.
    """

    data_type: DataType
    frequency: DataFrequency
    max_size: int = 1000
    ttl_seconds: int | None = None

    def __post_init__(self) -> None:
        """Validate channel configuration invariants."""

        if self.max_size <= 0:
            raise ValueError("max_size must be greater than 0")
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than 0 when provided")


@dataclass(slots=True)
class BufferConfig:
    """Configuration for a multi-channel buffer instance.

    Args:
        channels: Explicit channel definitions.
        default_max_size: Fallback max size used for dynamically added channels.
    """

    channels: list[ChannelConfig] = field(default_factory=list)
    default_max_size: int = 1000

    def __post_init__(self) -> None:
        """Validate buffer configuration."""

        if self.default_max_size <= 0:
            raise ValueError("default_max_size must be greater than 0")


class BufferChannel:
    """Thread-safe timestamp-ordered storage for a single type/frequency stream."""

    def __init__(self, config: ChannelConfig) -> None:
        """Initialise channel storage.

        Args:
            config: Channel retention and stream settings.
        """

        self.config = config
        self._records: list[DataRecord] = []
        self._timestamps: list[datetime] = []
        self._lock = RLock()

    def put(self, record: DataRecord) -> None:
        """Insert a record while preserving timestamp order.

        Args:
            record: Record to store.

        Raises:
            ValueError: If record data type/frequency does not match channel.
        """

        if record.data_type != self.config.data_type:
            raise ValueError(
                f"record data_type {record.data_type} does not match channel "
                f"{self.config.data_type}"
            )
        if record.frequency != self.config.frequency:
            raise ValueError(
                f"record frequency {record.frequency} does not match channel "
                f"{self.config.frequency}"
            )

        with self._lock:
            insert_index = bisect_right(self._timestamps, record.timestamp)
            self._timestamps.insert(insert_index, record.timestamp)
            self._records.insert(insert_index, record)
            self._prune_ttl_locked(reference_time=record.timestamp)
            self._enforce_max_size_locked()

    def get_latest(self, before: datetime | None = None) -> DataRecord | None:
        """Return the latest record optionally constrained by timestamp.

        Args:
            before: Optional inclusive timestamp constraint.

        Returns:
            Latest matching record if present, otherwise None.
        """

        with self._lock:
            if not self._records:
                return None

            if before is None:
                return self._records[-1]

            latest_index = bisect_right(self._timestamps, before) - 1
            if latest_index < 0:
                return None

            return self._records[latest_index]

    def get_range(self, start: datetime, end: datetime) -> list[DataRecord]:
        """Return records within an inclusive timestamp range.

        Args:
            start: Inclusive start timestamp.
            end: Inclusive end timestamp.

        Returns:
            Ordered records in the requested range.

        Raises:
            ValueError: If start is after end.
        """

        if start > end:
            raise ValueError("start must be less than or equal to end")

        with self._lock:
            left_index = bisect_left(self._timestamps, start)
            right_index = bisect_right(self._timestamps, end)
            return self._records[left_index:right_index]

    def clear(self) -> None:
        """Remove all records from the channel."""

        with self._lock:
            self._records.clear()
            self._timestamps.clear()

    def prune(self, before: datetime) -> int:
        """Remove records with timestamps earlier than ``before``.

        Args:
            before: Exclusive timestamp boundary.

        Returns:
            Number of removed records.
        """

        with self._lock:
            prune_count = bisect_left(self._timestamps, before)
            if prune_count <= 0:
                return 0

            del self._timestamps[:prune_count]
            del self._records[:prune_count]
            return prune_count

    def __len__(self) -> int:
        """Return retained record count."""

        with self._lock:
            return len(self._records)

    def _enforce_max_size_locked(self) -> None:
        """Enforce configured max-size constraint under lock."""

        overflow = len(self._records) - self.config.max_size
        if overflow <= 0:
            return

        del self._timestamps[:overflow]
        del self._records[:overflow]

    def _prune_ttl_locked(self, reference_time: datetime) -> None:
        """Prune records older than channel TTL under lock.

        Args:
            reference_time: Timestamp anchor used to evaluate record age.
        """

        if self.config.ttl_seconds is None:
            return

        cutoff = reference_time - timedelta(seconds=self.config.ttl_seconds)
        prune_count = bisect_left(self._timestamps, cutoff)
        if prune_count <= 0:
            return

        del self._timestamps[:prune_count]
        del self._records[:prune_count]


class MultiFrequencyBuffer:
    """Thread-safe multi-channel buffer for timestamp-aligned lookups."""

    def __init__(self, config: BufferConfig) -> None:
        """Initialize buffer channels from configuration.

        Args:
            config: Top-level buffer configuration.
        """

        self._config = config
        self._registry_lock = Lock()
        self._channels: dict[tuple[DataType, DataFrequency], BufferChannel] = {}

        for channel_config in config.channels:
            self.add_channel(channel_config)

    def add_channel(self, config: ChannelConfig) -> None:
        """Register a new channel.

        Args:
            config: Channel configuration to register.

        Raises:
            ValueError: If a channel already exists for type/frequency.
        """

        channel_key = (config.data_type, config.frequency)
        with self._registry_lock:
            if channel_key in self._channels:
                raise ValueError(
                    "channel already exists for "
                    f"{config.data_type.value}/{config.frequency.value}"
                )
            self._channels[channel_key] = BufferChannel(config=config)

    def put(self, record: DataRecord) -> None:
        """Route record to matching channel.

        Args:
            record: Record to buffer.

        Raises:
            ValueError: If no matching channel exists.
        """

        with self._registry_lock:
            channel = self._resolve_channel_locked(record)

        channel.put(record)

    def get_aligned_state(
        self,
        timestamp: datetime,
        channels: list[DataType] | None = None,
    ) -> dict[DataType, DataRecord]:
        """Return latest records at or before timestamp across channels.

        Args:
            timestamp: Alignment timestamp.
            channels: Optional data-type filter.

        Returns:
            Mapping from data type to latest aligned record.
        """

        selected_types = set(channels) if channels is not None else None
        grouped_channels = self._group_by_data_type()

        aligned: dict[DataType, DataRecord] = {}
        for data_type, type_channels in grouped_channels.items():
            if selected_types is not None and data_type not in selected_types:
                continue

            latest_record = self._latest_from_channels(
                channels=type_channels,
                before=timestamp,
            )
            if latest_record is not None:
                aligned[data_type] = latest_record

        return aligned

    def get_latest(self, data_type: DataType) -> DataRecord | None:
        """Return latest record across channels for a data type.

        Args:
            data_type: Data type to query.

        Returns:
            Most recent record for data type across all configured frequencies.
        """

        type_channels = self._group_by_data_type().get(data_type, [])
        return self._latest_from_channels(channels=type_channels, before=None)

    def get_all_latest(self) -> dict[DataType, DataRecord]:
        """Return latest record for each configured data type."""

        latest_by_type: dict[DataType, DataRecord] = {}
        for data_type, channels in self._group_by_data_type().items():
            latest_record = self._latest_from_channels(channels=channels, before=None)
            if latest_record is not None:
                latest_by_type[data_type] = latest_record

        return latest_by_type

    def clear_all(self) -> None:
        """Clear records from every channel."""

        for channel in self._snapshot_channels():
            channel.clear()

    def prune_all(self, before: datetime) -> int:
        """Prune old records from every channel.

        Args:
            before: Exclusive timestamp boundary.

        Returns:
            Total number of records removed.
        """

        removed_total = 0
        for channel in self._snapshot_channels():
            removed_total += channel.prune(before)

        return removed_total

    def get_memory_usage(self) -> dict[DataType, int]:
        """Return retained record counts grouped by data type."""

        usage: dict[DataType, int] = {}
        for data_type, channels in self._group_by_data_type().items():
            usage[data_type] = sum(len(channel) for channel in channels)

        return usage

    def _resolve_channel_locked(self, record: DataRecord) -> BufferChannel:
        """Resolve target channel for a record under registry lock.

        Args:
            record: Record to resolve.

        Returns:
            Matching channel.

        Raises:
            ValueError: If no unambiguous channel can be resolved.
        """

        if record.frequency is not None:
            key = (record.data_type, record.frequency)
            if key in self._channels:
                return self._channels[key]

        candidates = [
            channel
            for (data_type, _frequency), channel in self._channels.items()
            if data_type == record.data_type
        ]
        if len(candidates) == 1:
            return candidates[0]

        raise ValueError(
            "no matching channel for record "
            f"data_type={record.data_type.value}, frequency={record.frequency}"
        )

    def _snapshot_channels(self) -> list[BufferChannel]:
        """Return a stable snapshot of channels."""

        with self._registry_lock:
            return list(self._channels.values())

    def _group_by_data_type(self) -> dict[DataType, list[BufferChannel]]:
        """Group configured channels by data type."""

        grouped: dict[DataType, list[BufferChannel]] = {}
        with self._registry_lock:
            for (data_type, _frequency), channel in self._channels.items():
                grouped.setdefault(data_type, []).append(channel)

        return grouped

    @staticmethod
    def _latest_from_channels(
        channels: list[BufferChannel],
        before: datetime | None,
    ) -> DataRecord | None:
        """Return newest record among channels for the same data type."""

        latest_record: DataRecord | None = None
        for channel in channels:
            candidate = channel.get_latest(before=before)
            if candidate is None:
                continue
            if latest_record is None or candidate.timestamp > latest_record.timestamp:
                latest_record = candidate

        return latest_record
