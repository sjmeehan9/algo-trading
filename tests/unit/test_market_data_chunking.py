"""Unit tests for IB-compliant chunking, session windows, and request pacing.

Covers the regression behind error-001: historical-data requests must never
emit a seconds (``S``) duration above 86400, must split intraday ranges to the
IB step-size maxima, optionally split per trading-session day, and stagger
requests within IB pacing limits.
"""

from __future__ import annotations

from datetime import UTC, datetime
from threading import Event
from zoneinfo import ZoneInfo

import pytest

# Import the schema first so ``algotrading.api`` is fully initialised before the
# acquisition package, avoiding the package import-order cycle.
from algotrading.api.schemas.data_sources import (
    DEFAULT_REQUEST_PACING_SECONDS,
    MarketDataSourceConfig,
)
from algotrading.src.data_pipeline.acquisition import market_data
from algotrading.src.data_pipeline.acquisition.market_data import (
    _RequestPacer,
    _bar_interval_for_frequency,
    _chunk_ranges,
    _is_intraday_frequency,
    _max_request_span,
)
from algotrading.src.data_pipeline import DataFrequency
from algotrading.src.data_pipeline.sources.broker_source import BrokerDataSource

_SRC = BrokerDataSource(adapter=object())
_ET = ZoneInfo("America/New_York")
_SESSION = (
    datetime.strptime("09:30", "%H:%M").time(),
    datetime.strptime("15:30", "%H:%M").time(),
    _ET,
)


def _planned_durations(frequency: DataFrequency, start, end, session=None):
    """Replicate the production chunk -> duration pipeline and return strings."""

    padding = _bar_interval_for_frequency(frequency)
    max_span = _max_request_span(frequency) - 2 * padding
    if max_span.total_seconds() <= 0:
        max_span = _max_request_span(frequency)
    use_session = session if _is_intraday_frequency(frequency) else None
    chunks = list(_chunk_ranges(start, end, max_span=max_span, session=use_session))
    return [
        _SRC._duration_string(chunk_start - padding, chunk_end + padding)
        for chunk_start, chunk_end in chunks
    ], chunks


def _assert_all_valid(durations: list[str]) -> None:
    for duration in durations:
        amount_text, unit = duration.split()
        amount = int(amount_text)
        assert unit in {"S", "D", "Y"}
        if unit == "S":
            assert amount <= 86400, f"S duration exceeds IB limit: {duration}"


# --------------------------------------------------------------------------- #
# Duration string encoding
# --------------------------------------------------------------------------- #


def test_duration_string_uses_seconds_within_one_day() -> None:
    start = datetime(2026, 1, 5, 14, 0, tzinfo=UTC)
    end = datetime(2026, 1, 5, 20, 0, tzinfo=UTC)
    assert _SRC._duration_string(start, end) == "21600 S"


def test_duration_string_caps_seconds_at_one_day() -> None:
    start = datetime(2026, 1, 5, tzinfo=UTC)
    end = datetime(2026, 1, 6, tzinfo=UTC)  # exactly 86400s -> still S
    assert _SRC._duration_string(start, end) == "86400 S"


def test_duration_string_switches_to_days_above_one_day() -> None:
    start = datetime(2026, 1, 5, tzinfo=UTC)
    end = datetime(2026, 1, 12, tzinfo=UTC)  # 7 days
    assert _SRC._duration_string(start, end) == "7 D"


def test_duration_string_switches_to_years_above_one_year() -> None:
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, tzinfo=UTC)  # ~3 years (365-day years -> 4 Y)
    duration = _SRC._duration_string(start, end)
    amount, unit = duration.split()
    assert unit == "Y"
    assert int(amount) >= 3


# --------------------------------------------------------------------------- #
# Chunking respects IB limits (the error-001 regression)
# --------------------------------------------------------------------------- #


def test_minute_bug_scenario_never_exceeds_seconds_limit() -> None:
    # NVDA 1-min over ~one week previously produced "604920 S" -> rejected.
    start = datetime(2025, 5, 26, 0, 1, tzinfo=UTC)
    end = datetime(2025, 6, 2, 0, 1, tzinfo=UTC)
    durations, chunks = _planned_durations(DataFrequency.MINUTE_1, start, end)
    # ~one week of 1-min bars now splits into day-bounded requests instead of
    # a single oversized "604920 S" request.
    assert len(chunks) == 8
    _assert_all_valid(durations)
    assert durations[0] == "86400 S"


def test_sub_minute_frequencies_stay_within_step_size_table() -> None:
    start = datetime(2025, 5, 26, tzinfo=UTC)
    end = datetime(2025, 6, 2, tzinfo=UTC)
    for frequency, expected in (
        (DataFrequency.SECOND_1, "1800 S"),
        (DataFrequency.SECOND_5, "3600 S"),
        (DataFrequency.SECOND_10, "14400 S"),
        (DataFrequency.SECOND_30, "28800 S"),
    ):
        durations, _ = _planned_durations(frequency, start, end)
        _assert_all_valid(durations)
        assert durations[0] == expected


def test_daily_frequency_uses_year_chunks_and_ignores_session() -> None:
    start = datetime(2023, 6, 2, tzinfo=UTC)
    end = datetime(2026, 6, 2, tzinfo=UTC)
    durations, chunks = _planned_durations(
        DataFrequency.DAY_1, start, end, session=_SESSION
    )
    _assert_all_valid(durations)
    # Year-bounded daily chunks; the session window is ignored for daily bars.
    assert durations[0] == "365 D"
    assert all(unit == "D" for unit in (d.split()[1] for d in durations))


# --------------------------------------------------------------------------- #
# Trading-session day splitting
# --------------------------------------------------------------------------- #


def test_session_window_splits_into_weekday_sessions() -> None:
    # Mon 2025-05-26 .. Mon 2025-06-02 inclusive of four weekdays in between.
    start = datetime(2025, 5, 26, tzinfo=UTC)
    end = datetime(2025, 5, 31, tzinfo=UTC)  # Sat
    max_span = _max_request_span(DataFrequency.MINUTE_1)
    chunks = list(_chunk_ranges(start, end, max_span=max_span, session=_SESSION))
    # Mon-Fri = 5 weekday sessions; weekend produces none.
    assert len(chunks) == 5
    for chunk_start, chunk_end in chunks:
        local_start = chunk_start.astimezone(_ET)
        local_end = chunk_end.astimezone(_ET)
        assert (local_start.hour, local_start.minute) == (9, 30)
        assert (local_end.hour, local_end.minute) == (15, 30)
        assert local_start.weekday() < 5


def test_session_window_skips_weekends() -> None:
    # Sat + Sun only -> no sessions at all.
    start = datetime(2025, 5, 31, tzinfo=UTC)
    end = datetime(2025, 6, 1, 23, 0, tzinfo=UTC)
    max_span = _max_request_span(DataFrequency.MINUTE_1)
    chunks = list(_chunk_ranges(start, end, max_span=max_span, session=_SESSION))
    assert chunks == []


def test_session_window_subsplits_one_second_bars_per_day() -> None:
    start = datetime(2025, 5, 27, tzinfo=UTC)  # Tue
    end = datetime(2025, 5, 28, tzinfo=UTC)  # Wed
    durations, chunks = _planned_durations(
        DataFrequency.SECOND_1, start, end, session=_SESSION
    )
    _assert_all_valid(durations)
    # One trading day (Tue) of a 6h session at 1800s steps -> many chunks, all
    # within the 1-sec step-size maximum (1800 S), plus a small remainder.
    assert len(chunks) >= 12
    assert "1800 S" in durations
    assert durations.count("1800 S") >= 12


# --------------------------------------------------------------------------- #
# Schema: session window configuration
# --------------------------------------------------------------------------- #


def test_session_window_absent_by_default() -> None:
    config = MarketDataSourceConfig()
    assert config.session_window() is None
    assert config.request_pacing_seconds == DEFAULT_REQUEST_PACING_SECONDS


def test_session_window_parses_configured_values() -> None:
    config = MarketDataSourceConfig(
        session_start="09:30",
        session_end="15:30",
        session_timezone="America/New_York",
    )
    window = config.session_window()
    assert window is not None
    start_time, end_time, tz = window
    assert (start_time.hour, start_time.minute) == (9, 30)
    assert (end_time.hour, end_time.minute) == (15, 30)
    assert tz == ZoneInfo("America/New_York")


def test_session_bounds_must_be_supplied_together() -> None:
    with pytest.raises(ValueError):
        MarketDataSourceConfig(session_start="09:30")


def test_session_end_must_follow_start() -> None:
    with pytest.raises(ValueError):
        MarketDataSourceConfig(session_start="15:30", session_end="09:30")


def test_invalid_session_timezone_rejected() -> None:
    with pytest.raises(ValueError):
        MarketDataSourceConfig(
            session_start="09:30",
            session_end="15:30",
            session_timezone="Mars/Phobos",
        )


def test_from_config_reads_session_keys() -> None:
    config = MarketDataSourceConfig.from_config(
        {
            "session_start": "10:00",
            "session_end": "16:00",
            "session_timezone": "Europe/London",
            "request_pacing_seconds": 2.5,
        },
        frequency=DataFrequency.MINUTE_1,
        symbols=["AAPL"],
    )
    window = config.session_window()
    assert window is not None
    assert config.request_pacing_seconds == 2.5
    assert window[2] == ZoneInfo("Europe/London")


# --------------------------------------------------------------------------- #
# Request pacing
# --------------------------------------------------------------------------- #


def _patch_clock(monkeypatch) -> tuple[dict[str, float], list[float]]:
    clock = {"now": 0.0}
    sleeps: list[float] = []

    monkeypatch.setattr(market_data.time_module, "monotonic", lambda: clock["now"])

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(market_data.time_module, "sleep", fake_sleep)
    return clock, sleeps


def test_pacer_enforces_minimum_spacing(monkeypatch) -> None:
    _clock, sleeps = _patch_clock(monkeypatch)
    pacer = _RequestPacer(min_spacing=1.0)

    pacer.wait()  # first request, no wait
    assert sleeps == []
    pacer.wait()  # immediate second -> waits ~1s (in 0.25s slices)
    assert pytest.approx(sum(sleeps), abs=0.26) == 1.0


def test_pacer_respects_rolling_window_cap(monkeypatch) -> None:
    _clock, sleeps = _patch_clock(monkeypatch)
    pacer = _RequestPacer(min_spacing=0.0)

    # Fill the window with the maximum allowed requests at t=0.
    for _ in range(market_data._MAX_REQUESTS_PER_WINDOW):
        pacer.wait()
    assert sleeps == []
    # The next request must wait out the rolling window before proceeding.
    pacer.wait()
    assert sum(sleeps) == pytest.approx(market_data._PACING_WINDOW_SECONDS, abs=0.26)


def test_pacer_wait_is_interruptible_by_cancellation(monkeypatch) -> None:
    _clock, sleeps = _patch_clock(monkeypatch)
    pacer = _RequestPacer(min_spacing=100.0)
    cancel = Event()
    cancel.set()

    pacer.wait()  # first request, no spacing wait
    pacer.wait(cancel)  # would wait 100s, but cancellation aborts immediately
    # Only the tiny first slice (if any) runs before the cancel check returns.
    assert sum(sleeps) < 1.0


def test_build_pacer_uses_steady_spacing_for_large_jobs() -> None:
    config = MarketDataSourceConfig(request_pacing_seconds=1.0)
    request = _request_with(config)
    small = market_data._build_pacer(
        request=request, chunks_per_symbol=10, symbol_count=1
    )
    large = market_data._build_pacer(
        request=request,
        chunks_per_symbol=market_data._MAX_REQUESTS_PER_WINDOW + 10,
        symbol_count=1,
    )
    assert small._min_spacing == 1.0
    assert large._min_spacing == pytest.approx(market_data._STEADY_INTERVAL_SECONDS)


def _request_with(market_config: MarketDataSourceConfig):
    from algotrading.api.schemas.data_sources import TrainingDataRequest

    return TrainingDataRequest(
        symbols=["NVDA"],
        start_time=datetime(2025, 1, 1, tzinfo=UTC),
        end_time=datetime(2025, 2, 1, tzinfo=UTC),
        frequency=DataFrequency.MINUTE_1,
        market_source=market_config,
    )


# --------------------------------------------------------------------------- #
# Cancellation of the chunk loop
# --------------------------------------------------------------------------- #


class _FakeSource:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_batch(self, *, symbol, start, end, data_type):  # noqa: ANN001
        from algotrading.src.data_pipeline import DataBatch, DataType

        self.calls += 1
        return DataBatch(
            records=[],
            start_time=start,
            end_time=end,
            data_type=DataType.MARKET_BAR,
            symbol=symbol,
        )


def test_fetch_chunked_batch_aborts_when_cancelled_before_first_request() -> None:
    request = _request_with(MarketDataSourceConfig(request_pacing_seconds=0.0))
    chunks = market_data._plan_chunks(request)
    source = _FakeSource()
    cancel = Event()
    cancel.set()

    with pytest.raises(market_data.MarketDataAcquisitionCancelled):
        market_data._fetch_chunked_batch(
            source,
            request,
            "NVDA",
            chunks=chunks,
            pacer=_RequestPacer(min_spacing=0.0),
            cancel_event=cancel,
        )
    assert source.calls == 0


def test_fetch_chunked_batch_completes_without_cancellation() -> None:
    request = _request_with(MarketDataSourceConfig(request_pacing_seconds=0.0))
    chunks = market_data._plan_chunks(request)
    source = _FakeSource()

    batch = market_data._fetch_chunked_batch(
        source,
        request,
        "NVDA",
        chunks=chunks,
        pacer=_RequestPacer(min_spacing=0.0),
        cancel_event=Event(),
    )
    assert source.calls == len(chunks)
    assert batch.symbol == "NVDA"
