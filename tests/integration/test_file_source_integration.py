"""Integration tests for FileSource against existing historical data."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from algotrading.src.data_pipeline.sources.file_source import FileSource


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_file_source_loads_existing_saved_data() -> None:
    """FileSource loads existing repository historical data as DataBatch."""

    base_path = _repo_root() / "data" / "saved_data" / "pipeline_0001"
    source = FileSource(base_path=str(base_path), trim_percentage=0.038)
    source.connect()

    eastern = ZoneInfo("US/Eastern")
    batch = source.fetch_batch(
        symbol="AMD",
        start=datetime(2022, 1, 3, 9, 30, tzinfo=eastern),
        end=datetime(2022, 1, 3, 16, 0, tzinfo=eastern),
    )

    assert len(batch.records) > 0
    assert batch.records[0].symbol == "AMD"
    assert batch.records[0].timestamp.tzinfo is not None


def test_file_source_matches_legacy_trim_behavior_for_single_file() -> None:
    """FileSource row count and values align with legacy trim behavior."""

    base_path = _repo_root() / "data" / "saved_data" / "pipeline_0001"
    file_path = base_path / "AMD_NASDAQ_20220103.csv"

    original = pd.read_csv(file_path)
    rows_to_drop = int(len(original) * 0.038 / 2)
    expected = original.iloc[rows_to_drop:-rows_to_drop].reset_index(drop=True)

    source = FileSource(base_path=str(base_path), trim_percentage=0.038)
    source.connect()

    eastern = ZoneInfo("US/Eastern")
    batch = source.fetch_batch(
        symbol="AMD",
        start=datetime(2022, 1, 3, 0, 0, tzinfo=eastern),
        end=datetime(2022, 1, 3, 23, 59, tzinfo=eastern),
    )

    actual_df = batch.to_dataframe().reset_index(drop=True)

    assert len(actual_df) == len(expected)
    assert actual_df.loc[0, "close"] == expected.loc[0, "close"]
    assert (
        actual_df.loc[len(actual_df) - 1, "close"]
        == expected.loc[len(expected) - 1, "close"]
    )
