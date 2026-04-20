"""Unit tests for the custom strategy registry lifecycle."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from algotrading.src.models.registry import CustomStrategyRegistry, trading_strategy
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType


@trading_strategy(name="InMemory", signal_type=SignalType.POSITION)
class InMemoryStrategy:
    """In-memory strategy used for registry lifecycle tests."""

    def __init__(self) -> None:
        self.reset_called = False

    def predict(self, state: dict[str, object], market_data: object) -> ModelSignal:
        return ModelSignal(
            timestamp=datetime.now(tz=UTC),
            signal_type=SignalType.POSITION,
            value=1.0,
            metadata=SignalMetadata(model_id="in_memory", model_type="strategy"),
        )

    def reset(self) -> None:
        self.reset_called = True


def test_register_get_unregister_strategy() -> None:
    """Registry should support manual strategy registration and removal."""

    registry = CustomStrategyRegistry(strategy_dirs=[], auto_scan=False)
    strategy_id = registry.register_strategy(InMemoryStrategy)

    entry = registry.get(strategy_id)
    assert entry is not None
    assert entry.name == "InMemory"
    assert entry.signal_type == SignalType.POSITION

    assert registry.unregister(strategy_id) is True
    assert registry.get(strategy_id) is None


def test_instance_management_and_reset() -> None:
    """Registry should create singleton instance and invoke reset when available."""

    registry = CustomStrategyRegistry(strategy_dirs=[], auto_scan=False)
    strategy_id = registry.register_strategy(InMemoryStrategy)

    instance_a = registry.get_instance(strategy_id)
    instance_b = registry.get_instance(strategy_id)

    assert instance_a is instance_b
    assert instance_a.reset_called is False

    registry.reset_instance(strategy_id)
    assert instance_a.reset_called is True


def test_check_for_changes_and_reload(tmp_path: Path) -> None:
    """Registry should detect modified strategy files and reload them."""

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True)
    strategy_file = strategies_dir / "reloadable.py"

    strategy_file.write_text(
        "\n".join(
            [
                "from datetime import UTC, datetime",
                "from algotrading.src.models.registry import trading_strategy",
                "from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType",
                "",
                "@trading_strategy(name='Reloadable', version='1.0.0', signal_type=SignalType.POSITION)",
                "class ReloadableStrategy:",
                "    def predict(self, state, market_data) -> ModelSignal:",
                "        return ModelSignal(",
                "            timestamp=datetime.now(tz=UTC),",
                "            signal_type=SignalType.POSITION,",
                "            value=0.0,",
                "            metadata=SignalMetadata(model_id='reloadable', model_type='strategy'),",
                "        )",
            ]
        ),
        encoding="utf-8",
    )

    registry = CustomStrategyRegistry(
        strategy_dirs=[str(strategies_dir)], auto_scan=True
    )
    entries = registry.get_all()
    assert len(entries) == 1
    strategy_id = entries[0].strategy_id
    assert entries[0].version == "1.0.0"

    time.sleep(1.1)
    strategy_file.write_text(
        "\n".join(
            [
                "from datetime import UTC, datetime",
                "from algotrading.src.models.registry import trading_strategy",
                "from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType",
                "",
                "@trading_strategy(name='Reloadable', version='1.1.0', signal_type=SignalType.POSITION)",
                "class ReloadableStrategy:",
                "    def predict(self, state, market_data) -> ModelSignal:",
                "        return ModelSignal(",
                "            timestamp=datetime.now(tz=UTC),",
                "            signal_type=SignalType.POSITION,",
                "            value=0.0,",
                "            metadata=SignalMetadata(model_id='reloadable', model_type='strategy'),",
                "        )",
            ]
        ),
        encoding="utf-8",
    )

    changed = registry.check_for_changes()

    assert strategy_id in changed
    updated = registry.get(strategy_id)
    assert updated is not None
    assert updated.version == "1.1.0"
