"""Unit tests for strategy discovery and validation loader."""

from __future__ import annotations

from pathlib import Path

from algotrading.src.models.registry import StrategyLoader


def test_scan_directories_discovers_python_files(tmp_path: Path) -> None:
    """Loader should discover strategy Python files and skip dunder modules."""

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True)
    (strategies_dir / "alpha.py").write_text("x = 1\n", encoding="utf-8")
    (strategies_dir / "__init__.py").write_text("\n", encoding="utf-8")

    loader = StrategyLoader([str(strategies_dir)])
    files = loader.scan_directories()

    assert files == [strategies_dir / "alpha.py"]


def test_load_all_returns_only_valid_decorated_strategies(tmp_path: Path) -> None:
    """Loader should load valid decorated classes and ignore invalid classes."""

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True)

    valid_path = strategies_dir / "valid_strategy.py"
    valid_path.write_text(
        "\n".join(
            [
                "from datetime import UTC, datetime",
                "from algotrading.src.models.registry import trading_strategy",
                "from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType",
                "",
                "@trading_strategy(name='Valid')",
                "class ValidStrategy:",
                "    def predict(self, state, market_data) -> ModelSignal:",
                "        return ModelSignal(",
                "            timestamp=datetime.now(tz=UTC),",
                "            signal_type=SignalType.POSITION,",
                "            value=0.0,",
                "            metadata=SignalMetadata(model_id='valid', model_type='strategy'),",
                "        )",
            ]
        ),
        encoding="utf-8",
    )

    invalid_path = strategies_dir / "invalid_strategy.py"
    invalid_path.write_text(
        "\n".join(
            [
                "class InvalidStrategy:",
                "    def predict(self, state):",
                "        return None",
            ]
        ),
        encoding="utf-8",
    )

    loader = StrategyLoader([str(strategies_dir)])
    loaded = loader.load_all()

    assert len(loaded) == 1
    strategy_class, metadata, filepath = loaded[0]
    assert strategy_class.__name__ == "ValidStrategy"
    assert metadata["name"] == "Valid"
    assert filepath == valid_path


def test_validate_strategy_rejects_bad_predict_signature(tmp_path: Path) -> None:
    """Validation should fail when predict() does not accept required arguments."""

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True)
    bad_path = strategies_dir / "bad_signature.py"
    bad_path.write_text(
        "\n".join(
            [
                "from algotrading.src.models.registry import trading_strategy",
                "",
                "@trading_strategy()",
                "class BadSignature:",
                "    def predict(self, state):",
                "        return 0",
            ]
        ),
        encoding="utf-8",
    )

    loader = StrategyLoader([str(strategies_dir)])
    module = loader.load_module(bad_path)
    strategy_class = loader.find_strategies(module)[0]

    is_valid, errors = loader.validate_strategy(strategy_class)

    assert is_valid is False
    assert any("predict() must accept" in error for error in errors)
