---
name: phase7-seams
description: Reusable Phase 7 seams for data acquisition, env building, training, and backtest replay
metadata:
  type: project
---

Phase 7 wired real data/model lifecycle. Key reusable seams (all COMPLETE):

- **Data contract**: `api/schemas/data_sources.py` → `TrainingDataRequest`, `normalize_training_data_request(model_cfg, data_cfg)`, `normalize_backtest_data_request`. `api/schemas/backtesting.py` → `build_backtest_data_request(request, model_data_config, overrides=...)` (widens to whole DAYS — only has start_date/end_date; pops intraday time keys).
- **Local store**: `src/data_pipeline/storage/local_store.py` → `LocalDataStore` reads/writes `data/sourced/market|news/<provider>/<symbol>[/<freq>]/`. `find_market_data(request)` returns `dict[symbol, StoredMarketData]`; `StoredMarketData.batch.to_dataframe()`. Cache coverage is checked against the **manifest's requested_start/end**, so a day-wide request can't be satisfied by an intraday cache.
- **Acquisition**: `api/services/data_acquisition_service.py` → `DataAcquisitionService.ensure_market_data(req)`, `ensure_news_data(req, model=...)`, `.store`. Build via `create_default_data_acquisition_service(api_config=, project_root=)` then set `._supporting_registry`.
- **Env building (shared 7.4↔7.5)**: `api/training/factories.py` → `build_trading_environment(model, data_service, model_service, single_pass=, request=)` returns `BuiltTradingEnvironment(env, market_frame, observation_window, symbol, request)`. `single_pass=False` repeats frames for training; `True` walks each bar once for backtest replay. `build_core_rl_environment(...)` delegates to it.
- **Trainer**: `src/trainers/sb3_trainer.py` → `StableBaselines3Trainer(algorithm=SB3Algorithm, policy=)`; `.load(path, env=)` (resolves `.zip`), `.predict(obs, deterministic=True) -> (action_int, info)`. Dict-obs envs need `MultiInputPolicy`.
- **Generation model_path**: saved as `<root>/data/models/<model_id>/<job_id>.zip`; generation status must be `completed`/`evaluated` to backtest.

**7.5 replay (TradingEnv)**: action ints 0=hold,1=buy,2=sell (`TradingTools.stop_take` returns action unchanged when stop_take disabled). The env's `state_builder.state_df` holds the RAW (unscaled) current window; its LAST row gives the current bar's `date` + `close` — use these for fill price/timestamp. Env terminates when step==episode_length. `ModelDrivenBacktestExecutor` is the default in `create_default_backtest_service`; `ReplayFillModel` is long-only (cash≥0, position≥0). It preserves the model's intraday window when it fits the backtest dates so cached intraday IB/Alpaca bars satisfy coverage.

**How to apply:** Reuse these seams instead of reimplementing data/env/trainer logic. When a backtest/training touches intraday provider data, narrow the request to the model's intraday window or cache coverage will fail.
