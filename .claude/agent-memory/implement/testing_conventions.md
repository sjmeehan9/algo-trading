---
name: testing-conventions
description: Integration-first testing, provider-confirm gates, and the APIConfig env gotcha
metadata:
  type: feedback
---

Phase 7 is integration-first: do NOT add low-value unit tests; extend the integration suites and exercise the real default-service paths.

- **Provider-confirm gates** (`tests/conftest.py`, `tests/integration/conftest.py`): markers `requires_ib`/`requires_alpaca`/`requires_news_api` auto-skip unless `--ib-confirm`/`--alpaca-confirm`/`--news-api-confirm` is passed. Session fixtures `confirm_ib_gateway`/`confirm_alpaca_paper`/`confirm_news_api` call `input(...)` once — pipe `echo "yes" | pytest ... --ib-confirm -s`. IB needs TWS on 127.0.0.1:7497. Never print secrets.
- **Tiny trained artifact pattern**: train a PPO model end-to-end through the default service with file-backed market CSV (provider `file` + `explicit_files`), tiny hyperparams (`n_steps:4, batch_size:4, n_epochs:1`), `observation_window:4`, `total_timesteps:8`, `model_policy:MultiInputPolicy`. Poll `/api/v1/training/jobs/{id}` until `completed`, read `model_path` from `/api/v1/generations/{id}`. Mirror `test_default_service_trains_core_rl_with_file_backed_market_data`.

**Why:** A real provider run (IB) caught a real production bug the file-backed path missed (day-wide backtest request vs intraday cache coverage). Actually running `--ib-confirm` is worth it when TWS is up.

**APIConfig gotcha:** `algo-trading/.env/.env.local` sets `CORS_ORIGINS='[http://localhost:3000]'` (NOT valid JSON). If you `source` that env before a test, `APIConfig(...)` constructed WITHOUT an explicit `cors_origins=` will raise a pydantic validation error. Always pass `cors_origins=["http://localhost:3000"]` explicitly in test `APIConfig(...)` calls.
**How to apply:** Provider tests must skip cleanly without the flag and pass when run with it; always set `cors_origins` explicitly in test configs.
