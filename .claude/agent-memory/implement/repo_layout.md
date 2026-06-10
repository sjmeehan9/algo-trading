---
name: repo-layout
description: Where things live in the trading-app-dev / algo-trading repo
metadata:
  type: project
---

Repo root: `/Users/seanmeehan/Projects/trading-app-dev`.

- Docs live at the **repo root** `docs/` (NOT under `algo-trading/`): `docs/phase-7-component-breakdown.md`, `docs/implementation-context-phase-7.md`, `docs/components/phase-7-component-7-*-overview.md`, `docs/deployment/`, `docs/errors/`.
- Backend Python package: `algo-trading/app/algotrading/` with `api/` (FastAPI: routers, services, schemas, workers, training) and `src/` (broker, data_pipeline, data_sourcing, envs, trainers, models, reward_functions).
- `.venv` is at the **repo root**, not under `algo-trading`. Activate with `source /Users/seanmeehan/Projects/trading-app-dev/.venv/bin/activate`.
- Tests: `algo-trading/tests/` with `integration/`, `unit/`, `fixtures/`. Run via `cd algo-trading && python -m pytest -q tests/integration/...`.
- Env file for provider creds: `algo-trading/.env/.env.local`.

**Why:** The task prompt's suggested paths sometimes prefix `algo-trading/` to doc paths, but the real docs are at repo-root `docs/`.
**How to apply:** Read docs from `/Users/seanmeehan/Projects/trading-app-dev/docs/...`; run code/tests from the `algo-trading/` working dir with the root `.venv`.
