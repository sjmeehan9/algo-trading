#!/usr/bin/env bash
# Validate configuration, then run the FastAPI backend with Uvicorn.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=venv.sh
source "${SCRIPT_DIR}/venv.sh"
activate_venv

"${SCRIPT_DIR}/validate-config.sh"

API_HOST="$(python - <<'PY'
from algotrading.src.config.settings import get_settings

print(get_settings().api_host)
PY
)"
API_PORT="$(python - <<'PY'
from algotrading.src.config.settings import get_settings

print(get_settings().api_port)
PY
)"

exec uvicorn algotrading.api.main:create_app --factory --host "${API_HOST}" --port "${API_PORT}"
