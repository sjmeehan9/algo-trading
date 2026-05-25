#!/usr/bin/env bash
# Stop local API/frontend processes by port for the manual dev stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT="8000"
FRONTEND_PORT="${VITE_PORT:-3000}"

if [[ -f "${SCRIPT_DIR}/venv.sh" ]]; then
  # shellcheck source=venv.sh
  source "${SCRIPT_DIR}/venv.sh"
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    activate_venv
    API_PORT="$(python - <<'PY'
from algotrading.src.config.settings import get_settings

try:
    print(get_settings().api_port)
except Exception:
    print(8000)
PY
)"
  fi
fi

stop_port() {
  local port="$1"
  local label="$2"

  if ! command -v lsof >/dev/null 2>&1; then
    echo "lsof is required to stop ${label} by port." >&2
    return 1
  fi

  local pids
  pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN || true)"
  if [[ -z "${pids}" ]]; then
    echo "No ${label} process listening on port ${port}."
    return 0
  fi

  echo "Stopping ${label} process(es) on port ${port}: ${pids}"
  kill ${pids}
}

stop_port "${API_PORT}" "API"
stop_port "${FRONTEND_PORT}" "frontend"
