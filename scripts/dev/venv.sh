#!/usr/bin/env bash
# Resolve and activate the Python virtual environment used by dev scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

resolve_venv_dir() {
  if [[ -n "${ALGOTRADING_VENV_DIR:-}" ]]; then
    printf '%s\n' "${ALGOTRADING_VENV_DIR}"
    return 0
  fi

  if [[ -d "${ROOT_DIR}/.venv" ]]; then
    printf '%s\n' "${ROOT_DIR}/.venv"
    return 0
  fi

  if [[ -d "${ROOT_DIR}/../.venv" ]]; then
    printf '%s\n' "$(cd "${ROOT_DIR}/../.venv" && pwd)"
    return 0
  fi

  printf '%s\n' "${ROOT_DIR}/.venv"
}

VENV_DIR="$(resolve_venv_dir)"
VENV_PYTHON="${VENV_DIR}/bin/python"

activate_venv() {
  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Virtualenv not found at ${VENV_DIR}. Run make install first." >&2
    return 1
  fi

  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
}
