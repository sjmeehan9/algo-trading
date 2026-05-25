#!/usr/bin/env bash
# Install backend and frontend dependencies for manual local deployment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=venv.sh
source "${SCRIPT_DIR}/venv.sh"

INSTALL_BACKEND=true
INSTALL_FRONTEND=true

for arg in "$@"; do
  case "${arg}" in
    --backend-only)
      INSTALL_FRONTEND=false
      ;;
    --frontend-only)
      INSTALL_BACKEND=false
      ;;
    *)
      echo "Usage: scripts/dev/install.sh [--backend-only|--frontend-only]" >&2
      exit 2
      ;;
  esac
done

if [[ "${INSTALL_BACKEND}" == true ]]; then
  if ! command -v python3.11 >/dev/null 2>&1 && [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "python3.11 is required on PATH to create the backend virtualenv." >&2
    exit 1
  fi

  if [[ ! -x "${VENV_PYTHON}" ]]; then
    python3.11 -m venv "${VENV_DIR}"
  fi

  activate_venv
  python -m pip install --upgrade pip
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
  python -m pip install -e "${ROOT_DIR}"
fi

if [[ "${INSTALL_FRONTEND}" == true ]]; then
  if ! command -v pnpm >/dev/null 2>&1; then
    echo "pnpm is required on PATH to install frontend dependencies." >&2
    exit 1
  fi

  cd "${ROOT_DIR}/frontend"
  pnpm install
fi
