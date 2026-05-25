#!/usr/bin/env bash
# Run the Vite frontend dev server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if ! command -v pnpm >/dev/null 2>&1; then
  echo "pnpm is required on PATH to run the frontend dev server." >&2
  exit 1
fi

cd "${ROOT_DIR}/frontend"
exec pnpm dev
