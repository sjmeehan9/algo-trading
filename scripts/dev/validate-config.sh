#!/usr/bin/env bash
# Validate Component 6.7 runtime settings without printing secret values.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=venv.sh
source "${SCRIPT_DIR}/venv.sh"
activate_venv

python - <<'PY'
from algotrading.src.config.settings import get_settings
from algotrading.src.config.validation import validate_configuration

settings = get_settings()
is_valid, errors = validate_configuration(settings)
if not is_valid:
    print("Configuration validation failed:")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

print("Configuration validation passed.")
PY
