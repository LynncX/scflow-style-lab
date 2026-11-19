#!/usr/bin/env bash
# Background (2026-11): verify comparison orchestration offline before GPU deployment.
# Usage: PY=/path/to/python bash scripts/test_mock.sh
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: PY=python3 bash scripts/test_mock.sh\nOffline unittest suite; no installation, GPU or weights.\n'
    exit 0
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"${PY:-python3}" -m unittest discover -s "$ROOT/tests" -p 'test_*.py' -v
