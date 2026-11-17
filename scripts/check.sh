#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"

"$PY" -m compileall -q "$ROOT/src" "$ROOT/scripts"
"$PY" "$ROOT/src/runner.py" "$ROOT/configs/baseline_mock.json"
printf 'ok\n'
