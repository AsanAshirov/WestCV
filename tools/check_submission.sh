#!/usr/bin/env bash
# check_submission.sh — fast local gate before shipping the solution anywhere:
# byte-compile everything, run the unit tests, import solution.py (models load), validate an
# events file with the official evaluate.py if one is given.
#   tools/check_submission.sh [predictions.json]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY=python3
"$PY" -m compileall -q src solution.py
"$PY" -m pytest -q tests
"$PY" -c "import solution; print('solution imports, classes:', solution.CLASSES)"
if [ $# -ge 1 ]; then
  "$PY" evaluate.py --pred "$1" --validate-only
fi
echo "check_submission: OK"
