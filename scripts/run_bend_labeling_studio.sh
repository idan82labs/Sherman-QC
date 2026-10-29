#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
API_HOST="${BEND_LABELING_API_HOST:-127.0.0.1}"
API_PORT="${BEND_LABELING_API_PORT:-9000}"
WEB_HOST="${BEND_LABELING_WEB_HOST:-127.0.0.1}"
WEB_PORT="${BEND_LABELING_WEB_PORT:-3000}"

if ! "$PYTHON_BIN" -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
  echo "The selected Python runtime is missing backend dependencies."
  echo "Install requirements.txt or set PYTHON_BIN to the Sherman QC virtualenv Python."
  exit 1
fi

if [[ ! -d "$ROOT_DIR/frontend/react/node_modules" ]]; then
  echo "Frontend dependencies are missing. Run: npm --prefix frontend/react ci"
  exit 1
fi

cleanup() {
  trap - INT TERM EXIT
  kill "$API_PID" "$WEB_PID" 2>/dev/null || true
  wait "$API_PID" "$WEB_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

cd "$ROOT_DIR"
"$PYTHON_BIN" -m uvicorn apps.api.main:app --host "$API_HOST" --port "$API_PORT" &
API_PID=$!

npm --prefix "$ROOT_DIR/frontend/react" run dev -- --host "$WEB_HOST" --port "$WEB_PORT" &
WEB_PID=$!

echo "Bend Label Studio: http://$WEB_HOST:$WEB_PORT/bend-labeling"
wait "$API_PID" "$WEB_PID"
