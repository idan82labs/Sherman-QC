#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FRONTEND_DIR="${REPO_ROOT}/frontend/react"
EXPECTED_NODE_MAJOR="20"
MAX_SUPPORTED_NODE_MAJOR="25"

if ! command -v node >/dev/null 2>&1; then
  echo "error: node is not installed. Use Node ${EXPECTED_NODE_MAJOR}.x from ${REPO_ROOT}/.nvmrc." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "error: npm is not installed. Use Node ${EXPECTED_NODE_MAJOR}.x from ${REPO_ROOT}/.nvmrc." >&2
  exit 1
fi

NODE_MAJOR="$(node -p "process.versions.node.split('.')[0]")"
if (( NODE_MAJOR < EXPECTED_NODE_MAJOR || NODE_MAJOR > MAX_SUPPORTED_NODE_MAJOR )); then
  echo "error: supported frontend Node range is ${EXPECTED_NODE_MAJOR}.x through ${MAX_SUPPORTED_NODE_MAJOR}.x; found $(node -v)." >&2
  echo "hint: run 'nvm use' in ${REPO_ROOT} or install a supported Node version." >&2
  exit 1
fi
if [[ "${NODE_MAJOR}" != "${EXPECTED_NODE_MAJOR}" ]]; then
  echo "warning: frontend CI is pinned to Node ${EXPECTED_NODE_MAJOR}.x; local build is running on $(node -v)." >&2
fi

if [[ ! -d "${FRONTEND_DIR}" ]]; then
  echo "error: frontend workspace not found at ${FRONTEND_DIR}." >&2
  exit 1
fi

if [[ ! -f "${FRONTEND_DIR}/package-lock.json" ]]; then
  echo "error: package-lock.json is missing from ${FRONTEND_DIR}; npm ci cannot run deterministically." >&2
  exit 1
fi

if [[ ! -x "${FRONTEND_DIR}/node_modules/.bin/tsc" ]]; then
  echo "frontend dependencies missing or incomplete; running npm ci in ${FRONTEND_DIR}" >&2
  (
    cd "${FRONTEND_DIR}"
    npm ci --no-audit --no-fund
  )
fi

echo "running frontend typecheck" >&2
(
  cd "${FRONTEND_DIR}"
  npm run typecheck
)

echo "running frontend production build" >&2
(
  cd "${FRONTEND_DIR}"
  npm run build
)

echo "frontend toolchain check passed" >&2
