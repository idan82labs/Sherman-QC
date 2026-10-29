#!/usr/bin/env bash
# Local CI for the Sherman QC production line (plan §4 C11; GitHub Actions is not used).
#
# Runs, from the repo root, and exits non-zero when any step fails:
#   1. every tests/test_prod_*.py focused suite, one pytest process per file, with the plan's
#      PYT form (no conftest, no plugins, no bytecode, OMP_NUM_THREADS=1). The gated real-engine
#      E2E (tests/test_prod_e2e_engine.py) skips itself unless --engine is given;
#   2. frontend: tsc --noEmit, npm run lint, the production node checks
#      (check-station-view-model, check-onboarding-model, check-station-part-picker, check-station-layout,
#      check-onboarding-ui;
#      the browser checks print SKIP and exit 0 when Playwright's Chromium is missing). The two
#      browser checks (check-station-layout, check-onboarding-ui) contain timing assertions that can
#      flake under load: a failure is re-run ONCE and, when the re-run passes, the step is reported
#      as RETRIED (both logs kept, listed in the summary), never as a silent PASS;
#   3. a Vite build into a fresh $TMPDIR directory (never `npm run build`, which would empty the
#      served frontend/dist);
#   4. the G5 no-regression gate (scripts/production/g5_check.py; exit 3 INCOMPLETE counts as a failure).
#
# Usage: bash scripts/run_production_line_checks.sh [--engine] [--skip-frontend] [--skip-g5]
#                                                    [--only-python] [--pattern GLOB]
#   --engine          also run the gated real-engine E2E (one heavy run, ~1 min)
#   --pattern GLOB    restrict step 1 to tests/GLOB (default test_prod_*.py)
# Logs: $TMPDIR/sherman-line-checks-<timestamp>/ (one log per step). Nothing is written in the repo.
set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
ENGINE=0
FRONTEND=1
G5=1
PATTERN="test_prod_*.py"
while [ $# -gt 0 ]; do
  case "$1" in
    --engine) ENGINE=1 ;;
    --skip-frontend) FRONTEND=0 ;;
    --skip-g5) G5=0 ;;
    --only-python) FRONTEND=0; G5=0 ;;
    --pattern) shift; PATTERN="${1:?--pattern needs a glob}" ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "missing $PY" >&2; exit 2; }
TMP_BASE="${TMPDIR:-/tmp}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGS="$TMP_BASE/sherman-line-checks-$STAMP"
mkdir -p "$LOGS"
FAILED=()
RETRIED=()
PASSED=0
START=$(date +%s)

step() {  # step NAME LOGFILE CMD...
  local name="$1" log="$2"; shift 2
  local t0; t0=$(date +%s)
  if "$@" >"$log" 2>&1; then
    PASSED=$((PASSED + 1))
    printf 'PASS  %-58s %4ss\n' "$name" "$(( $(date +%s) - t0 ))"
  else
    local rc=$?
    FAILED+=("$name (exit $rc, log $log)")
    printf 'FAIL  %-58s %4ss  exit %s  -> %s\n' "$name" "$(( $(date +%s) - t0 ))" "$rc" "$log"
    tail -n 15 "$log" | sed 's/^/      | /'
  fi
}

step_retry_once() {  # step_retry_once NAME LOGFILE CMD...  (timing-sensitive browser checks)
  local name="$1" log="$2"; shift 2
  local t0; t0=$(date +%s)
  if "$@" >"$log" 2>&1; then
    PASSED=$((PASSED + 1))
    printf 'PASS  %-58s %4ss\n' "$name" "$(( $(date +%s) - t0 ))"
    return
  fi
  local rc1=$?
  printf 'FAIL  %-58s %4ss  exit %s  -> %s (re-running once)\n' "$name" "$(( $(date +%s) - t0 ))" "$rc1" "$log"
  tail -n 8 "$log" | sed 's/^/      | /'
  local log2="${log%.log}.retry.log"
  if "$@" >"$log2" 2>&1; then
    PASSED=$((PASSED + 1))
    RETRIED+=("$name (first run exit $rc1, log $log; re-run passed, log $log2)")
    printf 'RETRIED %-56s %4ss  first run failed, re-run passed -> %s\n' "$name" "$(( $(date +%s) - t0 ))" "$log2"
  else
    local rc=$?
    FAILED+=("$name (exit $rc1 then $rc on re-run, logs $log $log2)")
    printf 'FAIL  %-58s %4ss  exit %s on re-run -> %s\n' "$name" "$(( $(date +%s) - t0 ))" "$rc" "$log2"
    tail -n 15 "$log2" | sed 's/^/      | /'
  fi
}

pyt() {
  env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 \
    "$PY" -m pytest -c /dev/null --noconftest -p no:cacheprovider "$@" -q
}

echo "== Sherman QC production-line checks ($STAMP); logs in $LOGS"
echo "-- python focused suites (tests/$PATTERN)"
shopt -s nullglob
TESTS=(tests/$PATTERN)
[ ${#TESTS[@]} -gt 0 ] || { echo "no tests match tests/$PATTERN" >&2; exit 2; }
for t in "${TESTS[@]}"; do
  name="$(basename "$t" .py)"
  if [ "$name" = "test_prod_e2e_engine" ]; then
    if [ "$ENGINE" = 1 ]; then
      step "$t (real engine, gated)" "$LOGS/$name.log" env SHERMAN_E2E_ENGINE=1 PYTHONDONTWRITEBYTECODE=1 \
        PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:backend OMP_NUM_THREADS=1 \
        "$PY" -m pytest -c /dev/null --noconftest -p no:cacheprovider "$t" -q -s
    fi
    continue
  fi
  step "$t" "$LOGS/$name.log" pyt "$t"
done

if [ "$FRONTEND" = 1 ]; then
  echo "-- frontend"
  FE="$REPO/frontend/react"
  if [ ! -x "$FE/node_modules/.bin/tsc" ]; then
    FAILED+=("frontend: node_modules missing (run npm ci in frontend/react)")
    echo "FAIL  frontend: node_modules missing"
  else
    step "tsc --noEmit" "$LOGS/tsc.log" bash -c "cd '$FE' && ./node_modules/.bin/tsc --noEmit"
    step "npm run lint" "$LOGS/lint.log" bash -c "cd '$FE' && npm run lint --silent"
    for check in check-station-view-model check-onboarding-model check-station-part-picker; do
      step "node scripts/$check.mjs" "$LOGS/$check.log" bash -c "cd '$FE' && node scripts/$check.mjs"
    done
    for check in check-station-layout check-onboarding-ui; do
      step_retry_once "node scripts/$check.mjs" "$LOGS/$check.log" bash -c "cd '$FE' && node scripts/$check.mjs"
    done
    OUT="$TMP_BASE/sherman-line-vite-$STAMP"
    step "vite build -> \$TMPDIR" "$LOGS/vite-build.log" bash -c "cd '$FE' && ./node_modules/.bin/vite build --outDir '$OUT' --emptyOutDir"
  fi
fi

if [ "$G5" = 1 ]; then
  echo "-- G5 no-regression gate"
  step "scripts/production/g5_check.py" "$LOGS/g5.log" "$PY" scripts/production/g5_check.py --json "$LOGS/g5.json"
fi

echo "== $PASSED passed (${#RETRIED[@]} after one re-run), ${#FAILED[@]} failed in $(( $(date +%s) - START ))s"
if [ ${#RETRIED[@]} -gt 0 ]; then
  printf '   RETRIED (flaky; investigate): %s\n' "${RETRIED[@]}"
fi
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '   FAILED: %s\n' "${FAILED[@]}"
  exit 1
fi
exit 0
