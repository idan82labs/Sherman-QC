#!/bin/bash
# usage: guarded_slot.sh <scan path|sc_id> [--final-testb] [--cad-path] -- <engine/lane cmd...>
# Runs the gold-ledger guard (research/zeroshot/common/gold_guard.py) and, only if it allows the scan,
# runs the command under one geometry slot. Every programme engine/lane run on a scan goes through this
# (or calls gold_guard.require_runnable in Python). --final-testb is for the integrator's single TEST-B run only.
REPO=/Users/idantal/ShermanQC/FullProject
scan="$1"; shift; flags=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do flags+=("$1"); shift; done
[ "$1" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: guarded_slot.sh <scan> [--final-testb] [--cad-path] -- <cmd...>" >&2; exit 64; }
(cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m research.zeroshot.common.gold_guard check --scan "$scan" "${flags[@]}") || exit 3
exec /private/tmp/claude-501/sherman_slot.sh "$@"
