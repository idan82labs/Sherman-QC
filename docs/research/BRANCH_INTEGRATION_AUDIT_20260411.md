# Branch Integration Audit — 2026-04-11

## Goal

Track which open branches can safely move toward `main`, and which ones should remain out until they are split or ported.

## Current integration branch

- branch: `codex/main-integration`
- worktree: `/Users/idant/82Labs/Sherman QC/Full Project-main-integration`

This branch currently contains:
1. `main`
2. merged `codex/qc-clean-mainline`
3. merged follow-up `B14` recovery from `codex/qc-clean-mainline`

Verification on this branch:
- targeted QC suites: `104 passed`

## Branch-by-branch status

### `main`

Base branch:
- `/Users/idant/82Labs/Sherman QC/Full Project-main-integration`

Status:
- current production base

### `codex/qc-clean-mainline`

Status:
- merged into `codex/main-integration`

Reason:
- clean QC slice
- benchmark-positive
- documented root-cause notes
- regression coverage in place

### `origin/codex/unified-integration`

Status:
- already effectively in `main`

Evidence:
- `origin/codex/unified-integration` is an ancestor of `main`

Action:
- no additional merge required

### `origin/codex/observability-corpus-upgrade`

Status:
- not merged into `main`
- not merged into `codex/qc-clean-mainline`
- direct dry-run merge into `codex/main-integration` is not clean

Observed overlap / conflict surface:
- `backend/feature_detection/bend_detector.py`
- plus frontend/API feature files from the bend inspection UI slice

Assessment:
- parts of the observability codepath are already superseded by the QC clean branch
- the branch also carries corpus/spec metadata and enrichment scripts that need selective review

Action:
- do not merge this branch wholesale
- selectively port any still-useful corpus/spec assets later if needed

### `codex/enterprise-reorder`

Status:
- not mergeable to `main` as-is
- direct dry-run merge into `codex/main-integration` is not clean

Observed conflict surface:
- same frontend/API bend-inspection feature slice
- conflict at `backend/feature_detection/bend_detector.py`

Assessment:
- mixed UI/API/product work plus QC work
- too broad for safe direct merge

Action:
- do not merge wholesale
- split or port intentionally later

## Effective merge path

Safe path toward `main`:
1. `main`
2. merge `codex/main-integration`

Unsafe paths right now:
- `codex/enterprise-reorder` directly to `main`
- `origin/codex/observability-corpus-upgrade` directly to `main`

## Remaining technical follow-up

On the QC side, the next unresolved bend is:
- `CAD_B11`

That work should continue from:
- `codex/qc-clean-mainline`

and then be merged forward into:
- `codex/main-integration`
