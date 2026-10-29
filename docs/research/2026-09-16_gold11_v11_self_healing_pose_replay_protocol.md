# Gold11 V11 self-healing pose replay protocol — 2026-09-16

This protocol freezes a one-case mechanism replay for the V11 self-healing
path.  It uses the consumed Gold11 case `CG191DEC7F81` / `49001000`, whose
retained V11 registration contains two finite pose candidates.  The replay
holds raw scan, CAD, native topology, and scoring fixed while varying only
retained pose index (`0`, `1`) and support radius (`3.0`, `5.0` mm).

The full V11 attempt remains terminal and is not reopened.  This is a new
development diagnostic; it is not a replacement benchmark or promotion run.

## Frozen source and scope

Runner: `scripts/run_gold11_v11_self_healing_pose_replay_v1.py`, source SHA-256
`a96ff0e9a6371cc428adc79a3c03c76e3d33b581e58d92b80fc943f407c724ce`.

The runner authenticates all eleven reviewed cases, then accesses only the
hash-bound scan/CAD pair for `CG191DEC7F81`.  It retains candidate provenance
and does not relabel the rejected candidate as an accepted datum.  No new
parts, held-outs, email attachments, or unrelated datasets are read.

Fixed limits are 50,000 prepared scan rows and `OMP_NUM_THREADS=1`.  Gold and
the unchanged `UNCHANGED_MANUAL_GOLD_ONE_TO_ONE_MINIMUM_COST` matcher enter
only after all four inference replays return.

## Decision boundary

If pose 1 improves output at the same radius, the previous failure includes
pose-family selection despite the accepted pose's better global p95.  If only
5 mm improves output, the support gate is also implicated.  If neither does,
the remaining mechanism is native axis placement or correspondence.  No
single-case result authorizes changing the maintained engine; any change must
pass independent synthetic tests and a frozen paired multi-scan evaluation.

## Reproducible command

```text
OMP_NUM_THREADS=1 .venv/bin/python -B scripts/run_gold11_v11_self_healing_pose_replay_v1.py \
  --output docs/research/gold11_v11_self_healing_pose_replay_v1
```
