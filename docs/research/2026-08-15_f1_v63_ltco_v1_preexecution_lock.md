# F1 V63 LTCO V1 pre-execution lock

Status: **FROZEN BEFORE V5R3 GEOMETRY CONSTRUCTION OR INFERENCE**

This lock binds the one prospective intervention authorized by the preserved
GPT-5.6 Pro post-V5R2 ruling. The consumed V5R2 cases are diagnostic only and
must never be executed through V63. No V5R3 replacement geometry existed when
the implementation, thresholds, tests, and hashes below were frozen.

## Authority

- Preserved ruling: `docs/research/2026-08-15_gpt56_pro_post_v5r2_smooth_roll_ruling_verbatim.md`
- Ruling SHA-256: `9e2d161f539fd192748c3d4ccf9f5d27801265829b0a7841f3a63bdb24720cbb`
- Authorized profile: `V63_LOCALIZED_TURN_CORRIDOR_OWNERSHIP_V1` (`LTCO_V1`)
- Frozen configuration SHA-256: `778bda993566612fbe671390106962c97b58bbb92c5a8f49c0c6e07cdd1501c4`

## Scope and invariants

- Audits only final `cylindrical_consensus_recovery_v52` predictions.
- Deletion only: no line birth, movement, reorder, second NMS, or non-V52 mutation.
- Nine stations at fractions 0.10 through 0.90.
- Scan spacing is the deterministic median sixth-other-neighbor distance over
  at most 4096 evenly selected source samples, with `(distance, source_index)`
  tie-breaking.
- At most 32 audited V52 candidates, 4096 raw station-slice points, 65 routed
  profile points, and 2048 public scalar bytes per candidate. Overflow fails
  the prospective run closed.
- Source points, source indices, normals, angle profiles, and turn-density
  profiles are transient and are not serialized.
- Exact thresholds and constants are bound by the configuration hash above and
  the source hashes below. There is no tuning, sweep, rescue, or post-hoc
  threshold change.

## Frozen source hashes

| File | SHA-256 |
|---|---|
| `backend/f1_cylindrical_band_completion.py` | `6d1a82e6a8b37025e0e8d112a19d4214a6343e3653f5260dad39938fde7775e5` |
| `backend/f1_mesh_section_turn_locus_reselection_v55.py` | `77e44ca017f022164c7e7c5861c19ab6210b729ee2f0d09b832e2a3ee3a84621` |
| `backend/f1_localized_turn_corridor_ownership_v63.py` | `f35dbc2941bff52b3191bcda75b17313ab47fc26450b6e3fbeae3a5af1f8251c` |
| `tests/test_f1_cylindrical_band_completion.py` | `9f7a82688ccfb990a99c60f58b1aabdc861434f9f152eb6f062cd2e558c25de3` |
| `tests/test_f1_mesh_section_turn_locus_reselection_v55.py` | `3cc6ab9b1009831b1c2e2c4154445670bca7ca7e55a36dbcbb9fea0bf614ed54` |
| `tests/test_f1_localized_turn_corridor_ownership_v63.py` | `77c15b8ba28561ff44192d92a2955e903906fdc286d4e19f30cf2f8605cc737b` |

## Frozen verification

Command:

```text
.venv/bin/python -m pytest -q \
  tests/test_f1_cylindrical_band_completion.py \
  tests/test_f1_mesh_section_turn_locus_reselection_v55.py \
  tests/test_f1_localized_turn_corridor_ownership_v63.py
```

Result: `15 passed in 2.30s`.

The focused suite covers localized-turn acceptance, broad-roll rejection,
nearby-turn isolation, deterministic equal-distance spacing, nonfinite and
nonmonotone fail-closed behavior, deletion-only/order preservation, the
32-candidate capacity stop, the 2 KiB public-audit bound, transient support
nonserialization, and unchanged default V55 behavior.

## Prospective V5R3 rule

V5R3 must be newly constructed after this lock, retain triangle connectivity,
and pass a truth-free native-resolution visual pre-gate against real corpus
references before inference. If its morphology, topology, sampling density,
occlusion, or feature vocabulary still resembles the rejected toy-like V5/V5R2
lane, the entire batch is rejected before inference and replaced without
consuming the V63 scientific run. Once inference begins, cases are terminal:
no rerun, replacement, or rescue tuning.
