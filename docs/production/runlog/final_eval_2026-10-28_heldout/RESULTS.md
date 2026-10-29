# FE1 final held-out evaluation, 2026-10-28: results

This was a one-time run of both frozen zero-shot lanes, blind and without CAD, on the 27 scored FE1 scans. Each arm ran once per scan. Nothing was re-run or tuned. The evaluation is descriptive: no decision rule was pre-registered, and the promotion test was not invoked.

- **Ledger:** freeze seq 35 (authorization), 36 (engines, matcher and bootstrap), 37 (PREREG `c50f58e0…`), 42 (gold frozen; root manifest `70876b56…`), 43 (S1–S3 declared) and **44 (RUN `frz_35b0e2ad7d0a`, written before the first lane process)**. A NOTE after this file binds the sha256 of `results.json` and of this file.
- **Pre-run checks, all passed:**
  - both pins verify (`[]`): v3 rev3 pin `711bd079…` with fingerprint `871f23fa…`, and v1 pin `0087fb23…` with fingerprint `c712c15c…`;
  - the model, config, matcher and guard shas match seq 36;
  - the gold-root manifest sha matches seq 42 (`70876b56…`);
  - the gold-ledger sha matches seq 42 (`8241a85e…`);
  - the freeze ledger (43 entries) and the gold ledger (192 entries) verify;
  - all 27 staged copies match the gold-ledger sha256, and all 27 labels match the manifest.
- **Records:**
  - per-run outputs: `runs/<arm>/<gid>.json`, with lines, wall time, peak RSS and t;
  - scan list: `runs/SCANS.tsv`;
  - driver log: `runs/run_fe1.log`;
  - scored results: `results.json`, with per-case records in `runs/RESULTS_per_case.json`;
  - tools: `tools/run_fe1.py`, `tools/score_fe1.py` and `tools/visual_fe1.py`;
  - visual check: `visual/index.html`.

## Primary (PREREG §5): line-F1, 25 line scans, 17 families

The 95 % CIs come from a family bootstrap (unit part_id[:5], n 10000, seed 20261018).

| Arm | TP / FP / FN | P | R | Micro line-F1 [95 % CI] | Family-macro F1 [95 % CI] | Failures |
|---|---|---|---|---|---|---|
| **zs_lane_v3 rev3 (primary)** | 130 / 18 / 56 | 0.878 | 0.699 | **0.778 [0.583, 0.942]** | **0.860 [0.757, 0.943]** | 0 / 27 |
| zs_lane_v1 (comparator) | 140 / 17 / 46 | 0.892 | 0.753 | 0.816 [0.697, 0.945] | 0.885 [0.810, 0.953] | 0 / 27 |

**Paired v3 − v1** (paired family bootstrap over the same 17 families, n 10000, seed 20261018):

| | Δ | 95 % CI |
|---|---|---|
| Micro | −0.038 | [−0.127, +0.005] |
| Macro | −0.024 | [−0.069, +0.001] |

- Both intervals include 0.
- v3 is better on 1 family (16231: 0.655 → 0.667).
- v3 is worse on 2 families:
  - 31367: 0.571 → 0.255. On the lanced-tab strips `gb_68cd06`, v3 found 3 of 20 lines; v1 found 12 of 20.
  - 47949: 1.000 → 0.889, from one extra line on `gb_42dd99`.
- On every other family the two arms score the same.

Counting the predicted lines on zero-bend scans as FPs does not change either micro, because there are 0 such lines. No scan was `EXCLUDED_BEFORE_SCORING`, so the strict micro equals the micro.

## Sensitivity (declared at seq 43, reported regardless of direction)

| Variant | Gold lines dropped | v3 micro [CI] | v3 macro [CI] | v1 micro [CI] | v1 macro [CI] | Δ micro (v3 − v1) [CI] |
|---|---|---|---|---|---|---|
| Primary (frozen gold) | 0 | 0.778 [0.583, 0.942] | 0.860 [0.757, 0.943] | 0.816 [0.697, 0.945] | 0.885 [0.810, 0.953] | −0.038 [−0.127, +0.005] |
| **S1:** no NC1 open hems, no NC4 hem | 7 | 0.795 [0.597, 0.950] | 0.876 [0.772, 0.958] | 0.833 [0.716, 0.953] | 0.900 [0.826, 0.965] | −0.038 [−0.129, +0.005] |
| **S2:** no 180° hems at all | 10 | 0.803 [0.613, 0.950] | 0.877 [0.776, 0.958] | 0.841 [0.733, 0.953] | 0.902 [0.831, 0.965] | −0.038 [−0.131, +0.005] |
| **S3:** unseen families only (21 scans, 15 families) | n/a | 0.788 [0.566, 0.970] | 0.882 [0.772, 0.967] | 0.831 [0.696, 0.975] | 0.910 [0.833, 0.974] | −0.043 [−0.147, +0.006] |

- **The hem lines.**
  - Neither arm matched any of the 10 hem gold lines. S1 and S2 therefore lower FN only, by 7 and 10 lines, in both arms.
  - The dropped lines are listed in `results.json` under `sensitivity`. The derived gold copies are in `runs/derived_gold/`; the frozen gold was not edited.
- **S3 per arm** (unseen families, excluding group5 38173 and 26687):

  | Arm | TP / FP / FN |
  |---|---|
  | v3 | 115 / 17 / 45 |
  | v1 | 125 / 16 / 35 |

- **The 4 seen-family scans on their own:** micro 0.714 for both arms.

## Secondary (PREREG §5)

### Sub-sets

| Sub-set | Line scans / families | v3 TP/FP/FN | v3 micro [CI] | v3 macro | v1 TP/FP/FN | v1 micro [CI] | v1 macro |
|---|---|---|---|---|---|---|---|
| HOLDOUT10 | 9 / 9 | 26/4/0 | 0.929 [0.845, 1.0] | 0.957 | 26/3/0 | 0.945 [0.853, 1.0] | 0.970 |
| N001–N014 | 12 / 6 (+2 zero-bend) | 45/6/46 | 0.634 [0.396, 0.889] | 0.724 | 55/5/36 | 0.729 [0.618, 0.889] | 0.777 |
| EHO4 | 4 / 2 | 59/8/10 | 0.868 [0.667, 1.0] | 0.833 | 59/9/10 | 0.861 [0.655, 1.0] | 0.827 |

### Zero-bend and rolled parts

- **Zero-bend scans** (scorer disposition ZERO_BEND_CONTROL): there are 2, `gb_61b6da` and `gb_cdecd9`, both adjudicated rolled. Both arms predict **0 lines** on each.
- **False lines on rolled or flat parts:** 0 in both arms. The only scans adjudicated rolled or flat are those two.
- `gb_42dd99` is adjudicated mixed, with rolled facets. It is not in the rolled-or-flat count.
  - v3 has 1 FP there; v1 has 0.

### Where the misses are

FN causes:

| Arm | MISSED | SPAN < 60 % |
|---|---|---|
| v3 | 26 | 30 |
| v1 | 26 | 20 |

Most of the FN come from three groups:

| Scans | Gold lines | v3 found | v1 found |
|---|---|---|---|
| Lanced-tab strips `gb_68cd06` + `gb_959f4a` | 39 (both include 2 end hems) | 6 | 16 |
| The 10 hem lines | 10 | 0 | 0 |
| Short ears on `gb_138090` and `gb_1f3fe0` | 4 | 0 | 0 |
| Ladder webs `gb_b3ae40` and `gb_b97c8e` | 18 | 11 | 11 |

On `gb_68cd06`, v3 drew one continuous line through the row of 14 separate lanced tabs. That line is an FP, and the 14 tab lines are FN; `visual/gb_68cd06.png` shows it.

### Angles

The angle difference is the engine's included angle minus the gold's. The joggle `gb_437879` is flagged: its 2 lines are −2.2° and −2.0° in both arms. No hem line was matched.

| Arm | Matched lines, joggle excluded | Median abs | p90 abs | Within 1° | Mean signed |
|---|---|---|---|---|---|
| v3 | 128 | 0.09° | 0.40° | 126 | +0.29° |
| v1 | 138 | 0.09° | 0.39° | 137 | −0.04° |

- **v3 outlier:** on `gb_bbc3a8` B1, the line is in the right place (offset 0.14 mm) but its angle is wrong: engine 133.4° against gold 90.1°, +43.3°.
- **v1:** the largest difference is −2.9°, on `gb_42dd99` B3. v3 has the same −2.9° on that line.

### Runtime

`wall_s` is the time inside the worker, including the t estimate, and does not include the slot wait. The process limits were 16 GiB of RSS, 32 GiB for `gb_42dd99` and `gb_bbc3a8`, and 1800 s.

| Arm | Wall p50 | Wall p95 | Wall max | Total | Peak RSS p50 | Peak RSS max |
|---|---|---|---|---|---|---|
| v3 | 7.6 s | 32.8 s | 48.4 s (`gb_42dd99`) | 249 s | 1.5 GiB | 2.75 GiB |
| v1 | 10.8 s | 29.8 s | 222.8 s (`gb_42dd99`, 7.4 M faces, includes the Z2 t estimate) | 492 s | 1.6 GiB | 4.79 GiB |

- v3 reuses the t estimate from the v1 job on the same bytes, as frozen at seq 41.
- No job hit a limit.
- **Failures: none.** All 54 jobs returned status OK, with pins verified and no unpinned code loaded.

### Marker agreement and canary

These numbers are copied from the adjudication (`adjudication/ADJUDICATION_FE1.json`, seq 42).

- **Canary `gb_9528f8`:** marker A F1 1.000, marker B F1 1.000. It passed, and no canary label was written.
- **Marker A vs B:**

  | Set | Micro line-F1 | TP / FP / FN |
  |---|---|---|
  | 27 scored scans, line scans only | 0.9555 | 161 / 0 / 15 |
  | 27 scored scans, B's lines on the zero-bend scan counted as FP | 0.9499 | 161 / 2 / 15 |

- **Macro item-F1:** 0.923.
- **The one zero-bend disagreement:** `gb_437879`, adjudicated as a 2-line joggle.

### V/X pairs (context only, not a defect metric)

The engine line counts for v3 and v1 are given as V / X.

| V | X | Gold | v3 | v1 |
|---|---|---|---|---|
| gb_b21bad | gb_04738b | 6 / 6 | 7 / 8 | 7 / 8 |
| gb_138090 | gb_1f3fe0 | 4 / 4 | 2 / 2 | 2 / 2 |
| gb_1baa22 | gb_d11e33 | 3 / 3 | 2 / 2 | 2 / 2 |
| gb_231669 | gb_ac2239 | 14 / 14 | 13 / 13 | 14 / 13 |
| gb_44ddfb | gb_a0538b | 4 / 4 | 4 / 4 | 4 / 4 |
| gb_cdecd9 | gb_61b6da | 0 / 0 | 0 / 0 | 0 / 0 |
| gb_68cd06 | gb_959f4a | 20 / 19 | 4 / 4 | 13 / 4 |
| gb_d1093e | gb_a2dd83 | 20 / 21 | 20 / 21 | 20 / 21 |
| gb_b97c8e | gb_b3ae40 | 9 / 9 | 5 / 7 | 5 / 7 |

## Pooled v1 held-out: TEST-B (61) + N1 (4) + FE1 (27)

- **Sources:**
  - TEST-B and N1 come from their recorded one-time per-case records: `testb/RESULTS_per_case.json` `zs|testb` and `newscans_N1/RESULTS_per_case.json` `zs_lane_v1:zs (primary)`. Neither was re-run.
  - FE1 is this run's v1 arm.
- **Bootstrap:** family = part_id[:5], n 10000, seed 20261018.

| Pool | Line scans / families | TP / FP / FN | Micro [95 % CI] | Family-macro [95 % CI] |
|---|---|---|---|---|
| TEST-B | 61 / 49 | 266 / 39 / 51 | 0.855 [0.758, 0.928] | 0.852 [0.779, 0.915] |
| N1 | 1 / 1 (+3 zero-bend, 0 FP) | 21 / 2 / 5 | 0.857 | 0.857 |
| FE1 | 25 / 17 (+2 zero-bend, 0 FP) | 140 / 17 / 46 | 0.816 [0.697, 0.945] | 0.885 [0.810, 0.953] |
| **Pooled** | **87 / 66** (+5 zero-bend, 0 FP) | **427 / 58 / 102** | **0.842 [0.768, 0.910]** | **0.864 [0.807, 0.915]** |

- With the recorded family ids (68 families), the pool gives micro 0.842 [0.771, 0.907] and macro 0.862 [0.805, 0.913].
- **Caveats:**
  - The TEST-B and N1 v1 runs used the research lane CLI under the pin of the day. That pin had fingerprint `af87df2f`.
  - Lane code, models and config are unchanged since then. The re-pins of 10-26 and 10-28 re-hashed E3 files only.
  - FE1 ran `lane_cli.run_lane` blind.
  - Group 26687 is in both TEST-B and FE1, so it is one bootstrap unit.

## Visual check

`visual/index.html` has one iso image per scan for the primary arm (zs_lane_v3), labelled by gid only. The key:

| Colour | Meaning |
|---|---|
| Green, with the engine line dashed dark green | Gold line found |
| Orange | Gold line missed (FN) |
| Red | Engine line with no gold (FP) |

The images were rendered through the slot with `tools/visual_fe1.py`, using the N1 rendering helpers.

## Deviations and run notes

1. **Run entry point.**
   - PREREG §3 names the research `run_cohort.sh` tools.
   - Per the coordinator's instruction, the run used the production `lane_cli.run_lane` and `run_lane_v3` through the unchanged seq-41 worker (`cov_worker.py`, `100bdd99…`). That is the same pinned code, models and blind configuration as seq 34 and seq 41.
   - This is recorded in RUN seq 44.
2. **Shared slot.**
   - The seq-41 coverage run was still running, so FE1 jobs queued on the shared one-at-a-time slot lock. The longest wait was ~8.5 min, for the first job.
   - Jobs never overlapped. `wall_s` excludes the wait.
   - The driver's 7200 s outer limit includes the wait, and it was never reached.
3. **Per-job limit.** The 1800 s limit from the seq-41 budget was enforced inside the slot with `perl alarm`. No job came near it.
4. **Hem angles.** The hem gold lines use the D0 tangent-band construction, so their gold "incl" is not the hem angle. They are excluded from the angle summary. None was matched anyway.
5. **Consumed.** All 27 scored scans and the canary's batch are now consumed (PREREG §1).
