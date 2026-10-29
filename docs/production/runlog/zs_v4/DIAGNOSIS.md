# v1 vs v3 error taxonomy on the v4 pool (2026-10-29)

These are the pinned lanes as stored: zs_lane_v1 pin `0087fb23…` and zs_lane_v3 rev3 pin `711bd079…`, from the coverage run (seq 41) and FE1 (RUN seq 44). They are scored on all 217 pool scans. The numbers are **post-hoc**, and DEV is in-sample for both lanes.

Each FN and FP was attributed to a pipeline component by replaying the pinned selection on the candidate cache, step by step. The replay reproduces the stored outputs:
- v1: 217/217 exactly;
- v3: 215/217 exactly; the other 2 differ by ≤ 0.14 µm with identical scores (`results/CACHE_VALIDATION.json`).

Tools: `tools/diagnose.py` (writes `results/DIAGNOSIS.json`) and `tools/span_analysis.py` (writes `results/SPAN_ANALYSIS.json`).

## Totals

| Lane | TP / FP / FN | micro | family-macro |
|---|---|---|---|
| v1 | 1028 / 73 / 120 | 0.914 | 0.925 |
| v3 | 1023 / 66 / 125 | 0.915 | 0.926 |

## FN by component

| Component | What it means | v1 | v3 |
|---|---|---|---|
| NO_CAND:SPAN | a pooled candidate lies on the fold, but its span fails the 60 % gate | 51 | 46 |
| CLS_REJECT | a matching candidate exists, but the GBM gives p < 0.5 | 46 | 45 |
| NO_CAND:FAR | no candidate near the gold line (hems, short ears and others) | 19 | 15 |
| NO_CAND:OFFSET | the nearest candidate is > 4 mm away | 1 | 4 |
| EXT_DEDUPE | selected, then dropped by EXT's dedupe | – | 9 |
| EXT_ALTERED | selected, then EXT extended it past the gates | – | 3 |
| TAKEN | one-to-one assignment | 2 | 2 |
| SNAP_LOST | matches raw, not after the D1 snap | 1 | 1 |

## FP by component

| | v1 | v3 |
|---|---|---|
| Classifier-selected candidate | 73 | 58 |
| EXT line | – | 6 |
| Z3R rescue | – | 1 |
| v2 join | – | 1 |

FP causes for v1:
- SPAN < 60 %: 33
- SPURIOUS: 25
- duplicates (TAKEN_BY_OTHER): 13
- offset + span: 2

By source (v1): z2+z3 52, z3 16, z2 5.

## What v3 changed, line by line

### Where v3 loses to v1

- **31367, `gb_68cd06`, lanced tabs: 9 FN, all EXT_DEDUPE.** This is the set-C loss, caused by EXT plus its dedupe:
  - Z2 line 3 is 413 mm long and runs along the whole row of 14 separate tabs.
  - It pooled onto a 52.5 mm Z3 track that already merges two tabs; that track is an FP in both lanes.
  - EXT (Z2 line ≥ 1.3× and ≥ 0.6 contained) extended that track to 413 mm.
  - EXT's dedupe then dropped the 9 selected tab lines, each now "duplicated" by the 413 mm line.
  - Z3R would have re-added tabs 7 and 12 (p_z3 0.94 and 0.955), but they too duplicate the EXT line.

  The **v2 join was not involved**: every tab pair has flank-B plates that are not connected (cb = False), so nothing was joined. Without EXT, v3 would have matched 14 lines here (3 long + 9 tabs + 2 Z3R tabs), against v1's 12.
- **48340, `sc_521e2ee69b21` B1, and 43780, `sc_1d920838b34b` B1:** EXT_ALTERED. EXT extended a correct line beyond the gold extent. 43780 is a one-line part, so that one extension costs a whole family in the macro.
- **48900:** CLS_REJECT. The v3 GBM differs from v1's.
- **47949, `gb_42dd99`:** 1 extra FP. It is a 740 mm Z3-only line that v3's GBM selected on the budget-resampled mesh (v3 has 601 candidates there, v1 587 at full resolution). It is a classifier FP, not EXT or Z3R.

### Where v3 wins

| Family | Scans | How |
|---|---|---|
| 47085 (N1) | | EXT fixed 2 SPAN misses; Z3R rescued B14 |
| 12851, 48758, 48910 | TEST-B | EXT fixed a too-short line |
| 22352 | | classifier |

### Conclusion on EXT

EXT is right when Z3 lost part of a continuous fold. It is wrong when the Z2 extent runs over separate segments (tabs) or past where the fold ends. The cost is asymmetric: one bad extension can drop many lines.

## SPAN misses (51 for v1): what the right extent would have been

| Kind | Count | Fix that would have matched |
|---|---|---|
| too long | 21 | 12: the pooled Z2 line itself; 9: clipping |
| too short | 25 | 17: the union with collinear neighbours (join); 13: the Z2 extent or the union with it |
| both | 4 | – |

In 25 cases the Z2 line that pooling discarded matches the gold as is.

## Hems and short ears

- **Hems.** There are 11 hem gold lines: 10 in FE1, 1 in DEV. Neither lane emits any; this is a generator limit (paper `th:cor-floors` iv).
- **Short ears.** The FE1 short ears are 5–6 mm folds, and one marker did not mark them.
  - `gb_138090`: candidates exist with p ≈ 0.001.
  - `gb_1f3fe0`: no candidate has the right span.
