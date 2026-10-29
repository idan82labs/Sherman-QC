# Z3 axis-family slicing: iteration log (DEV only)

Code: `research/zeroshot/z3_slicing/`. Tests: `tests/test_prod_zeroshot_z3_slicing.py`.
Matcher: the frozen `score.py score_case` (5° / 4 mm / 0.6 of both spans, Hungarian, apex locus, raw frame).
Error budget: `score_broad.diagnose` (nearest counterpart within 15° / 15 mm).
"Candidate recall" = recall when every candidate is emitted (one-to-one matcher), "ratio" = candidates / gold.

Custody notes:
- Only SPLIT_FROZEN `dev.scans` are admitted (`custody.check_dev_scan`); TEST-B rows are dropped when INVENTORY is loaded.
- 18 DEV scans live under `incoming/2026-08-30_shaked_daily_scans/source_download/` (the daily28 consumed scans). The
  frozen split puts them in DEV and INVENTORY marks them usable DEV_CONSUMED. The HOLDOUT10 files in that folder are
  not in DEV, and my guard never lists or opens that folder; it only opens the DEV paths named by INVENTORY.
- Folds: the integrator's `dev_folds.json` (5 family-group folds) is used for everything fitted.
- Every scan also passes the programme harness guard `research/zeroshot/common/gold_guard.require_runnable`.

## Iteration 0: development on an 11-scan DEV debug subset (light runs outside the slot)

The slot semaphore was fully held by the Z4 pool rebuild, so single-scan debugging (1–3 s per scan, < 1.7 GiB) ran
directly with OMP_NUM_THREADS=1 and nice. Every full-DEV run goes through `sherman_slot.sh` (a code snapshot is queued:
`/private/tmp/claude-501/z3work/queue_run.sh`).

| Step | Change (general rule) | Subset cand. recall / ratio | Subset gated F1 |
|---|---|---|---|
| 0a | first pipeline: Gauss-map peaks → axis families → exact mesh sections every 2 mm → DP flats → corners → tracking (12 mm bridge) | 0.80 / 18.3× | 0.24 |
| 0b | track at most 4.5 mm gaps; chain collinear segments afterwards, across a gap only when both flanks continue (window/relief between the same two plates); drop tracks whose apex points do not line up along the family axis (was: forced onto the axis → ANGLE FPs) | 0.96 / 18.0× | 0.43 |
| 0c | sheet-edge walls (a ~t flat between antiparallel flats, or at a chain end) are not flanges; chords of a large-radius arc (significant quadratic bow, R < 8 L) are not flats; a normal peak must be compact (core ≥ 1.2 × ring) so arc bands give no axis families | 0.89 / 2.2× | 0.78 |
| 0d | canonical flank order per corner (cross(da, db) along +u): flank directions were averaged with random sign, which broke the continuity test (bad merges of separate collinear tabs) | 0.96 / 2.2× | 0.85 |

Gate used for "gated F1" here (`gate.simple`): length ≥ 8 mm, ≥ 4 supported slices, support fraction ≥ 0.5,
strip evidence ≥ 0.5, bend angle ≥ 9°. Not fitted; hand-set before any full-DEV score.

## Full-DEV iterations (121 scans: 116 complete-label line scans, 586 gold lines, 5 zero-bend controls)

Scored with `score_run.py` (frozen matcher). "simple" = the hand gate above (not fitted).

| Iter | Change | Cand. recall / ratio | simple-gate P / R / F1 (macro) | FN budget | FP budget | ctrl FP |
|---|---|---|---|---|---|---|
| it1 | iteration-0 code (3 point-cloud scans fail: no faces) | 0.887 / 4.57× | 0.576 / 0.870 / 0.693 (0.682) | MISSED 52, SPAN 19, OFFSET 3, ANGLE 2 | OFFSET 177, SPURIOUS 128, SPAN 52, ANGLE 16 | 22 |
| it2 | thick plates: a short flat between two same-direction turns, one of them < 35°, is a chord of one bend arc (merge); point clouds get a ball-pivoting surface; gold guard wired in | 0.910 / 3.01× | 0.752 / 0.892 / 0.817 (0.767) | MISSED 42, SPAN 16, OFFSET 4, ANGLE 1 | SPURIOUS 107, SPAN 38, OFFSET 19, ANGLE 5 | 21 |

it1 → it2: the 177 OFFSET FPs were the sub-arc corners of large-radius bends on 4–6 mm plate (46127000, 46128000,
49191008, 49783000): the multi-chord arc rule removed them (OFFSET FP 177 → 19).
| it4 | joggles: two parallel flats offset sideways by 0.5t–4t+4 mm through a short path give two opposite bends (web = chord between the tangent points); Gauss map on kNN-smoothed vertex normals (raw sub-mm face normals spread several degrees, which hid small flanges' peaks); peak floor 0.05 % of area, ≤ 20 families; tracks ≥ 9 mm use their own TLS direction (≤ 15° from the family axis) instead of the coarse family axis; sub-slice endpoints (walk 0.25 mm stations while the surface touches the apex line); collinear segments across a gap merge only when each flank's plate is one 8-connected component around the gap (windows bridged, separate tabs kept apart) | 0.966 / 5.16× | 0.698 / 0.942 / 0.802 (0.799) | MISSED 15, SPAN 13, OFFSET 6 | SPURIOUS 142, SPAN 45, OFFSET 37, ANGLE 10 | 29 |

it3 was superseded before it ran (its snapshot lacked the joggle and plate-connectivity rules).

Selection on it4 (out-of-fold, family-grouped 5 folds of `dev_folds.json`; model and threshold chosen on the 4 training folds):

| Selector | P | R | micro F1 [95 % CI] | family-macro F1 [CI] | TP / FP / FN | FN budget | FP budget | ctrl FP |
|---|---|---|---|---|---|---|---|---|
| LR (21 plain features) | 0.898 | 0.920 | 0.909 [0.871, 0.942] | 0.902 [0.856, 0.942] | 539 / 61 / 47 | MISSED 29, SPAN 12, OFFSET 5 | SPURIOUS 26, SPAN 23, OFFSET 7 | 2 |
| GBM (same features) | 0.939 | 0.925 | **0.932** [0.905, 0.955] | **0.914** [0.867, 0.953] | 542 / 35 / 44 | MISSED 25, SPAN 11, OFFSET 8 | SPURIOUS 17, SPAN 11, OFFSET 6 | 4 |

Unfitted birth filter (`gate.birth`: length ≥ 5 mm, ≥ 3 slices, bend ≥ 8°, support ≥ 0.4) on it4:
candidate recall **0.964 at 1.96×** the gold count (all candidates: 0.966 at 5.16×).

| it5 | (reverted) a flat < 1.2t+3 between two same-way turns treated as an arc chord whatever the angles; bevelled sheet ends (antiparallel flats ~t apart) made everything between them edge geometry. Aimed at 48565005 (5 mm plate, 45°/135° pseudo-corners) | 0.957 / 4.73× | 0.704 / 0.930 / 0.801 | MISSED 19, SPAN 14, OFFSET 6, ANGLE 2 | SPURIOUS 140, SPAN 43, OFFSET 30 | 29 |

it5 lost 5 born gold lines (49207000 B1/B2 252 mm 90° folds, a 47247004 tab lip, 23317000 B1, a 48333004 crease)
and did not fix 48565005, so both rules were reverted. OOF GBM on it5: 0.925 / 0.914 (it4: 0.932 / 0.914).
Kept from it5: the arc-chord test now also needs R_fit < 40 mm (a long bowed wall, R ~ 1 m, is never an arc
chord; previously only R < 8·L, which a 200 mm wall with 0.5 mm bow satisfies).

Ablation, slice step 1.5 mm instead of 2 mm (it5 code): candidates 8.3× (was 4.7×), OOF GBM 0.919 / 0.902 —
no gain, 2 mm stays.

| it6 | it4 code + the 40 mm arc-chord radius bound (it5 rules reverted) | 0.966 / 5.27× | 0.696 / 0.944 / 0.801 (0.816) | MISSED 14, SPAN 13, OFFSET 6 | SPURIOUS 145, SPAN 45, OFFSET 35, ANGLE 11 | 10 |

it6 selection (OOF): GBM 0.936 / 0.920 / **0.928** [0.902, 0.950], macro 0.909 [0.863, 0.948], TP 539 FP 37 FN 47,
FN MISSED 29 / SPAN 11 / OFFSET 7, FP SPURIOUS 17 / SPAN 12 / OFFSET 4 / TAKEN 3 / ANGLE 1, control FP 5.
LR 0.910 / macro 0.915. Birth stream (gate.birth): recall 0.964 at 1.93×.
Ablations on it6 (not adopted, within the CI): context features (competing parallel neighbours, coverage by a
longer collinear candidate, scan size, t) LR 0.921 / GBM 0.922; slice step 1.5 mm GBM 0.934 / macro 0.923 at 8.9×
candidates.
Unions on it6 (Z2 = the Z2 WP's `it04` eval_lines, present for 103/121 scans; A0 = phase-1 baseline):
plain NMS union Z3+Z2 0.903 (R 0.944, P 0.865), Z3+A0 0.885, Z3+Z2+A0 0.866; evidence-gated pooled selector
(external lines attached to Z3 candidates as agreement flags, external-only lines as extra candidates, one OOF GBM):
Z3+Z2 **0.937** [0.909, 0.962] / macro 0.924, Z3+Z2+A0 0.928.

| it7 (final) | joggle-aware plate test: a flank shorter than the bend zone + 6 mm (joggle web, small tab) is tested on the flank itself, not beyond it (it6 left the 3.5 mm-web joggle of 38169000 as 12 fragments) | **0.968 / 5.12×** | 0.724 / 0.942 / 0.819 (0.820) | MISSED 14, SPAN 11, OFFSET 9 | SPURIOUS 134, OFFSET 30, SPAN 29, ANGLE 11 | 10 |

it7 selection (OOF, family-grouped folds): GBM P 0.938 / R 0.906 / **F1 0.922** [0.896, 0.946], family-macro
**0.909** [0.865, 0.946]; TP 531 FP 35 FN 55; FN MISSED 32 / OFFSET 11 / SPAN 11 / TAKEN 1; FP SPAN 17 / SPURIOUS
15 / TAKEN 2 / OFFSET 1; control FP 5 (on 5 zero-bend scans). Per fold 0.918 / 0.891 / 0.953 / 0.919 / 0.946.
LR 0.915 / macro 0.912. it4 → it7 selector numbers move within ±0.01 (GBM threshold/fold variance); the geometric
candidate stage improved (simple-gate F1 0.802 → 0.819, candidate SPAN FPs 45 → 29).
Birth stream (gate.birth): recall **0.964 at 1.83×**.
Unions (it7): plain NMS Z3+Z2(it04) 0.903, Z3+A0 0.880; pooled OOF selector Z3+Z2(it05) **0.937** [0.910, 0.961]
/ macro 0.925; Z3+A0 0.926 / 0.911.
Runtime per scan (single thread): median 1.65 s, mean 2.7 s, P95 10.1 s, max 19.7 s; batch peak RSS 3.3 GiB.
Overlays of the 5 worst scans (`z3/overlays/`) were opened and checked: 47247004 (lattice-window fold: corners
absent along most of the fold), 49191019 (sloped segments of a fold that follows a bent web), 40500000 (4 FPs on the
vertical box-corner seams), 11979003 (Z-step inner-skin fold span), 48101000 (return flanges folded against a wall).
