# Jev experiment: Noul bend / not-bend on coordinate-free patch-pair descriptors

Status: **exploratory evidence only**. This is not wired into any engine, gate, threshold or
label. Run on 2026-10-17 with the owner's go-ahead ("maybe [Jev] can help"). Script:
`research/bench_labeling/jev_experiment.py` (`build` / `query` / `evaluate`).

## Question

One TypeSafe System One **Noul** per candidate pair of planar regions:

> Is the `candidate_pair` of planar regions joined by a real press-brake bend (not a
> sheet-thickness edge, hole wall, outline or artifact)?

The `criteria.true` text describes a fold between two flanges: a clear fold angle, a long
straight shared boundary, a small gap filled by a curved band whose normals sweep between
the two regions, and both regions wider than one thickness, on the same skin or opposite
skins. The `criteria.false` text lists five confusers: thickness edge, hole/slot wall, two
skins of one flange, one face split in two, and outline/corner-contact/noise/unrelated
regions. The full text and the shared `state` are in `results.json`.

## What was sent (custody)

- **Sent:** derived scalars and plain-language buckets for each pair. These are sheet thickness;
  whether outward normals are known; the relationship bucket; the angle between normals,
  the fold angle and the included angle; convex or concave corner; and for each region its
  area, length, width, width in thicknesses, thin-strip flag, share of the largest region,
  flatness RMS and normal spread. The pair-level values are the area ratio, shared-boundary
  length and its share of each region's length, closest and typical edge gap (with
  thickness buckets), the distance from the plane-intersection line to the near edges,
  transition-band area, width and normal-sweep share, the parallel-plane offset, and the
  segmenter's opposite-skin-twin flag.
- **Never sent:** coordinates, normal vectors, centroids, file names, SKUs, scan ids or patch
  ids. Before sending, `_assert_coordinate_free` rejects any list value, any forbidden key and
  any `sc_` or `.ply` text. Question keys are opaque hashes.
- **Data sources:** only the prepped artifacts under `label_prep/` (patches.json,
  points_ds.npz, prep_manifest.json), plus `label_lines/*/lines.json` and BENCHMARK_V2.json
  for labels. No raw scan was opened. Each prep manifest's `scan_path` went through the
  name-only `custody.check_path` guard. The label_prep selection had already excluded
  HOLDOUT10, sealed, N001-N014, EHO4, authorization and Eyal-50 material.
- **API key:** read from `.env.local` inside the script and never printed or logged.

## Candidates and labels

- **Candidate universe:** every adjacent patch pair (the `neighbors` graph from prep). This
  is a superset of prep's `candidate_pairs`, and it includes parallel twins, coplanar splits
  and thin strips as confusers. prep's `candidate_pairs` missed one labelled bend (scan
  sc_9018b86ccc9a, pair 1-3).
- **Size:** 1403 candidates, under the 1500 cap.
  - 1047 come from all 10 AI_GOLD_ACCEPTED scans: 5 bend-line scans with 32 labelled bends,
    and 5 zero-bend controls.
  - 356 come from 6 NEEDS_HUMAN_REVIEW scans whose verifier saw images. These are reported
    separately as `secondary_unverified` and are **not** evidence: their labels failed the
    strict gate (sc_822f083e3dfc has 0 labelled bends but 2 missed bends were seen).
- **Label variants (accepted tier):**
  - `flange` (primary): the pair straddles a labelled bend, with either skin of each flange
    allowed through the segmenter's twin links. 131 positives.
  - `strict`: only the exact labelled patch pair is positive. 32 positives. Its 99
    other-skin pairs count as negatives, which they physically are not.
  - `strict_excl_ambiguous`: strict positives, with those 99 other-skin pairs dropped. 32
    positives out of 948.
- **Models:** `jev-1.13.0` (= `jev-latest`) and `jev-1.14.0` (= `jev-preview`), pinned by ID.
  Requests used 40 Nouls each and the order was shuffled with a fixed seed. Each model took
  36 requests and about 1.34 M input tokens, roughly $0.11 for both models together. There
  was 1 transient `529 system_overloaded` on 1.14, which was retried successfully. The API
  was available throughout.

## Results: accepted scans (the only tier that counts)

| model | labels | n | pos | ROC-AUC | 95% CI (scan bootstrap) | best-F1 thr | P | R | F1 | P@0.5 | R@0.5 | Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev-1.13.0 | flange | 1047 | 131 | **0.945** | [0.804, 0.979] | 0.70 | 0.557 | 0.786 | 0.652 | 0.343 | 0.992 | 0.161 |
| jev-1.14.0 | flange | 1047 | 131 | 0.922 | [0.737, 0.978] | 0.57 | 0.467 | 0.916 | 0.619 | 0.432 | 0.962 | 0.131 |
| jev-1.13.0 | strict_excl_ambiguous | 948 | 32 | 0.939 | [0.777, 0.970] | 0.79 | 0.536 | 0.469 | 0.500 | 0.111 | 0.969 | 0.170 |
| jev-1.14.0 | strict_excl_ambiguous | 948 | 32 | 0.928 | [0.739, 0.974] | 0.71 | 0.284 | 0.594 | 0.384 | 0.153 | 0.938 | 0.132 |
| jev-1.13.0 | strict | 1047 | 32 | 0.897 | [0.732, 0.929] | 0.79 | 0.231 | 0.469 | 0.309 | 0.082 | 0.969 | 0.208 |
| jev-1.14.0 | strict | 1047 | 32 | 0.893 | [0.713, 0.933] | 0.71 | 0.186 | 0.594 | 0.284 | 0.103 | 0.938 | 0.164 |

The best-F1 threshold was picked on the same data, so those P/R/F1 values are optimistic.
No answer reached p ≥ 0.9; the highest was 0.85.

### Calibration (accepted, flange labels)

| bin | n (1.13) | mean p | observed yes (n) | n (1.14) | mean p | observed yes (n) |
|---|---|---|---|---|---|---|
| 0.0-0.1 | 152 | 0.063 | 0.000 (0) | 175 | 0.059 | 0.000 (0) |
| 0.1-0.2 | 195 | 0.152 | 0.000 (0) | 234 | 0.142 | 0.000 (0) |
| 0.2-0.3 | 65 | 0.232 | 0.000 (0) | 71 | 0.264 | 0.000 (0) |
| 0.3-0.4 | 66 | 0.357 | 0.000 (0) | 178 | 0.341 | 0.017 (3) |
| 0.4-0.5 | 190 | 0.448 | 0.005 (1) | 97 | 0.444 | 0.021 (2) |
| 0.5-0.6 | 101 | 0.534 | 0.050 (5) | 56 | 0.552 | 0.268 (15) |
| 0.6-0.7 | 93 | 0.651 | 0.237 (22) | 125 | 0.648 | 0.432 (54) |
| 0.7-0.8 | 129 | 0.740 | 0.457 (59) | 93 | 0.742 | 0.473 (44) |
| 0.8-0.9 | 56 | 0.831 | 0.786 (44) | 18 | 0.810 | 0.722 (13) |

The p values are not calibrated for this task. Below 0.5 the model is under-confident in the
safe direction: almost nothing there is a bend. Between 0.5 and 0.8 it is over-confident.
The real decision band is about 0.6-0.85 and has no high-confidence tail.

### Where the probability goes (accepted, flange labels)

| pair category | n | mean p 1.13 | share p≥0.5 1.13 | mean p 1.14 | share p≥0.5 1.14 |
|---|---|---|---|---|---|
| opposite-skin twin | 67 | 0.067 | 0.000 | 0.062 | 0.000 |
| parallel / coplanar | 114 | 0.074 | 0.000 | 0.062 | 0.000 |
| involves thin strip (thickness edge / hole wall) | 519 | 0.340 | 0.141 | 0.268 | 0.015 |
| non-parallel, both wide (negatives) | 216 | 0.611 | 0.815 | 0.566 | 0.731 |
| **POSITIVE (flange-level bend)** | 131 | 0.748 | 0.992 | 0.678 | 0.962 |

Jev handles the confusers that were named well: twins, coplanar splits and (on 1.14) thin
strips. The remaining errors are "non-parallel, both wide" negatives. **150 of those 216 come
from the zero-bend controls.** Their verifier summaries call all four multi-patch controls
"rolled half-cylinder, no discrete bends", and three of them are legacy reference meshes
with no part number. The segmenter cuts a rolled surface into flat facet strips, and
neighbouring facets meet at fold angles of about 10-17° (or 162-168°) with a smooth band
between them. Jev scores these 0.64 on average (1.13) against 0.75 for real bends.
"Faceting of a rolled or curved surface" was not one of the confusers in the criteria. The
other 66 come from bend scans: unlabelled pairs at about 90° (5-95% range 89.6-90.4°, for
example flange-to-flange corners), which the angle alone cannot separate from the positives
(89.1-90.9°).

### Per scan (accepted)

| scan | role | pairs | flange pos | AUC 1.13 | AUC 1.14 | p≥0.5 (1.13) | p≥0.5 (1.14) | max p neg (1.13) | min p pos (1.13) |
|---|---|---|---|---|---|---|---|---|---|
| sc_752e2bd54aad | bend_lines | 579 | 39 | 0.993 | 0.993 | 101 | 46 | 0.66 | 0.46 |
| sc_e080ae094f35 | bend_lines | 143 | 48 | 0.956 | 0.952 | 83 | 65 | 0.84 | 0.67 |
| sc_9567d0d3f33d | bend_lines | 68 | 28 | 0.973 | 0.992 | 44 | 43 | 0.76 | 0.70 |
| sc_85a7f8cca6bc | bend_lines | 24 | 12 | 1.000 | 1.000 | 14 | 12 | 0.54 | 0.61 |
| sc_9018b86ccc9a | bend_lines (point cloud, unoriented) | 6 | 4 | 1.000 | 1.000 | 4 | 0 | 0.07 | 0.50 |
| sc_b4719db3dd69 | zero_bend_control (rolled) | 129 | 0 | - | - | 83 | 76 | 0.82 | - |
| sc_2e584e9d1ffe | zero_bend_control (rolled) | 46 | 0 | - | - | 15 | 14 | 0.80 | - |
| sc_7a253806b10b | zero_bend_control (rolled) | 22 | 0 | - | - | 16 | 17 | 0.85 | - |
| sc_f098dbceaa6d | zero_bend_control (rolled) | 29 | 0 | - | - | 19 | 19 | 0.78 | - |
| sc_f99d883f2754 | zero_bend_control (flat blank) | 1 | 0 | - | - | 0 | 0 | 0.07 | - |

Inside one bend scan, the ranking is close to perfect. Across scans it is not: rolled-part
facets score as high as real bends, so no global threshold works. sc_9018b86ccc9a is the
point cloud with unknown normal signs. There 1.14 put every true bend below 0.5 (min 0.39),
while 1.13 put them at 0.50 or above.

### Against no-model baselines (accepted, flange labels)

| scorer | ROC-AUC | P@rule / P@0.5 | R |
|---|---|---|---|
| prep `candidate_pairs` membership | - | 0.200 | 0.969 |
| hand rule: non-parallel, no thin strip, not twins | - | 0.378 | 1.000 |
| Jev 1.13 p≥0.5 AND hand rule | - | 0.425 | 0.992 |
| Jev 1.13 inside the hand-rule set (347 pairs, 131 pos) | 0.778 | best-F1 0.652 | |
| Jev 1.14 inside the hand-rule set | 0.687 | best-F1 0.622 | |
| Jev 1.13 / 1.14 inside the rule set, bend scans only (197 pairs) | 0.822 / 0.866 | | |
| single feature inside the rule set: larger typical edge gap (direction chosen post hoc) | 0.844 | | |
| single feature inside the rule set: max shared-boundary share | 0.677 | | |
| single feature inside the rule set: fold angle | 0.628 | | |

Most of the headline AUC of about 0.94 comes from rejecting parallel, twin and thin-strip
pairs, which a three-line hand rule also does with 100% recall. On the hard residual, the
wide non-parallel pairs, Jev 1.13 reaches AUC 0.78. One post-hoc single descriptor (larger
typical edge gap) does better at 0.84. Jev's added value over code is therefore small and not
yet shown.

## Secondary tier (unverified labels; context only, not evidence)

Flange labels: AUC 0.905 for 1.13 and 0.912 for 1.14. Best-F1 is 0.85 at threshold 0.52
(1.13) and 0.87 at 0.41 (1.14). The hand rule alone gives P 0.784 and R 1.0. Per-scan AUC
ranges from 0.74 (sc_a5e534e6097c, point cloud) to 1.0. These labels did not pass the strict
gate, so the numbers are not reliable.

## Reading

1. **API availability:** the API worked. Two pinned model versions ran, with one transient
   529 that was retried.
2. **Named confusers:** given clean coordinate-free descriptors, Jev reliably rejects the
   confusers named in the criteria: twins, coplanar splits and, on 1.14, thickness/hole-wall
   strips. Within a single sheet-metal scan it ranks real bends above confusers (per-scan
   AUC 0.95-1.0).
3. **Not usable as a classifier as asked:**
   - Its probabilities are not calibrated for this task and never go above 0.85.
   - Shallow facets of rolled parts get bend-like scores, so global precision at useful recall
     is about 0.35-0.56.
   - On the hard subset it does not beat one simple descriptor.
4. **Small evidence base:** all of this rests on 5 bend scans (32 bends) and 5 controls, four
   of them rolled parts that look like the same kind of object. The scan-bootstrap CIs are
   wide (for example 0.80-0.98).
5. **If anyone continues (optional, not recommended as a priority):**
   - Add "facets of a rolled/curved surface (chains of small fold angles)" as a named
     confuser.
   - Send part-level context: how many fold angles the part has, and whether they chain.
   - Treat p as a feature for a small classical model rather than as a threshold, as the
     docs' composite-scoring pattern suggests.

   Any such change must be judged on scans **other** than these 10, because the confuser was
   identified on them.

## Files

- `candidates.jsonl`: per-candidate local key, descriptor (exactly what was sent), labels,
  and prep-candidate flag. The local scan id and patch pair are kept for joins; they were not
  sent.
- `raw_jev-1.13.0.jsonl`, `raw_jev-1.14.0.jsonl`: one line per request, with answers keyed
  by candidate hash, model, usage, latency and request id (failures included).
- `scores.jsonl`: merged p(yes) per model with labels.
- `results.json`: all metrics, the per-scan table, calibration and baselines.
- `selection.json`: the scans used, their tiers and pair counts.

Reproduce (light CPU, about 2 s for `build`; `query` resumes from the raw jsonl):

```
OMP_NUM_THREADS=1 .venv/bin/python -m research.bench_labeling.jev_experiment build
.venv/bin/python -m research.bench_labeling.jev_experiment query --model jev-1.13.0
.venv/bin/python -m research.bench_labeling.jev_experiment query --model jev-1.14.0
.venv/bin/python -m research.bench_labeling.jev_experiment evaluate --models jev-1.13.0 jev-1.14.0
```
