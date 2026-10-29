# Z3: axis-family slicing (zero-shot birth channel), DEV results, 2026-10-25

> **Ledger: consumed development.** Every number is on the 121 SPLIT_FROZEN DEV scans (116 line scans, 586 gold
> lines, 91 families, 5 zero-bend controls). Nothing fitted is reported in-sample: selectors are out-of-fold over the
> integrator's family-grouped `dev_folds.json`. No CAD is used anywhere. No TEST-B scan, label or output was opened.

Code: `research/zeroshot/z3_slicing/` (fingerprint `6cff04a9fb34f710`). Tests: `tests/test_prod_zeroshot_z3_slicing.py` (5 synthetic checks).
Iterations and every rule change: `ITER_LOG.md`. Machine-readable summary: `RESULTS.json`.

## Method (no part-specific logic; thresholds in mm, degrees or multiples of the measured t)

1. **Axis families.** Gauss map of kNN-smoothed vertex normals, then compact peaks (flanges) on the projective sphere. Family axes are the clustered cross products of peak pairs, ranked by pair area plus great-circle (bend-arc) support. There are at most 20 families.
2. **Slicing.** Exact mesh/plane sections every 2 mm perpendicular to each axis. They are chained through shared mesh edges into 2-D polylines that carry the outward face normals. Point-cloud scans get a ball-pivoting surface first.
3. **Corners.** Douglas–Peucker plus a collinear merge give the straight runs. The following are not flanges:
   - arc chords (significant quadratic bow, R < min(8 L, 40 mm));
   - sheet-edge walls (a flat about t long between antiparallel flats, or at a chain end).

   Several chords of one arc merge into one bend. Two parallel flats offset sideways become a **joggle**, which is two opposite bends. The included angle comes from the two flat directions. The apex is where the polyline crosses the corner bisector. The skin comes from the outward normals: an inner-skin corner is moved t along −bisector to its outer apex, and per-slice duplicates on the two skins are merged.
4. **Tracking.** Hungarian linking across slices, with gaps of 4.5 mm bridged. A TLS 3-D line goes through the apices. The span comes from slice support, then refined to sub-slice precision. Collinear segments are chained across a gap only when **each flank's plate is one 8-connected component around the gap**: windows and reliefs between the same two plates are bridged, while separate tabs on one base stay separate lines.
5. **Bend-strip evidence and selection.** A strip/flank evidence score is computed for each candidate, but on DEV it is ~1.0 almost everywhere, so it does not discriminate (see Next steps). Selection uses either:
   - the unfitted `gate.birth`, for the candidate stream;
   - or an out-of-fold GBM on 21 plain geometric features (length, support, bend angle, flank lengths, skin fractions, residuals, rank).
6. **Output.** `eval_lines.json` (`sherman.broad_baseline.eval_lines.v1`, raw scan frame, outer-skin apex locus), one per scan, in `runs/it7/<sid>/`. Alongside it:
   - `candidates_eval_lines.json`: the birth stream, scored with the out-of-fold probability;
   - `run_record.json`.

## Results (DEV, final code it7)

| What | P | R | micro F1 [95 % CI] | family-macro F1 [CI] |
|---|---|---|---|---|
| A0 baseline (phase 1) | 0.600 | 0.220 | 0.322 [0.264, 0.379] | 0.267 |
| **Z3 candidate stream** (`gate.birth`, unfitted) | 0.526 | **0.964** | (1.83× gold count) | |
| Z3 all candidates | | 0.968 | (5.12×) | |
| **Z3 standalone, OOF GBM selector** | 0.938 | 0.906 | **0.922** [0.896, 0.946] | **0.909** [0.865, 0.946] |
| Z3 standalone, OOF LR selector | 0.897 | 0.933 | 0.915 [0.870, 0.951] | 0.912 |
| Union Z3 + Z2 (`it04`), plain NMS | 0.870 | 0.939 | 0.903 | 0.885 |
| Union Z3 + A0, plain NMS | 0.852 | 0.911 | 0.880 | 0.866 |
| **Union Z3 + Z2 (`it05`), pooled OOF selector** | 0.945 | 0.930 | **0.937** [0.910, 0.961] | **0.925** |
| Union Z3 + A0, pooled OOF selector | 0.945 | 0.908 | 0.926 | 0.911 |

**Pooled union.** Each external line becomes an agreement flag on the Z3 candidate it duplicates. External lines with no Z3 counterpart become extra candidates with no Z3 features. One GBM, out-of-fold, selects from the pool. Z2 eval_lines exist for 103 of the 121 scans, and they are the Z2 WP's own DEV-tuned output.

**Stability.** GBM per fold (micro): 0.918 / 0.891 / 0.953 / 0.919 / 0.946. Across code iterations it4 → it7 the selector F1 stayed within 0.922–0.932.

**Zero-bend controls.** The final selector puts 5 FP lines on the 5 zero-bend controls.

**Error budget, Z3 standalone GBM (OOF).**
- FN 55: MISSED 32, OFFSET 11, SPAN 11, TAKEN 1.
  - Candidate-level misses: only 19 gold lines (3.2 %) are never born.
- FP 35: SPAN 17, SPURIOUS 15, TAKEN 2, OFFSET 1.

**Runtime per scan.** Single thread, load + detect:
- median 1.65 s, mean 2.7 s, P95 10.1 s, max 19.7 s;
- batch peak RSS 3.3 GiB.

## Learnings

- **Exact mesh sections beat point bands.** Chaining by shared mesh edges gives ordered profiles for free, which makes the two skins, edge walls and joggles explicit. A full DEV pass takes about 5 minutes on one thread.
- **What fixed what (iteration log).**
  - Most of the gap was birth plus span: chaining collinear segments and keeping tracking gaps short took candidate recall from 0.80 to 0.96.
  - Thick plate (4–6 mm) produced sub-arc corners on large radii. Arc-chord tests and multi-chord merging cut the OFFSET FPs from 177 to 19.
  - Raw sub-mm face normals spread several degrees, which hid small flanges in the Gauss map. Smoothed normals plus a low area floor recovered tiny parts and ears.
  - Joggles need an explicit rule. Their web is shorter than any flat threshold.
- **"Same two plates" is a connectivity question, not local presence.** A component test on each flank's plate separates windows (bridge them) from separate collinear tabs (keep them apart).
- **The flank arrays must be in a canonical order.** Order them by cross(da, db) along +u. Unordered averages silently broke every flank-based test.
- **Bend-strip (cylinder) evidence did not discriminate on these scans.** Arc normals exist near every corner, so `strip_frac` is ~1.0 for TPs and FPs alike. The useful evidence was slice support, flank lengths, skin agreement and the plate-connectivity test.
- **The selector is now the variance term.** Across iterations ±0.01 F1 comes from fold thresholds. The geometric stage still improves (simple-gate F1 0.80 → 0.82) without moving the OOF number.

## Not solved (worst scans, overlays in `overlays/`, checked by eye)

- **47247004**: lattice windows remove the wall next to most of the fold, so corners are absent along about 85 % of a 419 mm line.
- **49191019**: one fold follows a bent web, and its 40° sloped segments are missed or rejected.
- **40500000**: 4 FPs on the vertical box-corner seams (two separately bent walls meeting). These are not bends.
- **11979003**: Z-step fold on the inner skin; the span is 52 % of gold.
- **48101000**: return flanges folded flat against a wall, where gold uses inner-skin pairs. These are missed.
- Out of scope: 38898000, a rolled lap-seam joggle.

## Next steps

1. **Integrator.** Use `runs/it7/*/candidates_eval_lines.json` as the Z3 birth stream: recall 0.964 at 1.83×, with an out-of-fold score attached. A pooled selector with Z2 gave 0.937 / 0.925.
2. **Discriminative strip evidence.** Replace `strip_frac` with a per-station test of a monotone normal sweep across the bisector at the measured radius. This targets box-corner seams and emboss edges (most SPURIOUS FPs and the control FPs).
3. **Long folds broken by lattice windows and folds that follow a bent web.**
   - Per-station "virtual corner" test along a born line's extension, a 2-flat fit even when one flat is only 1–2 mm.
   - Slicing along the local direction of each web segment.
4. **Selector variance.** Use a fixed threshold from all-DEV nested CV, or an LR+GBM average. Report a family bootstrap of the threshold.
5. **New scans.** `lane.py` + `model_z3_gbm.pkl` (fit on all DEV) give a frozen standalone lane for an eventual one-time TEST-B run by the integrator. It has never been run on TEST-B.
