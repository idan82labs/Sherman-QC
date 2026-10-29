# Why the zero-shot lane missed 5 lines on 47085000 (N1): a stage-by-stage diagnosis

> **Label.** This is a post-hoc diagnosis of a consumed, one-time run (N1). It reports **no new N1 score**. Where a prototype is applied to N1, the result is marked "post-hoc, consumed" and is given per line only. All prototype numbers come from DEV out-of-fold runs (`dev_folds.json`, 125 scans, 619 gold lines, 5 zero-bend controls). TEST-B and the other sealed sets were not read. No frozen lane code, production engine code, gold or label was edited.

**Scan:** `sc_3d191f5f8ff8` (gid `gb_4dd5ba`, t = 0.98 mm).

**Primary result (v1 zs):** TP 21 / FP 2 / FN 5. The misses are B1, B4, B14, B25 and B26; the false lines are the short B1 and B4 lines.

## Method

**Re-running the lane with every stage dumped.**
- `research/zeroshot/n1_diag/diag_run.py` re-runs the frozen v1 stages on the scan and keeps every intermediate product. It calls the frozen modules read-only; nothing is reimplemented.
- The Z3 `detect()` is replayed call by call, and the replay gives the same 42 candidates as `slicer.detect`.
- The pooled selection reproduces the published v1 zs output exactly: 23 lines.

**Slice-level debugging.** `chain_debug.py` re-slices one family at the levels that cross a gold line. For each level it reports:
- the section chains;
- every straight run and the reason it was rejected;
- the corners found.

**Other tools:**
- GBM attribution: `attrib.py`.
- DEV failure classes: `dev_classes.py`.
- DEV prototypes: `proto.py`, `z2feat_*.py` and `ext_guard.py`.
- Renders: `render.py`.

All heavy runs went through `sherman_slot.sh`, with at most 3 at once and peak RSS of 1.7 GiB or less.

**Caps.** v1 and v2 have no line-count cap. The Z3 axis-family cap is 20; this scan has 5 families, and every miss lies on family 0 or 1, the two top-scored. Neither cap played a role.

## Summary

| Gold | Born by Z3? | Born by Z2? | Lost at | Class | DEV prevalence (v1 OOF, 33 FN) |
|---|---|---|---|---|---|
| B1 end flange, 80 mm | Partly: 36 mm, s = 48–80 | Yes, full 79.4 mm | **Pooling** keeps the Z3 extent and uses Z2 only as a vote. The root cause is at Z3 slicing. | POOL_EXTENT | **8 FN** in 4 families (9 lines flagged) |
| B4 lanced lip, 70 mm | Two 30 mm halves | Yes, full 69.1 mm | Z3 split, the Z3 merge refused, pooling dropped the Z2 extent, and the GBM rejected the 2nd half (0.39) | Z3_SPLIT + POOL_EXTENT | 3 FN Z3_SPLIT (overlapping POOL_EXTENT). v2's join already fixes most. |
| B14 hook lip, 25 mm | **Yes, full length; would match** | No: the lip is not segmented | **GBM selector**: p_zs 0.215 vs p_z3 0.904 | SEL_Z3ONLY_REJECT | **15 FN**, the largest class |
| B25/B26 joggle, 38 mm | **Yes, both corners; would match** | No: the web is not a flange | **GBM selector**: p_zs 0.016 / 0.009, p_z3 0.11 / 0.04 | Shallow joggle on a narrow strip | 0 FN of this pattern. 65 of 67 DEV joggle lines are found. |

## B1: end flange (`B1_closeup.png`)

**Z3 slicing.**
- One corner track runs for 17 slices, from s = 47.8 to 79.8 mm along the gold line, all inner-skin corners at 90.5°.
- From s = 6 to 46 mm, no slice has a corner within 4 mm of the fold. The slice debug shows why. At each of these levels the fold is split over two open chains:
  - the inner-skin chain carries the flange flat and the arc, but stops about 2 mm past the arc;
  - the outer-skin chain carries only the plate.
- No chain holds two flats, so no corner is formed. Slice k30 is shown in the PNG, next to the connected slice k10.
- A single isolated corner at s = 5.8 is dropped by `min_track_slices = 2`.
- The final candidate (pool index 18) is 36.25 mm long, with strip_frac 1.0.

**Z2.**
- Both plates are found: F0 is the plate and F26 the flange.
- The edge between them is accepted: span 79.4 mm, cover 1.0, ro 2.1 mm.

**Pooling.**
- The Z2 line duplicates candidate 18 under the pool rule (0.05°, 0.13 mm, 0.88 of the shorter span).
- It is therefore stored as the flags `in_z2` / `n_src` only. **The 79.4 mm geometry is discarded.**

**Later stages.**
- GBM: 0.959, selected.
- D1 snap: moved 0.12 mm; the extent is unchanged.
- Scorer: 0.03° and 0.07 mm, but gold overlap **0.405** and prediction overlap 0.892 against the 0.6 threshold. This gives FN + FP (SPAN).

**In plain language.** The engine saw the end flange, but only its last 36 mm. Along the first 40 mm the scanner captured each side of the sheet only partly. In every Z3 cross-section the fold is split between two loose pieces of surface, and no single piece shows both plates meeting. Z2 works from flat patches rather than cross-sections, and it found the whole 79 mm fold. But when the two detectors' results are merged, Z2's line only counts as a second-opinion vote on Z3's short line, and its length is thrown away. The output was in the right place but too short (40 % of the gold length), so it counted as one miss and one false line.

## B4: lanced lip (`B4_closeup.png`)

**Z3 slicing.**
- There are two tracks: 14 slices over s = 0.5–26.5 mm and 13 slices over s = 42.5–68.5 mm.
- In the 7 slices between them (s = 28.5–40.5) the saddle lip is less than 2 mm tall. Its straight runs are 1.6–2.0 mm long, which is under `min_flat_mm` 2.0, or they are rejected as arc bows. So there is no corner (slice k24, versus k36).

**Z3 merge.**
- The gap is 16 mm, more than 2.5 × the slice step, so `same_plates` must pass.
- With a 7 mm flank the test uses its "short flank" band of 2.8–7.7 mm, and it finds no lip material across the low middle. The pieces stay separate as candidates 21 (30.0 mm) and 25 (30.75 mm).

**Z2.** F0 and F20 are found and the edge is accepted, with a span of 69.1 mm.

**Pooling.** The Z2 line attaches to candidate 21 only. Candidate 25 has no Z2 flag.

**GBM.**
- Candidate 21 scores 0.993.
- Candidate 25 scores **0.392**, below 0.5; the Z3-alone model gives it 0.944.
- The effects that pull candidate 25 down are in_z2 +0.50, n_src +0.33, frac_both +0.30 and len_rel +0.27. These are the gains in probability if the feature took a typical DEV-positive value.

**Snap and scorer.**
- D1 snap: 0.18 mm.
- Scorer: gold overlap **0.38**, prediction overlap 0.88. This gives FN + FP.

**Why v2 fixes B4 but not B1.** v2's join merges two collinear pieces that are both in the pool when the same two plates continue across the gap.
- **B4:** the evidence for the pair (25, 21) shows a 15.2 mm gap with plate A and plate B both connected. v2 uses join_low 0, so the rejected 0.39 piece can take part, and the two pieces are joined into a 76.0 mm line. That line matches (gold overlap 1.00, prediction overlap 0.92).
- **B1:** there is only one Z3 piece. The rest of the fold has no track at all (one 1-slice corner, which is dropped), so there is nothing to join. Like v1, v2 still uses Z2 only as a vote and discards its extent. v2's B1 line is the same 36.3 mm line.

**In plain language.** The lip dips in the middle, and there it is shorter than the 2 mm Z3 needs before it calls something a flat side. So Z3 found the fold as two 30 mm halves with a 16 mm hole between them. Its own gap check looked for lip material in the hole, found none (the lip really is that low there), and did not join the halves. Z2 found the whole fold, but as with B1 its length was thrown away when the results were merged. The selector kept only the half that had Z2's vote. v2 adds a plate-continuity join that sees the same two plates on both sides of the hole, and that is why v2 got B4.

## B14: hook lip on one leg (`B14_closeup.png`)

**Z3.**
- A 12-slice track over s = 1.1–23.1 mm, included angle 63.6°, all inner-skin corners.
- Candidate 26 is 29.75 mm long. **Its snapped line matches the gold line:** gold overlap 1.00, prediction overlap 0.85, offset 0.02 mm.

**Z2.**
- Only one Z2 flange lies near the fold: F25, the foot, which is single-skin and 444 mm².
- 82 % of the scan points within 6 mm of the fold are unlabelled, so the 5 mm lip was never segmented and no edge was tested.
- Z2 missed the twin line B12 in the same way.

**GBM.**
- The pooled model scores candidate 26 at **0.215**; the Z3-alone model at 0.904.
- The top effects are in_z2 +0.62, frac_both +0.41, len_rel +0.34 and n_src +0.34.
- If Z2 had agreed (in_z2 = 1), the score would be 0.982.
- On DEV, Z3 candidates without Z2 agreement are real only **2.1 %** of the time (65 of 3029), against 93.5 % with agreement.

**Scorer.** The nearest engine line is the leg-to-foot fold, 13.8 mm away, so B14 is an FN.

**In plain language.** Z3 found this fold correctly, at the right place and full length, exactly as it found the five twins on the other legs. The difference is Z2. On this leg the foot was scanned on one side only, and the small lip was never recognised as a flat patch, so Z2 had nothing to say about this fold. The final selector learned on DEV that a Z3 line without a Z2 vote is almost always wrong, so it scored this line 0.22 and dropped it. Z3's own selector would have kept it (0.90).

## B25/B26: 3 mm edge joggle (`B25_B26_closeup.png`)

**Z3.**
- The joggle branch fired. There are two tracks of 15 and 14 slices covering the full 36 mm.
- Candidate 16 is 41.2 mm and candidate 14 is 40.2 mm, both with an included angle of about 134.5°.
- **If selected, both would match one-to-one:** candidate 16 to B25 at 1.15 mm, and candidate 14 to B26 at 1.07 mm.
- The D1 snap refuses both (shift 2.6 and 3.5 mm, over the 2.1 mm gate), but they match without it.

**Z2.**
- The plate (F0) and the strip (F27, F30) are found, but they are all parallel: sweep 179.4–179.9°, so each edge is rejected as SWEEP_OUT_OF_RANGE.
- The roughly 1.5 mm web is not a flange, and virtual risers are switched off. No fold is found.

**GBM.**
- Pooled model: 0.016 and 0.009. Z3-alone model: 0.112 and 0.040.
- For the Z3-alone model the top effects are frac_both +0.45, support_frac +0.25 and gap_med +0.23.
- support_frac is 0.74–0.79, which is the 4th–6th percentile of DEV positives.
- gap_med = 0 is the joggle-branch signature. On DEV, 910 candidates have it and only 3.3 % are real.

**Scorer.** MISSED: there is no engine line near either fold.

**In plain language.** Z3 did see both bends of this tiny step, in the right place and at the right length. But the step is only about 1.5 mm tall on a 4 mm strip. The cross-sections were noisy: only 74–79 % of the slices agreed. The step is also built by Z3's special joggle rule, and on DEV the lines from that rule are almost always false (97 %). So both selectors scored the pair near zero. Z2 sees the plate and the strip as two parallel flat patches with no wall between them, so it has no fold to report.

## Is each failure class general? (DEV only)

v1 DEV out-of-fold baseline: F1 0.9444 (TP 586 / FP 36 / FN 33). The 33 FN break down as follows (`evidence/dev_classes.json`):

| Class | DEV FN | Scans / families |
|---|---|---|
| SEL_Z3ONLY_REJECT (B14 class) | **15** | 12 families. 7 have p_z3 ≥ 0.7, and 2 have p_z3 ≥ 0.9. |
| POOL_EXTENT (B1 class) | **8** | 47247004 ×4, 40500000 ×2, 50757000, 45141000 |
| Z3_SPLIT (B4 class, without POOL_EXTENT) | 3 | 11979003, 48926001, sc_9b930fe58a1b |
| other selector rejects | 4 | |
| Z2 match / other | 2 | |
| not in pool | 1 | |

**Joggles.**
- DEV gold notes name 67 joggle lines, and v1 finds 65 of them.
- The 3 mm-offset joggle pairs on DEV (10839000, 38169000, 48333004) are all found.
- The B25/B26 pattern (both corners from the joggle branch, a web under 2 mm, low slice support) has **no DEV miss**.

## Proposed fixes, ranked

The ranking weighs DEV gain against the risk of false lines on rolled parts. "Paired CI" is a paired family bootstrap (4000 resamples) against the frozen OOF baseline of the same lane.

| # | Fix | Fixes | DEV v1 (Δ, paired CI) | DEV v2 (Δ, paired CI) | Rolled / zero-bend risk |
|---|---|---|---|---|---|
| 1 | **EXT: adopt Z2's extent at pooling.** A selected line whose candidate absorbed a Z2 line ≥ 1.3× longer, containing ≥ 75 % of it, takes the union of the two extents along its own axis. Lines it now duplicates are dropped. | B1, B4 (v1) | **0.9627, +0.0183 [+0.0022, +0.0403]**. FP 36→21, FN 33→25. Families 5 better, 1 worse. | 0.9643, +0.0065 [−0.0034, +0.0232]. Families 2 better, 1 worse. | **None by construction.** No line is added; 0 new lines on the controls and rolled-CAD scans. |
| 2 | **Z3R: rescue strong Z3-only lines.** Keep a rejected Z3-only candidate when the Z3-alone GBM gives ≥ 0.9 **and** v2's station sweep is ≥ 0.25. | B14 (borderline, p_z3 0.904) | +0.0017 [0.000, +0.0057]. Families 1 better, 0 worse. | +0.0017 [0.000, +0.0056]. Families 1 better, 0 worse. | **The sweep gate is required.** Without it, v2 adds a line on the rolled control 47479000; with it, 0 new control lines. On the N1 rolled shells the Z3-alone arm emitted 0 lines at p ≥ 0.5, so p ≥ 0.9 adds none (post-hoc, consumed). |
| 1+2 | EXT + Z3R | B1, B4, B14 | **0.9644, +0.0200 [+0.0024, +0.0439]**. TP 596 / FP 21 / FN 23. Control FP 5 → 5. | 0.9660, +0.0082 [−0.0019, +0.0251]. Control FP 1 → 1. | as above |
| 3 | Z3 birth: **two-skin corner synthesis.** Pair a flat that ends at a fold in one section chain with the matching flat in the other skin's chain (B1's root cause). | B1 when Z2 also misses | not prototyped (needs a Z3 DEV re-dump) | | unknown; must be checked on rolled shells |
| 4 | Joggle-pair rescue | B25/B26 | **−0.0090**: 14 DEV pairs, the 8 true ones already selected, +12 FP | −0.0092 | not recommended |

**Rejected prototypes.**
- **Z2-visibility features** in the pooled GBM: the fraction of points labelled as a Z2 flange near the line, the number of flanges there, and the Z2 edge status. DEV: −0.0052 [−0.0141, +0.0023], 9 families worse. Post-hoc, B14 would rise only to 0.435.
- **"Twin" rescue** (a rejected Z3-only line with ≥ 2 selected lines of the same signature): no change at all on DEV, so it cannot be tested.

**Notes on EXT.**
- Its single DEV regression is 48340008, a fold that exists only across an 8 mm neck, where Z2's line runs 15 mm along a slot edge.
- A flank-presence guard (`slicer.flank_continuity` on the extension) did not separate that case, because every extension scored 0.85–1.0 (`evidence/ext_flank_guard.json`).
- The containment threshold was first set to 0.9 a priori. DEV is flat from 0.5 to 0.75 and one line worse at 0.9. I revisited it after the post-hoc N1 check showed that B1's line is only 88 % contained, so treat the 0.75 choice as N1-aware.

**Post-hoc, consumed (N1 per line, not a score).**
- EXT recovers B1 (83.7 mm line, overlaps 1.00 / 0.95) and B4 in v1 (72.4 mm, 0.99 / 0.95).
- Z3R recovers B14.
- No new unmatched line appears.
- B25/B26 stay missed under every prototype.

## Files

**Renders** (about 1200 × 620 px, 107–129 KB each). Each has a local 3-D close-up, span bars along the gold line, and Z3 cross-sections at the slice where the track stops and at a good slice:
- `B1_closeup.png`
- `B4_closeup.png`
- `B14_closeup.png`
- `B25_B26_closeup.png`

Colours in the renders:

| Colour | Meaning |
|---|---|
| Green / orange | Gold lines |
| Red | Engine lines |
| Magenta | The absorbed Z2 line |
| Cyan | Rejected candidates |
| Blue | Z3 per-slice corners |

**Data:**
- `diag.json`: per failure, the stage where it was lost, the evidence numbers, the class, the DEV prevalence, the fix, the PNG and a plain-language paragraph.
- `evidence/`: the DEV class table, the prototype results (`proto*.json`, `z2feat_oof.json`), the GBM attributions, the v2 pair and sweep evidence on N1, the per-line post-hoc check and the flank-guard values.

**Code:** `research/zeroshot/n1_diag/`, a new research directory. The frozen lane code is untouched. Scratch dumps, including the mesh, the Z2 point labels and the pickled intermediates, are in the session scratchpad and are not kept in the repo.
