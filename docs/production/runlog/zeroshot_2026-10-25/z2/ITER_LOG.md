# Z2 self-recipe: iteration log (DEV only)

Cohort: the 103 DEV scans with custody DEV_CONSUMED / CONSUMED_DEVCHECK (99 line scans, 75 families, 494 gold lines;
4 zero-bend controls). The 18 CONSUMED_FRESH DEV scans are not opened: their gold lives under
`ai_bend_lines_fresh_v1_SEALED/`, and the WP rules forbid anything under `*_SEALED*` (same exclusion as D1's guard).
Folds: integrator `dev_folds.json` (family_group5, 5-fold). Matcher: frozen `score.py score_case` (5 deg / 4 mm / 0.6).
A0 on the same 99 scans: micro 0.340 / family-macro 0.273.
Guard: every scan passes `research.zeroshot.common.gold_guard.require_runnable` and the D1 path guard.

| iter | change | micro | macro | P | R | FN MISSED/SPAN/OFFSET/ANGLE | FP SPURIOUS/SPAN/OFFSET/ANGLE/TAKEN | flange-pair recall (GSv2 full) | runtime med/p95 s |
|---|---|---|---|---|---|---|---|---|---|
| it00 | first version: kNN-smoothed normals, flat seeds + plane growth, skin pairing, strip evidence, E3 _flat_start/_reseat_plane/_fit_side stations, measured outer apex | 0.757 | 0.727 | 0.705 | 0.818 | 76/5/9/0 | 117/16/31/1/4 | 0.930 | 1.8/3.8 |

## it00
- No fitted parameters (all thresholds are a-priori mm / deg / t multiples), so in-sample = out-of-fold. Per fold micro: 0.78/0.79/0.80/0.71/0.66.
- 3 scans are point clouds (no faces): lane FAILED -> 12 MISSED gold lines. Largest FP class: SPURIOUS (117).

## it01 (micro 0.863 / macro 0.788, P 0.903 R 0.826)
Changes (all general rules, from reading the it00 FP budget and 2 overlays):
1. Strip evidence uses only mid-sweep normals (the middle half of the arc). Cut walls and corner seams carry normals near either flange's direction, never the bisector.
2. Arc-facet suppression: a narrow flange lying inside another pair's arc reach, with its normal inside that pair's sweep, is a facet of a large-radius arc (thick 3.9 mm stock, Ro 9 mm split into 3 mm planar strips). Its edges are dropped.
3. Outer-skin choice and in-plane directions come from each flange's material *near* the other flange (a tab bent out of a window: the base centroid lay on the wrong side and the wrong skin was chosen, so the strip test failed).
4. Unfoldability: in a 3-cycle of bends with mutually non-parallel axes the shortest edge is dropped (box-corner seams between two flanges each bent off a common base).
- FN: MISSED 69 / OFFSET 14 / SPAN 3; FP: SPURIOUS 31 / SPAN 9 / OFFSET 3 / TAKEN 1. Per fold micro 0.87/0.93/0.86/0.86/0.70. Still no fitted parameters.
- Point-cloud loader (PCA normals, opposite-skin probe for orientation and t) was added after it01 launched; it enters it02.

## it02 (micro 0.852 / macro 0.791, P 0.899 R 0.810) -- a regression on micro, kept as a lesson
Changes: (1) cut-wall rejection (a one-sided region whose extent along the other flange's normal is about t);
(2) "purify": points off their region's robust plane released, regions keeping < 50 % dropped (curved / rolled
surfaces); (3) the 3-cycle rule only drops an edge that is short (< 0.5x the others' span) or thin (strip
density < 0.5x the strongest) -- it01 wrongly removed a real base-to-end-panel bend on a box with three real
bends meeting (sc_d27a); (4) point-cloud loader active.
- FN SPAN rose 3 -> 13: purify used the plane *distance* (0.3 mm), which strips the corners of a bowed base
  plate (sc_f247: 218 mm bends cut to 123 mm). Lesson from E3 again: large bases bow; global-plane distance
  tests must not decide membership.
- Point-cloud scans: 2 of 3 now correct (the third is a rolled arch, out of the planar model).

## it03 (micro 0.868 / macro 0.817, P 0.870 R 0.866; flange-pair recall GSv2-full 0.966)
Changes: (1) seed release by normal only (6 deg), with re-seeding of released points and re-splitting of what a
released riser had joined (joggle far plates); purify by normal only (bowed bases stay whole); (2) skin pairing
needs near-parallel skins (6 deg) and >= 60 % of the close footprint one thickness behind; (3) outer-skin choice
evaluated on the other flange's mid-surface points (robust at 15 deg joggles); (4) narrow unpaired regions are
walls only below 1.6 t (was 2.2 t) -- the edge-level CUT_WALL test does the rest; (5) min span 4 mm (5 mm tabs).
- Joggles (42288005) now 8/8; flange-pair recall milestone (>= 0.95) met.
- New FP source: rolled / cylindrical surfaces now split into planar facets (sc_fee6: 12 FP, zero-bend control
  sc_b471 rolled part: 22 FP lines).

## it04 (micro 0.878 / macro 0.835, P 0.880 R 0.876; flange-pair recall GSv2-full 0.976; zero-bend FP lines 4)
Changes: (1) rolled-kink rejection: measured outer radius (apex, else the strip's chord) > 17 t is not a
press-brake bend (gold rule: rolled kinks are not bends). On it03 every TP had Ro <= 14.3 t except one line on a
rolled arch; every faceted-cylinder FP had Ro >= 19.6 t. This threshold was read off DEV -> it is the one numeric
choice covered by the grouped-CV check below. (2) collinear narrow fragments (scan dropouts along a short web)
are merged into one strip; (3) a cut wall must stand square (within 20 deg) to the other flange.
- Zero-bend FP lines 22 -> 4; fee6 12 FP -> 2.

## it05 (micro 0.910 / macro 0.865, P 0.927 R 0.895; zero-bend FP lines 3)
Change: E3's own acceptance rule for a measured bend -- the station loop (E3 `_flat_start` + `_fit_side` on
each side) must give valid stations at >= max(3, ceil(0.6 n)) of the n stations (E3 LOW_DENSITY otherwise).
On it04, edges with 0 valid stations were 5 TP vs 31 FP: an edge without flat material on both sides past the
arc is an emboss, louvre, seam or facet, not a flange-to-flange bend.
- SPURIOUS 36 -> 18, FN MISSED 34 -> 33. Lost: sc_822f (37126000, joggle on a narrow strip), 2 TP.
- Per fold micro 0.94/0.94/0.92/0.88/0.83.

## it06 (experiment, not adopted): virtual riser discovery
Idea: two near-parallel flanges offset by >= t with unlabelled material between them (a fragmented joggle web,
the round bottom of a narrow U) get a virtual riser flange from the dominant normal deviation of that material,
measured as an E3 "model plane" (stations fitted on the real side only). On 4 target scans (10839000, 38173000,
42288005) it fixed 0-1 lines per scan and added 2-6 FP: the unlabelled material near both flanges is not
spatially coherent (corner rounds, cut walls, emboss sides all qualify), so the "riser" plane is often wrong.
Kept in code behind `GraphParams.risers=False`. A proper version needs slicing along the candidate axis (Z3-style
profile) rather than a global normal histogram.

## it07: grouped-CV check of the numeric thresholds read off DEV (same code as it05, riser flag off)
One segmentation per scan, 12 graph configs: max_ro_t in {12, 17, 25, off} x min_span in {4, 6, 8} mm.
All-DEV micro per config: 0.885-0.888 with the rolled filter off, 0.903-0.915 with it on (the rule matters, the
exact value does not). Nested family-grouped 5-fold (choose on 4 folds, score the 5th, pool):
**OOF micro 0.907 / family-macro 0.854** (P 0.926, R 0.889); chosen ro12_sp8 in 4 folds, ro25_sp8 in 1
(`z2/oof_it07.json`).
D1 gated outer-apex snap as a final step (`snap_d1.py`): 445 lines moved, median 0.044 mm; +1 TP / -1 FP
(0.912 / 0.870). Our measured outer apex was already on the D1 locus.

## it08 = final config (max_ro_t 12, min_span 8 mm, D1 snap on)
micro **0.917** [0.872, 0.953] / family-macro **0.867** [0.798, 0.927] (95 % family bootstrap, 10 000, seed 20261018),
P 0.944, R 0.891, 99 scans / 75 families / 494 gold lines; full-stage scans (85) 0.920. Zero-bend controls: 0 FP lines.
Flange-pair recall (proxy) GOLD_SET_V2 full scans 0.976 (all 0.966).
Error budget: FN MISSED 37 / OFFSET 11 / SPAN 5 / TAKEN 1; FP SPURIOUS 12 / SPAN 12 / OFFSET 2.
A0 on the same 99 scans: 0.340 / 0.273; FN MISSED 314 / SPAN 38 / OFFSET 22 / ANGLE 5; FP SPURIOUS 30 / SPAN 28 / OFFSET 4 / ANGLE 5.
Runtime per scan (one process, OMP 1): median 3.6 s, P95 6.8 s, max 14.3 s; peak RSS 2.6 GiB.
Honesty note: the structural rules (it01-it05) were designed by reading DEV overlays and error budgets across
all folds; only the numeric thresholds went through grouped CV. The OOF number therefore bounds threshold
selection, not rule design. TEST-B or new scans are the real check.

## Remaining errors (it08), by kind
- Rolled / cylindrical parts (28870000, 49125000, 49188000-like, two point-cloud arches): gold counts the
  flat-to-roll tangent lines; the planar model rejects rolled facets by design.
- Fragmented short webs / 48 deg double folds (10839000) and the round-bottom U (38173000): need a profile
  (slice) birth channel -- the global-histogram riser attempt (it06) failed.
- Tiny parts (< 8 mm bends: 46414000, 37126000 Z-fold on a 6 mm strip).
- A 560 mm plate with a fragmented edge profile and louvres (38169000): 0/5.
