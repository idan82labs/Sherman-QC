# Zero-shot bend-line detection on 3D scans of press-brake sheet metal

**Working paper: full internal draft, 2026-10-26.** Not for external release. Every number below is traceable to a repository file listed in the evidence index (§E). Ledger labels follow `AGENTS.md`:
- **DEV-OOF** means consumed development data, scored out of fold.
- **TEST-B** means the programme-sealed, consumed-by-rule test, run once.
- **POST-HOC** means analysis done after a one-time run. It informs future work only.

Nothing in this draft is an unseen-family accuracy claim in the sense of the 2026-10-19 custody rule (§10).

This draft supersedes the results placeholders of `2026-08-14_sherman_qc_zero_shot_bend_inspection_paper_draft.md` and keeps that draft's governance framing. The old draft's V51 confirmatory tranche was never executed; this paper reports a different architecture.

---

## Title options

1. **Slice, Track, Select: Zero-Shot Bend-Line Detection on Industrial Sheet-Metal Scans**
2. Exploiting Sheet-Metal Structure for Zero-Shot Bend-Line Recovery from 3D Scans
3. From 0.31 to 0.86: Axis-Family Slicing and Flange-Graph Metrology for CAD-Free Bend Inspection
4. Bend Lines Without CAD: A Pre-Registered Evaluation of Structure-Aware Detection on Real Press-Brake Parts
5. When Birth, Not Selection, Is the Bottleneck: Zero-Shot Bend Detection on Sheet-Metal Scans

Recommendation: option 1 for a vision or graphics venue, and option 4 for a manufacturing or metrology venue. Option 3 puts the delta in the title. That is honest only if the comparison is stated against the old labels: 0.893 vs 0.310 (§7.3).

---

## Abstract

**Problem.** Quality control of press-brake parts needs the location of every formed bend: its line, its included angle and its extent. Often there is no CAD model, or the CAD cannot be compiled or registered. We study **zero-shot bend-line detection**: given only a 3D scan of a sheet-metal part, emit one 3D line segment per formed bend, on the outer-skin apex. No CAD, part identity or expected count is available at inference.

**Method.** We exploit the structure of press-brake sheet metal: piecewise-planar, straight bend axes, few axis directions, constant thickness. The method has two candidate generators, a selector and a placement step:
- **Axis-family slicing (Z3).** Gauss-map axis families, exact mesh cross-sections every 2 mm, 2-D profile corner detection with explicit skin, edge-wall and joggle rules, and Hungarian tracking of corners across slices.
- **Flange-graph self-recipe (Z2).** Planar flange segmentation, then an adjacency graph whose edges are measured with the local skin-aware metrology of our CAD-assisted inspection engine.
- **Selector.** A gradient-boosted selector over 26 plain geometric features, fitted with family-grouped cross-validation.
- **Placement.** A thickness-aware outer-apex snap (D1).

**Data and protocol.** The data are 186 usable real scans of 147 part families from one manufacturer, with 915 gold lines.
- **Split.** Family-group-disjoint: DEV 125 scans / 94 families, and a sealed TEST-B of 61 scans / 50 families. No engine had ever run on any TEST-B scan.
- **TEST-B gold.** Re-made before any new engine output, by two blind AI markers plus an adjudicator.
- **Pre-registration.** Lane code (60 files), models, threshold, scorer and decision rule were pinned in a hash-chained ledger before the single run.

**Results.**
- **DEV, out of fold.** Micro line-F1 **0.944** [0.919, 0.966] and family-macro 0.944. The previous zero-shot lane scored 0.322 / 0.267.
- **TEST-B, one run.** Micro line-F1 **0.855** [0.764, 0.926] and family-macro **0.855** [0.782, 0.918] (P 0.872, R 0.839). That meets the pre-registered 0.85 point-estimate goal, narrowly: the CI reaches below 0.85.
- **Same TEST-B scans against the older labels the baseline was scored on.** The new lane scores 0.893; the old lane scored 0.310, at recall 0.198.
- **Runtime.** Median 8.0 s per scan on one CPU thread, against 229 s for the old lane.
- **Where recall comes from.** Most of the gain is candidate *birth*. The old lane never proposed 81 % of its misses and emitted nothing on 36 of 116 DEV scans. The new slicing channel proposes 96.4 % of gold lines at 1.83 candidates per gold line.
- **CAD-assisted overlay (DEV).** Filling bends the CAD engine withholds with zero-shot lines raises line-F1 from 0.955 to 0.987 and creates no PASS decision.
- **Angle fidelity (DEV, CAD cohort).** On 256 non-joggle bends the zero-shot included angle agrees with the CAD engine's measured angle to a median of 0.06° (P90 0.28°).
- **DEV saturation (post-hoc).** A later lane version gained +0.0135 on DEV, with a paired CI excluding 0. A post-hoc re-score on the consumed TEST-B gave exactly the same 0.855 (Δ 0.000).

**Limitations.** The gold is AI-made: there is no human annotation of the test set, and the markers share one preparation tool. The data come from one company and one scanner pipeline. The rules were designed on DEV. TEST-B is consumed by rule, and the data cannot be released.

---

## 1. Introduction

Press-brake forming turns a flat blank into a part by folding it about straight lines. Inspection must confirm that each intended bend exists, sits in the right place and has the right angle. With a compiled CAD model and a trustworthy registration, this reduces to per-flange metrology. Our CAD-assisted engine (E3) reaches 0.990 micro line-F1 on its development gold under those conditions [E-FIX3]. In practice, CAD is often missing, fails to compile (rolled parts, weldments), or cannot be registered (mirrors, symmetric datums, partial scans). In the one-time 2026-10-19 held-out evaluation, the engine lost most of its headline score to coverage: 0.527 headline against 0.857 on scans where it ran [E-FINAL]. A detector that needs no CAD is useful both as a standalone tool and to fill those gaps.

Generic machinery for primitives, creases and wireframes is mature [2, 12–20]. Bend lines on *formed sheet metal*, however, are almost absent from the literature (§2.4). Generic methods also do not use the strongest priors this domain offers:
- the surface is piecewise planar, with straight cylindrical bend strips;
- a part has only a few bend-axis directions;
- the sheet has a constant thickness *t*, so every flange has two skins *t* apart;
- the part is developable.

Our previous zero-shot lane, A0, used a patch graph with a CP-SAT typed union. It reached 0.322 micro F1 on DEV and 0.310 on TEST-B, both at recall ≈ 0.2 [E-BASE]. A long history of rule-level refinements on small cohorts did not transfer (§9).

This paper reports a redesign built around three observations from that history:
1. **Recall was the gap, and most misses were never born.** 81 % of A0's DEV false negatives had no prediction within 15° / 15 mm [E-BASE].
2. **The CAD engine's precision comes from local skin-aware plane fits once the flange graph is known.** Knowing which flanges exist is the hard part, not measuring them [E-LESS].
3. **Rules tuned on small consumed cohorts collapse on transfer.** Legacy 0.634 fell to 0.197 on a fresh daily cohort [E-V43]. Parameters must therefore be fitted under family-grouped CV, and the test must be sealed and pre-registered.

**Contributions.**
1. **Axis-family slicing with 2-D profile corner tracking (Z3).** A scan-only birth channel for press-brake bends. It turns 3-D segmentation into ordered 2-D profiles, where skins, cut-edge walls, large-radius arcs and joggles are explicit. It proposes 96.4 % of DEV gold lines at 1.83× the gold count.
2. **Flange-graph self-recipe (Z2).** A scan-derived flange adjacency graph measured with the CAD engine's skin-aware local metrology: flat-start windows, station fits and its LOW_DENSITY acceptance rule. It reuses engineering that had already solved skin swaps, larger-than-CAD radii and bowed bases. Flange-pair recall is 0.966.
3. **A pooled, family-grouped selector.** It integrates the two channels. It is evaluated with nested grouped CV, a fixed a-priori threshold and paired family bootstraps.
4. **A negative result on thickness modelling as an F1 lever.** The previous lane does put lines on the inner side of the sheet, a median 0.63 *t*. On this corpus's stock (0.9–3.2 mm) that never crosses the 4 mm gate, so outer-apex snapping is F1-neutral while cutting position error 24× (§7.4).
5. **A custody-first evaluation protocol for a small industrial corpus:**
   - a hash-chained gold ledger;
   - two blind markers plus adjudication with canaries;
   - a frozen matcher;
   - a family-disjoint sealed test, pre-registered with code and model hashes and run exactly once;
   - every invalidation and null result kept.
6. **A safe CAD-assisted integration.** Zero-shot lines fill bends the CAD engine withholds and flag unexpected bends. They never create a PASS. Registration seeding was rejected because it opened a false-PASS path (§7.7).

---

## 2. Related work

This section draws on `RESEARCH.md` [E-RES] and the 2026-08-14 draft's §10 [E-OLD]. Reference status is marked in §R.

### 2.1 Primitive fitting and multi-model segmentation

- **Sampling methods.** RANSAC [1] and Efficient RANSAC [2] fit planes, cylinders and other primitives under gross outliers.
- **Region growing** [21, 22] and **robust-statistics planar patches** [23] are deterministic alternatives.
- **Global energy methods** replace greedy sequential extraction: PEARL [24], label costs [25], Multi-X / Progressive-X [26] and Graph-Cut RANSAC [27]. They target the failure that sequential RANSAC lets the first large plane steal inliers from coplanar or parallel planes. On sheet metal, the "parallel plane" is the other skin.
- **Cylinder axes** come from the Gaussian image [28 (unverified), 29].
- **GlobFit** [30] imposes relations between primitives as constraints.

Our Z2 flange segmentation is in this lineage. It is not a global energy method; §9 explains why we did not need one.

### 2.2 Sharp features, edges and wireframes

- **Classical crease detection:** multi-scale covariance, Difference of Normals [31], Voronoi covariance [32], Gauss-map clustering [33], fast edge extraction [34] and learned contour classification [35].
- **Learned edge and wireframe methods:** EC-Net [5], PIE-NET [6], PC2WF [7], NerVE [8] and SepicNet [9].
- **Learned primitive segmentation:** SPFN [36], ParSeNet [37], HPNet [38] and Point2CAD [39].

Point2CAD recovers edges by intersecting neighbouring fitted surfaces, which is the idea Z2 shares. These networks are trained on clean, normalised CAD samples from ABC. At normalised scale a 1 mm sheet falls below their resolution, so both skins collapse into one surface. Licence checks (2026-10-25) found no pretrained weights clearly licensed for commercial use [E-RES §D9]. We therefore treat them as candidate offline comparators (§12), not baselines we ran.

### 2.3 Thin-walled parts and mid-surfaces

Mid-surface abstraction from solid models [40] and parameter-free mid-surface extraction [41] address thin walls in CAD. Laser-scan reconstruction of thin walls has known failure modes, such as TSDF collapse [42]. We use a much simpler ray-cast skin pairing (D1) to estimate *t* and place lines on the outer skin.

### 2.4 Bend measurement on formed sheet metal: the gap

The literature search found very little on recovering bend lines from scans of *formed* sheet metal [E-RES §0]:
- a Chinese patent that slices normal to a bend and fits two lines to measure its angle (CN106295058A [43]);
- an Aalborg MSc thesis on vision-based bend-quality inspection [44];
- in-process measurement for roll forming.

Sheet-metal CAD work exists: B-rep face-edge graphs with GNNs [45] and synthetic bending datasets [46 (unverified)]. It addresses CAD, not scans. **To our knowledge no published method detects all bend lines of a formed part from a scan, without CAD, and evaluates line-instance F1 on real parts.** The patent's slicing idea is the closest precedent for Z3. It measures the angle of a known bend; it does not discover bends, track their spans or handle skins and joggles.

### 2.5 Graph matching with CAD priors

Spectral matching [47], reweighted random walks [48] and factorized graph matching [49] can match a scan flange graph to a CAD flange graph without registration [E-RES §D5]. We tested only the cheapest CAD priors (count cap, family cap, angle windows) and leave graph matching to future work (§7.7).

### 2.6 Evaluation under group structure and leakage

Blocked cross-validation for hierarchical data [10] and the leakage taxonomy of Kapoor and Narayanan [11] motivate our family-grouped folds, sealed test and model-information reporting.

---

## 3. Problem and metric

**Input.** A triangle mesh, or in 3 DEV scans a point cloud, of one physical part in the scanner's raw frame. The scans are exported meshes from a ZEISS / Geomagic pipeline [E-SCAN].

**Output.** A set of 3-D line segments *P = {p_j}*, each with an included angle. They must lie in the raw scan frame, on the outer-skin arc apex.

**Gold.** Segments *G = {g_i}* drawn under the GOLD_SET_V2 rule [E-GOLD]:
- the line is on the outer-skin apex and parallel to the fold;
- it covers ≥ 80 % of the visible fold;
- formed folds ≥ 10° count;
- a joggle counts as 2;
- a fold split by windows or reliefs between the same two plates is 1 line;
- hole walls, edges, embosses, rolled kinks and weld-plate edges are not bends;
- partial scans count formed bends only.

D1 confirmed the apex convention on the scans themselves. Of 322 gold lines with a valid profile, 301 lie nearest the outer arc apex, at a median distance of 0.024 mm [E-D1 §2].

**Frozen matcher** (`research/broad_baseline/score.py score_case`, frozen since before this programme [E-BASE]). A prediction matches a gold line iff:
- the axis angle is ≤ 5°;
- the transverse offset is ≤ 4 mm;
- the overlap is ≥ 0.6 of **both** spans.

Matching is one-to-one by minimum-cost Hungarian assignment, in the raw scan frame.

**Aggregates.**
- *Micro F1* is computed from TP, FP and FN summed over complete-label scans that contain bends.
- *Family-macro F1* is the mean over part families of each family's micro F1.
- CIs are 95 % percentile bootstraps over families, with 10 000 resamples (seed 20261018 for the baseline, 20261025 for the programme).
- Zero-bend control scans are reported separately as FP-line counts.
- A failed or missing output counts as all false negatives. No scan is dropped [E-PRE].

**Error budget** (`score_broad.diagnose`, diagnostic only). For each error, find the nearest counterpart within 15° / 15 mm, then classify:
- FN: MISSED, SPAN < 60 %, OFFSET > 4 mm, ANGLE > 5°, or TAKEN;
- FP: SPURIOUS, or span, offset or angle failures.

---

## 4. Method

The final frozen lane, `zs_lane_v1`, is in `research/zeroshot/zs_lane_v1/`, with config `config_v1.json` and a 60-file code fingerprint `af87df2f…` [E-INT, E-PRE]. It runs:
1. Z3 detection;
2. Z2 detection;
3. pooling of Z2 lines into the Z3 candidate table;
4. one GBM with a fixed 0.5 threshold;
5. the D1 gated outer-apex snap.

No stage uses CAD, filenames or part-specific logic. Every threshold is in mm, degrees or multiples of the measured *t*.

### 4.1 Axis-family slicing with 2-D profile corner tracking (Z3)

Code: `research/zeroshot/z3_slicing/`, 2 218 lines, fingerprint `6cff04a9fb34f710` [E-Z3].

1. **Axis families.**
   - Smooth the vertex normals over k nearest neighbours. Raw sub-millimetre face normals spread several degrees and hid small flanges.
   - Find *compact* peaks on the projective Gauss sphere. A peak's core must be ≥ 1.2× its ring, so bend-arc bands produce no peaks. The peak area floor is 0.05 % of the scan area.
   - Candidate axes are the clustered cross products of peak pairs. They are ranked by pair area plus great-circle (bend-arc) support. At most 20 families are kept.
2. **Exact sections.**
   - Cut the mesh with planes perpendicular to each axis, every 2 mm (1.5 mm gave no gain).
   - Chain the sections through shared mesh edges into ordered 2-D polylines that carry outward face normals.
   - Point clouds get a ball-pivoting surface first.
3. **Corners.**
   - Douglas–Peucker simplification plus a collinear merge give the straight runs.
   - The following are *not* flanges:
     - a chord of an arc: significant quadratic bow with *R* < min(8 L, 40 mm);
     - a sheet-edge wall: a flat about *t* long between antiparallel flats, or at a chain end.
   - Several chords of one arc merge into a single bend. On 4–6 mm plate this rule cut OFFSET FPs from 177 to 19.
   - Two parallel flats offset sideways through a short path become a **joggle**, i.e. two opposite bends.
   - The included angle comes from the two flat directions. The apex is where the polyline crosses the corner bisector.
   - The skin comes from the outward normals. An inner-skin corner is moved *t* along −bisector to its outer apex, and per-slice duplicates on the two skins are merged.
4. **Tracking.**
   - Link corners across slices with the Hungarian algorithm, bridging gaps of up to 4.5 mm.
   - Fit a total-least-squares 3-D line through the apices. The span comes from slice support, refined to sub-slice precision with 0.25 mm stations.
   - Collinear segments are chained across a gap only when **each flank's plate is one 8-connected component around the gap**. Windows and reliefs between the same two plates are bridged, while separate tabs on one base stay separate.
   - Flank arrays must be in a canonical order (cross(d_a, d_b) along +u). Unordered averages silently broke every flank test.
5. **Birth filter** (`gate.birth`, hand-set before any full-DEV score): length ≥ 5 mm, ≥ 3 slices, bend ≥ 8°, support ≥ 0.4. This stream is the Z3 contribution to the pool.

A bend-strip (cylinder-normal) evidence score was implemented but does not discriminate on these scans. Arc normals exist near every corner, so `strip_frac` ≈ 1.0 for TPs and FPs alike [E-Z3].

### 4.2 Flange-graph self-recipe reusing local metrology (Z2)

Code: `research/zeroshot/z2_self_recipe/`, 1 870 lines. It imports `domains.production.engine.inspect.{config, geom, metrology}` from E3 [E-Z2, E-PRE].

1. **Segmentation.**
   - Smooth the normals over k nearest neighbours, then seed flat regions and grow planes.
   - Release points by normal only (6°), not by plane distance. Large bases bow 0.7–1.8°, and a distance test cut a 218 mm bend to 123 mm.
   - Re-split what a released riser had joined, which handles joggle far plates.
2. **Skin pairing.**
   - Pair regions that are near-parallel (6°) with ≥ 60 % of the close footprint one thickness behind.
   - Choose the outer skin from each flange's material *near* the other flange, evaluated on the other flange's mid-surface points.
3. **Graph edges** are flange pairs with bend-strip evidence. Only mid-sweep normals count: cut walls and seams carry normals near either flange's direction, never the bisector. Rules applied to the edges:
   - arc-facet suppression;
   - cut-wall rejection, which requires standing square to the other flange within 20°;
   - collinear fragment merging;
   - an unfoldability rule: in a 3-cycle of non-parallel axes, drop an edge only if it is short or thin.
4. **Metrology** uses E3's `_flat_start`, `_reseat_plane` and `_fit_side` station fits to place the measured outer apex and span.
5. **E3's own acceptance rule.** The station loop must give valid stations at ≥ max(3, ⌈0.6 n⌉) of the *n* stations; otherwise the edge is LOW_DENSITY. On it04, edges with 0 valid stations were 5 TP against 31 FP.
6. **Rolled-kink rejection.** A measured outer radius > 12–17 *t* is not a press-brake bend. This is the one numeric threshold read off DEV, and grouped CV chose its value (§7.1).

### 4.3 Learned selector with family-grouped CV

Each Z2 line is pooled into the Z3 candidate table:
- if a Z3 candidate duplicates it, it becomes an *agreement flag* on that candidate;
- otherwise it becomes an *extra candidate* with no Z3 features.

The selector is a gradient-boosted classifier over **21 Z3 features plus 5 pool flags** [E-PRE]:
- the Z3 features: length, slice support, bend angle, flank lengths, skin fractions, residuals and rank;
- the pool flags: has_z3, in_z2, in_a0 (always 0), n_src, ext_log_len.

Design choices:
- **Folds.** `dev_folds.json` has 5 folds grouped by five-digit family group (80 groups at freeze, 25/24/24/24/24 scans). It was frozen by the integrator before any lane fitted anything [E-README].
- **Threshold.** Fixed a priori at 0.5. Fold-tuned thresholds gave 0.939, against 0.944 fixed (§7.1).
- **No stacking, no NMS.** A parallel-twin NMS made F1 monotonically worse, because it removes real joggle neighbours [E-INT].
- **Final models** were fit on all 125 DEV scans. Their sha256 values are pinned in `PREREG.json`.

### 4.4 Two-skin thickness estimation and outer-apex placement (D1)

Code: `research/zeroshot/d1_two_skin/` [E-D1].

**Thickness.**
1. Sample 120 k area-weighted surface points.
2. Ray-cast along −n. Accept a hit only where the ray leaves the material through an antiparallel face (cos < −0.85), at 0.25–12 mm.
3. Take the smoothed histogram mode, then the median within ±20 %.

**Placement.** The *gated* snap moves each line to the nearest outer-apex locus, only if the shift is ≤ *t*/sin(θ/2) + 1 mm. The gate matters: ungated, the profiler latched onto a neighbouring bend about 8 mm away and destroyed 2 TPs.

The geometric motivation is that, for included angle θ, the two skins' apex lines are *d* = *t*/sin(θ/2) apart along the bisector:
- 1.41 *t* at 90°;
- 3.9 *t* at 30° [E-RES].

### 4.5 CAD priors (CAD-assisted variant only)

When a compiled CAD exists, rotation-invariant facts from `compiled_part.json` can constrain zs without registration [E-INT, E-CAD]:
- bend count *n*;
- nominal included angles;
- *t*;
- the number *K* of distinct axis directions.

The priors tested:
- a **count cap** (keep the top *n* lines, full-stage only);
- a **family cap** of *K* + 2 axis families for Z3, with CAD *t*;
- an **angle window**.

Section 7.7 reports which helped.

---

## 5. Dataset

### 5.1 Inventory

The census covered the repository, the source-asset tree and the local Downloads, Desktop and Documents folders. Nothing was downloaded. It found 289 unique mesh files by sha256, after custody exclusions [E-README]. **186 usable physical scans** remain, from **147 part families** in **132 five-digit family groups**:
- 169 full-stage and 17 partial;
- 60 gold scans have a STEP file, and 56 of those compile with E1;
- 122 gold scans have no CAD.

Excluded:
- Eyal50 quarantine scans (owner decision);
- one SKU overlapping HOLDOUT10;
- possible CAD reference meshes;
- synthetic LIDAR_DEMO scans and fixtures;
- duplicate twins.

Sheet thickness on the D1 subset, against CAD, is 0.9–3.2 mm, median about 1.5 mm [E-D1]. Elsewhere in DEV there is 4–6 mm plate [E-Z3].

**Every usable scan is consumed by rule.** Each had at least one prior label, or appears in an earlier development, DEV-CHECK or FRESH set. So no local set is "unseen" in the strict sense [E-PLAN OD-1].

### 5.2 Split (frozen before any new engine output)

`SPLIT_FROZEN.json`, sha256 `d25ffd4f…`, bound in freeze-ledger NOTE seq 11 [E-README].

| Portion | Scans | Families (5-digit groups) | Gold lines | Gold source | CAD |
|---|---|---|---|---|---|
| DEV (as frozen) | 121 (116 line + 5 zero-bend controls) | 91 line-scan families | 586 | 103 GOLD_SET_V2 audited vision; 18 accepted AI | 59 with STEP |
| DEV (as evaluated, + 4 newly gold-marked) | 125 (120 line + 5 controls) | 94 | 619 | + 4 via the §5.3 protocol | |
| **TEST-B (sealed)** | **61, all full-stage** | **50 (49)** | **317** re-audit (329 old FRESH) | re-audit gold of record | 1 STEP |

TEST-B consists of the 61 gold scans that **no engine or lane had ever run on**. Their five-digit family groups are disjoint from every DEV scan, and their labels were never read in any tuning loop. Before this programme, the only output ever produced on them was the frozen A0 baseline, run once, aggregate only [E-BASE].

### 5.3 How the gold was made

Protocol: `PLAN.md` §2 and `gold/MARKER_HOWTO.md` [E-PLAN, E-GOLD].

1. **Pre-registration.** Each scan was appended to the hash-chained `gold/GOLD_LEDGER.jsonl` (now 130 entries) as REGISTERED. A guard refuses any engine run on a scan that is not FROZEN, and refuses TEST-B without `--final-testb`.
2. **Two blind markers per scan** (A and B, separate AI agents labelled `claude-vision-ai-v1`).
   - Each worked on neutral-id scan copies (`gb_xxxxxx`), in batches of 10–11 scans plus 1 blind canary.
   - Markers saw only:
     - the raw scan;
     - a planar-patch "prep" measurement aid, which proposes no bends;
     - shaded renders, close-ups and perpendicular sections.
   - They never saw engine or lane output, CAD, prior labels or each other's files.
   - Each marker ran a missed-bend sweep on raw renders without the patch list.
3. **Adjudication.**
   - The frozen matcher compared A with B. Pairs within 1.5 mm and 2° were accepted.
   - Every disagreement went to an adjudicator, who returned one of ACCEPT_A, ACCEPT_B, MERGE, NOT_A_BEND or MISSING_ADD, with a written reason.
   - For TEST-B, the old FRESH label entered **only at adjudication**, as a third marking.
4. **Canary QC.** Each marker was scored against the canary's GOLD_SET_V2 gold. Marker-vs-canary F1 must be ≥ 0.95, or the batch is redone.
5. **Write and freeze.** Labels were written to `ai_bend_lines_zs_v1[_SEALED]`, with a provenance sidecar holding both markings and the adjudication. The ledger status was then set to FROZEN with the label sha256. Only then could any engine run on that scan.

**Gold QC records** (from `gold/*/ADJUDICATION_*.json` and per-scan adjudication files):

| Batch | A/B agreement (frozen matcher) | Canary (A / B / adjudicated) | FRESH vs re-audit | Notes |
|---|---|---|---|---|
| D0 (DEV, 4 + 1 canary) | 4 of 5 items F1 1.0; 1 item 0.90 | per-scan records only | n/a | |
| T1 | F1 0.911 (strict 1.5 mm / 2°: 0.893) | 0.615 / 1.0 / first pass **0.769 → redo** → 1.0 | | The redo reversed an end-tab MERGE; all TEST-B final lines were re-checked for the same error class |
| T2 | 10 TEST-B scans: 41 TP of 41 / 42 (F1 ≈ 0.988) | batch-level record not located | | |
| T3 | F1 0.991 | 1.0 / 1.0 / 1.0 | F1 0.953 (4 FRESH lines wrong, 1 missing) | |
| T4 | F1 1.000 | 1.0 / 1.0 / 1.0 | | **14 MISSING_ADD, all with a FRESH counterpart; 4 lines neither blind marker had** |
| T5 | F1 0.892 | 1.0 / 1.0 / 1.0 | | |
| T6 | F1 1.000 (4 pairs outside the 1.5 mm / 2° tolerance) | 1.0 / 1.0 / 1.0 | F1 0.906 (8 split lines merged) | |

Across all adjudication files the verdict counts are: MISSING_ADD 18, ACCEPT_B 8, ACCEPT_A 4, MERGE 3, NOT_A_BEND 1 (computed for this draft).

**Two caveats follow from these records and are carried into §10.**
1. **High A/B agreement partly reflects a shared tool.** Many accepted pairs match at 0.000 mm and 0°, because both markers ran the same `prep`/`lines` tool on the same patches. Agreement measures the choice of patch pairs and extents, not independent geometry.
2. **The re-audit is not fully independent of the old FRESH label.** In T4 the adjudicator added 14 lines that had FRESH counterparts, 4 of which neither blind marker had drawn. Every such change has a written reason grounded in sections.

The re-audit changed the line count on 5 TEST-B scans, from 329 to 317 lines in total. The main change was merging folds split by windows into one line, per the GOLD_SET_V2 window rule [E-TB].

---

## 6. Evaluation protocol

1. **Matcher frozen**, never changed during the programme (§3).
2. **DEV only for all design.** Every fitted quantity is chosen by 5-fold family-grouped CV on `dev_folds.json`, and only out-of-fold numbers are reported. Post-processing choices in v2 use *nested* grouped CV [E-V2].
3. **Each lane delivers:**
   - eval-lines for all DEV scans;
   - a run record with code fingerprint, wall time and RSS;
   - an error budget;
   - overlays of its 5 worst scans, opened and checked by eye [E-PLAN §3].
4. **Independent audit of Z3.** The audit [E-AUD]:
   - checked that the eval-lines equal the OOF selection on all 121 scans, with no mismatch;
   - rescored independently, with identical counts;
   - re-ran CV with random grouped folds: g5 ×10 gives mean 0.9285, range 0.912–0.942;
   - compared touched and untouched scans: scans named in the iteration log score 0.870 (28 scans), against 0.959 for the rest (71 scans);
   - checked determinism: identical endpoints on re-run.
5. **Pre-registration** (`testb/PREREG.json`, sha256 `86043f50…`, freeze-ledger NOTE seq 12 [E-PRE, E-TB]) fixed:
   - the code fingerprint (60 files: lane, Z3, Z2, D1, guard, E3 metrology modules and eval writer), the config sha and the model shas;
   - the threshold rule (fixed 0.5);
   - the scorer fingerprint, the bootstrap seed and the failed-scan rule;
   - the gold of record (61 label shas) and the secondary gold column;
   - the **decision rule**: the goal is met iff primary micro ≥ 0.85, as a point estimate with the CI reported;
   - an **expected range** of 0.84–0.92 for the primary.
   - The expected range was derived from DEV OOF 0.944, shrunk using the audit's touched-vs-untouched gap.
6. **CLI consistency.** The frozen CLI reproduced the DEV dump-pipeline selections exactly: 375/375 identical line sets (125 scans × 3 lanes), all hash pins OK.
7. **One-time run.**
   - Freeze-ledger RUN seq 13 was appended before the first process.
   - 61 scans × 3 lanes = 183 processes ran, each through the gold guard with `--final-testb`. The lane re-verified its pinned hashes at load.
   - Result: 183/183 OK, 0 failed.
   - Scoring ran once, and the results sha is bound at NOTE seq 14.
   - **TEST-B is now CONSUMED**: no re-run and no tuning on it.
8. **Post-hoc analysis** of TEST-B errors is labelled POST-HOC and changes nothing in the frozen lane.

---

## 7. Results

### 7.1 DEV out-of-fold

**Table 1. DEV OOF, 125 scans: 120 line scans / 94 families / 619 gold lines, plus 5 zero-bend controls [E-INT].** The A0 row is on the 116-scan frozen DEV.

| Arm | P | R | micro F1 [95 % CI] | family-macro [CI] | TP / FP / FN | ctrl FP | runtime median / P95 |
|---|---|---|---|---|---|---|---|
| A0 baseline (old lane) | 0.60 | 0.22 | 0.322 [0.262, 0.378] | 0.267 [0.207, 0.329] | | 0 | 229 / 432 s |
| Z2 alone (it08) | 0.934 | 0.897 | 0.915 [0.877, 0.947] | 0.867 [0.808, 0.919] | 555 / 39 / 64 | 0 | 3.7 / 7.3 s |
| Z3 alone (GBM) | 0.923 | 0.906 | 0.914 [0.885, 0.940] | 0.905 [0.860, 0.942] | 561 / 47 / 58 | 5 | 1.8 / 9.9 s |
| **zs_lane_v1 (Z3 + Z2 pooled)** | **0.942** | **0.947** | **0.944** [0.919, 0.966] | **0.944** [0.914, 0.969] | 586 / 36 / 33 | 5 | 5.7 / 16.5 s |
| zs_lane_v2 (post-v1, DEV only) | 0.961 | 0.955 | 0.958 [0.937, 0.975] | 0.951 [0.922, 0.974] | 591 / 24 / 28 | 1 | (CLI) 6.5 / 21.3 s |

- **zs − Z3, paired family bootstrap:** +0.030 micro [+0.013, +0.050] and +0.039 macro [+0.015, +0.070]. 25 families are better and 6 worse.
- **zs − Z2:** +0.029 [−0.005, +0.067].
- **v2 − v1:** +0.0135 [+0.0023, +0.0277]. 5 families are better, 0 worse, and control FP lines fall from 5 to 1 [E-V2].
- v2 was built after TEST-B, on DEV only. Its post-hoc re-score on the consumed TEST-B is unchanged (§7.3, §9.3).

**Subgroups** (zs / z3 / z2 micro) [E-INT]:

| Subgroup | zs | z3 | z2 | Note |
|---|---|---|---|---|
| FRESH-DEV, 17 line scans | 0.950 | 0.857 | 0.945 | Outside Z2's rule design |
| The 4 newly gold-marked scans | 0.841 | 0.724 | 0.817 | |
| The other 99 | 0.951 | 0.936 | 0.917 | |

**Optimism controls.**
- Z2's structural rules were designed by reading DEV overlays across all folds. Only its numeric thresholds went through nested grouped CV: OOF 0.907, against 0.917 for the final config [E-Z2].
- The Z3 audit's touched/untouched gap, 0.870 vs 0.959, is the best internal estimate of rule-design optimism [E-AUD].

### 7.2 Candidate birth: why recall jumped

**Table 2. Candidate recall on DEV** (every candidate kept, frozen one-to-one matcher).

| Source | Candidate recall | Candidates / gold | Source |
|---|---|---|---|
| A0 final output | 0.220 | | [E-BASE] |
| A0 liberal pool (`f1_uniform_scan_only_candidate_pool_v3`, median 642 merged candidates per scan) | **0.812** (476/586) | ≈ 100× | [E-Z4] |
| Z3 birth stream (`gate.birth`, unfitted) | **0.964** | **1.83×** | [E-Z3] |
| Z3 all candidates | 0.968 | 5.12× | [E-Z3] |
| Z2 flange-pair recall (proxy, GOLD_SET_V2 full / all) | 0.976 / 0.966 | | [E-Z2] |
| A0 pool + Z2 | 0.937 | | [E-Z4] |
| A0 pool + Z3 + Z2 | 0.995 (583/586) | | [E-Z4] |

Only 19 gold lines (3.2 %) are never born by Z3. The A0 pool held 81 % of gold lines but the lane kept 22 %. On the 36 scans where A0 emitted nothing, its pool still held 72.7 % of their lines [E-Z4]. Appendix A develops this.

### 7.3 TEST-B: the one-time result

**Table 3. TEST-B, 61 scans / 50 families, against the re-audit gold of record (317 lines) [E-TB].**

| Lane | P | R | **micro F1 [95 % CI]** | **family-macro [95 % CI]** | TP / FP / FN | Pre-registered expected range | Runtime median / P95 / max |
|---|---|---|---|---|---|---|---|
| **Primary: zs_lane_v1** | 0.872 | 0.839 | **0.855** [0.764, 0.926] | **0.855** [0.782, 0.918] | 266 / 39 / 51 | 0.84–0.92 (inside) | 8.0 / 20.2 / 24.0 s |
| Secondary: Z3 alone | 0.890 | 0.814 | 0.850 [0.746, 0.928] | 0.857 [0.787, 0.917] | 258 / 32 / 59 | 0.76–0.88 (inside) | 3.0 / 10.6 / 20.3 s |
| Secondary: Z2 alone | 0.800 | 0.808 | 0.804 [0.716, 0.875] | 0.785 [0.690, 0.872] | 256 / 64 / 61 | 0.82–0.92 (**below**) | 4.6 / 9.4 / 10.7 s |

**Table 4. Same lanes against the old FRESH labels (329 lines, 15° floor, not re-audited), and the A0 baseline on those labels.**

| Lane | P | R | micro F1 [CI] | family-macro [CI] | TP / FP / FN |
|---|---|---|---|---|---|
| zs | 0.928 | 0.860 | 0.893 [0.811, 0.952] | 0.875 [0.804, 0.936] | 283 / 22 / 46 |
| Z3 | | | 0.889 [0.791, 0.952] | 0.877 [0.810, 0.935] | 275 / 15 / 54 |
| Z2 | 0.847 | 0.824 | 0.835 [0.753, 0.902] | 0.803 [0.706, 0.887] | 271 / 49 / 58 |
| **A0 (old lane, frozen, run once)** | 0.722 | 0.198 | **0.310** [0.236, 0.376] | **0.252** [0.170, 0.339] | 65 / 25 / 264 |

**Reading.**
- **Verdict.** The pre-registered goal is met on the point estimate, 0.855 ≥ 0.85, but the CI lower bound is 0.764. With 61 scans and 50 families this is a **narrow pass, not a confident one**.
- **Integration benefit.** zs − Z3 on TEST-B is +0.005, within noise. The DEV margin of +0.030 did not replicate at this sample size.
- **Like-for-like versus the old lane.** On the labels the baseline was scored against, F1 rises from 0.310 to 0.893 and recall from 0.198 to 0.860.
- **Gold dependence.** All lanes score about 0.04 lower against the stricter re-audit, mostly through SPAN errors from the window-merge convention.

**Post-hoc: zs_lane_v2 on the consumed TEST-B** (POSTHOC_CONSUMED; not a held-out result) [E-V2].
- v2 was frozen after the v1 run and re-run once on TEST-B with nothing changed afterwards.
- Result: micro **0.8553** [0.7645, 0.9264], family-macro 0.862 [0.791, 0.923], TP/FP/FN 266 / 39 / 51. That is identical to v1 on micro (Δ 0.000 [−0.009, +0.010]; macro +0.007 [−0.009, +0.030]).
- Its 3 joins on TEST-B fixed one family (12851000, 0.5 → 1.0) and hurt another (49461000, 1.0 → 0.857). The error classes are unchanged in total.
- The TEST-B span class is mostly *not* the window-gap pattern the DEV-derived plate test recognises: 50278001 was not joined, and 26687021's over-merge was untouched.
- **Reading.** A DEV gain whose paired CI excludes 0 (+0.0135) transferred as exactly zero. DEV at about 0.95 is saturated as a design signal (§9.3, §10).

**Per-scan view** (read-only stats computed for this draft from `RESULTS_per_case.json`):
- zs is perfect on **36 / 61** scans and gets the exact count on 39 / 61; the median per-scan F1 is 1.0.
- 2 scans have zero TP: families 44001000 and 49566000.
- The worst families are 44001000 (0.0), 49566000 (0.0), 49553000 (0.40), 26687021 (0.41) and 50278001 (0.43).
- Gold lines per scan range 1–13, median 4.
- Peak RSS ≤ 5.9 GiB.

### 7.4 Ablations

**Table 5. Ablations** (DEV unless stated). The comparisons are within a cohort; cohorts differ between rows, as noted.

| Ablation | Cohort | Result | Source |
|---|---|---|---|
| **D1: A0 + gated outer-apex snap** | GOLD_SET_V1, 50 scans | F1 0.4020 → **0.4020**. Matched offset median 1.17 → 0.049 mm | [E-D1] |
| D1: A0 on the mid-surface | same | 0.4020 → **0.3316**. NO_PREDICTIONS rose from 13 to 21 scans | [E-D1] |
| D1: inner-skin-offset FNs among A0 FNs | same | **0 / 215**. 185 / 215 missed entirely | [E-D1] |
| D1 snap in zs | DEV 125 | 0.944 with snap vs 0.943 raw | [E-INT] |
| Z2 + D1 snap | DEV 99 | +1 TP / −1 FP (0.912 vs 0.907) | [E-Z2] |
| **Z4: GBM selector on the A0 pool** | DEV 116 | 0.584 → 0.675 (dedupe radius chosen by inner CV) → 0.686 (stage-2 stacking) | [E-Z4] |
| Z4: A0 pool + Z2 | DEV 116 | 0.837 / 0.782 | [E-Z4] |
| Z4: A0 pool + Z3 + Z2 | DEV 116 | 0.895 / 0.876. Below Z3 + Z2 alone (0.937): about 650 A0 candidates per scan dilute the table | [E-Z4] |
| Z3 + A0 pooled vs Z3 + Z2 pooled | DEV 121 | 0.926 vs **0.937** | [E-Z3] |
| **Z5: CP-SAT global consistency** (two-skin rule, flange pairs, strip penalty, near-tree bound) | DEV 116 | 0.675 → 0.677 (+0.002). Inner CV picked no flange term in any fold. On the Z3 + Z2 + A0 union it picked no term at all | [E-Z4] |
| Fixed vs fold-tuned threshold | DEV 125 | zs 0.944 vs 0.939; z3 0.914 vs 0.913 | [E-INT] |
| Parallel-twin NMS (none / 4 / 6 / 8 mm) | DEV 125 | 0.944 / 0.940 / 0.938 / 0.933 | [E-INT] |
| Blind collinear gap merge, G = 2 … ∞ mm | DEV 125 | non-monotone, 0.945 … 0.871. Rejected as noise | [E-INT] |
| Z3 slice step 1.5 vs 2 mm | DEV 121 | no gain (0.919 vs 0.925 OOF at it5), with 8.3× candidates | [E-Z3] |
| Z2 rolled-kink filter on / off | DEV 99 | 0.903–0.915 on vs 0.885–0.888 off (the rule matters, its exact value does not) | [E-Z2] |
| v2 flange-continuity join (nested CV) | DEV 125 | +0.0135 [+0.0023, +0.0277]. SPAN FP 22 → 9 (final) | [E-V2] |
| v2 normal-sweep features | DEV 125 | −0.0023 [−0.0067, +0.0017]. Not adopted as features; adopted as a 0.25 gate that removes 4 of 5 control FPs | [E-V2] |
| v2 split rule, tiny-bend rescue, scale-relative floors | DEV 125 | rejected by inner CV or inert | [E-V2] |

**The D1 null result.** The two-skin hypothesis is *true as geometry*. The median A0 line sits 1.05 mm (0.63 *t*) inside the outer apex, and 85 of 100 profiled A0 lines sit nearest an inner locus. It explains **none** of the F1 gap on this corpus, because the stock is thin enough that the offset never crosses the 4 mm gate. D1 survives as cheap infrastructure:
- thickness median absolute error 0.055 mm against CAD on 41 scans, 39 / 41 within 0.2 mm;
- the gated snap puts every lane on the gold convention.

The regime where it would matter (*t* ≥ 2.9 mm at 90°, or acute bends) is rare here.

### 7.5 Error analysis

**Table 6. Error budget** (diagnose).

| | FN: MISSED | FN: SPAN | FN: OFFSET | FN: other | FP: SPAN | FP: SPURIOUS | FP: OFFSET | FP: other |
|---|---|---|---|---|---|---|---|---|
| A0, DEV 116 (458 FN / 87 FP) | **373** | 52 | 26 | 6 ANGLE | 37 | 39 | 4 | 6 ANGLE |
| zs, DEV OOF 125 (33 / 36) | 13 | 13 | 5 | 2 TAKEN | 22 | 8 | 5 | 1 TAKEN |
| zs, TEST-B (51 / 39), POST-HOC | 16 | **28** | 7 | 0 | **24** | 13 | 0 | 2 TAKEN |

The old lane's errors are almost all MISSED. The new lane's errors are spread across classes, and on TEST-B they are dominated by **span convention**. The POST-HOC TEST-B classes [E-TB]:

1. **"Same two plates?"** About 28 FN and 24 FP. Two failure directions:
   - over-merge: separate 24–65 mm tabs on one base emitted as one 150–180 mm line;
   - under-merge: a 433 mm fold broken by windows emitted as 256 + 70 + 70 mm fragments.

   This is exactly where the re-audit gold differs most from the conventions the selector was trained on.
2. **Tiny bends next to a long fold.** 6 FN on one scan: six 6–7 mm bends 12 mm from a 100 mm line.
3. **Birth failure on a small, coarse part.** 6 FN on a 50 × 17 × 18 mm part with 20 k vertices. The Gauss-map peaks fell below the 20 mm² area floor, and Z2 found 0 lines.
4. **Selector miss.** 2 FN: Z3 had 152 candidates and the selector kept none.
5. **Spurious lines.** 13 FP: cut walls, seams and twins on featured parts.

**Remaining DEV failures by geometry** [E-Z3, E-Z2]:
- lattice windows removing the wall along about 85 % of a 419 mm fold;
- folds following a bent web;
- box-corner seams;
- return flanges folded flat against a wall;
- rolled or cylindrical parts, which are out of the planar model by design;
- tiny parts with bends < 8 mm.

### 7.6 Runtime

Single thread, per process, including interpreter start-up and model load:

| Lane | Median | P95 | Max | Peak RSS | Source |
|---|---|---|---|---|---|
| A0 (old) | 229 s | 432 s | 600 s (timeout) | ≤ 1.93 GiB | [E-BASE] |
| Z3 detect only (DEV) | 1.65 s | 10.1 s | 19.7 s | 3.3 GiB (batch) | [E-Z3] |
| Z2 (DEV) | 3.6 s | 6.8 s | 14.3 s | 2.6 GiB | [E-Z2] |
| zs_lane_v1, TEST-B | 8.0 s | 20.2 s | 24.0 s | 5.9 GiB | [E-TB] |
| zs + CAD *t* + *K* + 2 family cap (DEV, CAD cohort) | 5.8 s | 11.7 s | 12.8 s | | [E-CAD] |
| E3 (CAD engine, DEV CAD cohort) | 15.2 s | 90.9 s | 117.9 s | 2.48 GiB | [E-CAD, E-BASE] |

The new lane is roughly 30× faster at the median than the old lane. Most of A0's cost was the CP-SAT typed union: rebuilding the pool alone took 13–60 s per scan [E-Z4].

### 7.7 CAD-assisted overlay (related system)

**Cohort.** DEV scans with a compiled CAD: 49 line scans / 33 families / 302 gold lines, plus 2 controls. The zs probabilities are out of fold, and the 2 OD-3 cases were refused by the guard [E-CAD].

**Table 7. Line-F1 on DEV, CAD cohort.**

| Arm | P | R | micro F1 [CI] | family-macro [CI] | TP / FP / FN | Can create a PASS? |
|---|---|---|---|---|---|---|
| E3 alone (e3-1.12.0, re-run) | 0.993 | 0.921 | 0.955 [0.898, 0.986] | 0.845 [0.728, 0.943] | 278 / 2 / 24 | (authority) |
| zs alone (no CAD) | 0.955 | 0.977 | 0.966 [0.942, 0.987] | 0.948 | 295 / 14 / 7 | n/a |
| zs + count cap | 0.977 | 0.974 | 0.975 [0.957, 0.990] | 0.945 | 294 / 7 / 8 | no |
| zs + symmetric ±6° window + cap | 0.976 | 0.944 | 0.960 | 0.937 | 285 / 7 / 17 | no |
| **E3 + zs for withheld bends + zs on refused scans (option 2)** | 0.993 | 0.980 | **0.987** [0.969, 0.998] | **0.972** [0.940, 0.999] | 296 / 2 / 6 | **no** |
| Registration seeding, `add` | | | 0.952 | 0.843 | 276 / 2 / 26 | **yes: 7 new PASS** |
| Registration seeding, `only` | | | 0.930 | 0.836 | 264 / 2 / 38 | **yes: 8, one on a wrong pose** |

**Findings.**
- **The count cap is the useful prior.** It halves FPs for one TP.
- **A symmetric angle window hurts.** Real parts are under-bent by +7° to +13° against nominal, e.g. 152–157° measured on a 145° nominal.
- **Gap-fill is a clean win.** zs located 15 of 21 withheld or UNKNOWN bends, all of them TPs.
- **The cross-check is cheap.** It flags 3 / 49 VALID scans, all of which already carried non-PASS characteristics, so the incremental REVIEW rate is 0 / 49. Leave-one-bend-out sensitivity is 252 / 258 = 97.7 %.
- **Registration seeding was rejected.** It produced unverifiable new PASS decisions and one wrong-pose PASS, so it is a false-PASS path.

The adopted design is a post-E3, job-level overlay (`zeroshot_overlay_v1`). It can only lower a verdict to REVIEW, and its lines are never characteristics [E-OVL].

**Angle fidelity** [E-ANG]. This is DEV, consumed data, from stored outputs only. The cohort is 294 zs↔gold matches on 48 scans and 32 families; 284 of the bends have an E3 measured angle.

| Slice | n | zs vs E3: median / P90 | within 1° of E3 |
|---|---|---|---|
| All | 284 | 0.07° / 0.52° | 92.6 % |
| Excluding the 6 E3-wrong bends and the joggles | **256** | **0.06° / 0.28°** (bias +0.02°) | **98.8 %** |
| E3 firm PASS/FAIL bends only | 209 | 0.05° / 0.22° | 99.5 % |
| Joggles (10839000, 22 bends) | 22 | 4.04° / 10.6° | unreliable |

- zs against CAD nominal (median 0.45°, P90 2°) has the same size as E3 against CAD. So that spread is the parts' real over- or under-bend, not zs error.
- **zs is on the same side of the ±1° tolerance as E3 on 96.1 %** of the 256 ordinary bends. The misses sit on the tolerance edge.
- **Joggles are not measurable at this scale.** The web is all bend radius. zs reads 4–14° more open than E3, and gold disagrees with both. They should be displayed as "joggle, angle not measured".
- **Six large disagreements were E3 errors, confirmed by an independent cross-section:**
  - a reflex-side convention flip (299° for a 61° bend, 2 bends);
  - a fully formed 90° bend decoded as PARTIALLY_FORMED (152.9°);
  - station planes reaching into the next bend on a stepped profile with about 12 mm legs (3 bends).

  On all six, zs, gold and the section agree within about 1.5°. All six were E3 FAIL or REVIEW, so they are false rejections, not false PASSes. A zs–E3 angle disagreement above about 5° on a non-joggle bend could therefore serve as a display-only REVIEW trigger on E3 FAILs.
- **Correction to the CAD-path finding.** The "+7° to +13° under-bend on 145° nominals" cited above is the joggle class, as read by zs. E3 reads those bends near nominal. The advice against a symmetric angle window stands on the count-cap evidence, but its stated cause is the joggle misreading.

---

## 8. Discussion

**Why structure beats generality here.** Every successful component encodes a sheet-metal invariant:
- few axis directions (Z3 families);
- straight bends (tracking plus a TLS line);
- constant *t* (skin rules, D1);
- piecewise planarity (Z2 flanges);
- flat material on both sides of a real bend (E3's station acceptance rule).

The two strongest FP filters were E3's acceptance rule (SPURIOUS 36 → 18) and the arc-chord rule (OFFSET FP 177 → 19). Both are physical statements, not fitted cutoffs.

**Birth versus selection.** Earlier work disagreed on whether selection or birth was the bottleneck. The broad-set pool held about 85 % of gold, which pointed to selection; on the daily cohort 35 / 51 misses were never born, which pointed to birth [E-LESS, E-SRC].

The programme shows that both were true, and that fixing selection on a 100× pool is the harder path:
- a GBM on the A0 pool, with AUC 0.980, reached only 0.686;
- a 1.83× pool with 0.964 recall reached 0.914 with a similar selector.

A short, high-recall candidate list made the learning problem small enough for 120 labelled scans.

**Reusing the CAD engine's metrology.** Z2 inherits fixes that took the CAD engine several rounds to learn: flat-start windows past large real radii, local rather than global planes on bowed bases, and the outer-skin choice. The fixes transferred unchanged. This is a concrete case of measurement code being worth more than detection code.

**DEV is saturated.** v2 fixed real DEV failure classes with nested-CV-validated rules (+0.0135, paired CI excluding 0, 5 families better and 0 worse). It moved TEST-B by exactly 0.000. At about 0.95 DEV OOF, the remaining DEV errors are few and idiosyncratic, so rules that fix them describe DEV conventions rather than general geometry. Further progress needs new labelled data, not more DEV iteration.

**What did not transfer to TEST-B.** The DEV-to-TEST-B drop (0.944 → 0.855) matches the audit's touched/untouched gap (0.96 → 0.87). The new error mix on TEST-B (span convention) also points to label-convention shift as much as geometry. A human-annotated test is needed to separate the two (§12).

---

## 9. Negative results

These are reported as results. The old architecture's history is a strength of the paper: it documents how a small-data industrial detector fails, and why the redesign was necessary.

### 9.1 The old architecture: patch graph + CP-SAT typed union, and the V-chain (August–October 2026)

- **Architecture.** A0 is `backend/patch_graph_latent_decomposition.py`, 10 419 lines. It builds surfel atoms, a transition-atom graph and flange/bend hypotheses, then labels them with a CP-SAT typed union and many junction, corridor and birth recovery passes. A 3 222-line policy layer adds count "rescues" (`zero_shot_bend_profile.py`). On top of it, the V-chain added frozen intervention primitives: V16 routes, V43–V50 stitching and ownership, V51 single-axis parallel-return recovery [E-OLD, E-V16].
- **Process.** About 923 documents in `docs/research/`, with intervention labels up to V73. They record one-shot precommits, terminal stops and external architecture rulings.
- **What it achieved.**
  - V16 reached F1 0.717 on a 35-scan development corpus, with a minimum-family F1 of 0.0 [E-V16].
  - On GOLD_SET_V2, A0 scored 0.340 micro / 0.273 macro, at recall 0.23 [E-LESS].
- **Transfer collapse.** Frozen V51 scored 0.634 on Legacy37 and **0.197** on Today28 [E-V43]. Source recovery fit Today28 at 0.341 in-sample and **0.192** under grouped LOCO [E-SRC]. Axis/span recovery fell from 0.632 by resubstitution to 0.483 case-group-held-out [E-AXS].
- **Diagnosis** [E-LESS §4]:
  - wrong representation: no two-skin model, greedy sequential planes, spans from patch overlap;
  - selection by policy and cutoffs fitted on small consumed cohorts;
  - a circular evaluation: 45 of 71 "human" Gold11 lines were machine drafts from detector output;
  - complexity that produced runs with **no output at all**.
- **Protocol invalidations, kept on record.**
  - The V51 Gold37 blind-once attempt was invalidated before detector execution, because sealed fields were printed during a manifest check.
  - The first new-corpus precommit was invalidated by its own preflight, because of an import-path defect [E-OLD §4].

  Neither produced a score. Both made the eventual sealed-test discipline stricter.

### 9.2 Learned filters on top of geometric detectors

Jev (a TypeSafe System One judgement model) scored ranking AUC 0.69–0.88, against 0.93 for plain-feature LR. As a filter it left F1 flat (0.443 → 0.444). On E3 lines its AUC was 0.51, and every filter removed only true lines. The track was closed [E-LESS §1].

### 9.3 Within this programme

- **D1 as an F1 lever:** null (§7.4).
- **Mid-surface input to A0:** −0.070.
- **Z5 CP-SAT consistency:** +0.002. Its flange-pair term failed because the pool's plane table has a median of 7 planes per scan: only 27 % of true candidates find a flange pair, against 12 % of negatives.
- **Z4 on the A0 pool:** a ceiling of about 0.69. A0 adds nothing on top of Z3 + Z2.
- **Z2 virtual-riser discovery** (it06): fixed 0–1 lines per target scan and added 2–6 FP.
- **Z2 plane-distance purify** (it02): a regression that cut the spans of bowed bases.
- **Z3 rules aimed at one 5 mm plate part** (it5): lost 5 born gold lines elsewhere and were reverted.
- **Blind gap merge and parallel-twin NMS:** noise or harm.
- **v2:** a split rule, a tiny-bend rescue (7 positives against about 370 negatives) and a scale-relative floor were all rejected, or inert, on DEV.
- **v2 non-transfer (POSTHOC_CONSUMED).** The adopted v2 rules gained +0.0135 on DEV OOF and 0.000 on the consumed TEST-B (0.8553 both). This is the programme's clearest evidence of DEV saturation.
- **Registration seeding** for E3: rejected because of a false-PASS path.

---

## 10. Limitations

1. **The gold is AI-made, and there is no human annotation of the test set.**
   - All TEST-B gold comes from AI vision agents (`claude-vision-ai-v1`) under a written protocol.
   - The canaries pass at F1 1.0 after adjudication, but they are themselves AI-audited (GOLD_SET_V2) or AI-relabelled.
   - The historical human Gold11 labels were found mostly wrong or machine-drafted, so "human" is not automatically better. Still, no human has verified the test labels.
2. **The markers share the same prep.** Both blind markers use the same planar-patch `prep` tool and `lines` placement, so their agreement is inflated. Many pairs match at exactly 0 mm (§5.3). In one batch, a canary failure (0.769) was caught only by the QC rule.
3. **The re-audit leans partly on the old label.** The FRESH label entered as a third marking, and in T4 it contributed 14 added lines.
4. **Single company, single scanner pipeline.**
   - All parts come from one manufacturer's press-brake work.
   - All scans are exported meshes from one ZEISS / Geomagic pipeline.
   - Sheet stock is mostly 0.9–3.2 mm.
   - There are no hems in quantity, no thick-plate acute bends and no other scanners. Rolled parts are out of scope.
5. **TEST-B is consumed by rule.** Every TEST-B scan already had an old FRESH label made on 2026-10-17, and the corpus has been worked on for months. TEST-B is programme-sealed, but it is not "new data". It has now been used once and cannot be re-used.
6. **The rules were designed on DEV.** The structural rules of Z2 and Z3 were designed by reading DEV overlays across all folds. Grouped CV bounds parameter selection, not rule design. The touched/untouched gap (0.96 vs 0.87) shows the optimism this causes.
7. **DEV is saturated as a design signal.** The v2 result (+0.0135 on DEV, 0.000 on TEST-B, post-hoc) shows that DEV-driven rules no longer transfer. Every further improvement claim needs new data, and the DEV OOF numbers (0.944, 0.958) should be read as an upper bound, not an estimate.
8. **The sample is small.** With 61 scans and 50 families, the primary CI is [0.764, 0.926]. A narrow pass cannot be told apart from a narrow miss.
9. **Stacking leakage (minor).** The Z3 OOF probability used as a feature in Z4's iter-7 union came from models that saw the other folds. The frozen lane does not use that construction.
10. **The data cannot be released.** The scans are customer parts. Neither code nor data can currently be published, which limits external reproducibility (§12).
11. **Scope.** The method outputs lines and angles, not tolerance decisions. Angle fidelity is measured against E3 on DEV only, and joggle angles are unreliable (§7.7).

---

## 11. Reproducibility

- **Code.** `research/zeroshot/{z3_slicing, z2_self_recipe, d1_two_skin, z4_selector, zs_lane_v1, zs_lane_v2}`, with tests under `tests/test_prod_zeroshot_*`.
- **Pins.** The TEST-B lane fingerprint `af87df2f…` covers 60 files. Config `config_v1.json` has sha `dec044e8…`. The model sha256 values are in `PREREG.json`.
- **Ledgers.** Freeze-ledger seq 11 binds the split, seq 12 the pre-registration, seq 13 the run and seq 14 the results. The gold ledger has 130 entries and its chain verifies [E-CAD].
- **Determinism.**
  - A0 replayed identically [E-BASE].
  - Z3 gave identical endpoints on re-run [E-AUD].
  - The frozen CLI equals the dump pipeline on 375 / 375 runs.
- **Commands.**
  - `integration/run_cohort.sh <tsv> <out> <variant> <i> 3` runs a cohort.
  - `python -m research.zeroshot.zs_lane_v2.train …` trains.
  - The scorer is `research/zeroshot/zs_lane_v1/scoring.py`, which wraps the frozen `score_case`.
- **Not reproducible externally:** the scans and labels (customer data), and the A0 pools, which live only in temporary storage.

---

## 12. Conclusion

A structure-aware, scan-only detector of about 7 k lines replaced a 13 k-line patch-graph lane. Axis-family slicing and a flange graph measured with the CAD engine's own metrology took zero-shot bend-line detection from 0.31 to 0.86 micro F1 on a sealed, family-disjoint, pre-registered test, at about 8 s per scan. Against the older labels the gain is 0.31 → 0.89. The gain is almost entirely recall, and the recall came from *birthing* the right candidates rather than from a cleverer selection over a huge pool.

The same lane fills gaps in a CAD-assisted engine without ever creating a PASS. Its angles match the CAD engine's metrology to a median of 0.06° on ordinary bends. The result is a narrow pass on AI-made gold from one company, and a DEV-only improvement that left TEST-B unchanged shows the development set is exhausted as a design signal. The decisive next step is a human-annotated, genuinely new test set (Appendix B).

---

## Figure and table plan

| # | Type | Content | Data source |
|---|---|---|---|
| F1 | Diagram | Pipeline: scan → Z3 (Gauss map → sections → corners → tracks) ‖ Z2 (flanges → graph → E3 station metrology) → pool → GBM → D1 snap | method text |
| F2 | Figure | One slice profile with skins, edge walls, an arc-chord merge and a joggle marked; the tracked corners in 3-D | `z3/overlays/` |
| F3 | Figure | Two-skin geometry: *d* = *t*/sin(θ/2); histogram of the A0 line offset (median 0.63 *t*); gold apex convention (301 / 322) | `d1/d1_analysis.json` |
| F4 | Bar chart | Candidate recall vs candidates per gold: A0 output / A0 pool / Z3 birth / Z3 all / unions | Table 2 sources |
| F5 | Forest plot | Micro and macro F1 with CIs: A0, Z2, Z3, zs on DEV OOF and TEST-B (both golds) | `integration/README.md`, `testb/RESULTS.json` |
| F6 | Stacked bars | Error budgets for A0 DEV, zs DEV, zs TEST-B | Table 6 sources |
| F7 | Gallery | TEST-B failure classes, one overlay each (over-merge, under-merge, tiny bends, small part) | post-hoc; needs rendering from `testb/runs_TESTB_CONSUMED` (consumed, read-only) |
| F8 | Diagram + table | CAD overlay design: E3 ‖ zs → C06 overlay; Table 7 | `cad_path/CAD_PATH.md` |
| F9 | Timeline | Old architecture history: V16 → V51 → transfer collapse → programme | §9 sources |
| T1–T7 | Tables | As in §7 | as cited |
| T8 | Table | Gold QC per batch | `gold/*/ADJUDICATION_*.json` |

The gallery figures must use synthetic or owner-approved renders. Customer geometry may need redaction (Appendix B).

---

## References

Status codes:
- **[P]** listed with a DOI or URL in the 2026-08-14 draft's reference list;
- **[R]** listed with a URL or DOI in `RESEARCH.md` (2026-10-25);
- **[U]** listed without a resolvable identifier, or not verified. Check before submission.

None was re-resolved online for this draft.

1. M. A. Fischler, R. C. Bolles. Random Sample Consensus. *Commun. ACM* 24(6), 1981. doi:10.1145/358669.358692 [P]
2. R. Schnabel, R. Wahl, R. Klein. Efficient RANSAC for Point-Cloud Shape Detection. *CGF* 26(2), 2007. doi:10.1111/j.1467-8659.2007.01016.x [P][R]
3. L. Landrieu, M. Simonovsky. Large-Scale Point Cloud Semantic Segmentation with Superpoint Graphs. *CVPR* 2018. [P]
4. C. R. Qi et al. PointNet++. *NeurIPS* 2017. [P]
5. L. Yu et al. EC-Net: an Edge-Aware Point Set Consolidation Network. *ECCV* 2018. arXiv:1807.06010 [P][R]
6. X. Wang et al. PIE-NET: Parametric Inference of Point Cloud Edges. *NeurIPS* 2020. arXiv:2007.04883 [P][R]
7. N. Liu et al. PC2WF: 3D Wireframe Reconstruction from Raw Point Clouds. *ICLR* 2021. [P]
8. L. Zhu et al. NerVE: Neural Volumetric Edges. *CVPR* 2023. arXiv:2303.16465 [P][R]
9. K. Cherenkova et al. SepicNet. *CVPR Workshops* 2023. [P]
10. D. R. Roberts et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40(8), 2017. doi:10.1111/ecog.02881 [P]
11. S. Kapoor, A. Narayanan. Leakage and the Reproducibility Crisis in ML-based Science. *Patterns* 4(9), 2023. doi:10.1016/j.patter.2023.100804 [P]
12–20. Reserved for the classical crease and edge works below ([31]–[35]) and the learned primitive works ([36]–[39]). The numbering will be compacted at submission.
21. T. Rabbani et al. Segmentation of point clouds using smoothness constraint. *ISPRS* 2006. [R]
22. A.-V. Vo et al. Octree-based region growing for point cloud segmentation. *ISPRS J.* 2015. doi:10.1016/j.isprsjprs.2015.01.011 [R]
23. A. M. C. Araújo, M. M. Oliveira. A robust statistics approach for plane detection in unorganized point clouds. *Pattern Recognition* 2020. [R] (URL given, no DOI)
24. H. Isack, Y. Boykov. Energy-based geometric multi-model fitting. *IJCV* 2012. doi:10.1007/s11263-011-0474-7 [R]
25. A. Delong et al. Fast approximate energy minimization with label costs. *IJCV* 2012. doi:10.1007/s11263-011-0437-z [R]
26. D. Barath, J. Matas. Multi-X (ICCV 2017); Progressive-X (ICCV 2019), arXiv:1906.02290 [R]
27. D. Barath, J. Matas. Graph-Cut RANSAC. *CVPR* 2018. [R]
28. T. Chaperon, F. Goulette. Extracting cylinders in full 3D data using a random sampling method and the Gaussian image. *VMV* 2001. [U]
29. T. Rabbani et al. Efficient Hough transform for automatic detection of cylinders in point clouds. *ISPRS* 2005. [R]
30. Y. Li et al. GlobFit. *SIGGRAPH* 2011. doi:10.1145/2010324.1964947 [R]
31. Y. Ioannou et al. Difference of Normals as a multi-scale operator. *3DIMPVT* 2012. arXiv:1209.1759 [R]
32. Q. Mérigot, M. Ovsjanikov, L. Guibas. Voronoi-based curvature and feature estimation from point clouds. *TVCG* 2011. [R]
33. C. Weber, S. Hahmann, H. Hagen. Sharp feature detection in point clouds. *SMI* 2010. doi:10.1109/SMI.2010.32 [R]
34. D. Bazazian et al. Fast and robust edge extraction in unorganized point clouds. *DICTA* 2015. doi:10.1109/DICTA.2015.7371262 [R]
35. T. Hackel, J. D. Wegner, K. Schindler. Contour detection in unstructured 3D point clouds. *CVPR* 2016. doi:10.1109/CVPR.2016.178 [R]
36. L. Li et al. SPFN. *CVPR* 2019. arXiv:1812.07144 [R]
37. G. Sharma et al. ParSeNet. *ECCV* 2020. arXiv:2003.12181 [R]
38. S. Yan et al. HPNet. *ICCV* 2021. arXiv:2105.10620 [R]
39. Y. Liu et al. Point2CAD. *CVPR* 2024. arXiv:2312.04962 [R]
40. Y. Woo. Abstraction of mid-surfaces from solid models of thin-walled parts. *CAD* 2014. doi:10.1016/j.cad.2013.08.010 [R]
41. MidSurfer, 2024. arXiv:2405.19339 [R] (authors not recorded in RESEARCH.md)
42. Thin-surface laser-scan reconstruction pitfalls, PMC7014519 [R/U] (title and authors not recorded)
43. CN106295058A. Section-normal slicing and two-line fit for bend angle (patent). [R]
44. Aalborg University MSc thesis. Computer-Vision Based Bend Quality Inspection for Sheet Metal Parts. [R] (authors and year not recorded)
45. Ma, Yang. Sheet-metal B-rep face-edge graph features (GNN). *Sci. Rep.* 2024. [R]
46. BenDFM synthetic sheet-metal bending CAD dataset, arXiv:2603.13102 [U] (dated 2026 in RESEARCH.md; verify)
47. M. Leordeanu, M. Hebert. A spectral technique for correspondence problems using pairwise constraints. *ICCV* 2005. doi:10.1109/ICCV.2005.20 [R]
48. M. Cho, J. Lee, K. M. Lee. Reweighted random walks for graph matching. *ECCV* 2010. doi:10.1007/978-3-642-15555-0_36 [R]
49. F. Zhou, F. De la Torre. Factorized graph matching. *CVPR* 2012. doi:10.1109/CVPR.2012.6247777 [R]

Also listed in RESEARCH.md but not cited here: Lukács et al. (ECCV 1998), Taubin (PAMI 1991), Limberger & Oliveira (PR 2015), Weinmann et al. (ISPRS J. 2015), DEF, ComplexGen, SED-Net, Sonata/PTv3 and PartField.

---

## Appendix A (internal): architectural change, old pipeline → new pipeline

**Question.** Why did zero-shot recall go from 0.22 to about 0.9: DEV OOF 0.947, TEST-B 0.839 on re-audit gold and 0.860 on old labels?

### A.1 What changed

| Aspect | Old lane (A0 + V-chain) | New lane (zs_lane_v1) |
|---|---|---|
| Representation | Surfel / atom graph and transition atoms, flange and bend hypotheses, with no thickness model | Ordered 2-D cross-sections per axis family (Z3) plus a skin-paired planar flange graph (Z2) |
| Candidate birth | Greedy plane and patch extraction plus many recovery passes, about 650 merged pool candidates per scan | Corners in profiles, tracked across slices: 1.83× gold. Flange-pair edges |
| Selection | CP-SAT typed union with policy rules and count rescues, cutoffs fitted on Gold11 / Legacy37 | One GBM on 26 plain features, a fixed 0.5 threshold, family-grouped CV |
| Measurement | Line from patch intersections; span from patch overlap | E3 local skin-aware station fits (Z2); slice-support spans (Z3); D1 outer-apex snap |
| Abstention | Frequent: the solver often selected zero bends | None by design; a failed scan scores as all-FN |
| Code size | 10 419 + 3 222 lines | 2 218 (Z3) + 1 870 (Z2) + 1 337 (D1) + 1 336 (lane) |
| Runtime | 229 s median, 600 s timeouts | 8 s median |
| Labelled data | Gold11 / Legacy37 cohorts. Gold11 was partly machine-drafted (45 / 71 lines) | 120 DEV line scans / 619 lines, two-marker protocol, 5-fold family-grouped CV |
| Evaluation | In-sample on small cohorts, with overlapping calibration and evaluation (V52) | DEV OOF, independent audit, sealed pre-registered TEST-B |

### A.2 The evidence, step by step

1. **The old lane lost recall at selection, not only at birth.**
   - A0's own liberal pool contains **0.812** of DEV gold lines (476 / 586), but A0 keeps **0.220** [E-Z4, E-BASE].
   - **A0 emits nothing on 36 of 116 DEV scans** (33 NO_PREDICTIONS, 3 timeouts), losing 21 % of gold lines outright [E-BASE].
   - On 32 of those 36 line scans the CP-SAT union ran to OPTIMAL and selected zero bends from non-empty pools: 28 scans had 76–1130 candidates and 4 had near-empty pools [E-Z4].
2. **81 % of A0's misses were never born as a nearby final line.** 373 of 458 DEV FNs had no prediction within 15° / 15 mm [E-BASE]. D1 found the same on GOLD_SET_V1: 185 / 215 [E-D1]. The A0 pool shows that many of these existed upstream and were filtered out. The rest needed a new birth channel. The daily cohort had already shown this: 35 / 51 misses never appeared in the candidate stream [E-SRC].
3. **Repairing selection on the old pool hits a ceiling.** A family-grouped GBM over the A0 pool (AUC 0.980) reached **0.584**. Inner-CV dedupe raised it to 0.675 and stacking to **0.686** [E-Z4]. Most FPs were near-duplicate twins 4–6 mm from a matched line. The 100× pool is too noisy to select from with 116 scans.
4. **A structure-aware birth channel fixes recall at a small candidate count.** Z3 proposes **0.964** of gold at **1.83×** the gold count. Only 3.2 % of gold lines are never born [E-Z3]. The selector then reached 0.914–0.922 standalone.
5. **The flange graph supplies a second, complementary channel.** Z2's flange-pair recall is **0.966** (0.976 on GOLD_SET_V2 full scans). It reaches 0.915 alone [E-Z2, E-INT]. Pooling Z2 into Z3 adds +0.030 on DEV [E-INT]. Adding A0 *lowers* the score: Z3 + A0 scores 0.926 against 0.937 for Z3 + Z2, and the triple union 0.895 [E-Z3, E-Z4].
6. **E3's local fits did the measurement.** Z2 imports E3's `metrology` (`_flat_start`, `_fit_side` stations) and its LOW_DENSITY acceptance rule. That one rule halved SPURIOUS FPs, 36 → 18 [E-Z2]. The CAD engine reaches 0.990 once it knows the flanges [E-FIX3]; Z2 obtains the flanges from the scan instead.
7. **Thickness was not the missing piece.** D1 found 0 / 215 FNs explained by inner-skin offset on this stock [E-D1]. The architecture change is about birth and representation, not about skins.
8. **Labelled data and grouped CV replaced policy rules.**
   - The new lane learns one selector from 120 labelled DEV scans and 619 lines, fitted under 5-fold family-grouped CV, with a fixed threshold.
   - The old lane encoded count rescues and cutoffs fitted on cohorts of 28–37 scans, which collapsed on transfer: 0.634 → 0.197 [E-V43], and 0.341 → 0.192 under LOCO [E-SRC].
   - The Z3 audit's random regrouping (g5 ×10: 0.912–0.942) shows the new selector is stable under fold choice [E-AUD].

### A.3 One-paragraph summary

The old pipeline tried to get the bend set right through global combinatorial labelling over a noisy, two-skin-confused hypothesis graph. When the evidence was ambiguous, the optimum was often "no bends". The new pipeline asks a simpler local question many times: is there a corner in this profile, and is it the same corner in the next slice? It then measures each candidate with metrology already proven in the CAD engine, and leaves only a small, high-recall selection problem to a model trained on 5× more properly labelled scans. Recall rose because the right candidates were born, and because nothing in the pipeline is allowed to abstain.

---

## Appendix B (internal): what would make this publishable

| # | Item | Why | Status | Effort / dependency |
|---|---|---|---|---|
| 1 | **Human double annotation of a test set.** Two trained humans, blind to each other and to all engine output, plus an adjudicator. Report inter-annotator line-F1. Also have humans re-mark a sample of the AI-made TEST-B and DEV gold to measure AI-gold error | Limitation 1 is the first reviewer objection, and the shared-prep issue (§10.2) inflates the agreement statistic | Not started | About 2 × 40 scans of human time. Needs a labelling UI (Bend Label Studio exists) |
| 2 | **≥ 40 new scans**, ≥ 30 families, from new deliveries. Gold made by item 1 before any engine output; the frozen lane run once | The only honest unseen-family claim under the custody rule. TEST-B is consumed | Requested in PLAN §1.3; none on disk | Customer deliveries. Ask for STEP files too, which also tests the CAD overlay |
| 3 | **Baselines.** (a) Efficient RANSAC or region growing + adjacent-plane intersection (Schnabel; Open3D `detect_planar_patches` + intersection); (b) the old lane A0 (already have DEV and TEST-B aggregates); (c) a retrained open method (SPFN MIT, PIE-NET Apache, ComplexGen MIT) on synthetic sheet metal; (d) the CN106295058A-style slicing with a given axis, as an upper-bound angle comparator | Reviewers will ask how a generic primitive pipeline does | (b) done; (a) cheap, hours; (c) needs GPU and synthetic data; (d) optional | (a) must run on the new test set, not on consumed TEST-B |
| 4 | **A public or synthetic benchmark release.** Options: (i) a synthetic press-brake corpus with a scanner model (`scanner_model_card.json` exists; RESEARCH D8) and a released generator; (ii) a small set of owner-approved real scans of non-sensitive parts; (iii) release code plus metric plus the synthetic set only | Without data the work cannot be reproduced externally. A synthetic benchmark plus code is the usual compromise | Not started. Earlier synthetic corpora were visually rejected for domain gap (Aug 2026 records) | (i) about 1–2 days for a generator, plus a domain-gap audit; (ii) needs customer permission |
| 5 | **IP and patent timing.** Decide whether Z3 slicing and tracking, the Z2 self-recipe with E3 metrology, or the CAD-overlay gap-fill is patentable. File before any preprint or submission | Public disclosure ends novelty in most jurisdictions. CN106295058A is prior art for slicing-to-measure-angle and must be distinguished | Not started | Owner and patent counsel |
| 6 | **Customer data permission.** Written permission to publish aggregate results, figures of parts (or redacted renders) and any released scans. Confirm the part numbers in figures and tables are acceptable or anonymised | The scans are customer parts. The paper currently names part families | Not started | Owner |
| 7 | **Stronger statistics.** Pre-register the new-scan test with a power argument. At 61 scans the CI half-width is about ±0.08; tighter bounds need more families | The current pass is narrow | | Comes with item 2 |
| 8 | **Angle and position metrology claims.** The angle check is done (median 0.06°, P90 0.28° vs E3 on 256 non-joggle bends). Still to do: resolve the joggle convention, and add per-bend position error against the CAD engine | Inspection reviewers care about measurement, not only detection | Angle done on DEV; joggles and position open | Hours |
| 9 | **Rule-design optimism and DEV saturation.** Freeze the lane and test on new data (item 2). Report the touched/untouched split and the v2 null transfer as optimism estimates. Stop DEV-driven rule work | Limitations 6–7 | Evidence exists | none |
| 10 | **Venue fit.** Candidates: a manufacturing or metrology venue (e.g. CIRP-style or journal), or a 3D-vision workshop on structured reconstruction | Shapes framing and the baselines expected | To decide | |

---

## §E. Internal evidence index (not publication references)

Paths are relative to `/Users/idantal/ShermanQC/FullProject/`. `ZS` = `docs/production/runlog/zeroshot_2026-10-25`.

| Tag | Path | Used for |
|---|---|---|
| E-README | `ZS/README.md` | inventory (289 / 186 / 147 / 132; 915 lines; 169 / 17; 60 STEP), split sha, freeze-ledger seqs, dev_folds |
| E-LESS | `ZS/LESSONS.md` | A0 on GOLD_SET_V2 (0.340 / 0.273), pool 0.83–0.86, Jev, Gold11 machine drafts, E3 lessons, transfer collapse summary |
| E-BASE | `ZS/BASELINE.md`, `ZS/BASELINE.json` | A0 DEV and TEST-B aggregates, error budget (373 MISSED), 36 empty scans / 21 %, runtimes, determinism |
| E-PLAN | `ZS/PLAN.md` | OD-1..OD-4, split table, gold protocol, WP ranking, new-data requirement |
| E-RES | `ZS/RESEARCH.md` | literature, the gap finding, licence table, two-skin geometry |
| E-D1 | `ZS/d1/D1_REPORT.md`, `ZS/d1/d1_analysis.json`, `ZS/d1/ab_mid.json` | D1 null result, thickness accuracy, apex convention |
| E-Z2 | `ZS/z2/ITER_LOG.md`, `ZS/z2/oof_it07.json` | Z2 iterations, OOF 0.907, final 0.917, flange-pair recall, runtime |
| E-Z3 | `ZS/z3/README.md`, `ZS/z3/ITER_LOG.md`, `ZS/z3/RESULTS.json` | Z3 method, candidate recall 0.964 at 1.83×, standalone 0.922, unions, runtime |
| E-AUD | `ZS/audit_z3/audit_results.json`, `ZS/audit_z3/determinism.json` | rescore equality, random-fold CV, touched vs untouched, determinism |
| E-Z4 | `ZS/z4/ITER_LOG.md`, `ZS/z4/pool_recall_base.json`, `ZS/z4/a0_empty_diagnosis.json` | pool recall 0.812, A0-empty diagnosis, Z4 / Z5 ablations |
| E-INT | `ZS/integration/README.md`, `ZS/integration/oof_v1/DEV_OOF*.json` | DEV OOF table, subgroups, primary choice, NMS / merge ablations, CAD prototype |
| E-PRE | `ZS/testb/PREREG.json` | pins, features, threshold rule, decision rule, expected range |
| E-TB | `ZS/testb/RESULTS.md`, `ZS/testb/RESULTS.json`, `ZS/testb/RESULTS_per_case.json` | TEST-B results (both golds), runtime, post-hoc error classes; per-scan stats computed for this draft |
| E-GOLD | `ZS/gold/MARKER_HOWTO.md`, `ZS/gold/{D0,T1,T3,T4,T5,T6}/ADJUDICATION_*.json`, `ZS/gold/T2/*/adjudicator/adjudication.json`, `ZS/gold/T4/*/adjudication.json`, `ZS/gold/GOLD_LEDGER.jsonl` (line count only) | gold rule, protocol, QC per batch, verdict counts, FRESH dependence |
| E-CAD | `ZS/cad_path/CAD_PATH.md`, `ZS/cad_path/results/*.json` | CAD overlay options 1–4, E3 runtime |
| E-OVL | `ZS/overlay/CONTRACT_NOTE.md` | `zeroshot_overlay_v1` contract |
| E-V2 | `ZS/v2/ITER_LOG.md` (incl. "Post-hoc TEST-B re-score"), `ZS/v2/oof/OOF_base__final.json`, `ZS/v2/dev_cli/CONSISTENCY.json`, `ZS/v2/testb_posthoc_CONSUMED/RESULTS_POSTHOC_CONSUMED.json` | v2 DEV OOF 0.958, Δ vs v1, rejected rules, post-hoc TEST-B 0.8553 (Δ 0.000) |
| E-ANG | `ZS/angle_check/ANGLE_CHECK.md`, `ZS/angle_check/ANGLE_CHECK.json`, `ZS/angle_check/sections/` | angle fidelity vs E3 / CAD / gold, joggle class, the 6 E3-wrong bends |
| E-FINAL | `docs/production/runlog/final_eval_2026-10-19/REPORT.md` | E3 one-time held-out: 0.527 headline, 0.857 engine-ran |
| E-FIX3 | `docs/production/runlog/final_eval_2026-10-19/FIXES.md`, `FIXES2.md`, `FIXES3.md` | E3 GOLD_SET_V1 0.990 / 0.974, per-bend 192 / 203, compile and refusal fixes |
| E-SCAN | `docs/production/runlog/scanner_model_card.json` | scanner pipeline name (`zeiss_geomagic_export_model_card_v1`) |
| E-OLD | `docs/research/2026-08-14_sherman_qc_zero_shot_bend_inspection_paper_draft.md` | old architecture description, V-chain, invalidations, references 1–11 |
| E-V16 | `docs/research/2026-08-13_f1_v16_unified_route_professor_packet.md` | V16 dev F1 0.717, minimum-family F1 0.0 |
| E-V43 | `docs/research/2026-08-31_f1_v43_explosion_guard_development_result.md` | Legacy37 0.634 vs Today28 0.197; birth as the bottleneck |
| E-SRC | `docs/research/2026-08-31_f1_v51_source_recovery_v1_development_result.md` | source recovery 0.341 → LOCO 0.192; 35 / 51 never born |
| E-AXS | `docs/research/2026-08-31_f1_v51_today_axis_span_recovery_development.md` | axis/span 0.632 → 0.483 |
| (context) | `docs/RESEARCH_POSITIONING_BEND_WORK.md`, `docs/ONE_PAGER_PARTIAL_BEND_STATE_INFERENCE.md` | positioning: thesis-grade until benchmarked against baselines; partial-state inference is a separate open problem |
| (code sizes) | `backend/patch_graph_latent_decomposition.py`, `backend/zero_shot_bend_profile.py`, `research/zeroshot/*/` | line counts via `wc -l` |

**Custody note for this draft.** The following were not opened:
- any `*_SEALED*` label file;
- HOLDOUT10, N001–N016, EHO4 or Eyal50 data;
- face scans;
- any scan geometry.

TEST-B content was read only through aggregate and per-case *result* files and gold-adjudication records, which hold counts and reasons, not label geometry. No engine was run. The v2 post-hoc TEST-B numbers are quoted from the v2 agent's own record (`ZS/v2/ITER_LOG.md`). This draft did not re-score them.
