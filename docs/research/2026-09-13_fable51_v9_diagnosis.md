# V9 diagnosis: tested causal hypotheses behind the V8 blockers

Date: 2026-09-13. Author: Fable 5.1 worker `1519fdc2` (sole implementation
owner). Evidence sources: the terminal V8 attempt's DERIVED records only
(RESULT.json ledgers, plane equations, face box bindings, window records), the
frozen source, and synthetic fixtures. No raw scan/CAD, Gold, cache or
credential was opened. Nothing here relaxes a gate; every proposed mechanism
is a separately versioned candidate that still needs independent review,
freeze and PRE. Analysis scripts are retained in the job scratch directory
(`completion_analysis.py`, `completion_feasibility.py`, `flange_gate_synthetic.py`).

## A. Finite-start rejections (12 slots)

### Facts from the derived ledgers

1. Six scans (CG08506, CG64, CG65, CG84, CGEBA, CGF41; both arms identical)
   complete the context stage, propose 2,045–4,862 hypotheses each and pass
   none through the finite prefilter. Every hypothesis's weakest assigned
   support has possible box coverage upper ≈ 0 (histogram at 0.0 for all
   4,500/4,454/2,045/4,520/4,862/4,272 hypotheses; two CG84 hypotheses at 0.2).
2. Two-of-three hypotheses exist in five of the six scans: 3, 1, 0, 1, 3, 5
   (CG08506, CG64, CG65, CG84, CGEBA, CGF41). In each, two supports pass with
   coverage 1.0 and the third has every row definitely outside its face box.
3. In every such hypothesis the transformed third scan plane lies ON the
   assigned CAD plane: signed distance of the scan anchor from the CAD plane
   0.0–0.3 mm (the three-plane intersection fixes the pose). The third patch's
   centroid is outside the assigned face's box by 24.7–253.9 mm. So the third
   correspondence names the wrong finite face on the right (or a parallel)
   plane, not a wrong plane orientation.
4. No alternative third face was ever scheduled for those two passing
   correspondences: the four hypotheses sharing them are the sign variants of
   the same third face. The failed scan planes were assigned to 7–13 other CAD
   faces, but only inside other triplets. Quotas bind: 8,192 hypothesis records
   (6,720 triplet + 1,472 pair-slot reservation), each anchor schedules 2 of
   8,001 partner pairs, 754–960 eligible unsigned assignments.
5. CG65 has no two-of-three hypothesis (6,059 of 8,160 records rejected by
   the Wahba witness gate; five finite calls); it stays a negative for any
   completion mechanism.

### Feasibility test against exact face boxes (derived data)

For each two-of-three hypothesis: u = intersection direction of the two
passing CAD planes; for each CAD face j whose plane normal is within 8° of the
rotated third scan normal, λ_j is the shift along u that puts the third scan
plane onto face j; the completed pose is (R, t + λ_j u); a completion is
"feasible" when all three scan patch centroids fall inside their faces' root
boxes expanded by the 2 mm gate (plus an allowance). This is a centroid proxy
for the frozen row-by-row box screen and finite query, not the prefilter.

| Scan | Two-of-three hypotheses | Feasible completions (allowance 0 mm / 5 mm) | Examples (face, λ mm) |
| --- | ---: | ---: | --- |
| CG08506AE97D | 3 | 4 / 5 | coplanar face 125 (λ 0.14); parallel 141/143 (λ −2.0…−3.5) |
| CGEBA9A869AC | 3 | 2 / 2 | faces 116/118 (λ 0.7 / 2.2) |
| CGF41F49066A | 5 | 2 / 6 | coplanar 12/10 (λ 0.6 / 2.1); parallel 116/118 (λ −13…−15), 71/106 (λ −27/−47) |
| CG84C8FA12D2 | 1 | 0 / 0 | none |
| CG64C412CC73 | 1 | 2 / 2 | faces 322/359 (λ 13.0 / 14.0) |
| CG65CD575B1C | 0 | n/a | none |

Four of the six scans have at least one feasible completion under the exact
boxes. Whether a completion passes the real prefilter (≥ 75% of raw rows
within 2 mm of the face over all three supports) can only be observed in a
physical run.

### Context lane (the existing third-degree-of-freedom mechanism)

Per family, six endpoint slots (FACE0/FACE1/WHOLE MIN/MAX) are always
proposed and the two midpoint slots are absent whenever the heuristic λ
interval is empty (CG08506: 136/168 and 64/168 families; 200 absent slots).
This confirms the finite-pair reviewer's reading. All proposed slots fail
the prefilter: CG08506 1,096 hypotheses at 0/2 passing supports and 48 at 1/2;
CGEBA 1,998/16; CGF41 528/0; CG1E0 2,076/48. In all eight analyzed scans
(including CG1E0 and CG4BDD, whose passing starts came from the triplet lane)
the context lane produced no passing start. Sixteen of 184 CG08506 families
were rejected by `CENTRAL_NORMAL_WITNESS_OUTSIDE_DECLARED_GATE`; their recorded
residuals/allowances are available for the reviewer's order-biased
Gram-Schmidt hypothesis, which I have not reproduced and which is not the
dominant failure (the proposed slots, not the rejected families, fail).

### Mechanism V9-A: triplet completion lane (initializer V15)

After the ordinary prefilter, for every scheduled TRIPLET hypothesis with
exactly two PASSED supports and a non-parallel passing pair: enumerate CAD
faces whose plane normal is compatible with the rotated third scan normal
(same allowance as the Wahba witness gate); skip faces with |m_j·u| below a
declared floor (unresolvable along u); order candidates by |λ_j| then face id;
build completion hypotheses (same rotation, t + λ_j u, third correspondence
replaced, lane `TRIPLET_COMPLETION`, parent index recorded); run the same box
screen and finite query on all three supports; passing completions join the
exact start groups and the unchanged refinement schedule. Budgets: max
completions per parent, max completions total, and the existing prefilter
row/call caps, with explicit omission counters. No Gold, no case identity,
no changed gate; CAD guides correspondence only.

## B. Flange policy refusals (43 windows in 3 slots)

### Facts

- Frozen measurement policy: max fit residual δ = 0.05 mm, flange window
  [2.5, 12.5] mm from the nominal contact, search margin 3.0 mm on both sides
  of the nominal flange plane, plane exclusion 0.075 mm, tangency 0.1 mm.
- The weak-normal gate in `_plane` requires s2 > 10·δ·√N (RMS second in-plane
  spread > 0.5 mm) AND s3/s2 < 0.2; then max |residual| ≤ δ.
- 39 windows failed the weak gate, 4 the residual gate. The centered SVD is
  invariant to rigid pose for fixed rows (as the flange reviewer states); the
  pose enters only through crop membership.
- Scan noise is small: per-patch plane RMS over ~106-row noisy scan planes has
  median 0.0071 / 0.0038 / 0.0051 mm and p90 0.0151 / 0.0074 / 0.0091 mm for
  CG1E0 / CG4BDD / CG08506.

### Synthetic source-level tests (frozen `_plane`, frozen policy)

| Fixture (10 × 12 mm strip unless noted) | Outcome | s2/√N | s3/s2 |
| --- | --- | ---: | ---: |
| single skin, no noise | PASS | 2.889 | 0.000 |
| single skin, σ 0.01 mm | PASS (max residual 0.034) | 2.852 | 0.004 |
| single skin, σ 0.02 mm | NOT_PLANAR (max residual > 0.05) | 2.886 | 0.007 |
| two skins 1.5 mm apart, σ 0.02 | WEAK | 2.860 | 0.262 |
| two skins 1.0 mm apart | NOT_PLANAR (weak gate passes at 0.172) | 2.909 | 0.172 |
| two skins 0.8 mm apart | NOT_PLANAR | 2.938 | 0.136 |
| 20% rows from a second skin 1.5 mm away | WEAK | 2.864 | 0.210 |
| 5% rows from a second skin | NOT_PLANAR | 2.942 | 0.111 |
| single skin, crop only 1.5 mm wide | WEAK (narrow support) | 0.429 | 0.043 |
| single skin, crop 3 mm wide | NOT_PLANAR | 0.891 | 0.023 |

H1 (narrow support): on the SYNTHETIC full 10 mm strip s2/√N ≈ 2.9 ≫ 0.5, so
H1 is rejected only for such full-width fixture windows; the V8 records do not
establish the actual crop widths of the 43 refused windows, so H1 remains open
for them (a sliver left by the ROI or role filter would reproduce WEAK). H2 (crop
membership / wrong-context crop): a ≤ 1 mm pose error shifts the 3 mm margin
crop but does not by itself create s3/s2 ≥ 0.2; a wrong-context crop could.
H3 (contamination by a second surface inside the 3 mm margin): a second skin
1.2–1.8 mm away reproduces the observed WEAK outcome (ratio ≈ 0.26), and
partial or closer contamination reproduces NOT_PLANAR. None of these is
established by the V8 records, which carry no per-window spectrum; they are
competing hypotheses to be discriminated by the successor's diagnostics.

### Two-skin evidence in the derived scan-plane tables

Pairs of noisy scan planes with parallel normals (cos > 0.9995), centroids
within 15 mm and plane offset gap 0.3–4 mm:

| Scan | Scan planes | Parallel overlapping pairs (combinatorial) | Patches with ≥ 1 parallel partner | Gap median | Dominant gap bin | CAD small-extent faces |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| CG1E0BBA2933 | 6,686 | 105,805 | 97.5% | 1.62 mm | 1.5–1.8 mm (68,895) | 64 faces at 1.5 mm |
| CG4BDD1371B6 | 6,732 | 126,424 | 99.5% | 1.45 mm | 1.2–1.5 mm (110,271) | 52 faces at 1.5 mm |
| CG08506AE97D | 7,700 | 117,720 | 92.8% | 1.27 mm | 1.2–1.5 mm (81,021) | 48 faces at 1.5 mm, 12 at 1.0 |

The pair counts are combinatorial (one patch can partner many others within
15 mm) and must not be read as a count of two-skin sites; the per-patch
partner coverage column is the meaningful statistic. Nearly every scan patch
has a parallel partner about one sheet thickness away, and the CAD carries
thickness faces of the same size: the scans contain both skins of the sheet at
the whole-part level. Whether both skins enter the
specific flange crops of the 43 refused windows is NOT shown by the V8 records
(no per-window spectrum or selected-row provenance was published); it is the
leading hypothesis, consistent with the kernel's "explicit single-inner-face"
declaration, its 3.0 mm two-sided crop margin and the synthetic reproduction
of the observed reason, but it must be discriminated from narrow windows,
noise and wrong-context crops by the successor's diagnostics before any
recovery is claimed. If two skins are present, the arc crop (radial distance
≤ R + 3.0 mm) admits the outer arc (R + t) as well. The global registration
p95 (0.85 / 0.67 mm inlier-plane p95 on the admitted contexts) is not the
local residual and is not cited here as a cause.

### Mechanism V9-B: discriminating diagnostics plus declared material-side layer selection (measurement kernel V2)

Inside the versioned kernel, after the existing crops and before the
unchanged `_plane`/`_circle` gates. The diagnostics are published for EVERY
window, before and after any selection, so the selection is auditable and the
competing causes can be told apart; the selection rule is geometric (gap and
material side), never "the layer that fits better":

- Sign conventions (explicit). The nominal radial unit vector `radial` points
  from the bend axis outward to the inner flange contact; the material (sheet)
  lies on the +radial side of the inner face, so the outer skin sits at about
  +t along +radial. The frozen `_plane` is called with reference `-radial`,
  so every FITTED flange normal points toward the axis (air side); against
  that fitted normal the outer skin has signed distance about −t and the
  inner arc has signed distance R(1 − cos θ) ≥ 0. Layer splitting therefore
  uses the material-side-positive coordinate (the crop's TLS normal oriented
  toward +radial), NOT the fitted normal; the arc rule uses the fitted
  (axis-pointing) normals. Tests must cover both sweep signs.
- Flange crop: project the crop rows on the material-side-positive normal;
  enumerate EVERY qualifying gap (gap above the declared 0.3 mm floor with
  ≥ 12 rows on both sides; unqualified large gaps are published too, so a
  tiny outermost population stays auditable). Exactly one qualifying split
  whose two groups pass the scan-derived pair compatibility (per-group
  RMS/rank, parallel normals, offset inside the declared gap window,
  projected bounding-interval overlap — a bounding-box proxy, not a material
  footprint) keeps the group with the SMALLER material-side coordinate as
  `TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED` (the outer skin is
  always about +t further along +radial, independent of the pose error). Zero
  splits publish `SINGLE_LAYER_IDENTITY_UNRESOLVED` (inner-only and
  outer-only or missing-inner scans are indistinguishable there), two or more
  splits `AMBIGUOUS_MULTIPLE_LAYERS_IDENTITY_UNRESOLVED`, an incompatible pair
  `TWO_GROUPS_INCOMPATIBLE_IDENTITY_UNRESOLVED`; all three keep every row
  (abstention) and are never labelled inner. Fit quality never selects. No
  present status is inner-face-claim eligible: the two-layer selection is a
  side hypothesis retained as metadata, and every measured window is a generic
  conditional contact line without a nominal inner-radius deviation. Disabling
  the selection is the V1 control and grants no eligibility.
- Arc crop: after the two flange planes are fitted, keep only rows strictly on
  the axis side of BOTH fitted planes: signed distance against each
  axis-pointing fitted normal > plane_exclusion (the frozen rule keeps
  |signed| > plane_exclusion; the new rule additionally drops the material
  side). Inner-arc rows satisfy R(1 − cos θ) > plane_exclusion away from the
  contacts; an outer-arc row survives both tests only when
  (R+t)·cos θ < R − plane_exclusion and (R+t)·cos(φ−θ) < R − plane_exclusion,
  impossible under the idealized nominal model (exact common axis, exact
  tangent flanges, circular skins at R and R + observed gap, exact plane
  selection, no fit error) for sweeps φ < 2·arccos((R − plane_exclusion)/
  (R+t)) (about 143° at R 0.8, t 1.5, exclusion 0.075). The kernel publishes
  this as a nominal heuristic with `certificate=False`, never as an observed
  outer-arc exclusion certificate; actual isolation evidence remains the
  unchanged circle-residual and tangency gates against the selected planes.
- Diagnostics per window, for BOTH flange sides and both weak-gate branches
  regardless of which failure fires first: N, singular values as s_i/√N
  (population RMS), rank ratio, the two weak-gate booleans, reference angle,
  max/RMS residual, axial and ray extents of the selected rows, role-filter
  and layer-selection provenance (counts before/after, layer means, gap), and
  the arc material-side drop counts. Computed without raising; the legacy
  first-failure reason is kept as the primary reason. This addresses the
  flange audit's P2 items (short-span windows cannot pass the 0.5 mm minor
  spread gate; the current single reason conflates two branches and stops at
  the first flange).
- Deferred to a later successor, per the finite-start audit: the cone-feasible
  normal-witness fallback (triplet rowwise feasibility, analytic pair witness)
  and a finite-support-derived axial proposal search; V9 records their absence
  as a limitation rather than adding unreviewed mechanisms.
- All gates (0.05 mm residual, 10·δ spread, 0.2 ratio, tangency, sweep,
  condition) stay exactly as frozen.

## E. The frozen measurement policy cannot measure the cohort's nominal bend geometry under the exact-nominal clean model

All 99 native windows that V8 extracted (CG08506/CG84/CGE4B 10 each, CG1E0/
CG65/CGEBA/CGF41 13 each, CG4BDD 17; CG64 none) declare nominal inner radius
0.78–0.80 mm and sweep 90°. The frozen measurement policy is δ = 0.05 mm
(max fit residual), e = 0.075 mm (plane exclusion), with the feature-level
constraint 0 < δ < e < search margin.

Arithmetic (frozen kernel, no data): the arc rows within the exclusion band of
either fitted flange plane are removed, i.e. rows with r(1 − cos θ) ≤ e, so
the usable arc is φ − 2·arccos(1 − e/r). The free-circle path then refuses the
arc as `ARC_COMPATIBLE_WITH_FLAT` unless the arc's minimum enclosing strip
half-width (its sagitta / 2 for a clean arc) exceeds δ + guard. For r = 0.8,
φ = 90°: with e = 0.075 the usable arc is 40° and the sagitta 0.048 mm, so the
half-width 0.024 < 0.05. Because e > δ = 0.05 is required, the usable arc is
at most 49° and the sagitta at most 0.073 mm for ANY e: under the exact-nominal
clean model (exact r 0.78–0.80 mm, exact 90°, exact tangent flanges, ideal
fitted planes, single skin, no noise) the frozen policy cannot admit the arc.
This is a restricted theorem about that idealized model, NOT a claim that no
physical measurement could ever emerge regardless of pose, radius deviation or
sampling; a physical bend whose actual radius, sweep or flange geometry
departs from nominal could in principle pass the frozen gates. The frozen
kernel confirms the clean-model statement (`arc_policy_feasibility.py`,
spacing 0.3 mm):

| δ (mm) | e = δ+0.005 | e = δ+0.01 | e = δ+0.02 | e = 0.075 |
| ---: | --- | --- | --- | --- |
| 0.050 | flat-compatible | flat-compatible | rank-deficient arc | rank-deficient arc |
| 0.040 | flat-compatible | flat-compatible | flat-compatible | rank-deficient arc |
| 0.035 | flat-compatible | flat-compatible | flat-compatible | rank-deficient arc |
| 0.030 | measured r 0.800 | measured r 0.800 | measured r 0.800 | rank-deficient arc |
| 0.025 | measured r 0.800 | measured r 0.800 | measured r 0.800 | rank-deficient arc |

The kernel's own test matrix scales δ = 0.01·r and e = 0.015·r (r = 5 mm gives
the frozen 0.05 / 0.075); the frozen CONFIG applied those absolute values to
0.8 mm bends, where δ/r = 0.0625. This is a policy–geometry mismatch that
predates V7: on the exact-nominal clean model no V7/V8 window could have
produced a measurement whatever the registration did; for the physical windows
it is the leading explanation, not a proof. It is a derived, CAD-nominal-only
fact (radius and sweep are declared nominal values, allowed inputs); Gold plays
no role.

Consequence for V9 (for independent review): a separately derived measurement
policy for this geometry with the gate FORMS unchanged: δ = 0.03 mm, e = 0.04
mm (flange window, search margin and tangency unchanged). δ = 0.03 is the
largest 0.005-step value that the frozen kernel measures on perfect data with
margin (sagitta 0.086 mm versus 2δ = 0.06 mm). Sensitivity on synthetic
r = 0.8/90° bends at 0.3 mm sampling: noise σ 0.005 mm measures r 0.803 (single
skin) and r 0.789 (two skins with kernel-V2 layer selection); σ ≥ 0.010 mm
refuses at the flange residual gate (max residual over ~1,000 rows > 0.03).
The scan planes' per-patch plane RMS (median 0.004–0.007 mm, p90 0.007–0.015
mm) sits near that edge, so under the derived policy some windows are still
expected to refuse as NOT_PLANAR; that is the honest trade-off, not a tuning
target. δ is not chosen to make any particular case pass, and the choice is
published before any run. Without this change V9 cannot produce a
measurement on this cohort under the clean model, regardless of every other
mechanism. The candidate is one defensible prospective development value, not
a unique or calibrated one: the continuous half-width headroom above δ is only
~0.011–0.013 mm at r 0.78–0.80, and the finite three-column headroom of the
connected fixture is 0.0004 mm, so the outcome is sampling-phase dependent (a
half-step lattice phase leaves two admitted columns and refuses). Coupled
effects of the pair: tighter plane and circle max-residual gates, an eased
minor-spread gate (10·δ: 0.5 → 0.3 mm), eased flat-alternative rejection and
more near-contact arc rows admitted by e 0.04. A separately derived
finite-support policy (`max_angular_gap_deg` 5 → 30) admits the sparser
angular stations that the derived exclusion leaves (Part F).

Indicative proxy only (NOT an expected yield or refusal rate): from the
derived V8 scan-plane tables (per-patch plane RMS over ~106-row noisy scan
planes; the proxy takes a crop's max residual as 3.5 × RMS, which is only
indicative for 1,000–2,500-row flange crops and ignores waviness across a
10 mm window):

| Scan | patches with 3.5·RMS proxy > 0.05 | patches with 3.5·RMS proxy > 0.03 |
| --- | ---: | ---: |
| CG1E0BBA2933 | 12.5% | 37.5% |
| CG4BDD1371B6 | 3.1% | 7.0% |
| CG08506AE97D | 4.4% | 11.4% |
| CGEBA9A869AC | 2.9% | 14.9% |
| CGF41F49066A | 7.9% | 33.9% |
| CG84C8FA12D2 | 8.4% | 26.0% |

So the proxy suggests a substantial minority of windows may refuse at the
flange residual gate on the noisier scans; it predicts no yield, and the
kernel-V2 diagnostics will report the actual per-window residuals either way.

## F. The frozen finite-support angular gap cannot admit the derived-policy arc on the fixture

`cad_guided_instances.SupportPolicy.max_angular_gap_deg` (V8: 5°) bounds the
largest consecutive angular gap between distinct arc stations inside every
axial slab. On the connected synthetic fixture (five nominal angular columns
on r 0.8 mm / 90°, i.e. 22.5° steps; a cross-sectional chord of 0.3 mm would
subtend 21.6°, but generic 3D row spacing does not establish such a chord) the
ideal e 0.04 exclusion keeps three columns 22.5° apart, so the 5° cap refuses
(`DISCONNECTED_LOCAL_ANGULAR_SUPPORT`) and 25–45° admit; V9 declares 30° as a
development margin. This is fixture and nominal-arithmetic evidence only: it
does not prove that every physical carrier fails 5°, it is not phase-robust
(a half-step lattice phase leaves two columns, refused by the distinct-station
minimum at any cap; a missing interior column makes a 45° gap that still
refuses), for fixed fits and rows every support admitted at 5° stays admitted
at 30° (support-local monotonicity, not pipeline or F1 monotonicity), and the
generic sample-supported interpretation, abstentions, 4 mm axial gap, 20°
minimum sweep and 3-row minima are unchanged. The isolated adapter checks are
in `tests/test_cad_guided_instances_angular_gap_v9.py`.

## C. CPU-terminated registration (4 slots)

CG191 and CG4FAF spend > 80% of sampled CPU in surface V4 competitive-candidate
leaf pairing during refinement. V9 makes no change here; these remain retained
negatives. The completion lane adds refinement work only through the
unchanged `max_refined_start_groups` schedule.

## D. Plan and expected outcome classes

Modules (all new versions, frozen predecessors untouched): initializer V15
(V14 + completion lane), measurement kernel V2, context measurement V2,
support-cover measurement V3, contact scoring V3, report envelope V7, mesh
offline V16, projection V4, runner V9 with protocol/design and the derived
measurement policy of Part E and the support policy of Part F declared as
config deltas; focused tests; independent reviews (Codex source-only
reviewers on flange math and finite-start rejections plus wiring/runner
review); exclusive freeze; independent PRE; single execute; POST; result.
Honest expectation: completions may still fail the row-level prefilter; a
passing start must survive refinement and the 1.0 mm admission; layer
selection may expose the next refusal (arc fit, tangency, sweep); no numeric
F1 is promised, and none may be fabricated where the scorer says null.
