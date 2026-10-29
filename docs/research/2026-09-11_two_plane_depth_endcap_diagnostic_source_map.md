# Provisional depth/end-cap diagnostic source map — 2026-09-11

## Status and custody

SOURCE-ONLY PREPARATION; NO ASSERTED FAILURE CAUSE. Root released progress that
L-R shared registration completed in 1462.253261 CPU seconds with its oracle
true, and V2 measurement returned after 1.437055 CPU seconds with an
`AssertionError` in the original oracle and no reported execution/safety error.
L-A was still active when this task was assigned. These are relayed progress
observations, not this reviewer's inspection of terminal evidence. The identity
of the failed assertion, returned quantities, support counts and endpoints are
not known here.

Beyond canonical instructions/current status, only the named source files below
were read/hashed. No attempt, scientific
artifact, archive, raw/control input, project import, test, geometry execution or
replay was accessed. No frozen candidate source or native PRE was changed. The
native PRE remains BLOCKED. This map requests neither an exclusion change nor a
relaxed oracle; terminal custody and explicit release must precede record reads.

## Exact call and oracle chain

`tests/test_cad_guided_two_plane_depth_connected_v2.py:116` selects composite
planes plus scheduler V2 and validates their bindings and the completed context
V2 profile. Its packet loop at line 185 sends one shared registration and one
owned role preparation to both measurement arms. V2 uses the unchanged helper
`_measurement_call`, `_measurement_safety`, and `_measurement_oracle` in
`tests/test_cad_guided_support_cover_connected_v2.py:301`.

The registration oracle at helper line 285 requires completed conditional
contexts, not a true/accepted pose: `transform_scan_to_cad=None`,
`status=CONDITIONAL_MEASUREMENT_CONTEXTS`, nonempty
`evidence.measurement_context_indices`, complete candidate/numerical evaluation,
and unknown joint configuration/continuous pose-family coverage. Its success
does not require correct radius, flange attribution, or fixture endpoints.

For a future released L-R stage record, define `A = record.answer`,
`M = A.measurements`, `S = M.source_context_result`. The constructor's filenames
are `REGISTRATION_RESULT_02_SHARED.json`, `MEASUREMENT_RESULT_02_V1.json`, and
`MEASUREMENT_RESULT_02_V2.json`. These names were derived from source, not opened.

The recorder at helper line 141 retains `execution_error`, `safety_error`,
`oracle_error`, `oracle_passed`, `disposition`, `answer`, call/oracle timings and
binding. `_error` retains only type and up to 1000 message characters, not a
traceback. Many oracle assertions have no explicit message; therefore an empty
`AssertionError.message` alone need not identify an assertion. A failed oracle
does not retain its partially accumulated success return value.

Safety is separate from measurement accuracy. It requires publication validation,
unchanged shared inputs, exact V1/nested-V1 equality, `observed_bend_count=None`,
`production_use_allowed=False`, `conformance=INDETERMINATE`, unknown joint
configuration/continuous pose-family coverage, nested V1's evidence-only role,
and `context_quantifier=EXISTS_COMPLETED_EXISTING_COVER_CONTEXT`.

The positive oracle runs in this order; it does not select a favorable group:

| Order | Exact source test / returned fields | L-R discriminator |
|---|---|---|
| 1 | Helper line 378: retain every `M.support_groups` row whose `status` is `MEASURED_CONDITIONAL`; assert that list is nonempty. | No usable group is distinct from a bad numerical answer. Failure text is `M.reasons` or `NO_PUBLISHED_CANONICAL_MEASUREMENTS`. |
| 2 | Frozen fixture line 316, for **each** usable group's `measurements`: `fit_model=free_circle`, `tangency_check_role=FREE_CIRCLE_MODEL_COMPATIBILITY_CHECK`; then radius error and absolute signed-sweep error. | `abs(measured_radius_mm-5.30)<=.02`; `abs(abs(measured_sweep_deg)-90)<=.1`. These are the noisy L-R oracle limits, not the fit's .05 mm residual gate. |
| 3 | Same frozen oracle: nominal-radius substitution test and physical trim claim. | `abs(measured_radius_mm-5.0)>=.25`; `physical_trim_endpoints is None`. The nominal-angle substitution branch is inactive for L-R because its true and nominal sweeps are both 90 degrees. |
| 4 | Helper lines 380–392, for **each** usable group: raw geometry frame; radius/sweep intervals and datum deviation are `None`; then every association. | Frame is `ORIGINAL_SCAN_RAW_COORDINATES_ASSUMED_MM`; association registration is unaccepted; both canonical deviation equalities have absolute tolerance `1e-12`. |
| 5 | Helper line 394: localized case requires nonempty `M.predictions`. | A correct conditional radius with unavailable finite support can pass earlier quantities and fail `MISSING_REQUIRED_LOCALIZED_RAW_CARRIER`. |
| 6 | Frozen fixture line 334, for **every** prediction: literal contact-sector generator convention, no physical trim, manufacturing identity `UNKNOWN`; then generator and axis endpoint errors. | Both reversal-invariant maximum endpoint distances must be `<=.05 mm`. One bad published carrier fails even if another is accurate. |

The returned association equations are
`canonical_radius_deviation_mm = measured_radius_mm - nominal_radius_mm` and
`canonical_sweep_magnitude_deviation_deg = measured_sweep_magnitude_deg -
abs(nominal_sweep_deg)`. They are consistency checks, not datum/pose validation.
Keep all records in order when locating the first failing predicate. Do not
replace this universal check over published output by an existence/best-fit test.
Steps 2–3 run to completion for one group before the next; only after that whole
list succeeds does step 4 begin. A later group's numerical error can therefore
precede an earlier group's association/semantic check.

Before descending into groups, inspect `A.publication_validator_passed`,
`publication_error`, `input_custody_error`, `nested_v1_exact_pair`, then
`M.status`, `computation_status`, `reasons`, `discarded_measurement_prefix`,
`source_context_result_sha256`, `support_cover_plan` and `attempted_work` if
present. Source refusal, plan refusal and whole fatal-prefix clearing have
different schemas/receipts and must not be reduced to an empty prediction list.

## L-R construction and what inference does not receive

The original geometry is fixed in
`tests/test_cad_guided_connected_v12.py:122,154,231,275`:

- L-R is radius 5.30 mm, sweep 90 degrees, localized and noisy, under proper
  rigid scanner pose B. CAD remains separately constructed nominal radius 5 mm,
  sweep 90 degrees, axial span 0–30 mm and thickness 1 mm.
- The scan profile has 50 stations: first flange distances 0–20 mm, 25 angular
  stations over the sweep, and second flange distances through 29 mm, with
  contact seams welded. Eleven axial stations cover 0–30 mm in 3 mm steps.
- Raw rows 0–549 are the inner sheet. Its z=0 boundary uses IDs `11*i` for
  `i=0..49`. Two further 50-row profiles, IDs 550–599 and 600–649, have radii
  5.80 and 6.30 mm at z=0. Together the three profiles make the connected cap;
  50 of its 150 rows are shared with the sheet. The source fixture asserts
  650 rows and 1176 triangles, not a detached planar datum patch.
- Pose B is the source's quaternion construction about normalized `(-2,3,1)`,
  angle -1.17 radians, translation `(-17,31,47)`. Raw-coordinate perturbation
  depends only on original row number; its Euclidean ceiling is .002 mm, while
  the inference error-bound declaration remains .05 mm. This mismatch is an
  explicit constructed scenario, not a calibrated scanner claim.
- `cap_rows`, true pose, clean/model points, deformation and true endpoints live
  only in `truth`. Inference receives raw positions, raw triangulation, the .05
  bound and independently constructed CAD triangles/face/source IDs. No cap
  truth mask, exact scan normal, radius answer or endpoint answer is passed.

The independent predicate witness at frozen fixture line 481 is especially
useful but is **not** an observed runtime failure: before noise, at the exact
pose, the first nominal flange window admits 65 rows including ten cap-only
rows at least .5 mm off the inner sheet. Cap-only IDs `550+22` and `600+22`
also satisfy the stated angular/radial arc gates and are farther than .075 mm
from both actual flange planes. It proves that position-window tests alone do
not guarantee semantic support purity. It does not prove that a completed
runtime context admits those IDs after role filtering and its own crop.

## Cap-role admission: proposals, conflicts, incidence, then crop

The relevant seam is
`backend/cad_guided_composite_surface_roles_v1.py:300` followed by
`backend/cad_guided_context_surface_roles_v1.py:351` and
`backend/cad_guided_context_measurement_v1.py:257`.

Composite preparation binds both complete lanes and every qualified origin to
immutable raw supports. Published support kinds are `v7:CORE`, `v7:CLOSURE` or
`seeded_v1:SEEDED_REGION`; `record_id` is the composite plane ID. The composite
origin table links that ID back to lane/original kind/original record ID. Both
lanes' qualified but unresolved/noncap supports can oppose cap exclusion.

For each context's raw-frame nominal axis, let C be the union of triangles in
proposals whose entire stipulated unoriented normal cone lies inside the
15-degree axis cone, and U the union of triangles in all other qualified
proposals. The implemented condition is
`acos(abs(normal dot axis)) + normal_angle_bound + 1e-10 <= pi/12`.
Eligible cap triangles are `E = C \ U`, not C alone. A referenced raw vertex is
excluded only when **every** incident raw triangle is in E. Any noncap/unresolved
incident triangle keeps it; unreferenced vertices also remain. Coordinates and
raw IDs are not welded or deleted. A tight cap-contained proposal alone is
therefore insufficient if another qualified proposal conflicts with it.

The axis for this filter is the context-transformed feature axis, canonicalized
only in sign, not the later freshly fitted axis. Filters are cached by that
axis's exact bytes. Different context axes can produce different exclusion
bitmaps without a source mutation. This is conditional classification under a
common-plane/error-ball premise, not proof of a physical cap.

Terminal diagnostic fields in `S`:

| Field path | What it distinguishes |
|---|---|
| `surface_role_source_binding` | Points, triangulation, epsilon and full patch-metadata custody. |
| `surface_role_filters[j].axis`, `axis_cone_radians`, `proposal_dispositions` | The actual context axis and each proposal's kind/ID, role, nominal axis angle, normal-angle bound, support hashes/counts. Resolve composite IDs against the registration's retained origin metadata. |
| Filter `cap_proposed_triangle_count`, `conflicting_triangle_count`, `cap_like_triangle_count`, `unknown_triangle_count` | No cap proposal, opposed proposals, and final eligible cap triangles are different outcomes. Counts alone do not identify affected vertices. |
| Filter `excluded_row_count`, `kept_row_count`, `unreferenced_rows_kept`, raw-row digests and `excluded_raw_row_bitmap` | Actual whole-scan exclusion. The bitmap is base64 packbits, little bit order, zero-padded, with declared `bit_count`; do not infer raw IDs from counts. |
| Filter membership/incidence counters and completion flags | A completed selection versus whole preflight/selection failure. |
| `context_windows[*]`: `context_index`, `surface_role_filter_index`, `pre_role_filter_roi_rows`, `excluded_cap_like_roi_rows`, `local_input_rows`, `status`, `reasons`, `support_record_id` | Whole-scan exclusion is distinct from exclusion inside that context's ROI, and from subsequent fitting refusal. |

The crop at `backend/cad_guided_local_measurement_v1.py:36` is a conservative
superset of nominal finite flange/arc searches, with a floating-arithmetic guard,
not a measured-answer-tightened crop. Measurement uses `crop intersect kept`.
Neither the crop nor the inner kernel extends the nominal span by the declared
.05 mm coordinate-error bound. Boundary row survival must therefore be checked
from actual retained IDs, not assumed from the clean fixture endpoints. This is
a source-level diagnostic distinction, not a proposed change or asserted cause.

## Fixed-role measurement, grouping and quantity signs

`backend/cad_guided_measurement.py:284` first fits both complete flange-window
row sets, then admits arc rows within the guarded nominal angular window and
upper radial bound and outside both fitted plane exclusion bands. It does not
trim cap outliers from the selected roles. Relevant window reasons include
`FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL`, weak/rank-deficient flange support,
orientation/axis/sweep scope failures, `ARC_COMPATIBLE_WITH_FLAT`,
`ARC_NOT_SINGLE_CIRCLE_AT_DECLARED_RESIDUAL`, tangency/contact-sector failures
and insufficient angular/axial span. The complete strings are retained in each
window's `reasons` when `_measure` returns unavailable.

Successful windows produce `S.support_records[*]` with exact `raw_support` lists
(`arc_rows`, two `flange_rows`), fit policy and **all** associations. Association
fields retain `context_index`, `surface_role_filter_index`,
`ordered_flange_raw_rows`, `transform_scan_to_cad` and `per_context_measurements`.
These per-context quantities are not the canonical V2 answer. V1 later clears
overlapping records' canonical measurements/finite support with
`AMBIGUOUS_SHARED_RAW_OR_DUPLICATED_POSITION_SUPPORT`; `ownership_work` identifies
that stage. It does not erase raw roles or original associations. Thus a V1
record with null canonical output is not evidence that its fresh V2 cover fit
was attempted or failed. Ownership clearing can also overwrite that V1 record's
earlier canonical-failure reason; no erased result is reconstructed here.

`backend/cad_guided_exact_support_cover_v1.py:51` constructs maximal components
from all arc-related exact-position claims, without filtering members by their
old fit/finite status. Shared flange-only support does not form an arc ownership
edge. Cross arc/flange position claims veto the whole connected component.
Every eligible cover must be **one existing member** whose raw arc IDs equal
the component arc union and whose total flange IDs equal the component flange
union, with one direct, unique flange permutation for every member. Exact
coordinate equality detects conflicts; it cannot replace a missing raw ID.

Inspect `M.support_cover_plan.status/reasons/work` and every plan group's
`group_id`, `member_record_ids`, `cover_record_id`,
`member_flange_permutations`, `status/reasons`. The discriminating component
reasons are `CROSS_ROLE_POSITION_CONFLICT`, `INCOMPATIBLE_MEMBER_FIT_POLICIES`,
and `NO_EXISTING_WHOLE_COMPONENT_RAW_SUPPORT_COVER`. Do not diagnose all three
as poor circle fitting, split a failed component, or assemble a favorable novel
union. The seven-field fit policy is part of the cover proof.

For a covered group, V2 freshly calls `_canonical_fit` on that full existing
member's fixed raw roles. It uses no serialized radius or CAD center/axis in the
answer. The two fitted flange normals determine an axis whose largest absolute
component is positive; an orthogonal 2-D geometric circle fit determines center
and positive radius. The algebraic circle is only a solver seed; all-point max
residual, conditioning, flat-alternative and tangency checks remain active.
Canonical tangency uses radius-facing flange radials and requires each signed
distance-minus-radius residual within .1 mm. The contact sweep is
`atan2(axis dot (radial0 cross radial1), radial0 dot radial1)` in [-180,180]
degrees, admitted only at magnitude 20–160. Fixed arc rows must lie in that
measured contact sector and cover at least the policy angular extent.

`M.support_groups[*]` preserves member IDs/permutations and associations. An
uncovered group has null raw/fit/quantity/finite fields; a failed fresh cover fit
retains raw support/fit policy and a reason prefixed
`COVER_GEOMETRY_UNAVAILABLE:<type>:<reason>`. Inspect all group statuses/reasons,
`cover_refit_row_visits`, `raw_support`, and these quantity fields:
`measured_radius_mm`, `measured_sweep_deg`, `measured_sweep_magnitude_deg`,
`measured_axis_direction`, `measured_axis_point`, `circle_max_residual_mm`,
`flange_max_residual_mm`, `measured_tangency_residual_mm`,
`free_circle_jacobian_condition`, `free_circle_min_singular_per_sqrt_n`.

There is no signed physical radius: radius is positive, while radius deviation
can have either sign. Sweep sign follows canonical raw axis and raw role order,
not material handedness or manufacturing bend sense. The oracle deliberately
uses its absolute value. Residual limits/conditioning are not radius/sweep
uncertainty intervals; single-inner-face selection remains an unverified context
assumption. No count, thickness, physical trim or accepted joint pose follows.

## Finite support and endpoint localization

`backend/cad_guided_instances.py:68` consumes the same fixed role points and the
fresh measurement. It uses unique positions, rejects repeated/overlapping roles,
checks a unit axis and fitted radial consistency, unwraps the arc around its
largest circular gap, and sets axial stations from **arc rows only**. With
`z_i=(p_i-center) dot axis`, its segment is
`center + [min(z_i),max(z_i)]*axis`. Neither CAD trim nor flange extrema set the
segment. No finite disconnected interval is split into multiple bends.

The declared policy is maximum axial gap 4 mm, maximum local angular gap
5 degrees, minimum local arc sweep 20 degrees and at least three local points
from each of arc/flange0/flange1. The segment is divided into
`ceil(length/4)` equal closed slabs (at least one); boundaries share stations
only within a numerical guard. Each slab must satisfy all three-role counts and
angular requirements. Global axial gaps and local angular gaps test different
sampling limitations. These are engineering sample-support gates, not proof of
continuous material or full fixture/physical coverage.

Useful finite reasons include `NO_FINITE_AXIAL_SUPPORT`,
`DISCONNECTED_AXIAL_SUPPORT`, `INSUFFICIENT_LOCAL_ARC_OR_FLANGE_SUPPORT`,
`INSUFFICIENT_LOCAL_ANGULAR_STATIONS`, `INSUFFICIENT_LOCAL_ANGULAR_COVERAGE`,
`DISCONNECTED_LOCAL_ANGULAR_SUPPORT`, radial inconsistency and numeric/slab
budget refusal. `finite_support.status=UNAVAILABLE` retains its reasons and
`instance=None`; it does **not** invalidate an otherwise measured conditional
group. A finite-refused instance has no completed `local_coverage` diagnostics.

For `SUPPORTED`, canonical fitting preserves those axis endpoints as
`axis_support_line_start/end`, then adds the same offset
`radius * unit(radial0+radial1)` to form `line_start/end`. This is the literal
`MEASURED_CONTACT_SECTOR_MID_ANGLE_GENERATOR`, not a newly inferred inner/outer
material label. `M.predictions` is exactly the collection of supported instances
from measured V2 groups; nested V1 predictions are not added.

Inspect every group's `finite_support.status/reasons` and, if supported,
`instance` fields `line_start/end`, `axis_support_line_start/end`,
`axial_support_length_mm`, `max_axial_sample_gap_mm`, `unique_arc_support_count`,
`unique_flange_support_counts`, and every `local_coverage` slab's `arc_points`,
`flange_points`, `arc_sweep_deg`, `max_angular_gap_deg`.

The L-R source oracle's clean target is the pose-B transform of axis endpoints
`(0,0,0)` and `(0,0,30)`, and generator endpoints
`(5.3*cos(45 degrees),5.3*sin(45 degrees),0 or 30)`. Error is the smaller of
the two endpoint-orderings' maximum Euclidean errors. Both generator and axis
must be within .05 mm. Correct radius plus a supported but shortened/shifted
sample segment is not enough for this fixture oracle. Likewise, correct axis
endpoints need not imply the correct generator offset/phase. These are possible
distinctions to test against returned fields, not observations of L-R.

Publication validation checks the exact cover plan/raw-role binding and exact
prediction projection, typed finite quantity bounds and the carrier's internal
axis/direction/length/constant-perpendicular-radius-offset consistency. It does
not repeat the geometric fit or establish truth-endpoint accuracy. Its tiny
floating guards are not the oracle's .05 mm allowance and not calibrated error
certificates. Thus publication/safety success and an endpoint oracle failure
are logically compatible.

## Smallest terminal diagnostic handoff

After process-group-empty custody and explicit release, first retain the full
stage disposition/errors/bindings and match them to the fixed assertion order
above. Then summarize **all**, not selected, contexts/windows, role filters,
source support records, cover proposals/groups and predictions with their IDs.
Cross-tabulate retained raw IDs with the construction's sheet/welded-boundary/
cap-only ID ranges only as post-inference simulator diagnostics. Read the
actual exclusion bitmap, not just counts. Preserve original member statuses and
per-context answers alongside the fresh-cover result.

This separates (a) no usable context/roles, (b) complete but unavailable windows,
(c) component cover refusal, (d) fresh fixed-role fit refusal, (e) inaccurate
published quantities, (f) correct quantities but finite refusal/no predictions,
and (g) internally valid carriers that miss the fixture's endpoint oracle.
Multiple mechanisms may coexist across the retained denominator. An observed
bad support association would not alone prove it caused the first failed
assertion. No simulation, refit, crop change, proposal deletion or oracle
relaxation is authorized by this source map.

## Named source fingerprints

These hashes identify the source read for this map, not a fresh whole-97 PRE or
artifact attestation. All paths are repository-relative.

| Source | SHA256 |
|---|---|
| `tests/test_cad_guided_connected_v12.py` | `fbbbbc148c564c222013c68f81922e2aed19bdfc1033c7677117a1186582bdfd` |
| `tests/test_cad_guided_support_cover_connected_v2.py` | `c645c7f0ff96e726989eb929176a8b1286f45c1ef996c68b21473ec488707cb5` |
| `tests/test_cad_guided_two_plane_depth_connected_v2.py` | `8d84b443c265c4ebacb85d27654518393c36676884a2c7c3f1347d485c6247ee` |
| `backend/cad_guided_context_surface_roles_v1.py` | `c042408109d260888834ebacc9ad3b62256fda589f1ea8cbba31164b2bea9a62` |
| `backend/cad_guided_composite_surface_roles_v1.py` | `f87f83f9e235059f6d33a92fba4bd329197a61de6a23039212acd8ef90b3b68f` |
| `backend/cad_guided_local_measurement_v1.py` | `5b07e8c14d3eededfb7cbdb2cb628e6d631a30662280fc2d39b2528d862b541b` |
| `backend/cad_guided_measurement.py` | `d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb` |
| `backend/cad_guided_context_measurement_v1.py` | `a756289847125e94baa654b4c82b4c3daf5592b6529a1ec7f1a1072dd79575a9` |
| `backend/cad_guided_exact_support_cover_v1.py` | `70fc42d19f67346b6b5f4951832edef481aa4d94abaeec892b415cfc018cab47` |
| `backend/cad_guided_support_cover_measurement_v2.py` | `8783f95bfce161ae4d9cd0c5a2532114efa510907b50479a42896747e20d56aa` |
| `backend/cad_guided_instances.py` | `d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda` |
