# Independent non-author source review: Gold11 V9 measurement chain (kernel V2 / context V2 / support cover V3)

Date: 2026-09-13. Reviewer: independent Claude (Fable 5.1) source reviewer, non-author.
Repository: /Users/idantal/ShermanQC/FullProject (branch codex/enterprise-reorder, HEAD 5df325e; the six chain files are untracked working-tree files).
Method: read-only source and document inspection; unified diffs V1->V2 / V2->V3; SHA-256 bindings; `.venv/bin/python -c`
snippets that (a) parse test files with `ast` to count cases, (b) exercise pure kernel functions (`_measure`, `_select_flange_layer`,
`_flange_diagnostics`, `_flat_strip_width`, `_fit_policy`) on INVENTED arrays only. No test suite was executed, no geometry job or runner
was started, no data/attempt/cache/credential was opened. No numeric F1 or physical claim is made anywhere below.

## 1. Decision

**P1 present (two).** The chain cannot deliver a measurement to its own consumers as wired:

* **P1-1 (in scope)** `backend/cad_guided_support_cover_measurement_v3.py:179` binds the cover witness with `_v1._fit_policy(feature)` (7 keys) while every record written by context V2 carries `_fit_policy(feature, layer_policy)` (8 keys, `layer_selection` added at `backend/cad_guided_context_measurement_v2.py:59-63,341`). The comparison is always False for a real context-V2 result, so the first COVERED group raises `SUPPORT_COVER_FIT_POLICY_BINDING`, which the `except Exception` at `:201-211` converts into a fatal whole-clear (`discarded_measurement_prefix=True`, nested context result and all window diagnostics discarded). Only scans where no window measures avoid it; i.e. the defect fires exactly when the chain first succeeds.
* **P1-2 (adjacent scope, terminal consumer; unsound compatibility claim in the design record)** the frozen scorer pinned by the V9 runner (`scripts/run_gold11_mesh_development_v9.py:359-363`) rejects the chain's published identities: `scripts/gold11_contact_scoring_v2.py:31,66` require `schema == CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V2` (cover V3 publishes `_V3`, `cad_guided_support_cover_measurement_v3.py:81`), `:390` requires the nested context schema `_V1` (context V2 publishes `_V2`, `:165`), and `:82/:270` and `:95/:341` require `material_side_status == "CONTEXT_ASSUMED_NOT_VERIFIED"` on every quantity and carrier via `_claims` (`:207-211`), a value kernel V2 never emits when the layer policy is enabled (`backend/cad_guided_measurement_v2.py:554-562`; enabled by default in context V2 `:200-201` and by CONFIG `run_gold11_mesh_development_v9.py:243`). The scorer applies `_claims(measurements, _MEASUREMENT_CLAIMS)` on both the complete path (`:375`) and the refused-with-measurement-prefix path (`:545`), so any V9 case that reaches the measurement phase projects as `MALFORMED_RESULT` (`:555-560`, `:679`), never as REFUSED, COMPLETE_NO_LOCALIZED_CARRIERS or SCORED_CONDITIONAL. The design record's rows "consumer binding" / "scoring binding" and "The contact projection, matcher and all scoring thresholds are unchanged" (`docs/research/2026-09-13_gold11_mesh_development_v9_delta_design.md:26-29,44-48`) therefore assert a compatibility that the source contradicts.

Both P1s are invisible to the shipped suites because the seams are mocked (Section 6). Every other required check is either confirmed or a P2/note.

## 2. Bindings (SHA-256, bytes; final state at end of review)

| File | SHA-256 | Bytes |
| --- | --- | ---: |
| backend/cad_guided_measurement.py (frozen V1) | d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb | 25964 |
| backend/cad_guided_measurement_v2.py | 59555408cf92d3bdb7eae711b5ed7edfc68c0d9d518bb683c78e30dd61630d6d | 39282 |
| backend/cad_guided_context_measurement_v1.py (frozen) | a756289847125e94baa654b4c82b4c3daf5592b6529a1ec7f1a1072dd79575a9 | 39387 |
| backend/cad_guided_context_measurement_v2.py | cc10db40e376ed0786f27babcac40b8261633955a8c9926331f8c9d9414e4ce0 | 41188 |
| backend/cad_guided_support_cover_measurement_v2.py (frozen) | 8783f95bfce161ae4d9cd0c5a2532114efa510907b50479a42896747e20d56aa | 27567 |
| backend/cad_guided_support_cover_measurement_v3.py | 3f53d054165468d9197f7e0e0b3c6b564267880906377c1fb95f964457b2b74d | 27937 |
| tests/test_cad_guided_measurement_v2.py | 13e2fbf913313cef2ce78a61e2dc4d3a7f16afd75f405bb46188d9347a66f67b | 10693 |
| tests/test_cad_guided_context_measurement_v2.py | 720c6209654767911aebc254ba1f0ee3253c2f6ff8bccabd7aca9083d71ae2fd | 46930 |
| tests/test_cad_guided_support_cover_measurement_v3.py | 68ee5051fb0d2d394cb5bdb5f0ac4ab634d3ab8c1168d6459029d3122947b809 | 29859 |
| docs/research/2026-09-13_gold11_mesh_development_v9_delta_design.md | 1f4d52492a904703bf84302c8c5d6bc53cfb22157b0a87d9dd6272caeb574522 | 6347 |
| docs/research/2026-09-13_fable51_v9_diagnosis.md (final; see note) | 2d01be35ab89074d3b023d91a86aa81c7eb489bcae3877bbd9f74b1cb0962ece | 18650 |
| docs/research/2026-09-13_gold11_v8_flange_policy_codex_audit.md | 7e5f8ecc2b2e5709a20245175fa7edc89bff296e775119c8385d37d5146037c0 | 21623 |
| docs/research/2026-09-13_fable51_v8_codex_coordination.md (final; see note) | 5ef72697c1eb4011c7a8bc3a2dceb443c34f2364e80756bdfd293c326cb7b988 | 10030 |
| scripts/gold11_contact_scoring_v2.py (adjacent, read for P1-2) | 9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff | 50413 |
| scripts/gold11_schedule_projection_v4.py (adjacent, diff vs V3 only) | 06869ff7bb8761eae976f294e733713c3f2d94a83ee0c8d756c90d50afccfa90 | 16189 |
| backend/cad_guided_mesh_offline_v16.py (adjacent, wiring lines only) | 441c618d4998edaca0d401dd22f14d1444f248750a807386fecef19741708769 | 36874 |
| scripts/run_gold11_mesh_development_v9.py (adjacent, CONFIG/declaration lines only) | f8a9b7f28af81e842f187f6ff76ae87b80b59809561300785064ff63c25027e8 | 176957 |
| backend/cad_guided_instances.py (adjacent, `build_finite_support_instance` only) | d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda | 10296 |
| backend/cad_guided_local_measurement_v1.py (adjacent, `_crop_rows` only) | 5b07e8c14d3eededfb7cbdb2cb628e6d631a30662280fc2d39b2528d862b541b | 12215 |

Two documents changed on disk during the review (author edits, not reverted): the diagnosis was first read at
2dd922f7871d10e9718899ecb6719f57da6cdf72db5f89d8fa16ba807993a7f9 (17869 B) and re-read at the final hash above (delta: the
"Expected NOT_PLANAR share" table, lines 265-281); the coordination note was first read at
566fab207e2bf2b3abd233a460f5f1d5ac3abfcaf6356e44126d8d15a0848c50 (9386 B) and re-read at the final hash (delta: the 03:39 POST
correction section, lines 157-167; the 03:05 section is unchanged). Findings below cite the final texts.

## 3. Findings by priority (file:line, minimal repair)

### P1

**P1-1 Support cover V3 fit-policy binding is unsatisfiable for real context-V2 output.**
`backend/cad_guided_support_cover_measurement_v3.py:179` `_require(_v1._fit_policy(feature) == witness["fit_policy"], "SUPPORT_COVER_FIT_POLICY_BINDING")`.
Context V2 always normalizes `layer_policy` to a `LayerSelectionPolicyV1` (`cad_guided_context_measurement_v2.py:200-203`) and stores
`_fit_policy(feature, layer_policy)` in every record (`:341`) and in the equivalence digest (`:332-335`). Pure-function check on an invented
feature: record keys = 7 fit fields + `layer_selection`; cover check keys = 7 fit fields; equality False. Consequence: for every scan with at
least one COVERED cover group the whole support-cover result is fatally refused (`:189-211`), the nested `source_context_result` (with the new
per-window diagnostics) is dropped from the publishable object, V16 publishes status UNAVAILABLE / phase MEASUREMENT_REFUSAL
(`cad_guided_mesh_offline_v16.py:508-515`). Why the suites pass: the V3 test ledger builds record policies with `_fit_policy(feature)` and no
layer policy (`tests/test_cad_guided_support_cover_measurement_v3.py:142`) and mocks `inspect_generated_context_windows_v2` (`:184-197`);
the engine-integration suite mocks `inspect_support_cover_context_windows_v3` (`tests/test_cad_guided_completion_v1_engine_integration_v1.py:562`).
Minimal repair: validate/normalize `layer_policy` in `inspect_support_cover_context_windows_v3` exactly as context V2 does and compare against
`_v1._fit_policy(feature, layer_policy)` (equivalently bind `witness["fit_policy"]["layer_selection"] == original["layer_policy"]`); add the same
binding to `_validate_publication` (`:293-295`); add one unmocked context-V2 -> cover-V3 seam test on the synthetic bend fixture, and make
`_source_ledger` embed `layer_selection` so the existing 82 cases exercise the real record shape.

**P1-2 Terminal consumer (frozen scorer) rejects the chain's schema and status vocabulary; design record claims compatibility.**
Evidence: `scripts/gold11_contact_scoring_v2.py:31` `MEASUREMENT_SCHEMA = "CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V2"` enforced at `:66,375,545`;
`:390` `original.get("schema") == "CAD_GUIDED_GENERATED_CONTEXT_MEASUREMENTS_V1"`; `:79-83` and `:90-95` hard-code
`material_side_status="CONTEXT_ASSUMED_NOT_VERIFIED"`, enforced by `_claims` (`:207-211`) at `:270` (quantity) and `:341` (every carrier,
compact and full); the scorer test at `tests/test_gold11_contact_scoring_v2.py:298` confirms any other string is a conflict. The chain publishes
`_V3` (`cad_guided_support_cover_measurement_v3.py:81`), `_V2` (`cad_guided_context_measurement_v2.py:165`) and one of
SINGLE_LAYER_IDENTITY_UNRESOLVED / TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED / MIXED_LAYER_EVIDENCE_IDENTITY_UNRESOLVED whenever the
policy is enabled (`cad_guided_measurement_v2.py:554-562`); CONTEXT_ASSUMED_NOT_VERIFIED is emitted only with `enabled=False` (`:559-560`),
which V9 does not use (`run_gold11_mesh_development_v9.py:243,1451`). Projection V4 is V3 text with identity strings only (diff vs V3: SCHEMA,
INITIALIZATION/WORKFLOW_IMPLEMENTATION, function names) and delegates all claim checks to the scorer. Consequence: every V9 case reaching
FULL_RAW_LOCAL_MEASUREMENT (complete or refused with a measurement prefix) is `MALFORMED_RESULT` in the scorer (`:555-560,679`); the honest
refusal classes of V8 (REFUSED / COMPLETE_NO_LOCALIZED_CARRIERS) cannot even be reproduced. Also, if the constants were merely widened, the
scorer's own output would still declare `material_side_status="CONTEXT_ASSUMED_NOT_VERIFIED"` for every carrier (`:753`) and would score a
SINGLE_LAYER_IDENTITY_UNRESOLVED carrier exactly like an inner-verified one (endpoint matching only, `:751-756`), i.e. the silent
inner-treatment the coordination note forbids (`2026-09-13_fable51_v8_codex_coordination.md:152-155`). Minimal repair: a separately versioned
scorer/projection binding that (a) accepts the V3/V2 schemas, (b) declares which `material_side_status` values are admissible for scoring and
carries the per-carrier status into the score ledger instead of the constant at `:753`, (c) has fixtures using the new strings; the design
record's compatibility sentence must be corrected. This is outside the declared file scope but it is the chain's only consumer.

### P2

**P2-1 Diagnostics are not attached to every refusal.** Attached only for `_plane` failures (`cad_guided_measurement_v2.py:489-495`),
INSUFFICIENT_ARC_SUPPORT (`:538-542`) and `_measure_arc` failures (`:543-549`). Not attached: DEGENERATE_DIRECTION (`:497`),
MEASURED_AXIS_OUTSIDE_SEARCH_SCOPE (`:500-501`), MEASURED_SWEEP_OUTSIDE_MODEL_SCOPE (`:504-505`), AMBIGUOUS_FLANGE_WINDOWS (`:470-471`),
and any `LinAlgError`/`FloatingPointError` (not `_Unavailable`) from `_spectrum` in `_select_flange_layer` (`:254`, unguarded, unlike
`_flange_diagnostics` `:181-185`) or from `_plane`; in context V2 those escape the per-window `except _Unavailable` (`:322-325`) into the
whole-clear path (`:393-405`). The design-record claim "attached to the measurement or the refusal" (`design.md:43`) is therefore partial.
Repair: wrap `:496-505` in the same attach-and-reraise; guard `_spectrum` in `_select_flange_layer` and record `identity="SPECTRUM_UNAVAILABLE"`.

**P2-2 Context V2's own publication validator does not validate the new fields.** `_validate_context_measurement_publication_v2`
checks measured claims at `:547-553` and instance claims at `:569-576` without `material_side_status`; `record["material_side_status"]`
(`:339-342`), `window["material_side_status"]` and `window["diagnostics"]` (`:324-327`) are unchecked; membership in MATERIAL_SIDE_STATUSES
(`:45-46`) is enforced only by cover V3 for the group measurement/instance (`cad_guided_support_cover_measurement_v3.py:309,354-355`), never
for the nested records. A context publication claiming e.g. "VERIFIED_INNER" passes. PUBLICATION_FAULTS (`tests/test_cad_guided_context_measurement_v2.py:637-643`)
has no material-side fault. Repair: require membership and record == measurements == instance status consistency in the context validator;
add a fault case.

**P2-3 Derived-policy feasibility is sampling-dependent and the "with margin" claim is overstated.** Diagnosis `:252-254`: "delta = 0.03 is the
largest 0.005-step value that the frozen kernel measures on perfect data with margin (sagitta 0.086 mm versus 2 delta = 0.06 mm)". 0.086 mm is
the continuous sagitta of the 53.6 deg usable arc. With the fixture's 0.3 mm sampling at r 0.8 (about 21.5 deg per station) only three
stations (22.5/45/67.5 deg) survive the 18.19 deg exclusion, the frozen `_flat_strip_width` gives half-width 0.03045 mm, i.e. a pass by
0.45 um over delta 0.03; delta 0.0305 already refuses ARC_COMPATIBLE_WITH_FLAT; shifting the station phase by +5, -5 or +10 deg leaves two
stations and refuses ARC_SUPPORT_RANK_DEFICIENT (checked with the frozen V1 `_measure` on invented arrays). The admitted stations must span
at least 2*acos(1 - 2*delta/r) = 44.66 deg of a 53.61 deg usable window, so guaranteed feasibility needs an angular station spacing below
about 4.5 deg (about 0.063 mm of arc at r 0.8) or a favourable phase; the 0.05 mm fixture passes with half-width 0.0327 (delta 0.035 refuses
there too, consistent with the table's choice). The policy CHOICE (0.03/0.04) is the only 0.005 step consistent with both fixtures, but the
derivation (`diagnosis.md:224-232,250-263`; `run_gold11_mesh_development_v9.py:192-198`) omits the sampling condition, and the kernel's arc
diagnostics publish only counts (`cad_guided_measurement_v2.py:530-537`), so a real refusal cannot be attributed to sampling vs geometry.
Repair: state the station-span condition in MEASUREMENT_POLICY_DECLARATION; publish the kept arc's angular span and distinct-station count in
`arc_material_side_filter` before `_circle` runs; correct the "with margin" wording.

**P2-4 `enabled=False` does not reproduce V1 for the arc crop.** The one-sided arc filter (`:518-524`) is unconditional, while the docstring
(`:54`) and the design intent say the disabled policy reproduces V1 crops. Flange crops are V1-identical when disabled (`:251-253`), the arc
crop is not on any geometry with rows on the material side of a fitted plane inside the arc window (two skins with t <= 3 mm, adjacent
features). `test_disabled_policy_reproduces_v1_exactly_including_material_status` (`tests/test_cad_guided_measurement_v2.py:96-103`) uses a
single skin, where `material_side_dropped_rows` is 0 by construction, so it cannot discriminate. The unwired public entry
`inspect_cad_guided_scan` (`:641-742`) has no `layer_policy` parameter (`:690`) and thus silently combines V1 flange crops with the V2 arc
filter and V2 result keys; no V9 consumer calls it (V9 uses V1's entry in `backend/bend_inspection_pipeline.py:1783-1790` and
`backend/cad_guided_offline.py:48-51`). Repair: reword the docstring/record ("disabled reproduces V1 flange crops; the axis-side arc rule
is always applied") and add a two-skin disabled-policy test asserting the arc-filter difference; either wire `layer_policy` into
`inspect_cad_guided_scan` or mark it not-for-V9.

**P2-5 Dead policy field and binary-only layer report.** `max_layers_reported` (`:58,68`) is validated, digested into every record's
`fit_policy.layer_selection` and CONFIG (`run_gold11_mesh_development_v9.py:243`) but never used; `layers_detected` is 1 or 2 only
(`:265,270`). Probe (invented strips): three layers at 0/0.5/2.0 mm keep the two nearer layers (2788 of 4182 rows), report
`layers_detected=2`, and refuse NOT_PLANAR (max residual 0.25); two skins 0.25 mm apart report `layers_detected=1` and refuse NOT_PLANAR
(0.125). Refusals are recorded, but the layer count is a misstatement. Repair: drop the field or implement multi-gap reporting up to it.

**P2-6 Deviation rows carry no identity status.** `canonical_radius_deviation_mm` is computed against the nominal INNER radius for every
association regardless of status (`cad_guided_context_measurement_v2.py:372-374`; cover V3 `:120-122`), and associations carry no
`material_side_status`. An outer-only window (kernel test `:142-147` shows r 2.3 for R 0.8, t 1.5) publishes a deviation of about +t with
the status only on the record/instance. A consumer joining deviations without the record status would misread. Repair: copy the status
into association/deviation rows or null the deviation unless the status is the two-layer inner selection.

**P2-7 Declared noise sensitivity is untested.** `noise_sensitivity="SYNTHETIC_SIGMA_0P005_MEASURES_SIGMA_0P010_REFUSES..."`
(`run_gold11_mesh_development_v9.py:197`; diagnosis `:254-257`) has no fixture; kernel V2 tests use sigma 0.004 only (`:80,97,108,143,151`).
The arithmetic is plausible (expected max |noise| over N about sigma*sqrt(2 ln N): 0.019 / 0.028 / 0.037 mm for sigma 0.005 / 0.0075 / 0.010
at N 1000) but should be a test, not a declaration.

### Notes (wording / accounting)

* N-1 `ray_extent_mm` (`:210`) is the extent along the crop's second in-plane principal direction (`vh[1]`), not the nominal ray; rename or project on the declared ray. `principal_extent_mm` is undocumented in the diagnosis list (`diagnosis.md:199-204`).
* N-2 `_canonical_fit(..., material_side_status="CONTEXT_ASSUMED_NOT_VERIFIED")` default (`context_v2.py:80-81`) silently labels layer-selected rows as "assumed" if a caller omits it; both callers pass it (`:369`, cover `:186`); make it a required argument.
* N-3 `flange_support_counts` (`:617`) now counts post-selection rows under the V1 key name (V1 `:383` counted crop rows); crop counts moved to `diagnostics.flanges[].crop_rows`. Document the semantic change.
* N-4 `_finite_list` (`:159-160`) is dead code.
* N-5 Diagnosis Part B `:213-214` ("All gates (0.05 mm residual ...) stay exactly as frozen") conflicts in wording with Part E (delta 0.03); say "gate forms".
* N-6 Kernel row `"surface": "inner"` (`:684`) and scope string (`:712`) remain on the unwired public entry; a consumer of that entry reading `surface` would treat a SINGLE_LAYER_IDENTITY_UNRESOLVED result as inner. Not on the V9 path.
* N-7 `outer_arc_exclusion_status` is judged against the NOMINAL sweep and radius (`:528-537`); the name says so; the measured sweep is available two lines later and could be used as well.
* N-8 Design record test counts (18 / 168 / 82, `design.md:50-57`) agree with an `ast` parametrization count; this says nothing about what the fixtures discriminate (Section 6).

## 4. Required checks, with results

### 4.1 Kernel V2 vs V1 gates (check 1)
Unified diff hunks touch only: module docstring; imports (`asdict`); three identity constants and `LayerSelectionPolicyV1`; new helpers
inserted before `_plane`; `_measure` signature and body; the return dict. `_plane` (V1 `:116-134` = V2 `:277-295`), `circle_residual_jacobian`,
`_flat_strip_width` (flat witness, V2 `:309-334`), `_circle` (V2 `:337-375`, incl. rank/scale `:344`, flat witness `:348-350`, condition
`:369-370`, residual `:372-373`), `tangent_residual_jacobian`, `_tangent_circle` (`:389-442`) and the `CADGuidedFeature.__post_init__`
constraints (`:99-145`) are byte-identical; the in-`_measure` gates (AMBIGUOUS_FLANGE_WINDOWS `:470`, axis `:500`, sweep `:504`, arc guard /
radial window `:510-511`, tangency `:583`, contact sector `:586`, coverage `:589`, axial stations/span `:593-598`) are textually identical
(the tail moved into `_measure_arc`). Confirmed: every gate unchanged in form and threshold.
Behavioural differences (exhaustive): (i) optional layer selection of the flange crops before `_plane` (`:477-488`); (ii) one-sided arc
exclusion (`:524`, keeps `signed > e` instead of `|signed| > e`), unconditional; (iii) diagnostics SVDs run on the crops before `_plane`
(no gate effect; new exception surface, P2-1); (iv) `_measure(points, feature, layer_policy=None)` with a strict type check (`:445-449`);
(v) result gains `material_side_status`, three identity strings and `diagnostics` (`:628-632`); (vi) `flange_support_counts`,
`support_row_indices` and `_instance_support.flange_row_indices` are post-selection rows (`:617,633-637`); (vii) `_Unavailable` may carry
`.diagnostics`; (viii) `inspect_cad_guided_scan` runs the V2 `_measure` with the policy disabled and cannot enable it (P2-4).

### 4.2 Sign conventions (check 2) - independent derivation
Let a0 = axis, n0 = start radial (unit, orthogonal), k0 = a0 x n0, phi the signed sweep, n1 = cos(phi) n0 + sin(phi) k0 (`:450-455`).
`radial` in {n0, n1} is the unit vector from the axis to the inner contact C_i = R radial_i (`:465`, crop centred on `from_contact`).
With `surface="inner"` enforced (`:103-104`), the sheet occupies radial coordinate [R, R+t] along +radial in the flange region, so the
outer skin of flange i is the plane radial_i . p = R + t: material-side positive, as declared (`:9-14`).
`_plane(points[mask], -radial, feature)` (`:492`) orients the fitted normal m_i toward -radial (`:288-290`), i.e. toward the axis, and
the orientation gate (`:291-292`) admits at most 15 deg deviation from -radial.
(a) Outer-skin coordinate. Against the fitted (axis-pointing) normal with centre c on the kept plane (radial . c = R):
s = m . (p - c) = R - radial . p, so an outer-skin point has s = -t (negative), an inner-arc point at angle theta has
s = R - R cos(theta) = R(1 - cos theta) >= 0. Against the crop TLS normal oriented to +radial (`:259-261`) the coordinate is
q = +radial . (p - mean) so the outer skin has q = +t relative to the inner skin; the SMALLER q layer is the inner skin (`:267-274`).
The two conventions are used consistently: selection on +radial (keep smaller), arc filtering on the fitted axis-pointing normals.
Because the outer skin is at +t along the declared +radial irrespective of the pose error, the rule is pose-independent as long as the
15 deg orientation allowance holds (a 14 deg tilted two-skin strip still splits at 1.45 mm in my probe).
(b) Arc-row inequality. Row kept iff s_0 > e and s_1 > e, i.e. R - n0 . p > e and R - n1 . p > e. Inner arc (radius R):
R(1 - cos theta) > e and R(1 - cos(|phi| - theta)) > e, i.e. theta in (acos(1 - e/R), |phi| - acos(1 - e/R)); the usable inner arc is
|phi| - 2 acos(1 - e/R) (matches diagnosis `:224-226`). Outer arc (radius R+t, concentric): (R+t) cos theta < R - e and
(R+t) cos(|phi| - theta) < R - e, i.e. theta in (alpha, |phi| - alpha) with alpha = acos((R - e)/(R + t)); non-empty iff
|phi| > 2 alpha = 2 acos((R - e)/(R + t)) - exactly the diagnosis bound (`:193-196`) and the kernel's published bound (`:528-529`;
141.41 deg for R 0.8, t 1.5, e 0.04; 143.25 deg for e 0.075; 92.6 deg for t 0.3). For phi < 0 the construction mirrors (n1, rays and the
signed `angle` flip together, `:454-456,507`); the inequalities are sign-free because both fitted normals are oriented to their own
-radial. Verified on invented two-skin bends at sweeps +/-90 and +/-120: identical selection (1353/2706 rows kept per flange, 369 outer-arc
rows dropped, 123 kept, r 0.8000 recovered, status TWO_LAYER_INNER_SELECTED...). So: V2 selects the inner skin and the inner arc, and
outer-arc rows can survive only for sweeps above 2 acos((R - e)/(R + t)) (flagged INCOMPLETE_FOR_NOMINAL_SWEEP), or when the fitted planes
are not the inner planes (single outer-only layer: then the outer arc is the coherent arc, R+t is measured and the status is unresolved).
Caveat: the bound assumes a concentric outer skin (uniform t) and inner-plane fits; a thinned or non-concentric outer surface at the bend
reduces the effective t and the bound accordingly; the kernel conservatively uses the smaller observed flange gap (`:525-527`).

### 4.3 Single-layer identity (check 3)
`_split_layers` None -> `identity="SINGLE_LAYER_IDENTITY_UNRESOLVED"`, all rows kept (`:263-266`); `_measure_arc` maps any non-two-layer,
non-disabled combination to SINGLE_LAYER_IDENTITY_UNRESOLVED and one-of-two to MIXED_LAYER_EVIDENCE_IDENTITY_UNRESOLVED (`:554-562`);
no code path labels a single layer inner. Propagation: window (`context_v2.py:327`), record (`:342`), equivalence check (`:344-347`),
canonical measurement (`:140`) and instance (`:159`) via `:367-369`; cover group via `:184-186`; validators: cover V3 measurement `:309`
and instance `:354-355` (membership + consistency); context V2 validator none (P2-2). Outer-only scan: probed and asserted by the kernel
test (`:142-147`): single layer, r about R+t, SINGLE_LAYER_IDENTITY_UNRESOLVED. Could a consumer silently treat it as inner? Inside the
chain only through deviation rows (P2-6) and the unwired entry's `surface` field (N-6); at the terminal consumer, today it is rejected
outright (P1-2) and, if the constants were simply widened, it would be scored as if inner with a false constant status (P1-2).

### 4.4 Layer split rule (check 4)
Purely geometric: sort the +radial TLS coordinate, take the single largest consecutive gap, require gap > 0.3 mm and >= 12 rows on each
side, keep the smaller side (`:225-239,267-274`); no residual, spread or fit quality enters (the coordination constraint at
`coordination.md:101-105` is honoured). Failure modes (invented-array probes unless noted): tilt up to the 15 deg allowance - the TLS
normal of two parallel layers is the layer normal, gap scales by cos(tilt) (1.45 mm at 14 deg), split OK; narrow crops (minor in-plane RMS
below t/2) can make the TLS "normal" an in-plane direction, then no gap is found, identity SINGLE_LAYER..., `_plane` refuses WEAK with the
spectrum published - recorded, not mis-selected; noisy gaps - sigma 0.01 and 0.1 still split at about 1.5 mm; a spurious in-layer split
needs a > 0.3 mm step between consecutive normal coordinates, i.e. a physically separate surface, not roughness; three or more layers -
binary split at the largest gap, remainder may contain two layers -> NOT_PLANAR/WEAK refusal, `layers_detected=2` misreports (P2-5);
second skin beyond the 3 mm margin - never in the crop (`_crop_rows` / `:469`: |radial . from_contact| <= margin) nor in the arc window
(radial distance <= R + margin, `:511`), so SINGLE_LAYER..., correct; thin sheet t <= 0.3 mm - no split, both skins fitted together,
NOT_PLANAR for t > 2 delta (0.25 mm -> max residual 0.125), `layers_detected=1` misreports while `normal_extent_mm` about t reveals it
(P2-5); partial inner (30 rows) + full outer - the 30 inner rows are kept, then RANK_DEFICIENT/WEAK refusal; >= 12 stray rows NEARER the
axis than the true flange (burr, lip, debris, the adjacent bend's inner face) - the rule keeps the stray cluster and drops the flange,
labels it TWO_LAYER_INNER_SELECTED..._UNVERIFIED and relies on `_plane` (12-row spread), tangency (0.1 mm) and sweep gates to refuse; a
planar parallel lip >= 0.3 mm nearer the axis cannot pass tangency, so mis-selection is guarded but the primary reason would be misleading
(the diagnostics `kept_rows=12`, `dropped_rows=N` expose it). All of these end in recorded refusals or unresolved statuses; none silently
yields an inner-labelled measurement.

### 4.5 Diagnostics (check 5)
Both flanges, both branches: `spread_gate_passed` / `thickness_ratio_gate_passed` plus `spread_gate_minor_rms_mm`, threshold,
`thickness_ratio`, `rank_ratio` (`:191-197`); residual gate with max/RMS (`:204-206`); N (`rows`), `singular_over_sqrt_n` (population RMS,
`:163-168,186`), reference angle (`:201-203`), axial / principal / minor-in-plane / normal extents (`:207-211`); status ordering mirrors
`_plane` (`:212-221`); computed before and after selection (`:480,484`) without raising (guards `:178-185,188-190,193`); layer provenance
record (`:242-274`); arc drop counts and completeness bound (`:516-537`). Attachment to refusals: partial (P2-1). Primary reason: the
first `_plane` failure in flange order (`:490-495`), unchanged; `after_selection.first_failure` equals it by construction. Provenance of
dropped rows is count-only (kept rows are the record's raw support) - acceptable, but the diagnosis promised "selected-row provenance".

### 4.6 Derived measurement policy (check 6)
Arithmetic verified: usable arc = phi - 2 acos(1 - e/r); r 0.8, phi 90: e 0.075 -> 39.98 deg, sagitta 0.0482, half 0.0241 < 0.05;
e -> 0.05+ (the least e admitted by `0 < delta < e` at `:129`) -> 49.27 deg, sagitta 0.0728, half 0.0364 < 0.05: infeasible for every
admissible e at delta 0.05, as claimed; delta 0.03 / e 0.04 -> 53.61 deg, continuous sagitta 0.0860, half 0.0430 > 0.03; delta 0.035
/ e 0.04 -> 0.0430 > 0.035 continuous, but the sampled fixtures refuse (Section P2-3), consistent with the diagnosis table (`:235-241`)
being a fixture result, not an analytic one. e > delta is required by the feature validation (`:129`), so e cannot be lowered below delta
to widen the arc. Legitimacy: inputs are the declared nominal r and phi (allowed), the frozen witness inequality and the frozen feature
constraint; uniform CONFIG (`runner:238-243`), declared before any run (`:170-177,192-198`); no Gold and no per-case choice were found.
Character: NOT a gate relaxation - delta is tightened 0.05 -> 0.03 (flange and arc residual gates tighten; the 10 delta spread gate loosens
from 0.5 to 0.3 mm minor RMS, which is the audit's known coupling, `flange_audit.md:182-185`), e is tightened 0.075 -> 0.04 (band shrinks;
still > delta); the forms are unchanged. Noise sensitivity implied: the flange residual gate now refuses when max |noise| over ~10^3 rows
exceeds 0.03 mm (sigma about 0.0075-0.010 mm Gaussian), consistent with the diagnosis (`:254-263`) and the runner declaration (`:197`), and
NOT_PLANAR refusals on the noisier scans are expected and are honest negatives; noise does not threaten the flat witness (it widens the strip),
but arc sampling density/phase does (P2-3) and is undeclared.

### 4.7 Consumers (check 7)
Context V2 vs V1 (all hunks): module docstring; import from kernel V2 incl. `LayerSelectionPolicyV1`; `KERNEL_IMPLEMENTATION`,
`MATERIAL_SIDE_STATUSES`; `_fit_policy(feature, layer_policy=None)`; `_canonical_fit(..., material_side_status=...)` threading the status
into measurement and instance; `_base` schema V2 + `kernel_implementation`; entry renamed `..._v2` with `layer_policy` (default enabled,
validated, re-instantiated, published as `result["layer_policy"]`); `_measure(points[selected], raw_feature, layer_policy)`; window
`diagnostics`/`material_side_status`; digest and record fit_policy include `layer_selection`; CONTRADICTORY_SUPPORT_EQUIVALENCE_DIGEST now
also requires equal `material_side_status`; canonical binding uses `_fit_policy(feature, layer_policy)`; validator renamed, requires schema V2
and `kernel_implementation`. Nothing else changed (confirmed by the 11,988-byte unified diff). Cover V3 vs V2 (all hunks): docstring; `_v1`
rebound to context V2; schema V3; entry renamed with `layer_policy` passed through to context V2 and its V2 validator; `_canonical_fit`
called with `witness["material_side_status"]`; validator renamed; quantity and carrier claims accept exactly `_v1.MATERIAL_SIDE_STATUSES`
and require instance == measurement status. Nothing else changed - which is precisely the defect: `:179` was not updated (P1-1), and the
nested-record validation was not extended (P2-2).

## 5. Sign-convention and policy derivations - summary table (nominal geometry only)

| Quantity | Expression | Value(s) |
| --- | --- | --- |
| Outer skin vs fitted axis-pointing normal | s = R - radial . p | -t |
| Outer skin vs crop +radial TLS coordinate | q = radial . (p - mean) | +t relative to inner |
| Inner arc vs fitted normal | R(1 - cos theta) | >= 0; > e for theta > acos(1 - e/R) |
| Usable inner arc | phi - 2 acos(1 - e/R) | 39.98 deg (e .075), 49.27 (e .05+), 53.61 (e .04) |
| Continuous sagitta / 2 | R(1 - cos(usable/2)) / 2 | 0.0241, 0.0364, 0.0430 mm |
| Sampled half-width, 0.3 mm fixture, e .04 | frozen `_flat_strip_width`/2 on 3 stations | 0.03045 mm (pass over delta .03 by 0.45 um) |
| Station span needed for half-width > delta .03 | 2 acos(1 - 2 delta / R) | 44.66 deg of 53.61 |
| Outer-arc survival bound | 2 acos((R - e)/(R + t)) | 141.41 deg (t 1.5, e .04); 92.6 deg (t 0.3) |

## 6. Test gaps (check 8)

Discriminated by the shipped fixtures: single skin (V1-identical outputs, both sweep signs, 90/120), two skins with inner recovery (90,
-90, 120), outer-only -> unresolved, narrow crop -> spread branch not thickness branch, disabled policy on a single skin, frozen-policy
infeasibility and derived-policy success on sparse and dense samples, invalid policies. Gaps:
1. No unmocked context-V2 -> cover-V3 seam test; the V3 ledger fixture omits `layer_selection` (`test_..._v3.py:142`) and mocks discovery
(`:197`); the "policy binding" test (`:415-421`) passes for the wrong reason (both sides lack the key). Hides P1-1.
2. No projection/scorer fixture with the V3/V2 schemas or the new status strings (`tests/test_gold11_contact_scoring_v2.py:71,97,114,117`
use the old ones). Hides P1-2.
3. Context V2: no material-side or diagnostics fault in PUBLICATION_FAULTS; the mocked `_measure` (`:197-206`) always returns
SINGLE_LAYER_IDENTITY_UNRESOLVED, so the MIXED and TWO_LAYER statuses never traverse records/validators; the canonical-fit test asserts
CONTEXT_ASSUMED_NOT_VERIFIED only through the default argument (`:303`).
4. Kernel V2: no MIXED case (one flange two-layer, the other single); no three-layer, thin-sheet (t < 0.3), partial-inner, near-axis
stray-cluster, tilted-crop or beyond-margin second-skin case; no assertion that axis/sweep refusals carry diagnostics (they do not);
no two-skin disabled-policy case for the arc-filter difference; no `inspect_cad_guided_scan` V2 test; two-skin case lacks -120.
5. Derived policy: no station-phase / sampling-density case (a +5 deg phase at 0.3 mm refuses); no sigma 0.005 / 0.010 noise cases
backing the runner's `noise_sensitivity` declaration; no delta 0.035 case backing "largest 0.005-step".
6. Cover V3: no test that the nested context result of a MEASURED run retains its window diagnostics after cover publication (it would
fail today, P1-1).

## 7. What was not checked

No test was executed and no reported pass count was reproduced; no raw scan, CAD, Gold, attempt directory, cache or credential was opened;
the runner, engine and geometry jobs were not run. Not reviewed: initializer V15 / completion lane, V16 beyond the wiring lines quoted,
projection V4 beyond its diff from V3, the scorer beyond the claim constants and disposition flow quoted, `cad_guided_exact_support_cover_v1`,
`cad_guided_instances._build` internals beyond its two measurement keys, numerical behaviour of the least-squares solvers, and the physical
validity of the two-skin hypothesis or of any registration output. The line numbers cited refer to the hashed files in Section 2; the two
documents that changed during the review are cited at their final hashes.

## 8. Post-binding drift addendum (snapshot 2026-09-13 06:56:21 local)

While this review was being written the author rewrote five of the nine chain files (mtimes 06:51-06:55, after the 06:48 bindings in
Section 2). Sections 1-7 remain bound to the Section 2 hashes and are not rewritten. This addendum records the drift and re-checks each
finding against the drifted text by TARGETED reads only; the drifted files were not fully re-reviewed and need a fresh review once they stop moving.

| File | Drifted SHA-256 | Bytes | mtime |
| --- | --- | ---: | --- |
| backend/cad_guided_measurement_v2.py | 9a2e7da3a58838c5de74085b197547550bcbcd1b53804754b609bba5302c9baa | 48864 | 06:51:48 |
| backend/cad_guided_context_measurement_v2.py | b04affdb627bcf68586c9017337febc2cef76e18c6a7a86564c7b5a9e66c7edc | 44008 | 06:54:39 |
| backend/cad_guided_support_cover_measurement_v3.py | a298b123218d931911e064a2f35ab52e5553f2c1ad5b25a71e3c149c2d7d9234 | 29308 | 06:55:13 |
| tests/test_cad_guided_measurement_v2.py | 9e814c7eeb518757512b1984aa79b8a51032dc6e9552daea2cb3411f221bac27 | 14782 | 06:54:14 |
| tests/test_cad_guided_context_measurement_v2.py | 4e5dfbd5688179263b12f8baace9ae9720d02c8754b5ce78ecd44b12337676b1 | 47014 | 06:54:39 |
| tests/test_cad_guided_support_cover_measurement_v3.py | 2f2880006a401aae855e42b8224634afec19598c562af11609232a5b78dd7843 | 30482 | 06:55:13 |

Unchanged at the snapshot: frozen V1/V2 predecessors, design record, diagnosis, both reviewer documents, scorer V2, projection V4,
mesh offline V16, runner V9 (hashes as in Section 2).

What changed (targeted reads): kernel V2 is now 897 lines with a "qualifying gap partition + scan-derived pair compatibility" rule
(`_partition_layers` `:247-265`, `_layer_pair_compatibility` `:284-333`, `_select_flange_layer` `:336-388`), two more statuses
(AMBIGUOUS_MULTIPLE_LAYERS_IDENTITY_UNRESOLVED, TWO_GROUPS_INCOMPATIBLE_IDENTITY_UNRESOLVED, `:52-54`), connected eligibility
(`INNER_FACE_ELIGIBLE_STATUSES` `:50-51`, `inner_face_claim_eligible`/`carrier_role` `:704-705,774-775`), `radius_deviation_mm=None` plus
`radius_identity` for non-eligible identities (`:751-753`), a policy-gated arc filter (`:642-647`), diagnostics attached at the axis/sweep
refusals (`:608,622`), `layer_policy` on the public entry (`:788-789`), four new policy thresholds (`max_layer_normal_angle_deg` 3,
`max_layer_gap_mm` 6, `min_footprint_overlap` 0.5, `max_layer_rms_mm` 0.1, `:66-69`) and `ray_extent_mm` renamed. Context V2 now keeps the
seven-field `_fit_policy` (`:73-75`), binds the layer policy as a separate record/digest key (`:354,363,367,383-384`) and validates identity
fields on records, windows, measurements, associations and instances (`:51-60,541-543,561-564,586,609`); the canonical measurement and
instance now carry `inner_face_claim_eligible` and `carrier_role` (`:159,177`). Cover V3 binds `witness["layer_policy"] == original["layer_policy"]`
(`:187-188`), carries the identity triple into groups and validators (`:182-198,301-330,374-376`) and nulls inner deviations for
non-eligible identities (`:121-125`). Kernel tests add disabled-policy-on-two-skins (`:113-120`), three-layer ambiguity and tilted-band
incompatibility (`:179-198`), and the new policy validations (`:245-249`).

Status of the findings against the drifted text:

* P1-1: RESOLVED in the drifted combination (provisional). Both sides of `cad_guided_support_cover_measurement_v3.py:187` are again the
  seven-field policy and the layer policy is bound separately at `:188`; the cover tests' ledger now embeds `layer_policy` and a two-layer
  status (`tests/..._v3.py:27-28,158-159`). Still missing: an unmocked context-V2 -> cover-V3 seam test (the discovery mock remains).
* P1-2: STANDS and is STRENGTHENED. Scorer V2 is unchanged (`:31,66,82,95,207-211,270,341,375,390,545,753`). In addition to the schema and
  status conflicts, the canonical measurement and carrier now carry two extra keys, so the scorer's exact key-set checks
  `set(row) == _QUANTITY_KEYS` (`:271`) and `set(row) == _FULL_LINE_KEYS` (`:340`) also fail (INVALID_CONTACT_QUANTITY_FIELDS /
  INVALID_CONTACT_LINE_SCHEMA). Runner SCORING_SOURCES still pins scorer V2 (`run_gold11_mesh_development_v9.py:359-363`); the design record's
  compatibility sentence is unchanged (`design.md:26-29,44-48`). Every V9 case reaching the measurement phase still projects as
  MALFORMED_RESULT.
* P2-1: largely addressed (`:608,622,674,681`); `_spectrum` in `_select_flange_layer` (`:355`) remains unguarded against `LinAlgError`.
* P2-2: addressed in form (`_identity_fields` raises INVALID_MATERIAL_SIDE_STATUS for any string outside MATERIAL_SIDE_STATUSES; record,
  window, measurement, association and instance are bound). Not re-verified by test reading beyond the file hash.
* P2-3: OPEN. No station-span/count diagnostic was added (`grep station|angular_span|kept_arc` finds only the existing
  `sampled_arc_span_deg` of successful fits); the diagnosis text (`:252-254`) and the runner declaration (`:192-198`) are unchanged.
* P2-4: addressed (`:642-647,788-789`; test `:113-120`); the docstring at `:61` still says "reproduces V1 crops" and is now accurate.
* P2-5: addressed in form (`qualifying_split_count`, `qualifying_gaps_mm[:max_layers_reported]`, `layers_detected = splits + 1`, `:365-373`).
* P2-6: addressed in form (kernel `:751-753`; context `:397-400,562-564`; cover `:121-125`).
* P2-7: OPEN (kernel tests still use sigma 0.004 only; no sigma 0.005/0.010 case; no delta 0.035 case).
* N-1: addressed (`second_principal_extent_mm`, `flange_ray_extent_mm`, test `:221`). N-4: `_finite_list` still dead (`:175`).

New observations on the drifted text (not a full review):

* NEW P2-8 Undeclared thresholds. The drifted kernel applies four new selection thresholds by default (3 deg normal parallelism, 0.3-6 mm
  offset window, 0.5 footprint overlap, 0.1 mm per-group RMS), but the frozen runner CONFIG still declares only the four original fields
  (`run_gold11_mesh_development_v9.py:243`), `tests/test_gold11_mesh_development_v9_source.py:103` asserts exactly those four keys, and neither
  CONFIG_DELTA_FROM_V8 (`:170-190`) nor the design record names the new values. `LayerSelectionPolicyV1(**CONFIG["layer_policy"])` (`:1451`)
  would silently take the defaults. Either the runner declaration must enumerate all eight fields or the kernel must refuse an incomplete
  policy declaration.
* NEW note on the "never fit-quality based" constraint (`coordination.md:101-105`). `_layer_pair_compatibility` (`:284-333`) admits a split only
  if BOTH groups are individually planar (rank ratio >= min_plane_spread_ratio, RMS <= 0.1 mm), parallel within 3 deg, offset within
  [0.3, 6] mm and overlapping in footprint; on failure all rows are kept and the identity is TWO_GROUPS_INCOMPATIBLE. This is a symmetric
  admissibility test that can only BLOCK trimming, never trigger it because a fit improves, so it is consistent with the constraint; but it
  is residual-based (0.1 mm RMS, larger than delta 0.03) and must be declared as such in the design record. Footprint overlap is measured in
  the lower group's principal frame relative to the smaller extent (`:276-281,323-330`), so a small partial inner patch inside a full outer
  footprint scores overlap 1.0 - acceptable, since the subsequent `_plane` spread gate governs.
* `INNER_FACE_ELIGIBLE_STATUSES` includes CONTEXT_ASSUMED_NOT_VERIFIED (`:50-51`), i.e. the disabled-policy V1 control is inner-face eligible
  by declaration; this preserves V1 semantics but should be stated where the control arm is described.

Decision after drift: unchanged - P1 present (P1-2), P1-1 provisionally resolved in the drifted files pending a full re-review of the
drifted chain and an unmocked seam test.
