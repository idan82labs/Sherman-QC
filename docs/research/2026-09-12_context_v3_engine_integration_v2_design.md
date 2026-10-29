# Context V3 integration test V2: fixture-policy correction

Date: 2026-09-12. SOURCE-ONLY / NEW TEST REVISION / NOT EXECUTED.

Root reports the prior 94-outcome attempt is terminal: 21 passed and 73 failed
with `INVALID_WHOLE_PART_POLICY_ORDER`; its dense-support outcome passed.
This document does not replace the independent failure POST or reclassify those
outcomes. The prior test, candidate, harness, protocol, PRE and attempt remain
frozen. No old attempt/private artifact or physical data was read by this author.

Only these NEW files are authored here:

- `tests/test_cad_guided_context_v3_engine_integration_v2.py`
- This design note.

No production policy, engine implementation, scientific threshold or scoring
rule changed. No target import, collection or execution occurred. Checks were
source reading, standard-library AST/hash comparisons and textual diff only.

## Minimal correction and exact preservation

The original fixture's `WholePartRegistrationPolicyV1` set inlier p95 to 0.1 mm
and correspondence distance to 0.5 mm but inherited the 0.25 mm Huber delta.
Its actual constructor requires `huber_delta_mm <= max_inlier_plane_p95_mm <=
max_correspondence_distance_mm`, so fixture construction failed before the
intended integration seam. V2 explicitly sets the fixture-only Huber delta to
0.05 mm, making the tuple `0.05 <= 0.1 <= 0.5`. All other fixture values remain.

All 93 original parameter cases are retained. Nineteen of the 20 original test
function ASTs are unchanged, accounting for 89 cases. The four cases in the
remaining budget-selector function retain their original dict/bool/V1/V2 inputs
but no longer accept any arbitrary ValueError: each first constructs a valid
complete configuration, then wraps and delegates to the real context V3 budget
validator, requires exactly `INVALID_BUDGET_TYPE`, and asserts that validator
received the exact invalid object once. An earlier policy error cannot satisfy
those assertions.

One new constructor/configuration guard checks the valid three-value tuple,
point/iteration and remaining ordering constraints, and the owned reconstructed
configuration. A constructor-only negative control requires the exact ordering
error for the former invalid tuple; it does not invoke the old test or attempt.

Thus the NEW gate has 21 functions and 94 prospective outcomes: **93 preserved
cases plus one policy guard, not the former 93 plus dense capacity test**. The
already-passed dense-support test is neither imported nor collected. The static
source-order counts are:

`1,1,2,4,7,5,9,4,14,2,1,12,10,4,3,4,4,1,1,4,1`

The complete named-node AST delta is `_policy`, the four-case budget-selector
test, and the added policy guard. Other changes are the explanatory module
docstring. No function or parameter case was removed, no geometry was changed,
and no failing-only subset is proposed.

## Constructor, invariant and call audit

The complete current original test was read, including its single inert
`backend` graph, which preserves exact dataclass identity across relative and
absolute frozen imports. That graph is unchanged in V2. All explicit fixture
policy/budget constructors and the validators reached by configuration were
checked against source, including inherited workflow and native-policy checks.

| Fixture seam | Source-level constraint check |
| --- | --- |
| Whole-part registration | 50 coarse/refine points satisfy 36 minima and refinement ordering; one iteration is admitted. Explicit `0.05 <= 0.1 <= 0.5`; trim 0.85 is in [0.5,1], rotation step 0.15 <= 0.25; positive finite scalar and ratio defaults are valid. Reconstruction repeats the actual constructor. |
| Workflow V14 -> V2 -> V1 | 36 registration rows and 36 CAD samples meet the separate minima; default deflections 0.1/0.2 are within [0.0001,1]; native adapter V1 and provisional-mm declaration are exact. |
| Native policy -> CADGuidedFeature | Fit residual 0.05 < plane exclusion 0.075 < search margin 3; flange window 0 < 2.5 < 12.5; tangency 0.1 positive; free-circle mode admitted. The pure validation feature's nominal/vector defaults satisfy its constructor; no native fit is called. |
| Support policy | Axial gap 4, angular gap 5, minimum sweep 20 and three local points are inside the four declared ranges. Positional argument order matches its dataclass. |
| Noisy/patch/search policies | Unmodified noisy tiling (128,1024), halo one and scalar defaults satisfy their validators. CAD patch minimum support 12 is below each 30-vertex CAD sheet. Search minimum support three is below each 25-row scan sheet. Search bank cap two is permitted by the budget validator; it is not a minimum-three policy violation. |
| Initializer/surface/registration budgets | Default V12 limits and positive chunk size match its literal bounds. Default triangle-surface leaves/batches/ties satisfy admission. Default V3 execution budgets match its integer maxima. No cap was increased. |
| Context V3 budgets | Defaults satisfy 3072 queries <= 3*1024 families; chunk_rows=3 and max indexes seven are valid submaximal overrides. Max indexes two is the intentional connected capacity-refusal case. V1/V2 objects are intentional wrong-type inputs, now checked at the exact validator. |
| Schedule and composite selectors | V1 retains budgets=None. V2 supplies the exact default V2 type. Composite mode uses its default seeded/composite snapshots, within their own integer maxima. These do not change the context selector. |

Call plumbing was also read against current signatures: five positional arrays
and keyword-only bounds/policy/units for the public prefilter; `_FaceQueries`
has `(self, face, points)`, so the existing spy's `args[2]` is the point array;
the workflow's keyword-only inputs and schedule/context overrides match V14;
selected-context binding receives its explicit context index and expected
configuration. Native decode/extract/sample and registration/measurement remain
explicit mocks in the byte fixture, not newly claimed end-to-end work.

The 48-row byte fixture retains 36 unique selected raw rows and immutable
byte-backed transport arrays. Its CAD samples intentionally repeat three exact
dyadic interior locations with consistent oriented normals, face/triangle IDs
and barycentric weights (0.5,0.25,0.25). The live byte surface binder checks that
transport; real registration is mocked there, so this is not a claim that three
unique CAD samples satisfy the separate real registration input minimum. The
real numerical fixture remains 50 rows/64 scan triangles/80 CAD triangles,
with two independently triangulated sheets and the full two-match/eight-family/
64-slot schedule expectation. No capacity fixture is reduced or repeated.

No additional constructor or keyword defect was identified in this bounded
source audit. This is not an assertion that newly reached numerical or binding
paths have passed: the next exact run may reveal other failures, which must be
retained honestly rather than silently repaired or retried.

## Exact source handoff and next gate

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| Frozen original integration test | 42073 | `3b40f2a16a1a6ee48b19716473d57b64880e4f087059f7a7c3e14c370d2db76b` |
| New V2 integration test | 43998 | `67b705809d2b96a557981673d9c571a462c8b9a5ed196bf7593b53a1247e677b` |
| `backend/cad_guided_plane_initialization_v13.py` | 38651 | `79ee4de537f1dddcc1dee40d33b25fe3fe8fb667bcd4e30ecc0a811ebe2d989f` |
| `backend/cad_guided_mesh_offline_v14.py` | 33435 | `76378b93c44c046b773452b3d8cad400cb237ff8bc7039e258a90305bbf6c749` |

Root owns a new harness/protocol and independently frozen source/runtime map,
PRE, archive, exact denominator and resource limits before one qualification.
The unchanged backend static import superset remains applicable; replace the
collection target with the new V2 file, do not import/collect the frozen original
or the already-passed dense test. Source custody may separately bind historical
files for preservation without executing them. No invocation is released here.

The author of this V2 file also authored the original faulty fixture; root and
the PRE reviewer must independently review this correction. Passing a future
gate would qualify the connected invented integration seam only, not physical
accuracy, native fitting, F1 or promotion. The separately frozen small consumed
diagnostic and any later full Gold11 score remain distinct gates.
