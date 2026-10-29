# V11 consumer-chain integration — preliminary source audit

Date: 2026-09-13 23:42 UTC. Reviewer: Codex
`v9_completion_cost_postmortem`. Fable remains the sole implementation/test
editor. Scope: V18 native factory/bridge/measurement dispatch, layer-policy wire
grammar, fixed raw-support propagation and the named connected test. The other
reviewer owns layer-mode/refit mathematics. No test/import, geometry, physical
input, Gold, terminal artifact, control or cache read/execution occurred.

**Preliminary decision: two concrete declaration/validation corrections below;
no final acceptance at this active draft.** The full connected fixture is still
being developed, so its absence from the current test is a WIP coverage limit,
not a final author claim or a reason to invent another readiness programme.

## Snapshot pins

Workflow, bridge, context, cover and named test hashes were the same at initial
and final inspection. The kernel changed during early inspection; only its
later pinned type/policy/support interface is referenced here, not the moving
algorithm. Contact/projector hashes were stable across their two reads.

```text
b26d32c215c1c71edf163de1b704266de6541da85f22a36dc358758f4f290fdf backend/cad_guided_mesh_offline_v18.py
a2b0256cecc53bce5c153e3c4a7466ddc91cde26e7e9ac24cd1efb280eafe868 backend/cad_guided_native_feature_bridge_v2.py
117cd188d07f3f6b32b4fe65b3c8c08edafe4958e67c75943bd1965d0cd8d3f2 backend/cad_guided_measurement_v3.py
0084c9215f61d4dab57b6fc2cc4b8b9b3ea3716a382259597cb061486a1e57fb backend/cad_guided_context_measurement_v4.py
7a0b40cfb888811a039bc677e9c186c415195823db709525d702d76b349d95a6 backend/cad_guided_support_cover_measurement_v5.py
cb3072b4f93497de671a0b373c49aaa94443c77c30dcd3e889146502f2d05e2b scripts/gold11_contact_scoring_v5.py
adae1deb0e197b7331dab7bdda4d8248a72e8d184c7ef40db80a0d54b17ec395 scripts/gold11_schedule_projection_v6.py
72790f547a3c68b19b2db9bbf4c2894790d4850e879769306fabbaf20175624c tests/test_cad_guided_v11_connected_chain_v1.py
```

During final report writing, the kernel moved to
`51101ccec5b5c34b0a9c463b7a13e073c8524dcfe320093fff01fa2def688514`
and context V4 moved to
`7d12b92204334bce27e0012886fa41b75bc8b256f3c5c31de83cebdd243666b2`.
Those later drafts were not inspected here; statements/line references apply
to the reviewed pins above. All other final-write pins remained unchanged.

## Positive source connections

- Both native factories still construct kernel-V1 features
  (`cad_guided_native_features.py:17/310`, native V1:33/692). V18:587–595 invokes
  the selected real factory and then bridge V2. The bridge imports kernel V3's
  actual feature class (32), checks exact V1 input class and field grammar,
  rebuilds every field and verifies lossless field equality (82–106). Its cap
  remains 64, matching cover V5:150–153. The stale local alias `_KernelV2Feature`
  is a name only: the object it constructs is the V3 class. No V9-style class
  mismatch is present in this inspected chain.
- V18 imports the V3 layer policy and context V4 (33–36), reconstructs the
  validated policy (513–515), then passes the bridged tuple and that policy to
  real cover V5 at 691–697. Cover V5 imports context V4, which imports the V3
  feature/policy/measurement functions. It invokes context V4 and its validator
  before cover construction, and re-fits the fixed support through the same
  context implementation (cover:157–161, 184–207).
- Context V4's 13-field layer policy is separately serialized from the seven
  scalar fit fields (59–66, 154–162). The policy is included in raw-support
  equivalence digests and copied to records/cover witnesses (445–460;
  cover:188–195). Context publication reconstructs LayerSelectionPolicyV2 and
  requires exact field keys (577–581), then binds record policy to result policy
  (639). The evaluator's weaker validation is the exception identified below.
- The feature transform uses `R.T @ (origin - t)` with direction vectors rotated
  by `R.T` for `p_cad = R p_scan + t` (context:172–176). Layer-selected flange
  masks are mapped back to local point indices by the kernel (V3:775–797,
  992–995), then to original raw-row IDs using `selected[...]` in context
  441–443. Canonical/cover refits use those exact raw-role rows, not CAD-derived
  measured radius or axis (context:179–260; cover:184–207).
- Cap-role exclusions have an explicit raw bitmap and retained filter binding
  (context:393–423). Publication rejects any support row belonging to its
  associated excluded bitmap (673–674), overlapping raw roles (676–677) and
  prediction lists that differ from the exact finite-support projection
  (725–730). Layer exclusion is a distinct mechanism: selected raw flange IDs
  persist, while layer-mode/bridging counts and fitted group planes remain
  diagnostic context evidence. It must not be described as physical defect
  classification or calibrated outlier rejection.
- Canonical geometry is refit from raw coordinates; finite sampled endpoints
  are converted to a contact-sector generator using the measured radius and
  measured plane directions (context:197–258). Empty inner-face eligibility
  remains explicit through context/cover/contact; generic carriers cannot gain
  nominal inner-radius deviations, physical trim claims or observed-bend count.

## P2 — serialized evaluator accepts a broader layer-policy domain

At contact scoring V5:259–267, `_layer_policy` checks exact field keys, integer
types for three fields and positive finite scalar types for the others. It does
not enforce LayerSelectionPolicyV2's ranges or cross-field constraints
(kernel V3:129–151). Examples accepted by the helper's inspected predicates but
rejected by the kernel constructor include:

- `layer_refit_iterations = 0` or 9 (kernel requires 1–8).
- `max_bridging_row_fraction = 0.9` (kernel maximum 0.5).
- A bin width exceeding one third of the split gap, or membership band exceeding
  half the gap; inverted max-gap/split-gap relation.
- Invalid integer row/report bounds such as `min_layer_rows = 1`.

The evaluator calls this helper for serialized source policy at 462 and only
checks equality in nested records/groups at 464/490–491. Thus matching malformed
copies are not rejected by this domain validator. The live producer's constructor
does enforce its domain, so this is a defensive wire-contract mismatch, not a
claim that normal engine execution currently emits invalid policy or that a
full malformed prediction bypass was executed. No such probe ran here.

Small correction: mirror the canonical numeric domain in the inert serialized
validator (or use a shared side-effect-free validator if appropriate), preserving
strict JSON types and the no-engine-import scorer boundary. Add boundary and
cross-field cases, including mutation of all serialized copies together so the
test cannot pass merely on a stale-copy equality check. The existing connected
test at 409–411 mutates only a group copy and establishes mismatch rejection,
not constructor-domain equivalence.

## P2 — machine implementation labels still claim bridge V1

V18:48/55 and schedule projection V6:36 include `NATIVE_BRIDGE_V1`, while V18:36
and 595 actually use bridge V2 targeting kernel V3. V18's separately exported
`NATIVE_FEATURE_BRIDGE_IMPLEMENTATION` is correctly derived from the bridge, so
this is inconsistent provenance labelling, not an actual wrong-type dispatch.
The V18 header also says bridge V1 is unchanged (3), and the bridge function
docstring still says kernel V2 (73). Historical quoted headers may remain, but
current machine identifiers/current descriptions should say bridge V2. Change
the new producer/projector/test declarations together before any freeze; do not
edit frozen predecessors.

## Current tests: what is real and what remains WIP

The named connected test imports real kernel V3, context V4, cover V5, contact
V5 and projection V6 (fixture at 29–52). Its fitted geometry, canonical refit,
cover planner, finite-support builder and serialized scoring path are real.
The positive asserts measured radius and generic identity, then exact carrier
transport (223–255). Negative identity/stale-copy/old-schema cases and the
explicit subset grammar path are meaningful transport checks.

However, its features are directly constructed kernel-V3 objects (84–87), its
registration is handcrafted identity evidence (101–114), and its prepared roles
and selector are keep-all doubles (117–145). The scoring reference is copied from
the emitted carrier (195–215), so numeric synthetic F1 is transport arithmetic,
not independent localization accuracy. The ordinary-refusal coexistence test
also explicitly substitutes `_canonical_fit` for one feature (324–343). None
of this establishes real native factory -> bridge -> real registration -> real
surface roles -> measurement. The header labels these substitutes, though its
version names are stale.

Root has already requested the real connected positive/negative fixture and
Fable is developing it. The bounded acceptance should require one independently
constructed nonnominal scan with declared transform/geometry truth through that
actual chain, plus a deformed-patch negative that cannot disappear solely by
support selection. Assert nonempty positive output without conditionals/skips,
measured geometry against the independent construction rather than the emitted
carrier, native-to-V3 type identity, exact retained raw-row support/exclusion
counts, and honest unavailable/uncertain negative semantics. Keep copied-line
transport scoring as a separate claim. This is the requested fixture, not an
additional broad simulation programme.

## Handoff

Both concrete corrections were sent to root. No kernel-mode mathematics is
qualified here. The author should finish the in-progress fixture, close the two
wire/declaration issues and publish matching executed receipts/source pins for
a bounded delta review. Current source existence is not a passing receipt,
final V11 acceptance, PRE or run authority. Only this audit file was written.

## Preliminary full-bytes fixture review — 2026-09-14

Scope: source only, while the author runs a separate synthetic diagnostic. No
test/import/geometry execution, diagnostic output, physical data, attempt,
Gold/reference or cache was accessed. This appendage reviews the already-requested
connected fixture, not a new V11 gate. Fable remains sole implementation/test
editor. No executed result or final acceptance is asserted.

The new `tests/test_cad_guided_v11_full_bytes_connected_v1.py` really removes the
previous geometric substitutes: `_run` (162–173) constructs independent analytic
scan points/triangles, applies the quaternion oracle pose, writes binary PLY bytes,
constructs an OCP strip and serializes only that owned temporary STEP, then calls
the actual V18 byte entrypoint. No geometry monkeypatch is installed by its `m`
fixture (47–74). Constructor native imports occur inside their helpers, not at
module collection. Its runner-config import defines constants/functions; runner
execution remains under the `__main__` guard. The inspected constructor/import
seams do not load physical data. This is source inspection, not a general import
side-effect certification of every transitive dependency.

V18 confirms actual PLY preparation/topology and raw-byte binding (564–578), owned
STEP read/native V1 factory/bridge V2 (583–603), CAD sampling (604–616), initializer
V16 with real raw rows/triangles and registration V5 (632–646), registration binding
(650–678), then real context roles/cover V5 and publication validation (692–703).
The scan radius 5.06 versus independently constructed CAD radius 5.0, and the
known oblique pose, do not come from extracted features or output predictions.
The positive radius and carrier oracles are therefore materially stronger than
the previous factory-conditioned/copied-reference fixtures. Success is required,
not conditionally skipped. This file exercises the control mechanism only;
it does not qualify the composite challenger schedule or any physical accuracy.

### Concrete assertion and execution-contract gaps at this draft

1. **Original-row exclusion assertion is potentially vacuous.** Lines 291–296
   look for optional `flange_row_indices`/`flange_rows` inside
   `finite_support.instance`. The actual original-row provenance is the required
   `group.raw_support.flange_rows` (cover V5, 185–187); context V4 maps local kernel
   support back through `selected[...]` at 443–456. If both optional keys are absent,
   the negative assertion executes zero times. Require nonempty measured groups,
   the real raw-support fields, and exact disjointness from displaced original IDs
   in all retained roles. Do not treat a count or localized instance index as
   original-row lineage.

2. **The declared deformed patch and exact-count oracle are not yet what is tested.**
   Lines 117–122 randomly displace 12% of one flange's rows; this is scattered
   outlier deformation, not a spatially connected patch. Selection also advances
   the RNG before noise, so the clean call has different background noise from
   either deformed call. The positive/negative displacement calls do share their
   selected IDs and noise. Use a predeclared connected spatial mask with an
   independently seeded noise field for the required coherent-patch control, or
   narrow this test's claim to scattered defects without claiming it closes that
   patch check. `_crop_rows_role0` (126–130) assumes the CAD-nominal +x role and exact
   nominal pose, while the real crop is per recovered context and filtered by the
   real role mask (context V4, 389–425). The exact equality at 267 and lower bound
   at 284 do not establish exact row accounting through that path. Bind assertions
   to the correct context/role and original IDs; account separately for rows
   removed before the layer kernel and rows excluded by the layer kernel.

3. **A fresh allowance object is not a fresh CPU clock.** `_policies` (133–151)
   sets a 720 s guard and 900 s declared budget, but V5's guard is explicitly
   absolute process CPU, including imports/preparation (103–115, 156–159,
   209–224). Running all three tests in one pytest process does not provide three
   independent 720 s allowances; later cases inherit earlier CPU consumption.
   This module also does not itself enforce OMP, external CPU/wall or RSS limits.
   Final acceptance execution should declare fresh-process case isolation and
   fixed external limits before starting. Do not change an active diagnostic's
   limits. Source-level cooperative headroom is not a hard process cap.

### Claims to keep narrow (not additional architecture findings)

The positive assertion at 230–240 attributes every between-skin exclusion to the
constructed axial cut-edge rows. The real role selector can remove qualified
axial-cap-only rows before measurement (roles V1, 381–407; context V4, 415–425).
Whether these noisy cut-edge triangles obtain those proposal memberships is a
runtime question not answered here. Require row-based evidence for the claimed
route, rather than assuming all cut-edge rows reach the layer partition. This is
not a prediction that the active diagnostic must fail.

The noise bound is per coordinate (line 122), so its Euclidean norm bound is
sqrt(3)*0.005 mm, not 0.005 mm. The line oracle checks distance to the infinite
generator, direction and minimum length (176–179, 219–229); it does not independently
bound the axial location/endpoints of finite support. Do not describe that as
independent physical endpoint truth. Pose accuracy currently requires at least
one supported candidate (201–208), not every published conditional context.

### Source pins and disposition

The following hashes were unchanged between the initial and closing source reads
(cover V5 and registration V5 were additionally pinned at the closing read).
The candidate remains an active draft; later changes require a matching delta
review and executed receipts, not reuse of this preliminary snapshot.

| Source | SHA-256 |
| --- | --- |
| `tests/test_cad_guided_v11_full_bytes_connected_v1.py` | `53197348eb45247f9900427a585a4a2da668a393e9978c8bbc15d6edf2ded6d9` |
| `tests/test_cad_guided_native_feature_bridge_v1_factory_to_measurement_v18.py` | `a87d06d249e475817e80cf1b3632e0faeb16c5d139d4895c92e535edbcb5d38a` |
| `tests/test_cad_guided_mesh_connected_native_v1.py` | `11455ceda7283bcd1b84dbd037b7bd16bd2517bd8d44985ef8f774d2e5c100e8` |
| `backend/cad_guided_mesh_offline_v18.py` | `57ae0310135c915bab4820ed55bd0d7595ca069ac47e10e74978933163374fe6` |
| `backend/cad_guided_context_measurement_v4.py` | `7d12b92204334bce27e0012886fa41b75bc8b256f3c5c31de83cebdd243666b2` |
| `backend/cad_guided_context_surface_roles_v1.py` | `c042408109d260888834ebacc9ad3b62256fda589f1ea8cbba31164b2bea9a62` |
| `backend/cad_guided_support_cover_measurement_v5.py` | `7a0b40cfb888811a039bc677e9c186c415195823db709525d702d76b349d95a6` |
| `backend/cad_guided_global_registration_v5.py` | `ee05a03eb564c9fcc8c3cee631e4f429d0c66d9ec5468bc932377368105750e7` |
| `scripts/run_gold11_mesh_development_v11.py` | `e8de63e4c0172b7567f9e53709eabeef05ea94ac94ea24cf61355a9fd33cc71b` |

The full-byte integration path is present at source. The three concrete gaps
above were sent promptly to root; they concern whether the required fixture's
claims and final execution are nonvacuous and reproducible. They do not establish
an engine defect, a diagnostic outcome, PRE approval or a Gold rerun result.

## Revised full-bytes fixture: bounded source closure — 2026-09-14

Reviewed only the revision requested by root, with no imports/tests, diagnostic
outputs, attempts, physical data or caches accessed. Fixture SHA-256 at both
opening and closing reads:
`5e1eb4c0f4773da5472e4756f89a3177433cf708217a5a8d2ccf9e5480c5b782`.
The other eight constructor/connected-source hashes in the preceding table were
rechecked and remain unchanged. The author may continue editing; this is a
bounded draft snapshot, not an executed receipt or accepted V11 bundle.

### Closed at source

- **Required original-row support:** `_assert_group_pose_and_support` (250–267)
  now accesses the required `group.raw_support`, requires nonempty arc and both
  flange roles, validates original-row range/uniqueness, and checks every role
  against forbidden IDs. The between-skin test requires nonempty groups and
  invokes it with `run.chosen` (377–384). The optional-instance-key loophole is
  closed, using the actual cover V5 raw-support contract (185–187).
- **Connected patch and unchanged noise:** construction now tags original grids
  and selects a fixed rectangular region of the inner +x flange before noise
  (125–146); the same independently seeded noise array is generated for every
  displacement variant. This closes the scattered-points and unpaired-background
  defects. The rectangle is a connected patch of the triangulated grid.
- **Emitted-context pose binding:** `_group_contexts` links each group's member
  record IDs to source windows and their registration candidates (242–247), and
  every such candidate must be supported and satisfy the independently constructed
  pose-family error bound (254–258), not merely one unrelated successful pose.
  The declared symmetry swaps x/y and reverses z about height/2; its matrix is
  orthogonal with determinant +1. It preserves the symmetric nominal strip and
  the generator line, so this is a valid proper-rotation ambiguity, not reflection
  acceptance. Both positive emitted-group tests invoke the check.
- **Execution contract:** lines 33–35 and 61–66 explicitly require a fresh process
  per case, fixed 900 s external CPU / 1200 s wall, 720 s cooperative guard,
  4 GiB sampled RSS, OMP=1, disabled plugin autoload/cache, and a fresh-process
  preflight at 189–196. `_envelope_guard` (210–215) now fails the fixture if the
  actual disposition ledger contains expired/omitted/not-attempted candidates.
  The ledger keys match initializer V16's contract (863–877). This closes the
  misleading independent-allowance declaration. Actual fresh-process launch,
  external RSS sampling/enforcement and matching execution receipts remain root's
  execution review; the shown command itself does not implement RSS sampling.
- **Claim precision:** noise is now explicitly per coordinate with the correct
  norm bound; line distance/direction/minimum length are expressly not exact
  finite endpoint truth. The pose criterion has changed from 0.20 to 0.50 mm and
  its development-observation basis is disclosed at 18–20. This is a newly
  declared development criterion, not evidence that the former criterion passed
  or an independent calibration bound. No diagnostic numbers were verified here.

### Still partial: actual-context exclusion accounting

The prior exact-count finding is not fully closed. `_crop_rows` (150–156) still
counts tagged rows in the nominal CAD frame. Positive and negative assertions
compare those counts with every actual context (314–326, 349–365, 375–396),
without deriving tagged crop membership for each recovered pose. Its docstring
says the construction's margins are much larger than the pose bound, but the
declared grid disproves that premise: sheared rows have flange distances 2.7
and 12.3 mm, only 0.2 mm from the 2.5/12.5 crop boundaries, and the first sheared
axial level is `0.45*tan(35 degrees)`, about 0.315 mm from the axial boundary.
Those margins are smaller than the new 0.5 mm action-error allowance. This
does not prove an actual run fails; it means the nominal count is not a valid
general substitute for actual-context membership under the declared bound.

The zero pre-kernel role-exclusion assertion is now explicit, and the sheared
geometry is intended to exercise the layer partition after real role filtering.
However, aggregate equality with the number of sheared tags does not by itself
prove that those particular original rows, rather than equal-count replacements,
were excluded. Unlike patch IDs, sheared IDs are not checked against retained
group raw support. The air-side case permits either failure role but does not
independently check its non-emitting contexts against the pose family; it cannot
use the emitted-group pose bound as a premise for its nominal-count equality.

Smallest closure: derive each tested window's tagged membership using its actual
recovered context/crop (while retaining the independent pose-family check), test
the expected tagged original IDs against retained support, and reconcile the two
exclusion stages separately. Add the same pose-family check to the air-side
windows if their counts rely on that bound. Alternatively narrow the assertion
and narrative to aggregate conservation, without claiming exact tagged-row
exclusion. No engine redesign or new experiment programme is requested.

Disposition: the raw-ID nonvacuity, connected-patch/noise, explicit execution
contract and emitted-context pose-link findings are source-closed. Exact
actual-context/tagged-row accounting remains a concrete partial closure. Sent
to root promptly; only this audit was appended. Running-test results and final
V11 acceptance remain separate.

## Original consumer findings: source closure — 2026-09-14

Scope is the two original consumer findings only: serialized policy-domain
validation and bridge-V2 machine-label consistency. The author closure narrative
was read narrowly through its consumer section (lines 41–49); no full-bytes case
outputs, live diagnostics, attempts, physical/Gold/reference data, caches,
imports or tests were accessed/executed. This is not a full runner audit or
measurement-mathematics review. Root owns matching execution receipts. The
full-bytes accounting qualification above remains separate.

### P2 serialized layer-policy domain — CLOSED AT SOURCE

Contact V5, 263–279, now restates all nine scalar ranges, all three integer
ranges and the enabled boolean, with the same three cross-field comparisons
and binary64 division as `LayerSelectionPolicyV2.__post_init__` in kernel V3,
142–164. This includes the prior concrete examples: refits 0/9, bridging
fraction .9, too-small row count, oversized mode bin and membership band all
refuse. All scalars must be finite JSON numbers, booleans cannot masquerade as
numbers, and integer fields are exact ints. This remains inert validation;
contact V5 does not import the engine to validate an evidence payload.

The helper is reached by the real serialized result consumer at contact V5,
475. Every retained record is compared to that validated policy at 477 and
each covered group's policy is bound to the witness and source at 503–504.
Thus an invalid policy cannot evade the new domain check by making all its
serialized copies agree.

The prior test-vacuity gap is also closed at source: `_every_layer_policy`
walks the complete payload (724–736), and the rejection test mutates every
copy together, requires at least five copies, then requires both malformed
disposition and a layer-policy reason with zero predictions (748–756). The
accepted-boundary test retains two predictions (765–771). The direct
constructor/mirror test checks scalar/integer bounds and cross-field cases
against both validators (774–827). This is a meaningful acceptance matrix,
not merely a check for mismatched copies. No pass result is asserted here.

Two nonblocking wording corrections were sent to root: the author narrative
line 45 says 24 invalid cases, while the current parameter list at 739–747
contains 26; and exact domain agreement should be phrased as agreement on
serialized JSON numeric values. The Python constructor accepts float-convertible
numeric strings, while the scorer deliberately rejects those strings. That
stricter wire grammar is safe and does not reopen the broader-admission defect.

### P2 bridge-V1 machine labels on the V2 path — CLOSED AT SOURCE

Bridge V2 imports/builds the exact kernel-V3 class (32–37, 84–95), publishes
the V1-to-V3 target in its implementation/provenance (35–37, 100–104), and its
function docstring correctly names V3 (72–77). Historical V1 headers remain
explicitly quoted history, not active class or implementation declarations.

V18's implementation and comparator (50, 57) now both name bridge V2; the
workflow returns that implementation (439, 465). Projection V6 declares the
identical implementation (36) and requires it on received payloads (85–87).
The V11 runner's two arm definitions (181–195), comparator (198), arm-request
implementation (611) and score-request identities (2118–2119) consistently name
bridge V2/kernel V3. The active V11 header names V2, while V10/V9 labels remain
under explicitly historical prose or version-delta fields. No current
machine-label mismatch was found in the bounded named-source chain.

### Opening/closing pins and disposition

All eight listed sources/narrative were unchanged between this review's opening
and closing hash reads. Findings apply to these bytes, not a future edited
bundle. This audit does not independently attest authors' executed test results.

| Source | SHA-256 |
| --- | --- |
| `docs/research/2026-09-14_fable51_v11_review_response.md` | `e02d9db179fef77d79496b953b9c59e8d4bb95f06aa10918cdc8afe82a1b6c88` |
| `scripts/gold11_contact_scoring_v5.py` | `dd0963c2cbc8173fe94ce686b9fbe04f9bac9998ce1881afd5d4ec7cc1f000c7` |
| `tests/test_gold11_contact_scoring_v5.py` | `12e9944a0c77a4ec34e9b7df7abd22f3c2e6013d01426152086e6dd08222abdb` |
| `backend/cad_guided_measurement_v3.py` | `cf978d8617a190ca8db29d1f31bb859a3e123d14c7c21c4fade0cf322b2e1439` |
| `backend/cad_guided_native_feature_bridge_v2.py` | `ecac003e07b5cb7e91432352c6d88fe6b4bfd46f7fcfbeab19addaba671380a4` |
| `backend/cad_guided_mesh_offline_v18.py` | `57ae0310135c915bab4820ed55bd0d7595ca069ac47e10e74978933163374fe6` |
| `scripts/gold11_schedule_projection_v6.py` | `b419903ca8d7e1aa8d934d3e2e4088568db8c39391b8d7118b7f45e4f549bf9d` |
| `scripts/run_gold11_mesh_development_v11.py` | `e8de63e4c0172b7567f9e53709eabeef05ea94ac94ea24cf61355a9fd33cc71b` |

Both original P2 findings are closed at source; no additional blocking finding
in this limited seam. This does not close the separate full-bytes fixture
qualification, certify a physical score, constitute PRE or authorize execution.
Only this owned audit was appended.

## Full-bytes interval-accounting delta — 2026-09-14

Read AGENTS/current status first, then the requested source seams only. No
imports/tests/geometry or run-output/data/cache reads. Root independently owns
the completed fresh-process evidence; this section does not attest those logs.
Fixture opening/closing SHA-256 was unchanged:
`7ed7aa1756be0939e9f900c6fb8cb99ba4bc281100c35aa895c153dfd51159b9`.
It is still an active draft. The narrow kernel crop read was at
`8fd5c3e6a7f6437638f068052d8c71e085980b25fa7d14cca955cf5f66c5d6e6`,
which differs from the previous kernel pin; no broader kernel delta is approved
here. Context V4 remained
`7d12b92204334bce27e0012886fa41b75bc8b256f3c5c31de83cebdd243666b2`.

### Useful source closures

`_sheared_bounds` (168–174) replaces nominal exact counts with an eroded/dilated
rectangular crop. This is the right conservative construction: if a recovered
point's Euclidean error is at most b, each coordinate error is at most b.
Moving every crop half-space inward by b gives certainly-inside rows; moving
outward gives possibly-inside rows. The positive and symmetry-related flange
crops are equivalent up to the declared proper rotation/permutation. The kernel
flange masks (807–833 in the pinned kernel) use these nominal axial, flange-ray
and lateral bounds; their tiny axial numeric guard does not change membership
for this constructed grid. This is an engineering interval check for this
fixture, not certified floating-point metrology.

The group support helper now rejects every tagged sheared ID from all retained
roles (281–286), in addition to the existing dent-ID check. That directly closes
the prior possibility of claiming sheared-row exclusion using only a matching
aggregate count. The header at 23–26 appropriately narrows identity to no tagged
row retained plus interval aggregate accounting; it no longer claims the exact
per-context excluded sheared row set. Conservation and zero pre-kernel
cap-exclusion assertions remain. Every refused air-side context now gets its own
independent pose-family check (375–380), and the dent patch is explicitly checked
to lie in the eroded crop (371–373). RSS wording at 37–40 and 66–68 now correctly
describes a budget checked by HWM/ps rather than continuous enforcement.

For any diagnostic window not feeding an emitted group, the group helper does
not establish that window's pose bound. The interval claim should be attached
only to windows with that bound checked, as is now done explicitly for air-side
refusals. This is a scope qualification on the interval premise, not a demand
for additional geometry work.

### Concrete remaining draft assertion error

Positive line 335 requires `bounds[0][0] > 100`. The declared construction and
new half-millimetre erosion instead give exactly **90** certainly-inside sheared
rows per flange:

- The retained flange distances are 3.3, 3.9, ..., 11.7 mm: 15 columns.
- The first axial shear level is `0.45*tan(35 degrees)`, approximately 0.315 mm,
  so it lies outside the eroded 0.5–29.5 mm span at each end.
- The other three shear levels survive at each of two ends: 15 × 3 × 2 = 90.
- The outward crop admits 19 columns and all four shear levels at both ends:
  19 × 4 × 2 = 152 possibly-inside rows. Per-coordinate noise of at most 0.005 mm
  cannot change these memberships. The former nominal count was 136.

Thus the interval is [90, 152]; the `>100` precondition guarantees a fixture
failure independently of engine output. Correct the fixture assertion from
the declared grid mathematics; do not change the geometry, engine or experiment
cap to satisfy it. The nearby comment at 331–333 still says "exactly the
constructed count per role" and should follow the already-narrowed header.
This concrete error was reported immediately to root.

### Between-skin negative is not an accounting-only result

At this snapshot, the between-skin test still unconditionally requires PARTIAL,
nonempty predictions and measured groups (399–408), before its revised count
assertions. Root reports that the completed earlier case instead abstained with
`NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS` without CPU expiry. This review did
not inspect its output and makes no independent causal diagnosis. In particular,
an abstention before those assertions is not fixed by correcting 131-versus-136
accounting and must not be presented as an already successful measurement.

A deformation-control case need not emit. If retained evidence supports a
legitimate measurement refusal, a separately recorded development revision may
qualify that honest negative outcome, with the original failed emission
expectation preserved. Keep it distinct from the positive nonnominal measurement
case; require the relevant actual native/registration/measurement path and
specific refusal evidence, not a catch-all "any empty result passes" branch.
Do not claim measured radius, transported carrier or successful support exclusion
publication for an abstained case. The current source has not yet made that
expectation change. No new broad test programme or forced-emission engine change
is requested.

Disposition: interval method, tagged sheared-ID disjointness, refused-context
pose binding and honest RSS wording are improved/closed at source. The
90-versus-`>100` fixture assertion and between-skin expectation remain unresolved
at this pin. No executed pass, final V11 acceptance, PRE or run authority follows.

## Final-campaign fixture snapshot: conditional negative/cover check — 2026-09-14

Source-only review at root's request, after AGENTS/status and the newest author
ledger entry (2026-09-14 01:40 UTC). No tests/imports, live outputs, diagnostic
artifacts, physical/Gold/reference data or caches were read or executed. Root
separately owns the completed diagnostic evidence. The active campaign's results
are not inferred from source or from the prior diagnostic.

Opening/closing fixture SHA-256 was stable:
`6ff4d1540d4e7528943540249d6f6fccebf1f48d4f315d9ea783b259d2966110`.
Context V4 and cover V5 retained the preceding pins
`7d12b92204334bce27e0012886fa41b75bc8b256f3c5c31de83cebdd243666b2`
and `7a0b40cfb888811a039bc677e9c186c415195823db709525d702d76b349d95a6`.
The directly connected planner was read at
`70fc42d19f67346b6b5f4951832edef481aa4d94abaeec892b415cfc018cab47`
(`backend/cad_guided_exact_support_cover_v1.py`); author ledger at
`87447746bf88b1c6f666977af649f274fefb0bf2ed2da1a24a07840a842d575a`.

### Closed/improved at source

The incorrect `>100` interior precondition is replaced by the independently
derived constants 136 nominal / 90 interior / 152 possible (81–84, 351–352).
The interval calculation now checks equality with those declared constants.

Every window used for geometric interval/count attribution now has its own
pose-family error evaluated; claims apply only when it is at most 0.5 mm, and
each case requires at least one such window (positive 357–376; air-side
403–427; between-skin 457–480). Out-of-bound windows are not silently discarded:
they retain status, reasons, pose error and aggregate accounting in the receipt.
This is a conditional local mechanism claim, not accuracy across every admitted
pose. Positive and between-skin emitted groups still require every member
context to satisfy the pose-family bound (285–303).

The between-skin branch is no longer a catch-all empty-output pass. Before either
cover outcome, it requires at least one attributable MEASURED_CONTEXT window,
converged post-selection flange fits, exact whole-crop conservation, unverified
coverage semantics, interval accounting for the sheared population and the known
patch count, and nonempty source records. Every retained source record's raw
arc/flange support is checked against both original patch IDs and sheared IDs,
including records invalidated later by ownership (481–491). Thus a native import
failure, registration failure or arbitrary missing measurement cannot pass this
negative branch. A publication branch must satisfy the independent radius and
raw-support checks; an abstention branch requires the exact workflow, source,
record and cover-plan refusal reasons (503–512). This is meaningfully specific.

The defensible exclusion claim is no original patch/sheared ID retained in the
published source supports, plus conservative category counts consistent with
that construction. The code does not reconstruct every excluded row's complete
layer/category assignment, and should not be described as doing so.

### Remaining mathematical defect in the auxiliary no-cover witness

Lines 515–519 union flange column 0 and flange column 1 across all records,
then allow only one final swap against that union. The real cover planner
instead admits an independent role permutation for each member (149–175).
Sorting each record's two raw-ID lists does not guarantee corresponding physical
flanges always have the same column after subset cropping.

A small set counterexample makes the issue explicit: candidate flange sets
`{1,3}/{2,4}` cover member sets `{2}/{3}` by swapping that member's roles. Both
records can already be lexicographically ordered. The fixture's column unions
become `{1,2,3}/{2,3,4}`; neither candidate flange contains those unions, so it
would incorrectly certify no cover. The predicate also unions all records,
whereas the planner decides separately for each ownership component. These are
defects in the fixture's independent witness, not evidence that the actual
diagnosed abstention is wrong.

Root independently reports that the diagnosed case has one component and every
record misses 52–217 rows of its arc union. That gives a much smaller valid
witness, independent of flange permutations: assert the single component's
member set equals the source record set, compute that component's arc union,
and require a nonempty arc deficit for every candidate. Alternatively check
candidate containment per member with an independent permutation, separately
per component. No engine change, forced prediction or broad experiment is
needed. The current source's global-column-union predicate is not an exact
recomputation of the declared cover invariant.

### Air-side claim must stay local, not full-workflow rejection

The revised air-side case allows measured/emitted groups from out-of-bound
contexts, provided no attributable context feeds them (428–433). This is
explicitly reported as `predictions_from_unattributable_contexts`. It proves
refusal of the dent under the independently attributable contexts, not that the
whole workflow necessarily emits zero predictions for a deformed part. The
header's "without any prediction" (32) and unqualified emitted-group attribution
claim (42) overstate this branch; the between-skin function docstring's "EVERY
context window" (437) should say every attributable window. The old exact-count
comment (348–350) also remains stale. These are narrow claim corrections, not a
new blocking engine gate. No whole-workflow negative success should be reported
on this source contract alone.

Disposition: corrected interval constants, per-window attribution, source-record
raw-ID checks and the specific safe-abstention branch are sound improvements.
The auxiliary union witness needs the small correction above or an explicitly
separate validated arc-deficit witness; air-side claims must remain conditional
and local. Both points were sent promptly to root. No active-campaign pass,
final V11 approval, PRE or run authority is asserted.

## Final bounded oracle closure — 2026-09-14

**ACCEPTED in the requested source-review scope**, with the conditional claim
limits below. Opening and closing fixture hash was unchanged:
`8b0556ccfa2f152887a88505e0760f307c99f773c43fe02abcd811803204335a`.
The auxiliary source-only read of
`/Users/idantal/.claude/jobs/1519fdc2/tmp/oracle_offline_check.py` was pinned
unchanged at
`2a64f03960d00140cf5d9738459c908da8c61210537b905d9dcab49ec8e90f94`.
Neither file was executed/imported here; no dump, output, raw/Gold/reference,
attempt or cache was accessed.

The corrected fixture (513–521) now requires exactly one cover component, with
its member IDs equal to the complete source-record set, and at least two records.
It computes that component's arc union and requires every candidate record to
miss at least one union arc row. This is a sufficient, role-swap-invariant proof
of no existing whole-component cover: the planner requires a candidate's arc
support to equal the component arc union before testing any flange permutation
(planner V1, 149–170). A strict arc deficit fails that necessary condition.
The prior column-union/per-component defect is closed without introducing a new
flange-permutation oracle. The offline-check source mirrors the same component
and arc-deficit assertions against its designated retained evidence; this review
does not attest that script's execution result.

All preceding nonvacuity safeguards remain: specific measurement/ownership/cover
refusal reasons, nonempty source records, at least one attributable measured
window, original patch/sheared IDs absent from retained supports, and the
correct [90,152] interval. No arbitrary empty workflow can qualify this branch.

The accepted scientific claims are unchanged and deliberately narrow: real
full-bytes nonnominal positive measurement; local dent refusal on attributable
contexts; explicit between-skin support exclusion and a specifically justified
cover abstention when no member supplies all arc support. The air-side branch
still permits predictions exclusively from out-of-bound/unattributable contexts;
it does **not** establish guaranteed whole-workflow dent rejection. Some source
prose at 32, 42, 348–350 and 437 remains broader than that executable contract;
those words were not silently treated as corrected. They are nonblocking prose
qualifications under this explicit acceptance scope, not new geometry gates.

Evidence versions remain separate: root reports three completed real-geometry
case passes at fixture
`6ff4d1540d4e7528943540249d6f6fccebf1f48d4f315d9ea783b259d2966110`.
The present `8b0556cc...` change is the auxiliary oracle correction; no full
geometry rerun at those new bytes is asserted. Root/author own the separate
lightweight retained-evidence receipt and overall candidate acceptance. No
additional geometry campaign is needed solely for this oracle/prose closure.
This source acceptance is not PRE, physical scoring evidence, factory-floor
qualification or authorization to execute. Only this audit was appended.
