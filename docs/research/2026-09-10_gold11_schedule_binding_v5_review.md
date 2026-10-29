# Prospective V5 contact-runner schedule binding

Source-only integration review, not an exact freeze or permission to execute.
The V12/V13 engine sources may still change before their next freeze. No project
imports, tests, geometry, physical inputs or retained artifacts were accessed;
only this new review was written.

## Decision

**Keep the V3-custody-qualified contact scorer V2 unchanged as an inner
primitive, but add a separately versioned, independently qualified runner-level
schedule-binding envelope.** The current projector does not bind scheduling:
`_returned_configurations` checks only scan-plane configuration
(`scripts/gold11_contact_scoring_v2.py:240`), and
`project_gold11_contact_workflow_v2` has no expected schedule argument (`:489`).
A valid-looking result with unrequested schedule metadata is not rejected merely
because its schedule differs. Passing that result through the unchanged scorer
alone therefore does not qualify V5 schedule selection.

Do not add schedule fields to the qualified compact projection: `_projected_case`
requires its exact existing schema (`:583`). Its public paired scorer also
requires exactly one `v7` and one `composite_v1` request (`:710`). Thus a declared
V7/V1-schedule control versus composite/V2-schedule challenger can reuse it,
but measures a **combined candidate change**, not a schedule-only effect. A
same-scan-plane-mode schedule ablation, or direct schedule-aware scorer API,
requires a separately versioned scorer contract and new qualification. Never
mislabel the scan-plane modes to bypass that restriction.

## Exact construction and publication seams

Replace V4's V10/V11 split at
`scripts/run_gold11_mesh_development_v4.py:960` in a new V5 runner, not in V4.
Both arms must construct `MeshWorkflowPolicyV13` and call
`inspect_cad_guided_mesh_v13` with the same native V1 adapter, free-circle
measurement, connected contexts, two-plane lane enabled, input bytes and frozen
nonexperimental policies. Freeze independent per-arm scan-plane and schedule
declarations; do not derive expected declarations from returned output.

For schedule V1, pass `two_plane_schedule_mode="v1"` and
`two_plane_schedule_budgets=None` (the default path intentionally adds no new
schedule publication). For V2, pass `two_plane_schedule_mode="v2"` and the exact
`TwoPlaneScheduleBudgetsV2(**frozen_budget_dict)` instance, including inherited
and residual fields. No omitted/default-filled/mutated budget map is accepted as
the frozen request. `cad_guided_plane_initialization_v12.py:62` requires an
enabled pair lane for V2 and validates/owns the typed budget snapshot; `:205`
incorporates it into initialization configuration.

Bind the complete arm declaration and its canonical SHA into the V5 frozen
bundle, inference dispatch and STARTED/terminal metadata. The worker constructs
only from its frozen arm declaration and verifies the dispatch digest. Update
V4-equivalent `validate_leaf_binding` (`:880`), dispatch identity reconstruction
(`:1505`) and tree leaf schema together. An arm name alone must not silently
select a new scheduler under an old experiment identity.

For a completed V2 result, the independent expected value is
`{"mode":"v2","budgets":frozen_budget_dict}`. Validate exact keys, strict
integer types (not bool/float coercion) and exact values at every publication:

| Location in retained V13 JSON | Required binding |
| --- | --- |
| `policies.two_plane_schedule_configuration` | Exact frozen request |
| `registration.evidence.two_plane_schedule_configuration` | Same request |
| `registration.evidence.initialization.two_plane_schedule_configuration` | Same request |
| `registration.evidence.initialization.two_plane_schedule` | V2 schema, completed selected-policy evaluation, exact budget map and false acceptance/production/support-match authority claims |
| Every `registration_array_binding` entry | V2 mode, SHA of the canonical expected configuration, and the same two equation-bank digests as the schedule/breadth publication |

V13 currently publishes policy configuration at
`backend/cad_guided_mesh_offline_v13.py:142` and supplies the expected request to
the shared binder before measurement (`:218`). The shared V12 snapshot uses
sorted, compact, finite JSON (`cad_guided_mesh_offline_v12.py:151`). Its binder
(`:180`) independently reconstructs equation digests and validates the schedule;
V13 also checks the published initialization budget against its requested
budget. The runner should bind that published initialization budget to its own
frozen request as well, rather than trusting mutually consistent returned
copies. Do not normalize published scan coefficients again: the engine binder
explicitly avoids a second normalization when computing their canonical digest.

Require one binding entry for every declared measurement context, with the
existing context ledger intact; a valid first entry cannot excuse a missing or
malformed later one. Preserve the engine's conditional-context/no-accepted-pose
boundary. The runner's byte/configuration checks are not a replacement for the
engine's equation, raw-array or scheduling validation and must not claim to
independently reproduce those computations.

## Projection, scorer and parent wiring

Use a new pure script-level adapter; do not import backend initialization or
native geometry into the scorer leaf. Its inputs are exact retained result bytes,
the independently pinned result SHA, and frozen arm declarations. It performs
bounded strict JSON/configuration checks and calls the unchanged
`project_gold11_contact_workflow_v2` on the **same original bytes**. It must not
strip schedule fields and reserialize the workflow before projection.

Retain one versioned wrapper in the existing `SCORING_INPUT.json` slot, containing
the request/binding receipt and the untouched qualified contact projection only
when admitted. This keeps five files per successful arm and the planned
119-file campaign shape; no sixth per-arm artifact is needed. Use a distinct
outer schema, never extra keys masquerading as the inner projection schema.

Replace V4's `compact_result` call in `supervise_case` (`:1158`) with this adapter
after clean worker termination and exact result retention. Pin its result SHA
to retained-artifact and worker evidence, not an independently substituted
blob. Parent verification at `verify_parent_artifacts` (`:1476`) must reread the
exact `RESULT.json` bytes and reconstruct the entire wrapper using the same
frozen declarations, then compare its canonical bytes to retained
`SCORING_INPUT.json`. Two unrelated valid hashes do not prove this relationship.
For nonclean, partial or absent worker output, reconstruct the explicit
supervisor rejection instead of promoting retained JSON. Update the
`terminalize_case_records` (`:1221`) validator/fill path to the same outer schema.

Only after all 22 durable arm terminals may reference access and scoring begin.
Update `score_after_terminal` (`:1278`) to include explicit scan-plane requests
and schedule-bound wrappers in the scorer request. Update `scorer_worker`
(`:1262`) to validate the wrapper contract, unwrap only admitted qualified
projections, and call the unchanged `score_gold11_contact_workflows_v2` with its
existing argument schema. Keep the inner score unchanged; put schedule requests,
binding summaries and combined-change attribution in a distinct V5 outer report.
Update scorer leaf sources (`:825`), dispatch/request hashes, final report/schema
checks (`:1532`) and inventory/size contracts together. The new adapter and its
tests are new qualification scope; the inner scorer's old pass is not a pass
for this runner. Include actual package initializers in future test freezes.

## Fail-closed cases and required invented tests

- Reject an invalid frozen request before dispatch; do not count it as an
  engine failure. For a V1 request reject V2 configuration keys even when null,
  plus V2 schema/residual/breadth markers at all relevant locations, including
  policy and array-binding metadata.
- Completed V2 requires every binding above. Missing, changed, wrong-type,
  extra-key or all-copies-consistently-wrong configuration rejects the whole arm
  projection. Test each location, last context, mutated initialization limits,
  incorrect configuration SHA and divergent equation digests.
- An early explicit workflow refusal may precede configuration publication.
  Missing not-yet-reached evidence is not automatically malformed; any evidence
  actually present must still agree. It remains unscorable with no predictions,
  and must not receive a fabricated successful schedule-binding receipt.
- On runner schedule rejection, publish an explicit outer rejection with no
  admitted contact projection or successful count prefix. Feed the inner scorer
  its existing exact six-key supervisor `UNAVAILABLE` envelope, with a reason
  naming the runner schedule failure; do not forge a returned engine refusal or
  rewrite a qualified projector result. Preserve original result bytes for
  evidence and report the outer failure separately. All reference FNs remain.
- Test exact-byte/hash substitution, compact-wrapper substitution, cross-arm
  declaration swaps, stale dispatch/configuration digests, and a forged positive
  wrapper paired with genuine refused bytes. Verify parent reconstruction fails,
  not merely that either hash parses. Retain valid completed-no-carrier behavior.
- Test that the scorer leaf cannot import backend/native code or open inference
  inputs; a 21-terminal campaign cannot read references. Test the actual 22-arm
  request/report/inventory paths under fixed byte and time budgets, not only
  isolated tiny wrappers. Prove unchanged inner score output for admitted inputs
  and whole-case empty behavior for runner rejections.

No physical protocol, limits, gate denominator or execution is frozen here.
The source-only integration design does not establish physical F1 improvement,
historical comparability, measurement uncertainty or floor readiness.
