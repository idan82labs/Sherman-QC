# Independent source review: Gold11 V9 candidate — completion lane V1, mesh offline V16, projection V4, runner V9

Date: 2026-09-13. Non-author, read-only source review of the V9 candidate in
`/Users/idantal/ShermanQC/FullProject`. No test, runner mode or geometry job
was executed; no attempt directory, raw input, Gold/reference container,
private cache, `.env` or credential was opened. The only executed code was
`.venv/bin/python` snippets that tokenize/diff the checked-in text (recorded
below). Findings are bound to file:line of the hashed bytes in "Bindings".
This is not a PRE, a freeze, a physical claim, or a numeric-F1 statement.

## Decision

**P1 present (one): the V9 wiring cannot admit a completion-lane start to
measurement, and a completion-derived context refuses the whole case.**

- P1-1 — `backend/cad_guided_mesh_offline_v16.py:209-211,227-229` requires
  every selected measurement context to have a non-empty `hypothesis_indices`
  ledger bound to `initialization["hypotheses"]`; completion-lane start groups
  are created with `hypothesis_indices=[]` and only `completion_indices`
  (`backend/cad_guided_plane_initialization_v15.py:803-806`). A completion
  group that survives refinement and is SUPPORTED therefore raises
  `MISSING_SELECTED_START_SUPPORT_ORIGINS` inside the per-context binding loop
  (`v16.py:489-491`), which discards every context of the case, including
  main-lane contexts (`v16.py:492-501`, then `v16.py:517-519`). The intended
  success path of the new mechanism is unreachable, and the lane can regress
  cases that reached measurement contexts in V8.

Everything else reviewed is P2 or wording. The completion geometry, the
separate budgets, the unchanged main-lane text/counters/invariants, the V16
policy plumbing, projection V4 identity binding and the runner V9 identity /
config-delta / custody text all check out as claimed, with the accounting gaps
listed under P2.

## Bindings (SHA-256, bytes)

Hashed at review start and again at review end. Sixteen files were
byte-identical at both times. Two documents changed on disk during the review
(appended sections only; the sections relied upon are unchanged — see notes
N9); both hashes are given.

```text
ce4a61af90db1339ace0914cd359ea6e0afdf22dd5b9f038c1e2d9d7f1eeaba4   92740  backend/cad_guided_plane_initialization_v15.py
9e3ca41edd05bcc0b6fcdeb9ef0447dbacfe58df8c9458da20f9bb12c3d925bf   71025  backend/cad_guided_plane_initialization_v14.py  (matches finite-start audit snapshot)
441c618d4998edaca0d401dd22f14d1444f248750a807386fecef19741708769   36874  backend/cad_guided_mesh_offline_v16.py
c8f38f496fdd11148cec04a8b5ba0f1366cc548efca59a14b48451b5b5c57148   35461  backend/cad_guided_mesh_offline_v15.py
06869ff7bb8761eae976f294e733713c3f2d94a83ee0c8d756c90d50afccfa90   16189  scripts/gold11_schedule_projection_v4.py
763e0cd032521f237aaa999601f67e75644fe430b9837bedf1dc2c4d9be82d30   16058  scripts/gold11_schedule_projection_v3.py
f8a9b7f28af81e842f187f6ff76ae87b80b59809561300785064ff63c25027e8  176957  scripts/run_gold11_mesh_development_v9.py
0e9aa2f64f3fc14e0da137687e44c8cd0dc0712f9e8bd783726df4187b1360c5  172872  scripts/run_gold11_mesh_development_v8.py  (== SOURCE_BASIS_SHA)
9fd6c11e987cd260e0247154b901ebc67e600e0dba5766787a22b907ffbf2378   13584  tests/test_cad_guided_plane_initialization_v15_completion.py
6b36dba73a145435470f6854679805e5ef7070a2ff9fb61eda4cb12d8efb491c   14446  tests/test_cad_guided_plane_initialization_v15_registration.py
8507e7fbcd0870d05f7c62b1286b62b02332a5d297f12720821ffc26933f4be3   47961  tests/test_cad_guided_completion_v1_engine_integration_v1.py
2849158aad364c4282f309fb9fcf35ed9844b91c404ca2e38834f8dd9e853192   29774  tests/test_gold11_schedule_projection_v4.py
462619e1494e845ffcf3db68dfa1da6e1884fcf8682be41802fe0c78e0d238fe   13072  tests/test_gold11_mesh_development_v9_source.py
1f4d52492a904703bf84302c8c5d6bc53cfb22157b0a87d9dd6272caeb574522    6347  docs/research/2026-09-13_gold11_mesh_development_v9_delta_design.md
424e0d2d796fc5d4a7a3858bad56298bc3dee71993a407ecd9315245b395dd46    8387  docs/research/2026-09-13_gold11_mesh_development_v9_protocol.md
2dd922f7871d10e9718899ecb6719f57da6cdf72db5f89d8fa16ba807993a7f9   17869  docs/research/2026-09-13_fable51_v9_diagnosis.md  (at start; Part A read from this version)
2d01be35ab89074d3b023d91a86aa81c7eb489bcae3877bbd9f74b1cb0962ece   18650  docs/research/2026-09-13_fable51_v9_diagnosis.md  (at end; Part E table appended, lines 265-281)
3b4b4da990998adaf4e8dc6265436de91aba07ed617e149b49e2c987f8247823   20160  docs/research/2026-09-13_gold11_v8_finite_start_codex_audit.md
566fab207e2bf2b3abd233a460f5f1d5ac3abfcaf6356e44126d8d15a0848c50    9386  docs/research/2026-09-13_fable51_v8_codex_coordination.md  (at start)
5ef72697c1eb4011c7a8bc3a2dceb443c34f2364e80756bdfd293c326cb7b988   10030  docs/research/2026-09-13_fable51_v8_codex_coordination.md  (at end; 03:39 section appended, lines 157-167; 03:05 section unchanged)
```

Frozen modules consulted read-only for semantics (not in scope, not hashed
here): `backend/cad_guided_plane_search_v6.py` (`PlaneEquationV6` :21-28,
`_signed` :527-571, `_strata` :221-228), `backend/cad_guided_plane_initialization_v12.py`
(`_LIMITS`/budgets :32-59, `_publish_contexts` :1101-1120),
`backend/cad_guided_plane_initialization_v4.py` (`_pose` :207-214,
`_bound_support` :217-…, `_fail_prefilter` :384-394),
`backend/cad_guided_plane_initialization_v8.py` (`_bound_scan` :192-201),
`backend/cad_guided_face_box_screen_v1.py` (`decide_face_box_exclusion` :95-116),
`backend/cad_guided_report_envelope_v6.py` (:225-253),
`scripts/gold11_schedule_projection_v1.py` (:143-213).

## Findings by priority

### P1

**P1-1. Completion-origin start groups cannot be bound by mesh offline V16; a
SUPPORTED completion context refuses the whole case.**

Chain in source:

1. `v15.py:803-806` — a new completion group is
   `dict(exact_group_index=…, initialized_rotation=…, initialized_translation_mm=…,
   hypothesis_indices=[], refinement_disposition=…, lane="TRIPLET_COMPLETION",
   completion_indices=[])`; only `completion_indices` is filled (`:811`).
2. `v15.py:1006-1009` — the group can be scheduled for refinement and copied
   into `result["start_groups"]`; `v15.py:1317-1318` copies it into
   `evidence["candidates"]`; refinement, `_context_support` and
   `_publish_contexts` (`v12.py:1101-1120`) treat all candidates alike, so it
   can become `SUPPORTED_CONTEXT` and enter `measurement_context_indices`
   (`v15.py:1354-1357`, `:1104-1176`).
3. `v16.py:466-491` — every index in `measurement_context_indices` is bound
   with `_registration_binding(..., context_index=index)`.
4. `v16.py:205` requires `candidate["hypothesis_indices"] == group[...] ==
   all_groups[gi][...]` (passes: all `[]`), then `v16.py:209-211`
   `_require(type(origins) is list and origins and …, "MISSING_SELECTED_START_SUPPORT_ORIGINS")`
   fails for `origins == []`. Even if that were relaxed, `v16.py:227-229`
   requires `origins` to equal the main-lane rows with
   `exact_start_group_index == gi`, which is also `[]`; and `v16.py:212-226`
   only knows the lanes `TRIPLET`/`TWO_PLANE_CONTEXT`.
5. `v16.py:492-501` — the exception empties `measurement_context_indices`
   and `registration_array_binding` for the whole case and re-raises;
   `v16.py:517-519` publishes `WORKFLOW_EXCEPTION:ValueError`.

Consequences: (i) no completion can reach measurement, so the mechanism's
intended positive path is unreachable — contrary to the delta design
(`…v9_delta_design.md:65-67`, "still has to survive refinement, admission and
measurement") and absent from the protocol's expected negatives
(`…v9_protocol.md:95-102`); (ii) for a case with main-lane contexts, any
kept completion context (not collapsed as a pose-class duplicate by
`v15.py:1147-1151`) turns a V8-style admitted case into a whole-case refusal
— a regression class the runner's paired before/after would record as a
retained negative, not as a wiring defect. No test exercises a
completion-origin candidate through V16 (the integration fixture builds only
`hypothesis_indices=[index]` origins, `tests/test_cad_guided_completion_v1_engine_integration_v1.py:457-467`).

Minimal repair (V16 only, plus tests; no gate or threshold change):

- In `_registration_binding` after `v16.py:205`, also require
  `candidate.get("completion_indices", []) == group.get(...) == all_groups[gi].get(...)`.
- Replace `v16.py:209-229` with a two-ledger binding: `main = candidate["hypothesis_indices"]`,
  `completion = candidate.get("completion_indices", [])`; require
  `main or completion`, uniqueness of each; bind `main` exactly as today; for
  `completion` require `initialization["completion_lane"]["status"] == "COMPLETE"`
  and, for each `k`, the row `lane["hypotheses"][k]` to have
  `completion_index == k`, `proposal_lane == "TRIPLET_COMPLETION"`,
  `prefilter_status == "PASSED"`, three `correspondences` whose
  `scan_plane_id/cad_plane_id/normal_sign` equal `assigned_support`, all
  supports `PASSED`, `rotation`/`translation_mm` equal to the candidate's
  initialized pose, and `exact_start_group_index == gi`; require the ledger
  identities `main == [i for i,row in enumerate(hypotheses) if row.exact_start_group_index == gi and PASSED]`
  and `completion == [k for k,row in enumerate(lane["hypotheses"]) if … == gi and PASSED]`.
- Publish the origin lane(s) in the returned binding (`start_lanes`), so the
  projection/POST can count completion-origin contexts.
- Tests: a V16 plumbing case whose `_registration` fixture carries a
  completion-origin candidate (accept path) with tamper variants (missing
  lane ledger, wrong `completion_index`, non-PASSED support, pose mismatch,
  stale `exact_start_group_index`), and one actual end-to-end run through
  `register_measurement_contexts_v15` on a fixture that yields a real
  two-of-three parent (see T-1).
- Update the delta design/protocol wording once fixed.

### P2

**P2-1. A completion-lane failure is all-or-nothing for the entire
initialization, not for the lane.** `_complete_two_of_three` runs inside
`_prefilter` (`v15.py:1003`) under `_initialize`'s single `except`
(`v15.py:1059-1072`), which calls `_fail_prefilter` (`v4.py:384-394`: clears
`start_groups`, invalidates every group) and marks the initialization
unavailable; `register_whole_part_planes_v15` then refuses
(`v15.py:1294-1299`). Lane-internal fatal points that are not preflighted
against the lane's own inputs: `CAD_FACE_CACHE_BUDGET_EXCEEDED` /
`CAD_FACE_TRIANGLE_BUDGET_EXCEEDED` for `needed_faces` (`v15.py:459-461`),
face-build node/visit exhaustion (`v15.py:466-467,476`), the row/call/work
preflight itself (`v15.py:788-790`), `MISSING_PARENT_NORMAL_ALLOWANCE`
(`v15.py:715-717`), `INVALID_COMPLETION_CAD_NORMAL`/`_SCAN_NORMAL`
(`v15.py:701,710,724`). Under the V9 defaults the row/call preflight cannot
bind (≤ 4096×3 supports, ≤ 2 chunks each), but `needed_faces` can be far
larger than the main lane's `used_faces` (up to 64 candidate faces per
parent), so the face-cache/build caps are the realistic exposure. The
"additive" claim (`v15.py:3-6`, delta design `:46`) holds only on the
success path. Repair: either preflight `len(needed_faces)` and the summed
face triangles before `v15.py:792` and record a lane-level
`REFUSED_BY_LANE_BUDGET` status, or wrap the call at `v15.py:1003` so that on
any lane exception the lane is marked `FAILED` with the reason, `groups` is
truncated back to `main_group_count` (and its `lookup` keys removed),
`completion_passing_start_group_count=0`, and the main-lane result is kept —
never a partial set of completion groups. Test both branches.

**P2-2. Completion groups can displace main-lane groups from refinement,
and the omission ledger does not separate the lanes.** `v15.py:1006` runs
`_search._strata(len(groups), 48)` over the concatenated main+completion
list (`_strata`: `search_v6.py:221-228`). Whenever `M + C > 48`, the strata
over the combined list differ from V14's strata over the main list alone, so
main-lane starts that V14 would have refined can be omitted in favour of
completion starts. `omitted_passing_start_group_count` and
`surviving_start_group_count` (`v15.py:1010-1011`) are not split by lane.
For the twelve V8 no-passing-start slots `M = 0`, so no displacement there;
the exposure is on slots with main-lane starts. Repair: schedule the main
lane exactly as V14 (`_strata(M, 48)`) and give completion groups the
residual capacity (`_strata(C, 48 - len(main))`), or at minimum publish
`main_refined_group_count`, `completion_refined_group_count` and per-lane
omitted counts.

**P2-3. Lane ledger identity/accounting gaps (unresolved ambiguity is
counted but not always identifiable).**
- Omitted parents (`v15.py:682-683`) are counted, but their
  `hypothesis_index` values are not published; the prefix-by-index quota is
  an ordinal selection (the main lane forbids prefix quotas for itself,
  `v15.py:851-855`). Publish `omitted_parent_hypothesis_indices`.
- A parent whose candidates were all dropped by `max_completions_total`
  is labelled `SCHEDULED` with `scheduled_count == 0` (`v15.py:746-749,768`);
  per-parent `omitted_count` mixes the per-parent cap (`:745`) and the total
  cap (`:748`). Add `OMITTED_BY_TOTAL_COMPLETION_BUDGET` and split the two.
- `normal_incompatible_count` (`v15.py:725-727`) folds antiparallel faces
  (`m_j·n3w ≤ -cos(allowance)`, an orientation-only exclusion inherited from
  the parent's Wahba sign) together with genuinely tilted faces. Plane
  coincidence is orientation-free (see derivation), so publish a separate
  `antiparallel_normal_count`; the sibling parent with the opposite third
  sign generally cannot pass the Wahba gate for an orthogonal triplet
  (`search_v6.py:545-561`), so opposite-orientation coplanar/parallel faces
  are never completed — this should be visible in the ledger.

**P2-4. Report-byte exposure of the lane ledger is not preflighted or
tested.** `lane["hypotheses"]` (≤ 4096 rows × 3 support records with ~30
fields), `parents` and per-face `face_index_evidence` are added to the
initialization report under the unchanged 96 MiB cap
(`v15.py:1073`; runner `initialization_budgets.max_report_bytes`
`run_gold11_mesh_development_v9.py:289`), then the 128 MiB workflow and leaf
caps (`v16.py:252-270`, runner `:207`). Overflow returns the refusal
envelope `INITIALIZATION_REPORT_BUDGET_EXCEEDED`
(`cad_guided_report_envelope_v6.py:249-250`) and refuses the case, again
including main-lane contexts. V8 report sizes were not inspected (attempt
data out of scope), so the magnitude is unknown. Repair: bound the published
per-row detail of the lane ledger (keep per-support counts and coverage, drop
per-row float maxima for rejected completions) or declare a lane sub-budget
with an explicit truncation status; add a test at a small `max_report_bytes`
with a populated lane.

**P2-5. The completion tests do not prove the lane inside the real prefilter
with a real two-of-three parent.** `_parent_from`
(`tests/…_v15_completion.py:131-149`) fabricates the parent by relabelling a
passing triplet and shifting its translation by ±10 mm along the x=0/y=0
intersection. Under the fixture geometry (`:52-75`), at the +10 shift the
z=0 scan sheet (x 2–14, y 3–15) lies inside face 41's box (x 1–15, y 2–16 at
z=10), so the real prefilter would have PASSED support 41, while supports
17/29 (axial −6..6 shifted to 4..16, CAD axial extent ±14) would have been
box-excluded — the fabricated parent is not a faithful "two PASSED, third
outside" row, and `completions[wrong_face]["prefilter_status"] == "REJECTED"`
(`:171`) holds because the passing pair slides off its faces, not because
the third face is wrong. The recovery itself is real: face 37 at |λ| = 10
passes the actual `_FaceQueries` box screen and finite query on all three
supports (`v15.py:576-643,792-795`). The ledger test (`:117-128`) does not
assert `parent_count ≥ 1`, so the in-`_prefilter` path with a
search-produced parent is unproven (on this fixture the real search cannot
produce one). Repair: see T-1.

### Notes (wording / minor)

- N1 `v15.py:160`: reason string `INITIALIZATION_V14_IMPLEMENTATION_REQUIRED`
  retained in V15 (the compared constant is V15's; label only).
- N2 The replaced face is always re-enumerated (`v15.py:720-722` skips only
  the two passing faces) at `|λ| = |δ_old/(m·u)|`, which per the diagnosis is
  0.0–0.3 mm, so it is always the first scheduled candidate and a predictable
  re-rejection of the parent pose consuming one per-parent slot. Consider
  skipping the exact replaced face or tagging `replaced_face_retested`.
- N3 `min_axis_projection` (`v15.py:728-731`): `m_j ⊥ u` ⟺ `m_j ∈ span(m1,m2)`
  ⟺ a rank-deficient triplet, which the parent cannot be (search
  `min_normal_condition_ratio` 0.02 ≫ 1e-3), so the branch is a numeric
  guard, not the geometric class the docstring (`v15.py:91-92`) suggests.
- N4 `v15.py:711` uses `witness_offset_from_anchor_mm` unnormalised with a
  re-normalised `n3`; consistent because `PlaneEquationV6` requires
  `| |n| − 1 | ≤ 1e-8` (`search_v6.py:97`).
- N5 Runner: `SOURCE_PROFILE["source_delta"]` (`v9.py:56-61`) lists only
  identity/engine/projection strings and the inherited V7→V8 items; the
  measurement-policy, completion and layer config deltas are declared only
  in `CONFIG_DELTA_FROM_V8`/`MEASUREMENT_POLICY_DECLARATION` (fine for
  custody, but the profile text under-describes V9).
  `STAGE_TELEMETRY_STATUS = "UNCHANGED_SAMPLER_WIRING_V8_RUNNER_PATH_NAMED"`
  (`v9.py:122`) now names the V9 path.
- N6 `scripts/gold11_schedule_projection_v4.py:3-4` docstring duplicates the
  "V3 text with …" sentence; `tests/test_gold11_schedule_projection_v4.py:539`
  test name still says `v3_identities`.
- N7 `v15.py:1000-1001` vs `:811-812`: main-lane rows record
  `exact_start_group_index` on the hypothesis; completion rows record it on
  the completion row — consistent, but the two ledgers are only reconcilable
  after the P1 repair.
- N8 Delta design `:56` "(11: …)" matches the 11 parametrised cases in the
  completion test file.
- N9 The two documents that changed on disk during the review only gained
  appended sections (diagnosis Part E NOT_PLANAR-share table, coordination
  03:39 POST-correction note); Part A and the 03:05 section were read from
  the earlier bytes and are unchanged in the later bytes.

## Check 1 — V15 versus frozen V14 (textual)

Unified diff of `v14.py` → `v15.py` contains exactly these hunks; nothing
else differs:

1. Module docstring: new lines `v15.py:1-7`; the V14 text follows verbatim.
2. `from dataclasses import asdict, dataclass` (`:33`).
3. Identities `:65-67`: `INITIALIZATION_IMPLEMENTATION` changed to
   `…_V15_CONTEXT_V4_COMPLETION_V1`; new `COMPLETION_IMPLEMENTATION`,
   `COMPLETION_LANE`.
4. New `CompletionLanePolicyV1` (`:80-115`) and `_completion_configuration`
   (`:213-223`).
5. `_configuration`: two added kwargs (`:230-231`) and two added lines
   (`:240-241`).
6. `_base`: two added keys (`:252-253`).
7. New defs `:556-817` (`_unit_or_none`, `_completion_statistics`,
   `_evaluate_completion_supports`, `_fresh_support`, `_complete_two_of_three`).
8. `_prefilter`: the single insertion `:1002-1005` (`main_group_count`, the
   lane call, `stats.update` adding the two new keys).
9. Public entry renames `prefilter_plane_initializations_v15` (`:1179`),
   `register_whole_part_planes_v15` (`:1214`),
   `register_measurement_contexts_v15` (`:1399-1410`), the two new kwargs
   threaded through (`:1189,1200,1225,1268`), and two new evidence keys
   (`:1270-1271`).

Verbatim (by diff): aliases `:40-63`, `_canonical`, `_context_configuration*`,
`_context_publication`, `_bind_context_publication`,
`_frozen_configuration_v4`, `_validate_two_plane_context_configuration`,
`_two_plane_records` (`:276-450`), `_FaceQueries` (`:453-553`), the whole
`_prefilter` body before and after the insertion (`:820-1001,1006-1025`,
gates, counters and the two invariants `:1012-1022`), `_prepare`,
`_initialize`, `_context_support`, `_schedule_measurement_contexts`.
Additionally, a token-level comparison (comments/whitespace stripped) of the
main-lane per-support block `v15.py:914-991` against
`_evaluate_completion_supports` `:580-642` gives 697 = 697 tokens, identity
ratio 1.0; the `_fresh_support` key set (`:646-657`) equals the main-lane
support dict key set (`:885-900`).

## Check 2 — Completion geometry (independent derivation)

Conventions (verified): pose acts as `x_cad = R x_scan + t`, rows as
`x @ R.T + t` (`v15.py:586`); scan plane `k`: `n_k·(x − a_k) = 0` with
`a_k = anchor_k + offset_k·n_k` (`PlaneEquationV6`, `search_v6.py:23`;
`v15.py:711`); CAD face `j`: `m_j·(x − c_j) = 0` with `c_j` the area
centroid, declared on-plane by the same convention the main search uses
(`v15.py:317-318` builds CAD equations with anchor = centroid, offset 0).

Transformed scan plane `k`: `(R n_k)·(x − (R a_k + t)) = 0`.

Let `u = (m1 × m2)/|m1 × m2|` (`v15.py:699-702`; floor `min_pair_sine`),
`t' = t + λu`.

- Passing planes stay coincident: for `i ∈ {1,2}`, `m_i·u = 0`, so
  `m_i·(R a_i + t' − c_i) = m_i·(R a_i + t − c_i)` for every `λ`; the
  transformed scan plane `i` keeps exactly its (already accepted) offset from
  CAD plane `i`. What changes is the position of the finite scan patch inside
  its plane (it slides by `λ` along `u`), which is why the two passing
  supports must be re-screened — the code does so (`:580` iterates all three
  `assigned_support`, built from all three correspondences at `:778-781`).
- Third plane: `δ_j(λ) = m_j·(R a3 + t + λu − c_j) = δ_j + λ (m_j·u)`;
  `δ_j(λ) = 0 ⟺ λ_j = −δ_j/(m_j·u)` when `m_j·u ≠ 0` — exactly
  `v15.py:732-733` (`delta = mj@(a3w − c_j)`, `shift = −delta/mu`, with
  `a3w = R a3 + t` at `:713`). With `R n3` parallel to `m_j` within the
  allowance, the shifted third scan plane lies on face `j`'s plane up to the
  residual tilt. If `m_j·u = 0` the offset is `λ`-invariant (unresolvable
  along `u`), counted, not inferred (`:729-731`).
- Sign of `n3w`: the parent's witness gate (`search_v6.py:545-552`) measures
  `angle(R n_k, s_k m_k)`, which equals `angle(s_k R n_k, m_k)` for
  `s_k = ±1`. `n3w = s3·(R n3)` (`:712`) and the test
  `m_j·n3w ≥ cos(allowance_3)` (`:718,725`) is therefore the parent's own
  row-3 gate with `m_old` replaced by `m_j`. Consistent. (Plane coincidence
  itself is orientation-free — hence P2-3's antiparallel remark.)
- Threshold: `allowances[failed_position]` (`:714-718`) with
  `normal_alignment_allowances_radians = scan cone + CAD cone + extra`
  recorded in correspondence order (`search_v6.py:530-531,538-539,558`);
  `assigned_support` is appended in `correspondences` order (`:870-901`), so
  `failed_position` indexes the same row. `min(π/2, ·)` keeps `cos ≥ 0`.
- Branches: axis-parallel `:728-731`; shift limit `:734-736` (finite and
  `≤ max_shift_mm`); ordering `candidates.sort()` on `(|λ|, face_id, λ)`
  `:737-738`; dedup by exact `rotation.tobytes()+translation.tobytes()`
  against the shared `lookup` of the main lane `:800-812` (same key
  construction as `:993-999`; `_pose` round-trips the list-stored floats
  exactly, `v4.py:207-214`).
- Admission only after the full box screen and finite query on all three
  supports: `:794` (`_evaluate_completion_supports`), whose per-support text
  is token-identical to the frozen block (Check 1), with the same
  `policy.max_correspondence_distance_mm` and `policy.min_scan_coverage`
  (`:589,596,635`). Nothing is admitted on plane arithmetic alone.

## Check 3 — Budgets and custody

- Separate budget object: `_completion_configuration` (`:213-223`) requires
  `PlaneInitializationBudgetsV12`, validates every field against `_LIMITS`
  (`v12.py:32-39`), defaults to a copy of the base budgets. Runner
  `CONFIG["completion_budgets"]` (`v9.py:247-253` by diff offset; literal
  equal to `initialization_budgets` `v9.py:283-289`; the source test pins
  equality) is constructed at `v9.py:1449-1450` with the same
  `max_certificate_pairs` cap as the initialization budgets.
- Preflight before any query: planned rows/calls are accumulated while
  binding supports (`:777-785`) and checked against
  `max_prefilter_query_rows/calls/work_units` (`:788-790`) before the lane's
  `_FaceQueries` is built (`:792`) and before any `screen`/`__call__`.
- Separate `_FaceQueries` and statistics: `shim = dict(statistics=lane["statistics"], face_index_evidence=lane[...])`
  (`:791`); `_FaceQueries` reads/writes only `result["statistics"]` and
  `result["face_index_evidence"]` of the object it is given (`:458,471,475,
  489-490,493-504,526-527,531-540,547-553`), so all box/query/build charges
  land in `lane["statistics"]` (`:564-573` carries every key the class
  touches). Main `stats` is never written by the lane; the only main-lane
  change is the two additive keys at `:1004-1005`.
- Face/scan binding: `_old._bound_support(context, cad_patch=face)` for every
  needed face (`:786-787`, `v4.py:217-…`), `_bound_scan(context, patch)` per
  correspondence (`:780`, `v8.py:192-201`).
- Omission/unresolved counters: `:681-684` (parents), `:739-748,769-770`
  (candidates), `:740-742` (axis-parallel/incompatible/shift-limit),
  `:704-706` (parallel passing pair), plus the lane-level denominator
  invariant `:813-816`.
- Main-lane invariants `:1012-1022` reference only main counters and are
  unaffected; `hypotheses_evaluated == scheduled_hypothesis_count` holds
  because completion rows live in `lane["hypotheses"]`, not
  `result["hypotheses"]`. `exact_passing_start_group_count = len(groups)`
  (`:1010`) is now main + completion with the split published (`:1004-1005`).
  `max_refined_start_groups` strata: mechanism unchanged, population changed
  — see P2-2. Lane failure: all-or-nothing for the initialization — P2-1.

## Check 4 — Coplanar and parallel alternatives

- Enumeration is over every CAD patch except the two passing faces
  (`:720-722`); coplanar faces (same `m`, `λ ≈ λ_old ≈ 0`), parallel faces at
  other offsets, and the replaced face itself are all candidates (N2).
- No context-lane slot, midpoint or endpoint is consulted: parents are
  restricted to `proposal_lane == "TRIPLET"` (`:677`), and nothing is
  inferred from `ABSENT_CONTEXT_SLOT`/`AXIAL_SLOT_ABSENT_NOT_GEOMETRIC_EXCLUSION`
  rows. Compatible with the 03:05 constraint 4.
- Ambiguity retention: every passing completion becomes its own exact start
  group unless it is byte-identical in pose to an existing group
  (`:800-812`); several passing completions for one parent are retained as
  separate groups, never collapsed. Downstream, the unchanged V14
  measurement-context schedule may still collapse near-identical refined
  poses (`:1147-1151`), which is the pre-existing mechanism, not the lane.
- Orientation: faces antiparallel to the parent's signed third normal are
  excluded and counted with tilted faces (P2-3).

## Check 5 — Wiring (V16 vs V15; projection V4 vs V3)

V15 → V16 diff, exhaustively: docstring; imports (`_initialize_v15`,
`LayerSelectionPolicyV1` `v16.py:20-21`); identities `:34-38`
(`WORKFLOW_IMPLEMENTATION`, `KERNEL_IMPLEMENTATION`,
`COMPLETION_IMPLEMENTATION`, `REGISTRATION_IMPLEMENTATION` alias,
`SUCCESSOR_COMPARATOR`); `MeshWorkflowPolicyV16` rename; alias renames in
`_registration_binding` (`:84-87,178`); `inspect_cad_guided_mesh_v16` rename
and three new kwargs (`:273-281`); output identity fields (`:292-296`) and
reason string (`:302`); policy type checks (`:308,331-333` layer policy
validated/rebuilt); `_configuration` call passes completion policy/budgets
(`:351-352`); `output["policies"]` gains `completion_policy`,
`completion_budgets`, `layer_policy`, `kernel_implementation`,
`completion_implementation` (`:373-375`); `MAX_CONTEXT_WINDOWS/MAX_CROP_ROW_VISITS`
from context measurement V2 (`:434`); schedule label from V15 (`:444`);
`register_measurement_contexts_v15` call with completion policy/budgets
(`:446-459`); support cover V3 import/call with `layer_policy` and validator
V3 (`:506-512`). Nothing else differs — in particular `_registration_binding`
(`:81-249`) is the V15 text, which is the root of P1-1.

Projection V4 vs V3: docstring, `SCHEMA` V4, the two identity constants
(`:26-27`), two reason strings (`:78,104`), and the two function renames
(`:190,231,242`). The new V15/V16 publication keys (`completion_lane`,
`completion_implementation`, `completion_policy`, `completion_budgets`,
`layer_policy`, `kernel_implementation`) pass through V4's key filters
(`:98-101`: only `two_plane_context*`/`initialization_implementation` are
inspected; array bindings `:168-176`) and V1's (`projection_v1.py:156` only
forbids schedule-binding keys; `:169-175` compares
`initialization_budgets` only). Binding is by V16/V15 identity strings only.

## Check 6 — Runner V9 versus V8

- `SOURCE_BASIS = "scripts/run_gold11_mesh_development_v8.py"`,
  `SOURCE_BASIS_SHA = 0e9aa2f6…` (diff lines 26-30 → `v9.py:52-53`);
  the checked-in V8 file hashes to that value (Bindings). The freeze refuses
  a changed basis (`V9_FROZEN_V8_SOURCE_BASIS_CHANGED`, `v9.py:979,1031`).
- `SOURCES`: the V8 entry is replaced by the V9 runner; added: V7 runner
  (preservation oracle), kernel V2, context measurement V2, support cover
  V3, initializer V15, mesh offline V16, projection V4, the V9 delta design
  (`v9.py:66-…`, diff lines 43-62). The source test checks existence,
  uniqueness and that the imported `backend.*` closure (including the lazy
  imports) is a subset of the declared set
  (`tests/test_gold11_mesh_development_v9_source.py:113-137`). Projection V3
  is retained in `SOURCES` (frozen preservation) but removed from
  `SCORING_SOURCES` in favour of V4 (`v9.py:360-364`; test `:78-79`).
- CONFIG differs from V8 only in `measurement.max_fit_residual_mm` 0.05→0.03,
  `measurement.plane_exclusion_mm` 0.075→0.04, and the new `layer_policy`,
  `completion_policy`, `completion_budgets` entries (diff lines 169-185; the
  source test `:87-110` asserts wire-equality of everything else against the
  V8 AST literals, and equality of `LIMITS`, `CERTIFICATE_CAPS`, seeded /
  composite / schedule-V2 budgets and the two-plane declaration).
  `CONFIG_DELTA_FROM_V8` (diff 134-155) and `MEASUREMENT_POLICY_DECLARATION`
  (diff 156-162; `chosen_from_gold=False`, `tuned_to_case_identities=False`)
  are written into the frozen bundle (diff 283-286) and re-verified on every
  `verify_bundle` (diff 297-301) and echoed into the enriched report
  (diff 396-401).
- Custody/supervision/limits/scorer/terminalization: every remaining hunk is
  an identity-string substitution — error codes `V8_*`→`V9_*` (diff
  208-226, 435-460), `LEAF_SCHEMA`/`RUNNER_SOURCE` (diff 318-321), private
  cache prefixes `gold11-v9-*` (diff 329-330, 468-469), private scoring
  namespace `_gold11_v9_exact_scoring` (diff 242-243), stage-sampler module
  name (diff 374-375), custody/tree schemas `…_V9` (diff 412-413, 430-431),
  `experiment_schema`/arm identities (diff 392-404, 421-422),
  `runner_command` path (diff 383-384), projection module/functions and
  wrapper schema V4 (diff 202-203, 251-252, 260-261, 269-270). No
  supervision, limit, scorer, scoring-transport or terminalization logic
  changed. The V8 text itself is accepted from the independent V8 POST, not
  re-audited here.
- Identities are mutually consistent: `RUN` attempt path
  `gold11_mesh_development_v9_attempt`, `PROTOCOL` V9, bundle schema
  `GOLD11_MESH_DEVELOPMENT_V9`, `ONE_USE_PARENT_BOUND_LEAF_V9`,
  `ARM_DEFINITIONS`/`COMPARATOR`/report identities all name
  `CAD_GUIDED_MESH_OFFLINE_V16_CONTEXT_V4_COMPLETION_V1_KERNEL_V2` and
  `CAD_GUIDED_PLANE_INITIALIZATION_V15_CONTEXT_V4_COMPLETION_V1`, matching
  `v16.py:34`, `v15.py:65`, `projection_v4.py:26-27` and both tests.
- Engine wiring: `workflow_entry_and_policies` (`v9.py:1401-1466`) builds
  `CompletionLanePolicyV1(**CONFIG["completion_policy"])`,
  `PlaneInitializationBudgetsV12(**CONFIG["completion_budgets"], max_certificate_pairs=…)`,
  `LayerSelectionPolicyV1(**CONFIG["layer_policy"])` (`:1448-1451`) and
  returns `inspect_cad_guided_mesh_v16`; the leaf passes them as kwargs
  (`:1504,1539-1544`). It reads only `CONFIG`/declarations — no case token,
  part key, path or Gold enters (comment `:1531`). `CATALOG` is read only in
  freeze (`:987`, hash binding) and in the post-terminal scoring path
  (`:1728,1943`), unchanged from V8.

## Check 7 — Tests

Proven by the checked-in tests (as written; not executed here):

- Recovery of the correct third face for both shift signs through the real
  `_FaceQueries` box screen and finite query on all three supports, with the
  correct λ magnitude, group creation, `exact_start_group_index`, lane
  denominator invariants and face-index evidence
  (`…_v15_completion.py:152-178`) — but from a fabricated parent (P2-5).
- Disabled policy, per-parent omission, parent omission (`:181-198`);
  parallel passing pair (`:201-210`); axis-parallel candidates counted
  without any query (`:213-237`); invalid policy fields (`:240-244`);
  bounded ledger and `exact == main + completion` on the fixture (`:117-128`).
- V15 registration end-to-end on the two-sheet fixture and the measurement
  context cap (`…_v15_registration.py:50-80,202-226`) — a fixture with no
  completion parents.
- Textual verbatim-ness of `_prefilter`/`_context_support`/`_FaceQueries`
  with exactly one completion insertion (`…engine_integration_v1.py:184-196`),
  and the V16 byte-workflow plumbing with mocked boundaries — with main-lane
  origins only (`:457-467`).
- Projection V4: V3 suite with V16/V15 identities; tolerance of the V14/V15
  schedule keys (`…projection_v4.py:546-557`).
- Runner V9: identities, config deltas against V8 AST literals, closure,
  frozen arm requests, real constructor bindings (types of completion /
  layer policies and budgets, measurement δ/e), inert freeze/verify round
  trip binding the V8 basis (`…v9_source.py:53-217`).

## Test gaps

- T-1 (would have caught P1-1): no test carries a completion-origin start
  group through refinement and V16 `_registration_binding`. Needed: (a) a
  V16 plumbing case with a `lane="TRIPLET_COMPLETION"` candidate
  (`hypothesis_indices=[]`, `completion_indices=[k]`, matching
  `initialization["completion_lane"]["hypotheses"][k]`), accept and tamper
  variants; (b) an actual `register_measurement_contexts_v15` run on a
  fixture whose real search yields a two-of-three parent (e.g. CAD x=0/y=0
  faces with axial extent beyond the shift, ±24, and face 41 laterally
  offset so the third support is bound-excluded at the parent pose), asserting
  `completion_lane.parent_count ≥ 1`, recovery inside `_prefilter`, and a
  SUPPORTED completion context bound by V16.
- T-2 Coplanar wrong face (a second finite face on the same plane as the
  correct face, λ ≈ 0) — the diagnosis's dominant pattern (faces 125, 12/10)
  is untested; the fixture only has a parallel face at z = 10.
- T-3 `max_completions_total` omission, parent status under the total cap,
  `shift_limit_rejected_count`, `normal_incompatible_count` (tilted and
  antiparallel faces), and ordering ties by face id.
- T-4 Budget refusal paths and their consequence: lane preflight
  (`COMPLETION_PREFILTER_PREFLIGHT_BUDGET_EXCEEDED`), lane `_FaceQueries`
  cache/triangle/build caps, and the fact that main-lane groups are
  invalidated (P2-1).
- T-5 Dedup with the main lane (`duplicate_pose_count`, a completion whose
  pose equals a main-lane passing pose; mixed group with both ledgers).
- T-6 Report-byte budget with a populated lane (P2-4).
- T-7 Strata displacement when `M + C > max_refined_start_groups` (P2-2).
- T-8 Projection V4 tolerance of the new V15/V16 keys (`completion_lane`,
  `completion_policy`, `completion_budgets`, `layer_policy`,
  `kernel_implementation`, `completion_implementation`) — verified by reading,
  not pinned by a test.
- T-9 Runner: nothing exercises `engine_call_with_stage_sampler` with the
  V16 kwargs beyond constructor types (acceptable for a source test, noted).

## What was not checked

- Nothing was executed except the tokenize/diff snippets; no test outcome is
  asserted here.
- Measurement kernel V2, context measurement V2 and support cover V3
  internals (sign conventions, layer selection) — separate reviewers.
- Noisy-plane normal orientation semantics (whether scan normals are
  outward-oriented), which decides whether the inherited sign exclusion in
  P2-3 is physically justified.
- `_v12._scan_plane_binding`, `_previous._registration_binding`,
  `gold11_contact_scoring_v2` internals; the retained V8 runner text outside
  the diff (accepted from the independent V8 POST).
- Actual report sizes, parent counts, candidate-face counts or any V8/V9
  attempt data; no raw scan/CAD/Gold/cache/attempt directory was opened.
- No numeric F1, recovery rate, physical accuracy or calibration claim is
  made or implied; whether any completion passes the real prefilter on the
  cohort remains unobserved.
