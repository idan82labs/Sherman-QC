# V11 two-plane integration: exact focused PRE audit

Date: 2026-09-10. Reviewer: independent AI agent
`production_architecture_review`; not the author of the V11 runtime, the two
focused tests, this integration protocol, or the harness additions. The reviewer
previously authored the proposal-design narrative and independently audited the
two pure helper gates; that history is not a claim of independent design origin.

Decision: **GREEN_FOR_FIRST_ONCE_FOCUSED_V11_INTEGRATION_GATES**.
Blocking findings: none in the bounded release below. This is a prospective
source-only decision, not a passing result or human engineering approval.

## Exact release and stop conditions

Release only these two commands, serially, once each, under the same unchanged
46-file union below:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite plane_initialization_v11
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite mesh_offline_v11
```

Require independent retained-only POST-GREEN for the first gate before advancing
to the second. Expected outcomes are respectively **41 passes / zero skips** and
**85 passes / zero skips**. Each allowance is 60 seconds monitored wall time,
45 seconds worker CPU, 1,073,741,824 bytes sampled group RSS, and 1,048,576 bytes
captured output. No source, test, protocol or harness mutation between these
gates. A failure, skip, missing receipt, budget stop, or source drift is terminal;
preserve it and do not retry under this PRE or change its started limits.

Neither `mesh_physical_config_constructed_v11_source` nor
`mesh_physical_config_constructed_v11` is released by this document. Their route
definitions happen to share this harness file, but their test, full closure,
artifact budgets and separate protocol have not been admitted by this PRE.
No raw, physical, retained prediction, reference/Gold, native, runner, scorer,
server, or production execution is released.

## Review performed and prerequisites

Read the complete 779-line V11 initializer, 322-line V11 workflow, 448-line
initializer test, 299-line workflow test, and complete integration protocol.
Read the exact harness delta against the preserved predecessor and its selected
budget, namespace, test selection, environment, retention and completion paths.
Reviewed the actually used predecessor fixture/oracle and mock-loader seams.
Parsed the exact source imports and parameterizations without importing or
executing any target module. Expanded only the relevant literal harness binding
expressions with a restricted AST reader and hashed the declared source files.
No test collection, test execution, native call, inference, or original data
access occurred during this audit.

The two pure prerequisites are already independently retained-POST-GREEN and
remain terminal; no rerun is needed:

| Prerequisite | Released evidence | Evidence SHA256 |
| --- | --- | --- |
| Context helper, 110 passes / no skips | `docs/research/2026-09-10_two_plane_context_v1_post_run_audit.md` | `acbb33edaa9210cd4a5e37978f4ab40a3856f5c8099b941c409d46c888523198` |
| Pair scheduler, 68 passes / no skips | `docs/research/2026-09-10_two_plane_schedule_v1_post_run_audit.md` | `51fe47bc57d7d531c4e0f71261039b56e8bfc546264002b253cd51f8c3e1ff1d` |

Their current helper/scheduler/search bytes match the qualified hashes in the
closure below. This PRE relies on those completed independent helper-math audits;
it does not relabel their pose seeds as finite-support or accepted-registration
evidence. The original helper leaves axial ambiguity and incomplete context
explicit; the scheduler's finite declared coverage is not exhaustive search.

The prior harness is preserved at
`docs/research/source_snapshots/two_plane_v11_pre_24480e4b/check_cad_guided_measurement.py`,
SHA256 `24480e4b18e00a08a9e6efecd654041965c6a2c5a0f3ac10cc8f9f684b741171`.
The inspected delta adds the four V11 route selections, two source binding
groups, appropriate existing budget branches, namespace admission, native-only
opt-in/artifact handling and receipt retention. It does not rewrite the existing
supervisor. Only the two focused branches are admitted here.

## Critical integration findings

- Owned preparation remains `_prepare_triplets = _previous._prepare` and
  `_bound_scan = _previous._bound_scan`: the existing V8 scan/CAD extraction and
  triplet search run before the additive pair lane. New pair preparation binds
  the owned full raw support and complete CAD face, helper frames, explicit unit
  assumption, zero local origins, all five array digests, ordered assignments,
  sign-family identities, all eight slot identities and their candidate poses.
  No third support or nominal CAD bend measurement is fabricated.
- Joint preflight counts every retained triplet record and reserves the complete
  unsigned-ID namespace. Each ordered pair consumes 32 sign/slot records and one
  unsigned identity, with joint 8192/1024 ceilings before helper dispatch.
  Rejected signs, absent slots and duplicate poses do not refund these budgets.
  Triplets can starve the pair lane; no stronger fairness or true-pose coverage
  claim is made. Explicitly disabled pair mode bypasses additive reservation
  admission and retains the old eligible-proposed unsigned prefilter rule.
- Source AST comparison confirms V11 `_LIMITS` and the entire `_FaceQueries`
  class are identical to frozen V10. All proposed pair/triplet origins require
  exactly two/three distinct bound supports. The full eager CAD face index is
  retained; all support rows use the same represented transformed buffer for
  boxes and any subsequent finite query. Complete-support upper bounds can only
  reject. Otherwise every row is queried, coverage uses the existing directed
  lower fraction bound, and all assigned supports must pass. The finite hole
  test prevents a loose box from becoming positive surface evidence.
- Both lanes share one exact-pose grouping and at most 48 stratified starts;
  all passing origin IDs remain in selected and unselected group ledgers.
  Existing V3 refinement, finite diagnostics, observability and competitor
  selection remain the downstream acceptance mechanism. Passing two support
  checks is not accepted registration and does not resolve the remaining axial
  ambiguity by itself.
- Fatal initialization invalidates every usable start, clears selected/numerical/
  box completion and preserves known hypotheses plus attempted/completed work.
  Registration requires complete initializer and candidate evaluation. The byte
  workflow binds the requested lane flag, joint record totals, helper source
  digests, selected/all-group reciprocal origins, exact support tuples and the
  selected candidate's initialized and refined pose. A different proper rigid
  transform cannot borrow that candidate's diagnostics. Publication failure
  removes measurements and predictions; native unavailability retains its own
  phase, and observed bend count remains unknown.

These are reviewed contracts, not proof against arbitrary code mutation or
hostile process behavior. Failure injection covers selected corruptions rather
than every possible malformed internal object.

## Oracles, counts and import custody

Independent source enumeration found 14 top-level initializer test functions
expanding to 41 outcomes and 10 workflow functions expanding to 85. Neither new
test declares skip/xfail or imports predecessor test cases into pytest collection.
Only explicitly called predecessor bodies run.

The initializer's two new real positives retain all 50 raw rows on two broad
rectangles, a fixed proper permutation pose, stipulated uniform .05 mm error,
and either clean points or moment-balanced normal noise bounded by .002 mm.
Both support sets retain 25 rows. Actual extraction, scheduling, context seeds,
face boxes and finite queries are used; at least one retained start must have
clean action error at most 1e-7 mm. This does not run global two-plane registration
or free-circle measurement, nor does it substitute for the original thin part.
The other two positive regressions call the exact SHA-bound V10 noisy-positive
function body, changing only its private registration dispatch to V11 with pair
mode disabled. Its geometry, poses, noise and numeric oracles are not edited.

Remaining initializer checks explicitly inject schedules/hypotheses or finite
answers where appropriate: joint preflight boundaries, fifteen helper corruptions,
four full-support binding corruptions, absent slots without refill, shared 48
starts, late failure clearing, real shared-buffer chunking, real finite-hole
rejection, 49/50-row admission and attempted-versus-completed work. Their mocks
are not an accepted large-workload qualification.

The workflow reuses 42 unchanged predecessor refusal parameterizations (5+6+13+18).
It adds .05/full-raw/inverse mapping controls, strict policy snapshots, 29 joint/
origin corruptions, five box-contract corruptions and native-phase refusal.
The invented sampler contains three CAD triangles/IDs for the triplet mock;
pair origins have exactly two supports. Its registration, native and measurement
results are deliberately stipulated. No physical output follows from a mock pass.

Both new tests SHA-bind the predecessor files they actually load. The V2 fixture
creates an isolated synthetic package, uses invented byte constants and patches
the mesh loader, STEP reader, native extractors, sampler, registration and local
measurement boundaries. OCP imports in the closure are inside the patched/native
functions, not module-level imports. The current `backend/__init__.py` is absent.
The initializer imports the reviewed backend namespace; remaining external code
is stdlib, NumPy, SciPy and pytest. No legacy server/database startup is reached.

The harness uses one selected test path, `--noconftest`, `/dev/null` configuration,
disabled pytest plugin autoload/cache, `-I -S -B`, numerical threads set to one,
and private HOME/TMPDIR/cache. Focused routes set no native opt-in. Parent-owned
fresh attempt directories retain exclusive STARTED, OUTPUT and COMPLETED files;
before/after source maps and output hashes are required. Monitoring does not
promise a hard RSS allocation cap or a whole-CLI deadline covering all setup,
hashing, terminal I/O and cleanup. Missing terminal evidence cannot count as a
pass. The retained POST must independently verify actual counts, no skips/stops,
exit outcome, full three-file inventory and all 46 frozen source hashes.

## Exact shared source closure

`V11_FOCUSED_SOURCES` contains 45 unique files; each route's `source_bindings`
adds its already-present test plus the harness, yielding the same 46-key map.
Hashes below were independently observed before either V11 focused execution.
The prerequisite audit documents above are release evidence, not extra runtime
imports. The new native diagnostic test/protocol are not in this focused union.

| Source | SHA256 |
| --- | --- |
| `scripts/check_cad_guided_measurement.py` | `5675caa55fac23507c4d818cffb311af44020d0f71a0940acc30927e26545552` |
| `backend/cad_guided_plane_initialization_v9.py` | `c64ab591ee07b91d7f20cb0d663cc2bb2a98e5baa681e072f97353b01e82401d` |
| `backend/cad_guided_plane_initialization_v8.py` | `cd225b4642816f76cfd4b8bd725e3b8ffa5d9f6177ddbfa9b03fb6a22128c28b` |
| `backend/cad_guided_plane_initialization_v4.py` | `a4586cd02c490493472ea842622efeafe9ae18f246979ad4be364313bad72724` |
| `backend/cad_guided_global_registration_v3.py` | `088c0b5e9fd940094fd44462e7e5c831299fe2c69b87418d09c362772263bae0` |
| `backend/cad_guided_global_registration_v2.py` | `5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3` |
| `backend/cad_guided_global_registration_v1.py` | `7700541063c4bfb40c72fb66baf2a78fea1c631e5e3f93eabbb51edfdb59eff7` |
| `backend/cad_guided_triangle_surface_v3.py` | `8e3222bd425451e2e89b2fe0083db34a3702577018b3f557b1f026a86add9f10` |
| `backend/cad_guided_triangle_surface_v2.py` | `7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1` |
| `backend/cad_guided_triangle_distance_bounds_v3.py` | `2404278ebd8872dafa6752d9bf977879e7d0a21ea41752ec740501b7044058d7` |
| `backend/cad_guided_mesh_plane_proposals_v4.py` | `55356d900ce933b4dd4ff9fc8d055fdb46038422b69b6e05f7f9509d73f5ed48` |
| `backend/cad_guided_plane_hypotheses_v4.py` | `7e69fce8bf79216342071aacfbbb7c0d73972c78334cc81d7f0fe15c831f1370` |
| `backend/cad_guided_noisy_mesh_planes_v5.py` | `9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75` |
| `backend/cad_guided_noisy_mesh_planes_v7.py` | `b0822289cf3abbde9c7cbf263b45c8d8ef4c4f45d7913e452c47a46ed881449f` |
| `backend/cad_guided_plane_search_v6.py` | `aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676` |
| `backend/cad_guided_report_envelope_v6.py` | `c7cc45a362c5972aff78265b2ce63fa4d41ec023b662640d7d197c23003ba28e` |
| `backend/cad_guided_plane_initialization_v10.py` | `398c40055c00fd3432410242a7a2ee399d9d716ea8cb3c1755916904b6074331` |
| `backend/cad_guided_face_box_screen_v1.py` | `c8f66cc25f5c4cb6a6ba3c1e1beeb205540813796bb2ab30d72eb596cdf3abb7` |
| `docs/research/2026-09-10_face_box_v10_integration_protocol.md` | `675b7cf38c78b49ec8f759b167bfd77cdc38d887633e4749dbeda397f1f61261` |
| `tests/test_cad_guided_plane_initialization_v8.py` | `e78e5bc17705334b9e7dd1ebe1f70271ae1cdca0a118bbfbea24d95f633a684a` |
| `backend/cad_guided_mesh_offline_v9.py` | `82151485d6ac265e1c9d6538d7f7c2110451403c5e3119ca82506fb69d3b24af` |
| `backend/cad_guided_mesh_offline_v8.py` | `afdfc7f859eb5fdbf0fa7d796cc06a4180d40c02f6b0e590362f83e3f339e7ec` |
| `backend/cad_guided_mesh_offline_v2.py` | `fa30bc1ed575f9174414e3737eaaffecc6eafdcbcd9e3b1d68475f1af7ea8cbf` |
| `backend/cad_guided_mesh_offline_v1.py` | `55243b2a557e77a9c50e27a7fdef9257d60ed2468bc23c1788c391d7a836bdba` |
| `backend/cad_guided_mesh_input_v1.py` | `c4d936e2f60837d7c5221c9a46b64abd68c4ccee01ea72e9fd3c4050bde47917` |
| `backend/cad_guided_mesh_topology_v4.py` | `4d061b56a39aee6b7d9e8c04eb7af88b0573a7916d3d679c610c4aca81ee0734` |
| `backend/cad_guided_native_features.py` | `f8c681027af231546974633a02f12038d522c604bec14196f2ae5d49b439c99f` |
| `backend/cad_guided_native_features_v1.py` | `3804a583f5642df41cd3eb02be36b3f5024347b6514861d404389e85c1ccf98b` |
| `backend/cad_guided_measurement.py` | `d3a11bde2b14c6ece39414360e3146bf60e3801190b9e04cd6c5178ed903f2cb` |
| `backend/cad_guided_instances.py` | `d30b3fc8579b0a2d44d37bfe093343b3f7d703661793784d78b50efbc38eacda` |
| `backend/cad_guided_cad_surface_v2.py` | `b7364234a716f283f46246a452451f47c8efb07c18b2d37375c6fda0636da0d1` |
| `backend/cad_guided_cad_sampling_v1.py` | `f1d32159a50b455d6b9a0b5f646586f0d79e7f9c26e8fffbe67f97c5b74e181a` |
| `backend/cad_guided_local_measurement_v1.py` | `5b07e8c14d3eededfb7cbdb2cb628e6d631a30662280fc2d39b2528d862b541b` |
| `tests/test_cad_guided_mesh_offline_v2.py` | `10fdfeac837d32a4f61eff53e329fbfeb0b742fcf7d9834024819e6241bb9cfc` |
| `docs/research/2026-09-10_bound_aware_v9_focused_protocol.md` | `af07fd1b9cf4c950d9fdfa95cc7b4b102137b5c9bd859e8ab115716384c276a5` |
| `backend/cad_guided_mesh_offline_v10.py` | `81d9411ccfff00c9f9fe5aa9b6fc25d7384acf993d34cfdbb0349dfe109c7fa4` |
| `tests/test_cad_guided_mesh_offline_v9.py` | `b3bb60f5b5e5e2dceb5a1bb87e3b9290178218b169095b90f244a8f13c3ccd00` |
| `backend/cad_guided_plane_initialization_v11.py` | `31bc1b9c3fe58b1befd7432134ce0640fa9243e23f51b51c28ca4ae72aa7610e` |
| `backend/cad_guided_mesh_offline_v11.py` | `dbd60ebc8c2c159866c212506e46e35d56af17b196827ba5243c637c6dfe6dfa` |
| `backend/cad_guided_two_plane_context_v1.py` | `9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a` |
| `backend/cad_guided_two_plane_schedule_v1.py` | `1fcd5bb06922b64693de235cbf1e6e7df6fdd9c43795c6bced27af6d2e4b866f` |
| `tests/test_cad_guided_plane_initialization_v10.py` | `4882b701af3806fd03ddde9bb0b1af823bd0124e859e6655207f683db1563eb0` |
| `tests/test_cad_guided_mesh_offline_v10.py` | `0bba4eb137eede10d5a2724598285cddda6ebf0da9544bc4d01b18666240b9c2` |
| `tests/test_cad_guided_plane_initialization_v11.py` | `16a646404b4cf35975956b717a9143b52d8bca1e7ec5467f40096aefbc0edf1e` |
| `tests/test_cad_guided_mesh_offline_v11.py` | `dcd55ca3569be405f5b861b7838ba7866513b00a10d44d2281a934f4332f4cd9` |
| `docs/research/2026-09-10_two_plane_v11_integration_protocol.md` | `2748fed1bc8b7de8fc17bc93e3bb8b4c7628c0ec136fe12a93038fcf4d3f674b` |

## Advancement boundary

Only actual complete, budget-admitted, hash-stable receipts and independent POSTs
can qualify these two focused gates. Even both passing establishes neither
original thin-part .05/free-circle output, physical F1 improvement, all accepted
workload capacity, calibration, datum validity, count accuracy nor production
conformance. The separate original-geometry diagnostic must keep its original
positive-output criteria; successful recording of abstention is not a positive.
The spent physical V3 and recovered cached UNION control remain unchanged.
