# Two-plane V11 stage diagnostic: independent exact PRE

Date: 2026-09-10. Reviewer: independent AI agent
`production_architecture_review`, not the diagnostic-test, runtime, protocol or
harness author. No human engineering or manufacturing approval is claimed.

Decision: **GREEN_FOR_FIRST_SOURCE_GATE_ONLY_CONDITIONAL_DIAGNOSTIC_SEQUENCE**.
Blocking findings after the reviewed repair: none for this bounded scope.
Tests and diagnostic geometry are UNEXECUTED at this decision.

## Release and exact sequence

Release first, once:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite two_plane_v11_stage_diagnostic_source
```

Expected **32 source/mock passes and exactly one intentional recording skip**.
Only an independent retained POST-GREEN on those same frozen bytes releases the
first, once-only recording command:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite two_plane_v11_stage_diagnostic
```

Its expected recording matrix is **33 outcomes, zero skips**:32 source/mock
checks plus one two-case recording loop. Pytest success can establish complete
recording, not two successful original positives. Each original clean/noisy
oracle must separately retain PASS, FAIL or UNKNOWN. Neither command releases
engine qualification, workflow, native geometry, physical inputs, Gold11 scoring,
external consultation or production. No command was run by this reviewer.

Each route retains60s monitored wall/45s worker CPU/1,073,741,824B sampled group
RSS/1,048,576B output. Recording adds exactly two fresh case slots,16,777,216B
serialized JSON each/33,554,432B total. Capture limits remain1,000,000tree nodes,
depth64, arrays at most1,000,000values,128attempts and3,200rows per channel per
case. These limits do not change any engine policy or budget. No source edits
between routes; any failure, unexpected skip, stop, missing evidence or hash
drift is terminal and blocks advancement. Do not truncate, retry or widen caps.

## Review and resolved finding

Read the complete633-line repaired test, complete new protocol and all eight
harness diff hunks against preserved5675caa5. Reviewed the original fixture,
14-statement post-call oracle, inherited six helper functions, V8 preparation,
V11 public/prefilter/query seams and actual returned value types. Independently
counted18 test functions expanding to33 enabled outcomes by source AST. Checked
all48 current hashes and confirmed the45 original scientific files remain
identical to the earlier V11 focused freeze. No target module was imported,
test collected, geometry executed or original data accessed.

The earlier draft66fed801 had a real completeness defect: its observer captured
only after a box/query returned, so a raised method could lose the attempt and
still appear fully captured. The final8dbe3e7c source records each attempt and
counter snapshot before dispatch, then records returned/raised dispositions and
after counters. A raised method retains its exception and explicitly marks the
unreturned nested primitive payload missing; it vetoes capture completeness.
Eight runtime call/row counters across the two channels reconcile with observed
charged/completed deltas, requested/returned/raised totals and retained records.
Missing counter increments, lost records or overflow cannot become complete.

The new counter-only mocks cover both channels, exact argument/return identity,
attempt-before-call ordering, charged versus preflight exceptions, same
Exception/SystemExit identity, unavailable returned payloads, unobserved drift,
late call/row overflow and response-serialization failure. They do not call a
surface primitive or initializer. Ordinary recording errors preserve the
original scientific return/exception; this is not an asynchronous-signal or
hostile-object containment guarantee.

## Scientific and transport checks

- The opt-in extracts the exact source-bound original construction and four
  constants, not altered geometry. It preserves50raw rows/64scan triangles/
  4CAD triangles, clean then bounded moment-balanced noise, original proper
  pose, .05mm declaration, original policy50 and bank2.
- There is one public initializer call per case, with the same arguments and
  remaining defaults. Observers call the saved original methods once and return
  their original objects; they do not alter points, policies, candidate order
  or answers. Preparation context is retained by reference until after inference,
  so its final generated pair/triplet records are captured, not a pre-pair snapshot.
- The original14 post-call statements remain in order and run against the same
  returned answer. Known pose/action errors are evaluated only afterward. A
  failed original assertion remains FAIL; it cannot pass by empty starts.
- Actual successful query types are float64/bool/int64/U6 arrays, builtin face-ID
  tuples and JSON-compatible metadata; prepared context contains arrays and maps,
  not a surface-index/dataclass object. The transport retains full array values,
  dtype, shape and raw-byte SHA, including integer-key map identities. Unsupported,
  nonfinite or over-budget transport fails recording rather than projecting a
  favorable subset. It is not a general arbitrary-Python reconstruction codec.
- Both fixed case dispositions remain in bounded stdout if ordinary serialization
  or publication fails. Exclusive case filenames refuse overwrite. The protocol
  requires POST inventory/hash of every actual file, including partial writes;
  absent/full-corrupt artifacts remain UNKNOWN/incomplete, never successful-prefix
  evidence. No hidden inference retry fills missing query data.

The source branch imports only stdlib/NumPy/pytest and reads declared source or
tiny owned publication fixtures. Actual backend imports and extracted fixture
execution occur only in the opt-in recording function. No unselected predecessor
test bodies are collected. The harness sets the diagnostic opt-in and parent-owned
ARTIFACTS destination only for the recording route, and uses one selected test,
disabled plugin/cache/conftest, numerical threads1 and private HOME/TMPDIR.

## Custody and prior evidence

The original initializer failure remains independently authenticated40PASS/1FAIL:
`docs/research/2026-09-10_two_plane_v11_initializer_post_run_audit.md`, SHA256
`26cb80b06ba24463753a04560e765e485645abb99e749d60b828c39ad13e4dd7`.
It is not overwritten or qualified by this diagnostic. Original source/math/import
review is retained in the focused PRE
`cb232da25c94f2d4490a387811cece0b59da50e43ab495775bb0ae181f88fbe4`;
the pure context/scheduler POSTs remain prerequisites, not new execution requests.

The prior harness is preserved at
`docs/research/source_snapshots/two_plane_stage_pre_5675caa5/check_cad_guided_measurement.py`,
SHA256`5675caa55fac23507c4d818cffb311af44020d0f71a0940acc30927e26545552`.
The new delta only adds the two routes, shared binding, inherited limits and
namespace admission, recording-only flag/destination, and artifact fields in
COMPLETED. Existing supervisor logic is unchanged. Monitored wall excludes some
source/receipt I/O and cleanup; RSS is sampled, and inherited reaping is not a
hard-bounded cleanup proof. Missing terminal handling cannot count as success.

Both routes require new unique STARTED/OUTPUT/COMPLETED receipts, before/after
hash agreement and independent POST. Recording additionally requires the complete
two-case inventory, artifact hashes, array reconstruction/digest checks, original
input/policy identity and observer reconciliation. A mechanism conclusion must
come from actual retained evidence, not this source review.

## Exact48-file shared closure

The same45 original `V11_FOCUSED_SOURCES`, diagnostic test, new protocol and
current harness form48 unique paths. The selected test is already in that union.
The following exact bytes bind both prospective routes:

| Source | SHA256 |
| --- | --- |
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
| `tests/test_cad_guided_two_plane_v11_stage_diagnostic.py` | `8dbe3e7c2286fdb6807047efb9d56fdbd99cced338f7a954d12fb6ffe6cd7fca` |
| `docs/research/2026-09-10_two_plane_v11_stage_diagnostic_protocol.md` | `f57ff260fbee688d42293f626986ffbfdbd4945fa9989fa64f8361e0a03f633e` |
| `scripts/check_cad_guided_measurement.py` | `b96502ea02e234e9a801e62add142efb361f7cd21d89687af6bafea7a9344fa2` |

## Advancement limit

A recording result may distinguish useful-pose generation from finite-support
rejection and localize actual ambiguity flags. It does not itself prove a
retessellation mechanism, repair noisy output, establish global observability,
or beat the cached UNION control. No engine threshold, reference, physical
measurement truth, calibration, datum, count or conformance claim changes.

