# Gold11 full11 V9 - independent physical pre-run audit (PRE), 2026-09-13

Written 2026-09-13T05:34:53Z (UTC; local calendar date 2026-09-13).
Reviewer: `independent-ai-pre-auditor-v9` (Claude Fable 5.1, reviewer role; not
the author of the candidate, the freeze, the protocol or any review).
Attempt audited: `docs/research/gold11_mesh_development_v9_attempt` (frozen,
author-stated freeze time 2026-09-13 05:15:51 UTC; not executed). Companion
machine record: `PRE_RUN_AUDIT.json`, 27755 bytes, SHA-256
`750c85f519d15598c504597555a33b30eb20aba30eef4a61dfd1c472e04a037b`
(`written_utc` 2026-09-13T05:30:51Z).

## Decision

**GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN.** Blocking findings: none.
Bundle SHA-256 (canonical JSON of the decoded `FROZEN.json` object, sort_keys,
separators `(",", ":")`, allow_nan=False, UTF-8), recomputed independently:
`c50a8cb9684ba891a97005755529d952229e25b6db83e4549a17cb830ca4b47d` - agrees
with the value the freeze printed (as reported by the author); the
`ensure_ascii` True/False variants are byte-identical because the file is pure
ASCII (51206 canonical bytes).

This is an engineering audit of a frozen consumed-development attempt. It is
not manufacturing approval, calibration, conformance, generalization or
production evidence, and it establishes no F1 or physical improvement. It
authorizes only the single scoped `--execute` of this frozen attempt, exactly
as the frozen protocol's one-use boundary states, through the bound waiter:

```sh
.venv/bin/python scripts/gold11_v9_execute_waiter.py
# which runs, once: OMP_NUM_THREADS=1 .venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v9.py --execute
```

## Scope, method and what was not opened

Read-only. Nothing was executed except `shasum`, `wc`, `diff`, `cat`, `sed`,
`grep`, `ls -la`, `tar -tvf` (listing only), `ps -axo pid,pgid,etime,rss,command`,
`date -u`, `sysctl -n hw.memsize`, `.venv/bin/python --version` (plus a
`sys`/`platform` query of the same interpreter) and `.venv/bin/python` snippets
that hash files, parse JSON/AST, and exec only the runner's top-level constant
`Assign` nodes in an inert namespace (`math`, `json`, `dict`, `list`, `tuple`);
no runner function was called, no module under `backend/` or `scripts/` was
imported, no test, freeze, verify, execute, worker or geometry job was run.
Not opened, stat'ed, hashed, sampled or traversed: `data/` and any
`.ply`/`.step`, `data/bend_corpus/annotations/*` (Gold and reference
containers), `/private/tmp/gold11-*`, `/var/folders/*`, `.env` or credentials,
`output/`, `outputs/`, pools/manifests, HOLDOUT10, incoming-email/EHO parts,
and every `docs/research/*_attempt` other than the V9 attempt. Only the two
output files were written, both under `/Users/idantal/.claude/jobs/1519fdc2/tmp/`.
Process disclosure: one hashing snippet briefly wrote a transient helper file
(`_snapshot_sizes.json`, a size map) into that same tmp directory, outside the
repository; it was deleted immediately and the archive size comparison was
redone in memory. The pre-existing tmp `PRE_RUN_AUDIT.json` (20736 bytes,
SHA-256 `41fd61fc6157e44823477e04da034d4b378307055cb325e0bcb19f1d7c7d6554`,
the V8 machine record) was replaced by the V9 record, as instructed.

## Check 1 - frozen bundle vs runner constants and live environment

- `FROZEN.json`: 51207 bytes, mode 0600, file SHA-256
  `67a253ad14702d15853942d70f94260d06e91713b76e4d84f26242ceed37a26e` (below the
  runner's 256 KiB read cap); decodes with duplicate-key and non-finite
  rejection; 24 top-level keys; `schema` = `GOLD11_MESH_DEVELOPMENT_V9`;
  `production_use_allowed` = false; `historical_comparability` =
  `UNESTABLISHED`.
- `source_sha256`: 85 entries (none under `data/`, none `.ply`/`.step`). For
  every entry the working-tree file (`/Users/idantal/ShermanQC/FullProject/<path>`),
  the snapshot copy (`.../gold11_mesh_development_v9_attempt/source/<path>`) and
  the declared hash agree: 85/85, 0 mismatches. The snapshot tree holds exactly
  those 85 regular files (backend 59, scripts 21, docs 3, tests 2; 0 extra, 0
  missing, no symlinks).
- `SOURCES` tuple of the frozen `scripts/run_gold11_mesh_development_v9.py`
  (parsed with `ast`, literal constants plus the three names `SOURCE_BASIS`,
  `REVIEW`, `PROTOCOL` resolved from their own literal assignments): 85
  elements, 85 distinct, key set identical to the `source_sha256` map; it
  includes `scripts/gold11_v9_execute_waiter.py` and the V8 basis.
- `SOURCE_BASIS` = `scripts/run_gold11_mesh_development_v8.py`,
  `SOURCE_BASIS_SHA` =
  `0e9aa2f64f3fc14e0da137687e44c8cd0dc0712f9e8bd783726df4187b1360c5`; the
  working-tree V8 runner and the snapshot V8 runner both hash to that value;
  the snapshot V8 runner is 172872 bytes.
- Frozen V9 runner: 181565 bytes, SHA-256
  `4567a0f584e15b75339238b3f50ebb19209f127bfcb3a55e5fa73fd36b4a85b1` (working
  tree, snapshot, delta table and Codex follow-up agree). Removing the single
  line `"scripts/gold11_v9_execute_waiter.py",` (one occurrence) reproduces
  `ad61e12b0a06c0cafb7e515687019e42c6e137e26a419753aaeef2360f5b4241`, the
  runner hash root reviewed at 04:50 UTC - the post-review runner delta is
  exactly the waiter addition, as the coordination record claims.
- Runner constants (inert exec of 52 top-level assignments; `ROOT`, `RUN`,
  `CATALOG`, `STAGE_SAMPLE_SOURCE_PATHS`, `CONTEXT_REQUESTS_BY_ARM` skipped as
  non-inert and read instead) equal the bundle for every compared field:
  `source_profile`, `config`, `limits`, `certificate_caps`, `arm_definitions`,
  `arms`, `comparator`, `two_plane_declaration`, `config_delta_from_v8`,
  `measurement_policy_declaration`, `pose_admission_declaration`,
  `position_error_declaration`, `scoring_transport`, `artifact_policy`,
  `retained_control_policy`, `catalog_sha256`, `selected_reference_sha256`
  (17/17 equal after JSON round trip) plus the `schema` literal in `freeze()`.
- `arm_requests`: rebuilt from the exec'd constants by replicating the
  `arm_request()` composition (read, not called; `CONTEXT_REQUESTS_BY_ARM`
  replicated as a JSON deep copy of `CONTEXT_V4_REQUEST` per arm); canonical
  bytes equal the bundle's for both arms. Canonical SHA-256 of each frozen arm
  request: control
  `b1df28220670a37fc9964fac3b64d914d12c69dc40cc1ded9e9d0ef3403eeb14` (9055
  bytes; scan planes v7, schedule v1), challenger
  `2bcf0f4a0c39ab867466445ebddb9ed2fc82bdc3462cfa17a0e597b2154ebb74` (10323
  bytes; composite_v1 planes, schedule v2); all-arm digest
  `f6ca1b6c8f9d2388a243ae38783a65c8db97da9e3bf7b1e310b6d4c90fe589b5`. Both
  context mode v4, both `implementation` =
  `CAD_GUIDED_MESH_OFFLINE_V16_CONTEXT_V4_COMPLETION_V1_KERNEL_V2`, both
  `registration_implementation` =
  `SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V4`, both
  `context_request.budgets.max_extrema_coordinate_visits` = 512000000.
- Declared config deltas are present in the frozen values:
  `measurement.max_fit_residual_mm` 0.05 -> 0.03, `measurement.plane_exclusion_mm`
  0.075 -> 0.04, `support.max_angular_gap_deg` 5.0 -> 30.0 (V8 5.0 recorded in
  the delta; `max_axial_gap_mm` 4.0, `min_arc_sweep_deg` 20.0,
  `min_local_points` 3 unchanged); NEW `layer_policy` with exactly 8 fields
  (`enabled` true, `layer_split_gap_mm` 0.3, `min_layer_rows` 12,
  `max_layers_reported` 8, `max_layer_normal_angle_deg` 3.0, `max_layer_gap_mm`
  6.0, `min_projected_interval_overlap` 0.5, `max_layer_rms_mm` 0.1); NEW
  `completion_policy` (`enabled` true, `max_parents` 256,
  `max_completions_per_parent` 64, `max_completions_total` 4096, `max_shift_mm`
  1000.0, `min_axis_projection` 0.001, `min_pair_sine` 0.001); NEW
  `completion_budgets` (15 fields, equal to `initialization_budgets`) and
  `completion_budget_declaration` (10 active fields, 5 declared-inactive
  fields, work-unit formula, combined-allowance and ledger-cap statements).
  `CONFIG_DELTA_FROM_V8` carries six keys (the three value deltas, the
  completion lane, the layer selection and `inherited_v7_to_v8`).
  `MEASUREMENT_POLICY_DECLARATION` (`chosen_from_gold` false,
  `tuned_to_case_identities` false) is frozen. The 1.0 mm pose admission gate
  is unchanged from V8: `CONFIG.registration.max_inlier_plane_p95_mm` = 1.0 =
  `POSE_ADMISSION_DECLARATION.max_inlier_plane_p95_mm`;
  `surface_budgets.max_ties_per_query` 256 and
  `workflow.max_measurement_contexts` 8 are unchanged.
- Helper pins equal the frozen map: supervision
  `scripts/gold11_owned_process_supervision_v3.py`
  `9eec779c6ba14c969fb37950aac2e7aab8af844d3f573233b1b22c92991472ea`; terminal
  artifacts `scripts/gold11_terminal_artifacts_v4.py`
  `3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838`
  (= `artifact_policy.helper_sha256`); stage sampler
  `scripts/gold11_engine_stage_sampler_v1.py`
  `80316d0026bcc2fabcf1d8861d010d5f434f720c303f5969384cc45f7809b6b0`; all
  seven `SCORING_SOURCES` (f1 contract v2, evaluate_f1, mesh scoring v1,
  contact scoring v2, projection v1, contact scoring v3, projection v4) are in
  the frozen denominator.
- Accepted source-closure hashes (coordination section "Source closure
  ACCEPTED - 05:08-05:14 UTC" and the review response's delta table): the
  frozen snapshot files carry exactly the twelve accepted values -
  `cad_guided_measurement_v2.py` 12bb9ce9..., `cad_guided_context_measurement_v2.py`
  9b48c72b..., `cad_guided_support_cover_measurement_v3.py` 6c6bd758...,
  `cad_guided_plane_initialization_v15.py` 505f313d...,
  `cad_guided_mesh_offline_v16.py` 469c444f..., `cad_guided_report_envelope_v7.py`
  b7abe653..., `gold11_contact_scoring_v3.py` 4f0cbea8...,
  `gold11_schedule_projection_v4.py` 32029cbb...,
  `run_gold11_mesh_development_v9.py` 4567a0f5..., `gold11_v9_execute_waiter.py`
  85913855..., protocol e6fcd96a..., delta design d00991ce... - 12/12 (full
  digests in the JSON companion). The frozen contact scorer V2
  `scripts/gold11_contact_scoring_v2.py` is present at
  `9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff` and the
  exact cover planner V1 `backend/cad_guided_exact_support_cover_v1.py` at
  `70fc42d19f67346b6b5f4951832edef481aa4d94abaeec892b415cfc018cab47`; 14/14
  match, working tree = snapshot = expected.
- Release gate constants: `RELEASE_AUTHORIZED` = True, `TIMING_RELEASE_STATUS`
  = `REVIEWED_V9_EXACT_TIMING_RELEASE`, integer CPU limits, terminal reserve
  2400 s below the 104400 s campaign wall; `STAGE_SAMPLE_SOURCE_PATHS` names
  `scripts/run_gold11_mesh_development_v9.py`; `LEAF_SCHEMA` =
  `ONE_USE_PARENT_BOUND_LEAF_V9`.
- Bounding the runner delta: a full `diff -U0` of the frozen V8 runner against
  the frozen V9 runner yields 58 hunks (311 diff lines), all of the reviewed
  classes (identity strings and error codes; attempt/protocol/basis paths;
  `SOURCES` additions - V7 runner as preservation oracle, the eight new
  modules, the V9 design, the waiter; `ARM_DEFINITIONS`/`COMPARATOR`
  identities; `CONFIG_DELTA_FROM_V8` and `MEASUREMENT_POLICY_DECLARATION`; the
  CONFIG measurement/support/layer/completion deltas; `SCORING_SOURCES`
  contact V3 + projection V4 and wrapper schema V4; engine wiring imports for
  V16/V15/kernel V2 policies; scorer/projection function names; report,
  custody and comparison schema identities; private cache prefixes). No hunk
  touches `LIMITS`, custody, supervision, terminal fill, scoring transport
  sizes, the matcher, the pose gate or any other CONFIG value.
- `runtime_metadata` equals the live environment: `.venv/bin/python`
  (symlink to `.../cpython-3.11-macos-aarch64-none/bin/python3.11`) resolves to
  `/Users/idantal/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11`,
  version `3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]`, platform
  darwin, machine arm64, numpy 2.4.6, scipy 1.17.1, cadquery-ocp 7.9.3.1.1;
  exactly one dist-info of each named distribution is discoverable under
  `.venv/lib/python3.11/site-packages` (`numpy-2.4.6`, `scipy-1.17.1`,
  `cadquery_ocp-7.9.3.1.1`; a separately named `cadquery_ocp_proxy-7.9.3.1.1`
  also exists). Attestation is version metadata only, as the bundle declares.

## Check 2 - case authority (frozen review source only)

- `source/scripts/review_gold11_inputs_v1.py`: 25616 bytes, SHA-256
  `41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955`, equal to
  the runner's `REVIEW_SHA`, to the frozen map entry and to the protocol.
- Exactly one `CASES` assignment, of the form `json.loads(<string literal>)`
  (string recovered with `ast.literal_eval`, then `json.loads`): 11 cases, 11
  distinct `scan_sha256`, 6 distinct `cad_sha256`, 11 distinct case tokens.
- `{scan_path: scan_sha256, cad_path: cad_sha256}` over all cases: 17 distinct
  absolute paths; canonical SHA-256
  `2459f4e93fe4204889ade6f8a4150d3543ec9f96625f39aaec811f003db4d157` -
  IDENTICAL to the V8 PRE's raw17 digest (cohort unchanged).
- Canonical digest of the case list:
  `943268be4ee3d33301ab44dfb3e6f85fb0d34de852f788464f1d438ffe43813e`, identical
  for the review source and `FROZEN.json["cases"]` (order and content
  identical) and IDENTICAL to the V8 PRE's case-list digest.
- Case tokens in order: CG08506AE97D, CG191DEC7F81, CG1E0BBA2933, CG4BDD1371B6,
  CG4FAF273093, CG64C412CC73, CG65CD575B1C, CG84C8FA12D2, CGE4B0549E41,
  CGEBA9A869AC, CGF41F49066A. Families (`part_key`) in order: 10839000,
  49001000, 38172000, 37361005, 49024000, 40500000, 38172000, 10839000,
  10839000, 38172000, 38172000. Reference ids in order: corpus/03_10839000,
  corpus/10_49001000, shaked_xv/C05, corpus/05_37361005, corpus/11_49024000,
  corpus/08_40500000, corpus/06_38172000, corpus/01_10839000,
  corpus/02_10839000, corpus/07_38172000, shaked_xv/C06.
- Declared reference bindings in the runner constants: `CATALOG_SHA`
  `4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36`,
  `SELECTED_SHA`
  `d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc`; both
  equal the bundle's `catalog_sha256` / `selected_reference_sha256` and the
  protocol. These are declarations only; the runner verifies them in
  `selected_references()` at scoring time (71 reference lines). No raw
  scan/CAD file and no Gold or reference container was opened by this audit.

## Check 3 - attempt tree and external archive

- Attempt directory top level: exactly `FROZEN.json` (mode 0600) and `source/`.
  No `STARTED.json`, `COMPLETED.json`, `PRE_RUN_AUDIT.json`, `FINAL_GATE.json`,
  `FINALIZATION_FAILED.json`, `COORDINATOR_DISPATCH.json`, `cases`,
  `leaf_dispatch` or `coordinator_work` marker exists.
- Snapshot: 85 regular files under `source/` (backend 59, scripts 21, docs 3,
  tests 2), none outside the map, no symlinks.
- External archive
  `docs/research/gold11_mesh_development_v9_physical_frozen.RdikX7AH/SOURCE.tar`
  (directory mode 0700, sole entry): SHA-256
  `412c9f8e662c845e172cdd19a134031c62a5c4a12866041fd82981da2c6c2326`, 3451904
  bytes (both equal the author's freeze figures). `tar -tvf` lists 92 entries:
  6 directory entries (`source/`, `source/tests/`, `source/backend/`,
  `source/docs/`, `source/scripts/`, `source/docs/research/`) and 86 file
  members = `FROZEN.json` plus exactly the 85 `source/<path>` members (0 extra,
  0 missing); every listed member size equals the on-disk snapshot size
  (86/86); all file members are mode `-rw-------`. Listing only; nothing was
  extracted.

## Check 4 - protocol and delta design (frozen copies)

- Protocol `docs/research/2026-09-13_gold11_mesh_development_v9_protocol.md`
  14198 bytes, SHA-256
  `e6fcd96aeb85bf3d91543f63f2001511ca22ff13a46c2bd1f1a51ca810fef54f`; delta
  design `docs/research/2026-09-13_gold11_mesh_development_v9_delta_design.md`
  11932 bytes, SHA-256
  `d00991ce3ec38d89c668830eeb28cb505086cb64ab0c62c11c0addfb8c48efce`; both
  equal the frozen map, the working tree and the accepted delta table.
- Limits table equals the runner's `LIMITS` and the bundle's `limits`
  value-for-value: per-case wall/CPU 3540/1860 s; worker sampled tree /
  coordinator HWM / outer tree 8/4/12 GiB (8589934592 / 4294967296 /
  12884901888); scan/CAD input 64/8 MiB (67108864 / 8388608); leaf log /
  result 1/128 MiB (1048576 / 134217728); projection and receipt 8 MiB
  (`compact_case_bytes` 8388608; `scoring_transport.inner_projection_bytes`
  and `retained_schedule_wrapper_bytes` 8388608); scorer 180 s / 120 s / 2 GiB
  / 192 MiB request / 32 MiB result (2147483648 / 201326592 / 33554432);
  coordinator log 2 MiB (2097152) / parent read 180 s; campaign 104400 s /
  reserve 2400 s. Under these limits 22*3540 + 180 + 2400 = 80460 < 104400
  holds (the V9 protocol does not restate the arithmetic).
- The declared deltas are described in both documents: measurement policy
  delta 0.05 -> 0.03 / e 0.075 -> 0.04 with its derivation, coupled effects
  and non-uniqueness; `support.max_angular_gap_deg` 5 -> 30 scoped to fixture
  and nominal arithmetic; the additive two-of-three completion lane with its
  separate budget envelope and declared inactive fields; kernel V2 layer
  selection with the empty `INNER_FACE_ELIGIBLE_STATUSES` eligibility contract
  (every measured window a generic conditional contact line, no inner-face
  role); consumer-chain bindings (context V2, cover V3, contact V3, projection
  V4, envelope V7, mesh offline V16). The protocol keeps the 1.0 mm gate as a
  provisional development admission, the combined-change interpretation and
  the scan-plane-lane-only arm scope.
- Release boundary: the protocol names the waiter launch
  `.venv/bin/python scripts/gold11_v9_execute_waiter.py`, which runs once
  `OMP_NUM_THREADS=1 .venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v9.py --execute`,
  the exclusive `O_EXCL` reservation of the three outer records, the one-time
  token / exclusive lock / still-pending admission of the waiting parent, the
  refusal of a direct `--waiter` invocation, and the outer exit record path
  `docs/research/gold11_mesh_development_v9_attempt_outer_exit.json` (beside,
  never inside, the attempt directory); an unobserved exit is recorded as
  unobserved. The three outer records
  (`..._outer_exit.json`, `..._outer_pid.txt`, `..._outer_stdout.txt`) do NOT
  yet exist.
- Expected retained negative outcomes (to be reported as such, never grounds
  for a change after launch): windows above 0.03 mm flange residual refuse
  `NOT_PLANAR`; single-layer, ambiguous or incompatible partitions measure as
  generic unresolved carriers or refuse; sparse angular stations refuse
  (`ARC_COMPATIBLE_WITH_FLAT`, `INSUFFICIENT_DISTINCT_ANGULAR_SUPPORT`,
  `DISCONNECTED_LOCAL_ANGULAR_SUPPORT` beyond 30 deg); completion candidates
  may fail the prefilter, refinement or admission or be omitted by the
  non-exhaustive ranking (omissions published); CG65 has no two-of-three
  parent; the CGE4B membership-cap, CG64 `INVALID_NATIVE_SOLID` and
  CG191/CG4FAF surface-query CPU blockers persist. Positive geometry on every
  case is not a prerequisite.

## Check 4a - execute waiter (frozen copy, read-only)

- `source/scripts/gold11_v9_execute_waiter.py`: 9179 bytes, SHA-256
  `85913855908ac3a965cb37d7fdd36a8c3464aabd05621184ecd9b4b6e8107688` (working
  tree, snapshot, map, delta table and Codex follow-up agree).
- `COMMAND` (lines 32-33) is `[<ROOT>/.venv/bin/python, -I, -S, -B, -u,
  scripts/run_gold11_mesh_development_v9.py, --execute]`, run with
  `cwd=<ROOT>` and `OMP_NUM_THREADS=1` set in the child environment (line
  166) - exactly the protocol's execute command.
- `launch()` (lines 83-125): refuses on process overlap (pattern
  `run_gold11_|check_|pytest|cad_guided|_gold11_v\d_exact_scoring`, its own
  name excluded), on a non-pristine attempt tree (entry names must be exactly
  `FROZEN.json`, `PRE_RUN_AUDIT.json`, `source`; names only), and when any of
  the three outer records exists (`os.path.lexists`); then reserves
  `_outer_pid.txt`, `_outer_exit.json` (pending record, nonce) and
  `_outer_stdout.txt` (empty) with `O_WRONLY|O_CREAT|O_EXCL` (lines 62-65,
  101-103; a `FileExistsError` refuses) BEFORE spawning the detached waiting
  parent (`start_new_session=True`, lines 112-114) with the one-time token and
  the configuration in its environment; a spawn failure atomically replaces
  the exit record with `UNOBSERVED_WAITER_LAUNCH_FAILED` (lines 118-122).
- `waiter()` (lines 128-188): refuses without the token/config (131-134),
  without the reservation (137-141), when the reservation is already locked
  (`LOCK_EX|LOCK_NB`, held for its whole life, 142-147), and when the
  reservation line is not `RESERVED <token>`, the exit record is not the
  pending record for that nonce, or the stdout log is not empty (155-159) -
  touching no record on refusal; it then claims the reservation, runs the
  command as its direct child, writes `RUNNING ... runner_pid=...`, waits, and
  replaces the exit record atomically (temp file + `os.replace`, 68-72) with
  the observed return code (`OBSERVED_BY_WAITING_PARENT`, 181-183); any
  failure before observation replaces it with
  `UNOBSERVED_WAITING_PARENT_FAILED:<type>` and re-raises (176-180).
- Records live beside, never inside, the attempt directory (46-47). The
  runner's own one-shot gate (`STARTED_OR_TERMINAL_ATTEMPT_NO_RETRY`) is
  unchanged. Limitation (also stated by the protocol and the Codex
  runner/waiter review): no crash-proof guarantee - a SIGKILL or power loss
  leaves `UNOBSERVED_PENDING`, which the POST must keep as unobserved.

## Check 5 - independent reviews and closures on record

Documents read in full (working tree, hashed at audit time):

| SHA-256 | bytes | file |
|---|---:|---|
| e3cba171b5e86eda2d1a9360e90e08821282eae218dfd88406f39941ea39f2c5 | 25935 | 2026-09-13_fable51_v9_codex_coordination.md |
| 5298f201a70edd1d59a44b707e85341726561f7c5888a68f13831321c655ee3b | 26284 | 2026-09-13_fable51_v9_review_response.md |
| 2fe8d44b538ec2c6492c44f60adb8132dfc3de8a08be6ef04005fb3faf1072a9 | 4806 | 2026-09-13_gold11_v9_material_chain_delta_acceptance_codex_audit.md |
| 4c9c50e5bd0acae0d01dad6f4ee141a63399a9ed3cdf1952b651af384c81a73a | 8061 | 2026-09-13_gold11_v9_completion_delta_acceptance_codex_audit.md |
| d47f66af5fb2b724bc4140ed6dcf5c36c824bf5f4816e429b7a436862b2d5ed8 | 7529 | 2026-09-13_gold11_v9_runner_waiter_final_closure_codex_audit.md |

- The three Codex acceptance documents hash exactly to the values named in
  the coordination section (3/3 match). Material chain: "Accept the scoped
  source fixes" (prior P1 - unverified two-layer status granting inner-face
  eligibility - closed by the empty `INNER_FACE_ELIGIBLE_STATUSES`; the two
  boolean P2 sites strict) with one test-evidence correction (two cover
  parametrize names), closed by the 05:12 UTC append-only correction.
  Completion: "ACCEPTED ... No blocking finding remains" (full V16 binding and
  workflow dispatch exercised for completion-only and mixed origins; the one
  skip is a disclosed non-applicable mutation; one stale test comment noted as
  nonblocking). Runner/waiter: both P2s (waiter outside the frozen inventory;
  direct `--waiter` entry could overwrite exit evidence) closed in the 05:08
  UTC follow-up - "No remaining blocker in this runner/waiter delta".
- Coordination decision (05:08-05:14 UTC): "the reviewed V9 candidate's
  source-closure gate is accepted"; no unresolved P1 and no unresolved P2 is
  recorded; freeze, independent PRE, one waiter-launched execute and POST are
  the named next steps; the decision itself is "NOT the physical PRE".
- Test receipts are author-observed (855 passed / 1 skipped / 0 failed across
  thirteen suites at the delta hashes, 05:05 UTC; 2 passed / 82 deselected
  for the two cover eligibility-integer cases, 05:12 UTC) or root-observed (9
  fake-child waiter tests passed); none was rerun here, by rule.
- The review-response hash moved from 94ec97a9... (05:06 UTC, the version the
  Codex documents cite) to 5298f201... through the append-only 05:12 UTC
  test-wiring correction, which the coordination decision section itself
  records; no source file changed. All 13 delta-table test files (with the
  corrected cover test `f53fe50dcbe70e8a6119c583b3d574c43caae76a57e5f7812f1a1163e2c41732`)
  and the diagnosis document hash as declared in the working tree at audit
  time (14/14); they are outside the frozen denominator by design.

## Check 6 - no-overlap process check

`ps -axo pid,pgid,etime,rss,command` at 2026-09-13T05:20:44Z, patterns
`run_gold11_|check_|pytest|cad_guided|_gold11_v[0-9]_exact_scoring|gold11_v9_execute_waiter`:
no matching line. The only Python process present is unrelated
(`/Users/idantal/tavnit/.venv/bin/python3 .venv/bin/tavnit serve --port 9017`,
pid 81241, elapsed 02-16:08:04). Machine memory `hw.memsize` = 137438953472
bytes (128 GiB); the protocol's 12 GiB campaign outer tree (12884901888) and
the repository's 16 GiB ceiling (17179869184) are below it, and available RAM
does not relax them. The coordinator must repeat a fresh no-overlap check
immediately before the waiter launch (the launcher repeats it too).

## Standing constraints (bundle and protocol)

The all-22 slot denominator (11 cases x 2 arms, review-source order, control
then challenger), the unchanged 71 reference lines (declared hashes above),
both fresh arms (control: V7 scan planes + schedule v1; challenger:
composite_v1 planes + schedule v2, same V16 engine), fixed budgets and limits,
retained negatives (refusals, timeouts, empty outputs contribute zero
predictions and keep their false negatives; null stays null) and no Gold in
inference (reference access only after all 22 durable terminals) are kept.

## Non-blocking observations (residual, recorded for POST)

1. The launcher prints `LAUNCHED` after a fixed 3 s sleep without confirming
   that the detached waiting parent claimed the reservation. After launch the
   operator must verify that `_outer_pid.txt` reads `RUNNING ... runner_pid=`
   and that the runner process exists; a refused waiter leaves the exit record
   `UNOBSERVED_PENDING`, and any later launch is refused by the existing
   records (a custody decision, never an automatic retry).
2. The waiting parent copies its own environment (including
   `GOLD11_V9_WAITER_TOKEN`/`GOLD11_V9_WAITER_CONFIG`) into the runner child;
   harmless under `python -I`, recorded for completeness.
3. No crash-proof (SIGKILL/power-loss/adversarial-filesystem) guarantee exists
   for the outer exit record; unobserved stays unobserved in the POST.
4. Author-reported and root-reported test results were not reproduced here; no
   test was run, by rule.
5. Runtime attestation is version metadata, not binary attestation, as the
   bundle itself declares; a separately named `cadquery_ocp_proxy` dist-info
   sits beside `cadquery_ocp` (one dist-info per named distribution).
6. The completion reviewer's stale-comment note (completion test line 475) is
   documentation-only and remains open as nonblocking.
7. `DRAFT_RELEASE_BLOCKERS` and `STAGE_TELEMETRY_STATUS` are unconsumed
   declarations; `require_release`/`verify_pre_run_audit`/
   `prestart_tree_admission` govern release.
8. The V9 protocol does not restate the campaign arithmetic; it holds under the
   frozen `LIMITS` (80460 s < 104400 s).
9. Process disclosure of this audit: a transient helper file was written to
   and deleted from the tmp output directory (outside the repository); the
   tmp V8 machine record was replaced by the V9 one (see Scope).

## Placement and one-use instructions for the coordinator

- Copy `PRE_RUN_AUDIT.json` verbatim to
  `docs/research/gold11_mesh_development_v9_attempt/PRE_RUN_AUDIT.json` as a
  regular file (no symlink; the runner opens it with `O_NOFOLLOW`, caps it at
  256 KiB, requires `decision`, `bundle_sha256`, a non-empty `reviewer` and an
  empty `blocking_findings` - all present, predicate replicated here - and
  re-reads it during binding). Verify its SHA-256
  `750c85f519d15598c504597555a33b30eb20aba30eef4a61dfd1c472e04a037b` after
  copying.
- Do NOT place this narrative report (or anything else) inside the attempt
  directory: `prestart_tree_admission` and the waiter's `PRISTINE_TREE` check
  admit exactly `FROZEN.json`, `PRE_RUN_AUDIT.json` and `source/` at the
  attempt root and refuse any other name. This report belongs in
  `docs/research/`.
- Run the fresh no-overlap check, then the single waiter launch above, once;
  confirm the reservation record shows `RUNNING` with the runner pid. On any
  failure preserve the tree and the outer records, diagnose and freeze a
  justified successor; never reopen this attempt.

## Authorization statement

This PRE binds the frozen source/runtime bundle (85 sources, bundle SHA-256
`c50a8cb9684ba891a97005755529d952229e25b6db83e4549a17cb830ca4b47d`), the raw17
path/hash declaration, the full11 case order, the unchanged 71-line reference
hashes (declared), both arm requests, the scorer sources, the limits, the
failure policy, the control envelope and the waiter launch path, and finds them
consistent. It is an engineering audit only: it is not manufacturing approval,
calibration, conformance, generalization or production evidence, and it
authorizes only the single scoped `--execute` of this frozen V9 attempt through
the bound waiter. Independent POST remains required before any score is
accepted.
