# Gold11 full11 V10 - independent physical pre-run audit (PRE), 2026-09-13

Written 2026-09-13T16:20:51Z (UTC; local calendar date 2026-09-13).
Reviewer: `independent-ai-pre-auditor-v10` (Claude Fable 5.1, reviewer role; not
the author of the candidate, the freeze, the protocol or any review).
Attempt audited: `docs/research/gold11_mesh_development_v10_attempt` (frozen;
`FROZEN.json` mtime 2026-09-13 16:03 UTC, archive manifest `written_utc`
2026-09-13T16:04:02Z; not executed). Companion machine record:
`PRE_RUN_AUDIT.json`, 38389 bytes, SHA-256
`9c14099a88506210d471a326ccc5397d541ae5afea145713749acb1acb3cfa0b`
(`written_utc` 2026-09-13T16:16:42Z; pure ASCII; decodes with duplicate-key and
non-finite rejection; the runner's `verify_pre_run_audit` predicate was
replicated against it and holds). Audit window 16:06:02Z-16:20:51Z.

## Decision

**GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN.** Blocking findings: none.
Bundle SHA-256 (canonical JSON of the decoded `FROZEN.json` object, sort_keys,
separators `(",", ":")`, allow_nan=False, UTF-8), recomputed independently:
`ded97290231a6e8fab9ba1680790fd3feead45dde9c9912048bde41462546928` - equal to
the author-stated value; the `ensure_ascii` True/False variants are
byte-identical because the file is pure ASCII (56924 canonical bytes; the file
itself is 56925 bytes with a trailing newline).

This is an engineering audit of a frozen consumed-development attempt. It is
not manufacturing approval, calibration, conformance, generalization or
production evidence, and it establishes no F1 or physical improvement. It
authorizes only the single scoped `--execute` of this frozen attempt, exactly
as the frozen protocol's one-use boundary states, through the bound waiter:

```sh
.venv/bin/python scripts/gold11_v10_execute_waiter.py
# which runs, once: OMP_NUM_THREADS=1 .venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v10.py --execute
```

## Scope, method and what was not opened

Read-only. Nothing was executed except `shasum -a 256`, `wc`, `ls -la`, `cat`,
`sed -n`, `grep`, `diff -U0`, `tar -tvf` (listing only), `tar -xOf` (single
member payload streamed to stdout for hashing, 97 times),
`ps -axo pid,pgid,etime,rss,command`, `date -u`, `sysctl -n hw.memsize`,
`.venv/bin/python --version` (plus a `sys`/`platform` query of the same
interpreter) and `.venv/bin/python` snippets that hash files, parse JSON/AST,
and exec only the runner's top-level constant `Assign` nodes in an inert
namespace (`math`, `json`, `dict`, `list`, `tuple`; empty builtins); no runner
function was called, no module under `backend/` or `scripts/` was imported, no
test, freeze, verify, execute, worker, geometry or native job was run.
Not opened, stat'ed, hashed, sampled or traversed: `data/` and any
`.ply`/`.step`, `data/bend_corpus/annotations/*` (Gold and reference
containers), `/private/tmp/gold11-*`, `/var/folders/*`, `.env` or credentials,
`output/`, `outputs/`, pools/manifests, HOLDOUT10, incoming-email/EHO parts,
and every `docs/research/*_attempt*` other than the V10 attempt (the V9 PRE
report was read as the template; the V9 attempt directory was not opened).
Nothing in the repository was modified; git was not touched. Only the two
output files were written, both under
`/Users/idantal/.claude/jobs/1519fdc2/tmp/pre_v10/`.
Process disclosure: two scratch files were written in that same output
directory (outside the repository) and are left in place for traceability:
`_tar_listing.txt` (9954 bytes, SHA-256
`736ffa9b4b21af2a1a68ff338c5e0e7e1c6717bff12592148b4788cfef2502a5`, the
`tar -tvf` listing) and `_tar_member_payload_hashes.txt` (11220 bytes, SHA-256
`8c998f9391b149dfd1b52e696b817ac78f52c2042167014b5a733f4fedffa0db`, the
streamed member payload hashes). They are audit working files only and must
never be copied into the attempt directory. The pre-existing tmp-root
`PRE_RUN_AUDIT.json` (27755 bytes, the V9 record) was left untouched.

## Check 1 - frozen bundle vs runner constants and live environment

- `FROZEN.json`: 56925 bytes, mode 0600, file SHA-256
  `dd93ccdd9ed0a90c85247b0b76859836f4e05b776e66c492713076592574b980` (equal
  to the author-stated value; below the runner's 256 KiB read cap); decodes
  with duplicate-key and non-finite rejection; 24 top-level keys; `schema` =
  `GOLD11_MESH_DEVELOPMENT_V10`; `production_use_allowed` = false;
  `historical_comparability` = `UNESTABLISHED`.
- `source_sha256`: 96 entries (none under `data/`, none `.ply`/`.step`). For
  every entry the working-tree file (`/Users/idantal/ShermanQC/FullProject/<path>`),
  the snapshot copy (`.../gold11_mesh_development_v10_attempt/source/<path>`)
  and the declared hash agree: 96/96, 0 mismatches. The snapshot tree holds
  exactly those 96 regular files (backend 65, scripts 25, docs 4, tests 2;
  0 extra, 0 missing, no symlinks; every snapshot file mode 0600; 3724627
  bytes in total).
- `SOURCES` tuple of the frozen `scripts/run_gold11_mesh_development_v10.py`
  (parsed with `ast`; 52 top-level constant assignments exec'd inertly;
  `ROOT`, `RUN`, `CATALOG`, `STAGE_SAMPLE_SOURCE_PATHS`,
  `CONTEXT_REQUESTS_BY_ARM` skipped as non-inert and read as text): 96
  elements, 96 distinct, key set identical to the `source_sha256` map; it
  includes `scripts/gold11_v10_execute_waiter.py`, the V9 basis, the protocol
  and the review script. Relative to the V9 `SOURCES` (85) it adds twelve
  entries (context V3, registration V5, mesh offline V17, native bridge V1,
  initializer V16, cover V4, the V10 design and protocol, contact V4,
  projection V5, the V10 waiter, the V10 runner) and drops one (the V9
  protocol); the eight predecessor runners V2-V9 are retained as preservation
  oracles.
- `SOURCE_BASIS` = `scripts/run_gold11_mesh_development_v9.py`,
  `SOURCE_BASIS_SHA` =
  `4567a0f584e15b75339238b3f50ebb19209f127bfcb3a55e5fa73fd36b4a85b1`; the
  working-tree V9 runner, the snapshot V9 runner and the frozen map entry all
  hash to that value (181565 bytes); `source_profile.unchanged_source_basis*`
  carry the same path and hash.
- Frozen V10 runner: 188070 bytes, SHA-256
  `f7fd083f8b74a22e7fd1afe872ae8b9b2bfbc94f4b7024fc8c2b46494208da92` (working
  tree, snapshot, frozen map, the author's closure table and the Codex
  runner-declaration closure agree).
- Runner constants equal the bundle for every compared field: `source_profile`,
  `config`, `limits`, `certificate_caps`, `arm_definitions`, `arms`,
  `comparator`, `two_plane_declaration`, `config_delta_from_v9`,
  `measurement_policy_declaration`, `pose_admission_declaration`,
  `position_error_declaration`, `scoring_transport`, `artifact_policy`,
  `retained_control_policy`, `catalog_sha256`, `selected_reference_sha256`
  (17/17 equal after JSON round trip) plus the `schema` literal
  `GOLD11_MESH_DEVELOPMENT_V10` in `freeze()` (line 1071) and `verify`
  (line 1100). `COMPARATOR` =
  `FRESH_V17_ALLOWANCE_V1_BRIDGE_V1_KERNEL_V2_V7_SCHEDULE_V1_VS_COMPOSITE_SCHEDULE_V2_NATIVE_V1`.
- `arm_requests`: `common_config` equals `CONFIG` for both arms; the identities
  `workflow`, `implementation`, `initialization_implementation`,
  `registration_implementation`, `two_plane_context_mode`,
  `native_adapter_version` and `scan_plane_mode` equal `ARM_DEFINITIONS`, and
  `two_plane_schedule_mode` equals `schedule_request.mode` (v1 control / v2
  challenger); `registration`, `measurement` and `inference` are descriptive
  identities not carried in the request. Rebuilt from the exec'd constants by
  replicating the `arm_request()` composition (lines 550-568; read, not
  called; `CONTEXT_REQUESTS_BY_ARM` replicated as the JSON deep copy of
  `CONTEXT_V4_REQUEST` per arm that line 426 declares; `encoded` = sort_keys,
  compact separators, allow_nan=False): canonical bytes equal the bundle's for
  both arms. Canonical SHA-256 of each frozen arm request: control
  `463dab0ea3bcf8cb4c79731b0d91a3ad0a8e99bf3ff4f1f193d5c0010ac3077b` (10055
  bytes; scan planes v7, schedule v1, `scan_plane_configuration` null),
  challenger `88380c13c3268116cdd6ac62e051db66b540b483ab906a3c2a3d057cbf237ec8`
  (11323 bytes; composite_v1 planes, schedule v2); all-arm digest
  `39b8db734ef6a79b6e564ac5fedb81396ca4cc4888377fe1e6b70351e0488713`. Both
  context mode v4; both `implementation` =
  `CAD_GUIDED_MESH_OFFLINE_V17_CONTEXT_V4_COMPLETION_V1_KERNEL_V2_NATIVE_BRIDGE_V1_ALLOWANCE_V1`;
  both `initialization_implementation` =
  `CAD_GUIDED_PLANE_INITIALIZATION_V16_CONTEXT_V4_COMPLETION_V1_ALLOWANCE_V1`;
  both `registration_implementation` =
  `SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V5_CHUNKED_QUERIES_REUSE_V1_ALLOWANCE_V1`;
  both `context_request.budgets.max_extrema_coordinate_visits` = 512000000.
- Declared configuration deltas are present in the frozen values: NEW
  `CONFIG["refinement_allowance"]` = {`enabled` true, `main_refined_groups` 12,
  `completion_refined_groups` 4, `main_query_rows` 6000000,
  `completion_query_rows` 3000000, `cpu_budget_seconds` 1860.0,
  `cpu_guard_seconds` 1500.0, `cooperative_chunk_rows` 32768};
  `cpu_budget_seconds` == `LIMITS["case_cpu_seconds"]` == 1860 and
  `cpu_guard_seconds` 1500 < 1860. The corrected
  `completion_budget_declaration` lists 13 active fields (the V9 ten plus
  `max_node_visits`, `max_triangle_pairs`, `max_normal_pairs`), 2 inactive
  fields (`max_prefilter_unsigned_assignments`, `max_refined_start_groups`), a
  new `finite_query_charging` statement, a new `allowance_interaction`
  statement and reworded `inactive_scope` / `work_units_formula` prose;
  `completion_budgets` and `initialization_budgets` are byte-identical to the
  V9 runner (15 fields each).
- CONFIG against the V9 runner (working-tree `scripts/run_gold11_mesh_development_v9.py`,
  same inert extraction, leaf-level diff): exactly 7 leaf differences, all
  inside `refinement_allowance` (added) and `completion_budget_declaration`
  (the two lists, the two reworded strings, the two added strings); every
  other CONFIG value is identical - in particular `measurement.max_fit_residual_mm`
  0.03, `measurement.plane_exclusion_mm` 0.04, `support.max_angular_gap_deg`
  30.0, `layer_policy`, `completion_policy`, `surface_budgets.max_ties_per_query`
  256, `workflow.max_measurement_contexts` 8, and the 1.0 mm pose admission
  gate (`registration.max_inlier_plane_p95_mm` = 1.0 =
  `POSE_ADMISSION_DECLARATION.max_inlier_plane_p95_mm`).
  `CONFIG_DELTA_FROM_V9` carries seven keys (`refinement_allowance`,
  `native_feature_bridge`, `query_reuse`, `completion_budget_declaration`,
  `worker_stage_checkpoints`, `inherited_v8_to_v9`, `inherited_v7_to_v8`);
  `MEASUREMENT_POLICY_DECLARATION` (`chosen_from_gold` false,
  `tuned_to_case_identities` false) is identical to V9.
- `LIMITS` is identical to the V9 runner (19 values, listed under Check 4) and
  `CERTIFICATE_CAPS`, `ARTIFACT_POLICY`, `RETAINED_CONTROL_POLICY`,
  `POSE_ADMISSION_DECLARATION`, `POSITION_ERROR_DECLARATION`,
  `TWO_PLANE_DECLARATION`, `MEASUREMENT_POLICY_DECLARATION`,
  `CONTEXT_V4_REQUEST`, `SEEDED_PLANE_BUDGETS`, `COMPOSITE_PLANE_BUDGETS`,
  `SCHEDULE_V2_BUDGETS` and `ARMS` are identical too. Changed besides CONFIG:
  `ARM_DEFINITIONS` (V17/V16/V5 identity strings, 10 leaves), `COMPARATOR`,
  `SOURCE_PROFILE` (`implementation` V10, `scorer_change` true with
  `scorer_change_scope`, `source_delta`, basis), `SCORING_TRANSPORT.wrapper_schema`
  V4 -> `GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V5`,
  `STAGE_LOG_DECLARATION` (+`checkpoint_stdout_bytes` 65536,
  +`checkpoint_cap_notice_bytes` 128, +`checkpoint_scope`,
  `other_output_remaining_bytes` 520192 -> 454528; the five byte fields still
  sum to the unchanged `whole_leaf_log_bytes` 1048576), `SCORING_SOURCES`
  7 -> 9 (+contact V4, +projection V5), `LEAF_SCHEMA` =
  `ONE_USE_PARENT_BOUND_LEAF_V10`, `TIMING_RELEASE_STATUS` =
  `REVIEWED_V10_EXACT_TIMING_RELEASE`, the first `DRAFT_RELEASE_BLOCKERS`
  entry.
- Helper pins equal the frozen map: supervision
  `scripts/gold11_owned_process_supervision_v3.py`
  `9eec779c6ba14c969fb37950aac2e7aab8af844d3f573233b1b22c92991472ea`; terminal
  artifacts `scripts/gold11_terminal_artifacts_v4.py`
  `3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838`
  (= `artifact_policy.helper_sha256`); stage sampler
  `scripts/gold11_engine_stage_sampler_v1.py`
  `80316d0026bcc2fabcf1d8861d010d5f434f720c303f5969384cc45f7809b6b0`; all
  nine `SCORING_SOURCES` (f1 contract v2, evaluate_f1, mesh scoring v1,
  contact v2/v3/v4, projection v1/v4/v5) are in the frozen denominator; 19
  helper/predecessor pins checked against the map (V9 and V8 runners, V9
  waiter, envelope V7, registration V4, surface V4, stage policies V2, native
  features r0/v1, mesh offline V16, initializer V15, context V2, cover V3,
  measurement r0/v2 and the three helpers) all equal.
- Release gate constants: `RELEASE_AUTHORIZED` = True, integer CPU limits,
  terminal reserve 2400 s below the 104400 s campaign wall;
  `STAGE_SAMPLE_SOURCE_PATHS` (line 145-146) names
  `scripts/run_gold11_mesh_development_v10.py`; `RUNNER_SOURCE` likewise.
- Bounding the runner delta: `diff -U0` of the frozen V9 runner against the
  frozen V10 runner yields 59 hunks (370 diff lines), all of the reviewed
  classes: header; attempt/protocol/basis paths; `SOURCE_PROFILE`; `SOURCES`
  additions; `TIMING_RELEASE_STATUS`; `STAGE_SAMPLE_SOURCE_PATHS`;
  `STAGE_LOG_DECLARATION` checkpoint fields; `DRAFT_RELEASE_BLOCKERS`;
  `ARM_DEFINITIONS`/`COMPARATOR`; `CONFIG_DELTA_FROM_V9`; the two CONFIG
  deltas; `SCORING_SOURCES` and wrapper schema V5; V10 error codes;
  `arm_request` identities; the private scoring namespace
  `_gold11_v10_exact_scoring`; projection/contact V5/V4 function names;
  freeze/verify schema and delta keys; `LEAF_SCHEMA`/`RUNNER_SOURCE`; private
  cache prefixes `gold11-v10-`; engine wiring imports (mesh offline V17,
  initializer V16, registration V5 `RefinementAllowanceV1` constructed from
  `CONFIG["refinement_allowance"]` and refused with
  `FROZEN_REFINEMENT_ALLOWANCE_BINDING` unless its encoding equals the frozen
  map, `cpu_budget_seconds` equals `LIMITS["case_cpu_seconds"]` and `enabled`
  is True, lines 1536-1539); the new bounded `stage_checkpoint_printer`
  (lines 1625-1647, cap 65536 bytes, one cap-notice line, unencodable record
  omitted whole); report/custody/comparison schema identities and the
  `comparison_scope_note` naming the historical 0.4580 as non-comparable. No
  hunk touches `LIMITS`, custody, supervision, terminal fill, scoring
  transport sizes, the matcher, the pose gate or any other CONFIG value.
- `runtime_metadata` equals the live environment: `.venv/bin/python` is a
  symlink to `/Users/idantal/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3.11`,
  whose directory is itself a symlink to `cpython-3.11.15-macos-aarch64-none`,
  i.e. the bundle's `python_executable`
  `/Users/idantal/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11`;
  version `3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]`, platform
  darwin, machine arm64 (`macOS-26.5-arm64-arm-64bit`), numpy 2.4.6, scipy
  1.17.1, cadquery-ocp 7.9.3.1.1; exactly one dist-info of each named
  distribution is discoverable under `.venv/lib/python3.11/site-packages`
  (`numpy-2.4.6`, `scipy-1.17.1`, `cadquery_ocp-7.9.3.1.1`; a separately named
  `cadquery_ocp_proxy-7.9.3.1.1` also exists). Attestation is version metadata
  only, as the bundle declares.

## Check 2 - case authority (frozen review source only)

- `source/scripts/review_gold11_inputs_v1.py`: 25616 bytes, SHA-256
  `41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955`, equal to
  the runner's `REVIEW_SHA` (line 54), to the frozen map entry, to the working
  tree and to the protocol (line 70).
- Exactly one `CASES` assignment, of the form `json.loads(<string literal>)`
  (string recovered with `ast.literal_eval`, then `json.loads`; no import):
  11 cases, 11 distinct `scan_sha256`, 6 distinct `cad_sha256`, 11 distinct
  case tokens. The runner's `approved_cases_from_review` (lines 1031-1049)
  binds exactly this shape and these counts.
- `{scan_path: scan_sha256, cad_path: cad_sha256}` over all cases: 17 distinct
  absolute paths; canonical SHA-256 (sort_keys, separators `(",", ":")`)
  `2459f4e93fe4204889ade6f8a4150d3543ec9f96625f39aaec811f003db4d157` -
  IDENTICAL to the V9 PRE's raw17 digest as quoted in the ledger
  `docs/research/2026-09-12_fable51_progress.md` (lines 227-228 and 490, the
  05:35 UTC entry). The method was reproduced exactly: the compact
  sorted-key canonical form gives the quoted value, the default-separator and
  indented variants do not.
- Canonical digest of the case list:
  `943268be4ee3d33301ab44dfb3e6f85fb0d34de852f788464f1d438ffe43813e`,
  identical for the review source and `FROZEN.json["cases"]` (`==` over the
  full list: same order, same tokens, part keys, reference ids, paths and
  sha256 declarations) and IDENTICAL to the V9 PRE's case-list digest.
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
  protocol (lines 72, 74). These are declarations only; the runner verifies
  them in `selected_references()` at scoring time (line 1865: 71 reference
  lines and the selected-snapshot digest). No raw scan/CAD file and no Gold
  or reference container was opened by this audit.

## Check 3 - attempt tree and external archive

- Attempt directory top level: exactly `FROZEN.json` (mode 0600) and `source/`
  (subdirectories `backend`, `docs`, `scripts`, `tests`). No `STARTED.json`,
  `COMPLETED.json`, `PRE_RUN_AUDIT.json`, `FINAL_GATE.json`,
  `FINALIZATION_FAILED.json`, `COORDINATOR_DISPATCH.json`, `cases`,
  `leaf_dispatch` or `coordinator_work` marker exists.
- Snapshot: 96 regular files under `source/` (backend 65, scripts 25, docs 4,
  tests 2), none outside the map, no symlinks, all mode 0600.
- External archive
  `docs/research/gold11_mesh_development_v10_physical_frozen.q4bxmlh3/SOURCE.tar`
  (directory mode 0700, holding `SOURCE.tar` and the author's manifest):
  SHA-256 `2cb7d6fd17012b23e539d754d3dfdea779fc334ad8e2524a74a5ef43bb36e255`,
  3858944 bytes (both equal the author's stated figures). `tar -tvf` lists
  103 entries: 6 directory entries (`source/`, `source/tests/`,
  `source/backend/`, `source/docs/`, `source/scripts/`,
  `source/docs/research/`) and 97 file members = `FROZEN.json` plus exactly
  the 96 `source/<path>` members (0 extra, 0 missing); no AppleDouble `._`
  entry, no non-regular entry; every listed member size equals the on-disk
  snapshot size (97/97); all file members are mode `-rw-------`.
- Member payloads: every one of the 97 file members was streamed with
  `tar -xOf` and hashed; payload SHA-256 == on-disk snapshot file SHA-256 ==
  declared hash (`bundle["source_sha256"]` for the 96 sources, the file hash
  `dd93ccdd...` for `FROZEN.json`): 97/97, 0 mismatches. Nothing was
  extracted to disk.
- Author manifest `SOURCE.tar.members.sha256.json`: 37841 bytes, SHA-256
  `e62eda42f1313784792fd1b2ce55408782890d97ba4b8f65dab27458dbe4290c` (equal to
  the author-stated value); schema
  `GOLD11_V10_FROZEN_ARCHIVE_MEMBER_PAYLOAD_HASHES_V1`, `written_utc`
  2026-09-13T16:04:02Z; 97 member records, each carrying `payload_sha256`,
  `on_disk_sha256`, `bytes` and `mode`: all 97 agree with this audit's
  independent streamed hashes, on-disk sizes and modes; its
  `bundle_sha256_canonical` and `frozen_json_file_sha256` equal the values
  above; `mismatches`, `missing_from_archive`, `unexpected` and
  `extra_in_archive` are empty; `source_count_archived` =
  `source_count_declared` = 96.
- The three outer records (`..._outer_exit.json`, `..._outer_pid.txt`,
  `..._outer_stdout.txt`) do NOT exist beside the attempt directory.

## Check 4 - protocol and delta design (frozen copies)

- Protocol `docs/research/2026-09-13_gold11_mesh_development_v10_protocol.md`
  13127 bytes, SHA-256
  `e59aff47a8312bfce44139a0a1977a33dff4a0a86039502b8c34fe5f1f27889b`; delta
  design `docs/research/2026-09-13_gold11_mesh_development_v10_delta_design.md`
  17298 bytes, SHA-256
  `0a68d42ac50264cff41189b67960a6f42d4d282c761c227bee11989b2e23d23c`; both
  equal the frozen map, the working tree, the author's closure table and the
  Codex runner-declaration closure.
- Resource table (protocol lines 164-177) equals the runner's `LIMITS` and the
  bundle's `limits` value-for-value: per-case wall/CPU 3540/1860 s; allowance
  CPU guard 1500 s (CONFIG, inside the cap); worker sampled tree / coordinator
  HWM / outer tree 8/4/12 GiB (8589934592 / 4294967296 / 12884901888);
  scan/CAD input 64/8 MiB (67108864 / 8388608); leaf log / result 1/128 MiB
  (1048576 / 134217728); stage-checkpoint stdout 65536 bytes
  (`STAGE_LOG_DECLARATION.checkpoint_stdout_bytes`); projection and receipt
  8 MiB (`compact_case_bytes` 8388608; `scoring_transport.inner_projection_bytes`
  and `retained_schedule_wrapper_bytes` 8388608); scorer 180 s / 120 s / 2 GiB
  / 192 MiB request / 32 MiB result (2147483648 / 201326592 / 33554432);
  coordinator log 2 MiB (2097152) / parent read 180 s; campaign 104400 s /
  reserve 2400 s. Under these limits 22*3540 + 180 + 2400 = 80460 < 104400
  holds (the V10 protocol does not restate the arithmetic). `LIMITS` is
  unchanged from V9, as the protocol states.
- The declared deltas are described in both documents and match the runner
  constants: the refinement allowance values (design lines 51-55: 12 / 4 /
  6 000 000 / 3 000 000 / 1860 / 1500 / 32 768 = `CONFIG["refinement_allowance"]`;
  "one allocation for the whole completion lane"; not a raised cap); the
  corrected completion budget declaration with every budget VALUE unchanged
  (verified byte-identical to V9); the native feature bridge (cap 64, fail
  closed before CAD sampling, frozen factories untouched); exact bound query
  reuse with fatal integrity failure; bounded stage checkpoints (65536 bytes
  inside the unchanged 1 MiB leaf log); the explicit evaluated-subset
  registration contract consumed by context V3, cover V4, contact V4,
  projection V5 and the V17 binder. Unchanged from V9 per the protocol and
  verified in CONFIG: the 0.03 mm / 0.04 mm measurement policy, the 30 deg
  support gap, the completion lane, the kernel-V2 layer selection with EMPTY
  inner-face eligibility, the 1.0 mm pose-admission gate and all gate forms.
- Release boundary: the protocol names the waiter launch
  `.venv/bin/python scripts/gold11_v10_execute_waiter.py`, which runs once
  `OMP_NUM_THREADS=1 .venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v10.py --execute`
  (lines 206-207), the exclusive `O_EXCL` reservation of the three outer
  records, the one-time token / exclusive lock / still-pending admission of
  the waiting parent, the refusal of a direct `--waiter` invocation, and the
  outer exit record path
  `docs/research/gold11_mesh_development_v10_attempt_outer_exit.json` (line
  200; beside, never inside, the attempt directory); an unobserved exit is
  recorded as unobserved.
- Immutable authority hashes are quoted consistently: review script
  `41a79da1...` (protocol line 70, runner line 54, frozen map), catalog
  `4ad243c1...` (protocol line 72, runner line 56, bundle `catalog_sha256`),
  selected snapshot `d16390b2...` (protocol line 74, runner line 57, bundle
  `selected_reference_sha256`), basis `4567a0f5...` (protocol line 101,
  design line 12, runner line 59, bundle `source_profile`). The catalog was
  not opened.
- Scoring declaration: contact scoring V4 with the unchanged legacy
  one-to-one minimum-cost matcher (axis angle <= 5 deg, transverse offset
  <= 4 mm, span overlaps >= .60/.60) and the unchanged metric-aggregation
  contract; `source_profile.scorer_change` = true with the declared scope
  (V3 text + registration-evaluation contract check + one integer diagnostic
  `subset_registration_count`); null stays null; the historical 0.4580152672
  is contextual only.
- Expected retained negative outcomes (to be reported as such, never grounds
  for a change after launch): candidates omitted or expired by row quotas or
  the CPU guard stay unknown; an expired completion lane publishes no
  completion group; retained CPU kills inside one cooperative chunk or one
  face-index build; windows above 0.03 mm flange residual refuse `NOT_PLANAR`;
  single-layer, ambiguous-partition or incompatible-pair windows measure as
  generic unresolved carriers or refuse; sparse angular stations refuse;
  completion candidates may fail the prefilter, refinement or admission; CG65
  has no two-of-three parent; the CGE4B, CG64, CG191 and CG4FAF blockers
  persist. Positive geometry on every case is not a prerequisite.

## Check 4a - execute waiter (frozen copy, read-only)

- `source/scripts/gold11_v10_execute_waiter.py`: 9213 bytes, SHA-256
  `e7ade783ea5adae1d2a97f9d1696913dd614120209a8b875d97147ed6b1282bc` (working
  tree, snapshot, map, closure table and Codex runner-declaration closure
  agree).
- `ATTEMPT` (line 31) is `docs/research/gold11_mesh_development_v10_attempt`;
  `COMMAND` (lines 32-33) is `[<ROOT>/.venv/bin/python, -I, -S, -B, -u,
  scripts/run_gold11_mesh_development_v10.py, --execute]`, run with
  `cwd=<ROOT>` and `OMP_NUM_THREADS=1` set in the child environment (line
  166) - exactly the protocol's execute command. `PRISTINE_TREE` (line 34) =
  `FROZEN.json`, `PRE_RUN_AUDIT.json`, `source`; `TOKEN_ENV` / `CONFIG_ENV`
  (lines 35-36) = `GOLD11_V10_WAITER_TOKEN` / `GOLD11_V10_WAITER_CONFIG`;
  `OVERLAP_PATTERN` (line 37) =
  `run_gold11_|check_|pytest|cad_guided|_gold11_v\d+_exact_scoring` (the
  multi-digit namespace correction D3 is present); `SCHEMA` (line 38) =
  `GOLD11_V10_OUTER_EXIT_RECORD_V1`.
- `launch()` (lines 83-125): refuses on process overlap (its own name
  excluded, 99), on a non-pristine attempt tree (entry names must be exactly
  the `PRISTINE_TREE`; names only, 98), and when any of the three outer
  records exists (`os.path.lexists`, 97); then reserves `_outer_pid.txt`
  (`RESERVED <nonce> <stamp>`), `_outer_exit.json` (pending record with the
  nonce) and `_outer_stdout.txt` (empty) with `O_WRONLY|O_CREAT|O_EXCL`
  (lines 62-65, 101-103; a `FileExistsError` refuses, 96) BEFORE spawning the
  detached waiting parent (`start_new_session=True`, lines 112-114) with the
  one-time token and the configuration in its environment; a spawn failure
  atomically replaces the exit record with `UNOBSERVED_WAITER_LAUNCH_FAILED`
  (lines 118-122).
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
- Records live beside, never inside, the attempt directory (lines 46-47). The
  runner's own one-shot gate (`STARTED.json` exclusive publication and the
  terminal-marker checks, lines 1428-1449 and 2111-2131) is unchanged in
  substance. Limitation (also stated by the protocol and the Codex
  runner/declaration review): the waiter cannot be crash-proof - a SIGKILL or
  power loss leaves `UNOBSERVED_PENDING`, which the POST must keep as
  unobserved.

## Check 5 - independent reviews and closures on record

Documents read in full (working tree, hashed at audit time):

| SHA-256 | bytes | file |
|---|---:|---|
| ff6c94daa686b72335562888f0594a9218d3d369c103e94b41e4cd6c71b49e5b | 13742 | 2026-09-13_fable51_v10_review_response.md (author-stated value equal) |
| 3e148087e0e4972921b6d9fa8db7b68413599df9d617be950750515272405cf9 | 30845 | 2026-09-13_fable51_v10_codex_coordination.md (author-stated value equal) |
| 7315fc11851cdfbb1a47967533cccde528cd44f72770a0b85edbaaefde0091a9 | 26079 | 2026-09-13_gold11_v10_allowance_codex_audit.md |
| 17920e5406695937ce6a985d550faf86f1b42ac32e244bc3edd9f87b3ea96a23 | 17956 | 2026-09-13_gold11_v10_query_reuse_codex_audit.md |
| 36b00f4b1346aec23c0543eb538e29251adadcb841c330fa6b04fcf01101684a | 15880 | 2026-09-13_gold11_v10_native_bridge_codex_audit.md |
| c3e5fdcd6a0684d09b09a496fdd08eacb1757f34918dcf092f6a15e1295d4684 | 14895 | 2026-09-13_gold11_v10_runner_declarations_codex_audit.md |

- The four Codex audit files hash exactly to the values the coordination
  record quotes for their accepted final states (allowance and runner
  declarations in the 16:01 UTC closure section, query reuse in the 15:42 UTC
  section, native bridge in the 14:49 UTC section): 4/4 match.
- The author's closure table (review response lines 40-64) holds 25 hashes.
  The 12 entries that are in `SOURCES` (native bridge V1 `fef3980a...`, mesh
  offline V17 `5c4fa6b8...`, registration V5 `ee05a03e...`, initializer V16
  `7dad12ae...`, context V3 `e67d1c35...`, cover V4 `f6972fd5...`, contact V4
  `c0bdfada...`, projection V5 `996fc87c...`, runner `f7fd083f...`, waiter
  `e7ade783...`, design `0a68d42a...`, protocol `e59aff47...`) equal the
  frozen bundle's `source_sha256` AND the working-tree hash: 12/12. The 13
  test files (not in `SOURCES` by design) equal their working-tree hashes at
  audit time: 13/13. Mismatches: none.
- Codex decisions on record: native bridge B1 - initial snapshot not accepted
  (P1: cap 16 against the 64-feature cover boundary; P2 evidence limits),
  append-only delta ACCEPT at bridge `fef3980a...` (unchanged since) with
  explicit 17/64/65 boundary tests, an independently constructed geometry
  case, the retained 1 mm-skin refusal and a real bridge fault through V17
  dispatch; query reuse B3 - not accepted (P1 corrupted retained evidence
  silently recomputed; P2 mutable owned copy), delta with one remaining
  ordering blocker, final ACCEPT of the different-valid-index fix (final V5
  pin `ee05a03e...` accepted in the coordination closure; "no B3 source
  blocker remains"); allowance B2/B4 - NOT GREEN (P1 rollback provenance, four
  P2s, vacuous acceptance tests), conditional delta acceptance, then the
  15:59 UTC final residual closure "ACCEPT ... No remaining finding" at V16
  `7dad12ae...`, V5 `ee05a03e...`, allowance test `cc34fc7f...`; runner
  declarations - `GREEN_BOUNDED_RUNNER_DECLARATIONS_WITH_NONBLOCKING_CORRECTIONS`
  (D1 allowance-check timing wording, D2 "unchanged scorer" wording, D3
  single-digit overlap regex), then the 16:00 UTC
  `ACCEPTED_BOUNDED_RUNNER_DECLARATION_DELTA_NO_BLOCKING_FINDINGS` at the
  frozen hashes, with the D2 residual shorthand in protocol line 141 recorded
  as nonblocking.
- Coordination decision (16:01 UTC, last section): "ACCEPT the final V10
  candidate for exclusive freeze and independent physical PRE. No remaining
  blocking finding in the reviewed V10 delta"; all 25 table hashes
  independently matched by root before and after the final review; the
  named next steps are the exclusive freeze, independent non-author PRE with
  verified archive member payload hashes, a fresh no-overlap check, ONE
  waiter-launched execute and independent POST; the decision itself is "not
  PRE, permission to skip PRE, an execution receipt, a new physical score".
  No unresolved P1 and no unresolved P2 is recorded.
- Test receipts are author-observed (885 passed / 1 skipped / 0 failed across
  thirteen suites, pytest exit 0, 15:52:07-15:55:11 UTC, with the full
  receipt history including one failed run at 15:18-15:20 UTC and the 13-pass
  B1 and 14-pass B3 focused receipts); root read but did not rerun them; none
  was rerun here, by rule.

## Check 6 - no-overlap process check

`ps -axo pid,pgid,etime,rss,command` at 2026-09-13T16:06:25Z and again at
2026-09-13T16:15:23Z, pattern
`run_gold11_|check_|pytest|cad_guided|_gold11_v[0-9]+_exact_scoring|gold11_v10_execute_waiter`:
no matching line (the first check matched only the grep process itself). The
only Python process present at both times is unrelated
(`/Users/idantal/tavnit/.venv/bin/python3 .venv/bin/tavnit serve --port 9017`,
pid 81241, elapsed 03-02:53:45 then 03-03:02:43). Machine memory `hw.memsize`
= 137438953472 bytes (128 GiB); the protocol's 12 GiB campaign outer tree
(12884901888) and the repository's 16 GiB ceiling (17179869184) are below it,
and available RAM does not relax them. The coordinator must repeat a fresh
no-overlap check immediately before the waiter launch (the launcher repeats
it too).

## Standing constraints (bundle and protocol)

The all-22 slot denominator (11 cases x 2 arms, review-source order, control
then challenger), the unchanged 71 reference lines (declared hashes above),
both fresh arms on the same V17 engine (control: V7 scan planes + schedule v1;
challenger: composite_v1 planes + schedule v2; the arm comparison is
scan-plane lane / schedule only, with timing-dependent CPU-guard stops
reported as such), fixed budgets and limits (unchanged from V9; the allowance
allocates work inside them and raises no cap), retained negatives (refusals,
timeouts, omissions, expiries and empty outputs contribute zero predictions
and keep their false negatives; omitted or expired candidates are UNKNOWN,
never rejected; null stays null), no retry/reopen/rescore of V9 or any earlier
terminal attempt, one heavy job, the historical 0.4580152672 as non-comparable,
and no Gold in inference (reference access only after all 22 durable
reconstructed terminals) are kept. This audit authorizes only the single
scoped execute through the bound waiter.

## Non-blocking observations (residual, recorded for POST)

1. The launcher prints `LAUNCHED` after a fixed 3 s sleep (line 123) without
   confirming that the detached waiting parent claimed the reservation. After
   launch the operator must verify that `_outer_pid.txt` reads
   `RUNNING ... runner_pid=` and that the runner process exists; a refused
   waiter leaves the exit record `UNOBSERVED_PENDING`, and any later launch is
   refused by the existing records (a custody decision, never an automatic
   retry).
2. The waiting parent copies its own environment (including
   `GOLD11_V10_WAITER_TOKEN`/`GOLD11_V10_WAITER_CONFIG`) into the runner child
   (lines 165-166); harmless under `python -I`, recorded for completeness.
3. No crash-proof (SIGKILL/power-loss/adversarial-filesystem) guarantee exists
   for the outer exit record; unobserved stays unobserved in the POST.
4. Author-reported test results were not reproduced here; no test was run, by
   rule.
5. Runtime attestation is version metadata, not binary attestation, as the
   bundle itself declares; a separately named `cadquery_ocp_proxy` dist-info
   sits beside `cadquery_ocp` (one dist-info per named distribution).
6. Protocol line 141 keeps the shorthand "unchanged scorer returns one"
   although `scorer_change` is declared true with its exact scope; the
   adjacent lines 128-131 state that only the matcher and metric-aggregation
   contract are unchanged. Recorded by the Codex runner-declaration closure as
   a nonblocking residual; it is not a scoring-rule change.
7. `DRAFT_RELEASE_BLOCKERS` and `STAGE_TELEMETRY_STATUS` are unconsumed
   declarations (only their assignments occur in the runner text);
   `require_release`/`verify_pre_run_audit`/`prestart_tree_admission` govern
   release.
8. `CONFIG.refinement_allowance.cpu_budget_seconds` / `cpu_guard_seconds` are
   floats (1860.0 / 1500.0) while `LIMITS.case_cpu_seconds` is the int 1860;
   numerically equal; the runner's worker-side binding compares numerically
   and the frozen registration V5 `RefinementAllowanceV1.__post_init__`
   (lines 126-139) accepts int or float for both fields, bounds the three
   integer quotas (12 in 1..48, 4 in 0..48, 32768 in 1024..200000) and
   requires guard < budget.
9. CPU-guard stops are timing dependent by declaration; arm differences may
   include guard-stop differences that are not mechanism differences; the
   realized trace (disposition ledgers, cost receipts, stage checkpoints) is
   to be published and bound by the POST.
10. `STAGE_LOG_DECLARATION.other_output_remaining_bytes` moved 520192 ->
    454528 to hold the 65536 + 128 checkpoint bytes inside the unchanged
    1048576 whole-leaf log cap (sum verified); no cap raised.
11. The archive directory holds the author's member manifest beside
    `SOURCE.tar`; all 97 manifest records agree with this audit's independent
    streamed hashes, sizes and modes.
12. The V10 protocol does not restate the campaign arithmetic; it holds under
    the frozen `LIMITS` (80460 s < 104400 s).
13. Process disclosure of this audit: two scratch files were written to the
    tmp output directory (outside the repository; hashes in Scope); the tmp
    root's V9 machine record was left untouched.

## Placement and one-use instructions for the coordinator

- Copy `PRE_RUN_AUDIT.json` verbatim to
  `docs/research/gold11_mesh_development_v10_attempt/PRE_RUN_AUDIT.json` as a
  regular file (no symlink; the runner opens it through `bounded_read` with
  `O_NOFOLLOW`, caps it at 256 KiB, requires `decision` ==
  `GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN`, `bundle_sha256` ==
  `digest(encoded(bundle))`, a non-empty `reviewer` and an empty
  `blocking_findings` - all present, predicate replicated here - and re-reads
  it during binding). Verify its SHA-256
  `9c14099a88506210d471a326ccc5397d541ae5afea145713749acb1acb3cfa0b` (38389
  bytes) after copying.
- Do NOT place this narrative report, the scratch files or anything else
  inside the attempt directory: `prestart_tree_admission` and the waiter's
  `PRISTINE_TREE` check admit exactly `FROZEN.json`, `PRE_RUN_AUDIT.json` and
  `source/` at the attempt root and refuse any other name. This report
  belongs in `docs/research/`.
- Run the fresh no-overlap check, then the single waiter launch above, once;
  confirm the reservation record shows `RUNNING` with the runner pid. On any
  failure preserve the tree and the outer records, diagnose and freeze a
  justified successor; never reopen this attempt.

## Authorization statement

This PRE binds the frozen source/runtime bundle (96 sources, bundle SHA-256
`ded97290231a6e8fab9ba1680790fd3feead45dde9c9912048bde41462546928`, file
SHA-256 `dd93ccdd9ed0a90c85247b0b76859836f4e05b776e66c492713076592574b980`),
the raw17 path/hash declaration, the full11 case order, the unchanged 71-line
reference hashes (declared), both arm requests, the refinement allowance, the
scorer sources, the limits, the failure policy, the control envelope, the
external archive (container and every member payload) and the waiter launch
path, and finds them consistent. It is an engineering audit only: it is not
manufacturing approval, calibration, conformance, generalization or production
evidence, and it authorizes only the single scoped `--execute` of this frozen
V10 attempt through the bound waiter. Independent POST remains required before
any score is accepted.
