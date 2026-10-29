# Gold11 full11 V8 - independent physical pre-run audit (PRE), 2026-09-13

Written 2026-09-12T23:47:26Z (UTC; local calendar date 2026-09-13).
Reviewer: `independent-ai-pre-auditor-v8` (Claude Fable 5.1, reviewer role; not
the author of the candidate, the freeze, the protocol or the reviews).
Attempt audited: `docs/research/gold11_mesh_development_v8_attempt` (frozen,
not executed). Companion machine record: `PRE_RUN_AUDIT.json`, 20736 bytes,
SHA-256 `41fd61fc6157e44823477e04da034d4b378307055cb325e0bcb19f1d7c7d6554`.

## Decision

**GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN.** Blocking findings: none.
Bundle SHA-256 (canonical JSON of the decoded `FROZEN.json` object, sort_keys,
separators `(",", ":")`, allow_nan=False), recomputed independently:
`43173e8a9c59c95c188e17c0c2901b8067fecf349ec363c05ab024c261fa9326` - agrees
with the value the freeze printed and with the progress ledger.

This is an engineering audit of a frozen consumed-development attempt. It is
not manufacturing approval, calibration, conformance, generalization or
production evidence, and it establishes no F1 or physical improvement. It
authorizes only the single scoped `--execute` of this frozen attempt, exactly
as the frozen protocol's one-use boundary states:

```sh
.venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v8.py --execute
```

## Scope, method and what was not opened

Read-only. Nothing was executed except `shasum`, `wc`, `diff`, `cat`, `sed`,
`grep`, `ls -la`, `tar -tvf` (listing only), `ps -axo pid,pgid,etime,rss,command`
and `.venv/bin/python` snippets that hash files, parse JSON/AST, and exec only
the runner's top-level constant `Assign` nodes in an inert namespace (`math`,
`json`, `dict`, `list`, `tuple`); no runner function was called, no module under
`backend/` or `scripts/` was imported, no test, freeze, verify, execute, worker
or geometry job was run. Not opened, stat'ed, hashed, sampled or traversed:
`data/` and any `.ply`/`.step`, `data/bend_corpus/annotations/*` (Gold and
reference containers), `/private/tmp/gold11-*`, `/var/folders/*`, `.env` or
credentials, and every `docs/research/*_attempt` other than the V8 attempt.
Only the two output files were written, both under
`/Users/idantal/.claude/jobs/1519fdc2/tmp/`.

## Check 1 - frozen bundle vs runner constants and live environment

- `FROZEN.json`: 39670 bytes, file SHA-256
  `03f58d23a3733434cfac6d51c031974088e660d0c880542ceb5c8182f3032961` (below the
  runner's 256 KiB read cap); decodes with duplicate-key and non-finite
  rejection; `schema` = `GOLD11_MESH_DEVELOPMENT_V8`; `production_use_allowed`
  = false; `historical_comparability` = `UNESTABLISHED`.
- `source_sha256`: 74 entries. For every entry the working-tree file
  (`/Users/idantal/ShermanQC/FullProject/<path>`), the snapshot copy
  (`.../gold11_mesh_development_v8_attempt/source/<path>`) and the declared
  hash agree: 74/74, 0 mismatches. The snapshot tree holds exactly those 74
  files (0 extra).
- `SOURCES` tuple of the frozen `scripts/run_gold11_mesh_development_v8.py`
  (parsed with `ast`, literal constants plus the three names `SOURCE_BASIS`,
  `REVIEW`, `PROTOCOL` resolved from their own literal assignments): 74
  elements, 74 distinct, key set identical to the `source_sha256` map.
- `SOURCE_BASIS` = `scripts/run_gold11_mesh_development_v7.py`,
  `SOURCE_BASIS_SHA` =
  `e910d562e55a62de80f5e448f944c239b5bdf95ef11c6d0b6f0b83b24bfc7a52`; the
  working-tree V7 runner and the snapshot V7 runner both hash to that value;
  the snapshot V7 runner is 169180 bytes as the protocol states.
- Frozen V8 runner: 172872 bytes, SHA-256
  `0e9aa2f64f3fc14e0da137687e44c8cd0dc0712f9e8bd783726df4187b1360c5` (working
  tree, snapshot and ledger final table agree).
- Runner constants (inert exec) equal the bundle for every compared field:
  schema, `source_profile`, `config`, `limits`, `certificate_caps`,
  `arm_definitions`, `arms`, `comparator`, `two_plane_declaration`,
  `config_delta_from_v7`, `pose_admission_declaration`,
  `position_error_declaration`, `scoring_transport`, `artifact_policy`,
  `retained_control_policy`, `catalog_sha256`, `selected_reference_sha256`
  (17/17 equal).
- `arm_requests`: rebuilt from the exec'd constants by replicating the
  `arm_request()` composition (read, not called); canonical bytes equal the
  bundle's for both arms. Canonical SHA-256 of each frozen arm request:
  control `b467981797a5c2f5303dcd462e1f0d116f4939fbfc18327b9fdf0cee05cf6565`
  (scan planes v7, schedule v1), challenger
  `2243dc4775a0571cc757a75d20bd97db75bfead1116faea7fc60d004062aa52b`
  (composite_v1 planes, schedule v2); both context mode v4, both
  `registration_implementation` =
  `SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V4`.
- Declared config deltas are present in the frozen values:
  `registration.max_inlier_plane_p95_mm` = 1.0, `surface_budgets.max_ties_per_query`
  = 256, `workflow.max_measurement_contexts` = 8, and
  `context_request.budgets.max_extrema_coordinate_visits` = 512000000 in both
  arm requests. `POSE_ADMISSION_DECLARATION.max_inlier_plane_p95_mm` = 1.0
  equals CONFIG.
- Helper pins equal the frozen map: supervision
  `scripts/gold11_owned_process_supervision_v3.py`
  `9eec779c6ba14c969fb37950aac2e7aab8af844d3f573233b1b22c92991472ea`; terminal
  artifacts `scripts/gold11_terminal_artifacts_v4.py`
  `3f4c575458b5033cf3d99baaeb980fb8591c03a54c55cbff0fbc881858115838`
  (= `artifact_policy.helper_sha256`); stage sampler
  `scripts/gold11_engine_stage_sampler_v1.py`
  `80316d0026bcc2fabcf1d8861d010d5f434f720c303f5969384cc45f7809b6b0`; all six
  `SCORING_SOURCES` (f1 contract v2, evaluate_f1, mesh scoring v1, contact
  scoring v2, projection v1, projection v3) are in the frozen denominator.
- Release gate constants: `RELEASE_AUTHORIZED` = True, `TIMING_RELEASE_STATUS`
  = `REVIEWED_V8_EXACT_TIMING_RELEASE`, integer CPU limits, terminal reserve
  2400 s below the 104400 s campaign wall.
- `runtime_metadata` equals the live environment: python executable resolves
  to `/Users/idantal/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3.11`,
  version `3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]`, platform
  darwin, machine arm64, numpy 2.4.6, scipy 1.17.1, cadquery-ocp 7.9.3.1.1;
  exactly one distribution of each is discoverable under
  `.venv/lib/python3.11/site-packages` (the runner refuses ambiguity).
  Attestation is version metadata only, as the bundle declares.

## Check 2 - case authority (frozen review source only)

- `source/scripts/review_gold11_inputs_v1.py`: SHA-256
  `41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955`, equal to
  the runner's `REVIEW_SHA` and to the frozen map entry.
- Exactly one `CASES` assignment, of the form `json.loads(<string literal>)`
  (string recovered with `ast.literal_eval`, then `json.loads`): 11 cases, 11
  distinct `scan_sha256`, 6 distinct `cad_sha256`, 11 distinct case tokens.
- `{scan_path: scan_sha256, cad_path: cad_sha256}` over all cases: 17 distinct
  absolute paths; canonical SHA-256
  `2459f4e93fe4204889ade6f8a4150d3543ec9f96625f39aaec811f003db4d157` (as
  required).
- Canonical digest of the case list:
  `943268be4ee3d33301ab44dfb3e6f85fb0d34de852f788464f1d438ffe43813e`, identical
  for the review source and `FROZEN.json["cases"]`; order and content identical.
- Case tokens in order: CG08506AE97D, CG191DEC7F81, CG1E0BBA2933, CG4BDD1371B6,
  CG4FAF273093, CG64C412CC73, CG65CD575B1C, CG84C8FA12D2, CGE4B0549E41,
  CGEBA9A869AC, CGF41F49066A. Families (`part_key`) in order: 10839000,
  49001000, 38172000, 37361005, 49024000, 40500000, 38172000, 10839000,
  10839000, 38172000, 38172000 (38172000 x4, 10839000 x3, 49001000, 37361005,
  49024000, 40500000 x1 each). Reference ids in order: corpus/03_10839000,
  corpus/10_49001000, shaked_xv/C05, corpus/05_37361005, corpus/11_49024000,
  corpus/08_40500000, corpus/06_38172000, corpus/01_10839000,
  corpus/02_10839000, corpus/07_38172000, shaked_xv/C06.
- Declared reference bindings in the runner constants: `CATALOG_SHA`
  `4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36`,
  `SELECTED_SHA`
  `d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc`; both
  equal the bundle's `catalog_sha256` / `selected_reference_sha256`. These are
  declarations only; the runner verifies them in `selected_references()` at
  scoring time (71 reference lines). No raw scan/CAD file and no Gold or
  reference container was opened by this audit.

## Check 3 - attempt tree and external archive

- Attempt directory top level: exactly `FROZEN.json` (mode 0600) and `source/`.
  No `STARTED.json`, `COMPLETED.json`, `PRE_RUN_AUDIT.json`, `FINAL_GATE.json`,
  `FINALIZATION_FAILED.json`, `COORDINATOR_DISPATCH.json`, `cases`,
  `leaf_dispatch` or `coordinator_work` marker exists.
- Snapshot: 74 regular files under `source/` (backend 53, scripts 17, docs 2,
  tests 2), none outside the map.
- External archive
  `docs/research/gold11_mesh_development_v8_physical_frozen.x9er0oPw/SOURCE.tar`
  (directory mode 0700, sole entry): SHA-256
  `35cad3685658cfab5f1ab824c438c6284a0ba47e7970d4fa812164065bfae103`, 2839040
  bytes (both equal the ledger's freeze entry). `tar -tvf` lists 81 entries: 6
  directory entries (`source/`, `source/tests/`, `source/backend/`,
  `source/docs/`, `source/scripts/`, `source/docs/research/`) and 75 file
  members = `FROZEN.json` plus exactly the 74 `source/<path>` members; every
  listed member size equals the on-disk snapshot size (75/75). Listing only;
  nothing was extracted.

## Check 4 - protocol and delta design (frozen copies)

- Protocol `docs/research/2026-09-13_gold11_mesh_development_v8_protocol.md`
  8794 bytes, SHA-256
  `a39e501a01890bfa73daec461a0f27bec8eebb166b5709c3d97248211beef385`; delta
  design `docs/research/2026-09-13_gold11_mesh_development_v8_delta_design.md`
  9410 bytes, SHA-256
  `060e3eaa595cda4e96ef1eb21fb16d74e7ab47f5f8a98c6d79920b13ae83a061`; both
  equal the frozen map and the ledger's final pre-freeze table.
- Limits table equals the runner's `LIMITS` and the bundle's `limits`
  value-for-value: per-case wall/CPU 3540/1860 s; worker sampled tree /
  coordinator HWM / outer tree 8/4/12 GiB (8589934592 / 4294967296 /
  12884901888); scan/CAD input 64/8 MiB (67108864 / 8388608); leaf log /
  result 1/128 MiB (1048576 / 134217728); projection and receipt 8 MiB
  (`compact_case_bytes` 8388608; `scoring_transport.inner_projection_bytes`
  and `retained_schedule_wrapper_bytes` 8388608); scorer 180 s / 120 s / 2 GiB
  / 192 MiB request / 32 MiB result (2147483648 / 201326592 / 33554432);
  coordinator log 2 MiB (2097152) / parent read 180 s; campaign 104400 s /
  reserve 2400 s. The protocol's arithmetic 22*3540 + 180 + 2400 = 80460 <
  104400 holds.
- The four declared config deltas are described in both documents
  (registration p95 gate 0.5 -> 1.0 mm, surface ties 32 -> 256, context
  extrema coordinate visits 256M -> 512M, NEW `workflow.max_measurement_contexts`
  = 8), the protocol now says "the four declared deltas". The pose-admission
  declaration is described: "Any pose admitted under the 1.0 mm gate is
  labelled a provisional development admission with unresolved false-pose
  uncertainty, never conformance"; the report carries
  `pose_admission_declaration`, `result_interpretation` and
  `arm_comparison_scope` (present in the frozen runner at lines 1905-1907).
  The design states the pose-class dedup applies regardless of the cap and
  records the 4S + 2P index-visit model.
- Expected retained negative outcomes (to be reported as such, never grounds
  for a cap change): CGE4B both arms (noisy-plane triangle-membership cap 20M);
  CG64/CG65 both arms (completed finite prefilter with no passing start);
  CG191, CG64, CG4FAF (`INVALID_NATIVE_SOLID`); per-case wall/CPU termination
  at 3540/1860 s; and the new class of measurement-consumer preflight refusals
  (`CONTEXT_WINDOW_PREFLIGHT_BUDGET`, `FULL_RAW_CROP_WORK_PREFLIGHT_BUDGET`,
  aggregate local-row budgets) from more contexts admitted under the 1.0 mm
  gate. Positive geometry on every case is not a prerequisite.

## Check 5 - independent reviews and ledger closures

Documents read in full (working tree, hashed at audit time):

| SHA-256 | bytes | file |
|---|---:|---|
| 436e21a89b18211fa9c5a5e08d0c83d51d2aea11cb00f0fee6b3c8efb4c9236c | 22035 | 2026-09-13_fable51_wiring_v14_v15_independent_review.md |
| 576007a86daf10bd51757b58f3c268c40d70fbcb6808bf221a237ec90acbce65 | 37980 | 2026-09-13_fable51_runner_v8_independent_review.md |
| 40ad51fbdab0beb8d0a971296e59028e3f0f6ff9599024e7d1d6904d3248defb | 14076 | 2026-09-13_fable51_review_delta_independent_recheck.md |
| 7f7b7ed886c7e086d7385c653a4155fe505928ab4993be1c4d7c05881088814e | 29806 | 2026-09-13_fable51_context_schedule_independent_review.md |
| 01766ee2fc4d5fa868c91c5094b7f38bd18fc442378767ae83b76eea33219a78 | 4739 | 2026-09-12_fable51_codex_review_receipt.md |
| c12f2da1fe3156d0d05e720ae18e5d323e16cf499f921449e8fbc636c0eeb7f0 | 17357 | 2026-09-12_fable51_progress.md |

- P1: none in any review. Unresolved P2: none.
  - Wiring V14/V15 review: no P1; P2-1 (wrong registration label untested),
    P2-2 (verbatim assertion omits the configuration copy) - closed per ledger
    22:58 UTC; P3-1..5 applied or noted.
  - Runner V8 review: no P1; P2-1 stale sampler path - closed (frozen
    `STAGE_SAMPLE_SOURCE_PATHS` names `scripts/run_gold11_mesh_development_v8.py`);
    P2-2 projection V3 test - closed (`tests/test_gold11_schedule_projection_v3.py`
    exists, ledger hash verified); P2-3 gate adversarial evidence - closed at
    the `_context_support` gate-test half, while the "two near-duplicate
    SUPPORTED contexts -> one cover prediction" half was superseded by the
    unconditional pose-class dedup schedule (reviewed with no P1); P2-4
    protocol outcome class - closed (present in the frozen protocol).
  - Review-delta recheck: no P1, no P2; P3-1..4 closed per ledger (leaf128
    figure retained, wording, test comment, 4S+2P accounting in the design).
  - Context-schedule review: no P1; P2-1 cap-branch tests, P2-2 unmocked cap
    assertions, P2-3 `SOURCE_PROFILE.source_delta` (fifth entry
    `WORKFLOW_MEASUREMENT_CONTEXT_SCHEDULE_POSE_CLASS_DEDUP_AND_EVIDENCE_RANKED_CAP_8`
    is in the frozen runner and bundle), P2-4 hash drift recorded - all closed
    per ledger 23:34 UTC; P3-1..6 applied or noted.
  - Codex receipt: items 1-4 closed at source (receipt "Subsequent bounded
    delta closure" and the recheck); the receipt itself grants no execution
    permission, consistent with this PRE being required.
- Reviewer-bound bytes vs frozen bytes: every review bound earlier bytes of
  V14 (eb1ad06a.../46248b94...), V15 (5edaae99.../681ad505...), the V8 runner
  (98ae00a0.../ff052f72...), protocol and design, because closures were applied
  after each review. The ledger's final pre-freeze hash table (20 rows) equals
  the frozen `source_sha256` for all 11 rows inside the frozen denominator and
  equals the working tree for all 20 rows (the 9 test-file rows are outside the
  frozen denominator by V7-text design and were verified in the working tree
  at audit time). The ledger's freeze entry states the bundle SHA-256, the
  `FROZEN.json` file SHA-256, the archive SHA-256 and size and "74 frozen
  sources" exactly as measured here.
- Bounding the post-review runner delta: a full `diff -U0` of the frozen V7
  runner against the frozen V8 runner yields 62 hunks, all of the reviewed
  classes (identity strings and error codes; engine wiring imports and
  constructors for V15/V14/context V4/surface V4/registration V4; projection
  V3 transport names; the four declared CONFIG deltas; declarations and
  report labels). No hunk touches custody, supervision, terminal fill, limits,
  scoring transport sizes, the matcher or any other CONFIG value.

## Check 6 - no-overlap process check

`ps -axo pid,pgid,etime,rss,command` at 2026-09-12T23:47:26Z, patterns
`run_gold11_|check_|pytest|cad_guided`: no matching line. The only Python
process present is unrelated (`/Users/idantal/tavnit/.venv/bin/python3
.venv/bin/tavnit serve --port 9017`, pid 81241, elapsed 02-10:24:11). The
coordinator must repeat a fresh no-overlap check immediately before
`--execute`, as the protocol requires.

## Non-blocking observations (residual, recorded for POST)

1. The V14/V15 closure deltas after the last review (46248b94 -> 9e3ca41e,
   681ad505 -> c8f38f49) cannot be byte-diffed because the pre-closure bytes
   are unrecoverable (untracked); the described closures are present in the
   frozen bytes (`_schedule_measurement_contexts`,
   `measurement_context_schedule_policy`, `pose_class_collapse`, the
   `MEASUREMENT_CONTEXT_CAP` phase, `finite_surface_implementation`,
   `residual_gate_value_scope`) and the ledger describes them as test,
   docstring, label and ordering changes. Accepted as a residual, not a blocker.
2. Runner review P2-3's second half (near-duplicate context cover test) is a
   test gap superseded by the dedup schedule; the physical run's retained
   ledgers remain the only efficacy evidence, as the protocol states.
3. The context-schedule review's binding table carries a truncated hash
   string for the recheck document (cosmetic; the ledger's full hash matches).
4. `DRAFT_RELEASE_BLOCKERS` and `STAGE_TELEMETRY_STATUS` are unconsumed
   declarations (per the reviews); `require_release`/`verify_pre_run_audit`
   govern release.
5. Author-reported test results (657 passed) were not reproduced here; no
   test was run, by rule.
6. Runtime attestation is version metadata, not binary attestation, as the
   bundle itself declares.

## Placement and one-use instructions for the coordinator

- Copy `PRE_RUN_AUDIT.json` verbatim to
  `docs/research/gold11_mesh_development_v8_attempt/PRE_RUN_AUDIT.json` as a
  regular file (no symlink; the runner opens it with `O_NOFOLLOW`, caps it at
  256 KiB and re-reads it during binding). Verify its SHA-256
  `41fd61fc6157e44823477e04da034d4b378307055cb325e0bcb19f1d7c7d6554` after
  copying.
- Do NOT place this narrative report (or anything else) inside the attempt
  directory: `prestart_tree_admission` admits exactly `FROZEN.json`,
  `PRE_RUN_AUDIT.json` and `source/` at the attempt root and refuses any other
  name. This report belongs in `docs/research/`.
- Run the fresh no-overlap check, then the single `--execute` above, once.
  On any failure preserve the tree, diagnose and freeze a justified successor;
  never reopen this attempt.

## Authorization statement

This PRE binds the frozen source/runtime bundle (74 sources, bundle SHA-256
`43173e8a9c59c95c188e17c0c2901b8067fecf349ec363c05ab024c261fa9326`), the raw17
path/hash declaration, the full11 case order, the unchanged 71-line reference
hashes (declared), both arm requests, the scorer sources, the limits, the
failure policy and the control envelope, and finds them consistent. It is an
engineering audit only: it is not manufacturing approval, calibration,
conformance, generalization or production evidence, and it authorizes only the
single scoped `--execute` of this frozen V8 attempt. Independent POST remains
required before any score is accepted.
