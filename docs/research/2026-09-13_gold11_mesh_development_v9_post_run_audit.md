# Gold11 full11 V9 - independent physical POST (execution custody and scoring ledger)

Date: 2026-09-13 (audit window 13:36-13:52 UTC by `date -u`). Reviewer:
`independent-ai-post-auditor-v9` (Claude Fable 5.1, reviewer role; not the
author of the candidate, the freeze, the PRE or the run).
Attempt audited: `docs/research/gold11_mesh_development_v9_attempt` (terminal;
frozen 2026-09-13 05:15:51 UTC; executed ONCE through the bound waiter
`scripts/gold11_v9_execute_waiter.py`). This is an engineering audit of
execution custody and of the retained negative scoring ledger. It is not
manufacturing approval, calibrated metrology, conformance, generalization or
production evidence. Every V9 outcome below is a COMBINED consumed-development
change relative to V8 (initializer V15 completion lane, measurement kernel V2
with layer selection and diagnostics, context measurement V2 / support cover
V3 / contact scoring V3 / projection V4 / report envelope V7 / mesh offline
V16, the derived measurement policy delta 0.05 -> 0.03 mm and e 0.075 -> 0.04
mm, and `support.max_angular_gap_deg` 5 -> 30 acting together on top of the
unchanged V8 mechanisms and the unchanged provisional 1.0 mm admission gate).
Nothing here is a mechanism-only speedup or a calibrated accuracy improvement,
the in-run arm comparison is scan-plane lane and scheduler only on the same V16
engine (not old engine versus new engine), and the historical F1 0.4580152672
is not comparable and is not beaten. No numeric F1 exists: the unchanged scorer
returned null for both arms and null stays null.

Method: read-only. Only `cat`/`sed`/`grep`/`shasum`/`wc`/`ls -la`/`diff`/
`tar -tvf` (listing only), one `ps -axo pid,pgid,etime,rss,command`, `date -u`,
`git status --porcelain` on named paths, and `.venv/bin/python` snippets that
read, parse (json) and hash files under the attempt, the job tmp directory and
the named repository documents (including the frozen `source/` snapshot, read
with `sed`/`grep` only). No runner, projection, scorer, test, freeze, verify,
execute, worker or geometry code was executed and no module under `backend/`
or `scripts/` was imported. Not opened, stat'ed, hashed, sampled or traversed:
`data/**`, any `.ply`/`.step`, the Gold and reference containers under
`data/bend_corpus/annotations`, `/private/tmp/gold11-*` (including the V9
coordinator cache named in the custody records), `/var/folders/*`, `.env`,
credentials, `output/`, `outputs/`, pools/manifests, HOLDOUT10,
incoming-email/EHO parts, and every other `docs/research/*_attempt` (V8 and
earlier are terminal and were not reopened). The attempt's own
`SELECTED_REFERENCES.json` (released derived snapshot) was read. V8 figures
come only from the accepted V8 POST
(`docs/research/2026-09-13_gold11_mesh_development_v8_post_run_audit.md`) and
its addendum. Process disclosure: one early `tar -tvf` listing was redirected
into a transient scratch file in the job tmp directory (outside the
repository); it was deleted immediately and the archive comparison was redone
in memory. This report is the sole retained write.

## Decision

ACCEPT the execution custody of the terminal V9 campaign and its complete
negative scoring ledger, as an engineering audit only, with the following
limits:

1. The parent process exit status is now OBSERVED (returncode 0, recorded by
   the waiting parent 52251 for runner 52252; check 1). The V8 limitation
   ("exit unobserved") is closed for V9. The runner's own release condition
   `OBSERVED_PARENT_EXIT_ZERO_FINAL_GATE_NO_VETO_AND_EXACT_INDEPENDENT_POST_AUDIT`
   is met by observation on its first two clauses (observed exit 0;
   `FINAL_GATE.json` present with no veto marker); its third clause (the
   exact independent POST) is the subject of this document and is for the
   coordinator to bind. The outer exit record proves the outer process return
   code only; it is not a score and not an audit (its own `scope` field says
   so).
2. Both arms emitted zero supports: TP 0 / FP 0 / FN 71, precision null,
   recall 0.0, F1 null (undefined, not zero), minimum-family recall 0.0. This
   is the unchanged scorer's own null, not a numeric result and not an
   improvement over V8's identical ledger.
3. Fourteen of the 22 workers were CPU-terminated at the frozen 1860 s limit
   (exit -24, no returned result; four in V8); their internal stages are
   unknown beyond sampled stacks.
4. The only slot that reached the measurement consumer (CG4BDD1371B6 control,
   one supported context) was refused before any window was attempted by a
   feature-type identity guard inside the frozen support cover V3
   (`SUPPORT_COVER_MEASUREMENT_FAILURE:ValueError:INVALID_SUPPORT_COVER_FEATURES`).
   Consequently the V9 measurement-side changes (kernel V2 both-flange
   diagnostics and layer selection, context measurement V2, cover V3, the
   0.03/0.04 mm policy and the 30 deg support gap) were NOT exercised on any
   physical window in this campaign; no per-window diagnostic exists and the
   material-side identity counts are null (no carrier), not zero. This is a
   retained zero-prediction outcome under the frozen failure policy, not a
   custody or ledger inconsistency, and it cannot be repaired on this terminal
   attempt (finding 3).
5. The coordinator private cache `/private/tmp/gold11-v9-coordinator-gtgs3ilh`
   is retained opaque outside the attempt (declared by the runner; not
   inspected here).
6. Raw scan/CAD bytes and the Gold catalog were not re-read here; input binding
   rests on the frozen declarations and each receipt's `source_binding_valid`.

No administrative failure, denominator exclusion, retry, resumed start, relaxed
limit, gate change after launch or edited frozen source was found (details in
Findings). Blocking findings: none.

## Bindings

Canonical digest = SHA-256 of `json.dumps(obj, sort_keys=True,
separators=(",", ":"), allow_nan=False).encode()` of the decoded object.
File SHA-256 values are of the raw bytes.

| Artifact (attempt root) | Bytes | File SHA-256 |
| --- | ---: | --- |
| FROZEN.json | 51207 | `67a253ad14702d15853942d70f94260d06e91713b76e4d84f26242ceed37a26e` |
| PRE_RUN_AUDIT.json | 27755 | `750c85f519d15598c504597555a33b30eb20aba30eef4a61dfd1c472e04a037b` |
| STARTED.json | 944 | `47db84ef21b924d52f012481e7c883ea62d7b876d937c636c55920fc2662c1b4` |
| COORDINATOR_DISPATCH.json | 221 | `72afc1dfa65c05536495cf06213430d5ee1898383bf919dd096e4056bd416e96` |
| COORDINATOR_VERIFIED.json | 10321 | `07385591a3976c8de8e2f7acddb5cdd5338e71eaadc17b60e8aa99bd61a30e8a` |
| COORDINATOR_COMPLETED.json | 185 | `d6533ccc7fed2f5e2f6e2a6756caecb147f942d06028fc6ac8eff4a010f9742c` |
| COORDINATOR_OUTPUT.txt | 3713 | `67c2b1c2e856aa5e2606e7d308ca7efec4ccaad88b43738105a1fdc1ac40068f` |
| INFERENCE_TERMINAL.json | 1257 | `620d29e6ad6d5c78f1f5705d31610198128040876a6f67876ee60c82420a41bb` |
| SCORER_RECEIPT.json | 3210 | `96f4507472d5e40417c2f36cfc6ed46a8ebbc480203a993a14a5c13ca2576a10` |
| SCORER_RESULT.json | 29758 | `f561f65c620e1d526129a7e3ab549ebcfba391f90a4f33e83c4b38218bbab63d` |
| SCORER_OUTPUT.txt | 115 | `0dc2fd1278bfcb072c15ff419aaac56aafd53080c90d94c724261abf07ba5820` |
| SCORER_ARTIFACT_RETENTION.json | 248 | `ea67bbd79fa8007af386631e036b68af22449bbbeef7bec2e3027d977edb1690` |
| SCORED.json | 106125 | `f89a9d7b29b0a18545ea5da10c9de23873dd37ada4af275131afd2841c43695c` |
| SELECTED_REFERENCES.json | 24751 | `d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc` |
| COMPLETED.json | 199674 | `792d35ad231db751c42f86a2326c8ff77d8ee8be74d4b8e1575aafd73fdbfe29` |
| FINAL_GATE.json | 1057 | `1226ec0dda6d63e322c6146d21affaf3d64f7185fb9d975ca57bb025ccc0e1eb` |

Outer records beside the attempt directory (`docs/research/`):
`gold11_mesh_development_v9_attempt_outer_exit.json` 542 B
`034ec01f2e2b1982bf5ecdd8f06001df74d7e8a8c23be1903df1675dfdad9194`;
`..._outer_pid.txt` 113 B
`96dae33782a1091ea204b6ad7993cf35e4ed50e4cc33e6b256065085d91c4d5f`;
`..._outer_stdout.txt` 466 B
`3611e7bae75c8a0bc59ccefb630e80b797ee2163b86dba7e6b11e8a757619fa6`. No
`.tmp.<pid>` leftover exists beside them.

## Check 1 - outer exit evidence (new versus V8)

- `_outer_exit.json`: `schema` `GOLD11_V9_OUTER_EXIT_RECORD_V1`, `scope`
  `OUTER_PROCESS_RETURN_CODE_ONLY_NOT_A_SCORE_OR_AUDIT`, `command` =
  `[.venv/bin/python, -I, -S, -B, -u, scripts/run_gold11_mesh_development_v9.py, --execute]`
  (the protocol's exact command), `nonce` `25e6c406b3f52ced8bddd9173b9d7345`,
  `observation` `OBSERVED_BY_WAITING_PARENT`, `observed_by`
  `WAITING_PARENT_PROCESS_52251`, `returncode` 0, `runner_pid` 52252,
  `started` `2026-09-13 05:36:07 UTC`, `finished` `2026-09-13 13:34:04 UTC`.
- `_outer_pid.txt` holds exactly one line:
  `FINISHED 25e6c406b3f52ced8bddd9173b9d7345 waiter_pid=52251 runner_pid=52252 returncode=0 2026-09-13 13:34:04 UTC`.
  The frozen waiter (`source/scripts/gold11_v9_execute_waiter.py`, 9179 B,
  `85913855908ac3a965cb37d7fdd36a8c3464aabd05621184ecd9b4b6e8107688`) rewrites
  this reservation file in place at each transition (RESERVED by the launcher,
  then CLAIMED, RUNNING and FINISHED by the waiting parent via
  seek/write/truncate under an exclusive `flock`), so the earlier states are
  not retained as separate lines; the FINISHED state is present and its nonce,
  pids, return code and timestamp equal the exit record. The RESERVED/RUNNING
  history is therefore inferred from the frozen source, not observed here.
- `_outer_stdout.txt` (466 B) is exactly the runner's final status line:
  `{"status": "COMPUTED_PENDING_INDEPENDENT_POST_AUDIT", "attempt": ".../gold11_mesh_development_v9_attempt", "release_allowed": false, "durable_terminal_case_records": 22, "final_campaign_inventory_sha256": "4c63a73d8c21ed72349b8db5bc2c7ba6a4d20a8dd91cef031dd127a8dd64b594", "final_campaign_inventory_entries": 284, "final_scientific_manifest_entries": 105, "final_inventory_stdout_is_not_durable_runner_artifact": true}`.
  In the frozen runner's `execute()` this line is printed only after
  `COMPLETED.json`, `FINAL_GATE.json` and the post-FINAL_GATE inventory
  succeed, and `return 0` follows one more wall/RSS veto whose failure would
  write `FINALIZATION_FAILED.json` (absent) and return 1; the observed
  returncode 0 is consistent with that path.
- Process identity: runner 52252 = `COORDINATOR_DISPATCH.parent_pid` =
  `COORDINATOR_VERIFIED.outer_pid`; coordinator 52253 in every leaf
  dispatch/claim record; first inference worker 52254; scorer worker 57813 =
  `SCORER_RECEIPT.child_pid`.
- Timestamps (UTC, from `lstat` mtime_ns): `PRE_RUN_AUDIT.json` placed
  05:35:55.324 (after the 05:15:51.713 freeze, before launch); waiter
  `started` 05:36:07; `STARTED.json` 05:36:08.028 (1.03 s later);
  `COORDINATOR_DISPATCH.json` 05:36:08.029; `COORDINATOR_VERIFIED.json`
  05:36:08.084; `COMPLETED.json` 13:34:03.848; `FINAL_GATE.json` 13:34:04.034;
  outer stdout 13:34:04.217, outer exit 13:34:04.247, outer pid 13:34:04.247;
  waiter `finished` 13:34:04 (1 s resolution). FINAL_GATE - STARTED = 28676.0 s;
  `COMPLETED.parent_wall_seconds` 28675.583; `FINAL_GATE.parent_wall_seconds`
  28675.770; outer span 28677 s. All consistent.
- Live check: `ps -axo pid,pgid,etime,rss,command` at 13:38:07 UTC matched no
  `run_gold11_|check_|pytest|cad_guided|_gold11_v[0-9]_exact_scoring|gold11_v9_execute_waiter`
  process.
- `COMPLETED.release_requires` =
  `OBSERVED_PARENT_EXIT_ZERO_FINAL_GATE_NO_VETO_AND_EXACT_INDEPENDENT_POST_AUDIT`;
  `FINAL_GATE.final_release_requires` =
  `POST_FINAL_GATE_FULL_TREE_RESCAN_OBSERVED_EXIT0_AND_INDEPENDENT_POST`;
  `FINAL_GATE.status` `PENDING_INDEPENDENT_POST_AUDIT`. Observed exit 0, no
  `FINALIZATION_FAILED.json`, no `SCORING_UNAVAILABLE.json`, no
  `coordinator_work`, no `RESULT_PARTIAL.bin`, no `SCORER_RESULT_PARTIAL.bin`
  anywhere in the tree. The exit observation is the waiting parent's (the
  author's launcher on the same host), verified here for internal consistency
  against the runner's own records; it was not observed by this reviewer.

## Check 2 - frozen bundle, PRE bindings, source drift, archive

- `FROZEN.json` 51207 B / `67a253ad...7a26e`; canonical(FROZEN) =
  `c50a8cb9684ba891a97005755529d952229e25b6db83e4549a17cb830ca4b47d` =
  `STARTED.bundle_sha256` = `COORDINATOR_VERIFIED.bundle_sha256` =
  `PRE_RUN_AUDIT.bundle_sha256` = `SCORED.bundle_sha256`. `schema`
  `GOLD11_MESH_DEVELOPMENT_V9`, `production_use_allowed` false,
  `historical_comparability` `UNESTABLISHED`, `selected_reference_sha256`
  `d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc`,
  `catalog_sha256`
  `4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36`.
- `PRE_RUN_AUDIT.json` 27755 B / `750c85f5...a037b`, byte-identical (`diff`)
  to the PRE copy in the job tmp directory; canonical(PRE) =
  `39d6d16222a72f207bcc6457d3e1d7ac1e971560e0a8d6149ced050e465e29c0` =
  `STARTED.audit_sha256` = `COORDINATOR_VERIFIED.audit_sha256`. PRE fields:
  `decision` `GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN`, `reviewer`
  `independent-ai-pre-auditor-v9`, `blocking_findings` `[]`, `written_utc`
  2026-09-13T05:30:51Z.
- Source drift (three-way): all 85 `FROZEN.source_sha256` entries (backend 59,
  scripts 21, docs 3, tests 2) equal the `source/` snapshot copy AND the
  working-tree file at `/Users/idantal/ShermanQC/FullProject`: 85/85, 0
  mismatches, 0 missing, 0 extra files in `source/` (85 regular files, 0
  symlinks, 6 directories). `COORDINATOR_VERIFIED.source_sha256` ==
  `FROZEN.source_sha256`; `COMPLETED.final_source_audit`: `complete` true,
  `unchanged` true, `runtime_metadata_unchanged` true, `errors` {}, `current`
  == `frozen` == `FROZEN.source_sha256` (85 entries), `controls` = the five
  file hashes {COORDINATOR_DISPATCH, COORDINATOR_VERIFIED, FROZEN,
  PRE_RUN_AUDIT, STARTED} equal to the table above. No working-tree drift of
  any frozen source was found. Frozen V9 runner 181565 B /
  `4567a0f584e15b75339238b3f50ebb19209f127bfcb3a55e5fa73fd36b4a85b1`; V8 basis
  172872 B / `0e9aa2f64f3fc14e0da137687e44c8cd0dc0712f9e8bd783726df4187b1360c5`.
- Runtime metadata equal in FROZEN and COORDINATOR_VERIFIED (python 3.11.15
  arm64 darwin, numpy 2.4.6, scipy 1.17.1, cadquery-ocp 7.9.3.1.1; version
  metadata, not binary attestation).
- External archive
  `docs/research/gold11_mesh_development_v9_physical_frozen.RdikX7AH/SOURCE.tar`:
  `412c9f8e662c845e172cdd19a134031c62a5c4a12866041fd82981da2c6c2326`, 3451904
  B, mtime 05:15:51.778 UTC (expected values). `tar -tvf` lists 92 entries: 6
  directories and 86 file members = `FROZEN.json` (51207 B) plus exactly the
  85 `source/<path>` members (0 extra, 0 missing); every member size equals
  the on-disk snapshot size (86/86); all file members mode `-rw-------`.
  Nothing was extracted.
- `STARTED.limits` == `FROZEN.limits` (case wall/CPU 3540/1860 s; worker
  sampled tree / coordinator HWM / outer tree 8589934592 / 4294967296 /
  12884901888; scan/CAD 67108864 / 8388608; log/result 1048576 / 134217728;
  projection and receipt 8388608; scorer 180 s / 120 s / 2147483648 /
  201326592 request / 33554432 result; coordinator log 2097152 / parent read
  180 s; campaign 104400 s / reserve 2400 s). All 22 case `STARTED.json`
  records carry the same limits, `no_resume` true and the frozen arm request
  digest (control
  `b1df28220670a37fc9964fac3b64d914d12c69dc40cc1ded9e9d0ef3403eeb14`, 9055
  canonical bytes; challenger
  `2bcf0f4a0c39ab867466445ebddb9ed2fc82bdc3462cfa17a0e597b2154ebb74`, 10323
  bytes); canonical(all arm requests) =
  `f6ca1b6c8f9d2388a243ae38783a65c8db97da9e3bf7b1e310b6d4c90fe589b5` =
  `STARTED.arm_requests_sha256` = `INFERENCE_TERMINAL.arm_requests_sha256`.
  `SCORED.arm_requests`, `arm_definitions`, `config_delta_from_v8`,
  `measurement_policy_declaration`, `pose_admission_declaration`,
  `scoring_transport`, `certificate_caps`, `two_plane_declaration`,
  `retained_control_policy` and `artifact_policy` are object-equal to FROZEN.
- Frozen config deltas present in the bundle and in every returned policy
  snapshot: `measurement.max_fit_residual_mm` 0.03, `plane_exclusion_mm` 0.04,
  `support.max_angular_gap_deg` 30.0 (axial gap 4.0, sweep 20.0, min points 3
  unchanged), `layer_policy` (8 fields, enabled), `completion_policy`
  (enabled, 256 parents, 64 per parent, 4096 total, 1000 mm shift),
  `completion_budgets` == `initialization_budgets` (15 fields) with the
  declaration of five inactive fields, registration gate 1.0 mm, ties 256,
  extrema coordinate visits 512000000, `workflow.max_measurement_contexts` 8.

## Check 3 - execution custody

- canonical(STARTED) = `b6e9193b34602e02eb59de707381f03367f57a503d9effc592da25f831f76eab`
  = `COORDINATOR_VERIFIED.started_sha256`; canonical(COORDINATOR_DISPATCH) =
  `f3ba006c4b615f0e7bdbe6ad536b9ebc9a2511c0007347d45f27b958d5f150f3` =
  `COORDINATOR_VERIFIED.coordinator_dispatch_sha256`; `COORDINATOR_VERIFIED`
  schema `ONE_USE_PARENT_BOUND_LEAF_V9`, canonical =
  `1de12f17f77ddf3c9b347a14da566296225d7141e7d8069ee659398cef16eb48` (the
  `context_sha256` of every leaf dispatch record).
- Coordinator dispatch: cache `/private/tmp/gold11-v9-coordinator-gtgs3ilh`,
  identity device 16777231 / inode 35839343 / uid 501, parent pid 52252;
  `COMPLETED.coordinator_private_cache_custody` names the same path and
  identity with status `OPAQUE_PRIVATE_CACHE_RETAINED_OUTSIDE_ATTEMPT`,
  `contents_enumerated_or_read` false (not inspected here either).
- COMPLETED: `status` `COMPUTED_PENDING_INDEPENDENT_POST_AUDIT`, `failures`
  `[]`, `parent_wall_seconds` 28675.58346658433, `parent_peak_rss_bytes`
  1262747648, `no_resume` true, `production_use_allowed` false; canonical
  `5e544c5dcf0e092c4cce6164ca385add0ac99a8876616593c9b4a8cd5ab6ae97` =
  `FINAL_GATE.completed_canonical_sha256`; canonical(final_source_audit) =
  `bc5b6c2fca1dd2a4e714fe836634251b45d43c2815027fce8a538651cad8bfab` =
  `FINAL_GATE.source_and_control_audit_sha256`; canonical(owned_artifact_custody)
  = `06ceca84b2c03800ff2b77e2997f124c91473089e3b690edbc3ba7acc0a20d15` =
  `FINAL_GATE.owned_artifact_custody_sha256`. FINAL_GATE
  `scored_report_sha256` = SCORED file hash, `scorer_result_sha256` =
  SCORER_RESULT file hash = `SCORER_OUTPUT.result_sha256` =
  `SCORER_ARTIFACT_RETENTION` artifact hash (29758 B,
  `COMPLETE_RESULT_BYTES_RETAINED`, no leftover partial),
  `reference_snapshot_sha256` `d16390b2...b45dc`, `final_gate_self_hash_claim`
  false, `inventory_boundary`
  `AFTER_COMPLETED_BEFORE_FINAL_GATE_WITH_EXACT_COMPLETED_BYTES`.
  `COORDINATOR_COMPLETED.json`: same status, `parent_wall_seconds`
  28653.912550666835, `no_resume` true, `historical_comparability`
  `UNESTABLISHED`.
- Coordinator receipt (supervision V3): exit_code 0, budget_stop null,
  child_pid 52253, wall 28653.993390665855 s, direct_child_reaped true,
  owned_group_empty_confirmed true (group 52253), requires_process_attention
  false, foreign escape / unknown identity false, cleanup/control errors [],
  553029 RSS samples, sampled tree peak 2357755904 B + parent 42270720 B =
  budget peak 2400010240 B (below 12 GiB), log 3713 B bound.
  `COORDINATOR_OUTPUT.txt` holds exactly the 22 case lines in frozen order
  (walls and `budget_stop` equal to the 22 receipts' `parent_wall_seconds` /
  null) plus the final status line.
- Owned artifact custody (`GOLD11_OWNED_ARTIFACT_CUSTODY_V9`): `complete`
  true, 105 declared files (391123124 B) re-hashed here, 0 mismatches; 24
  logs all `COMPLETE_LOG_BYTES_BOUND` (0 mismatches); `scoring_request_sha256`
  `ec7681548edf8f98552ac1b2d39c12d48c732c1bdc078682f31a7e73172ea64d`;
  campaign tree inventory `COMPLETE_CAMPAIGN_TREE_CUSTODY_V9` complete, 282
  entries (241 regular files: 105 scientific, 85 source, 46 dispatch, 5
  control; 41 directories), errors [], `deadline_expired` false,
  `first_unrecorded_path` null, read_bytes 394403818 (exact). All 282
  inventory fingerprints (device, inode, mode, links, bytes, mtime_ns,
  ctime_ns) equal the current `lstat` values and all 241 recorded file hashes
  equal current bytes; the only on-disk names outside that inventory are the
  two later publications `COMPLETED.json` and `FINAL_GATE.json` (284 total).
- Inventory reproduction: appending the current `COMPLETED.json` fingerprint,
  hash and bytes to the retained pre-COMPLETED inventory exactly as
  `verify_campaign_inventory` would (one more CONTROL slot and its 256 KiB cap,
  `published_control_names` ["COMPLETED.json"], +199674 read bytes, the name
  added to the root directory's expected/initial/final lists) reproduces
  `FINAL_GATE.completed_publication_inventory_sha256` =
  `a90474c1f6e85deaed04290ef1d0d1fcf61e9d6f0e62aa76223095f9ac7915c1` (283
  entries); adding `FINAL_GATE.json` (+1057 read bytes) reproduces the parent
  stdout's `final_campaign_inventory_sha256` =
  `4c63a73d8c21ed72349b8db5bc2c7ba6a4d20a8dd91cef031dd127a8dd64b594` with 284
  entries / 105 scientific. The final on-disk identities of both control files
  are therefore the ones the runner itself inventoried.
- Terminal fill: all 22 `terminal_case_records` durable, every record
  `PREEXISTING_VALID_NO_WRITE` (no fallback terminal was written).
- Leaf dispatch: `leaf_dispatch/` holds 46 regular files (the directory link
  count 48 reported by `ls -la` counts `.` and `..`) = 23 issued/claimed pairs
  (22 INFERENCE + 1 SCORER); every id equals canonical({role, binding}),
  `context_sha256` = canonical(COORDINATOR_VERIFIED), claim `dispatch_sha256`
  = canonical(issued record), `no_retry` true, coordinator pid 52253, worker
  pids equal the receipts' `child_pid` (23 distinct), private cache path and
  identity equal the receipts' custody records, 23 distinct caches / 23
  distinct inodes, all reported `REMOVED_AFTER_QUIESCENCE_AND_DERIVED_RETENTION`;
  each inference binding's scan/CAD hashes equal the frozen case declaration
  and its `arm_request_sha256` the frozen arm digest; the scorer binding's
  `request_sha256` equals `SCORED.scoring_request_sha256`; the 8 clean
  inference ids are exactly `required_leaf_dispatch_ids` (CG4BDD1371B6,
  CG65CD575B1C, CG84C8FA12D2, CGE4B0549E41, both arms).
- INFERENCE_TERMINAL: case_count 11, arm_count 2, terminal_record_count 22,
  `outcomes_sha256`
  `086fae714f1ca88a7bf24aecbf0a9a35e3b1f1aeb4d4a8017edfcd59767cae34`
  reproduced here as canonical({arm: {case: SCORING_INPUT object}}) from the
  22 retained `SCORING_INPUT.json` objects; 8 result bindings equal the
  receipts and the actual RESULT.json hashes, 14 null; parent_wall
  28633.066532082856 s (sum of the 22 per-case parent walls 28633.036 s).
- Scoring transport: `SCORED.projection_bindings` equal the receipts on
  arm_request_sha256 / scoring_input_sha256 / result_sha256 / projection_kind
  for all 22; `scorer_input_canonical_sha256` equals canonical(SCORING_INPUT
  .contact_projection) for the 8 wrapped projections
  (`GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V4` wrappers, inner
  `GOLD11_CONTACT_WORKFLOW_PROJECTION_V3`, `result_binding.matched` true with
  supplied bytes equal to the RESULT file size) and canonical(refusal) for the
  14 supervisor refusals (the exact six-key object, 159 B, one hash
  `3a4dd85d5df2c5ccbeac7047f13d17d607889b5e718b725ccd17d19ee1ddeea1` in all
  14). Rebuilding the V6 scoring request as `make_scoring_request` composes it
  (inner objects, the snapshot's `bends`, the frozen case->family map, the
  frozen arm requests, the scan-plane requests by arm, the projection bindings
  and the selected-reference digest) gives canonical
  `ec7681548edf8f98552ac1b2d39c12d48c732c1bdc078682f31a7e73172ea64d` (75179
  B) = `SCORED.scoring_request_sha256` = the scorer leaf binding =
  `owned_artifact_custody.scoring_request_sha256`.
  `SCORED.workflow_stage_attribution` equals the 22 receipts'
  `workflow_stage_diagnostics`. canonical(SELECTED_REFERENCES) = the file
  bytes = `d16390b2...b45dc` = `FROZEN.selected_reference_sha256`; 11 cases in
  frozen order, 71 bends (6,5,10,4,3,7,6,8,4,8,10), every case's
  `scan_sha256` equal to the frozen declaration, 9 `reference_mesh_raw` and 2
  `scan_raw` frames (CG1E0BBA2933, CGF41F49066A), equal to
  `SCORED.case_declared_coordinate_frames`. The catalog hash
  `4ad243c1...b36` is a declaration verified by the runner's
  `selected_references()` at scoring time; the catalog was not opened here.
- Scoring-after-terminal ordering (epoch ns from the inventory fingerprints,
  equal to current mtimes): last case terminal
  `cases/CGF41F49066A/challenger/COMPLETED.json` 1789306401384533473
  (13:33:21.385Z); `INFERENCE_TERMINAL.json` 1789306401385882479
  (13:33:21.386Z); `SELECTED_REFERENCES.json` 1789306421982833440
  (13:33:41.983Z, +20.597 s); scorer dispatch issued 13:33:41.990 and claimed
  13:33:42.030; `SCORER_OUTPUT.txt` 13:33:42.201; `SCORER_RESULT.json`
  13:33:42.218; `SCORER_ARTIFACT_RETENTION.json` and `SCORER_RECEIPT.json`
  13:33:42.219; `SCORED.json` 13:33:42.231; `COORDINATOR_COMPLETED.json` and
  `COORDINATOR_OUTPUT.txt` 13:33:42.232; `COMPLETED.json` 1789306443848437987
  (13:34:03.848Z); `FINAL_GATE.json` 1789306444034354508 (13:34:04.034Z). No
  file under `cases/` has an mtime after `INFERENCE_TERMINAL.json`; all 44
  inference dispatch files precede it and only the scorer pair follows it.
  The 22 slots ran strictly serially (each `STARTED.json` after the previous
  slot's `COMPLETED.json`, frozen order, control then challenger).

## Check 4 - resource receipts against the frozen LIMITS

All 22 slots table (frozen case order, control C then challenger H). `wall` =
leaf receipt `wall_seconds`; `pwall` = per-case parent wall; `cpu` =
worker-reported `worker_cpu_seconds` from the log's `INFERENCE_TERMINAL` line
(returned workers) or the LAST SAMPLED process CPU (killed workers, marked
`~`, not a final value); `hwm` = worker-reported `worker_peak_rss_bytes`
(returned) or last sampled high-water RSS (`~`); `tree` = supervisor
`sampled_tree_peak_rss_bytes`; `smp` = sampler records in OUTPUT.txt; `log` =
OUTPUT.txt bytes (= forwarded = stored = accepted = attempted); `RESULT` =
retained RESULT.json bytes and hash prefix. Every receipt has `status`
TERMINAL, `budget_stop` null, `source_binding_valid` true,
`derived_result_retention_complete` true, `private_cache_cleanup_complete`
true, no cleanup/identity/control errors, `requires_process_attention` false.

| Case | Family | Arm | exit | wall s | pwall s | cpu s | hwm B | tree B | smp | log B | RESULT |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CG08506AE97D | 10839000 | C | -24 | 1861.374 | 1861.422 | ~1860.095 | ~1803534336 | 1801486336 | 62 | 97594 | none |
| CG08506AE97D | 10839000 | H | -24 | 1861.049 | 1861.096 | ~1860.276 | ~2049196032 | 2045247488 | 62 | 97851 | none |
| CG191DEC7F81 | 49001000 | C | -24 | 1861.280 | 1861.313 | ~1860.294 | ~1409204224 | 1409187840 | 62 | 97598 | none |
| CG191DEC7F81 | 49001000 | H | -24 | 1861.307 | 1861.340 | ~1860.324 | ~1607319552 | 1602486272 | 62 | 97379 | none |
| CG1E0BBA2933 | 38172000 | C | -24 | 1860.992 | 1861.031 | ~1860.369 | ~1631256576 | 1631256576 | 62 | 97634 | none |
| CG1E0BBA2933 | 38172000 | H | -24 | 1861.631 | 1861.670 | ~1860.320 | ~1824358400 | 1819459584 | 62 | 97506 | none |
| CG4BDD1371B6 | 37361005 | C | 0 | 1626.818 | 1631.566 | 1625.392 | 1469743104 | 1467596800 | 54 | 86753 | 60624780 `583eb18fd87a` |
| CG4BDD1371B6 | 37361005 | H | 0 | 702.230 | 706.937 | 701.538 | 1618444288 | 1614659584 | 23 | 37666 | 61733466 `32dd73a4565f` |
| CG4FAF273093 | 49024000 | C | -24 | 1861.798 | 1861.804 | ~1859.736 | ~513671168 | 513671168 | 62 | 97317 | none |
| CG4FAF273093 | 49024000 | H | -24 | 1861.791 | 1861.796 | ~1859.718 | ~541360128 | 541360128 | 62 | 97522 | none |
| CG64C412CC73 | 40500000 | C | -24 | 1862.024 | 1862.047 | ~1859.714 | ~1064861696 | 1059733504 | 62 | 97606 | none |
| CG64C412CC73 | 40500000 | H | -24 | 1862.027 | 1862.051 | ~1859.818 | ~1233797120 | 1229864960 | 62 | 97302 | none |
| CG65CD575B1C | 38172000 | C | 0 | 30.827 | 35.506 | 30.687 | 1592573952 | 1592573952 | 0 | 1645 | 57300936 `c192260e3fe6` |
| CG65CD575B1C | 38172000 | H | 0 | 47.060 | 51.687 | 46.913 | 1911160832 | 1911160832 | 1 | 3109 | 58946054 `a8c0c9e226d8` |
| CG84C8FA12D2 | 10839000 | C | 0 | 42.823 | 49.005 | 42.684 | 1771044864 | 1771044864 | 1 | 3232 | 74637411 `8b026a808693` |
| CG84C8FA12D2 | 10839000 | H | 0 | 63.654 | 69.618 | 63.487 | 1944829952 | 1944829952 | 2 | 4482 | 75811102 `12e8eca3b912` |
| CGE4B0549E41 | 10839000 | C | 0 | 12.502 | 12.551 | 12.417 | 1697153024 | 1697153024 | 0 | 1636 | 85765 `2aa1d4633952` |
| CGE4B0549E41 | 10839000 | H | 0 | 12.750 | 12.794 | 12.662 | 1754529792 | 1754529792 | 0 | 1639 | 90951 `0c0d0dd8789f` |
| CGEBA9A869AC | 38172000 | C | -24 | 1861.891 | 1861.920 | ~1859.680 | ~1496825856 | 1496088576 | 62 | 97454 | none |
| CGEBA9A869AC | 38172000 | H | -24 | 1861.901 | 1861.931 | ~1859.701 | ~1642217472 | 1637154816 | 62 | 97702 | none |
| CGF41F49066A | 38172000 | C | -24 | 1861.911 | 1861.946 | ~1859.634 | ~1650147328 | 1647509504 | 62 | 97553 | none |
| CGF41F49066A | 38172000 | H | -24 | 1861.970 | 1862.006 | ~1859.885 | ~1775009792 | 1770160128 | 62 | 97096 | none |

Sums: leaf walls 28601.609 s (of which the 14 CPU-terminated slots 26062.946
s and the 8 returned slots 2538.664 s); parent walls 28633.036 s; known worker
CPU for the 8 returned workers 2535.780 s (no total including the killed
workers is formed). Largest returned CPU 1625.392 s (CG4BDD C), largest
returned worker high-water RSS 1944829952 B (CG84C H), largest sampled tree
2045247488 B (CG08506 H): all below 1860 s / 8 GiB; all leaf walls below 3540
s (largest 1862.027 s); all logs below 1 MiB (largest 97851 B); largest RESULT
75811102 B below 128 MiB; COMPLETED 199674 B and FINAL_GATE 1057 B below 256
KiB; SCORED 106125 B and SCORER_RESULT 29758 B below 32 MiB. Scorer receipt:
exit 0, budget_stop null, wall 0.22751912521198392 s, sampled tree peak
88866816 B (below 180 s / 2 GiB), log 115 B bound, cache
`/private/tmp/gold11-v9-scorer-404xooo4` (inode 35852641) removed. Parent:
lifetime high-water RSS 1262747648 B, wall 28675.583 s and FINAL_GATE wall
28675.770 s (below 104400 s; 28675.8 + 2400 s reserve = 31075.8 s < 104400
s). Sampler records total 949 (302 in V8). Every OUTPUT.txt line is JSON;
each returned worker has exactly one `PUBLIC_WORKFLOW_CALL_BEGIN` line, one
stopped sampler receipt (`samples_attempted` = `samples_written`,
`failure_type` null, `sample_limit_reached` false) and one
`INFERENCE_TERMINAL` line whose `result_sha256` equals the receipt, the
retention artifact, the actual file hash, `INFERENCE_TERMINAL.result_bindings`
and the projection's `result_binding.supplied_sha256`. Every termination was
handled as a retained negative: 14 supervisor refusals with `budget_stop`
null (the OS CPU limit, not a parent budget, ended them) and no per-slot wall
timeout occurred. Sampled values are not continuous guarantees.

Per-slot RESULT.json checks (8 files): `cad_sha256` and
`scan_topology.source_sha256` equal the frozen case declaration; `schema`
`CAD_GUIDED_MESH_OFFLINE_V13` (transport) with `implementation`
`CAD_GUIDED_MESH_OFFLINE_V16_CONTEXT_V4_COMPLETION_V1_KERNEL_V2`,
`initialization_implementation`
`CAD_GUIDED_PLANE_INITIALIZATION_V15_CONTEXT_V4_COMPLETION_V1`,
`registration_implementation`
`SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V4`,
`kernel_implementation` `CAD_GUIDED_MEASUREMENT_V2_LAYER_SELECTION_AND_DIAGNOSTICS`,
`completion_implementation`
`TWO_OF_THREE_TRIPLET_THIRD_FACE_COMPLETION_ALONG_PASSING_PAIR_AXIS_V1`,
`two_plane_context_configuration_sha256`
`12900ddac835ccd99208019b77d4bcce3443ae3c5c9dd0ea82aba89ee5f5db66` in all 8;
policy snapshots carry the frozen deltas (0.03 / 0.04 mm, 30 deg gap, layer
policy, completion policy, gate 1.0 mm, ties 256, 512M coordinates, cap 8);
derived `measurement_context_cap.value` = min(8, 4096 // features,
64000000 // (features x raw rows)) = 6 (CG4BDD, 17 features, 581263 rows), 6
(CG65C, 13 / 715498), 7 (CG84C, 10 / 863115), 7 (CGE4B, 10 / 837407). All 8
have `status` UNAVAILABLE, `predictions` [], `observed_bend_count` null,
`conformance` INDETERMINATE, `native_features.status` PARTIAL with no
reasons.

### CPU-terminated slots: what exists and what is unknown

Fourteen slots: CG08506AE97D, CG191DEC7F81, CG1E0BBA2933, CG4FAF273093,
CG64C412CC73, CGEBA9A869AC, CGF41F49066A (both arms each). Records present:
`STARTED.json` (limits and arm request bound); `COMPLETED.json` receipt with
`exit_code` -24, `budget_stop` null, `failure_type` ValueError, refusal
`INFERENCE_WORKER_FAILED`, `inference_started` `UNKNOWN_UNTIL_RETURNED_RESULT`,
`retained_result_artifacts` `NO_RETURNED_RESULT_INTERNAL_STAGES_UNKNOWN`,
`workflow_stage_diagnostics` `NO_RETURNED_WORKFLOW_INTERNAL_STAGES_UNKNOWN`,
`projection_kind` `SUPERVISOR_REFUSAL_NO_USABLE_WORKFLOW`, private cache
removed, walls 1860.992-1862.027 s (below the 3540 s wall cap);
`SCORING_INPUT.json` = the six-key supervisor refusal (159 B, hash
`3a4dd85d...deea1`, the same bytes as V8's refusals); `OUTPUT.txt`
(97096-97851 B) with one `PUBLIC_WORKFLOW_CALL_BEGIN` line and 62 sampler
records (indices 0-61, 30 s cadence) and NO sampler receipt and NO
`INFERENCE_TERMINAL` line; no `RESULT.json`, no `RESULT_PARTIAL.bin`. Exit
-24 is the Darwin signal-style return for SIGXCPU; the frozen worker sets
`RLIMIT_CPU` to `LIMITS["case_cpu_seconds"]` (runner line 1579), and the last
samples read CPU 1859.634-1860.369 s at elapsed 1860.51-1861.17 s, which
supports CPU-limit attribution under the frozen (not relaxed) limit. In all
14 the last sampled stack sits inside `register_whole_part_planes_v15` at
lines 1460 (coarse `_v3._refine`), 1462 (fine `_v3._refine`) or 1465
(`_v3._diagnose`), below registration V4 `_Queries.__call__` 245 and triangle
surface V4 `query_closest_surface` / `walk` / `leaf` / `_paired_candidates`,
called from `register_measurement_contexts_v15` 1535 <-
`inspect_cad_guided_mesh_v16` 559. Over the 62 samples the V15 frame sits at
1460 / 1462 / 1465 / 1467 (coarse refine / fine refine / diagnose / context
support) in 19/36/4/2 (CG08506 C), 19/35/4/2 (CG08506 H), 23/34/2/2 (CG191
C), 22/32/4/2 (CG191 H), 25/30/4/2 (CG1E0 C), 25/30/3/2 (CG1E0 H), 18/34/6/4
(CG4FAF C), 17/32/7/6 (CG4FAF H), 41/21/0/0 (CG64C C), 41/20/0/0 (CG64C H),
25/29/4/3 (CGEBA C), 25/29/4/3 (CGEBA H), 20/36/3/2 (CGF41 C), 20/35/3/2
(CGF41 H) samples, with one sample at line 1414 (`_initialize`) in seven
slots (CG08506 C/H, CG191 C, CG1E0 C/H, CG64C H, CGEBA H) and one sample at
1430 (surface-role preparation) in three (CG08506 H, CG191 H, CGF41 H); the
remaining samples had the V15 frame outside the 12 named frames. Because line
1460 follows the `NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS` requirement at
line 1425, at least one passing start group existed in each killed run, and
in the twelve slots with samples at line 1467 at least one candidate reached
context-support evaluation; CG64C (both arms) never sampled beyond
refinement. Nothing else is known: final CPU, candidate counts and origins
(main lane or completion lane), whether any context was supported,
native-feature status and every work counter are unknown for these 14 slots
(V8 had `INVALID_NATIVE_SOLID` for CG64C both arms; V9 cannot confirm or deny
it).

## Check 5 - scoring ledger

`SCORED.json` schema `GOLD11_CONTACT_GEOMETRIC_DEVELOPMENT_COMPARISON_V3`,
status `COMPUTED_DEVELOPMENT_DIAGNOSTIC`, experiment_schema
`GOLD11_MESH_DEVELOPMENT_V9`, control identity
`FRESH_V16_COMPLETION_V1_KERNEL_V2_V7_SCAN_PLANES_SCHEDULE_V1_NATIVE_V1`,
challenger identity
`FRESH_V16_COMPLETION_V1_KERNEL_V2_COMPOSITE_SCAN_PLANES_SCHEDULE_V2_NATIVE_V1`,
comparator
`FRESH_V16_COMPLETION_V1_KERNEL_V2_V7_SCHEDULE_V1_VS_COMPOSITE_SCHEDULE_V2_NATIVE_V1`,
wrapper schema `GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V4`, request
schema `GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_REQUEST_V6`, `result_interpretation`
`COMBINED_CONSUMED_DEVELOPMENT_CHANGES_NOT_MECHANISM_ONLY_SPEEDUP_OR_CALIBRATED_ACCURACY`,
`arm_comparison_scope`
`SCAN_PLANE_LANE_AND_SCHEDULE_ONLY_SAME_V16_ENGINE_NOT_OLD_VS_NEW_ENGINE`,
`combined_change_attribution` true, `historical_score_beaten_claim_allowed`
false, `historical_score_comparability` `UNESTABLISHED`,
`accepted_registration_claim_allowed` false, `production_use_allowed` false,
`floor_release_allowed` false, `strict_physical_engine_targets_met` false,
`observed_bend_count` null, `conformance` INDETERMINATE, `uncertainty_status`
UNCALIBRATED, `reference_line_count` 71, `family_count` 6, `case_count` 11,
`reference_snapshot_sha256` `d16390b2...b45dc`, `all_declared_input_bytes_verified`
true, `source_before_after_consistent` true. Matcher unchanged
(`UNCHANGED_LEGACY_ONE_TO_ONE_MINIMUM_COST`, axis 5.0 deg, transverse 4.0 mm,
Gold-span overlap 0.6, prediction-span overlap 0.6). Scoring scope
`GENERIC_CONDITIONAL_CONTACT_GENERATOR_LOCALIZATION_NOT_INNER_ARC_OR_MANUFACTURING_EVENT`,
`inner_face_specific_claims`
`NONE_UNRESOLVED_CARRIERS_SCORED_AS_GENERIC_CONTACT_LINES_ONLY`,
`carrier_identity_policy`
`ELIGIBILITY_IS_A_DECLARED_FUNCTION_OF_MATERIAL_SIDE_STATUS_NEVER_A_DEFAULT`,
`material_side_status` `PER_CARRIER_DECLARED_UNVERIFIED_SEE_CASE_DISPOSITIONS`.
All 37 original `SCORER_RESULT.json` fields are canonically identical in
`SCORED.json` (0 differing fields; 31 runner-added fields only);
`SCORER_RESULT.json` is in canonical byte form.

| Metric | Control | Challenger | Paired delta |
| --- | ---: | ---: | ---: |
| True positives | 0 | 0 | 0 |
| False positives | 0 | 0 | 0 |
| False negatives | 71 | 71 | 0 |
| Precision | null | null | null |
| Recall | 0.0 | 0.0 | 0.0 |
| F1 | null | null | null |
| Minimum-family recall | 0.0 | 0.0 | 0.0 |
| Emitted support line count | 0 | 0 | - |
| Exact emitted-support-count accuracy | 0.0 | 0.0 | 0.0 |
| Exact localized target-set accuracy | 0.0 | 0.0 | 0.0 |
| Refused / malformed / complete-empty cases | 11 / 0 / 0 | 11 / 0 / 0 | - |
| Physical observed counts unknown | 11 of 11 | 11 of 11 | - |

Per-family false negatives (both arms): 10839000 18 (3 cases), 37361005 4,
38172000 34 (4 cases), 40500000 7, 49001000 5, 49024000 3 (sum 71); family
paired deltas all 0 / null. Per-case dispositions (both arms): all 11
`REFUSED` with `prediction_count` 0, `false_negative` = `gold_count`
(6,5,10,4,3,7,6,8,4,8,10), every Gold id listed under `unmatched_gold_ids`,
precision and F1 null, recall 0.0, `observed_bend_count` null; reason
`EXPLICIT_SUPERVISOR_UNAVAILABLE` (phase SUPERVISOR, `result_binding` null,
diagnostics null) for CG08506, CG191, CG1E0, CG4FAF, CG64C, CGEBA and CGF41;
`EXPLICIT_WORKFLOW_UNAVAILABLE` with `result_binding.supplied_sha256` equal to
the RESULT hash for CG4BDD (reported phase `MEASUREMENT_REFUSAL` control,
`WHOLE_PART_REGISTRATION` challenger), CG65C, CG84C and CGE4B
(`WHOLE_PART_REGISTRATION`). In those four cases the projection diagnostics
(`registration_context_count`, `source_context_window_count`,
`source_support_record_count`, `support_group_count`, `measured_group_count`,
`covered_group_count`, `emitted_carrier_count`,
`identity_unresolved_carrier_count`, `inner_face_eligible_carrier_count`,
`finite_supported/refused_group_count`) are all null: no window, support
record or carrier exists anywhere in the campaign, so the material-side
identity counts are null (absent), not zero. Null precision and F1 are
preserved as null (undefined), never coerced to zero, in SCORER_RESULT,
SCORED, every per-case, per-family and paired field; no numeric F1 exists for
this run. All 22 slots are in the denominator and the 71 references are
retained as false negatives in both arms.

## Check 6 - per-slot outcome taxonomy (from the attempt's own records)

Mutually exclusive over all 22 slots, from the retained RESULT ledgers
(workflow -> registration -> initialization -> completion lane chain) and the
supervisor receipts. Codes: CPU = CPU-terminated, no returned result; PF =
completed finite prefilter (main lane plus completion lane) with no passing
start; MW = scan-plane proposal triangle-membership refusal (completion lane
not attempted); RG = every candidate (main and completion) rejected by the
1.0 mm gate / proximity policy, no supported context; MG = supported context
scheduled, measurement consumer refused by the support cover V3 feature guard
before any window.

| Cause | Slots | Cases |
| --- | ---: | --- |
| CPU | 14 | CG08506 C/H, CG191 C/H, CG1E0 C/H, CG4FAF C/H, CG64C C/H, CGEBA C/H, CGF41 C/H |
| PF | 4 | CG65C C/H, CG84C C/H |
| MW | 2 | CGE4B C/H |
| RG | 1 | CG4BDD H |
| MG | 1 | CG4BDD C |

Chain details (both arms identical unless stated):

- PF (4): workflow phase `WHOLE_PART_REGISTRATION`, reasons
  [`PROVISIONAL_REGISTRATION_UNAVAILABLE`, `NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS`];
  registration UNAVAILABLE with 0 candidates; initialization COMPLETE
  (search_complete, numerical_consumer_complete, box_screening_complete all
  true) with `exact_passing_start_group_count` 0; two-plane context V4
  COMPLETE. Main-lane prefilter counters (hypotheses evaluated / assigned
  supports / bound-excluded supports (rows) / completed finite calls (rows) /
  passing starts): CG65C 2045 / 5935 / 5930 (645303) / 5 (533) / 0; CG84C
  4522 / 12342 / 12127 (1484752) / 215 (23418) / 0 - both equal to V8's
  released values to the unit. Context V4 `extrema_coordinate_visits`
  25099205 (CG65C) and 79883302 (CG84C), equal to V8. Native `PARTIAL` with
  no reasons (`supported_native_candidate_count` 7 and
  `unresolved_candidate_count` 11 for CG65C; 6 and 14 for CG84C).
- MW (2): CGE4B C reasons [`PROVISIONAL_REGISTRATION_UNAVAILABLE`,
  `NOISY_PLANE_INITIALIZATION_UNAVAILABLE`], initialization UNAVAILABLE
  `NOISY_SCAN_PLANE_PROPOSALS_UNAVAILABLE` (`evidence_scope`
  `ATTEMPTED_WORK_ONLY_ALL_STARTS_INVALIDATED`, all counters zero); CGE4B H
  `COMPOSITE_SCAN_PLANE_PROPOSALS_UNAVAILABLE`. The string
  `MAX_TRIANGLE_MEMBERSHIPS_EXCEEDED` occurs once in each RESULT and
  `COMPOSITE_V7_LANE_UNAVAILABLE` once in the challenger RESULT (string
  presence in the encoded `patch_evidence.scan` wire; not decoded here).
  Completion lane `NOT_ATTEMPTED` (`attempted_child_count` 0,
  `passing_children_invalidated` true, `usable` false). Unchanged from V8 (the
  20M membership cap was not part of the V9 delta).
- RG (1): CG4BDD H (main-lane hypothesis 4533 plus completion children 7 and
  14) - three candidates attempted, each coarse 30/30 and refine 30/30
  `ITERATION_LIMIT`, surface p95 upper 1.1958633372293805 / 1.3183404592163683
  / 1.3375951378099122 mm (CAD coverage 0.983184814453125 / 0.97991943359375
  / 0.97894287109375; scan coverage lower 0.9997093586514241 /
  0.9995155977523735 / 0.9995155977523735) all above the 1.0 mm gate:
  `SURFACE_RESIDUAL_OUTSIDE_DECLARED_ALGORITHM_GATE`, context
  `REJECTED_CONTEXT` (`CONTEXT_PROXIMITY_RESIDUAL_OUTSIDE_GATE`), registration
  UNAVAILABLE `NO_GEOMETRICALLY_SUPPORTED_MEASUREMENT_CONTEXTS`;
  query_resources 189 calls / 609993 rows, 53285260 certificate pairs,
  13321315 triangle pairs, 353072084 work units. The main candidate's value
  equals V8's CG4BDD H value (1.1958633372293805 mm) to every printed digit.
  This is a completed residual-policy rejection, not proof of physical
  nonconformance and not a capacity refusal.
- MG (1): CG4BDD C - four candidates attempted (main-lane hypothesis 4533 plus
  completion children 25, 26, 59), all 30/30 + 30/30 `ITERATION_LIMIT`,
  `numerical_consumer_status` `COMPLETE_V3_INTERVAL_SUPPORT_POLICY`. Candidate
  0 (main): surface p95 upper 0.6658972409525943 mm, CAD coverage
  0.990753173828125, scan coverage lower 0.9999031195504746;
  `conditional_context_support` `SUPPORTED_CONTEXT` with context p95 upper
  0.6956471254551617 mm (rms upper 0.34599589418224896), coverage lower
  0.9999031195504746, reverse coverage 0.990753173828125, 20642 of 20644 rows,
  `possible_eligible` true - every value equal to V8's CG4BDD C admission to
  the printed digit. Candidates 1 and 2 (completion 25, 26): p95 upper
  0.7380226977855356 / 0.7204521183958937 mm but CAD coverage
  0.566558837890625 / 0.561431884765625 and scan coverage lower
  0.5837047083898469 / 0.5767777562487889 (12050 / 11907 of 20644 rows):
  context `REJECTED_CONTEXT` `INSUFFICIENT_CONDITIONAL_CONTEXT_PROXIMITY`,
  candidate reason `INSUFFICIENT_FINITE_SURFACE_SUPPORT`. Candidate 3
  (completion 59): p95 upper 1.3062819637418486 mm, coverage 0.9805908203125 /
  0.9999031195504746: `SURFACE_RESIDUAL_OUTSIDE_DECLARED_ALGORITHM_GATE`,
  `CONTEXT_PROXIMITY_RESIDUAL_OUTSIDE_GATE`. query_resources 252 calls /
  813324 rows, 373995224 certificate pairs, 93498806 triangle pairs,
  1502006322 work units. `measurement_context_schedule`: supported 1,
  scheduled 1 (index 0, `SCHEDULED`), `max_measurement_contexts` 6,
  pose_class_rows 4183 (V8: 4183), no duplicate or omitted disposition;
  registration `CONDITIONAL_MEASUREMENT_CONTEXTS`, `measurement_context_indices`
  [0]. The workflow then entered `FULL_RAW_LOCAL_MEASUREMENT` and returned
  status UNAVAILABLE, phase `MEASUREMENT_REFUSAL`, reason
  `SUPPORT_COVER_MEASUREMENT_FAILURE:ValueError:INVALID_SUPPORT_COVER_FEATURES`;
  `measurements` (schema `CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V3`): status
  UNAVAILABLE, `computation_status` absent, `source_context_result` null,
  `source_context_result_sha256` null, `support_cover_plan` null,
  `support_groups` [], `predictions` [], `attempted_work` {attempted_groups 0,
  cover_plan_work null, cover_refit_row_visits 0},
  `discarded_measurement_prefix` true with null discarded digest and 0
  discarded records; `support_cover_publication_validation`
  {cover_plan_revalidated false, fatal_refusal true}. No window, layer,
  spectrum or finite-support record exists in this RESULT (no key containing
  "layer" under `measurements`; the strings `NOT_PLANAR`,
  `FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY`,
  `FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL`, `ARC_COMPATIBLE_WITH_FLAT`,
  `INSUFFICIENT_DISTINCT_ANGULAR_SUPPORT`, `DISCONNECTED_LOCAL_ANGULAR_SUPPORT`
  and `TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED` occur in none of
  the 8 RESULT files). V9's `native_features.features` has 17 entries.
- CPU (14): as described in check 4.

Absent everywhere in the 8 RESULT files: `INVALID_NATIVE_SOLID`,
`BUDGET_EXTREMA_COORDINATE_VISITS`, `NEAR_TIE_BUDGET_EXCEEDED`,
`CONTEXT_WINDOW_PREFLIGHT_BUDGET`, `FULL_RAW_CROP_WORK_PREFLIGHT_BUDGET`,
`SUPPORTED_CONTEXT_DUPLICATE_POSE_CLASS`,
`SUPPORTED_CONTEXT_OMITTED_BY_MEASUREMENT_BUDGET`, `MEASUREMENT_CONTEXT_CAP`,
`NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS`,
`NO_NONCONFLICTING_CANONICAL_SCAN_MEASUREMENTS`.

### What the frozen source says about the MG refusal (read-only)

`source/backend/cad_guided_support_cover_measurement_v3.py` lines 143-145
raise `INVALID_SUPPORT_COVER_FEATURES` unless `type(features) in (list,
tuple) and 1 <= len(features) <= 64 and all(type(feature) is
_v1.CADGuidedFeature for feature in features)`, where `_v1` is
`cad_guided_context_measurement_v2` (line 20), whose `CADGuidedFeature` is
imported from `cad_guided_measurement_v2` (its line 21; class defined at
`cad_guided_measurement_v2.py` line 99). `cad_guided_mesh_offline_v16.py`
passes `features["features"]` (line 621) produced by
`cad_guided_native_features_v1.extract_ocp_guided_features_v1` (lines 525-529
for `native_adapter_version` `v1`, the frozen value), which constructs
`cad_guided_measurement.CADGuidedFeature` (its line 33 import; class defined
at `cad_guided_measurement.py` line 31) - a different class object. The first
two conjuncts hold on this input (the list has 17 entries; V16 line 550 uses
the same list for `feature_count` 17, which the RESULT's context cap
records), so the identity conjunct failed; the raise is caught by cover V3's
outer `except Exception` (lines 213-222), which returns the refusal seen in
the RESULT. In V8 the same features passed the corresponding guard in cover
V2 (line 139) because cover V2's `_v1` is `cad_guided_context_measurement_v1`
(line 17), whose `CADGuidedFeature` is the `cad_guided_measurement` class
(its line 17). This is a deterministic frozen-source outcome for any V9 slot
that reaches measurement with native-adapter features; it is stated here from
the records and the frozen bytes, and it is not a runtime observation of the
exception. Whether the pre-freeze fixture tests constructed the V2 class
directly is outside this audit's scope.

### Completion-lane ledgers (`initialization.completion_lane`, the 8 returned slots)

| Slot | Lane status | Parents (hyp. idx) | Children scheduled | Evaluated / passing / rejected | Normal-incompatible / shift-limit / omitted / duplicate | New start groups | Refinement schedule (main + completion of 48) | Lane bytes | Lane work units |
| --- | --- | --- | ---: | --- | --- | ---: | --- | ---: | ---: |
| CG4BDD C | COMPLETE, usable | 3 (2355, 2545, 2803) | 64 (18 + 28 + 18) | 64 / 3 / 61 | 140 / 0 / 0 / 0 | 3 | 1 + 3 | 340974 | 2287764 |
| CG4BDD H | COMPLETE, usable | 1 (2638) | 18 | 18 / 2 / 16 | 50 / 0 / 0 / 0 | 2 | 1 + 2 | 103145 | 192044 |
| CG65C C | COMPLETE, usable | 0 | 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 | 0 + 0 | 2951 | 0 |
| CG65C H | COMPLETE, usable | 0 | 0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 | 0 + 0 | 2951 | 0 |
| CG84C C | COMPLETE, usable | 1 | 44 | 44 / 0 / 44 | 68 / 0 / 0 / 0 | 0 | 0 + 0 | 245510 | 570548 |
| CG84C H | COMPLETE, usable | 1 | 44 | 44 / 0 / 44 | 68 / 0 / 0 / 0 | 0 | 0 + 0 | 245774 | 570548 |
| CGE4B C | NOT_ATTEMPTED | - | 0 | - | - | 0 | - | - | - |
| CGE4B H | NOT_ATTEMPTED | - | 0 | - | - | 0 | - | - | - |

Every lane carries `ranking`
`NONEXHAUSTIVE_LOCAL_SCHEDULE_ABS_SHIFT_THEN_FACE_ID_NO_UNIQUENESS_OR_COVERAGE_CLAIM`,
`scope` `ADDITIVE_PROPOSALS_RESCREENED_AND_REQUERIED_ON_ALL_THREE_SUPPORTS_NOT_ACCEPTED_REGISTRATION`,
the frozen policy and budgets, `axial_unconstrained_count`,
`axis_parallel_unresolved_count` and `parallel_passing_pair_count` 0, and
`ledger_bytes` below the 100663296 B cap. Main-lane statistics record the
split: CG4BDD C `main_passing_start_group_count` 1 +
`completion_passing_start_group_count` 3 = `exact_passing_start_group_count`
4 (hypotheses evaluated 4938, assigned 12532, bound-excluded 12264 (1585607
rows), completed calls 268 (28879 rows)); CG4BDD H 1 + 2 = 3 (4928 / 12576 /
12366 (1680860) / 210 (25164)). The completion start groups are tagged `lane`
`TRIPLET_COMPLETION` with `completion_indices` and empty
`hypothesis_indices`; `refinement_schedule.rule`
`MAIN_LANE_STRATA_FIRST_THEN_COMPLETION_STRATA_IN_REMAINING_SLOTS`. Layer
selection identities and finite-support refusals: none exist in any record
(no window was measured; see MG).

## Check 7 - standing constraints

- No Gold in inference: the frozen runner gives workers only a dispatch id
  and token; the issued leaf record binds arm plus scan/CAD hashes; the only
  catalog reader `selected_references()` is called by `score_after_terminal`
  after `INFERENCE_TERMINAL.json`; `SELECTED_REFERENCES.json` was written
  20.597 s after `INFERENCE_TERMINAL.json` and after all 22 terminals, and no
  case file changed afterwards. This rests on frozen-source reading and
  ordering evidence, not on a runtime observation of file accesses.
- All-22 denominator: 22 case directories (11 x 2), 22 durable terminal
  records, 22 projections, 22 dispositions in both arms of SCORED; refused
  cases keep every reference as a false negative.
- Unchanged 71 references: snapshot digest `d16390b2...b45dc` equal to V8's
  and to the frozen declaration; 71 lines, same per-case counts as V8.
- Both fresh arms: 22 fresh inference workers (23 distinct pids and caches
  with the scorer), no cached control loaded (`retained_control_policy`
  `NOT_ASSUMED_OR_LOADED_BY_THIS_RUNNER`).
- No gate change after launch, no resume/retry: single `STARTED.json` per
  slot, 22 one-use dispatch ids each claimed once, every `STARTED.limits`
  equal to the frozen limits, CPU kills at the frozen 1860 s, `no_resume`
  true everywhere, `FROZEN.json` and `PRE_RUN_AUDIT.json` unchanged (check 2),
  final source audit unchanged.
- No reopening of V8 or earlier attempts by this audit (V8 figures from the
  released V8 POST and addendum only); no raw, Gold or cache access by this
  audit.

## Paired comparison with V8

Same eleven inputs, same 71-line reference snapshot (`d16390b2...b45dc` in
both), same matcher; both campaigns TP 0 / FP 0 / FN 71 in both arms,
precision and F1 null, recall 0. The paired score difference V9 - V8 is zero
on every defined metric and null on precision/F1. The immediate causes moved
without any slot reaching an emitted support:

| V8 cause (released POST) | V8 slots | V9 cause | V9 slots |
| --- | ---: | --- | ---: |
| No passing start (PF) | 12 | PF | 4 |
| Membership cap (MW) | 2 | MW | 2 |
| Measurement-consumer window refusal (MC) | 3 | - | 0 |
| Residual rejection (RG) | 1 | RG | 1 |
| CPU kill | 4 | CPU | 14 |
| - | 0 | Cover feature-guard refusal (MG) | 1 |

Per slot (V8 cause and leaf wall from the released V8 POST -> V9 cause and
leaf wall): CG08506 C PF 43.818 -> CPU 1861.374, H PF 63.043 -> CPU 1861.049;
CG191 C CPU 1861.348 -> CPU 1861.280, H CPU 1861.316 -> CPU 1861.307; CG1E0 C
MC 356.501 -> CPU 1860.992, H MC 392.212 -> CPU 1861.631; CG4BDD C MC 253.856
-> MG 1626.818, H RG 288.536 -> RG 702.230; CG4FAF C CPU 1860.930 -> CPU
1861.798, H CPU 1860.923 -> CPU 1861.791; CG64C C PF 26.350 -> CPU 1862.024,
H PF 35.097 -> CPU 1862.027; CG65C C PF 32.758 -> PF 30.827, H PF 50.446 ->
PF 47.060; CG84C C PF 45.253 -> PF 42.823, H PF 67.105 -> PF 63.654; CGE4B C
MW 13.526 -> MW 12.502, H MW 13.581 -> MW 12.750; CGEBA C PF 32.564 -> CPU
1861.891, H PF 47.677 -> CPU 1861.901; CGF41 C PF 36.951 -> CPU 1861.911, H
PF 54.737 -> CPU 1861.970.

Observations (combined-change attribution only, from the records):

- Ten slots that completed quickly in V8 with no passing start (CG08506,
  CG64C, CGEBA, CGF41; both arms) or with a scheduled context (CG1E0; both
  arms) ran to the frozen 1860 s CPU limit inside the V15 registration
  candidate loop in V9, i.e. each now had at least one passing start group
  and no returned ledger. The records do not say where those start groups
  came from; the V8 ledgers for the same inputs had zero main-lane passing
  starts, and on the two returned PF cases the V9 main-lane prefilter
  reproduced V8's counters exactly, which is consistent with, but not proof
  of, completion-lane origin. The CPU-kill count rose from 4 to 14 and the
  campaign inference wall from 9298.529 s (V8 leaf sum) to 28601.609 s, of
  which the 14 kills are 26062.946 s. For the 8 slots returned in both
  campaigns the worker CPU rose from 1850.329 s (V8, 18 returned) to
  2535.780 s (V9, 8 returned); this is not a speed-up.
- CG4BDD C: the main-lane candidate reproduces the V8 admission to every
  printed digit (surface 0.6658972409525943 mm, context 0.6956471254551617
  mm, 20642/20644 rows, pose_class_rows 4183); V8 then refused 17 native
  windows in the measurement consumer, V9 refused the whole measurement by
  the cover V3 feature guard before any window. CG4BDD H reproduces the V8
  rejection (1.1958633372293805 mm) and adds two completion candidates
  rejected at 1.318 and 1.338 mm. The three completion candidates that
  passed the V9 prefilter on CG4BDD C were rejected for insufficient finite
  surface support (coverage ~0.57-0.58, two) or residual 1.306 mm (one).
- CG65C and CG84C reproduce the V7/V8 prefilter counters exactly; the
  completion lane found no two-of-three parent on CG65C (as the protocol
  anticipated) and one parent with 44 rescreened children, none passing, on
  CG84C (both arms).
- The measurement-side deltas (kernel V2, context V2, cover V3, layer
  selection, 0.03/0.04 mm policy, 30 deg gap) produced no window evidence at
  all: the expected retained-negative classes of the protocol (`NOT_PLANAR`,
  arc-support refusals, single-layer / ambiguous partitions) neither occurred
  nor could occur, because no window was reached.
- Pose-class dedup and the context cap were again not exercised (supported =
  scheduled = 1 in the one scheduled slot; cap 6).
- Arm comparison: 21 of 22 slots have identical immediate causes in both
  arms; the only arm difference is CG4BDD (C supported at 0.696 mm then MG; H
  rejected at 1.196 / 1.318 / 1.338 mm). Challenger walls exceed control
  walls in three of the four returned pairs (CG65C, CG84C, CGE4B; composite
  scan-plane lane); CG4BDD H is shorter than C (3 versus 4 candidates, no
  measurement attempt). Nothing here separates the declared V9 changes from
  one another.

## Findings

1. Exit status observed (new). The waiting parent 52251 recorded returncode
   0 for runner 52252 (`OBSERVED_BY_WAITING_PARENT`, nonce
   `25e6c406...7345`, started 05:36:07 UTC, finished 13:34:04 UTC), consistent
   with STARTED/COMPLETED/FINAL_GATE timestamps, the parent walls and the
   runner's `execute()` return path. The V8 evidence limitation is closed;
   the record proves the outer return code only. The pid record retains only
   its final FINISHED state (single-line state file by design).
2. Fourteen CPU-terminated slots versus four in V8, each an explicit
   supervisor refusal with no returned or partial result; `budget_stop` null
   in all 14; internal stages unknown beyond sampled stacks (Python-frame
   samples every 30 s, not continuous profiling). Every kill occurred inside
   the V15 registration refine/diagnose loop at the frozen limit; CG64C never
   sampled beyond refinement.
3. Material non-blocking finding - new retained outcome class MG (1 slot,
   CG4BDD C): a supported context under the 1.0 mm gate reached the frozen
   support cover V3 and was refused by its `type(feature) is
   cad_guided_measurement_v2.CADGuidedFeature` identity guard, because the
   frozen V16 workflow passes `cad_guided_measurement.CADGuidedFeature`
   instances from native adapter V1 (the class V8's cover V2 compared
   against). Zero windows, zero support records, zero predictions; no
   per-window diagnostic, no layer identity, no finite-support refusal exists
   in the campaign; `identity_unresolved_carrier_count` and
   `inner_face_eligible_carrier_count` are null (absent), not the zero the
   protocol expected. The V9 measurement-side changes were therefore not
   tested by V9 on physical data. This is a deterministic frozen-source
   outcome handled as a retained negative by the frozen failure policy; it is
   not a custody or ledger inconsistency, it must not be described as a
   geometric or measurement refusal, and it cannot be repaired on this
   terminal attempt (a successor requires its own review, freeze, PRE and
   run). While this audit was in progress the author's `docs/CURRENT_STATUS.md`
   top entry (13:45 UTC) began to mention "a V9 wiring defect found"; its body
   was not read and this finding derives from the records and frozen bytes.
4. Completion lane: exercised and published on the 6 returned initializations
   (3 + 1 + 0 + 0 + 1 + 1 parents; 64 + 18 + 0 + 0 + 44 + 44 children; 3 + 2
   + 0 + 0 + 0 + 0 passing; no omission, duplicate or shift-limit rejection;
   lane bytes at most 340974 B). The five new completion start groups on
   CG4BDD all completed refinement and were rejected (two for insufficient
   finite surface support, three by the 1.0 mm gate). Not attempted on CGE4B
   (proposals unavailable).
5. Main-lane prefilter and context V4 reproduce V8 exactly on the three
   returned initializations with V8 counterparts (CG65C, CG84C, CG4BDD
   counters and the CG4BDD candidate numerics to every printed digit),
   consistent with unchanged behaviour of those stages on these inputs.
6. The measurement-context schedule (pose-class dedup, cap) was not
   exercised: one supported, one scheduled, cap 6; no duplicate or omitted
   disposition.
7. Native `INVALID_NATIVE_SOLID` is absent from all 8 returned RESULTs; the
   V8 native refusal for CG64C cannot be confirmed because both CG64C workers
   returned nothing.
8. No retry, resumed start, relaxed limit or edited frozen source: each of
   the 22 case directories holds exactly one `STARTED.json`, one receipt, one
   projection, one log and (8 cases) one RESULT, in the frozen order; 22
   unique one-use dispatch ids each claimed once; every `STARTED.limits`
   equals the frozen limits and every arm request digest equals the frozen
   request; the CPU kills occurred at the frozen 1860 s; the working tree,
   `source/` snapshot and archive members are hash-identical for all 85
   sources; the parent's final source audit and both inventory passes report
   no change; `COORDINATOR_OUTPUT.txt` holds exactly 22 case lines.
9. Gold/reference isolation as in check 7: no evidence that Gold or reference
   data reached inference; frozen-source reading and ordering evidence, not a
   runtime observation of file accesses.
10. Coordinator private cache `/private/tmp/gold11-v9-coordinator-gtgs3ilh`
    (device 16777231 / inode 35839343) declared
    `OPAQUE_PRIVATE_CACHE_RETAINED_OUTSIDE_ATTEMPT`, not enumerated by the
    runner or by this audit; the 23 leaf caches are declared removed.
11. 84 of the 85 frozen source files exist in the working tree only as
    untracked files (`git status` `??`; one file is clean-tracked), as do the
    attempt directory, the archive directory, the protocol and the three
    outer records; custody rests on the hash-verified snapshot and archive,
    not on version control. Not a blocker; recorded for the root.
12. `PRE_RUN_AUDIT.json` bytes are not in canonical JSON form (the runner
    binds its raw-byte hash and its canonical digest separately; both
    reconcile). The outer records are mode 0644 while the attempt's controls
    are 0600. Cosmetic.
13. Resource envelopes: parent lifetime high-water RSS 1262747648 B and
    sampled outer tree peak 2400010240 B (below 12 GiB); parent wall
    28675.583 s and FINAL_GATE wall 28675.770 s (below 104400 s and below the
    reserve boundary); all per-slot, log, result, control, scorer and report
    sizes below their frozen caps (check 4). Sampled values are not continuous
    guarantees.
14. Combined-change caveat: V9 differs from V8 in the completion lane, the
    measurement kernel and consumer chain, the measurement policy and the
    support gap at once; no per-change attribution is possible from these
    ledgers, and slower or faster refusals are not geometry improvements. The
    campaign spent 3.08x V8's inference wall to produce the same zero-support
    ledger.
15. Process disclosure of this audit: one transient `tar -tvf` listing was
    written to and deleted from the job tmp directory (outside the
    repository); the archive comparison was redone in memory.

## What was not checked

- The outer exit observation itself: the waiting parent's record was verified
  for internal consistency, not observed by this reviewer; no crash-proof
  guarantee is claimed by the waiter and none is claimed here.
- No runner, projection (`gold11_schedule_projection_v4`), scorer or engine
  code was executed: inner projections were bound by canonical-hash identity
  with the retained `contact_projection` objects, not re-projected from RESULT
  bytes; the matcher was not rerun; the cover V3 exception was inferred from
  the records and the frozen bytes, not reproduced.
- Raw scan/CAD bytes were not read or hashed; the Gold catalog was not
  opened; input binding rests on the frozen declarations,
  `source_binding_valid` receipts and the snapshot digest.
- Private caches under `/private/tmp` were not stat'ed or listed; one `ps`
  query was run (no matching process); no live-writer check beyond that.
- The V8 attempt directory was not opened; V8 figures are from the released
  V8 POST and its addendum only.
- The encoded `patch_evidence.scan` wire, `hypotheses`, `completion_lane.hypotheses`,
  `face_index_evidence` and `unsigned_assignments` arrays were not decoded or
  reviewed beyond counts and string presence; the supervision, artifact,
  sampler, kernel V2, context V2 and cover V3 helpers were not reviewed line
  by line (their frozen hashes were verified; only the lines cited above were
  read).
- Sampler records were counted and their frames read; sampler policy
  conformance was not otherwise verified.
- `completed_publication_inventory_sha256`, the stdout inventory digest and
  `scoring_request_sha256` were reproduced by data transformation of the
  retained records, not by running the runner's functions.
- Pre-freeze fixture tests and the independent source reviews were not
  re-examined for whether the cover V3 class-identity guard was exercised
  against native-adapter features; that is a successor-design question.
