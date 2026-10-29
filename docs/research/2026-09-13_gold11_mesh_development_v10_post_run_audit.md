# Gold11 full11 V10 - independent physical POST (execution custody and scoring ledger)

Date: 2026-09-13/14 (audit window from 23:06:46 UTC on 2026-09-13 by `date -u`;
the end of the window is the `written_utc` of the companion
`POST_RUN_AUDIT.json`, written before this report). Reviewer:
`independent-ai-post-auditor-v10` (Claude Fable 5.1, reviewer role; not the
author of the candidate, the freeze, the PRE, the run or any review).
Attempt audited: `docs/research/gold11_mesh_development_v10_attempt` (terminal;
frozen 2026-09-13 16:03:04 UTC; executed ONCE through the bound waiter
`scripts/gold11_v10_execute_waiter.py`, launched 16:26:45 UTC, observed exit
0 at 23:04:19 UTC). This is an engineering audit of execution custody and of
the retained zero-prediction scoring ledger. It is not manufacturing approval,
calibrated metrology, conformance, generalization or production evidence.
Every V10 outcome below is a COMBINED consumed-development change relative to
V9 (native feature bridge V1, refinement allowance V1 with its 1500 s
cooperative CPU guard inside the unchanged 1860 s cap, exact reuse of the
diagnosis full-scan query, the evaluated-subset registration contract with its
disjoint disposition ledger, the corrected completion-budget declaration and
bounded stage checkpoints, acting together on the unchanged V9 mechanisms,
policies and the provisional 1.0 mm admission gate). Nothing here is a
mechanism-only speedup or a calibrated accuracy improvement; the in-run arm
comparison is scan-plane lane and scheduler only on the same V17 engine (not
old engine versus new engine); the historical F1 0.4580152672 is not comparable
and is not beaten. No numeric F1 exists: the frozen scorer returned null for
both arms and null stays null.

Method: read-only. Only `cat`/`sed -n`/`grep`/`shasum -a 256`/`wc`/`ls -la`
(`-laT` for second-resolution mtimes)/`diff`/`tar -tvf` (listing) and
`tar -xOf` (single-member streaming into `shasum`), two
`ps -axo pid,pgid,etime,rss,command` queries, `date -u`,
`git status --porcelain` on named paths, and `.venv/bin/python` snippets that
read, parse (json), hash, `lstat` and tabulate files under the attempt, the
outer records, the external archive directory, the job scratch directory and
the named repository documents (the frozen `source/` snapshot was read with
`sed`/`grep` only: the runner's inventory/scoring-request text and the lines
of `cad_guided_mesh_offline_v17.py` cited below). No runner, projection,
scorer, test, freeze, verify, execute, worker or geometry/native code was
executed and no module under `backend/` or `scripts/` was imported. Not
opened, stat'ed, hashed, sampled or traversed: `data/**`, any `.ply`/`.step`,
the Gold and reference containers under `data/bend_corpus/annotations`,
`/private/tmp/gold11-*` (including the V10 coordinator cache named in the
custody records), `/var/folders/*`, `.env`, credentials, `output/`,
`outputs/`, pools/manifests, HOLDOUT10, incoming-email/EHO parts, and every
other `docs/research/*_attempt*` directory (V9 and earlier are terminal and
were not reopened; V9 figures come only from the released V9 POST, its
addendum and the V9 result narrative). The attempt's own
`SELECTED_REFERENCES.json` (the runner's post-terminal reference snapshot
copy) was hashed and its per-case bend counts and scan digests compared; no
reference geometry is transcribed here. The author's V10 result narrative and
the Codex registration postmortem, both written while this audit was in
progress, were not read. Process disclosure: four scratch files were written
in the output directory outside the repository (listed at the end); nothing in
the repository was modified. This report and `POST_RUN_AUDIT.json` are the
only retained writes.

## Decision

ACCEPT_WITH_FINDINGS: ACCEPT the execution custody of the terminal V10
campaign and its complete zero-prediction scoring ledger, as an engineering
audit only. Every finding below is non-blocking (blocking findings: none). The
limits of the acceptance:

1. The parent process exit is OBSERVED (returncode 0, recorded by the waiting
   parent 70512 for runner 70513, nonce `60084d20c0a6ffa9af8a8d66fc8f7e4b`;
   check 1). The runner's own release condition
   `OBSERVED_PARENT_EXIT_ZERO_FINAL_GATE_NO_VETO_AND_EXACT_INDEPENDENT_POST_AUDIT`
   is met by observation on its first two clauses (observed exit 0;
   `FINAL_GATE.json` present with no veto marker); its third clause is this
   document and is for the coordinator to bind. The outer exit record proves
   the outer return code only (its own `scope` field says so).
2. Both arms emitted zero supports: TP 0 / FP 0 / FN 71, precision null,
   recall 0.0, F1 null (undefined, not zero), minimum-family recall 0.0. This
   is the frozen scorer's own null under the frozen failure policy; it is
   identical to V9's ledger and is not an improvement over it.
3. No worker was CPU-terminated (V9: 14). All 22 leaves returned a complete
   `RESULT.json`; the cooperative CPU guard stopped refinement in 15 leaves at
   about 1500 s of process CPU and published 15 EXPIRED and 28 OMITTED
   candidates, which remain UNKNOWN by declaration (never rejected).
4. For the first time in this line of attempts the measurement consumer ran
   on physical data: three slots (CG1E0BBA2933 both arms, CG4BDD1371B6
   control) evaluated 43 context windows through bridge V1, kernel V2,
   context V3 and cover V4; every window refused at the role-0 flange plane
   fit (32 `FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY`, 11
   `FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL`), so no support record, carrier or
   prediction exists. The V9 cover feature-type guard refusal (class MG) did
   not recur.
5. The coordinator private cache `/private/tmp/gold11-v10-coordinator-qxw4iju4`
   is retained opaque outside the attempt (declared by the runner; not
   inspected here).
6. Raw scan/CAD bytes and the Gold catalog were not re-read here; input
   binding rests on the frozen declarations, each receipt's
   `source_binding_valid` and the RESULT-embedded scan/CAD digests.

No administrative failure, denominator exclusion, retry, resumed start,
relaxed limit, gate change after launch or edited frozen source was found.

## Bindings

Canonical digest = SHA-256 of `json.dumps(obj, sort_keys=True,
separators=(",", ":"), allow_nan=False).encode()` of the decoded object.
File SHA-256 values are of the raw bytes.

| Artifact (attempt root) | Bytes | File SHA-256 |
| --- | ---: | --- |
| FROZEN.json | 56925 | `dd93ccdd9ed0a90c85247b0b76859836f4e05b776e66c492713076592574b980` |
| PRE_RUN_AUDIT.json | 38389 | `9c14099a88506210d471a326ccc5397d541ae5afea145713749acb1acb3cfa0b` |
| STARTED.json | 944 | `395706986810364d2df35f9e5b2523fd70adec92fb4ff12bf36df715f6a4737a` |
| COORDINATOR_DISPATCH.json | 222 | `18ff2f6dc34f1d8ccb9e05e4ec5632effbaeb81435b70ef6dc44916deb636438` |
| COORDINATOR_VERIFIED.json | 11583 | `4fe60c14dc7d6c3f276c00ee6af58f3465e73f31a1a29f8b14c90db4845a8efd` |
| COORDINATOR_COMPLETED.json | 184 | `fea2373e4ea6b939a7ccc84e2b005dccf1f236a3e5a31d04097e0e79e2783100` |
| COORDINATOR_OUTPUT.txt | 4090 | `67edd8261f3312ef192620f7d3ac0ec7c82b85bbb11ef4a001cf3520223bf6c5` |
| INFERENCE_TERMINAL.json | 2125 | `2a60a1732ff2253101199d722aae2450f184fbc7cdd097c383811e37b5e9f36a` |
| SCORER_RECEIPT.json | 3211 | `14dae8f05dd475940c596ba6a1e275702b8e9808ca7c3b4ffa04d125894aae18` |
| SCORER_RESULT.json | 40981 | `b0fae7aa48b24bd897d72b5e1836ef40dc6e9475359971c4ced5e6677068e46f` |
| SCORER_OUTPUT.txt | 115 | (in `POST_RUN_AUDIT.json` bindings) |
| SCORER_ARTIFACT_RETENTION.json | 248 | (in `POST_RUN_AUDIT.json` bindings) |
| SCORED.json | 181854 | `6d01d2250edb160494743a9c205524882411d11d4c5054598bd68952d651e99e` |
| SELECTED_REFERENCES.json | 24751 | `d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc` |
| COMPLETED.json | 215560 | `39508761b17bbea26fddc804c239e906ee39eb901440e235e49ae3a8f324307b` |
| FINAL_GATE.json | 1057 | `1fa0d90a069719de21b4ada4a89e5069e17448e1d7a598ca46318cc205083ab7` |

Outer records beside the attempt directory (`docs/research/`):
`gold11_mesh_development_v10_attempt_outer_exit.json` 544 B
`94d346887385fae2d89d7f06b94b8e2a529cb5bd7c8786a2e2f24fd757dfa8d9`;
`..._outer_pid.txt` 113 B
`e09176ee3ca5e5c6ededc0df199064e71f9f1947fd0626bfd6e84131db7f430a`;
`..._outer_stdout.txt` 467 B
`693398d895732753d0eb8a7a73ce9b45c806c4f7a12179d60216583f6df50bcb`. No
`.tmp.<pid>` leftover exists beside them (the V9 outer records also stand
there, unchanged from 2026-09-13 13:34 UTC). PRE report
`docs/research/2026-09-13_gold11_mesh_development_v10_physical_pre_run_audit.md`
40038 B `313525fc3284159ba8f1232d6dbc5fb0e97436d90248aa2812f5c1ab8ab63a23`.
All hashes and byte counts equal the values declared in the commission and in
the progress ledger.

## Check 1 - outer exit evidence

- `_outer_exit.json`: `schema` `GOLD11_V10_OUTER_EXIT_RECORD_V1`, `scope`
  `OUTER_PROCESS_RETURN_CODE_ONLY_NOT_A_SCORE_OR_AUDIT`, `command` =
  `[.venv/bin/python, -I, -S, -B, -u, scripts/run_gold11_mesh_development_v10.py, --execute]`
  (the protocol's exact command), `nonce` `60084d20c0a6ffa9af8a8d66fc8f7e4b`,
  `observation` `OBSERVED_BY_WAITING_PARENT`, `observed_by`
  `WAITING_PARENT_PROCESS_70512`, `returncode` 0, `runner_pid` 70513,
  `started` `2026-09-13 16:26:45 UTC`, `finished` `2026-09-13 23:04:19 UTC`.
- `_outer_pid.txt` holds exactly one line:
  `FINISHED 60084d20c0a6ffa9af8a8d66fc8f7e4b waiter_pid=70512 runner_pid=70513 returncode=0 2026-09-13 23:04:19 UTC`
  (nonce, pids, return code and timestamp equal the exit record; the waiter
  rewrites this single-line reservation file in place, so earlier RESERVED /
  RUNNING states are not retained; the progress ledger records the RUNNING
  line observed at launch).
- `_outer_stdout.txt` is one line (467 B), the runner's final status line:
  `status` `COMPUTED_PENDING_INDEPENDENT_POST_AUDIT`, `release_allowed`
  false, `durable_terminal_case_records` 22,
  `final_campaign_inventory_sha256`
  `008b07bc680830656b50e2e64f9ab9cd3eccd5683a39b563c57bef2bf763da3c`,
  `final_campaign_inventory_entries` 309,
  `final_scientific_manifest_entries` 119,
  `final_inventory_stdout_is_not_durable_runner_artifact` true. In the
  frozen runner (`source/scripts/run_gold11_mesh_development_v10.py` lines
  3018-3058) this line is printed only after `COMPLETED.json`,
  `FINAL_GATE.json` and the post-FINAL_GATE inventory succeed, and `return 0`
  follows one more wall/RSS veto whose failure would write
  `FINALIZATION_FAILED.json` (absent); the observed returncode 0 is consistent
  with that path. Both inventory digests are reproduced in check 3.
- Process identity: runner 70513 = `COORDINATOR_DISPATCH.parent_pid` =
  `COORDINATOR_VERIFIED.outer_pid`; coordinator 70514 =
  `COORDINATOR_VERIFIED.coordinator_pid` = the `coordinator_pid` of every leaf
  dispatch/claim record = the coordinator receipt's `child_pid` and
  `owned_group_id`; first inference worker 70515; scorer worker 80824 =
  `SCORER_RECEIPT.child_pid`.
- Timestamps (UTC; from the inventory `lstat` fingerprints, equal to current
  mtimes): `FROZEN.json` 16:03:04.215; `PRE_RUN_AUDIT.json` placed
  16:26:14.339; waiter `started` 16:26:45; `STARTED.json` 16:26:45.425;
  `COORDINATOR_DISPATCH.json` 16:26:45.426; `COORDINATOR_VERIFIED.json`
  16:26:45.474; first case `STARTED.json` 16:26:45.475; last case terminal
  23:02:10.349; `INFERENCE_TERMINAL.json` 23:02:10.350; `COMPLETED.json`
  23:04:18.537; `FINAL_GATE.json` 23:04:19.004; outer records 23:04:19; waiter
  `finished` 23:04:19 (1 s resolution). FINAL_GATE - STARTED = 23853.6 s;
  `COMPLETED.parent_wall_seconds` 23853.009159167297;
  `FINAL_GATE.parent_wall_seconds` 23853.477465209085; outer span 23854 s.
  All consistent.
- Live check: `ps -axo pid,pgid,etime,rss,command` at 23:07:04 UTC matched no
  `run_gold11_|waiter|_gold11_v10_exact_scoring|cad_guided|gold11` process and
  neither pid 70512 nor 70513.
- `COMPLETED.release_requires` =
  `OBSERVED_PARENT_EXIT_ZERO_FINAL_GATE_NO_VETO_AND_EXACT_INDEPENDENT_POST_AUDIT`;
  `FINAL_GATE.final_release_requires` =
  `POST_FINAL_GATE_FULL_TREE_RESCAN_OBSERVED_EXIT0_AND_INDEPENDENT_POST`;
  `FINAL_GATE.status` `PENDING_INDEPENDENT_POST_AUDIT`;
  `COMPLETED.failures` `[]`. No `FINALIZATION_FAILED.json`, no
  `SCORING_UNAVAILABLE.json`, no `RESULT_PARTIAL.bin`, no
  `SCORER_RESULT_PARTIAL.bin` anywhere in the tree (309 on-disk paths = the
  307 inventoried entries plus `COMPLETED.json` and `FINAL_GATE.json`). The
  exit observation is the waiting parent's (the author's launcher on the same
  host), verified here for internal consistency; it was not observed by this
  reviewer.

## Check 2 - frozen bundle, PRE bindings, source drift, archive

- `FROZEN.json` 56925 B / `dd93ccdd...4b980`; canonical(FROZEN) =
  `ded97290231a6e8fab9ba1680790fd3feead45dde9c9912048bde41462546928` =
  `STARTED.bundle_sha256` = `COORDINATOR_VERIFIED.bundle_sha256` =
  `PRE_RUN_AUDIT.bundle_sha256` = `SCORED.bundle_sha256`. `schema`
  `GOLD11_MESH_DEVELOPMENT_V10`, `comparator`
  `FRESH_V17_ALLOWANCE_V1_BRIDGE_V1_KERNEL_V2_V7_SCHEDULE_V1_VS_COMPOSITE_SCHEDULE_V2_NATIVE_V1`,
  `production_use_allowed` false, `historical_comparability` `UNESTABLISHED`,
  `selected_reference_sha256`
  `d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc`,
  `catalog_sha256`
  `4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36`;
  `config_delta_from_v9` keys `completion_budget_declaration`,
  `inherited_v7_to_v8`, `inherited_v8_to_v9`, `native_feature_bridge`,
  `query_reuse`, `refinement_allowance`, `worker_stage_checkpoints`;
  `config.refinement_allowance` = {enabled true, main_refined_groups 12,
  completion_refined_groups 4, main_query_rows 6000000,
  completion_query_rows 3000000, cooperative_chunk_rows 32768,
  cpu_budget_seconds 1860.0, cpu_guard_seconds 1500.0}. Runtime metadata
  (python 3.11.15 arm64 darwin, numpy 2.4.6, scipy 1.17.1, cadquery-ocp
  7.9.3.1.1; version metadata, not binary attestation) equal in FROZEN and
  COORDINATOR_VERIFIED.
- `PRE_RUN_AUDIT.json` 38389 B / `9c14099a...3cfa0b`; canonical(PRE) =
  `3b3918ecbcc580594c1b3eff110f2042d4910e4df3881d501036d850bc6826d3` =
  `STARTED.audit_sha256` = `COORDINATOR_VERIFIED.audit_sha256`. PRE fields:
  `decision` `GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN`, `reviewer`
  `independent-ai-pre-auditor-v10`, `blocking_findings` `[]`, 13 non-blocking
  notes, `written_utc` 2026-09-13T16:16:42Z, checks 1-6 all PASS. The PRE
  bytes are not canonical JSON (35500 B canonical); the runner binds the raw
  hash (`final_source_audit.controls`) and the canonical digest
  (`STARTED.audit_sha256`) separately and both reconcile.
- Source drift (three-way): all 96 `FROZEN.source_sha256` entries (backend 65,
  scripts 25, docs 4, tests 2) equal the `source/` snapshot copy AND the
  working-tree file at `/Users/idantal/ShermanQC/FullProject`: 96/96 and
  96/96, 0 mismatches, 0 missing, 0 extra files in `source/` (96 regular
  files). `COORDINATOR_VERIFIED.source_sha256` == `FROZEN.source_sha256`;
  `COMPLETED.final_source_audit`: `complete` true, `unchanged` true,
  `runtime_metadata_unchanged` true, `errors` {}, `current` == `frozen` ==
  `FROZEN.source_sha256` (96 entries), `controls` = the five file hashes
  {COORDINATOR_DISPATCH, COORDINATOR_VERIFIED, FROZEN, PRE_RUN_AUDIT, STARTED}
  equal to the table above. No working-tree drift and no snapshot drift of
  any frozen source. Frozen V10 runner 188070 B /
  `f7fd083f8b74a22e7fd1afe872ae8b9b2bfbc94f4b7024fc8c2b46494208da92`
  (equal to the closure table of the review response); declared V9 basis
  `4567a0f584e15b75339238b3f50ebb19209f127bfcb3a55e5fa73fd36b4a85b1`.
- External archive
  `docs/research/gold11_mesh_development_v10_physical_frozen.q4bxmlh3/SOURCE.tar`:
  `2cb7d6fd17012b23e539d754d3dfdea779fc334ad8e2524a74a5ef43bb36e255`, 3858944
  B (expected values). `tar -tvf` lists 103 entries: 6 directories and 97 file
  members = `FROZEN.json` (56925 B) plus exactly the 96 `source/<path>`
  members (0 extra, 0 missing). EVERY member payload was streamed with
  `tar -xOf` and hashed: 97/97 equal the author's manifest
  `SOURCE.tar.members.sha256.json` (37841 B,
  `e62eda42f1313784792fd1b2ce55408782890d97ba4b8f65dab27458dbe4290c`, schema
  `GOLD11_V10_FROZEN_ARCHIVE_MEMBER_PAYLOAD_HASHES_V1`, `mismatches` [],
  `missing_from_archive` [], `unexpected` []) on both `payload_sha256` and
  `on_disk_sha256`; 96/96 source members equal `FROZEN.source_sha256`; the
  `FROZEN.json` member hashes to `dd93ccdd...4b980`; all 97 listed sizes
  equal the manifest sizes. Nothing was extracted to disk (the listing and
  the member hashes were written to two scratch files outside the
  repository, listed at the end).
- `STARTED.limits` == `FROZEN.limits` == the limits of all 22 case
  `STARTED.json` records (case wall/CPU 3540/1860 s; worker sampled tree /
  coordinator HWM / outer tree 8589934592 / 4294967296 / 12884901888;
  scan/CAD 67108864 / 8388608; log/result 1048576 / 134217728; projection and
  receipt 8388608; scorer 180 s / 120 s / 2147483648 / 201326592 request /
  33554432 result; coordinator log 2097152 / parent read 180 s; campaign
  104400 s / reserve 2400 s): value for value the V9 limits of the released
  V9 POST and the protocol's resource table (19 values; the guard 1500 s is
  `CONFIG.refinement_allowance.cpu_guard_seconds`, the checkpoint 65536 B is
  the `STAGE_LOG_DECLARATION` inside every leaf log). Every case
  `STARTED.json` carries `no_resume` true and the frozen arm request digest
  (control `463dab0ea3bcf8cb4c79731b0d91a3ad0a8e99bf3ff4f1f193d5c0010ac3077b`,
  challenger `88380c13c3268116cdd6ac62e051db66b540b483ab906a3c2a3d057cbf237ec8`);
  canonical(all arm requests) =
  `39b8db734ef6a79b6e564ac5fedb81396ca4cc4888377fe1e6b70351e0488713` =
  `STARTED.arm_requests_sha256` = `INFERENCE_TERMINAL.arm_requests_sha256`.
  `SCORED.arm_requests`, `arm_definitions`, `config_delta_from_v9`,
  `measurement_policy_declaration`, `pose_admission_declaration`,
  `scoring_transport`, `certificate_caps`, `two_plane_declaration`,
  `retained_control_policy`, `artifact_policy` and
  `position_error_declaration` are object-equal to FROZEN.
- Frozen policy values present in every returned policy snapshot (22/22):
  `measurement` {0.03 mm max fit residual, 0.04 mm plane exclusion,
  free_circle, flange window 2.5-12.5 mm}, `support.max_angular_gap_deg`
  30.0, registration gate `max_inlier_plane_p95_mm` 1.0, and
  `registration.evidence.refinement_allowance` ==
  `initialization.refinement_allowance` == `CONFIG.refinement_allowance`.

## Check 3 - execution custody

- canonical(STARTED) = `aac14b59c82cfcab14c8a2ddac1485b3e53a8be1ecf3b2ac0cd193747ed6b701`
  = `COORDINATOR_VERIFIED.started_sha256`; canonical(COORDINATOR_DISPATCH) =
  `f2d614c86c56831571645927287616fdd9c63a643d3388ac8fa32ac8b4ee9251` =
  `COORDINATOR_VERIFIED.coordinator_dispatch_sha256`; `COORDINATOR_VERIFIED`
  schema `ONE_USE_PARENT_BOUND_LEAF_V10`, canonical =
  `01981e40304f0bb767bb6262fdec2c868b2749250caed8375050726084d502aa` = the
  `context_sha256` of all 23 leaf dispatch records.
- Coordinator dispatch: cache `/private/tmp/gold11-v10-coordinator-qxw4iju4`,
  identity device 16777231 / inode 35878705 / uid 501, parent pid 70513;
  `COMPLETED.coordinator_private_cache_custody` names the same path and
  identity with status `OPAQUE_PRIVATE_CACHE_RETAINED_OUTSIDE_ATTEMPT`,
  `contents_enumerated_or_read` false (not inspected here either).
- COMPLETED: `status` `COMPUTED_PENDING_INDEPENDENT_POST_AUDIT`, `failures`
  `[]`, `parent_wall_seconds` 23853.009159167297, `parent_peak_rss_bytes`
  1595310080, `no_resume` true, `production_use_allowed` false; the file is
  canonical JSON plus one newline (as are FINAL_GATE, STARTED, both
  COORDINATOR_* JSON controls, INFERENCE_TERMINAL, SCORED, FROZEN and the two
  scorer receipts; SCORER_RESULT and SELECTED_REFERENCES are canonical with
  no newline; PRE_RUN_AUDIT is not canonical). canonical(COMPLETED) =
  `2aca770f09107828ba68759088984eb12a23afc5276e9d3c6345272fedcc38ef` =
  `FINAL_GATE.completed_canonical_sha256`; canonical(final_source_audit) =
  `7269720ae927ef66d912ea4e46da714a6ad78774e7482139a538a520ca765e19` =
  `FINAL_GATE.source_and_control_audit_sha256`;
  canonical(owned_artifact_custody) =
  `2c95122fe1b093792f3d7915203f13a7cc6e402a2d147de5856d182d4cf32b54` =
  `FINAL_GATE.owned_artifact_custody_sha256`. FINAL_GATE
  `scored_report_sha256` = SCORED file hash, `scorer_result_sha256` =
  SCORER_RESULT file hash = `SCORER_OUTPUT.result_sha256` =
  `SCORER_ARTIFACT_RETENTION` artifact hash (40981 B,
  `COMPLETE_RESULT_BYTES_RETAINED`, no leftover partial),
  `reference_snapshot_sha256` `d16390b2...b45dc`, `final_gate_self_hash_claim`
  false, `inventory_boundary`
  `AFTER_COMPLETED_BEFORE_FINAL_GATE_WITH_EXACT_COMPLETED_BYTES`.
  `COORDINATOR_COMPLETED.json`: same status, `parent_wall_seconds`
  23787.54732716689, `no_resume` true, `historical_comparability`
  `UNESTABLISHED`.
- Coordinator receipt (supervision V3): exit_code 0, budget_stop null,
  child_pid 70514, wall 23787.62068737531 s, direct_child_reaped true,
  owned_group_empty_confirmed true (group 70514), requires_process_attention
  false, foreign escape / unknown identity false, cleanup/control errors [],
  457283 RSS samples, sampled tree peak 2474196992 B + parent 42188800 B =
  budget peak 2516353024 B (below 12 GiB), log 4090 B bound.
  `identity_counters`: admitted 24, verified_absence_retirements 23,
  events_discarded 7 (16-event cap), tracked_peak 2, membership races 0;
  `exit_transition_diagnostics` 4 entries (pids 72904, 76042, 78242, 78326)
  each resolved as absence within 0.0015-0.0106 s. `COORDINATOR_OUTPUT.txt`
  holds exactly the 22 case lines in frozen order (status ADMITTED, phase
  `SEE_EXACT_RESULT_AND_SCHEDULE_PROJECTION`, walls equal to the 22 receipts'
  `parent_wall_seconds`, `budget_stop` null) plus the final status line.
- Owned artifact custody (`GOLD11_OWNED_ARTIFACT_CUSTODY_V10`): `complete`
  true, 119 declared files re-hashed here, 0 mismatches; 24 logs all
  `COMPLETE_LOG_BYTES_BOUND`; `scoring_request_sha256`
  `955f249070bb8427279b84224c22815b06d4a3c66ba29b59c26b10bda18d0732`;
  campaign tree inventory `COMPLETE_CAMPAIGN_TREE_CUSTODY_V10` complete, phase
  `ALL_SCHEMA_BRANCHES_VISITED`, 307 entries (266 regular files: 119
  scientific, 96 source, 46 dispatch, 5 control; 41 directories), errors [],
  `deadline_expired` false, `first_unrecorded_path` null, read_bytes
  1213905186 (exact), schema slots 332 / metadata limit 460. All 307
  inventory fingerprints (device, inode, mode, links, bytes, mtime_ns,
  ctime_ns) equal the current `lstat` values, all 42 directory identities
  equal, and all 266 recorded file hashes equal current bytes; the only
  on-disk names outside that inventory are the two later publications
  `COMPLETED.json` and `FINAL_GATE.json` (309 total).
- Inventory reproduction: adding the current `COMPLETED.json` fingerprint,
  hash and bytes to the retained pre-COMPLETED inventory exactly as
  `verify_campaign_inventory` composes a published control (one more CONTROL
  slot with its 256 KiB cap, `published_control_names` ["COMPLETED.json"],
  +215560 read bytes, the name added to the root directory's expected /
  initial / final name lists, `digest(encoded(completed)+b"\n")` = the file
  hash) reproduces `FINAL_GATE.completed_publication_inventory_sha256` =
  `68d0b3f5493a0526cfb603ccbd1a847b82194e31f62ba9f2e83c35300ddee14c` (308
  entries); adding `FINAL_GATE.json` (+1057 read bytes) reproduces the parent
  stdout's `final_campaign_inventory_sha256` =
  `008b07bc680830656b50e2e64f9ab9cd3eccd5683a39b563c57bef2bf763da3c` with 309
  entries / 119 scientific / 1214121803 read bytes. The final on-disk
  identities of both control files are therefore the ones the runner itself
  inventoried.
- Terminal fill: all 22 `terminal_case_records` durable, every record
  `PREEXISTING_VALID_NO_WRITE` (no fallback terminal was written), each
  binding the `COMPLETED.json` and `SCORING_INPUT.json` hashes verified here.
- Leaf dispatch: `leaf_dispatch/` holds 46 regular files (the directory link
  count 48 reported by `ls -la` counts `.` and `..`) = 23 issued/claimed pairs
  (22 INFERENCE + 1 SCORER); every id equals canonical({role, binding}),
  `context_sha256` = canonical(COORDINATOR_VERIFIED), claim `dispatch_sha256`
  = canonical(issued record), `no_retry` true, coordinator pid 70514, worker
  pids equal the receipts' `child_pid` (23 distinct: 70515, 71133, 71694,
  72339, 72904, 73523, 74128, 74720, 75045, 76042, 76690, 77428, 78189, 78214,
  78242, 78290, 78326, 78328, 78339, 78982, 79603, 80212 and scorer 80824),
  private cache path and identity equal the receipts' custody records, all
  reported `REMOVED_AFTER_QUIESCENCE_AND_DERIVED_RETENTION`; each inference
  binding's scan/CAD hashes equal the frozen case declaration and its
  `arm_request_sha256` the frozen arm digest; the scorer binding's
  `request_sha256` equals `SCORED.scoring_request_sha256`; the 22 inference
  ids are exactly `required_leaf_dispatch_ids`. Each of the 22 case
  directories holds exactly one `STARTED.json`, one receipt, one projection,
  one log and one `RESULT.json`; the 22 slots ran strictly serially in the
  frozen order, control then challenger (each `STARTED.json` mtime after the
  previous slot's `COMPLETED.json`).
- INFERENCE_TERMINAL: case_count 11, arm_count 2, terminal_record_count 22,
  `outcomes_sha256`
  `65ff9ca8042bee7533fb33b5e70d9d73e77c786089d6552cc468e8bf238733af`
  reproduced here as canonical({arm: {case: SCORING_INPUT object}}) from the
  22 retained `SCORING_INPUT.json` objects; 22 non-null result bindings equal
  the receipts, the projections and the actual RESULT.json hashes;
  parent_wall 23724.775835749693 s (sum of the 22 per-case parent walls
  23724.742859 s).
- Scoring transport: `SCORED.projection_bindings` equal the receipts on
  arm_request_sha256 / scoring_input_sha256 / result_sha256 / projection_kind
  (`EXACT_RETURNED_RESULT_SCHEDULE_WRAPPER`) for all 22;
  `scorer_input_canonical_sha256` equals canonical(SCORING_INPUT
  .contact_projection) for all 22 (wrapper
  `GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V5`, inner
  `GOLD11_CONTACT_WORKFLOW_PROJECTION_V4`, wrapper `disposition` ADMITTED,
  `result_binding.matched` true with supplied bytes equal to the RESULT file
  size). Rebuilding the V6 scoring request as `make_scoring_request` composes
  it (runner lines 1925-1942: the 22 inner objects, the snapshot's `bends`,
  the frozen case->family map, the frozen arm requests, the scan-plane
  requests by arm, the projection bindings and the selected-reference digest)
  gives canonical `955f2490...0732` (126186 B) = `SCORED.scoring_request_sha256`
  = the scorer leaf binding = `owned_artifact_custody.scoring_request_sha256`.
  `SCORED.workflow_stage_attribution` equals the 22 receipts'
  `workflow_stage_diagnostics`. canonical(SELECTED_REFERENCES) = the file
  bytes = `d16390b2...b45dc` = `FROZEN.selected_reference_sha256`; 11 cases,
  71 bends (6,5,10,4,3,7,6,8,4,8,10), every case's `scan_sha256` equal to the
  frozen declaration, 9 `reference_mesh_raw` and 2 `scan_raw` frames
  (CG1E0BBA2933, CGF41F49066A), equal to
  `SCORED.case_declared_coordinate_frames` and to the V9 snapshot. The
  catalog hash `4ad243c1...b36` is a declaration verified by the runner's
  `selected_references()` at scoring time; the catalog was not opened here.
- Scoring-after-terminal ordering (epoch ns from the inventory fingerprints):
  last case terminal `cases/CGF41F49066A/challenger/COMPLETED.json`
  23:02:10.349Z; `INFERENCE_TERMINAL.json` 1789340530350232786
  (23:02:10.350Z); `SELECTED_REFERENCES.json` 1789340592864669077
  (23:03:12.865Z, +62.514 s); scorer dispatch issued 23:03:12.873 and claimed
  23:03:12.913; `SCORER_OUTPUT.txt` 23:03:13.089; `SCORER_RESULT.json`
  23:03:13.106; `SCORER_ARTIFACT_RETENTION.json` and `SCORER_RECEIPT.json`
  23:03:13.107-108; `SCORED.json` 23:03:13.121; `COORDINATOR_COMPLETED.json`
  and `COORDINATOR_OUTPUT.txt` 23:03:13.122; `COMPLETED.json` 23:04:18.537;
  `FINAL_GATE.json` 23:04:19.004. No file under `cases/` has an mtime after
  `INFERENCE_TERMINAL.json`; all 44 inference dispatch files precede it and
  only the scorer pair follows it.

## Check 4 - resource receipts against the frozen LIMITS

All 22 slots (frozen case order, control C then challenger H). `wall` = leaf
receipt `wall_seconds`; `pwall` = per-case parent wall; `cpu` =
worker-reported `worker_cpu_seconds` from the log's `INFERENCE_TERMINAL` line
(every worker returned, so every value is final); `hwm` = worker-reported
`worker_peak_rss_bytes`; `tree` = supervisor `sampled_tree_peak_rss_bytes`;
`ckpt` = `STAGE_CHECKPOINT` lines in OUTPUT.txt; `smp` = sampler records;
`log` = OUTPUT.txt bytes (= forwarded = stored = accepted = attempted);
`guard` = `cost_receipts.cpu_guard_tripped`; `RESULT` = retained RESULT.json
bytes and hash prefix. Every receipt has `status` TERMINAL, `exit_code` 0,
`budget_stop` null, `supervisor_failure_reason` null, `inference_started`
true, `source_binding_valid` true, `derived_result_retention_complete` true,
`retained_result_artifacts.status` `COMPLETE_RESULT_BYTES_RETAINED` (no
leftover partial), `private_cache_cleanup_complete` true, no
cleanup/identity/control errors, `requires_process_attention` false.

| Case | Family | Arm | wall s | pwall s | cpu s | hwm B | tree B | ckpt | smp | log B | guard | RESULT |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CG08506AE97D | 10839000 | C | 1510.066 | 1515.824 | 1508.734 | 1863417856 | 1863417856 | 13 | 50 | 83288 | yes | 71237474 `5e2abde35d61` |
| CG08506AE97D | 10839000 | H | 1513.074 | 1518.819 | 1511.724 | 2062778368 | 2058944512 | 13 | 50 | 83106 | yes | 72687899 `e50ab55e10be` |
| CG191DEC7F81 | 49001000 | C | 1515.216 | 1519.333 | 1513.900 | 1435172864 | 1434288128 | 17 | 50 | 84304 | yes | 54887107 `c7f91f89168c` |
| CG191DEC7F81 | 49001000 | H | 1514.714 | 1519.207 | 1513.387 | 1652490240 | 1647640576 | 17 | 50 | 83991 | yes | 56283317 `4836d6971752` |
| CG1E0BBA2933 | 38172000 | C | 1510.468 | 1516.060 | 1509.146 | 1531838464 | 1531838464 | 11 | 50 | 83025 | yes | 78582907 `6c4dfe9afdc7` |
| CG1E0BBA2933 | 38172000 | H | 1522.089 | 1527.615 | 1520.687 | 1790689280 | 1790181376 | 11 | 50 | 82483 | yes | 79941681 `1fc6b27587ca` |
| CG4BDD1371B6 | 37361005 | C | 1513.467 | 1518.271 | 1512.109 | 1449852928 | 1449852928 | 13 | 50 | 83031 | yes | 69032640 `b73d4481c605` |
| CG4BDD1371B6 | 37361005 | H | 669.882 | 674.479 | 669.159 | 1650245632 | 1644838912 | 11 | 22 | 38635 | no | 61749740 `cbea64595445` |
| CG4FAF273093 | 49024000 | C | 1503.479 | 1504.391 | 1502.213 | 559529984 | 559529984 | 22 | 50 | 85131 | yes | 17083245 `70d74b3ab911` |
| CG4FAF273093 | 49024000 | H | 1503.400 | 1504.370 | 1502.135 | 576913408 | 576913408 | 20 | 50 | 84652 | yes | 17650277 `f078e2fef9d7` |
| CG64C412CC73 | 40500000 | C | 1533.147 | 1536.393 | 1531.793 | 1075314688 | 1075003392 | 8 | 51 | 83528 | yes | 48847597 `9e5162bccd92` |
| CG64C412CC73 | 40500000 | H | 1557.612 | 1561.009 | 1556.227 | 1222377472 | 1218052096 | 8 | 51 | 83676 | yes | 49645848 `c5afc9599283` |
| CG65CD575B1C | 38172000 | C | 30.885 | 35.571 | 30.720 | 1622376448 | 1622376448 | 2 | 1 | 3271 | no | 57311147 `4e3986f2126f` |
| CG65CD575B1C | 38172000 | H | 47.472 | 52.454 | 47.315 | 1840414720 | 1840414720 | 2 | 1 | 3763 | no | 58956282 `d1daa3213fe3` |
| CG84C8FA12D2 | 10839000 | C | 43.090 | 49.230 | 42.942 | 1664122880 | 1664122880 | 2 | 1 | 3866 | no | 74646648 `9bbbd0d899f5` |
| CG84C8FA12D2 | 10839000 | H | 63.612 | 69.930 | 63.442 | 1944403968 | 1944403968 | 2 | 2 | 4885 | no | 75820321 `cac29d04f4bb` |
| CGE4B0549E41 | 10839000 | C | 12.473 | 12.517 | 12.387 | 1698906112 | 1698906112 | 0 | 0 | 1814 | no | 92971 `bb39d03ab312` |
| CGE4B0549E41 | 10839000 | H | 12.673 | 12.718 | 12.586 | 1755201536 | 1755201536 | 0 | 0 | 1817 | no | 98148 `ee1ccb8afc74` |
| CGEBA9A869AC | 38172000 | C | 1517.257 | 1521.851 | 1515.908 | 1466302464 | 1464467456 | 9 | 50 | 82323 | yes | 62455708 `430cee85b73d` |
| CGEBA9A869AC | 38172000 | H | 1509.166 | 1513.991 | 1507.819 | 1617592320 | 1612529664 | 9 | 50 | 82306 | yes | 63763946 `eeeca9fbbb4a` |
| CGF41F49066A | 38172000 | C | 1508.462 | 1513.886 | 1507.148 | 1555595264 | 1555283968 | 9 | 50 | 82557 | yes | 67996166 `ff900f386bee` |
| CGF41F49066A | 38172000 | H | 1521.276 | 1526.821 | 1519.944 | 1785036800 | 1780350976 | 9 | 50 | 82029 | yes | 69294451 `3cda2123b8cb` |

Sums: leaf walls 23632.980 s; parent walls 23724.743 s; worker CPU (all 22
workers, every value final) 23611.428 s. Largest worker CPU 1556.227 s (CG64C
H), largest worker high-water RSS 2062778368 B (CG08506 H), largest sampled
tree 2058944512 B (CG08506 H): all below 1860 s / 8 GiB; all leaf walls below
3540 s (largest 1557.612 s); all logs below 1 MiB (largest 85131 B); largest
RESULT 79941681 B below 128 MiB; largest SCORING_INPUT 10965 B and largest
leaf COMPLETED.json 10394 B below 8 MiB; COMPLETED 215560 B and FINAL_GATE
1057 B below 256 KiB; SCORED 181854 B and SCORER_RESULT 40981 B below 32 MiB.
Scorer receipt: exit 0, budget_stop null, wall 0.23281520884484053 s, sampled
tree peak 89341952 B (below 180 s / 2 GiB), log 115 B bound, cache
`/private/tmp/gold11-v10-scorer-tihgp5hl` (inode 35898383) removed. Parent:
lifetime high-water RSS 1595310080 B (below the 4 GiB coordinator HWM cap and
the 12 GiB outer tree cap), wall 23853.009 s and FINAL_GATE wall 23853.477 s
(below 104400 s; 23853.5 + 2400 s reserve = 26253.5 s < 104400 s). Sampler
records total 779 (V9 949); RSS samples 455174. Every OUTPUT.txt line is JSON
of one of five kinds (`PUBLIC_WORKFLOW_CALL_BEGIN`, `OBSERVED_STACK`,
`STAGE_CHECKPOINT`, `ENGINE_STAGE_SAMPLER_TERMINAL`, `INFERENCE_TERMINAL`) or
blank; each worker has exactly one `PUBLIC_WORKFLOW_CALL_BEGIN` line (arm
equal to the slot), one stopped sampler receipt (`samples_attempted` =
`samples_written`, `failure_type` null, `sample_limit_reached` false,
`startup_failure_type` null) and one `INFERENCE_TERMINAL` line whose
`result_sha256` equals the receipt, the retention artifact, the actual file
hash, `INFERENCE_TERMINAL.result_bindings` and the projection's
`result_binding.supplied_sha256`. No CPU-limit termination and no per-slot
wall timeout occurred (`budget_stop` null in all 22; exit 0 in all 22).
Sampled values are not continuous guarantees.

### Stage checkpoints (new V10 evidence)

Every leaf log carries the same `log_declaration` (`whole_leaf_log_bytes`
1048576 = `sample_stream_bytes` 524288 + `receipt_bytes` 4096 +
`checkpoint_stdout_bytes` 65536 + `checkpoint_cap_notice_bytes` 128 +
`other_output_remaining_bytes` 454528; `checkpoint_scope`
`V10_BOUNDED_STAGE_CHECKPOINT_STDOUT_LINES_EMITTED_EVIDENCE_OF_A_KILLED_LEAF_NEVER_A_RESULT`).
208 `STAGE_CHECKPOINT` lines in total (13, 13, 17, 17, 11, 11, 13, 11, 22,
20, 8, 8, 2, 2, 2, 2, 0, 0, 9, 9, 9, 9 in frozen order); distinct stages
`COMPLETION_PREFILTER_BEGIN/END`, `REFINEMENT_BEGIN`,
`REFINE_CANDIDATE_BEGIN/END/OMITTED`, `CONTEXT_PUBLICATION_BEGIN/END`; no
cap-notice line anywhere; the (stage, lane) sequence of each log equals the
`registration.evidence.stage_checkpoints.records` retained in that slot's
RESULT.json (22/22; ledger `truncated` false everywhere, at most 22 records /
2633 bytes of the 512 / 65536 caps). Because no leaf was killed, the "last
checkpoint of a killed leaf" question is vacuous: the 16 leaves with passing
start groups all reached `CONTEXT_PUBLICATION_END` (at process CPU
1501.0-1552.9 s in the 15 guard-stopped leaves and 664.7 s in CG4BDD H), the
four no-start leaves (CG65C, CG84C) stopped at `COMPLETION_PREFILTER_END`
(24.8-55.3 s), and the two CGE4B leaves emitted no checkpoint (initialization
unavailable before the completion prefilter). Guard-stop points differ
between arms (e.g. CG08506 C stopped at 1503.7 s, H at 1506.3 s; the expired
CG08506 candidate cost 225.19 s in C and 176.74 s in H), as the declared
timing dependence predicts.

Per-slot RESULT.json checks (22 files): `cad_sha256` and
`scan_topology.source_sha256` equal the frozen case declaration; `schema`
`CAD_GUIDED_MESH_OFFLINE_V13` (transport) with `implementation`
`CAD_GUIDED_MESH_OFFLINE_V17_CONTEXT_V4_COMPLETION_V1_KERNEL_V2_NATIVE_BRIDGE_V1_ALLOWANCE_V1`,
`initialization_implementation`
`CAD_GUIDED_PLANE_INITIALIZATION_V16_CONTEXT_V4_COMPLETION_V1_ALLOWANCE_V1`,
`registration_implementation`
`SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V5_CHUNKED_QUERIES_REUSE_V1_ALLOWANCE_V1`,
`kernel_implementation` `CAD_GUIDED_MEASUREMENT_V2_LAYER_SELECTION_AND_DIAGNOSTICS`,
`two_plane_context_configuration_sha256`
`12900ddac835ccd99208019b77d4bcce3443ae3c5c9dd0ea82aba89ee5f5db66` in all 22;
`registration.evidence.publication_contract`
`EVALUATED_ALLOWANCE_SUBSET_EXPLICIT_DISJOINT_DISPOSITION_LEDGER_V1` and
`candidate_disposition_ledger.schema` `candidate_disposition_ledger_v1` in all
22; `status` UNAVAILABLE, `predictions` [], `observed_bend_count` null,
`conformance` INDETERMINATE in all 22.

## Check 5 - scoring ledger

`SCORED.json` schema `GOLD11_CONTACT_GEOMETRIC_DEVELOPMENT_COMPARISON_V4`,
status `COMPUTED_DEVELOPMENT_DIAGNOSTIC`, experiment_schema
`GOLD11_MESH_DEVELOPMENT_V10`, control identity
`FRESH_V17_ALLOWANCE_V1_BRIDGE_V1_COMPLETION_V1_KERNEL_V2_V7_SCAN_PLANES_SCHEDULE_V1_NATIVE_V1`,
challenger identity
`FRESH_V17_ALLOWANCE_V1_BRIDGE_V1_COMPLETION_V1_KERNEL_V2_COMPOSITE_SCAN_PLANES_SCHEDULE_V2_NATIVE_V1`,
comparator
`FRESH_V17_ALLOWANCE_V1_BRIDGE_V1_KERNEL_V2_V7_SCHEDULE_V1_VS_COMPOSITE_SCHEDULE_V2_NATIVE_V1`,
wrapper schema `GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V5`, request
schema `GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_REQUEST_V6`,
`result_interpretation`
`COMBINED_CONSUMED_DEVELOPMENT_CHANGES_NOT_MECHANISM_ONLY_SPEEDUP_OR_CALIBRATED_ACCURACY`,
`arm_comparison_scope`
`SCAN_PLANE_LANE_AND_SCHEDULE_ONLY_SAME_V17_ENGINE_NOT_OLD_VS_NEW_ENGINE`,
`comparison_scope_note` (both arms run the same V17 engine under the same
allowance; any difference is the scan-plane lane or schedule plus
timing-dependent CPU-guard stops; historical 0.4580 not comparable, never
claimed beaten), `combined_change_attribution` true,
`historical_score_beaten_claim_allowed` false, `historical_score_comparability`
`UNESTABLISHED`, `accepted_registration_claim_allowed` false,
`production_use_allowed` false, `floor_release_allowed` false,
`strict_physical_engine_targets_met` false, `observed_bend_count` null,
`conformance` INDETERMINATE, `uncertainty_status` UNCALIBRATED,
`reference_line_count` 71, `family_count` 6, `case_count` 11,
`reference_snapshot_sha256` `d16390b2...b45dc`, `bundle_sha256`
`ded97290...6928`, `all_declared_input_bytes_verified` true,
`source_before_after_consistent` true, `phase_attribution_caveat`
`MISSING_OR_KILLED_INTERNAL_STAGE_UNKNOWN_USE_RETURNED_NESTED_LEDGERS`.
Matcher unchanged (`UNCHANGED_LEGACY_ONE_TO_ONE_MINIMUM_COST`, axis 5.0 deg,
transverse 4.0 mm, Gold-span overlap 0.6, prediction-span overlap 0.6).
Scoring scope
`GENERIC_CONDITIONAL_CONTACT_GENERATOR_LOCALIZATION_NOT_INNER_ARC_OR_MANUFACTURING_EVENT`,
`inner_face_specific_claims`
`NONE_UNRESOLVED_CARRIERS_SCORED_AS_GENERIC_CONTACT_LINES_ONLY`,
`carrier_identity_policy`
`ELIGIBILITY_IS_A_DECLARED_FUNCTION_OF_MATERIAL_SIDE_STATUS_NEVER_A_DEFAULT`,
`material_side_status` `PER_CARRIER_DECLARED_UNVERIFIED_SEE_CASE_DISPOSITIONS`,
`failure_policy` `WHOLE_CASE_ZERO_PREDICTIONS_ALL_REFERENCE_FNS_NO_PREFIX_OR_DROPPED_CASE`.
All 37 original `SCORER_RESULT.json` fields are canonically identical in
`SCORED.json` (0 differing fields; 32 runner-added enrichment fields only);
`SCORER_RESULT.json` is in canonical byte form. The scan-plane requests by arm
(control mode v7, configuration null; challenger `composite_v1` with the
frozen composite and seeded budgets) are identical in SCORER_RESULT, SCORED
and every retained projection.

Exactly as the scorer returned them (identical for both arms):

| Metric | Control | Challenger | Paired delta |
| --- | ---: | ---: | ---: |
| True positives | 0 | 0 | 0 |
| False positives | 0 | 0 | 0 |
| False negatives | 71 | 71 | 0 |
| Precision | null | null | null |
| Recall | 0.0 (0 of 71) | 0.0 (0 of 71) | 0.0 |
| F1 | null | null | null |
| Minimum-family recall | 0.0 | 0.0 | 0.0 |
| Emitted support line count | 0 | 0 | - |
| Exact emitted-support-count accuracy | 0.0 (0 of 11 cases) | 0.0 (0 of 11) | 0.0 |
| Exact localized target-set accuracy | 0.0 (0 of 11 cases) | 0.0 (0 of 11) | 0.0 |
| Refused / malformed / complete-empty cases | 11 / 0 / 0 | 11 / 0 / 0 | - |
| Physical observed counts unknown | 11 of 11 (`UNMET_NO_PHYSICAL_OBSERVED_COUNTS`, rate 1.0) | 11 of 11 | - |

Per-family false negatives (both arms): 10839000 18 (3 cases), 37361005 4,
38172000 34 (4 cases), 40500000 7, 49001000 5, 49024000 3 (sum 71); family
paired deltas all 0 / null. Per-case dispositions (both arms): all 11
`REFUSED` with reason `EXPLICIT_WORKFLOW_UNAVAILABLE` (no supervisor refusal
exists in V10; V9 had 14), `prediction_count` 0, `false_negative` =
`gold_count` (6,5,10,4,3,7,6,8,4,8,10), every Gold id listed under
`unmatched_gold_ids`, precision and F1 null, recall 0.0, `observed_bend_count`
null, `result_binding.matched` true with the RESULT hash. Reported phases:
`WHOLE_PART_REGISTRATION` for CG08506, CG4FAF, CG64C, CG65C, CG84C, CGE4B,
CGEBA, CGF41 (both arms) and CG4BDD challenger; `NATIVE_FEATURES` for CG191
(both arms); `COMPLETE` for CG1E0 (both arms) and CG4BDD control. In those
three `COMPLETE` slots the projection diagnostics are integers over zero
carriers: `registration_context_count` 1, `source_context_window_count` 13 /
13 / 17, `subset_registration_count` 1 (the registration published an
evaluated allowance subset), `support_group_count`, `measured_group_count`,
`covered_group_count`, `emitted_carrier_count`,
`identity_unresolved_carrier_count`, `inner_face_eligible_carrier_count`,
`finite_supported/refused_group_count` and `source_support_record_count` all
0; in the other 19 slots every diagnostic is null (no context reached
measurement). `inner_face_eligible_carrier_count` is therefore 0 where
defined, as the protocol expected, but over zero carriers. Null precision and
F1 are preserved as null (undefined), never coerced to zero, in SCORER_RESULT,
SCORED, every per-case, per-family and paired field; NO NUMERIC F1 EXISTS for
either arm of this run, under scorer status `COMPUTED_DEVELOPMENT_DIAGNOSTIC`.
All 22 slots are in the denominator and the 71 references are retained as
false negatives in both arms; every reference false negative of every refused
case is accounted for by that case's `prediction_count` 0 in the per-slot
projection (ADMITTED wrapper, contact projection REFUSED /
`EXPLICIT_WORKFLOW_UNAVAILABLE`, `predictions` [] in all 22).

## Check 6 - per-slot outcome taxonomy (from the attempt's own records)

Mutually exclusive over all 22 slots, from the retained RESULT ledgers
(workflow -> registration -> initialization -> completion lane -> allowance
ledgers -> measurement) and the supervisor receipts. Codes: PF = completed
finite prefilter (main plus completion lane) with no passing start; MW =
scan-plane proposal triangle-membership refusal (completion lane not
attempted); RG = candidates refined, none supported (1.0 mm gate or finite
support / coverage), registration UNAVAILABLE; EX = every selected candidate
expired or omitted by the CPU guard, none completed; NF = supported context
but native features UNAVAILABLE (`INVALID_NATIVE_SOLID`), nothing measurable;
MR = supported context, measurement dispatched through bridge V1, every
window refused at the role-0 plane fit. CPU (kill) and MG (cover guard) are
V9 classes that do not occur in V10.

| Class | Slots | Cases | Guard-stopped |
| --- | ---: | --- | ---: |
| RG | 9 | CG08506 C/H, CG4BDD H, CG4FAF C/H, CGEBA C/H, CGF41 C/H | 8 (all but CG4BDD H) |
| EX | 2 | CG64C C/H | 2 |
| NF | 2 | CG191 C/H | 2 |
| MR | 3 | CG1E0 C/H, CG4BDD C | 3 |
| PF | 4 | CG65C C/H, CG84C C/H | 0 |
| MW | 2 | CGE4B C/H | 0 |

Dominant causes: 11 slots with no supported context (RG 9 + EX 2), 5 slots
with a supported context that produced no measurement support (NF 2, MR 3),
6 slots that never produced a candidate (PF 4, MW 2). Predictions: 0 in all
22. 10 of 11 case pairs share the same class across arms; the exception is
CG4BDD (C MR versus H RG), the same single differing pair as in V9.

Allowance ledgers over the 22 slots (`candidate_disposition_ledger_v1`,
partition verified: selected 74 = completed 31 + expired 15 + omitted 28 +
failed 0 + not_attempted 0): every EXPIRED candidate carries
`EXPIRED_BY_CPU_GUARD` and every OMITTED candidate `OMITTED_BY_CPU_GUARD`;
`candidate_omission_status` `ALLOWANCE_OMISSIONS_RETAINED_UNKNOWN`,
`candidate_set_scope` `EVALUATED_ALLOWANCE_SUBSET_NOT_WHOLE_FINITE_SET`;
`evaluated_subset_complete` true in the 16 slots with candidates and false in
the 6 without; strict `candidate_evaluation_complete` true only for CG4BDD H
(the one slot whose guard did not trip). Before refinement the deterministic
lane quotas (main 12, completion 4 of the 48-group schedule) omitted a
further 27 passing start groups (CG191 C 8 and H 14 completion groups, CG4FAF
C 5 main groups; `omitted_passing_start_group_count`). Cost receipts:
`cpu_guard_tripped` true in 15 slots (`lane_stopped` completion:CPU_GUARD in
11, main:CPU_GUARD in 4), `process_cpu_seconds_now` at publication
1501.0-1552.9 s in those 15, 664.7 s in CG4BDD H and 12.4-57.7 s in the six
no-candidate slots; lane CPU totals completion 15798.64 s, main 6846.26 s;
per-candidate refinement cost 193.96-1422.29 s CPU for the 31 completed
candidates (sum 16964.55 s) and 29.46-1512.91 s for the 15 expired ones (sum
5660.54 s); every completed candidate reused the diagnosis full-scan query
exactly once (`reused_query_calls` 1,
`REUSED_DIAGNOSIS_FULL_SCAN_QUERY_EXACT_BINDING`; `query_resources
.reused_query_calls` = completed candidates per slot); `chunked_query_calls`
0 in all 22; initialization `cost_receipts.cpu_guard_tripped` false in all
22 (the guard tripped only inside registration refinement);
`completion_lane.refinement_schedule.lane_transaction` null in all 22 (no
rollback occurred).

Chain details (both arms identical unless stated; C = control, H =
challenger; p95 = surface `inlier_plane_p95_mm`; cov = CAD coverage / scan
coverage lower):

- RG, CG08506 C/H (family 10839000): native PARTIAL, 10 features bridged; main
  lane 0 passing of 4500 hypotheses; completion lane 3 parents, 132
  scheduled, 4 passing, 204 normal-incompatible; 4 selected (0 main + 4
  completion); 3 completed and rejected at p95 1.3136 / 1.3092 / 1.3149 mm,
  cov 0.667-0.669 / 0.747-0.749 (`INSUFFICIENT_FINITE_SURFACE_SUPPORT` +
  `SURFACE_RESIDUAL_OUTSIDE_DECLARED_ALGORITHM_GATE`, context
  `CONTEXT_PROXIMITY_RESIDUAL_OUTSIDE_GATE`), the fourth EXPIRED (225.19 s C,
  176.74 s H); guard tripped (completion lane 1463.5 / 1432.7 s CPU).
- NF, CG191 C/H (49001000): native UNAVAILABLE `INVALID_NATIVE_SOLID` (bridge
  0 features, no refusal); main lane 5 passing of 2876 hypotheses, completion
  lane 7 / 12 parents, 35 / 60 scheduled, 12 / 18 passing; quota omitted 8 /
  14 completion groups; 9 selected (5 main + 4 completion); 2 completed: one
  SUPPORTED_CONTEXT at p95 0.3637 mm (cov 0.794 / 0.908; C candidate 0, H
  candidate 1) and one rejected at 1.2412 / 1.2411 mm; 1 EXPIRED (223.16 /
  204.55 s), 6 OMITTED; guard tripped (main lane 1470.9 / 1440.6 s);
  registration `CONDITIONAL_MEASUREMENT_CONTEXTS`, one context scheduled
  (cap 8); the workflow then refused at phase `NATIVE_FEATURES`
  (`NATIVE_FEATURES_UNAVAILABLE`): nothing to measure.
- MR, CG1E0 C/H (38172000): native PARTIAL, 13 features bridged; main lane 1
  passing of 4876; completion lane 3 parents, 96 scheduled, 2 passing; 3
  selected; candidate 0 (main) SUPPORTED at p95 0.8480 mm (cov 0.981 /
  0.984), candidate 1 (completion) rejected at 0.6492 mm with cov 0.457 /
  0.451 (`INSUFFICIENT_FINITE_SURFACE_SUPPORT` /
  `INSUFFICIENT_CONDITIONAL_CONTEXT_PROXIMITY`), candidate 2 EXPIRED (152.88
  / 124.68 s); guard tripped (completion 1187.4 / 1160.3 s after main 277.4
  / 281.0 s); one context scheduled (cap 6, pose_class_rows 4448);
  measurement (`CAD_GUIDED_SUPPORT_COVER_MEASUREMENTS_V4`,
  `computation_status` COMPLETE, source context result
  `CAD_GUIDED_GENERATED_CONTEXT_MEASUREMENTS_V3`): 13 planned = 13 evaluated
  windows, all UNAVAILABLE at `failure_phase` `PLANE_FIT_ROLE_0` (8
  `FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY`, 5
  `FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL`); 0 support records, 0 support
  groups; workflow reason `NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS`.
  The two arms' measurement records are identical (same supported candidate).
- MR, CG4BDD C (37361005): native PARTIAL, 17 features bridged; main lane 1
  passing of 4938; completion lane 3 parents, 64 scheduled, 3 passing, 140
  normal-incompatible; 4 selected (1 + 3); candidate 0 (main) SUPPORTED at
  p95 0.6658972409525935 mm (cov 0.991 / 0.988; V9 admitted the same
  candidate at 0.6658972409525943 mm), candidates 1-2 (completion) rejected
  at 0.7380 / 0.7205 mm with cov 0.567 / 0.571 and 0.561 / 0.564
  (insufficient finite support), candidate 3 EXPIRED (207.14 s; V9 had
  refined it at 1.306 mm); guard tripped (completion 1280.8 s after main
  194.0 s); one context scheduled (cap 6, pose_class_rows 4183, as V8/V9);
  measurement: 17 planned = 17 evaluated windows, all UNAVAILABLE at
  `PLANE_FIT_ROLE_0` (16 WEAK, 1 NOT_PLANAR); 0 support records.
- RG, CG4BDD H: native PARTIAL 17; main 1 passing of 4928, completion 1
  parent, 18 scheduled, 2 passing; 3 selected, all completed and rejected at
  the gate: 1.195863337229378 / 1.3183 / 1.3376 mm with cov 0.98 / 0.99 (V9:
  1.1958633372293805 / 1.3183 / 1.3376); guard not tripped (664.7 s); the
  only slot with strict whole-set completeness.
- RG, CG4FAF C/H (49024000): native UNAVAILABLE `INVALID_NATIVE_SOLID`; main
  lane 17 / 11 passing (C: 5 omitted by the main quota of 12), completion
  lane 0 parents (C) and 1 parent / 14 scheduled / 0 passing (H); 12 / 11
  selected; 4 / 3 completed and rejected at the gate (C 1.4644 / 1.4116 /
  1.3280 / 1.4644 mm, cov 0.84-0.89 / 0.79-0.93; H 1.5501 / 1.5494 / 1.4503
  mm, the last also insufficient support at cov 0.586 / 0.530); 1 EXPIRED
  (78.53 / 224.53 s), 7 OMITTED; guard tripped (main 1490.0 / 1489.4 s).
- EX, CG64C C/H (40500000): native UNAVAILABLE `INVALID_NATIVE_SOLID` (as V8
  reported for this case; V9 unknown); main lane 0 passing of 4454;
  completion lane 1 parent, 51 scheduled, 2 passing; 2 selected; candidate 0
  EXPIRED after 1503.02 / 1512.91 s (33 query calls, 83535 rows attempted),
  candidate 1 OMITTED; 0 completed; guard tripped (completion 1504.0 /
  1513.9 s). Nothing beyond the completion prefilter is known for this case.
- PF, CG65C C/H (38172000): 0 candidates; main-lane prefilter 2045
  hypotheses / 5935 assigned supports / 5930 bound-excluded (645303 rows) / 5
  completed finite calls (533 rows) / 0 passing - equal to V9 and V8 to the
  unit; completion lane 0 parents (no two-of-three parent, as the protocol
  anticipated); native PARTIAL 13 (supported 7 / unresolved 11 candidates).
- PF, CG84C C/H (10839000): 0 candidates; prefilter 4522 / 12342 / 12127
  (1484752) / 215 (23418) / 0 - equal to V9 and V8; completion lane 1 parent,
  44 scheduled, 0 passing; native PARTIAL 10.
- MW, CGE4B C/H (10839000): initialization UNAVAILABLE
  (`NOISY_SCAN_PLANE_PROPOSALS_UNAVAILABLE` C,
  `COMPOSITE_SCAN_PLANE_PROPOSALS_UNAVAILABLE` H), all counters zero,
  completion lane `NOT_ATTEMPTED`, no checkpoint; unchanged since V8.
- RG, CGEBA C/H (38172000): native PARTIAL 13; main 0 passing of 4862;
  completion lane 3 parents, 80 scheduled, 2 passing; 2 selected; candidate 0
  completed and rejected at p95 0.6025 mm (inside the gate) with cov 0.602 /
  0.599 (`INSUFFICIENT_FINITE_SURFACE_SUPPORT`, 1422.29 / 1417.02 s CPU),
  candidate 1 EXPIRED (57.02 / 29.46 s); guard tripped.
- RG, CGF41 C/H (38172000): native PARTIAL 13; main 0 passing of 4272;
  completion lane 5 parents, 152 scheduled, 2 passing; 2 selected; candidate
  0 rejected at 0.7111 mm with cov 0.575 / 0.568 (984.82 / 984.97 s),
  candidate 1 EXPIRED (479.15 / 461.58 s); guard tripped.

Summary of the 31 completed candidates: 5 SUPPORTED_CONTEXT (p95 0.3637,
0.3637, 0.8480, 0.8480, 0.6659 mm) and 26 REJECTED_CONTEXT: 18 by the 1.0 mm
gate (p95 1.196-1.550 mm; 7 of them also insufficient finite support) and 8 by
finite support only (p95 0.602-0.738 mm inside the gate, CAD coverage
0.457-0.602 and scan coverage lower 0.451-0.599 against the frozen
`min_scan_coverage` 0.75). Main-lane prefilter counters and completion-lane
counts reproduce V9 exactly on the four comparable cases (CG65C, CG84C, CG4BDD
C/H), and the CG4BDD residuals agree with V9 to 13-15 significant digits.

Native features and bridge provenance (22 slots): `native_feature_bridge`
present in all 22 with `implementation`
`NATIVE_FEATURE_BRIDGE_V1_EXACT_KERNEL_V1_TO_KERNEL_V2_LOSSLESS_FIELD_RECONSTRUCTION`,
`source_class` `backend.cad_guided_measurement.CADGuidedFeature`,
`target_class` `backend.cad_guided_measurement_v2.CADGuidedFeature`,
`truncation` `NEVER_OVER_LIMIT_SETS_REFUSE_WHOLE`, `native_metadata_retained_unchanged`
true, no refusal in any slot; `feature_count` equals the native feature count:
10 (CG08506, CG84C, CGE4B), 13 (CG1E0, CG65C, CGEBA, CGF41), 17 (CG4BDD), 0
with `native_status` UNAVAILABLE (CG191, CG4FAF, CG64C; reason
`INVALID_NATIVE_SOLID`). The derived `measurement_context_cap.value` is 7
(10 features), 6 (13 and 17 features) and 8 (UNAVAILABLE slots, feature_count
1).

Measurement-window diagnostics (43 windows, the first per-window evidence of
this attempt line): every window records `failure_phase` `PLANE_FIT_ROLE_0`
with both flanges' layer-selection diagnostics retained. Layer identities:
CG1E0 (13 windows, each arm) role 0 `SINGLE_LAYER_IDENTITY_UNRESOLVED` 9,
`TWO_LAYER_INNER_SELECTED_BY_MATERIAL_SIDE_UNVERIFIED` 3,
`TWO_GROUPS_INCOMPATIBLE_IDENTITY_UNRESOLVED` 1; role 1 two-layer 9, single 3,
incompatible 1; CG4BDD C (17 windows) role 0 single 16, two-layer 1; role 1
two-layer 12, single 5. Flange fit status after selection: role 0 WEAK 8 /
NOT_PLANAR 5 (CG1E0, each arm) and WEAK 16 / NOT_PLANAR 1 (CG4BDD C) - no
role-0 fit passed; role 1 PASS 1 / WEAK 4 / NOT_PLANAR 8 (CG1E0, each arm) and
PASS 11 / WEAK 5 / NOT_PLANAR 1 (CG4BDD C). Arc, tangency and support gates
were never reached; no carrier exists; every identity string is "unverified"
or "unresolved", consistent with the declared empty inner-face eligibility.
These are the frozen kernel's retained outputs; their physical meaning was not
assessed here.

Frozen V17 order (read-only, `source/backend/cad_guided_mesh_offline_v17.py`):
native extraction at phase `NATIVE_FEATURES` (line 581), the bridge at
`NATIVE_FEATURE_BRIDGE` (588-596; refuses only on class / duplicate id / limit
/ malformed faults, so an UNAVAILABLE factory result passes through with zero
features), CAD sampling, `WHOLE_PART_REGISTRATION` (624), and only then the
availability test `features["status"] not in ("AVAILABLE", "PARTIAL") or not
features["features"]` -> `NATIVE_FEATURES_UNAVAILABLE` (682-684) before
`FULL_RAW_LOCAL_MEASUREMENT` (685). The records agree: the six
`INVALID_NATIVE_SOLID` slots ran registration to the guard (worker CPU
1513.90 / 1513.39, 1502.21 / 1502.13, 1531.79 / 1556.23 s; sum 9119.65 s =
38.6 percent of the campaign's worker CPU) and CG191's supported contexts,
the lowest residuals of the campaign, could not be measured by construction.
This is stated from the records and the frozen text, not from a runtime
observation.

## Check 7 - standing constraints

- No Gold in inference: the frozen runner gives workers only a dispatch id
  and token; the issued leaf record binds arm plus scan/CAD hashes; the only
  catalog reader `selected_references()` (runner line 1847) is called by
  `score_after_terminal` (line 2003) after `INFERENCE_TERMINAL.json`;
  `SELECTED_REFERENCES.json` was written 62.514 s after
  `INFERENCE_TERMINAL.json` and after all 22 terminals, and no case file
  changed afterwards. This rests on frozen-source reading and ordering
  evidence, not on a runtime observation of file accesses.
- All-22 denominator: 22 case directories (11 x 2), 22 durable terminal
  records, 22 projections, 22 dispositions in both arms of SCORED; refused
  cases keep every reference as a false negative.
- Unchanged 71 references: snapshot digest `d16390b2...b45dc` equal to V9's
  and to the frozen declaration; 71 lines, same per-case counts and frames.
- Both fresh arms: 22 fresh inference workers (23 distinct pids and caches
  with the scorer), no cached control loaded (`retained_control_policy`
  `NOT_ASSUMED_OR_LOADED_BY_THIS_RUNNER`).
- One heavy job: serial leaves; `OMP_NUM_THREADS=1` in the authorized command;
  fresh no-overlap check recorded at launch; `ps` quiescent at audit time.
- Limits unchanged from V9: 19 LIMITS values identical to the V9 POST
  bindings and to the protocol table; the allowance allocates work inside the
  1860 s cap (guard 1500 s) and raised no cap; no CPU or wall termination
  occurred and none was extended.
- No gate change after launch, no resume/retry: single `STARTED.json` per
  slot, 22 one-use dispatch ids each claimed once, every `STARTED.limits`
  equal to the frozen limits, `no_resume` true everywhere, `FROZEN.json` and
  `PRE_RUN_AUDIT.json` unchanged (check 2), final source audit unchanged.
- Historical comparability `UNESTABLISHED` in FROZEN, SCORER_RESULT and
  SCORED; `historical_score_beaten_claim_allowed` false;
  `production_use_allowed` false everywhere; `release_allowed` false on the
  runner's stdout; `FINAL_GATE.status` `PENDING_INDEPENDENT_POST_AUDIT`.
- No reopening of V9 or earlier attempts by this audit (V9 figures from the
  released V9 POST, its addendum and the V9 result narrative only); no raw,
  Gold or cache access by this audit.

## Paired comparison with V9

Same eleven inputs, same 71-line reference snapshot (`d16390b2...b45dc` in
both), same matcher; both campaigns TP 0 / FP 0 / FN 71 in both arms,
precision and F1 null, recall 0. The paired score difference V10 - V9 is zero
on every defined metric and null on precision/F1. The immediate causes moved
without any slot reaching an emitted support:

| V9 cause (released POST) | V9 slots | V10 cause | V10 slots |
| --- | ---: | --- | ---: |
| CPU kill inside refinement | 14 | RG (guard-stopped, none supported) | 8 |
| | | EX (guard-stopped, none completed) | 2 |
| | | NF (supported, native unavailable) | 2 |
| | | MR (supported, all windows refused) | 2 |
| Cover feature-guard refusal (MG) | 1 | MR | 1 |
| Residual rejection (RG) | 1 | RG | 1 |
| No passing start (PF) | 4 | PF | 4 |
| Membership cap (MW) | 2 | MW | 2 |

Per slot (V9 cause and leaf wall from the released V9 POST -> V10 cause and
leaf wall): CG08506 C CPU 1861.374 -> RG 1510.066, H CPU 1861.049 -> RG
1513.074; CG191 C CPU 1861.280 -> NF 1515.216, H CPU 1861.307 -> NF 1514.714;
CG1E0 C CPU 1860.992 -> MR 1510.468, H CPU 1861.631 -> MR 1522.089; CG4BDD C
MG 1626.818 -> MR 1513.467, H RG 702.230 -> RG 669.882; CG4FAF C CPU 1861.798
-> RG 1503.479, H CPU 1861.791 -> RG 1503.400; CG64C C CPU 1862.024 -> EX
1533.147, H CPU 1862.027 -> EX 1557.612; CG65C C PF 30.827 -> PF 30.885, H PF
47.060 -> PF 47.472; CG84C C PF 42.823 -> PF 43.090, H PF 63.654 -> PF 63.612;
CGE4B C MW 12.502 -> MW 12.473, H MW 12.750 -> MW 12.673; CGEBA C CPU 1861.891
-> RG 1517.257, H CPU 1861.901 -> RG 1509.166; CGF41 C CPU 1861.911 -> RG
1508.462, H CPU 1861.970 -> RG 1521.276.

Observations (combined-change attribution only, from the records):

- CPU kills 14 -> 0; slots reaching measurement windows 0 -> 3 (43 windows;
  V9 had none because CG4BDD C was refused by the cover grammar and CG1E0 was
  killed); slots with predictions 0 -> 0. The 14 V9 kills became 8 RG, 2 EX, 2
  NF and 2 MR under the guard: the kill sites of V9 (surface-V4 competitive
  pairing inside refinement) are the same frames the V10 samplers show at the
  guard stop (`_paired_candidates` / `leaf` / `walk` /
  `query_closest_surface`), but V10 published what had been refined before the
  stop and marked the rest EXPIRED/OMITTED.
- Worker CPU: V10 total over all 22 returned workers 23611.428 s (a campaign
  total exists for the first time since V8); V9 had 2535.780 s over its 8
  returned workers and no final value for the 14 killed ones (no V9 total).
  Paired over the 8 slots returned in both campaigns (CG4BDD, CG65C, CG84C,
  CGE4B, both arms): V9 2535.780 s -> V10 2390.661 s; per slot CG4BDD C
  1625.392 -> 1512.109 (-113.283, the fourth candidate expired instead of
  being refined), H 701.538 -> 669.159 (-32.379), CG65C C 30.687 -> 30.720, H
  46.913 -> 47.315, CG84C C 42.684 -> 42.942, H 63.487 -> 63.442, CGE4B C
  12.417 -> 12.387, H 12.662 -> 12.586. Leaf wall sum 28601.609 s -> 23632.980
  s. None of this is a speed-up claim: the guard-bounded walls are a declared
  allocation inside the unchanged caps, and the omitted work is unknown.
- The main-lane prefilter, the completion lane and the CG4BDD residuals
  reproduce V9 (and V8) on the four comparable cases; the differing last
  digits of the CG4BDD residuals (0.6658972409525935 vs ...943; 1.195863337229378
  vs ...3805) are consistent with a changed summation path in the reused /
  chunk-capable query, not with a different candidate.
- Arm comparison: 10 of 11 case pairs share the same class; CG4BDD differs (C
  MR versus H RG) exactly as in V9 (C MG versus H RG). CG191's arms differ
  only in which candidate index was supported. Challenger walls exceed
  control walls in 5 of the 8 non-guard-limited comparisons (CG65C, CG84C,
  CGE4B pairs and CG4BDD H shorter than C because nothing was measured);
  guard-stop points differ between arms by timing. Nothing here separates the
  declared V10 changes from one another.

## Findings

Ranked most severe first; each labelled for the acceptance of the custody and
ledger. Blocking: none.

1. Non-blocking. No numeric F1; second consecutive zero-prediction terminal
   ledger. Both arms TP 0 / FP 0 / FN 71, precision null, recall 0.0, F1
   null, minimum-family recall 0.0, emitted supports 0; all 11 cases REFUSED
   in both arms; null preserved everywhere. This is the frozen scorer's own
   output under the frozen failure policy, not a custody or ledger
   inconsistency; the historical 0.4580152672 is non-comparable and not
   beaten.
2. Non-blocking. The measurement side was exercised on physical data for the
   first time and refused every window. Three slots (CG1E0 C/H, CG4BDD C)
   evaluated 43 windows through bridge V1 (17/13/13 features reconstructed;
   no bridge refusal anywhere): 43 UNAVAILABLE at `PLANE_FIT_ROLE_0` (32
   `FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY`, 11
   `FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL`), 0 support records, 0 carriers;
   identity counts 0 over zero carriers in those slots, null elsewhere. The
   V9 MG defect did not recur. Retained negative outcome.
3. Non-blocking. Allowance and guard: 0 CPU kills (V9 14); guard tripped in
   15 leaves at process CPU 1501.0-1552.9 s; ledgers over 22 slots selected
   74 = completed 31 + expired 15 + omitted 28 (+ failed 0, not_attempted 0),
   all by the CPU guard; 27 further start groups omitted deterministically by
   the lane quotas; `evaluated_subset_complete` true in 16 / false in the 6
   candidate-less slots; strict completeness true only for CG4BDD H.
   Expired/omitted candidates are unknown, never rejected; guard stops are
   timing dependent and differ between arms.
4. Non-blocking. Frozen V17 order spends the registration allowance before the
   native-availability check: in the six `INVALID_NATIVE_SOLID` slots (CG191,
   CG4FAF, CG64C, both arms) registration ran to the guard (9119.65 s worker
   CPU, 38.6 percent of the campaign) although the workflow refuses at
   `NATIVE_FEATURES_UNAVAILABLE` (line 682-684) only after registration
   (line 624); CG191's supported contexts (p95 0.3637 mm, the best in the
   campaign) were unmeasurable by construction. Observation for a successor;
   no change to this terminal attempt is requested.
5. Non-blocking. The author's progress-ledger entry of 23:15 UTC differs from
   the records on three counts: "guard tripped in 16 slots" (records 15),
   "16 expired" (records 15), "p95 1.20-1.62 mm" (records 1.196-1.550 mm over
   the 18 residual-gate rejections; 1.62 occurs in no candidate's
   `inlier_plane_p95_mm`; nested context-proximity residual fields were not
   extracted). The author's separate result narrative was not read. The
   scoring ledger is unaffected; informational for the coordinator.
6. Non-blocking. Registration outcomes: 5 supported contexts of 31 completed
   candidates; 18 rejections by the 1.0 mm gate (p95 1.196-1.550 mm), 8 by
   finite support only (p95 0.602-0.738 mm inside the gate, CAD coverage
   0.457-0.602, scan coverage lower 0.451-0.599); every completed candidate
   reused the diagnosis query exactly once; no chunked query was needed;
   per-candidate refinement cost 193.96-1422.29 s CPU. Completed-residual
   policy rejections are not physical nonconformance and not capacity
   refusals.
7. Non-blocking. Not exercised: pose-class dedup and the context cap
   (supported = scheduled = 1 in the 5 supported slots; caps 6/8);
   completion-lane rollback (`lane_transaction` null in all 22; the
   initialization guard never tripped); the checkpoint cap notice (no ledger
   truncated); completion candidates passing on CG4FAF (0 of 0 and 0 of 14)
   or a parent on CG65 (none, as anticipated); CGE4B never reached the lane
   (unchanged since V8).
8. Non-blocking. Custody cosmetics: 95 of the 96 frozen sources are untracked
   in git (`??`; one tracked-clean), as are the attempt directory, the archive
   directory, the PRE report, the protocol, the runner and the waiter;
   custody rests on the hash-verified snapshot and archive (96/96 three ways,
   97/97 member payloads). `PRE_RUN_AUDIT.json` is not canonical JSON (both
   digests reconcile). Outer records are mode 0644, attempt controls 0600.
   Coordinator cache opaque; leaf caches declared removed.
9. Non-blocking. Coordinator supervision counters observed, not interpreted:
   admitted 24 / verified-absence retirements 23 / events discarded 7 /
   tracked peak 2; 4 exit-transition entries resolved as absence within 11
   ms; `requires_process_attention` false.
10. Non-blocking. Resource envelopes: every per-slot, scorer, coordinator,
    parent, log, result, control and report figure below its frozen cap
    (check 4); campaign 23853.5 s + 2400 s reserve = 26253.5 s below 104400
    s. Sampled values are not continuous guarantees.
11. Non-blocking. Process disclosure of this audit: four scratch files in the
    output directory outside the repository (listed below); no repository
    write; no code execution.

## What was not checked

- The outer exit observation itself: the waiting parent's record was verified
  for internal consistency (nonce, pids, timestamps, walls and the frozen
  runner's return path), not observed by this reviewer; no crash-proof
  guarantee is claimed by the waiter and none here.
- No runner, projection (`gold11_schedule_projection_v5`), scorer
  (`gold11_contact_scoring_v4`), engine, test, freeze, verify, execute or
  worker code was executed and no module under `backend/` or `scripts/` was
  imported: inner projections were bound by canonical-hash identity with the
  retained `contact_projection` objects, the matcher was not rerun, and the
  scoring request, both inventory digests and the terminal outcomes digest
  were reproduced by data transformation of the retained records.
- Raw scan/CAD bytes were not read or hashed; the Gold catalog was not
  opened; input binding rests on the frozen declarations, the receipts'
  `source_binding_valid` and the RESULT-embedded scan/CAD digests; the
  reference snapshot was hashed and counted only.
- Private caches under `/private/tmp` were not stat'ed or listed; two `ps`
  queries were run (none matching); no live-writer check beyond that.
- The V9 attempt directory was not opened; V9 figures are from the released
  V9 POST, its addendum and the V9 result narrative only.
- Encoded `patch_evidence.scan` wires, `hypotheses`,
  `completion_lane.hypotheses`, `face_index_evidence`,
  `unsigned_assignments`, two-plane context/schedule arrays and the
  per-window flange fit values were not decoded beyond counts, statuses,
  reasons and layer-selection identities; nested context-proximity residual
  fields were not extracted; no kernel, cover, context, bridge, registration
  or initializer module was reviewed line by line (their frozen hashes were
  verified; only the V17 lines and the runner inventory / scoring-request
  text cited above were read).
- Sampler policy conformance was not verified beyond counting records and
  reading the last frame of each slot.
- The author's V10 result narrative and the Codex registration postmortem
  written during this audit were not read; the coordination and diagnosis
  documents were only grepped for POST expectations.
- Whether the per-window refusals are correct physical statements was not
  assessed; they are the frozen kernel's retained outputs.

## Outputs of this audit

- `/Users/idantal/.claude/jobs/1519fdc2/tmp/post_v10/POST_RUN_AUDIT.json`:
  280901 bytes, SHA-256
  `ddabac9ee54474bff0d2842e33cfc4c342905b9bba4368491d06bfa1dd180c05`
  (schema `GOLD11_V10_INDEPENDENT_POST_RUN_AUDIT_V1`, decision
  `ACCEPT_WITH_FINDINGS`, 22 slot rows, exact per-arm scorer numbers with
  nulls preserved, all bindings, resources, taxonomy, paired comparison,
  findings and not-checked lists).
- This report: `/Users/idantal/.claude/jobs/1519fdc2/tmp/post_v10/2026-09-13_gold11_mesh_development_v10_post_run_audit.md`
  (its own hash and size cannot be embedded; they are printed in the
  reviewer's reply to the coordinator).
- Scratch files (outside the repository, disclosed; sizes and hashes in
  `POST_RUN_AUDIT.json.scratch_files`): `scratch_tar_tvf.txt` (the `tar -tvf`
  listing), `scratch_tar_member_hashes.txt` (97 streamed member hashes),
  `scratch_result_hashes.txt` (22 RESULT.json hashes and sizes),
  `scratch_slot_ledgers.json` (the per-slot ledger extraction from the 22
  RESULT.json files).
