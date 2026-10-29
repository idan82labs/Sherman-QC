# Gold11 full11 V8 - independent physical POST (execution custody and scoring ledger)

Date: 2026-09-13. Reviewer: independent AI post-run auditor (Claude Fable 5.1,
reviewer role; not the author of the candidate, the freeze, the PRE or the run).
Attempt audited: `docs/research/gold11_mesh_development_v8_attempt` (terminal).
This is an engineering audit of execution custody and of the retained negative
scoring ledger. It is not manufacturing approval, calibrated metrology,
conformance, generalization or production evidence. Every V8 outcome below is
a COMBINED consumed-development change relative to V7 (new extrema V3 / context
V4, surface V4 / registration V4, declared capacities 512M coordinates and 256
ties, the provisional 1.0 mm pose-admission gate and the new measurement-context
schedule acting together). Nothing here is a mechanism-only speedup or a
calibrated accuracy improvement, the in-run arm comparison is scan-plane lane
and scheduler only on the same V15 engine (not old engine versus new engine),
and the historical F1 0.4580152672 is not comparable and is not beaten.

Method: read-only. Only `cat`/`sed`/`grep`/`shasum`/`wc`/`ls`/`diff`/`tar -tvf`
(listing only) and `.venv/bin/python` snippets that read, parse and hash files
under the attempt, the job tmp directory and the named repository documents.
No runner, projection, scorer, test or geometry code was executed. Not opened,
stat'ed, hashed or traversed: `data/**`, any `.ply`/`.step`, the Gold and
reference containers under `data/bend_corpus/annotations`, `/private/tmp/gold11-*`,
`/var/folders/*`, `.env`, credentials, and every other `docs/research/*_attempt`.
The attempt's own `SELECTED_REFERENCES.json` (released derived snapshot) was
read. V7 figures come only from the released documents
`docs/research/2026-09-12_gold11_mesh_development_v7_result.md` and the
scientific diagnosis it links. This file is the sole write.

## Decision

ACCEPT the execution custody of the terminal V8 campaign and its complete
negative scoring ledger, with the following limits:

1. The parent process exit status is UNOBSERVED. The launch was detached
   (`launch_v8_execute.py`, `start_new_session=True`, no waiting parent), so no
   shell exit code was captured. `FINAL_GATE.json` present plus
   `FINALIZATION_FAILED.json` absent are consistent with a zero return from
   `execute()`, but they do not prove the process exit status. The runner's own
   release condition recorded in `COMPLETED.json` (`release_requires` =
   `OBSERVED_PARENT_EXIT_ZERO_FINAL_GATE_NO_VETO_AND_EXACT_INDEPENDENT_POST_AUDIT`)
   is therefore NOT met by observation. This limitation cannot be repaired by a
   rerun: the attempt is terminal and must not be rerun, resumed or rescored.
2. Both arms emitted zero supports: TP 0 / FP 0 / FN 71, precision null,
   recall 0.0, F1 null (undefined, not zero), minimum-family recall 0.0. This
   is the unchanged scorer's own null, not a numeric result and not an
   improvement over V7's identical ledger.
3. Four workers were CPU-terminated (exit -24, no returned result); their
   internal stages are unknown beyond sampled stacks.
4. The coordinator private cache `/private/tmp/gold11-v8-coordinator-6ebyksl4`
   is retained opaque outside the attempt (declared by the runner; not
   inspected here).
5. Raw scan/CAD bytes and the Gold catalog were not re-read here; input binding
   rests on the frozen declarations and each receipt's `source_binding_valid`.

No administrative failure, denominator exclusion, retry, resumed start,
relaxed limit or edited frozen source was found (details in Findings).

## Bindings

Canonical digest = SHA-256 of `json.dumps(obj, sort_keys=True,
separators=(",", ":"), allow_nan=False).encode()` of the decoded object.
File SHA-256 values are of the raw bytes.

| Artifact (attempt root) | Bytes | File SHA-256 |
| --- | ---: | --- |
| FROZEN.json | 39670 | `03f58d23a3733434cfac6d51c031974088e660d0c880542ceb5c8182f3032961` |
| PRE_RUN_AUDIT.json | 20736 | `41fd61fc6157e44823477e04da034d4b378307055cb325e0bcb19f1d7c7d6554` |
| STARTED.json | 944 | `d6531597fa9a622a7291f722ad0d8c282b7f89b0477d015804a0582b5e39ed3f` |
| COORDINATOR_DISPATCH.json | 221 | `c4c0246b91d89e5c1f533c5c3e0ff36733b3b62d3d7db0df8da8882fd7ddf762` |
| COORDINATOR_VERIFIED.json | 9078 | `5d8e00c731f6f69e6fe2235776b57ea18dd63b87c36cf2e88286138d55c98886` |
| COORDINATOR_COMPLETED.json | 183 | `c40f3a94dab301a70534be6bb6abd31f0b081802f844a1c3c6e501400f2232e8` |
| COORDINATOR_OUTPUT.txt | 3983 | `4039b8c9c6895269eaab0c9359149bae9e20cfd7d7a749a7179224ed339715ca` |
| INFERENCE_TERMINAL.json | 1876 | `aa197fccc684abd2b7b17c95d6ba3fb6b09f4cd6445b9d52aced12bab6365c70` |
| SCORER_RECEIPT.json | 3209 | `4132d13fb0f5eb31322609a8e64a825b2bde1187dff9cb69c6f06ca029b49cff` |
| SCORER_RESULT.json | 35447 | `d20a950a9d955d3fe6e5df2f0726666996758c8ce971be15791f2d466f689292` |
| SCORER_OUTPUT.txt | 115 | `bbe0637056900782f804139aca197486749d7e18cd0709363ced7a182b09e727` |
| SCORER_ARTIFACT_RETENTION.json | 248 | `e1df067ece7226955d526147e170bee02cc52966240030eae47a5a62e7eda155` |
| SCORED.json | 146867 | `2538f9ba84ded29899f2089820124e218d645df167818651fcfa22fb8f1787ca` |
| SELECTED_REFERENCES.json | 24751 | `d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc` |
| COMPLETED.json | 203850 | `379a6be373e08dce8a9a0cde61783839ad6b79bb65b13c22dadf5e1aa03e0509` |
| FINAL_GATE.json | 1056 | `106907225365f23c6719f8ee166f256bc36be7c24f7709ef60f784a331b8d19f` |

Control custody against PRE (check 7): `PRE_RUN_AUDIT.json` in the attempt is
20736 B / `41fd61fc...d554`, byte-identical (`diff`) to the PRE copy in the job
tmp directory; `FROZEN.json` is 39670 B / `03f58d23...2961`. Both equal the
expected values. The external archive
`docs/research/gold11_mesh_development_v8_physical_frozen.x9er0oPw/SOURCE.tar`
hashes to `35cad3685658cfab5f1ab824c438c6284a0ba47e7970d4fa812164065bfae103`,
2839040 B (expected values). `tar -tvf` lists 81 entries: 6 directories and 75
file members = `FROZEN.json` (39670 B) plus exactly the 74 `source/<path>`
members; every member size equals the on-disk `source/` file size (75/75).
Nothing was extracted.

Canonical bindings:

- canonical(FROZEN) = `43173e8a9c59c95c188e17c0c2901b8067fecf349ec363c05ab024c261fa9326`
  = `STARTED.bundle_sha256` = `COORDINATOR_VERIFIED.bundle_sha256` =
  `PRE_RUN_AUDIT.bundle_sha256` = `SCORED.bundle_sha256`.
- canonical(PRE_RUN_AUDIT) = `b68eaab4a58d7222e731408a7b107edcb3885c9416603548d4871891fd7eca69`
  = `STARTED.audit_sha256` = `COORDINATOR_VERIFIED.audit_sha256`. PRE fields:
  `decision` = `GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN`, `reviewer` =
  `independent-ai-pre-auditor-v8 (...)`, `blocking_findings` = `[]`.
- canonical(STARTED) = `COORDINATOR_VERIFIED.started_sha256`;
  canonical(COORDINATOR_DISPATCH) = `COORDINATOR_VERIFIED.coordinator_dispatch_sha256`;
  `COORDINATOR_VERIFIED.source_sha256` == `FROZEN.source_sha256` and
  `runtime_metadata` equal (python 3.11.15 arm64 darwin, numpy 2.4.6, scipy
  1.17.1, cadquery-ocp 7.9.3.1.1; version metadata, not binary attestation).
- `STARTED.limits` == `FROZEN.limits`; all 22 case `STARTED.json` records carry
  the same limits, `no_resume=true` and the frozen arm request digest
  (control `b467981797a5c2f5303dcd462e1f0d116f4939fbfc18327b9fdf0cee05cf6565`,
  challenger `2243dc4775a0571cc757a75d20bd97db75bfead1116faea7fc60d004062aa52b`);
  canonical(all arm requests) = `837aed555b41d40a209d96f94c001fee97b7c889a93d5fe5a09af6212725989b`
  = `STARTED.arm_requests_sha256` = `INFERENCE_TERMINAL.arm_requests_sha256`.
- Process identity: `v8_execute_pid.txt` 37160 = `COORDINATOR_DISPATCH.parent_pid`
  = `COORDINATOR_VERIFIED.outer_pid`; coordinator pid 37162 in every leaf
  dispatch/claim record; scorer worker pid 42521 = `SCORER_RECEIPT.child_pid`.
- COMPLETED: `status` = `COMPUTED_PENDING_INDEPENDENT_POST_AUDIT`, `failures`
  = `[]`, `parent_wall_seconds` = 9512.744064625353, `parent_peak_rss_bytes` =
  1557233664, `no_resume` true, `production_use_allowed` false. canonical(COMPLETED)
  = `c20a3c4e5ee5c56e6cd2037a56e32c638b4a49b2faf58d06f010d6e1875dc632` =
  `FINAL_GATE.completed_canonical_sha256`. `final_source_audit`: complete true,
  unchanged true, runtime_metadata_unchanged true, errors {}, `current` ==
  `frozen` == `FROZEN.source_sha256` (74 entries), controls =
  {COORDINATOR_DISPATCH, COORDINATOR_VERIFIED, FROZEN, PRE_RUN_AUDIT, STARTED}
  file hashes equal to the table above. canonical(final_source_audit) =
  `22d043cd454c2829c59d2e8fd346c8824a8edcf434b0059f7d8599b31cee9447` =
  `FINAL_GATE.source_and_control_audit_sha256`; canonical(owned_artifact_custody)
  = `b910b1c24706b267f8b2187f3a4e684ad9415c5be381d944f575f536da81556e` =
  `FINAL_GATE.owned_artifact_custody_sha256`. FINAL_GATE `status` =
  `PENDING_INDEPENDENT_POST_AUDIT`, `scored_report_sha256` = SCORED file hash,
  `scorer_result_sha256` = SCORER_RESULT file hash, `reference_snapshot_sha256`
  = `d16390b2...b45dc`, `parent_wall_seconds` 9513.197944208048,
  `final_gate_self_hash_claim` false. `FINALIZATION_FAILED.json`,
  `SCORING_UNAVAILABLE.json`, `coordinator_work`, `RESULT_PARTIAL.bin` and
  `SCORER_RESULT_PARTIAL.bin` are absent.
- Source drift (check 2): all 74 `FROZEN.source_sha256` entries equal the
  `source/` copy and the working-tree file at `/Users/idantal/ShermanQC/FullProject`
  (0 mismatches; `source/` holds exactly those 74 files). The frozen V8 runner
  is 172872 B / `0e9aa2f64f3fc14e0da137687e44c8cd0dc0712f9e8bd783726df4187b1360c5`;
  the V7 basis `e910d562e55a62de80f5e448f944c239b5bdf95ef11c6d0b6f0b83b24bfc7a52`.
- Owned artifact custody: 115 declared files (1056305265 B) re-hashed here,
  0 mismatches; 24 logs all `COMPLETE_LOG_BYTES_BOUND`; campaign tree inventory
  complete, 281 entries (240 regular files + 41 directories: 115 scientific,
  74 source, 46 dispatch, 5 control), errors [], read_bytes 1058993503. All 281
  inventory fingerprints (device, inode, mode, links, bytes, mtime_ns,
  ctime_ns) equal the current `lstat` values; all 240 recorded file hashes
  equal current bytes; the only on-disk names outside that inventory are the
  two later publications `COMPLETED.json` and `FINAL_GATE.json` (283 total).
- Inventory reproduction: appending the current `COMPLETED.json` fingerprint,
  hash and bytes to the retained pre-COMPLETED inventory exactly as
  `verify_campaign_inventory` would (one more control slot, +203850 read bytes)
  reproduces `FINAL_GATE.completed_publication_inventory_sha256` =
  `87bca33c36d20ec2edb5483b1ffcd844dba68a7571a6b4afd10a04caf15bea5b`; adding
  `FINAL_GATE.json` (+1056 read bytes) reproduces the parent stdout's
  `final_campaign_inventory_sha256` =
  `dbeb824d1929e6f3b638cfc3cc1f8ed13899fc5a5da39a62d5fbc7a2912445bb` with 283
  entries / 115 scientific. The final on-disk identities of both control files
  are therefore the ones the runner itself inventoried.
- Terminal fill: all 22 `terminal_case_records` durable, every record
  `PREEXISTING_VALID_NO_WRITE` (no fallback terminal was written).
- Coordinator receipt (supervision V3): exit_code 0, budget_stop null,
  child_pid 37162, wall 9451.938938874751 s, direct_child_reaped true,
  owned_group_empty_confirmed true (group 37162), requires_process_attention
  false, foreign escape / unknown identity false, cleanup/control errors [],
  sampled tree peak 2549350400 B + parent 41959424 B = budget peak 2591260672 B
  (below 12 GiB), log 3983 B bound. `COORDINATOR_OUTPUT.txt` holds exactly the
  22 case lines in frozen order plus the final status line.
- Leaf dispatch: 46 files = 23 issued/claimed pairs (22 INFERENCE + 1 SCORER),
  ids equal canonical(role, binding), context_sha256 = canonical(COORDINATOR_VERIFIED),
  claim `dispatch_sha256` = canonical(issued record), `no_retry` true, worker
  pids equal the receipts' `child_pid`, private cache path/identity equal the
  receipts' custody records, 23 distinct caches (23 distinct inodes), all
  reported `REMOVED_AFTER_QUIESCENCE_AND_DERIVED_RETENTION`; the 18 clean
  inference ids are exactly `required_leaf_dispatch_ids`.
- INFERENCE_TERMINAL: case_count 11, arm_count 2, terminal_record_count 22,
  `outcomes_sha256` `b1d3c7bb2c8bde2709fdca88bb8deca6592ad4bee7711bd296db5be2df06b879`
  reproduced here from the 22 retained `SCORING_INPUT.json` objects; 18 result
  bindings equal the receipts and the actual RESULT.json hashes, 4 null;
  parent_wall 9392.117845084053 s (sum of the 22 per-case parent walls
  9392.080 s).
- Scoring transport: `SCORED.projection_bindings` equal the receipts on
  arm_request_sha256 / scoring_input_sha256 / result_sha256 / projection_kind
  for all 22; `scorer_input_canonical_sha256` equals canonical(SCORING_INPUT
  .contact_projection) for the 18 wrapped projections and canonical(refusal)
  for the 4 supervisor refusals. Rebuilding the V6 scoring request from those
  inner objects, the snapshot's `bends`, the frozen family map, the frozen arm
  requests and scan-plane requests gives canonical
  `26939e683c1493b677f48875892f843ed2cc6be705f6cbf89d625a77f04fd007`
  (103965 B) = `SCORED.scoring_request_sha256` = the scorer leaf binding
  `request_sha256`. `workflow_stage_attribution` equals the 22 receipts'
  `workflow_stage_diagnostics`. canonical(SELECTED_REFERENCES) = the file bytes
  = `d16390b2...b45dc`; 11 cases, 71 bends (6,5,10,4,3,7,6,8,4,8,10 in frozen
  case order), 9 `reference_mesh_raw` and 2 `scan_raw` frames, equal to
  `SCORED.case_declared_coordinate_frames`.

## All 22 slots table

Frozen case order, control (C) then challenger (H). `wall` = leaf receipt
`wall_seconds`; `pwall` = per-case parent wall; `cpu` = worker-reported
`worker_cpu_seconds` (returned workers) or the LAST SAMPLED process CPU (killed
workers, marked `~`, not a final value); `hwm` = worker `ru_maxrss` (returned)
or last sampled high-water RSS (`~`); `smp` = sampler records in OUTPUT.txt;
`log` = OUTPUT.txt bytes (= stored = forwarded); `RESULT` = retained RESULT.json
bytes and hash prefix. Every receipt has `status` TERMINAL, `budget_stop` null,
`source_binding_valid` true, `derived_result_retention_complete` true,
`private_cache_cleanup_complete` true, no cleanup/identity/control errors.

| Case | Family | Arm | exit | wall s | pwall s | cpu s | hwm B | smp | log B | RESULT |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CG08506AE97D | 10839000 | C | 0 | 43.818 | 50.716 | 43.126 | 1649704960 | 1 | 3219 | 70531657 `d2fb0f041f64` |
| CG08506AE97D | 10839000 | H | 0 | 63.043 | 70.072 | 62.863 | 1911668736 | 2 | 4223 | 71981171 `97121d001d37` |
| CG191DEC7F81 | 49001000 | C | -24 | 1861.348 | 1861.381 | ~1860.224 | ~1417363456 | 62 | 97537 | none |
| CG191DEC7F81 | 49001000 | H | -24 | 1861.316 | 1861.349 | ~1860.247 | ~1639366656 | 62 | 97312 | none |
| CG1E0BBA2933 | 38172000 | C | 0 | 356.501 | 362.940 | 356.198 | 1607548928 | 11 | 18832 | 77971784 `76e047ec8812` |
| CG1E0BBA2933 | 38172000 | H | 0 | 392.212 | 398.626 | 391.851 | 1831124992 | 13 | 21419 | 79329905 `d643b4f70666` |
| CG4BDD1371B6 | 37361005 | C | 0 | 253.856 | 259.450 | 253.608 | 1488928768 | 8 | 14262 | 68532131 `8307e8dd06ae` |
| CG4BDD1371B6 | 37361005 | H | 0 | 288.536 | 293.915 | 288.276 | 1627406336 | 9 | 15681 | 61612979 `4637dca7af0a` |
| CG4FAF273093 | 49024000 | C | -24 | 1860.930 | 1860.936 | ~1860.196 | ~513409024 | 62 | 97262 | none |
| CG4FAF273093 | 49024000 | H | -24 | 1860.923 | 1860.930 | ~1860.329 | ~539459584 | 62 | 97549 | none |
| CG64C412CC73 | 40500000 | C | 0 | 26.350 | 30.101 | 26.224 | 1056129024 | 0 | 1636 | 48547279 `264391688376` |
| CG64C412CC73 | 40500000 | H | 0 | 35.097 | 38.952 | 34.967 | 1129578496 | 1 | 2964 | 49345009 `ae66a3edf733` |
| CG65CD575B1C | 38172000 | C | 0 | 32.758 | 38.332 | 32.607 | 1688420352 | 1 | 2615 | 57296120 `ee8562a39bfb` |
| CG65CD575B1C | 38172000 | H | 0 | 50.446 | 55.984 | 50.297 | 1874624512 | 1 | 3255 | 58941238 `6dc5830c2a7b` |
| CG84C8FA12D2 | 10839000 | C | 0 | 45.253 | 52.068 | 45.105 | 1770733568 | 1 | 3227 | 74390034 `2f259f62f670` |
| CG84C8FA12D2 | 10839000 | H | 0 | 67.105 | 74.071 | 66.949 | 1957478400 | 2 | 4445 | 75563461 `bc68b9a602a2` |
| CGE4B0549E41 | 10839000 | C | 0 | 13.526 | 13.574 | 13.429 | 1696301056 | 0 | 1645 | 83784 `17dd6ceaa3fc` |
| CGE4B0549E41 | 10839000 | H | 0 | 13.581 | 13.634 | 13.486 | 1754906624 | 0 | 1639 | 88970 `9130dfa641d9` |
| CGEBA9A869AC | 38172000 | C | 0 | 32.564 | 38.123 | 32.422 | 1381679104 | 1 | 2607 | 62004334 `6b1facf555dd` |
| CGEBA9A869AC | 38172000 | H | 0 | 47.677 | 53.130 | 47.530 | 1470644224 | 1 | 3112 | 63311959 `c5c0015f88f7` |
| CGF41F49066A | 38172000 | C | 0 | 36.951 | 42.764 | 36.806 | 1538965504 | 1 | 2952 | 67202059 `caac54b5402f` |
| CGF41F49066A | 38172000 | H | 0 | 54.737 | 61.031 | 54.586 | 1696759808 | 1 | 3254 | 68499279 `34f9874bb9a3` |

Sums: leaf walls 9298.529 s; parent walls 9392.080 s; known worker CPU for the
18 returned workers 1850.329 s (no total including the killed workers is
formed). Largest returned CPU 391.851 s (CG1E0 H), largest returned worker
high-water RSS 1957478400 B (CG84 H), all below 1860 s / 8 GiB; all leaf walls
below 3540 s. Sampler records total 302 (188 in V7). Every OUTPUT.txt line is
JSON; the 18 returned workers each have exactly one `PUBLIC_WORKFLOW_CALL_BEGIN`
line, one stopped sampler receipt and one `INFERENCE_TERMINAL` line whose
`result_sha256` equals the receipt, the retention artifact, the actual file
hash, `INFERENCE_TERMINAL.result_bindings` and the projection's
`result_binding.supplied_sha256` (matched true, supplied bytes equal file size).

Per-slot RESULT.json checks (18 files): `cad_sha256` and
`scan_topology.source_sha256` equal the frozen case declaration;
`implementation` `CAD_GUIDED_MESH_OFFLINE_V15_CONTEXT_V4`,
`initialization_implementation` `CAD_GUIDED_PLANE_INITIALIZATION_V14_CONTEXT_V4`,
`registration_implementation` `SURFACE_V4_VERTEX_BOUND_TIES256_WIDTH_UNCERTAINTY_CONSUMER_V4`,
`two_plane_context_configuration_sha256`
`12900ddac835ccd99208019b77d4bcce3443ae3c5c9dd0ea82aba89ee5f5db66` in all 18.
The four declared deltas are present in every returned policy snapshot:
`policies.registration.max_inlier_plane_p95_mm` 1.0, initialization
`surface_budgets.max_ties_per_query` 256 (and registration
`surface_query_budgets.max_ties_per_query` 256 where registration ran),
context budget `max_extrema_coordinate_visits` 512000000 (16 slots that built a
context), `measurement_context_cap.policy_maximum` 8. The derived cap value
equals min(8, 4096 // features, 64000000 // (features x raw rows)) in all 18
(7, 6, 6, 8, 6, 7, 7, 7, 6 per case in frozen order).

### CPU-terminated slots: what exists and what is unknown

CG191DEC7F81 (both arms) and CG4FAF273093 (both arms). Records present:
`STARTED.json` (limits and arm request bound); `COMPLETED.json` receipt with
`exit_code` -24, `budget_stop` null, `failure_type` ValueError, refusal
`INFERENCE_WORKER_FAILED`, `inference_started` `UNKNOWN_UNTIL_RETURNED_RESULT`,
`retained_result_artifacts` `NO_RETURNED_RESULT_INTERNAL_STAGES_UNKNOWN` (no
artifacts), `derived_result_retention_complete` true, private cache removed,
`workflow_stage_diagnostics` `NO_RETURNED_WORKFLOW_INTERNAL_STAGES_UNKNOWN`,
walls 1861.348 / 1861.316 / 1860.930 / 1860.923 s (below the 3540 s wall cap);
`SCORING_INPUT.json` = the exact six-key supervisor refusal (159 B, same hash
`3a4dd85d...deea1` in all four); `OUTPUT.txt` (97537 / 97312 / 97262 / 97549 B)
with one `PUBLIC_WORKFLOW_CALL_BEGIN` line and 62 sampler records (indices
0-61, 30 s cadence) and NO sampler receipt and NO `INFERENCE_TERMINAL` line;
no `RESULT.json`, no `RESULT_PARTIAL.bin`. Exit -24 is the Darwin signal-style
return for SIGXCPU; the frozen worker sets `RLIMIT_CPU` to the frozen 1860 s,
and the last samples read CPU 1860.224 / 1860.247 / 1860.196 / 1860.329 s at
elapsed 1860.50 / 1860.50 / 1860.47 / 1860.59 s, which supports CPU-limit
attribution under the frozen (not relaxed) limit. Last sampled stack in all
four: triangle surface V4 `_paired_candidates` (line 354/355) <- `leaf` 534 <-
`walk` 603 <- `query_closest_surface` <- registration V4 `_Queries.__call__`
245 <- `_refine` 376 (three slots) or `_diagnose` 440 (CG4FAF H) <- V14
`register_whole_part_planes_v14` line 999 (fine refinement call) or 1002
(diagnosis call) <- V14 1072 <- V15 430. Across the 62 samples the V14 frame
sits at lines 997/999/1002/1004 (coarse refine / fine refine / diagnose /
context support) in 21/34/4/2, 22/33/3/2, 17/34/6/5 and 18/33/7/4 samples,
with one CG191 C sample at line 951 (initialization). Because line 997 follows
the start-group requirement at V14 line 961, at least one passing start group
existed in each killed run; nothing else is known: final CPU, candidate
counts, whether any context was supported, native-feature status and every
work counter are unknown for these four slots (V7 had `INVALID_NATIVE_SOLID`
for CG191 both arms and CG4FAF H; V8 cannot confirm or deny it).

## Immediate causes

Mutually exclusive over all 22 slots, from the retained RESULT ledgers
(workflow -> registration -> initialization -> two-plane context chain) and the
supervisor receipts. Codes: PF = completed finite prefilter with no passing
start; MW = scan-plane proposal triangle-membership refusal; MC = supported
context scheduled, measurement consumer refused every window; RG = completed
candidate rejected by the (1.0 mm) residual gate; CPU = CPU-terminated, no
returned result.

| Cause | Slots | Cases |
| --- | ---: | --- |
| PF | 12 | CG08506 C/H, CG64C C/H, CG65C C/H, CG84C C/H, CGEBA C/H, CGF41 C/H |
| MW | 2 | CGE4B C/H |
| MC | 3 | CG1E0 C/H, CG4BDD C |
| RG | 1 | CG4BDD H |
| CPU | 4 | CG191 C/H, CG4FAF C/H |

Chain details (both arms identical unless stated; all returned slots have
`status` UNAVAILABLE, `predictions` [], `observed_bend_count` null,
`conformance` INDETERMINATE):

- PF (12): workflow phase `WHOLE_PART_REGISTRATION`, reasons
  [`PROVISIONAL_REGISTRATION_UNAVAILABLE`, `NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS`];
  registration UNAVAILABLE with 0 candidates; initialization COMPLETE
  (search_complete, numerical_consumer_complete, box_screening_complete all
  true) with `exact_passing_start_group_count` 0; two-plane context V4 COMPLETE
  with all planned signed families completed. Per case: hypotheses evaluated /
  assigned supports / bound-excluded supports (rows) / completed finite calls
  (rows) / passing starts: CG08506 4500 / 12356 / 12159 (1280652) / 197 (20196)
  / 0; CG64C 4454 / 12358 / 12263 (1350961) / 95 (10317) / 0; CG65C 2045 / 5935
  / 5930 (645303) / 5 (533) / 0; CG84C 4522 / 12342 / 12127 (1484752) / 215
  (23418) / 0; CGEBA 4862 / 12572 / 12440 (1411542) / 132 (14242) / 0; CGF41
  4272 / 12288 / 12118 (1408908) / 170 (19292) / 0. Context V4 statistics
  (`extrema_coordinate_visits` / `extrema_queries` / `query_rows_evaluated` /
  signed families completed of planned): CG08506 89178513 / 219 / 7005730 /
  184 of 184; CG64C 52758560 / 252 / 4637800 / 160 of 160; CG65C 25099205 / 54 /
  272232 / 32 of 32; CG84C 79883302 / 174 / 5811800 / 192 of 192; CGEBA
  52584781 / 471 / 3659238 / 304 of 304; CGF41 27774069 / 135 / 439367 / 80 of
  80. No registration query resources exist (no start passed). CG64C both arms
  additionally report native `UNAVAILABLE` / `INVALID_NATIVE_SOLID` (0
  features); the other PF slots have native `PARTIAL` with no reasons.
- MW (2): CGE4B C reasons [`PROVISIONAL_REGISTRATION_UNAVAILABLE`,
  `NOISY_PLANE_INITIALIZATION_UNAVAILABLE`], initialization UNAVAILABLE
  `NOISY_SCAN_PLANE_PROPOSALS_UNAVAILABLE`; CGE4B H initialization
  `COMPOSITE_SCAN_PLANE_PROPOSALS_UNAVAILABLE`. The string
  `MAX_TRIANGLE_MEMBERSHIPS_EXCEEDED` occurs once in each RESULT and
  `COMPOSITE_V7_LANE_UNAVAILABLE` once in the challenger RESULT (string
  presence in the encoded `patch_evidence.scan` wire; the wire was not
  decoded here). All initialization/context counters are zero; no context was
  built (the 20M membership cap is unchanged from V7, as the delta design
  deferred).
- MC (3): initialization COMPLETE with exactly one passing start group;
  registration status `CONDITIONAL_MEASUREMENT_CONTEXTS`, 1 candidate
  attempted, coarse 30/30 and refine 30/30 iterations both `ITERATION_LIMIT`
  (not convergence-certified), `numerical_consumer_status`
  `COMPLETE_V3_INTERVAL_SUPPORT_POLICY`. CG1E0 C/H (hypothesis 745): surface
  p95 upper 0.8479920393527555 mm, scan coverage lower 0.9840254161446214,
  CAD coverage 0.9810791015625; `conditional_context_support` =
  `SUPPORTED_CONTEXT` with context p95 upper 0.9008134371607799 mm, coverage
  lower 0.9992393055306962, reverse coverage 0.9810791015625, 22331 of 22348
  rows; registration `query_resources` 63 calls / 216484 rows, 14841916
  certificate pairs, 3710479 triangle pairs, 7392 normal pairs, 100629414 work
  units; `measurement_context_schedule`: supported 1, scheduled 1 (index 0),
  `max_measurement_contexts` 6, pose_class_rows 4448, no duplicate or
  omitted disposition. CG4BDD C (hypothesis 4533): surface p95 upper
  0.6658972409525943 mm, coverage lower 0.9882774656074403, CAD coverage
  0.990753173828125; context SUPPORTED with p95 upper 0.6956471254551617 mm,
  coverage lower 0.9999031195504746, reverse 0.990753173828125, 20642 of 20644
  rows; query_resources 63 / 203331 rows, 17409008 certificate pairs,
  116522428 work units; schedule 1 of 1, cap 6, pose_class_rows 4183. All
  three then reached measurement: workflow phase `COMPLETE`, status
  UNAVAILABLE with reason `NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS`;
  `measurements.source_context_result` UNAVAILABLE
  `NO_NONCONFLICTING_CANONICAL_SCAN_MEASUREMENTS` with 1 attempted context and
  every planned window UNAVAILABLE: CG1E0 13 windows (11
  `FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY`, 2
  `FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL`), crop rows 9598732 attempted =
  completed, 243839 local input rows, 4 surface-role filters; CG4BDD C 17
  windows (17 `FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY`), crop rows
  9881471, 144910 local rows, 3 filters; 0 support records, cover plan
  COMPLETE with 0 components, 0 support groups, 0 predictions. The two window
  reasons are raised by the frozen `backend/cad_guided_measurement.py` plane
  fit (lines 125 and 133: a flange-support spread/residual policy against
  `max_fit_residual_mm` 0.05, and the planarity check). These are
  measurement-consumer window refusals, not the preflight-budget class the
  protocol anticipated: the strings `CONTEXT_WINDOW_PREFLIGHT_BUDGET` and
  `FULL_RAW_CROP_WORK_PREFLIGHT_BUDGET` occur in no RESULT.
- RG (1): CG4BDD H (hypothesis 4533) completes 30/30 + 30/30 but surface p95
  upper 1.1958633372293805 mm exceeds the 1.0 mm gate:
  `SURFACE_RESIDUAL_OUTSIDE_DECLARED_ALGORITHM_GATE`, context `REJECTED_CONTEXT`
  (`CONTEXT_PROXIMITY_RESIDUAL_OUTSIDE_GATE`, p95 upper 1.2009319036096364 mm,
  coverage lower 0.9997093586514241, reverse 0.983184814453125), registration
  UNAVAILABLE `NO_GEOMETRICALLY_SUPPORTED_MEASUREMENT_CONTEXTS`; query_resources
  63 / 203331 rows, 17664604 certificate pairs, 117422706 work units. This is a
  completed residual-policy rejection, not proof of physical nonconformance and
  not a capacity refusal.
- CPU (4): as described above.

Absent everywhere in the 18 RESULT files: `BUDGET_EXTREMA_COORDINATE_VISITS`,
`NEAR_TIE_BUDGET_EXCEEDED`, `UNSUPPORTED_COMPETITIVE_TRIANGLE_INTERVAL_WIDTH`,
`SUPPORTED_CONTEXT_DUPLICATE_POSE_CLASS`,
`SUPPORTED_CONTEXT_OMITTED_BY_MEASUREMENT_BUDGET`, `MEASUREMENT_CONTEXT_CAP`.
`INVALID_NATIVE_SOLID` occurs only in CG64C (both arms).

## Scores

`SCORED.json` schema `GOLD11_CONTACT_GEOMETRIC_DEVELOPMENT_COMPARISON_V2`,
status `COMPUTED_DEVELOPMENT_DIAGNOSTIC`, experiment_schema
`GOLD11_MESH_DEVELOPMENT_V8`, control identity
`FRESH_V15_CONTEXT_V4_SURFACE_V4_V7_SCAN_PLANES_SCHEDULE_V1_NATIVE_V1`,
challenger identity
`FRESH_V15_CONTEXT_V4_SURFACE_V4_COMPOSITE_SCAN_PLANES_SCHEDULE_V2_NATIVE_V1`,
comparator `FRESH_V15_CONTEXT_V4_SURFACE_V4_V7_SCHEDULE_V1_VS_COMPOSITE_SCHEDULE_V2_NATIVE_V1`,
wrapper schema `GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V3`, request
schema `GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_REQUEST_V6`, arm definitions with
initializer V14 / context v4 / registration V4 identities, `config_delta_from_v7`
(0.5 -> 1.0 mm, 32 -> 256 ties, 256M -> 512M coordinates, new
`workflow_max_measurement_contexts` 8), `result_interpretation`
`COMBINED_CONSUMED_DEVELOPMENT_CHANGES_NOT_MECHANISM_ONLY_SPEEDUP_OR_CALIBRATED_ACCURACY`,
`arm_comparison_scope` `SCAN_PLANE_LANE_AND_SCHEDULE_ONLY_SAME_V15_ENGINE_NOT_OLD_VS_NEW_ENGINE`,
`historical_score_beaten_claim_allowed` false, `historical_score_comparability`
`UNESTABLISHED`, `accepted_registration_claim_allowed` false,
`production_use_allowed` false, reference_snapshot_sha256 `d16390b2...b45dc`,
matcher unchanged (axis 5 deg, transverse 4 mm, overlaps 0.60/0.60). All 34
original `SCORER_RESULT.json` fields are canonically identical in `SCORED.json`
(0 differing fields; 30 runner-added fields only).

| Metric | Control | Challenger | Paired delta |
| --- | ---: | ---: | ---: |
| True positives | 0 | 0 | 0 |
| False positives | 0 | 0 | 0 |
| False negatives | 71 | 71 | 0 |
| Precision | null | null | null |
| Recall | 0.0 | 0.0 | 0.0 |
| F1 | null | null | null |
| Minimum-family recall | 0.0 | 0.0 | 0.0 |
| Exact emitted-support-count accuracy | 0.0 | 0.0 | 0.0 |
| Exact localized target-set accuracy | 0.0 | 0.0 | 0.0 |
| Refused / malformed cases | 11 / 0 | 11 / 0 | - |
| Physical observed counts unknown | 11 of 11 | 11 of 11 | - |

Per-family false negatives (both arms): 10839000 18 (3 cases), 37361005 4,
38172000 34 (4 cases), 40500000 7, 49001000 5, 49024000 3 (sum 71). Per-case
dispositions (both arms, identical): all 11 `REFUSED` with `prediction_count`
0, `false_negative` = `gold_count` (6,5,10,4,3,7,6,8,4,8,10), precision and F1
null, recall 0.0, `observed_bend_count` null; reason
`EXPLICIT_SUPERVISOR_UNAVAILABLE` (phase SUPERVISOR, `result_binding` null,
diagnostics null) for CG191 and CG4FAF, `EXPLICIT_WORKFLOW_UNAVAILABLE` for the
other nine with `result_binding.supplied_sha256` equal to the RESULT hash;
reported phase `COMPLETE` for CG1E0 C/H and CG4BDD C (diagnostics:
registration_context_count 1, source_context_window_count 13/13/17, all group
and carrier counts 0) and `WHOLE_PART_REGISTRATION` for the rest. Null
precision and F1 are preserved as null (undefined), never coerced to zero; no
numeric F1 exists for this run.

Scoring-after-terminal ordering (from the runner's own inventory fingerprints,
equal to current mtimes; epoch nanoseconds, UTC): last case terminal
`cases/CGF41F49066A/challenger/COMPLETED.json` 1789266519518540472
(2026-09-13T02:28:39.519Z); `INFERENCE_TERMINAL.json` 1789266519520071532
(02:28:39.520Z); `SELECTED_REFERENCES.json` 1789266578956332200
(02:29:38.956Z, +59.436 s); scorer dispatch issued 1789266578964183209 and
claimed 1789266579015645024; `SCORER_OUTPUT.txt` 1789266579219637596;
`SCORER_RESULT.json` 1789266579241462353; `SCORER_RECEIPT.json`
1789266579242935162; `SCORED.json` 1789266579256166735; `COORDINATOR_COMPLETED.json`
1789266579258142758; `COMPLETED.json` 1789266640096804710 (02:30:40.097Z);
`FINAL_GATE.json` 1789266640549706425 (02:30:40.550Z). No file under `cases/`
has an mtime after `INFERENCE_TERMINAL.json`; all 44 inference dispatch files
precede it and only the scorer pair follows it. `STARTED.json` is
1789257127125356136 (2026-09-12T23:52:07.125Z), 48 s after the PRE placement.
The scorer receipt carries no absolute timestamp (wall 0.2759697078727186 s).

## Paired comparison with V7

Same eleven inputs, same 71-line reference snapshot
(`d16390b2...b45dc` in both), same matcher; both campaigns TP 0 / FP 0 / FN 71
in both arms, precision and F1 null, recall 0. The paired score difference
V8 - V7 is zero on every defined metric and null on precision/F1. The immediate
causes moved without any slot reaching an emitted support:

| V7 cause (released) | V7 slots | V8 cause | V8 slots |
| --- | ---: | --- | ---: |
| Extrema coordinate-work refusal (EC) | 10 | - | 0 |
| No passing start (PF) | 4 | PF | 12 |
| Membership cap (MW) | 2 | MW | 2 |
| Wide certificate interval (IW) | 2 | - | 0 |
| Near-tie storage (NT) | 2 | - | 0 |
| Residual rejection (RG) | 1 | RG | 1 |
| CPU kill | 1 | CPU | 4 |
| - | 0 | Measurement-consumer window refusal (MC) | 3 |

Per slot (V7 cause and leaf wall from the released diagnosis -> V8 cause and
leaf wall): CG08506 C EC 36.358 -> PF 43.818, H EC 55.840 -> PF 63.043; CG191 C
IW 1355.268 -> CPU 1861.348, H IW 530.090 -> CPU 1861.316; CG1E0 C EC 31.674 ->
MC 356.501, H EC 48.648 -> MC 392.212; CG4BDD C RG 222.185 -> MC 253.856, H NT
120.831 -> RG 288.536; CG4FAF C CPU 1861.971 -> CPU 1860.930, H NT 1316.186 ->
CPU 1860.923; CG64C C PF 25.004 -> PF 26.350, H PF 33.514 -> PF 35.097; CG65C C
PF 30.737 -> PF 32.758, H PF 46.856 -> PF 50.446; CG84C C EC 38.493 -> PF
45.253, H EC 59.006 -> PF 67.105; CGE4B C MW 12.471 -> MW 13.526, H MW 12.642 ->
MW 13.581; CGEBA C EC 27.996 -> PF 32.564, H EC 43.171 -> PF 47.677; CGF41 C EC
32.818 -> PF 36.951, H EC 50.204 -> PF 54.737.

Observations (combined-change attribution only):

- All ten V7 extrema refusals completed their context under V4 (largest
  `extrema_coordinate_visits` 89178513 for CG08506, all planned signed families
  completed; V7 stopped these at 256,001,094-256,001,863 attempted coordinates
  with 69-115 of 80-320 families). The V4 coordinate counter has different
  semantics from V1 (`NEW_INDEX_WORK_ONLY_18M_PLUS_M_PER_PARTITION_DEPTH_BUILD_PLUS_9E_QUERY...`),
  so the two counters are not the same unit and no speed-up ratio is stated.
  Every one of those ten slots then failed the unchanged finite prefilter with
  zero passing starts, i.e. the blocking stage moved downstream; V8 did not
  turn any of them into a registration candidate.
- CG64C and CG65C reproduce the V7 prefilter counters exactly (4454 / 12358 /
  12263 / 95 calls and 2045 / 5935 / 5930 / 5 calls; 0 passing), consistent
  with unchanged behaviour of that stage on these inputs.
- CG4BDD C: the V8 candidate values equal the released V7 values to every
  printed digit (surface p95 upper 0.665897241 mm, context p95 upper
  0.6956471254551617 mm, 63 calls / 203331 rows). V7 rejected this context at
  the 0.5 mm gate; V8 admits it under the provisional 1.0 mm gate and then
  refuses every measurement window. This equality is evidence that the new
  surface V4 / registration V4 consumer reproduced the V3 candidate numerics
  through the gate on this input; it is not evidence that the pose is correct.
- CG1E0 C/H: first-ever supported context on this case (p95 upper 0.9008 mm,
  admitted only because the gate is 1.0 mm; it would have been rejected at
  0.5 mm) followed by 13 refused windows. Both admitted contexts carry the
  frozen `pose_admission_declaration` label (`PROVISIONAL_DEVELOPMENT_ADMISSION_GATE_NOT_CALIBRATED_CONFORMANCE`,
  unresolved false-pose risk); neither is an accepted registration.
- CG4BDD H: the V7 near-tie refusal (32 ties) is gone with 256 ties; the
  candidate now completes and is rejected at 1.196 mm. CG191 C/H and CG4FAF H:
  the V7 interval-width and near-tie refusals are gone (surface V4 converts
  wide intervals to per-row uncertainty) and the candidate loops then ran to
  the frozen 1860 s CPU limit; CG4FAF C was CPU-killed in both campaigns. The
  CPU-kill count rose from 1 to 4 and the campaign inference wall from about
  5992 s (V7 leaf sum) to 9298.529 s, of which the four kills are 7444.517 s.
  For the 18 slots returned in both campaigns the worker CPU rose from 925.750 s
  (V7, released values) to 1850.329 s; this is not a speed-up.
- Pose-class dedup and the context cap were never exercised (supported =
  scheduled = 1 in the three scheduled slots; caps 6, 6, 6); the new-outcome
  class the protocol anticipated (preflight-budget refusals) did not occur; the
  new class that did occur is the flange window refusal inside the unchanged
  measurement consumer.
- Arm comparison: 20 of 22 slots have identical immediate causes in both arms
  and eight of the nine returned pairs (16 slots) have identical
  initialization and context counters; the only arm difference is CG4BDD (C
  admitted at 0.696 mm and refused at measurement; H rejected at 1.196 mm,
  with slightly different prefilter counters 4938 versus 4928 hypotheses). Challenger walls exceed
  control walls in every returned pair (composite scan-plane lane). Nothing
  here separates the four declared V8 changes from one another.

## Findings

1. Exit status unobserved (limitation, not repairable). No shell exit code
   exists for the detached parent. The runner's `release_requires` condition
   `OBSERVED_PARENT_EXIT_ZERO_FINAL_GATE_NO_VETO_AND_EXACT_INDEPENDENT_POST_AUDIT`
   is not met by observation. The coordinator reports that `ps` observed the
   parent's disappearance at 2026-09-13 02:30:42 UTC (supplied evidence, not
   observed by this reviewer; it is 1.5 s after the `FINAL_GATE.json` mtime).
   The attempt is terminal; a rerun cannot supply the missing observation.
2. Four CPU-terminated slots versus one in V7 (CG191 C/H, CG4FAF C/H), each an
   explicit supervisor refusal with no returned or partial result; internal
   stages unknown beyond the sampled stacks (Python-frame samples every 30 s,
   not continuous profiling). `budget_stop` is null in all four: the OS CPU
   limit, not a parent budget, ended them.
3. New retained outcome class MC (3 slots): a supported context under the
   1.0 mm gate reached the frozen measurement consumer and every window was
   refused (`FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY`,
   `FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL`), zero support records, zero
   predictions. Not the preflight-budget class anticipated by the protocol.
4. Two contexts admitted only because of the 1.0 mm gate (CG1E0 0.9008 mm,
   CG4BDD C 0.6956 mm); both are labelled provisional development admissions
   with unresolved false-pose risk in the frozen report fields; no accepted
   registration exists in any slot (`accepted_registration` false everywhere).
5. The ten V7 extrema-coordinate refusals all complete under context V4 but
   every one lands in the "no passing start" class; the capacity bottleneck
   moved downstream. The V1/V4 coordinate counters are different units.
6. The measurement-context schedule (pose-class dedup, cap) was not exercised:
   no duplicate or omitted disposition anywhere; caps computed consistently.
7. Native `INVALID_NATIVE_SOLID` is retained only for CG64C (both arms) in V8;
   the V7 native refusals for CG191 and CG4FAF cannot be confirmed because
   those workers returned nothing.
8. No retry, resumed start, relaxed limit or edited frozen source: each of the
   22 case directories holds exactly one `STARTED.json`, one receipt, one
   projection, one log and (18 cases) one RESULT, in the frozen order; 22
   unique one-use dispatch ids each claimed once; every `STARTED.limits` equals
   the frozen limits and every arm request digest equals the frozen request;
   the CPU kills occurred at the frozen 1860 s; the working tree, `source/`
   snapshot and archive members are hash-identical for all 74 sources; the
   parent's final source audit and both inventory passes report no change;
   `COORDINATOR_OUTPUT.txt` holds exactly 22 case lines.
9. Gold/reference isolation: the frozen runner gives workers only a dispatch
   id and token; the issued leaf record binds arm plus scan/CAD hashes; the
   worker installs a Python audit hook restricting opens to the frozen
   `backend` snapshot, its private cache and the runtime roots (a Python-level
   guard, not a syscall sandbox); `selected_references()` (the only catalog
   reader) is called solely by `score_after_terminal` after
   `INFERENCE_TERMINAL.json` is verified. `SELECTED_REFERENCES.json` was
   written 59.4 s after `INFERENCE_TERMINAL.json` and after all 22 terminals;
   no case file changed afterwards. No evidence that Gold or reference data
   reached inference; this rests on frozen-source reading and ordering
   evidence, not on a runtime observation of file accesses.
10. Coordinator private cache `/private/tmp/gold11-v8-coordinator-6ebyksl4`
    (identity device 16777231 / inode 35822845) is declared
    `OPAQUE_PRIVATE_CACHE_RETAINED_OUTSIDE_ATTEMPT`, not enumerated by the
    runner or by this audit; the 23 leaf caches are declared removed.
11. All 74 frozen source files exist in the working tree only as untracked
    files (`git status` `??`), as do the attempt directory and the archive
    directory; custody rests on the hash-verified snapshot and archive, not on
    version control. Not a blocker; recorded for the root.
12. `PRE_RUN_AUDIT.json` bytes are not in canonical JSON form (the runner
    binds its raw-byte hash and its canonical digest separately; both
    reconcile). Cosmetic.
13. Resource envelopes: parent lifetime high-water RSS 1557233664 B and
    sampled outer tree peak 2591260672 B (below 12 GiB); parent wall 9512.744 s
    and FINAL_GATE wall 9513.198 s (below 104400 s); COMPLETED 203850 B and
    FINAL_GATE 1056 B below 256 KiB; SCORED 146867 B and SCORER_RESULT 35447 B
    below 32 MiB; largest RESULT 79329905 B below 128 MiB; largest log 97549 B
    below 1 MiB; scorer wall 0.276 s and sampled RSS 87556096 B below 180 s /
    2 GiB. Sampled values are not continuous guarantees.
14. Combined-change caveat: V8 differs from V7 in mechanisms, capacities and
    one policy gate at once; no per-change attribution is possible from these
    ledgers, and faster or slower refusals are not geometry improvements.

## What was not checked

- Process exit status of the parent (unobservable; see Decision).
- No runner, projection (`revalidate_gold11_schedule_projection_v3`), scorer or
  engine code was executed: inner projections were bound by canonical-hash
  identity with the retained `contact_projection` objects, not re-projected
  from RESULT bytes; the matcher was not rerun.
- Raw scan/CAD bytes were not read or hashed; the Gold catalog
  (`bend_line_gold_catalog_v1.json`) was not opened; input binding rests on the
  frozen declarations, `source_binding_valid` receipts and the snapshot digest.
- Private caches under `/private/tmp` were not stat'ed or listed; no `ps` or
  other process query was run by this reviewer; no live-writer check.
- The V7 attempt directory was not opened; V7 figures are from released
  documents only.
- The encoded `patch_evidence.scan` wire, `proposal_dispositions`, hypothesis
  and family arrays were not decoded or reviewed beyond counts and string
  presence; the 42 KB supervision helper and the artifact helper were not
  reviewed line by line (their frozen hashes were verified).
- Sampler records were counted and their frames read for the four killed
  slots; sampler policy conformance was not otherwise verified.
- `completed_publication_inventory_sha256` and the stdout inventory digest were
  reproduced by data transformation of the retained inventory, not by running
  the inventory function.

## Check 8 - parent stdout capture and exit status

`/Users/idantal/.claude/jobs/1519fdc2/tmp/v8_execute_stdout.txt` (466 B)
contains exactly one line, the runner's final status:
`{"status": "COMPUTED_PENDING_INDEPENDENT_POST_AUDIT", "attempt": ".../gold11_mesh_development_v8_attempt", "release_allowed": false, "durable_terminal_case_records": 22, "final_campaign_inventory_sha256": "dbeb824d1929e6f3b638cfc3cc1f8ed13899fc5a5da39a62d5fbc7a2912445bb", "final_campaign_inventory_entries": 283, "final_scientific_manifest_entries": 115, "final_inventory_stdout_is_not_durable_runner_artifact": true}`.
In `scripts/run_gold11_mesh_development_v8.py` `execute()`, that line is printed
only after `COMPLETED.json`, `FINAL_GATE.json` and the post-FINAL_GATE inventory
succeed, and `return 0` follows one more wall/RSS veto whose failure would try
to write `FINALIZATION_FAILED.json` (absent) and return 1. The recorded wall
(9513 s) and parent RSS (1.56 GB) are far below that veto's limits (104400 s,
12 GiB). These facts are consistent with a zero return from `execute()`, and I
reproduced the printed inventory digest from the retained records. They do
NOT constitute an observation of the process exit status: the launch was
detached and no exit code was captured, so the runner's
`OBSERVED_PARENT_EXIT_ZERO...` release condition remains unmet by observation
and cannot be met by any later action on this terminal attempt.
