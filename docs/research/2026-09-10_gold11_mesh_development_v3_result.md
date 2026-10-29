# Gold11 physical V3: negative engine observations; formal release blocked

Date: 2026-09-10. Status: root observed parent exit0; independent retained
post-audit BLOCKS formal release because44retained partial control files are
absent from the parent's artifact manifest. Required source/result/scorer hashes
remain intact; the negative observations below are separately authenticated,
not a fully released comparison. Do not rerun, clean or overwrite this terminal
attempt. A successful runner exit is not a scientific or integration release.
The durable [independent post-audit](gold11_mesh_development_v3_attempt/POST_RUN_AUDIT.json)
records `BLOCKED_FORMAL_RELEASE_INCOMPLETE_ARTIFACT_CUSTODY`; raw SHA-256
`0c2a2bb7cb54ebef66b1dd4ea9d60b1cc98f4cdb6011583202919d69bbf1327d`.

## Outcome

All11 consumed cases and both fresh arms completed with durable dispositions.
Neither arm emitted a usable line. The fixed71-reference scorer reports:

| Arm | Lines | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Fresh V8 control, nativeV1 | 0 | 0 | 0 | 71 | undefined | 0 | undefined |
| Fresh V10 challenger, nativeV1 | 0 | 0 | 0 | 71 | undefined | 0 | undefined |

Each arm has11refusals,0malformed scored cases, minimum-family recall0 and
0/11exact emitted-line totals. Physical observed-bend count remains UNKNOWN in
all11cases, not zero. Physical bend-by-bend accuracy is not evaluated without
measurement truth. No positive F1, comparable paired gain or production promotion.

The separately independently POST-GREEN
[cached UNION control](2026-09-10_gold11_cached_control_v2_result.md) remains
30TP/30FP/41FN, F1 `0.4580152671755725`. Its numerical headline is reproduced;
old engine invocation/runtime/reference-history attestation remains incomplete.
V3 does not beat that control and does not establish historical comparability.

## Exact challenger attribution

The rows below partition all11cases/71references. A CPU-killed worker's internal
stage remains unknown; do not assign it the last stage observed in another arm.

| Earliest retained outcome | Cases | References | Evidence |
| --- | ---: | ---: | --- |
| Completed finite screening, no passing pose | 6 | 45 | Local numerical consumers finish, but every generated start fails finite-support screening. No refinement candidate or measurement is produced. |
| Worker exit-24 under150s CPU cap, no returned RESULT | 3 | 19 | Entire cases retained as unavailable. Internal stages, counters and possible in-memory progress are unknown. |
| Expanded membership admission failure | 1 | 4 | Same20M logical cap;19,518,222 admitted,21,392cores/15closures/9,450proposals invalidated as a whole. |
| No pose from the scheduled triplet search | 1 | 3 |82-plane bank,158scheduled/evaluated triplets, zero proposed poses. |

Per-case details (all TP0/FP0 and all listed references retained as FN):

| Case | Family | Refs | Challenger outcome | Parent case seconds |
| --- | --- | ---: | --- | ---: |
| CG08506AE97D |10839000|6| Finite screening complete, zero starts |37.832484|
| CG191DEC7F81 |49001000|5| Exit-24, no RESULT, internal stages unknown |150.999727|
| CG1E0BBA2933 |38172000|10| Exit-24, no RESULT, internal stages unknown |150.787208|
| CG4BDD1371B6 |37361005|4| Exit-24, no RESULT, internal stages unknown |150.707849|
| CG4FAF273093 |49024000|3| No generated pose from scheduled triplets |2.806912|
| CG64C412CC73 |40500000|7| Finite screening complete, zero starts |24.261864|
| CG65CD575B1C |38172000|6| Finite screening complete, zero starts |32.501700|
| CG84C8FA12D2 |10839000|8| Finite screening complete, zero starts |41.372338|
| CGE4B0549E41 |10839000|4| Expanded membership admission failure |13.825607|
| CGEBA9A869AC |38172000|8| Finite screening complete, zero starts |29.295903|
| CGF41F49066A |38172000|10| Finite screening complete, zero starts |35.729188|

Across the six completed finite-screening cases only:18,541proposed poses,
55,623assigned supports,54,951box-excluded supports,672completed finite-query
supports,3,340,144certificate pairs, and zero passing start groups. These are
not counters for the three CPU-killed cases. Boxes are rejection-only; their
exclusion counts do not establish that a true pose was generated or excluded.

Root's bounded read-only aggregation of the same six complete RESULTs further
finds17,918poses with zero passing assigned supports,610with one,13with two,
and none with three. Of672fully queried supports,636pass individually and36fail;
the54,951box-excluded supports are separate. The72,941queried rows contain
69,320definite distance inliers,3,621definite outside rows and zero threshold-
unresolved rows.10,465region-uncertain/boundary rows and109ambiguous rows overlap
those distance categories; do not add them as a disjoint partition. Maximum
returned distance-interval width is8.438169424415311e-6mm.

This localizes the observed joint-support rejection more closely than saying
"numeric uncertainty": most generated assignments are spatially excluded,
and some individual face correspondences pass, but no generated triple does.
It does NOT establish the correct physical pose, whether plane normal/offset
estimates or correspondence scheduling missed it, or whether the13two-support
seeds would pass unchanged whole-part refinement/rank/ambiguity checks. No
support gate or threshold was changed and no model was rerun for this inspection.
This additional support histogram/row-detail aggregation was NOT independently
recomputed by the post-reviewer; distinguish it from the independently verified
six-case stage and headline counter partition above.

The fresh control retains the previous9+1+1 admission pattern. Thus six cases
move beyond the old numerical-admission blocker, while three now consume their
CPU allocation without a returned result. This is intermediate stage progress
and newly exposed runtime cost, not improved predictions. The unchanged
constructed V10 success did not establish this physical operating range.

Native validity remains an overlapping blocker: control artifacts independently
report INVALID_NATIVE_SOLID for CG191DEC7F81, CG4FAF273093 and CG64C412CC73
(15references); all other control inventories are PARTIAL. The latter two also
have returned challenger native failures. The first challenger's native stage
is unknown because its worker returned no RESULT; do not copy the control's
stage into that missing challenger record. Successful registration would still
not qualify unavailable native measurement windows.

## Execution, custody and resources

Executed FIRST ONCE after exact independent PRE-GREEN:

```bash
.venv/bin/python -I -S -B scripts/run_gold11_mesh_development_v3.py --execute
```

This is a reproducibility record, NOT a repeatable command against the spent
attempt. A successor requires a new engineering reason, frozen protocol and audit.
Attempt: `docs/research/gold11_mesh_development_v3_attempt`.

The parent reports no campaign failure, no outer budget stop, no cleanup error,
all38current/frozen sources unchanged, and a manifest of116owned files and24logs.
Its complete-custody claim is incorrect: independent directory inventory found
44additional partial control files, described below. There are22durable case
dispositions and19full returned case RESULTS;
the three killed workers returned no complete or partial RESULT. Final scoring
followed the all22-inference-terminal barrier with the unchanged71-reference
snapshot. The required records/negative metrics remain independently checkable;
that does not cure the incomplete frozen artifact ledger.

Parent completion wall959.111951s; final-gate observation959.113147s (about16min).
Sampled monitored group/descendant-plus-parent peak2,012,168,192B, versus12GiB
frozen global limit. Parent high-water944,046,080B is a separate measurement;
peaks need not be simultaneous. RSS is sampled, not a hard allocation cap or a
continuous peak proof. Three case CPU terminations do not imply an outer campaign
budget failure. Case caps remain180wall/150CPU; none was widened or retried.

Key retained SHA-256 bindings:

| Artifact | SHA-256 |
| --- | --- |
| Canonical FROZEN bundle |`ffee4c1012393ff67607160c888bd5d00ea0ac5b81784b6f4ab32d40d3fa9a27`|
| Raw PRE_RUN_AUDIT |`6cb07d52cc35739ef908fdb5b2c73fdbf3c00796f2bc2f8cb4bfe7c7dee3018a`|
| Raw COMPLETED |`9570277bb6781406a5199d4a5c37a2aaeea560437558543f58a9a10ec53412fc`|
| Raw FINAL_GATE |`82e6a8534f308783c9a057b731af7ff38373ac228e36e646a2541e7b32900021`|
| Original SCORER_RESULT |`2ebf8c0d62281323aaf301ef6c737a7fb18db21f998c4f18cdc91a1da1ca4f2d`|
| Enriched SCORED |`5f4d0359bdf8ebd079fa6dcfda174d4d24db5950dac42068352b9b331cdfd166`|
| COORDINATOR_OUTPUT |`f6e1f7a4b2ce83607ef3249675b0f3e4e6392a76e66643872f55e9c90d3fc104`|
| Selected71references |`d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc`|

## Independent post-audit: control-publication defect

Reviewer `/root/production_architecture_review` independently found44additional
regular files totaling7,348bytes: one `.SCORING_INPUT.json.partial-*` and one
`.COMPLETED.json.partial-*` under each of22case/arm directories. These are NOT
hardlink aliases of the canonical files. They contain fallback unavailable /
unknown terminalization JSON with reason GLOBAL_ABORT_OR_MISSING_TERMINAL_RECORD.
The canonical files were not overwritten and the fallback partials did not enter
scoring. All116manifested hashes and76current/frozen source-copy hashes match.

Exact cause: `terminalize_case_records` unconditionally attempts exclusive
publication even when a terminal target already exists. `publish_bytes` writes
the partial before its exclusive hardlink fails with FileExistsError. The
terminalizer then validates the existing canonical file but leaves the newly
created partial unaccounted. The parent inventory verifies expected files but
does not reject these extras. This is a connected publication/test-coverage
defect, not a geometry failure or an explanation for zero engine predictions.

The complete frozen ledger would require160entries, above its128-entry cap.
Do not edit the manifest, enlarge the spent cap or delete the44files. Formal run
release remains blocked even though the negative engine observations are intact.
A separately versioned runner repair must avoid unnecessary existing-target
publication and validate the complete allowed directory inventory. The existing
repeated terminal-fill test exercises real publication but checks canonical
bytes/counts, not all directory entries; it therefore misses the extra files.
Tests must assert the complete inventory during real atomic publication,
including existing targets, non-aliased partials and unknown-file controls. Use the
repair in the next justified engine successor; do not rerun this physical
candidate merely to repair its historical custody ledger.

## Consequence for the roadmap

Do not promote V10 as an accuracy improvement. Retain its demonstrated numerical
mechanism separately from physical prediction performance. Prioritize useful
label-independent pose generation AND complete finite-support/runtime behavior,
not additional tests of already qualified numerical primitives. The two-plane
context helper is independently source-reviewed with110invented tests authored
but unexecuted; it is neither integrated nor a demonstrated solution. Its tests
explicitly show that asymmetric partial context can miss the true pose, and
central normal proposals do not enclose all feasible rotations.

Next connected gates must use the declared physical .05mm/free-circle regime,
include real accepted/overlapping closure workloads, and preserve partial-scan
ambiguity, signed/material-side semantics and all downstream rank checks.
Investigate CPU cost with a separately frozen diagnostic that retains stage
evidence before termination; do not infer the three missing stages or extend
their spent limits. Develop compact membership admission as an explicit new
policy with unchanged expanded lineage, not a renamed old cap. Keep native local
facts distinct from executable inner-surface measurements.

The [failure-factor audit](2026-09-10_gold11_failure_factor_audit.md) and
[execution plan](2026-09-10_gold11_output_recovery_execution_plan.md) remain active.
Common-mm/frame/carrier compatibility, physical uncertainty, datum stability and
manufacturing tolerances remain unresolved as declared. No held-outs/email,
reference edits, product default change, hardware or factory-floor release.
