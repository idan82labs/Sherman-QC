# Gold11 V5: independent terminal failure POST

Date: 2026-09-12.
Reviewer: Codex independent technical reviewer `/root/cad_blocker_closure_review` (AI).

## Decision and terminal boundary

**RED — TERMINAL FAILURE; NO NEW F1 PRODUCED.** The sole frozen V5 attempt remains consumed and must not be resumed, rerun, rescored, or reinterpreted as a completed comparison.

Root observed execution session2399 terminate with exit1 at11:08:34 UTC and independently confirmed parent14761, coordinator14763 and PGID14763 absent at11:08:46 UTC. Root then explicitly released the fixed terminal controls, eleven/twenty-two case artifacts and named freeze archive for this POST. No final success stdout was emitted. Those external termination observations are attributed to root; this reviewer did not infer quiescence from a favorable score or kill the run.

The campaign's first recorded stop is its process-identity guard, not its frozen global wall/RSS allowance. Independently of that abort, **all eight actual returned engine outputs refused whole-part registration, with no measurements and zero predictions**. The scorer never ran. There is neither a numeric F1 nor a completed null-valued scorer report to publish.

## Scope and exact custody

Attempt: `docs/research/gold11_mesh_development_v5_attempt`.

I used bounded stable regular-file reads, hashing, JSON inspection, source inspection, stdlib-only runtime metadata and in-memory archive verification. All fixed case directories were checked for their declared artifact names. No target import, test, native/geometry execution, inference, refit, matcher invocation or rescoring occurred. No original raw input, Gold catalog/reference label, private cache or `coordinator_work` payload was opened. The exceptional cache path below is known only from the retained parent inventory; I did not enter or stat that child.

The accepted physical [PRE](2026-09-12_gold11_mesh_development_v5_pre_run_audit.md) remains12992B SHA256 `3cd9e08797476661cf238336a91b7cfff579bec239a5333e5cf2e18e3927dc53`. Its accepted85-outcome source qualification is not rerun or promoted into physical accuracy evidence.

| Root artifact | Bytes | SHA256 |
| --- | ---: | --- |
| FROZEN.json | 33512 | `4e4308f70d4a4fc24c388fc8293a9c2f5c71c723c9fc8756160dc672bcb6702c` |
| PRE_RUN_AUDIT.json | 269 | `a434292e8508aa4158d5ab62e9b31030ababa3987fea758713e1670d5bc75414` |
| STARTED.json | 944 | `c5359e924db90c794765072882d8cf3d7eaf13209195267f68462ee908164373` |
| COORDINATOR_DISPATCH.json | 103 | `eb54a46370bab2b7ec2c188f2bef30a0d2fb1da98e0d6cecde877288a7ef9034` |
| COORDINATOR_VERIFIED.json | 7605 | `562446a160b4aeefc0fdd1fd0fbd6973371ac09e5012f3b0ccc562934835ad7c` |
| COORDINATOR_OUTPUT.txt | 1584 | `b5403b881e92ade339bf9b44a37f2c493e343f96b41098da5e99e1450898898f` |
| COMPLETED.json | 127384 | `78b6b4480dc99e0c03f9299c16c8517336b620f2d513d6f28c58d9f99886d0f2` |
| FINALIZATION_FAILED.json | 353 | `6787bba729cb19608703a8bc3fcba5960d7f2d44dfb65dea0e082fff3e2f7055` |

Independent reads bound80 root/case artifacts,410690629 bytes: the eight root files above plus44 terminal/projection records, ten case STARTED files, ten case logs and eight full RESULTs. The canonical path-to-`{bytes,sha256}` map has SHA256 `5f812664552e5d2ad1d693922161722401d729f25bc297cae1441a616fba8426`. Its scope excludes source files, leaf controls and private caches. Canonical maps use sorted-key compact UTF-8 JSON with nonfinite values forbidden and no trailing newline.

All20 issued/claimed leaf-control files for the first five case pairs were separately bound. Their same-format ledger SHA256 is `f225226f197c60807500267070304d39cfd9230cbf563b94f176a7592a4d1098`. Declared request/arm/input hashes, coordinator context hash, claim-to-dispatch hash and no-retry fields reconcile. Recorded worker PIDs are14766/15831,17452/18288,19567/20509,21948/28806,32667/86033 in case-pair order. This authenticates retained neutral control bindings, not the identity or contents of any current cache or a process birth certificate.

All61 live and frozen source copies match the accepted bundle and each other:1778581 payload bytes; canonical source-map SHA256 `149070a6aeba9283fb056567cd4530d6bd7f0208f3f571d490e2dad863cf959e`. All five parent-bound controls also match. The repository-root initializer remains absent.

The named archive `docs/research/cad_engine_v1_synthetic_checks/gold11_mesh_development_v5_frozen.v_qrbrat/SOURCE.tar` is1884160B SHA256 `a54cf23f01a3085c57a5d74936144d2481a9cb5e8cb57ab735cf23ab8dd3ce19`. I verified exactly64 unique regular payloads,1825354 bytes, against all61 source copies plus exact FROZEN, approval marker and PRE. No extraction or archive mutation occurred.

The canonical bundle remains `42f6d6ad3faf9944c403d65349174f70f2dc72add930636cb8bd7d598fe28b5a`; CONFIG `9411366c98a7664a9efdea3f1cd4368ea0459350dab553b9e9792c88e0275575`; limits `8fd437267181e3df7a4a2b516649889ecb71d16c98fbf70adcb703a17f436865`; case declaration `943268be4ee3d33301ab44dfb3e6f85fb0d34de852f788464f1d438ffe43813e`.

Fresh stdlib-only metadata equals FROZEN: Python3.11.15, darwin/arm64, NumPy2.4.6, SciPy1.17.1, cadquery-ocp7.9.3.1.1, and the same resolved application executable. No packages were imported to perform that check. This remains version metadata, not binary/bytecode attestation.

## Campaign abort and incomplete finalization

The parent receipt records:

- `budget_stop=OBSERVED_PROCESS_IDENTITY_CHANGED_OR_REUSED`, coordinator exit-9, `observed_process_identity_change_detected=true`, peak one identity-unknown PID;
- direct child reaped and owned group confirmed empty; cleanup/control-close error lists empty;
- foreign-group escape flags false and identity-unknown-still-present false after cleanup;
- coordinator supervision wall3493.949967375025s; sampled tree peak1948598272B, sampled budget peak1988624384B; parent terminal wall3494.163027250208s and peak46891008B.

The frozen supervisor source compares previously retained PID-to-PGID entries with later process-table samples and selects this stop when a known numeric PID appears under another PGID. This is the evidenced guard mechanism. The terminal records do not retain the offending PID's birth identity or full historical sample, so **actual escape versus ordinary PID reuse/identity ambiguity is unresolved**. The named reason is not proof of malicious escape or proof of a harmless false alarm. A future source-level repair must preserve that distinction.

The first inventory has `complete=false`, `phase=ALL_SCHEMA_BRANCHES_VISITED`, no unattempted schema directories, no deadline expiry,202 observed entries and74 scientific entries against128. Its two errors are:

- `UNDECLARED_CAMPAIGN_NAME`: `coordinator_work/gold11-mesh-v5-thx0pl0m`;
- `QUALIFIED_LEAF_INVENTORY_UNAVAILABLE`: `coordinator_work`.

It reports412366988 read bytes and exact counter scope. This is an explicitly incomplete inventory, not a successful prefix. The unrecognized working-directory entry remains a custody/cleanup limitation; neither its payload nor raw-input cleanup is established by this POST.

The parent retained both `TERMINAL_EVIDENCE:COORDINATOR_EXIT_RESOURCE_OR_GROUP_CUSTODY_FAILURE` and `COMPLETE_CAMPAIGN_INVENTORY:COMPLETE_CAMPAIGN_TREE_CUSTODY_REQUIRED`, then published the terminal veto. Only the failed pre-COMPLETED inventory exists. The post-COMPLETED and post-FINAL_GATE success phases were not reached and must not be counted as passed.

## Exact22 dispositions: engine yield is separate from wrapper admission

All22 COMPLETED/SCORING_INPUT pairs exist and match their terminal-fill byte/hash records. Nine pairs were preexisting and retained without rewrite; thirteen pairs were newly filled after abort. All eight clean-return wrappers bind the exact original RESULT hashes and byte counts, and their inner projections bind the same bytes. I inspected those complete RESULT objects directly; no geometry or projection function was rerun.

| Case | Control | Challenger |
| --- | --- | --- |
| CG08506AE97D | Returned UNAVAILABLE; context BUDGET_INDEX_VISITS | Same |
| CG191DEC7F81 | Returned UNAVAILABLE; context BUDGET_INDEX_VISITS | Same |
| CG1E0BBA2933 | Returned UNAVAILABLE; context BUDGET_INDEX_VISITS | Same |
| CG4BDD1371B6 | Returned UNAVAILABLE; NO_GEOMETRICALLY_SUPPORTED_MEASUREMENT_CONTEXTS | Returned UNAVAILABLE; FINITE_SURFACE_QUERY_UNAVAILABLE:NEAR_TIE_BUDGET_EXCEEDED |
| CG4FAF273093 | Began; exit-24/SIGXCPU; no returned RESULT | Began; interrupted by campaign abort before own receipt/RESULT; global terminal fill |
| CG64C412CC73 | Global terminal fill; no retained dispatch/start/log/result | Same |
| CG65CD575B1C | Global terminal fill; no retained dispatch/start/log/result | Same |
| CG84C8FA12D2 | Global terminal fill; no retained dispatch/start/log/result | Same |
| CGE4B0549E41 | Global terminal fill; no retained dispatch/start/log/result | Same |
| CGEBA9A869AC | Global terminal fill; no retained dispatch/start/log/result | Same |
| CGF41F49066A | Global terminal fill; no retained dispatch/start/log/result | Same |

The eight returned workflows all have `status=UNAVAILABLE`, `phase=WHOLE_PART_REGISTRATION`, `measurements=null`, and zero predictions. Their outer schedule-wrapper `ADMITTED` status means structural transport admission, **not** engine success. Every inner contact projection is `REFUSED` with `EXPLICIT_WORKFLOW_UNAVAILABLE` and zero predictions.

For the first three pairs the two-plane schedule reports COMPLETE, but context generation reports `BUDGET_INDEX_VISITS`, propagating through `TWO_PLANE_CONTEXT_UNAVAILABLE` and `NOISY_PLANE_INITIALIZATION_UNAVAILABLE`. For CG4BDD1371B6 both context generation and initialization report COMPLETE before the distinct downstream registration refusals shown above. None reaches a usable measurement or raw-support-union fit.

CG4FAF273093/control's numeric exit-24 is SIGXCPU on the bound platform. Its public-workflow-begin log, frozen1860s CPU limit and1863.4591998751275s case wall are consistent with the OS CPU limit; the receipt's sampled `budget_stop` is nevertheless null and there is no worker CPU footer or returned internal-stage ledger. Do not invent its exact consumed CPU or deepest internal stage. Challenger's bound dispatch/claim and begin log establish that it began, but its own terminal/result did not survive the campaign stop. The generic fill receipt's unknown fields remain unknown.

The final six pairs have no retained dispatch/start/output/result and were not reached by the frozen serial route before abort. The thirteen global-fill receipts comprise the interrupted challenger plus those twelve remaining opportunities. Terminal storage completeness is not 22 completed inferences.

## Exact retained full results and resources

| Case/arm | RESULT bytes | SHA256 |
| --- | ---: | --- |
| CG08506AE97D/control | 53254124 | `82f89e2b92a394906aaf55e7f6a6e29078b4267b266eef560f5ee86173239bec` |
| CG08506AE97D/challenger | 54675081 | `bfd9ec155f56ad7b8ea2b59a49ad8cdf04b7d3ed4059e42ff0d331f101523d4c` |
| CG191DEC7F81/control | 41687648 | `00627cf46625cdd0e0ae992047a2fd78abeeb9b5e67be94424bd2147b0dd5f6d` |
| CG191DEC7F81/challenger | 42932262 | `ad74f208e714373b0f6f204a977502cb233708020e22890dfaab83e714f98782` |
| CG1E0BBA2933/control | 47421106 | `715f88c8ab1181e79cde81849abbef5893966f75456b543892fafd33b3e72645` |
| CG1E0BBA2933/challenger | 48685581 | `b49980ca92e402315b330b9dc44dcf086da7733112bfe65cea916bef16687963` |
| CG4BDD1371B6/control | 60191496 | `3593c3b93c3391716b611ad4564272083eab97268efa2049fd90f9a919512f08` |
| CG4BDD1371B6/challenger | 61545607 | `ac48d6192e9db98a0078bc84d28021765df183994f783a29754cecf8ddc2c170` |

No declared fixed case RESULT_PARTIAL.bin exists. This does not assert there are no incomplete bytes in the prohibited private cache; its contents remain uninspected.

Completed-worker CPU footers in table order are34.989339,54.546764,27.439516,42.492148,30.831291,48.252042,232.103632 and127.403337 seconds, totaling598.058069s for those eight only. Their maximum reported worker RSS is1793654784B. Missing ninth/tenth CPU footers mean no exact total campaign CPU is available.

All seven limits remain unchanged: caseCPU1860/wall3540, scorerCPU120/wall180, parent-read180, terminal reserve2400 and campaign wall104400s. The identity-triggered campaign failure happened around58m14s, well before the global wall allowance and below its sampled12GiB tree threshold. Sampled memory is not an OS allocation bound, and these observations do not justify retroactively increasing any begun cap.

## Score absence and permitted conclusion

Exact fixed-path inspection confirms absence of FINAL_GATE, COORDINATOR_COMPLETED, INFERENCE_TERMINAL, SELECTED_REFERENCES, SCORER_RECEIPT, SCORER_OUTPUT, SCORER_ARTIFACT_RETENTION, SCORER_RESULT, SCORER_RESULT_PARTIAL and SCORED. SCORING_UNAVAILABLE is also absent: the killed coordinator did not publish that ordinary exception-path record. The outer COMPLETED/veto records are the actual failure evidence.

The22-before-Gold scoring barrier was never satisfied by a completed coordinator, and no fresh selected71 snapshot or scoring request/result was published. The71 denominator remains the frozen authorized expectation, not a newly evaluated confusion matrix. Do not report inferred TP/FP/FN or manufacture F1=0/null as an actual scorer result. In a completed unchanged scorer, TP=0 would give null F1, but that arithmetic convention does not substitute for this missing execution.

The defensible outcome is: **ten workflows began, eight returned registration refusals, one hit SIGXCPU, one was interrupted, twelve remaining opportunities were terminal-filled without being reached, and no F1 was computed**. Source/bundle stability is verified; campaign publication/custody and physical-output success are not green. The identity-guard mechanism and early registration resource refusals are separate prospective repair targets. The future union planner is not demonstrated to fix these pre-measurement failures.

Preserve this attempt, veto, accepted PRE, frozen archive and all prior RED evidence. Any future correction must be separately versioned and prospectively reviewed; this failure POST grants no retry, cached recovery, rescoring, new-data access or production promotion.
