1. RULING

AUTHORIZE_FROZEN_V51_ON_NEW_UNTOUCHED_CORPUS_ONLY

2. VALIDITY ASSESSMENT

The Gold37 event is terminally invalid. Per-case manual_bends were exposed before detector-artifact sealing; detector and replay counts remained 0/37, no score was produced, and the incident record expressly prohibits replacement execution. Consequently, Gold37 can support neither a V51 performance claim nor a V51 geometric pass/fail conclusion and must never receive a V51 rerun. 

Pasted text +1

The incident invalidated the evaluation corpus for this architecture, not the already-frozen architecture itself. There was no detector observation from which V51 could have been adapted, and the packet supplies no scored or per-case evidence that falsifies V51. Its only current evidence remains the hypothesis-generating P01–P07 sequence: P06 supports the narrow parallel-return mechanism visually, while P07 is an independent no-op, but neither constitutes locked F1 evidence. 

Pasted text

A V52 selected now would therefore add a new adaptive degree of freedom without a valid V51 transfer result. Opening a new sensing capability is also premature because the present scan geometry is sufficient to test the frozen hypothesis; what was lost is a valid holdout, not geometric observability.

The frozen pre-exposure V51 bytes may be executed only on a future corpus assembled after this ruling and never on Gold37. The result will be interpretable as a locked finite-corpus transfer test, not as an unbiased population-level guarantee. No P08-style visual development run or other real-scan invocation may intervene between this ruling and the new corpus execution.

3. RESEARCH BASIS

Cawley and Talbot show that adapting model choices to a finite evaluation criterion can overfit that criterion and bias subsequent performance estimates. The adapted principle here is strict separation: Gold37, once opened, cannot remain the confirmatory evaluation for V51, even though no detector score was produced. 
Journal of Machine Learning Research

Dwork and colleagues formalize how adaptively generated analyses undermine ordinary holdout validity. Their reusable-holdout algorithms are not being implemented here; therefore the defensible operational remedy is the simpler one: discard Gold37 as a V51 holdout, keep V51 unchanged, and disclose a fresh holdout only once after primary and replay outputs are sealed. 
arXiv
+1

Recht and colleagues demonstrate that a newly collected test set can also introduce distribution shift even when intended as a replication. The new Sherman corpus must consequently reproduce the intended production acquisition, reconstruction, cleaning, and annotation process under a precommitted sampling rule; otherwise its score would confound V51 transfer with a changed scan-generation process. 
Proceedings of Machine Learning Research

Published point-cloud work supports extracting cylindrical primitives from unorganized geometry, and published NMS work supports deterministic suppression of competing hypotheses. Neither establishes that different CB<n> identifiers correspond to different physical bends, nor does either validate V51’s exact length, angle, offset, overlap, activation, ranking, or ownership predicates. Those remain Sherman-specific hypotheses to be falsified by the fresh corpus. 
Wiley Online Library
+1

4. EXACT ACTION

Architecture identity and precommit. Create one append-only event with token V51_NEW_UNTOUCHED_CORPUS_BLIND_ONCE_LOCKED. Bind the exact pre-exposure V51 detector, inherited selector and model-bundle bytes, candidate-pool builder, pair_metrics, serializers, renderer, scorer, dependency lock, CPU/thread configuration, and canonical result schema. Verification must use the pre-exposure V51 precommit hashes—not a post-incident rebuild. The present packet archive and manifest remain the incident-evidence anchor. 

Pasted text

 If any executable byte cannot be matched to the pre-exposure identity, stop before corpus contact.

Gold37 truth, Gold37 scoring artifacts, and every historical truth directory must be absent from the detector container or process account at the filesystem-permission boundary. Personnel exposed during the incident may mechanically launch the sealed runner, but may not select corpus cases, define exclusions, annotate truth, change the harness, inspect pre-score predictions, or adjudicate annotations.

Frozen composition and feature schema. Execute exactly:

V16 → V43 → V44 → V50(V45 semantics with longest-first incumbents) → V47 → V48 → V49 → V51.

V46 remains historical and is not reinserted. No learned feature is added or removed: V16’s imported FEATURE_NAMES ordering and its V9/V4 model-profile checks remain byte-bound. 

Pasted text

V51 itself receives only the post-V49 predictions and the source-only candidate pool. Its decision inputs remain:

post-V49 prediction count;

incumbent source_hypothesis_id or bend_id, source_kind, and line endpoints;

scan_bbox_mm;

candidate candidate_id, representative_source, span_endpoints, and score;

derived incumbent and candidate lengths, parsed CB<n> family, axis_angle_deg, transverse_offset_mm, and shorter_span_overlap.

No filename, case identity, part or family identity, expected count, annotation, CAD, drawing, or benchmark result enters these computations.

Exact V51 predicate. Preserve the implementation without reinterpretation:

Activate only when the post-V49 result contains exactly one prediction.

Require the incumbent to have source_kind == "cylindrical_standard" and a candidate-ID leaf matching ^(CB\d+)_.

Require incumbent length to be at least 0.75 × max(scan_bbox_mm).

Consider only pool rows with representative_source == "cylindrical_standard", a recognized CB<n> family different from the incumbent family, and:

candidate length at least 0.75 × incumbent_length;

undirected axis angle at most 2°;

transverse offset at least 8 mm;

transverse offset at most 0.08 × incumbent_length;

shorter-span overlap at least 0.80.

Rank eligible rows by descending score, then descending length, then _row_key.

Append at most the first eligible row and canonically sort the resulting predictions. 

Pasted text +1

V50 retains longest-first incumbent ownership. V47 retains its elongated-lane consolidation and stitching. V48 and V49 retain their separate frozen duplicate-suppression relations. V51 performs no post-append NMS, does not rerun ownership, and does not alter any earlier suppression result. 

Pasted text

Output and cache contract. Preserve the V51 detector result schema exactly: schema version, result kind, profile, v49_retained, candidate_count, predictions, single_axis_parallel_return_recovery_audit, leakage flags, source_only, research_only, and v51_result_sha256. A V51-appended prediction retains the frozen bend_id, endpoints, selection route, score, source hypothesis, and source kind fields. 

Pasted text +1

Store scan hash, point count, cache-byte audit, elapsed time, peak process-tree RSS, invocation event, and primary/replay digests in a separate harness manifest so that runtime measurements do not contaminate the byte-identical semantic artifact. No source point, source cloud, vertex list, face list, or unbounded candidate geometry may be serialized.

Source-free preflight. Before any new-corpus detector invocation:

rerun the frozen 25/25 packet-selector tests and the frozen 45/45 focused/inherited suite;

specifically retain the V51 tests for highest-support deterministic recovery, inactivity with multiple incumbents, rejection of same-family or too-close candidates, and exact candidate-pool permutation identity; 

Pasted text

prove detector-process denial of both Gold37 and new-corpus truth stores;

verify invocation count zero, exact source/config/dependency hashes, compact-cache enforcement, and the 200,000-point cap.

No source-free fixture or test may be added, removed, or changed using anything remembered or derived from Gold37.

New corpus. Assemble the smallest corpus satisfying the stipulated enterprise coverage floor: exactly 30 scans across exactly 15 held-out part families, two scans per family. The neutral corpus custodian must precommit, before case selection, the scanner configuration, calibration, reconstruction and cleaning pipeline, static acquisition-failure criteria, and deterministic sampling rule. Use the first 15 eligible families and the first two eligible scans from each under that rule; expected bend count, apparent difficulty, V51 output, and visual preference are prohibited selection variables.

Corpus membership must be formed after this ruling. A scan is admissible only if its bytes are created after the ruling or were already held under an independent cryptographic seal with access logs proving that no V51 developer or incident-exposed operator had seen its geometry, identity, count, truth, or diagnostic render. A custodian-only hash audit must prove scan and family disjointness from Gold37 and every V16–V51 development or prospective case. The detector receives only opaque case IDs and scan geometry.

Truth and scoring order. Before detector invocation, two independent annotators must produce scan-geometry-only bend-line truth without access to V51 predictions; a separate adjudicator resolves every non-identical annotation set. Seal the final 30-case truth manifest and family map before any prediction is generated.

The scorer may use a newly written corpus-generic manifest reader because the included historical evaluator has hard-coded case plumbing, but that harness must be frozen before corpus assembly and may call only the unchanged matcher semantics:

one-to-one minimum-cost assignment;

axis angle at most 5°;

symmetric transverse offset at most 4 mm;

gold-span overlap at least 0.60;

prediction-span overlap at least 0.60. 

Pasted text +1

Run all 30 primary cases in the opaque order without scoring, human inspection, early stopping, or case-dependent intervention. Then run all 30 replay cases under the identical environment. Seal both result sets, canonical digests, resource manifests, and automatically generated prediction-only renders. Only afterward may the separate read-only scorer open truth and family membership and write the complete aggregate and per-case score in one transaction.

After the score is immutable, generate native-resolution prediction/truth overlays and manually audit all 30 cases. The audit may identify geometric or rendering failure modes, but it may not change truth, waive a metric, alter a prediction, modify the matcher, or trigger another V51 execution.

5. FALSIFICATION GATES

The execution is scientifically valid only if all of the following hold:

Every detector, model, configuration, dependency, test, matcher, renderer, and serializer hash matches the precommit.

The 30 scans and 15 families pass the custodian’s untouched, disjointness, and deterministic-sampling audits.

Neither Gold37 nor new-corpus truth, family mapping, filenames, expected counts, or scoring outputs are accessible to the detector process.

Both frozen test groups pass completely before invocation.

All 30 primary cases and all 30 replay cases complete. Because the required job-success rate is at least 0.995, this corpus requires 30/30.

Every precommitted canonical replay result is byte-identical to its primary result.

Working points are at most 200,000 per case; compact-cache rules pass; forbidden source geometry is never serialized.

Peak process-tree RSS is strictly below 16 GiB and P95 latency is at most 300 seconds.

Native-resolution overlays exist and receive an append-only manual ruling for all 30 cases.

No Gold37-derived predicate, threshold, ordering, exclusion, corpus choice, annotation decision, visual diagnosis, or scorer change is introduced.

The valid semantic transfer gate passes only if all of these hold simultaneously, using exact aggregate integers before decimal rounding:

exact bend-count accuracy at least 27/30;

line precision at least 0.90;

line recall at least 0.90;

recall of every one of the 15 frozen families at least 0.80;

render-admissible marker accuracy at least 0.99 under the precommitted denominator;

deterministic replay exactly 1.0;

all resource, point, cache, and job-success gates pass.

These are the packet’s existing program thresholds applied to the smallest conforming 30-scan/15-family tranche. 

Pasted text

 F1 must be reported from the frozen aggregate TP, FP, and FN counts, but there is no alternative near-pass or compensating-metric rule.

Missing any semantic threshold terminally retires V51. Visual coherence cannot override the score; no family waiver, threshold sweep, predicate correction, reordered ownership, V51b, or post-score rescue is permitted.

A hash mismatch, leakage, failed test, incomplete primary or replay set, replay mismatch, resource violation, scorer change, or missing all-case audit makes the event invalid, not a V51 geometric success or failure. There is no automatic replacement execution: any failure before truth opening requires a new adjudication, while any failure after truth opening permanently consumes both the corpus evaluation and the V51 identity. Passing all gates authorizes only a subsequent research-promotion ruling; runtime/main and production remain closed.

6. EXPECTED FAILURE MODE

The most likely V51-specific systematic error is a false parallel-return append caused by proposal-family multiplicity rather than a second physical bend. A different CB<n> can be another fit to the same transition, the opposite skin, a long rim, or a nearby cylindrical structure; P06’s 27 eligible candidates already show that family inequality does not establish unique physical ownership. 

Pasted text

The exposing artifact is a native-resolution stack of cross-sections perpendicular to the incumbent axis, overlaying the incumbent, appended V51 axis, and their local source support by cylindrical family. A true return produces two persistent, spatially separate transition bands. The predicted failure shows both axes attached to one transition or skin, one axis following a rim, or the appended axis occupying an unsupported offset locus.

7. NEXT STEP AFTER FAILURE

After any valid semantic failure, terminally retire V51 and open one fresh architecture-selection packet using only the sealed new-corpus error taxonomy, with Gold37 and every V51 constant, route, ordering, and predicate permanently closed to rescue tuning.
