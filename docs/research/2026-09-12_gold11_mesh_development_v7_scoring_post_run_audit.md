# Gold11 mesh development V7 — independent scoring POST

Date: 2026-09-12. Reviewer: `context_v3_adversarial`, independently assigned by
the integrating root. Scope is the sole completed retained attempt
`docs/research/gold11_mesh_development_v7_attempt`.

## Decision

**ACCEPT_RETAINED_NEGATIVE_SCORING_LEDGER. No scoring/result/projection binding
blocker found. This is not scientific success, a numeric F1 result, promotion,
or authority to retry.**

Both freshly executed arms retain TP **0**, FP **0**, FN **71**, recall **0.0**,
precision **null**, F1 **null**, and zero emitted support lines. All eleven cases
remain in each arm: eleven refused, zero malformed, zero complete-with-no-carrier.
The challenger has no measured gain. Historical-score comparability remains
unestablished. Physical observed bend counts remain unknown for 11/11 cases in
both arms, not observed zero bends.

This subaudit was performed only after root supplied actual outer exit0
(`6ce55b`) and fresh external whole-writer quiescence (`75df5d`). The separately
owned complete physical custody POST remains responsible for all process,
dispatch, source/runtime/archive, log, retention, timing, and final-inventory
attestations. This document does not replace that audit.

## Exact evidence and read boundary

The frozen physical protocol was read in full. Its 13,957 bytes SHA256
`32c84d3dcc12a3f25024fafebeb65eb242c098eeb4d4a50548570ccfbc67319a`
match the live named protocol and the bundle pin. The retained PRE is 31,934 bytes
SHA256 `7712fe7bc1ddcabfe0b7b16f6ff582658283e5dff9a7687fcb0c21eab1eb96bf`;
its decision is `GREEN_FOR_FROZEN_CONSUMED_DEVELOPMENT_RUN`, with no blocking
findings. Its bundle binding equals the independently recomputed canonical
FROZEN digest
`65c69cf82063358476dd7a6a3ef406b190ab8492baeafe1583d10c7f5f7efc49`.

| Retained artifact | Bytes | SHA256 |
| --- | ---: | --- |
| FROZEN.json | 36954 | `230593dc3a5d6dd10216fcd728e9881f732d07065885a9fcc68b177216882182` |
| INFERENCE_TERMINAL.json | 2062 | `10efe6007b61fd93b9af93fe09185873a82b497492103fe26ea8e12815b942ff` |
| SELECTED_REFERENCES.json | 24751 | `d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc` |
| SCORER_RESULT.json | 37619 | `cecd9c139edbacf7fe46d355a60da00f27e029d4335ce9926f62ee0414e21e7d` |
| SCORER_ARTIFACT_RETENTION.json | 248 | `3f7cd4e5499dfb54655536baa8173945df048bca96e712e1fb821564b8bbd2d6` |
| SCORER_RECEIPT.json | 3210 | `b6da3604c8726830b8850d603cbcd6bb7dbf034a571626e71239c3cc00828432` |
| SCORED.json | 158617 | `473798ad25b59530d576edbc4029cd7f8430299a060d8c436aa0f52fcadbd8c9` |

The principal independent stdlib pass read eight top-level JSON artifacts
(including PRE), all 22 COMPLETED and SCORING_INPUT files, all 21 RESULT files
with a non-null result binding, and eight named frozen source/protocol files.
These 81 retained files total 974,975,316 bytes. Its sorted compact-JSON
path-to-`{bytes,sha256}` map SHA256 is
`33232d637030aa91241f21ba9cbd156712963d21d4e07cfdfdad8a3035c69a67`.
The main pass used no-follow bounded regular/single-link reads and compared
device/inode/mode/link-count/size/mtime/ctime before/after and against the named
path; atime was deliberately excluded. Later narrow metadata checks rebound
each decoded RESULT to its retained receipt hash.

Only the selected retained reference snapshot was decoded. No original catalog,
raw scan/CAD, private coordinator/scorer/leaf cache, other attempt, or undeclared
Gold was read. No target module was imported or executed; no projector,
revalidator, matcher, scorer, inference, fitting, or test suite was rerun. Inline
stdlib JSON/hash/arithmetic checks are independent audit work, not a new target
qualification or benchmark execution.

## Source dependencies

The following named files were independently hashed, matched byte-for-byte
between live and frozen copies, and matched the FROZEN source map. Relevant
projection, reference-selection, request/reconstruction, refusal, aggregation,
and report-enrichment functions were read as source.

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| scripts/run_gold11_mesh_development_v7.py | 169180 | `e910d562e55a62de80f5e448f944c239b5bdf95ef11c6d0b6f0b83b24bfc7a52` |
| scripts/gold11_schedule_projection_v2.py | 14677 | `728920fe179751aa69e271a572d71acf98269a807904953ada0e38801874a3db` |
| scripts/gold11_schedule_projection_v1.py | 17231 | `a34431d167c086daa04b5e69e0c873f69df74fb27afd641148f7a91d4e899245` |
| scripts/gold11_contact_scoring_v2.py | 50413 | `9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff` |
| scripts/gold11_mesh_scoring_v1.py | 13492 | `aa34a2f71bac4edacba41a7e82af0b0a6d4c3d17560fb5e05dab8991ccd0b234` |
| scripts/evaluate_f1_against_manual_gold.py | 13134 | `1e9420558b91f9b136fb32ed012b56c64cdaeecf7599ee1ef9fe336915d75c62` |
| scripts/f1_physical_evaluation_contract_v2.py | 8329 | `28522ee7c63d6886304042a97c736397ea6f043bf77f064cb42efc17a8b1a9b7` |

The eighth named file is the physical protocol pinned above. Full70-source
verification is separately owned: the custody reviewer reports unchanged
2,284,566 source bytes, map
`e50c182e5db3139e9df54e67b10ce0ee70ba804b803d84eca91befeac24fac31`,
runtime `79a084f7e8a3681abeafaad5d4369e04c5734dc674df11ce205475e60af25907`,
and exact PRE/archive agreement. Those broader custody findings are attributed,
not a claim that this reviewer re-audited all runtime/source payloads.

## All22 result and scorer-input bindings

The exact eleven case tokens and both arms match FROZEN, terminal records,
selected references, scorer case dispositions, and report projection bindings.
For each of 22 slots, the independent pass checked the canonical full arm-request
hash, exact retained projection-byte hash, and terminal result hash. For each of
21 bound RESULT files, its complete bytes agree with the leaf receipt, retained
artifact descriptor, wrapper binding, and inner contact binding, including byte
length. Every such wrapper is `ADMITTED`; every inner contact projection is a
`REFUSED` projection of an actual returned `UNAVAILABLE` workflow at
`WHOLE_PART_REGISTRATION`, with no measurements and no predictions. Returned
workflow reasons agree with the contact projection.

The remaining slot is `CG4FAF273093/control`: no usable bound RESULT, with the
exact retained supervisor refusal and empty retained-result artifact list. It
is not replaced with an inferred workflow result or silently excluded. Its
resource/process disposition is audited separately by the custody reviewer.

Requested scan, schedule, and context configurations agree with each arm's
frozen request. All four context-publication parents were checked at their
actual schema paths: workflow, policies, registration evidence, and
`registration.evidence.initialization`. Their implementation identity,
configuration and configuration digest agree. The shared context digest is
`e65c1552d8985238af14fc2b770764aeca35d2e0a0d8ebf21c714fd68bcf460c`.
Initialization budgets and all three scan/schedule publication parents agree
with the frozen requests; control remains schedule v1/scan v7 and challenger
schedule v2/composite scan, both context V3/workflow V14/initializer V13.

The narrow nested-ledger follow-up found no discrepancy. Of the 21 returned
results, 19 publish a COMPLETE schedule: nine have a COMPLETE context helper
and ten an UNAVAILABLE context helper. The two `CGE4B0549E41` results have no
schedule/context helper publication, permissible for these incomplete workflows.
No returned result has a selected measurement context: all reported context
index lists are empty, and registration-array bindings are absent. Present
helpers' schemas, arithmetic declarations, budgets, state/completion flags and
non-authority flags agree with the requested metadata. This is metadata
consistency, not replay of scheduling, raw support, or successful registration.

Independently collecting the exact stored 22 outer projections yields canonical
SHA256 `092118f9e583f59ddb43b17b44f0d437f8a079cb33f6415104d0df197250ea3b`,
identical to INFERENCE_TERMINAL. The full arm-request digest is
`54cbece489ca6631357d212c2cd4a0b2eae3d762970c1340f2dbde7584aa490b`.

Using only retained selected bends, case-family identities, the 21 admitted
inner contact projections plus the exact supervisor refusal, frozen requests,
and reconstructed receipt/hash bindings independently produces a 115,597-byte
canonical scorer request. Its SHA256
`8e940bf54fbd7b079e3156f2a684718747500879a8666dd62642c94d308ba31f`
equals SCORED.scoring_request_sha256; all 22 reconstructed projection bindings
equal the report. This does not read or infer contents of the deleted private
request cache. Nor does it claim independent execution of the qualified
projector/revalidator: the retained bytes, source contract, and actual runner
receipts establish the bounded reconstruction evidence described here.

The source's `score_after_terminal` explicitly reconstructs all22 durable
records and checks terminal counts/outcome/request digests before selected
reference access. The retained terminal has 11 cases, two arms, 22 records and
the matching digests above. Actual ordering/dispatch/quiescence is additionally
covered by the complete custody POST; this audit does not deduce chronology
merely from filesystem timestamps.

## Denominators and arithmetic

The selected snapshot is exactly the frozen SHA256, with eleven unique cases,
71 bend records and six families. Every reference case identity, part/family,
scan identity hash/path and reference ID agrees with FROZEN. Each declared
bend_count equals its retained bend-list length, with unique bend IDs per case.

| Family | Cases | References = FN per arm |
| --- | ---: | ---: |
| 10839000 | 3 | 18 |
| 37361005 | 1 | 4 |
| 38172000 | 4 | 34 |
| 40500000 | 1 | 7 |
| 49001000 | 1 | 5 |
| 49024000 | 1 | 3 |
| Total | 11 | 71 |

Case reference counts in frozen order are
`CG08506AE97D:6, CG191DEC7F81:5, CG1E0BBA2933:10, CG4BDD1371B6:4,
CG4FAF273093:3, CG64C412CC73:7, CG65CD575B1C:6, CG84C8FA12D2:8,
CGE4B0549E41:4, CGEBA9A869AC:8, CGF41F49066A:10`.

For each case in both arms, all retained reference IDs appear in unmatched_gold_ids
in source order, with empty matches/unmatched-prediction IDs. Prediction count,
TP and FP are zero; FN equals that case's complete reference count. Family and
global integer totals agree independently. This empty-prediction arithmetic
does not require running the geometric matcher or interpreting reference
coordinates anew.

The frozen `score_localized_cases` computes precision as TP/(TP+FP), returning
null at zero denominator. Recall is TP/(TP+FN), hence 0/71 = 0.0. F1 is computed
from precision and recall only when both are defined and their sum is nonzero;
therefore it is null here. Replacing it with the alternate direct-count F1
convention would change the frozen policy and is not done. Every case, family
and aggregate retains the corresponding null/zero values. Minimum-family recall
is zero. Both exact emitted-support-count accuracy and exact localized-target-set
accuracy are zero; neither is a physical observed-bend count.

Global and six family paired deltas agree: integer TP/FP/FN deltas zero, recall
and exact-accuracy deltas zero, precision/F1 deltas null. No case/family was
discarded to create a more favorable score.

## Scorer/report equality and interpretation

SCORER_ARTIFACT_RETENTION binds the exact 37,619-byte SCORER_RESULT. The scorer
receipt records exit0, no budget stop, successful derived retention and private
cache cleanup. Its reported wall is 0.22620945796370506 seconds; detailed process
and resource custody remains in the separate audit.

All 34 original scorer top-level fields are canonically identical in SCORED.
There are 26 additional runner enrichment fields, not changed original metrics.
The report's workflow-stage attribution equals the retained per-leaf receipt
diagnostics; declared coordinate frames equal the selected snapshot. Bundle,
scorer-result, request, arm-request and selected-reference bindings agree as
specified above. The retained report keeps provisional frame interpretation,
unknown physical observed count, `production_use_allowed=false`,
`floor_release_allowed=false`, `accepted_registration_claim_allowed=false`,
`historical_score_beaten_claim_allowed=false`, and
`strict_physical_engine_targets_met=false`.

Audit housekeeping: initial read-only inspectors used an incorrect scorer
retention field and assumed a top-level initialization/incorrect arm-request
key. Source inspection corrected those local inspector assumptions; the final
nested-ledger pass uses the actual schema and has no discrepancies. One
intermediate print was truncated and was not treated as complete evidence.
These were not target failures, target retries, source changes, or scoring runs.

Conclusion: accept this complete consumed-development negative ledger, preserve
all refusals and undefined metrics, and join it with the independently owned
physical custody and scientific-diagnosis audits. Any future mechanism change or
physical invocation requires new explicit scope; this POST releases neither.
