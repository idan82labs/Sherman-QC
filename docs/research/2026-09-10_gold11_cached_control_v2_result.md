# Gold11 cached control V2: numerical baseline recovered

Date:2026-09-10. Execution completed once with observed parent exit0. Independent
retained post-audit is GREEN for a new reproducible cached control. The original
run records retain their pending status; the separate
[POST_RUN_AUDIT](gold11_cached_control_v2_attempt/POST_RUN_AUDIT.json) supplies the
review decision without rewriting them. No fresh inference was performed.

Reviewer production_architecture_review independently checked42current/frozen
source files,47scientific artifacts,10controls, all44case dispositions/24family
summaries/four aggregates, exact snapshots and original row/endpoints/order.
There was no fusion/matcher rerun or original-input reopening during post-audit.
POST_RUN_AUDIT SHA256:
`bdc2e3a67559358a035beb35e13de9b6c13dad72a899939efedd1c9a69595f92`.

## Computed result and comparison boundary

The predeclared UNION reports F1 **0.4580152671755725**, exactly the historical
headline number:30TP/30FP/41FN from60retained predicted lines and71references.
Precision0.5; recall30/71=0.4225352112676056; minimum-family recall0.2.
All11cases/sixfamilies remain included; no reconstruction or scorer refusals.
All33per-case A0/CAD/UNION cardinality comparisons equal the historical reports.
Nothing was adjusted to obtain these counts or this numerical agreement.

| Fixed variant | Lines | TP | FP | FN | Precision | Recall | Localized F1 | Minimum-family recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scan-only A0 |33|19|14|52|0.5757575758|0.2676056338|0.3653846154|0.1111111111|
| Gated CAD |44|20|24|51|0.4545454545|0.2816901408|0.3478260870|0|
| Intersection |17|10|7|61|0.5882352941|0.1408450704|0.2272727273|0|
| UNION, predeclared control |60|30|30|41|0.5|0.4225352113|0.4580152672|0.2|

Table decimals are display rounding; the retained JSON has full values and
unchanged undefined per-case/family metrics. All four variants are reported,
not selected after observing scores. UNION exact cached-line-count equality is
1/11; exact localized reference-set equality is0/11. Neither is physical observed
bend count, which remains unknown for all cases.

This establishes numerical reproduction from the recovered current prediction
bytes and exact September9 reference snapshot. It does NOT reconstruct the old
inference invocation/runtime closure or independently attest old reference-frame
history. Historical-engine comparability therefore remains UNESTABLISHED. The
new control is reproducible retained-endpoint geometry under provisional common-mm/
raw-frame/carrier assumptions, not calibration or manufacturing-valid metrology.
It is not new-engine progress or proof that the old score has been beaten.

## Evidence, execution and retention

The [protocol](2026-09-10_gold11_cached_control_v2_protocol.md) incorporates the
unchanged scientific contract; its source/guard and shared-supervisor gates passed
independent pre/post-review before the first source-only freeze. The independent
AI technical [pre-run audit](gold11_cached_control_v2_attempt/PRE_RUN_AUDIT.json)
bound the exact21-file source bundle, fixed22prediction declarations, reference
snapshot and limits. It is not a human manufacturing signature.

Executed once, after that audit:

`.venv/bin/python -I -S -B scripts/run_gold11_cached_control_v2.py --execute`

This exact attempt is terminal: do not invoke it again. Reproducibility is provided
by retained input/source/config/scorer bytes and commands, not an in-place rerun.
Any future repeated execution needs a documented engineering purpose and new
independently frozen one-use protocol; original data/results remain untouched.

All22input files passed declared-byte preflight before prediction interpretation
or catalog access. The fixed catalog passed its hash check; only the selected
11case/71reference snapshot was projected. No raw scan/CAD, held-out, email or
unrelated geometry was opened. All22owned input snapshots,22check records,
preflight, selected references and complete report survive:47scientific artifacts,
204932bytes total. Complete report111229bytes;137logbytes match stored/observed
counts. No scientific truncation, omitted case or successful-prefix scoring.

No runtime/budget/source/control/cleanup error was reported. Worker was reaped,
owned group confirmed empty, no changed/foreign identity was observed. Five child
RSS samples; peak worker tree81461248B and simultaneous parent-plus-tree116031488B.
Worker CPU0.284271s; final-gate observation parent wall0.34712441731244326s,
aggregate CPU0.4314929999999999s, parent high-water34783232B and reaped-child
high-water81887232B. These are distinct resource observations, not a continuous
allocation or persistence guarantee. Final-gate measurements precede its own
publication and were rechecked before observed exit0, as the protocol declares.

| Retained record | SHA256 |
| --- | --- |
| FROZEN | `f6129f40e48f6c1c95712bfdaac4433903b2fa7ef3d1f529ede37b88ed2b7213` |
| PRE_RUN_AUDIT | `089550b1c9b51a1f0d596754a8562a34a64fd2cf973ba206c386f35de6f1c94b` |
| STARTED | `d1b9aa12703bd0fa17a229f14bf1c59ab2f19d3f942a1094abdc503ec0f8c87a` |
| COMPLETED | `b572ec7e39c15f612a8a0137efc9b8fca89fee17bd980ed89cd49975d7e05a5e` |
| FINAL_GATE | `f925e267c53644c4b241a79e56238ccb8e95f402c1d6bf90ee8eb8f125af538f` |
| OUTPUT | `fd760aabc3eb631d165d1bba5d18c36034c5bc1a34f5796778f4e68371c32833` |
| SCORER_REPORT | `6307f41ef3677a822e69969ba8da03b3ce5664375fbfc6ec642f52de6cb956b3` |
| INPUT_PREFLIGHT | `4ca1c32cdecd1f5a6fb2ec6ec89ce6ed2cc51591ac8349eb3bf786509dc0f575` |
| SELECTED_REFERENCES | `d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc` |

## What this says about the next engine work

Relative to retained A0, UNION has net11moreTP and16moreFP: the CAD contribution
improves recall but also adds substantial false-positive cost. This is an aggregate
comparison, not a new row-level attribution claim or permission to choose Gold-
matched predictions. Intersection is not a solution: it loses20of UNION's30TP.

For the fixed71reference denominator, a defined future F1 exceeds60/131 iff
`101*TP > 30*FP + 2130`. One additional TP at30FP gives62/132=0.469697; one fewer
FP at30TP gives60/130=0.461538. Such small numerical gains would not satisfy the
locked offline targets; P/R/F1>=0.90 requires at least64TP and at most7FP at64TP,
plus the separate family/count/measurement conditions. No Gold-based filtering,
case-identity tuning or threshold change follows from this arithmetic.

The immediate next scientific run is the separately frozen fresh V8/V10 all11
comparison, preserving the physical configuration and every refusal. It must
show which previously blocked cases reach registration/scan measurement and
whether emitted lines improve on this authenticated cached control under the
same endpoint-scoring assumptions. Two-plane/context proposals and native/work
coverage repairs remain parallel engine work, not a reason to postpone all11.
No new physical measurement truth, calibrated uncertainty, production integration
or factory-floor release is established here.
