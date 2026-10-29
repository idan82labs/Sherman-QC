# Gold11 full11 V10 — prospective consumed-development scoring protocol

Date: 2026-09-13. This protocol is not PRE, source freeze or launch authority.
All prior terminal attempts (V9 included), R0, raw data, references and
historical results stay unchanged. This is a new full11 run, never a
resume/retry of an old attempt.

## Engineering reason and question

V9 completed on all 22 slots with F1 null: 14 slots were killed by the
unchanged 1860 s CPU cap inside surface-V4 candidate refinement with nothing
published (V15 orders every refinement before any publication, so a kill loses
already-passing contexts); the one admitted measurement (CG4BDD control) was
refused as a whole by the support-cover grammar BEFORE any window was
evaluated (`INVALID_SUPPORT_COVER_FEATURES`: the exact-type guard rejected the
kernel-V1 feature objects that both frozen native factories return) — the V9
measurement-side changes were never exercised on physical data. The independent cost postmortem also showed the
diagnosis full-scan query performed twice per candidate and a false completion
budget declaration. V10 changes exactly these things, plus the consumer-chain
bindings they require, and nothing else:

- a NEW native feature bridge reconstructs kernel-V2 feature objects losslessly
  from the frozen factories' kernel-V1 objects (cap 64 = the cover/context
  grammar boundary; wrong class, duplicate id, over-limit or malformed input
  refuses before any costly work; frozen factories untouched);
- registration V5 / initializer V16 add a declared refinement allowance INSIDE
  the unchanged leaf caps: deterministic per-lane start-group and query-row
  quotas within the frozen 48-group schedule (main strata first; one completion
  allocation of query rows — enumeration, binding and index builds charge no
  rows and are covered by the CPU/phase checks), a cooperative CPU guard at
  1500 s of the 1860 s absolute process
  CPU (timing dependent by declaration; realized trace published), chunked
  cooperative full-scan queries with typed stops between chunks, exact bound
  reuse of the diagnosis full-scan query (immutable payload; corrupted retained
  evidence is fatal), typed expiry only (everything else stays fatal),
  transactional completion-lane rollback that restores main groups verbatim,
  attempted/completed row accounting, bounded stage checkpoints and cost
  receipts;
- the registration publishes the evaluated subset explicitly: strict
  `candidate_evaluation_complete` stays FALSE with any omission,
  `evaluated_subset_complete` and a disjoint completed/failed/expired/omitted/
  not_attempted ledger (`candidate_disposition_ledger_v1`) travel with the
  result under the contract
  `EVALUATED_ALLOWANCE_SUBSET_EXPLICIT_DISJOINT_DISPOSITION_LEDGER_V1`, and
  every new consumer (context V3, cover V4, contact V4, projection V5, the V17
  binder) validates it; omitted/expired candidates are UNKNOWN, never rejected;
- the completion budget declaration is corrected (the finite face queries
  charge node visits, triangle pairs, normal pairs and work units; values
  unchanged) and the worker prints bounded stage checkpoints.

The derived δ = 0.03 mm / e = 0.04 mm measurement policy, the 30° support gap,
the completion lane, the kernel-V2 layer selection with its EMPTY inner-face
eligibility (every measured window is a GENERIC conditional contact line), the
1.0 mm pose-admission gate and every sample-model gate form are unchanged from
V9.

The question is the complete same-input contact-localization score of both
fresh V17 arms, retaining every refusal, timeout, omission, expiry and empty
output, the per-window diagnostics, the disposition ledgers and cost receipts
that attribute CPU to lanes and candidates, and the paired before/after against
V9's retained all-22 ledger. Positive geometry on every case is not a
prerequisite; several cases are expected to keep refusing (delta design,
deferred blockers).

## Immutable authority and complete denominator

Use only the eleven consumed Gold11 scans, six distinct CADs and unchanged 71
reference lines bound by the September 9 input review. The sole case authority
is the literal CASES in `scripts/review_gold11_inputs_v1.py`, SHA256
41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955.
Reference-container SHA256 remains
4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36;
selected-record snapshot SHA256 remains
d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc.
No mutable manifest may add or substitute a case. No reference placement,
label, endpoint, frame declaration or count is edited.

Keep the exact review-source case order, control then challenger per case:
22 serial inference slots. Original full scan/CAD bytes are bounded and
hash-verified for each slot before dispatch. Gold is accessed only by the
frozen post-terminal scoring path after all 22 owned results/refusals are
reconstructed; it is never available to inference, pose, datum, candidate/
carrier selection, plane construction, layer selection, allowance decisions or
measurement.

## Candidate, comparator and interpretation

V10 results are COMBINED consumed-development changes (bridge, allowance,
reuse, subset contract, corrected declarations) on top of the unchanged V9
mechanisms and policies and the unchanged provisional 1.0 mm admission gate;
not an isolated mechanism attribution, a speedup measurement, a calibrated
accuracy improvement or a production result. The arm comparison inside V10
isolates only the scan-plane lane and scheduler on the same V17 engine under
the same allowance; timing-dependent CPU-guard stops can add arm differences
that are not mechanism differences and are reported as such. Any admitted pose
remains a provisional development admission; global pose uniqueness is never
established; an omitted or expired candidate is unknown; the derived policy is
pre-declared and never adjusted on Gold or per case.

Runner: `scripts/run_gold11_mesh_development_v10.py` (basis: the V9 runner,
SHA256 4567a0f584e15b75339238b3f50ebb19209f127bfcb3a55e5fa73fd36b4a85b1).
Both fresh arms use native V1 through the bridge V1, mesh offline V17,
initializer V16, registration V5 (surface V4), context V4, stage policies V2,
kernel V2 / context measurement V3 / support cover V4, contact scoring V4 and
schedule projection V5. Control uses scan-plane V7 and schedule V1; challenger
uses composite_v1 scan planes and schedule V2. The complete CONFIG (including
`refinement_allowance`, the deltas recorded in `CONFIG_DELTA_FROM_V9`, the
inherited `MEASUREMENT_POLICY_DECLARATION` and the corrected
`completion_budget_declaration`), certificate caps, arm requests, sampler,
scorer and failure handling are bound to exact source/runtime hashes by the
freeze. The allowance's `cpu_budget_seconds` equals the unchanged leaf CPU cap
and is checked by the constructor seam; no cap is raised. Do not alter
thresholds, quotas, lanes or scheduling in response to Gold, omit a failed
family, infer an observed count from CAD, or substitute a nominal CAD answer.

## Scoring, negative evidence and claims

Use contact scoring V4 (the V3 text recognizing the support-cover V4 / context
V3 grammar, refusing any registration whose evaluation contract is neither
strict nor the explicit subset contract, and counting subset registrations in
the integer diagnostic `subset_registration_count`; scoring scope: GENERIC
conditional contact generator lines, not inner arcs) and the unchanged legacy
one-to-one minimum-cost matcher: axis angle <= 5 degrees, transverse offset
<= 4 mm, Gold-span overlap >= .60 and prediction-span overlap >= .60. Every
carrier is a generic contact line with an unverified material-side hypothesis
(`identity_unresolved_carrier_count`); `inner_face_eligible_carrier_count` is
retained in the grammar and is expected to be zero; no count changes an
endpoint or a match. The F1 itself comes from the unchanged matcher and
metric-aggregation contract (`evaluate_f1_against_manual_gold`,
`f1_physical_evaluation_contract_v2`) that contact V4 imports unchanged; the
V4 scorer schema and diagnostics changed, the metric arithmetic did not.
Preserve null/undefined metrics; do not coerce them to zero or omit their
cases. Report each arm's TP/FP/FN, precision/recall/F1,
exact emitted-support count and localized target-set accuracy, minimum-family
recall, full per-case and per-family dispositions, paired differences,
resources, returned stage evidence, the disposition ledgers, cost receipts and
stage checkpoints of every slot, and the per-window diagnostics and identity
counts of every prediction. Refused/malformed/empty cases contribute zero
predictions and retain all their reference false negatives. Historical F1
.4580152672 is contextual only; do not claim this run beats it. A numeric F1
exists only if the unchanged scorer returns one; null stays null; a low numeric
score is a baseline with its dominant failures reported, not completion.

## Expected retained negative outcomes

Candidates omitted or expired by the row quotas or the CPU guard stay unknown
(published as such); a completion lane that expires at a phase boundary
publishes no completion group; leaves whose main-lane refinement still exceeds
the leaf CPU cap inside one cooperative chunk or one face-index build are
retained kills; windows whose flange residual exceeds 0.03 mm refuse
NOT_PLANAR; single-layer, ambiguous-partition or incompatible-pair windows
measure as generic carriers with unresolved identity or refuse; arcs whose
finite sampling phase leaves too few or too sparse angular stations refuse;
completion candidates may fail the row-level prefilter, refinement or the
admission gate; CG65 has no two-of-three parent; CGE4B, CG64, CG191 and CG4FAF
blockers persist. All are retained zero-prediction outcomes of this
configuration, never grounds for a change after launch.

## Fixed prospective resource budget

Unchanged from V9 (one heavy geometry job, serial leaves, numerical threads
one, repository 16 GiB ceiling, stricter 12 GiB campaign outer limit):

| Resource | Frozen value |
| --- | ---: |
| Inference per-case wall / CPU | 3540 / 1860 seconds |
| Refinement allowance CPU guard (inside the CPU cap) | 1500 seconds |
| Worker sampled tree / coordinator HWM / outer tree | 8 / 4 / 12 GiB |
| Scan / CAD input buffer | 64 / 8 MiB |
| Leaf output log / complete-or-partial result | 1 / 128 MiB |
| Stage-checkpoint stdout lines inside the leaf log | 65536 bytes |
| Retained per-case projection and leaf receipt | 8 MiB each |
| Scorer wall / CPU / sampled tree | 180 / 120 seconds / 2 GiB |
| Complete scorer request | 192 MiB |
| Scorer result / enriched report | 32 MiB each |
| Coordinator log / parent bounded read | 2 MiB / 180 seconds |
| Campaign wall / terminal reserve | 104400 / 2400 seconds |

The allowance allocates work inside these limits and never extends them; a CPU
or wall termination remains a retained negative outcome.

## Exact release and one-use execution boundary

Required in order: independent bounded source reviews of the V10 delta (bridge
V1, registration V5, initializer V16, context V3, cover V4, mesh offline V17,
contact V4, projection V5, runner V10, waiter V10) with their closure deltas
accepted; passing focused tests recorded in the progress ledger with stable
hashes; exclusive `--freeze`; independent non-author exact physical PRE binding
the frozen source/runtime bundle, raw17 declaration and full11 order, unchanged
71 reference hashes, both arm requests, the allowance, scorer, limits, failure
policy and control sizes; verified external archive; fresh no-overlap process
check. Only then, once, through the durable waiter
`scripts/gold11_v10_execute_waiter.py` (its bytes are in the frozen source
inventory), whose launcher first reserves the three outer records exclusively
(`O_EXCL`; any existing record refuses the launch and is never appended to or
truncated) and whose detached waiting parent, admitted only with the
launcher's one-time token, an exclusive lock on the reservation and a
still-pending exit record, runs the exact command below as its direct child
and atomically records the ACTUAL outer return code in
`docs/research/gold11_mesh_development_v10_attempt_outer_exit.json` (beside,
never inside, the attempt directory). A direct `--waiter` invocation refuses.
An unobserved exit status is recorded as unobserved, never inferred from a
final-gate file:

```sh
.venv/bin/python scripts/gold11_v10_execute_waiter.py
# which runs, once: OMP_NUM_THREADS=1 .venv/bin/python -I -S -B -u scripts/run_gold11_mesh_development_v10.py --execute
```

Actual scorer output, enriched report, all 22 terminal records and the complete
final tree must reconcile before any score is accepted. Independent POST then
checks every case/arm, raw/reference/source/runtime bindings, result/refusal/
projection evidence, disposition ledgers and cost receipts, scorer/report
equality, resource receipts and final inventory. On failure preserve it,
diagnose the mechanism and freeze a justified successor; never overwrite or
reopen this attempt.
