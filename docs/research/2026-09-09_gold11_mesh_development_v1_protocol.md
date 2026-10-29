# Gold11 mesh-workflow development V1 — pre-run protocol

Status: PREPARED FOR SOURCE FREEZE, NOT EXECUTED. Independent pre-run audit of
the exact resulting bundle is required. This document is included in that
exclusive source bundle and must not be edited afterward. It does not reopen
an earlier attempt. Completion status belongs in separate run/checkpoint records.

## Authority and hypothesis

The user's September9 autonomous mandate authorizes a new development comparison
on exactly the consumed11 scans/six CADs/unchanged71 reference lines. No email,
held-out, unrelated data, original/Gold mutation, external deployment, hardware,
shared infrastructure, contacts or new paid services are authorized.

Engineering reason: the retained R0 parser rejects all11 mesh PLY inputs;10
exceed its row cap, and native face/tree bounds reject several actual CADs.
V1 adds byte-validated mesh input, coverage-receipted registration sampling,
full-original-row measurement, bounded whole-part registration, realistic native
inspection windows and a fixed scan-derived surface carrier. The comparison
isolates the native extractor under one common new workflow. It asks whether
the V1 native extractor improves usable finite scan-supported outputs and the
declared geometric line score over the R0 extractor under that same wrapper.

The control is **not** the untouched R0 pipeline or reconstructed historical
incumbent. Both arms use new mesh input, CAD sampling, provisional registration,
full-raw fitting and measured carrier. Only the native extractor changes.
Historical `.4580152672` comparability remains UNESTABLISHED; no result of this
protocol can substantiate an old-score-beaten claim. See the
[historical binding audit](2026-09-09_gold11_historical_comparison_binding.md).

Known pre-run engineering limitation: the three-case actual STEP-to-scan
constructed test returns1/3 at8192 CAD samples, and its separately versioned
32768-sample successor also1/3 under unchanged strict synthetic policies.
The first has an accurate nominal pose but insufficient finite-sample coverage;
the second has an incorrect, rank-deficient nominal refinement. Missing-arc
abstention is retained, but its shifted inner/outer solution is not verified side
identity. These are negative connected evidence, not passed workflow checks.
The physical configuration below differs (including free-circle fitting).
Execution here is an honest all-case diagnostic, not conditional on a favorable
score or a claim that constructed end-to-end accuracy has already been closed.

## Immutable cohort and reference custody

The exact case/path/hash constant is parsed, not executed, from
`scripts/review_gold11_inputs_v1.py`, whose SHA256 must remain
`41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955`.
This binds11 scan hashes and six CAD hashes reviewed September9; no mutable
manifest chooses cases. Each used raw byte snapshot is independently hash-checked
on descriptor-consistent, no-final-symlink bounded reads. Workers receive neutral
snapshot names and geometry bytes, not source names, case/family identities or
reference lines. No hidden case-specific policy or retries are permitted.

Gold container SHA256 is
`4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36`;
selected-record canonical SHA256 is
`d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc`.
All11 selected case records and71 bends must match. Unselected geometry is never
used or emitted. Gold is loaded only after all22 inference attempts are durable
and terminal. It never enters the inference process, pose, datum, carrier or
threshold selection. The current selected snapshot is not proven identical to
the missing historical snapshot.

## Candidate, configuration and execution

`scripts/run_gold11_mesh_development_v1.py` owns exact policy dictionaries and
the named source closure. `--freeze` reads sources only, creates an exclusive
attempt directory, captures source bytes and writes FROZEN.json. It does not read
raw geometry or Gold. No frozen bytes may change after independent audit.
`--execute` requires an audit bound to that exact bundle and refuses an existing
STARTED marker. No resume, arm-only, case-only, retry, dynamic budget or input-path
options exist. A later engineering successor needs a new version, reason,
protocol and independent audit; V1 remains terminal.

Internal inference/scoring entrypoints require a one-use neutral capability
issued by the verified live coordinator. Role, geometry/request hashes, actual
parent/process group, STARTED/audit/source/runtime bindings and private cache
inode are checked. Exclusive issue/claim records prevent replay; leaves do not
receive original case/family names or open the frozen case catalog. This guards
the reviewed workflow, not a hostile local user rewriting source or control files.

The policies are geometry-only and identical across cases/arms. Registration:
50000 maximum actual scan representatives,32768 CAD interior samples;24 proper
PCA starts;2048/8192 optimization points;30 coarse+30 refinement iterations;
trim .85; Huber .25mm; correspondence2mm; inlier plane-p95 .5mm;
scan/CAD coverage .75/.25; observability ratio .005; ambiguity RMS .05mm and
coverage .03; pose class .1mm. These are exploratory algorithm gates, not
manufacturing tolerances or calibrated uncertainty.

CAD triangulation requests .1mm/.2rad approximation, serial, no cleanup or
hidden remeshing retry. Native feature extraction uses the reviewed V1 boundary
policy or the frozen R0 extractor. Local measurement uses the unchanged
free-circle kernel, flange window[2.5,12.5]mm,3mm search margin,.05mm fit residual,
.075mm plane exclusion,.1mm tangency gate. Finite support uses axial gap4mm,
angular gap5degrees, minimum sweep20degrees, minimum local points3. Full original
XYZ rows, not registration samples, enter each bounded local fit. Source-row
support and failures remain retained in full per-arm result files.

Units are `PROVISIONAL_COMMON_MM_NOT_CALIBRATED`: CAD is imported with an explicit
1mm system-unit target; scans have no verified export-unit declaration. No scale
is estimated. Registration is whole-part best-fit, not stable-datum metrology;
it can absorb positional/orientational deformation and bias ROI selection.
Intrinsic radius/angle remain conditional scan-derived fits, not physical truth.

## Fixed budgets and failure handling

One native/geometry worker at a time,22 fixed case/arm attempts in reviewed case
order with R0 thenV1. Per-case wall180s, CPU150s, sampled worker-tree RSS8GiB,
log1MiB, result64MiB; raw scan64MiB/CAD8MiB. Overall wall4200s includes30s
reserved for terminal-record completion; coordinator supervision uses4170s.
The sampled complete-run cap is12GiB including the outer supervisor, coordinator
and all worker descendants, below the repository16GiB ceiling. Coordinator
current and peak RSS are separately limited to4GiB. Workers inherit the
coordinator's dedicated process group for atomic whole-run termination; per-case
monitoring measures the active worker's descendant tree. Missing/malformed
monitor evidence fails closed. Limits are never changed after execution starts.

Scoring runs separately after inference:60s wall,50s CPU,2GiB sampled worker-tree
RSS,192MiB request,32MiB result. Parent scoring projections are capped at8MiB per
case, source/raw preparation reads and writes at20s, coordinator log2MiB,
dispatch16KiB. Global abort retains22 terminal case/arm outcomes and never opens
Gold or retries inference. Source/runtime metadata are verified after scoring as
well as before execution. Runtime version metadata is not binary attestation.
The Python audit hook is a reviewed accidental-I/O boundary, not OS-level
protection against hostile native code.

Every selected case remains in both arms. Hash, parsing, native, registration,
measurement, timeout, memory, output and budget refusals are terminal and retained.
No nominal fallback, failed-row dropping, successful-prefix recovery, fabricated
count or geometry repair is permitted. A malformed emitted prediction invalidates
the entire case's scored prediction set. Out-of-budget cases receive an explicit
zero-prediction abstention, not silent exclusion. Parent retains compact scoring
projections and full-file hashes, not22 duplicated full support payloads in RAM.

## Scoring and decision

Call the unchanged geometric matcher (`evaluate_f1_against_manual_gold.py`):
undirected angle<=5degrees, symmetric transverse offset<=4mm, overlap>=.60 of
**both** reference and prediction spans, one-to-one assignment. Use unchanged
`score_localized_cases`; undefined metrics stay undefined. No changed cost,
threshold, Gold alignment, line deduplication, carrier selection or family split.

Predictions use the measured inner-arc mid-angle generator, constructed from
scan-fitted geometry, transformed back to original scan raw coordinates. Gold
is provisionally compared in its retained raw frame under the common-mm
assumption. Its carrier/frame compatibility is **not verified**. Therefore this
is an exploratory geometric-development score, not manufacturing-valid accuracy.

Report every per-case TP/FP/FN, P/R/F1, exact emitted-support count, family recall,
minimum-family recall, paired deltas, stage failures and resource receipts.
`observed_bend_count` remains unknown; emitted window/instance count is not a
physical observed-bend count. Thus the95% physical count target is unmet even
if emitted-count equality is high. Physical measurement truth and calibrated
uncertainty are absent and separately unmet.

A candidate earns a positive paired geometric result only if both arms have
defined aggregate localized F1, precision, recall and minimum-family recall;
challenger F1 is strictly greater; and challenger precision, recall and
minimum-family recall each do not decrease. Aggregate exact emitted-support-count
accuracy must also not decrease, but it is not the observed-bend-count target.
No rounding is applied to these comparisons. Any undefined required metric makes
the paired gain unestablished, not zero; mixed tradeoffs do not earn promotion
under this protocol. All numeric outputs and refusals remain reported regardless.

This fixed rule identifies mechanism-development evidence only. It does not
automatically promote code. Promotion to the maintained explicit local path also
requires post-run source/math/connected audit, negative controls, no failed-case
concealment and retained reproducible before/after commands/results. Missing
physical truth, reference-convention compatibility and calibrated uncertainty
remain explicit limitations. This does not authorize default activation or floor use.
In particular, the retained connected1/3 failures block promotion of this source
snapshot even if its exploratory physical score is positive. All-abstention and
undefined-score results are valid terminal diagnostic outcomes, never permission
to reopen this attempt. Closing connected accuracy requires separately versioned
engineering evidence and a new independent promotion decision.
If no improvement occurs, retain all failures, diagnose the stage/mechanism and
prepare a justified separately audited successor. Do not tune to case identities.

## Pre-freeze evidence and commands

Runner source SHA256 `100b9dc7dfcb5dfe6a3f4bdd77b03e375ff878278efdf9524d0582299a856837`:
60 source/orchestration tests pass in `gold11_mesh_runner_v1-ptn141_o`;
the separate actual guarded-runtime smoke also passes (61 total checks) in
`gold11_mesh_guard_smoke_v1-xl2fzf3s`. Parent0.414694/0.849296s,
sampled group peak46415872/355991552B; neither had a budget stop or source drift.
The smoke imports NumPy/SciPy/OCP.gp and checks an invented point plus owned
temporary-file cleanup; it does not perform physical inference or establish
connected geometry accuracy. Per-case scorer49 checks pass in
`gold11_mesh_scoring_v1-08auo_vl`. All receipts are under
`docs/research/cad_engine_v1_synthetic_checks/` and retain exact source bindings.

From the repository directory, once each only:

```sh
.venv/bin/python -I -S -B scripts/run_gold11_mesh_development_v1.py --freeze
```

Then obtain the independent audit bound to the resulting canonical bundle hash
in `PRE_RUN_AUDIT.json`. Only after its GREEN decision:

```sh
.venv/bin/python -I -S -B scripts/run_gold11_mesh_development_v1.py --execute
```

These commands are intentionally not a resume/retry interface. Reproducing a
terminal physical evaluation needs an explicitly new frozen audited successor,
with the same inputs/scorer retained and the reason/version recorded.
