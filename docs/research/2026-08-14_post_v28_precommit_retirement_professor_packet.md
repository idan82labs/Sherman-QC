# GPT-5.6 Sol Pro packet — post-V28 precommit retirement

Date: 2026-08-14

## Decision requested

Act as a skeptical professor advising an industrial 3D scan bend-inspection
research program. Ground the ruling in primary research or authoritative
technical literature where relevant, distinguish established methods from
Sherman-specific hypotheses, and select **exactly one** next action that can
materially improve strict zero-shot line-instance precision/recall under the
constraints below.

Do not rescue V28, tune against known answers, propose a threshold sweep, use
part identity, or relabel an old result as prospective. Prefer a quickly
falsifiable action with mandatory native-resolution visual inspection. If no
source-only action is scientifically defensible, state the smallest capability
opening required and why it is necessary. A formal STOP is acceptable only if
you can show that all remaining permitted actions are invalid.

## Binding objective and constraints

The end state is a locked benchmark of at least 30 scans across at least 15
known or near-neighbor families with:

- at least 90% exact bend-count accuracy;
- at least 90% bend-line precision and recall;
- at least 99% render-admissible marker accuracy;
- 100% deterministic replay;
- less than 16 GiB full-process-tree peak RSS;
- at most 300 s P95 latency;
- at least 99.5% job success;
- no major family below 80% recall.

Inference remains scan-only, deterministic CPU, approximately 200,000 bounded
input points, and compact scalar/line/hash caches with no source-point
serialization. Gold, CAD, expected counts, filenames, part/family identity,
good/bad status, benchmark feedback, runtime/main, and production are closed
unless you explicitly justify the smallest scientifically necessary opening.

The causal limit `SCAN_SURFACE_CAUSAL_NONIDENTIFIABILITY_STOP_V21` remains
binding: byte-identical permitted surface observations must produce
byte-identical inference even if hidden manufacturing semantics differ.

## Current retained evidence

The retained V16 full-training result is descriptive only: 132 TP, 60 FP,
44 FN, precision 0.6875, recall 0.75, F1 0.717391. It is not a family-held-out
generalization estimate.

The strict 35-case family-held-out V22 result is 89 TP, 100 FP, 87 FN,
precision 0.470899, recall 0.505682, F1 0.487671, exact count 3/35.

V27 collinear fragment stitching, evaluated on the same strict LOFO predictions,
produced 89 TP, 97 FP, 87 FN, precision 0.478495, recall 0.505682,
F1 0.491713, exact count 3/35. It removed three false positives without an
observed recall loss. V27 is frozen at
`d67db9e952d8e127cebe4c8117fe3d604830a83a084d73c6aeeb8ec812015527`
and remains research-only. It has not met its separate 30-trigger/eight-family
zero-false-merge promotion gate.

V28 paired-return completion is frozen at
`0f0408bb5040d11b3cd6180a4ab46224d720f2a82ff431b6f6b6eff1b06cc9af`.
It had one post-hoc, drawing-confirmed U04 mechanism success and zero triggers
in the historical 35-case LOFO set and 26 recent source pools. Those facts were
hypothesis-generating, not prospective F1 evidence.

## Your previous selected action and its terminal outcome

You authorized exactly one action:
`V28_FROZEN_PROSPECTIVE_PAIRED_RETURN_REPLICATION_1`.

Your ruling required a static precommit audit before any fixture or scan and
said that the action must terminate without a patch if:

- equal candidates lack a deterministic total order;
- construction depends on input iteration or nondeterministic reduction;
- ordinary NMS does not run after V28;
- an existing selected lane is not guaranteed to survive a duplicate collision.

The frozen hashes matched. The audit extracted the V28 rank as absolute
separation error ascending, span descending, and candidate ID ascending. The
candidate-ID tiebreak is explicit.

The action nevertheless failed three precommit conditions:

1. qualifying fragments are accumulated in `all_rows` input order, and the
   frozen V27 fitter consumes component endpoints in index order and anchors
   axis sign to `indices[0]`; canonical fitted-axis bytes are not order- or
   endpoint-reversal invariant;
2. bounded-complex NMS occurs inside upstream `select_unified`, before V27/V28;
3. after V28, `build_result` only sorts and serializes, so no post-V28 NMS or
   old-lane-wins collision rule exists.

Terminal token:
`V28_PRECOMMIT_PIPELINE_CONTRACT_FAILURE_RETIRE`.

No detector, analytic fixture, benchmark case, or real scan was executed. No
visual artifact was generated because the action correctly stopped before the
source-free and native-resolution visual gates. V28 is terminally retired
under its current architecture; no V28b or repair is authorized.

Machine evidence:

- precommit manifest SHA-256:
  `1d6d8fa5da8d7da3fe3e873f35943ebce29e87c6c75284ee246940aee99549bb`;
- audit wrapper SHA-256:
  `17509d4678c398feb3e5d6444e21cbde46a3833f4aec3538b5d28378a28a9393`;
- focused audit tests: four passed;
- detector invoked: false;
- fixtures executed: false;
- real scans executed: false;
- prohibited information used: false.

## Required response

Return these sections:

1. `RULING` — exactly one next action or a formal STOP.
2. `RESEARCH BASIS` — primary/authoritative grounding and the boundary between
   established geometry methods and the Sherman-specific hypothesis.
3. `CAUSAL SCOPE` — exactly what the action may and may not claim.
4. `EXACT SPECIFICATION` — insertion point, deterministic ordering,
   canonicalization, NMS/ownership behavior, resource/cache bounds, and
   prohibited inputs.
5. `FAST FALSIFICATION GATES` — source-free positives, hard negatives,
   permutation/endpoint/rigid-transform tests, clean-process replay, resource
   limits, and required 4096px native visual artifacts.
6. `REAL-SCAN GATE` — if authorized, a single prospective selection and
   source-only visual-adjudication protocol with no replacement input.
7. `RETAIN/RETIRE RULE` — terminal conditions with no post-hoc rescue.

Select a fresh architectural mechanism rather than repairing or renaming V28.
The practical priority is action: test quickly, inspect visually, retire what
fails, and fund only what survives.
