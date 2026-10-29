# Gold11 V10 — bounded runner/declaration source audit

Date: 2026-09-13 15:44 UTC. Reviewer: Codex `v9_post_score_ledger`.
Independent source-only seam; Fable remains the sole candidate editor.

## Decision at the reviewed snapshot

`GREEN_BOUNDED_RUNNER_DECLARATIONS_WITH_NONBLOCKING_CORRECTIONS`.

No blocking contradiction was found between the V10 runner, waiter and
protocol concerning immutable cohort/reference declarations, 22 retained
case/arm outcomes, fixed outer budgets, terminality, same-engine paired-arm
interpretation, null metrics, or independent PRE/POST requirements.

Three small delta-specific wording/matching corrections are recorded below.
They do not require a new research phase or alter the engineering decision
rule. This audit is **not** a physical PRE, source freeze, launch authority,
test-pass receipt, full import-closure audit, or review of the geometry and
evaluated-subset implementation. Candidate files were still being finalized;
the decision applies only to the pinned bytes below and needs an append-only
delta note if the relevant bytes change.

## Scope and evidence actually inspected

Read canonical `AGENTS.md` and the latest `docs/CURRENT_STATUS.md` checkpoint
(V9 terminal/POST accepted), the complete V10 protocol and delta design, the
runner's V9-to-V10 text diff plus relevant unchanged release/scoring sections,
the full V10 waiter and its V9 diff, and both named source/waiter test files.
The working-tree V9 runner/waiter hashes match the previously established
frozen V9 source hashes; no attempt directory was accessed for this comparison.

Only static text reads, diffs, filename discovery in `tests/`, timestamps and
source-file hashes were used. No runner, freeze, waiter, scorer, projection,
test suite, geometry, source-import helper or physical process was executed.
No raw input, Gold/reference content, attempt/control file, opaque cache,
email part, held-out or unrelated dataset was read, hashed or modified.

## Pinned source snapshot

Paths are relative to the repository root. These hashes were checked at the
start and again at 15:44 UTC (the first four remained unchanged during review).

| File | SHA256 |
| --- | --- |
| `scripts/run_gold11_mesh_development_v10.py` | `163c46658bf2ebe3aa258866d3e7ef431bc9a22974d499b79de10cbe7753db71` |
| `scripts/gold11_v10_execute_waiter.py` | `c2648778195b34ee39b45e217c8b566d0998e4d721634866501305ad6b4952fc` |
| `docs/research/2026-09-13_gold11_mesh_development_v10_protocol.md` | `1d6ab2c9dab27b23545dfa4f8e0ac6e9a2c203e97c485763040b24ce894364e0` |
| `docs/research/2026-09-13_gold11_mesh_development_v10_delta_design.md` | `6476ed7fde1a6f86d9e22fd2e530d1728965b8cf40378cda6ef284bb491241f7` |
| `tests/test_gold11_mesh_development_v10_source.py` | `7116db3500a9541b133c04dd9e3f57ae7e188060bd466df2e3b3160d87e6e37e` |
| `tests/test_gold11_v10_execute_waiter.py` | `9a1542b378118dfec12b245fb5b87ea9b4c758be942e87ddb170e9206e855d37` |
| V9 runner preservation basis | `4567a0f584e15b75339238b3f50ebb19209f127bfcb3a55e5fa73fd36b4a85b1` |
| V9 waiter preservation basis | `85913855908ac3a965cb37d7fdd36a8c3464aabd05621184ecd9b4b6e8107688` |

## Confirmed declaration consistency

- **Comparator:** both arms are V17 with native bridge V1, initializer V16,
  registration V5, kernel V2, context measurement V3, cover V4, contact V4 and
  projection V5. Control remains scan-plane V7/schedule V1; challenger remains
  composite/schedule V2. The protocol, design and enriched-report declaration
  identify the V9-to-V10 change as combined engineering changes, not an
  isolated speedup or calibrated accuracy improvement. The inner paired
  comparison is not old engine versus new engine. Timing-dependent CPU-guard
  stopping is explicitly separated from deterministic group/row quotas.
- **Preserved cohort:** runner constants retain review-source hash
  `41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955`,
  reference-container hash
  `4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36`,
  and selected-reference hash
  `d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc`.
  The unchanged case-source parser requires 11 distinct scan hashes and six
  CAD hashes; coordinator loops all cases and both arms in their fixed order.
  These are source declarations, not new raw/reference verification.
- **Scoring boundary:** the runner retains the all-22 durable reconstruction
  gate before opening the reference catalog; complete-map checks remain
  11 cases / two arms / 22 terminal records and 71 references. Whole-case
  refusals remain zero predictions with all reference false negatives, not
  dropped cases. Original scorer fields must still equal enriched-report
  fields. V10 declares `scorer_change=True` with consumer-contract/schema and
  diagnostic scope; it does not declare a changed matcher, threshold or
  arithmetic convention. Null remains null; a low numeric score would be a
  baseline, not roadmap completion.
- **Budgets:** the `LIMITS` assignment is unchanged in the runner diff:
  3540 s leaf wall, 1860 s leaf CPU, 8 GiB worker / 4 GiB coordinator /
  12 GiB outer RSS, 1 MiB leaf log, 128 MiB result, 8 MiB compact projection,
  180 s scorer wall / 120 s CPU / 2 GiB RSS, 192 MiB scorer request,
  32 MiB scorer result, 104400 s campaign / 2400 s terminal reserve.
  The new 1500 s absolute-process CPU guard is inside the unchanged leaf cap;
  360 s headroom is explicitly not a completion guarantee. Quotas are
  12 main groups / four completion groups, 6,000,000 / 3,000,000 query rows,
  cooperative chunks at most 32,768 rows. The declarations say completion
  receives one lane allocation, not a fresh allocation per candidate.
- **Checkpoint capacity:** 524288 sample bytes + 4096 sampler receipt bytes +
  65536 checkpoint bytes + 128 cap-notice reserve + 454528 other-output bytes
  sum to the unchanged 1048576-byte whole-leaf cap. The runner bounds emitted
  checkpoint lines and emits at most one cap notice; checkpoints are explicitly
  diagnostic stdout, never a result, measurement, durability guarantee or
  success claim. The unchanged whole-log cap remains the enforcement boundary.
- **Terminality and audit release:** V10 uses fresh paths, schema identities,
  private namespaces and waiter environment keys. V9 is a hash-bound
  preservation basis, not a runtime retry. The source/protocol preserve
  exclusive freeze, non-author exact physical PRE, external archive,
  no-overlap check, one waiter-launched execute, actual outer exit observation
  and independent POST. A failed or missing parent observation stays
  unobserved; a launch print is not a score. Existing terminal attempts and
  failed selected cases are not reopened or erased.
- **Claim limits:** generic conditional contact geometry only; CAD does not
  supply the measured answer or observed physical count; units/common frame,
  pose admission and uncertainty remain provisional/uncalibrated. Omitted or
  expired candidates stay unknown rather than rejected. Historical
  0.4580152672 comparability remains unestablished; no historical-beaten,
  production or floor-release claim is authorized by this run.

## Concrete nonblocking delta corrections

### D1 — Allowance-check timing is misstated in the design

`..._delta_design.md:27` says an allowance mismatch refuses "before any leaf".
The only runtime call of `workflow_entry_and_policies` is in `worker()`
(`run_gold11_mesh_development_v10.py:1658`), after leaf authorization and
`initialize_worker`, but before that worker reads scan/CAD buffers and invokes
the public workflow. The equality check is at runner lines 1535–1538.

Use "inside the authorized worker, before input-buffer reads and workflow
dispatch." No code change is required to preserve fail-closed inference;
this is a correction to where the existing check occurs, not a proposal for
another launch gate.

### D2 — Numeric-F1 wording should name the unchanged metric contract

`..._protocol.md:135` says "unchanged scorer", although the versioned scorer
grammar intentionally changes and the runner explicitly declares
`scorer_change=True`. `..._delta_design.md:123` says the matcher returns F1;
F1 is an aggregation output, not the endpoint matcher's return value.

Use "a numeric F1 exists only if the frozen, unchanged metric-aggregation
contract returns one; null stays null." The surrounding protocol already
describes the exact contact-V4 consumer delta and preserves arithmetic, so
this wording issue does not imply a scoring-rule change or block the design.

### D3 — Waiter private-namespace overlap alternative is single-digit

`gold11_v10_execute_waiter.py:36` retains `_gold11_v\d_exact_scoring`, which
matches the V9 namespace but not the new `_gold11_v10_exact_scoring` namespace
named in the V10 runner. Widen that alternative to `_gold11_v\d+_exact_scoring`
and add a string-only V10 namespace regression when touching the waiter.

The real authorized V10 runner and scorer-child commands still match the
separate `run_gold11_` alternative, so this does not defeat the normal V10
no-overlap check. It is a small version-rollover coverage inconsistency, not
evidence that two physical benchmark jobs were launched or an additional
physical readiness gate.

## Test-source review and limits

The source test covers predecessor hash binding, declared deltas, unchanged
CONFIG/LIMITS outside those deltas, arm identities/requests, module-closure
presence, real constructor binding, an in-memory freeze/verify fixture and
checkpoint byte accounting. The waiter test uses temporary trees/fake children
for exclusive reservations, duplicates, wrong/missing tokens, advisory locks,
overlap/pristine-tree refusal, observed return codes and unobserved failures.
These tests were **read, not run** by this reviewer; no passing result is
inferred from their existence. The root/Fable must use actual stable-hash
receipts from their separately owned test work.

The review does not independently validate the bridge's geometry, query reuse,
CPU metering, typed subset partition or each consumer's validator. Those are
separate bounded source seams. It adds no dataset access or execution
authority. Fable can close D1–D3 with narrowly versioned source/doc edits;
the integrating root can append their exact delta acceptance, then retain the
already required freeze/PRE/one-shot/POST sequence.

## Handoff note — 2026-09-13 15:47 UTC

Root confirmed D1–D3 were queued to Fable; the author suite was still running,
with the narrow documentation/waiter changes to follow. This report does not
claim those edits or tests have completed. D3's source citation above should
read waiter **line 37**, not line 36, at the pinned hash.

Root also identified a separate wording correction in the protocol/design
intro and runner's bridge rationale: V9's invalid-feature validation refused
the whole cover before any individual window was evaluated. The wording
"every window refused" must not imply window-by-window physical measurement
occurred. That correction was queued with these findings; it does not change
the preserved negative ledger or add a new run gate.

## Final closure delta acceptance — 2026-09-13 16:00 UTC

Decision: `ACCEPTED_BOUNDED_RUNNER_DECLARATION_DELTA_NO_BLOCKING_FINDINGS`.
This supersedes the pending-correction status for the narrow final changes,
not the scope limitations above. It is not physical PRE or launch authority.

Read the complete 123-line author closure
`2026-09-13_fable51_v10_review_response.md`, SHA256
`ff6c94daa686b72335562888f0594a9218d3d369c103e94b41e4cd6c71b49e5b`,
and the exact final D1–D3/whole-cover/row-quota declaration and test edits.
Independently hashed these final files and matched the closure's records:

| File | Final SHA256 |
| --- | --- |
| `scripts/run_gold11_mesh_development_v10.py` | `f7fd083f8b74a22e7fd1afe872ae8b9b2bfbc94f4b7024fc8c2b46494208da92` |
| `scripts/gold11_v10_execute_waiter.py` | `e7ade783ea5adae1d2a97f9d1696913dd614120209a8b875d97147ed6b1282bc` |
| `docs/research/2026-09-13_gold11_mesh_development_v10_protocol.md` | `e59aff47a8312bfce44139a0a1977a33dff4a0a86039502b8c34fe5f1f27889b` |
| `docs/research/2026-09-13_gold11_mesh_development_v10_delta_design.md` | `0a68d42ac50264cff41189b67960a6f42d4d282c761c227bee11989b2e23d23c` |
| `tests/test_gold11_v10_execute_waiter.py` | `0df6e722723441c0b4b13e795a51479cc1cae50bfd31f8b3a988856e60649cf9` |

- **D1 closed.** The design now locates the allowance check inside the
  authorized worker, after authorization/initialization and before input
  buffer reads and workflow dispatch. It separately distinguishes the inert
  constructor test from a launched leaf.
- **D3 closed.** The waiter uses `_gold11_v\d+_exact_scoring`. The added
  string-only regression covers V9, V10, the V10 runner command and an
  unrelated-shell negative. The normal exclusive-reservation/waiting-parent
  path is not changed by this regex correction.
- **D2 substantively clarified, with one harmless residual shorthand.** The
  protocol explicitly names the unchanged matcher/metric-aggregation modules
  and says contact V4's schema/diagnostics changed but metric arithmetic did
  not. The design names the same contract and preserves null propagation.
  Protocol line 141 still says "unchanged scorer returns one"; read in
  isolation that is imprecise, but the adjacent explicit qualification
  resolves the intended meaning. This remaining wording is not a scoring
  change or blocker and does not warrant another readiness gate. It should
  not be described as a literally removed phrase.
- **Whole-cover wording closed.** Runner bridge rationale and protocol now
  say the V9 support cover refused as a whole before any window was evaluated.
  They no longer imply per-window physical measurement was attempted.
- **Row-quota scope reconciled.** Runner `allowance_interaction`, protocol and
  design now explicitly distinguish query rows (screens/queries charged
  before work, complete only after a complete answer) from enumeration,
  binding and index builds, which charge no rows and use CPU/phase checks.
  This accepts the declaration consistency, not a new implementation proof
  of those checks.

The author closure reports the final 13-suite run at these hashes as
885 passed / one inherited skip / zero failed, pytest exit zero, from
15:52:07 to 15:55:11 UTC. This reviewer did not rerun tests or inspect their
job artifacts; that is an attributed author receipt, not an independent
test execution. Root owns the actual constructor-to-workflow seam review.

No broad source audit, imports, execution, raw/Gold/attempt/control/cache
access or candidate edits were performed in this final delta review. The
original all-22/71, immutable-budget, terminality, same-engine paired-arm and
PRE/POST conclusions remain intact within the previously declared scope.
