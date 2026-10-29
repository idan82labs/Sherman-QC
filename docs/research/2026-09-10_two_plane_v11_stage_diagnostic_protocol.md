# Two-plane V11 stage diagnostic: prospective protocol

Date: 2026-09-10. Purpose: distinguish missing useful pose proposals from finite
support rejection in the retained V11 noisy50-row positive failure. This is a
new diagnostic experiment, not a retry of the original qualification, engine
repair, workflow/native promotion, or Gold11 evaluation. The original terminal
`plane_initialization_v11-eyeny0mn` remains40PASS/1FAIL and is never overwritten.

## Authority and engineering reason

The standing local-engine mandate authorizes synthetic diagnostics and audits.
The newest user request is a strategic failure-pattern/architecture review and
Astra Pro consultation. That model choice is pending; no substitute consultation
or redesign is inferred. This bounded diagnostic supplies missing evidence for
that review without changing the engine or accessing any physical input.

The original failed gate retained only its traceback. It established completed
finite prefiltering/no start, but not whether any correct pose was generated,
which support was rejected or which ambiguity flag mattered. The successor adds
pass-through stage capture to the same source-bound clean/noisy cases and keeps
their original post-inference oracle. It cannot turn a failed positive into a
qualification pass; recording success is a separate outcome.

## Exact fixed science

- Candidate `backend/cad_guided_plane_initialization_v11.py`, SHA256
  `31bc1b9c3fe58b1befd7432134ce0640fa9243e23f51b51c28ca4ae72aa7610e`.
- Original fixture/oracle `tests/test_cad_guided_plane_initialization_v11.py`,
  SHA256`16a646404b4cf35975956b717a9143b52d8bca1e7ec5467f40096aefbc0edf1e`.
- Original inherited helpers `tests/test_cad_guided_plane_initialization_v10.py`,
  SHA256`4882b701af3806fd03ddde9bb0b1af823bd0124e859e6655207f683db1563eb0`.
- Extract the original `_two_rectangles` function and its four constants by AST;
  preserve original50raw rows,64scan triangles,4CAD triangles, two face IDs,
  clean then moment-balanced <=.002mm normal-noise case, fixed proper pose.
- Exactly one public `prefilter_plane_initializations_v11` call per case; original
  .05mm per-row declaration, original `_policy(50)`, original common-mm assumption,
  `PlaneSearchBudgetsV6(max_bank_planes=2)`, unchanged default remaining policies.
  No additional inference, known-pose proposal injection, endpoint changes,
  triangulation changes, support inset, alternative thresholds or selected retry.
- Keep every original14post-call statement in original order, evaluated after
  that same answer. The1e-7mm true-start ceiling and all support assertions stay.
  Store the first exception as FAIL, not a successful abstention. Post-evaluation
  pose-action errors may be computed for all generated/retained groups, but never
  feed back into inference or change which candidates are queried/retained.

## Diagnostic capture and completeness

The different-author test is
`tests/test_cad_guided_two_plane_v11_stage_diagnostic.py`. It preserves full input
arrays, digests, preparation context, generated slots and returned public answer.
Pass-through observers record actual face-box and finite-query arguments/results
without recomputing geometry, modifying arguments or replacing return objects.
Record attempted/raised calls and reconcile with runtime attempted/completed
counters. A missing primitive result after an exception is explicitly incomplete,
not a claim of complete capture. Recorder errors must not change the scientific
return value; they do fail the diagnostic-recording ledger.

Retain every selected case including failure, rejection and absence. Neither
clean nor noisy may be removed if recording fails. Partial evidence stays partial;
do not reconstruct a missing query by a second call. An exception before a full
case artifact exists remains an explicit UNKNOWN_IF_NOT_DURABLY_RETAINED case
disposition in bounded stdout. The POST inventories and hashes all actual files,
including incomplete publication remnants, and refuses a successful-prefix claim.

No raw scan/CAD/Gold, retained physical predictions, email, held-out or unrelated
dataset is accessed. No OCP/Open3D/native geometry path, server/DB import, network,
child experiment, API model call or hardware. NumPy arithmetic is limited to the
invented controls. The source-only route must not import the geometry candidate.

## Budgets fixed before either route executes

Each route independently receives60s monitored child wall,45s worker CPU,
1,073,741,824B sampled worker-group RSS and1,048,576B stdout/stderr. All numeric
thread settings remain1. Run serially; no concurrent heavy process.

The recording route owns exactly two case slots,16,777,216B maximum serialized
JSON per case,33,554,432B total. The test constructs only those fresh fixed names
under its parent's new ARTIFACTS directory; existing names must refuse overwrite.
Transport allows at most1,000,000tree nodes and depth64; arrays <=1,000,000values;
each box/query channel <=128observations and<=3,200rows per case. These are
prospective diagnostic limits, not alterations of the engine's fixed budgets.
Overflow invalidates recording, not grounds to truncate or raise the cap.

The existing harness has sampled, not hard, RSS enforcement. Its monitored wall
starts before worker launch; source hashing, initial/terminal receipt writes and
temporary-cache cleanup lie outside that stopwatch. Reaping is inherited and
not a hard-bounded cleanup proof. These known synthetic-harness limitations are
not broadened into production-supervisor qualification. The array-only test
does not launch descendants or access physical data. Any unexpected stop/error
is retained and blocks the diagnostic interpretation/release.

## Source closure and release sequence

Root archived the original harness exactly at
`docs/research/source_snapshots/two_plane_stage_pre_5675caa5/check_cad_guided_measurement.py`,
SHA256`5675caa55fac23507c4d818cffb311af44020d0f71a0940acc30927e26545552`.
New harness routes add only dispatch, fresh artifact destination, exact source
bindings and inherited budget selection; no existing route is run or requalified.

Both routes freeze the same48distinct files: original45`V11_FOCUSED_SOURCES`,
new diagnostic test, this protocol and current harness. The independent PRE
must enumerate exact paths/hashes, verify closure and prospective test counts,
read original constructors/oracle and changed observer/dispatch/publication paths.
The PRE is the final binding of current diagnostic/harness bytes; no edits after
release or between its two routes. Before/after hashes must match its table.

1. Independent exact PRE GREEN releases first source-only execution:

   `.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite two_plane_v11_stage_diagnostic_source`

   All declared source/mock outcomes must pass with exactly the one intentional
   opt-in recording skip; no stop/source drift or unexpected skips. This does
   not execute geometry and is not evidence of a successful noisy positive.
2. Independent retained POST GREEN for that source attempt, same frozen bytes,
   releases first recording execution:

   `.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite two_plane_v11_stage_diagnostic`

   Source/mock checks plus one two-case recording test; no skip. Both original
   positives receive separate PASS/FAIL/UNKNOWN dispositions. Pytest success
   indicates recording completeness only, not two positive engine outcomes.
3. Independent retained POST checks all files, exact source closure, full array
   digests, input count/policy identity, observer accounting, original oracle,
   resource observations, both case artifacts/dispositions and no hidden retry.
   Interpret only what survived. No workflow/native/Gold11 route is released.

Fresh unique attempt directories are generated by the harness, never an old
attempt. Every first source or recording execution is terminal regardless of
outcome. No repairing/overwriting a spent attempt, cap changes or silent retries.

## Predeclared diagnostic questions

For each original case, report generation and support separately:

1. Did preparation choose the intended two complete25-row supports? If not,
   report actual row identities and selection evidence without reselecting them.
2. Was any generated pose within the unchanged oracle's1e-7mm action-error
   ceiling? This is truth used by evaluation only, not a new pose injected into
   inference. Report all generated pose errors, not only a favorable one.
3. For generated true-pose candidates, which assigned support gates rejected,
   and were rows definitely outside, distance-unresolved, normal-ambiguous,
   closest-point-ambiguous or uncertainty-excluded? Categories may overlap.
4. Does the rejected support coincide with the proposed coplanar-diagonal seam
   mechanism? Distinguish observed flags from source-based causal interpretation.
   This diagnostic alone cannot prove invariance under retessellation; that
   would require a separately frozen falsifier without changing this fixture.
5. Did the original clean/noisy positive statuses reproduce? If not, preserve
   the discrepancy and investigate, without replacing the old negative or
   claiming a fixed engine (the candidate was unchanged).

Decision: the result can localize a mechanism and inform the requested strategic
consultation. It cannot establish improved F1, global pose uniqueness, physical
measurement accuracy, calibrated error, a datum, observed bend count, a tolerance
decision, production integration or floor readiness.
