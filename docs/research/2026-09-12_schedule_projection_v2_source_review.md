# Schedule/context projection V2: independent source review

Date: 2026-09-12. Scope: new pure transport binding and invented-byte test
authoring only. This document is not a PRE, test result, execution receipt,
physical-input release, engine qualification, or scoring/promotion decision.
Frozen schedule V1, contact scorer V2, initializer V12, workflow V13, and all
terminal attempts remain unchanged by this reviewer.

## Reviewed boundary

The candidate is `scripts/gold11_schedule_projection_v2.py`. Its complete source
and complete immediate schedule/contact projection implementations were read.
The review also inspected the named legacy import seams, inert test package,
ordinary V1/contact test constructors, the context V1/V2/V3 policy declarations,
and the new initializer V13/workflow V14 implementation and binding contract
from the preceding integration task. No candidate or test module was imported,
collected, compiled, or executed. No raw, Gold, private, physical, output,
manifest, control, authorization, or terminal-attempt artifact was accessed.

The candidate composes the frozen V1 schedule projector, which composes the
frozen V2 contact projector, using the same original supplied RESULT bytes.
It does not rewrite a workflow schema, clone a function namespace, monkeypatch
a backend, recompute geometry, or convert a contact-sector generator into an
inner-arc carrier. The workflow transport schema remains V13 for the unchanged
consumer. A separately mandatory workflow implementation identifies V14;
initializer identity and context V3 configuration distinguish its semantics.

The declared successor comparator uses context V3 in both arms. The intended
control is scan-plane v7/schedule v1; the intended challenger is composite v1/
schedule v2. Two additional cross-combinations in the tests check orthogonality,
not extra physical comparison arms or a historical paired-control claim.

## Closed request and publication contract

`_context_request` requires exactly twelve configuration keys: mode,
implementation, schema_version, budgets, and the eight explicit arithmetic,
scope, chunking, guard, membership, and memory declarations. The independently
declared budget map has 23 exact integer values with maxima matching effective
context V3 policy. Booleans and floats are not integer aliases; chunk_rows must
be at least three; max_extrema_queries cannot exceed three times the requested
max_signed_families. Valid smaller requested values remain bound rather than
being replaced by defaults. An invalid caller request raises before case
attribution; malformed returned evidence yields a whole-result refusal.

The configuration SHA256 is calculated from sorted-key compact JSON with
allow_nan=False and the default ensure_ascii=True, encoded as UTF-8. This
matches the initializer's canonical program for these ASCII declarations.
The maximum-size configuration is independently bounded by the existing
32,768-byte minimum wrapper budget during digesting; the tests use their own
ordinary JSON serializer rather than the candidate's digest helper.

Completed workflows require matching configuration and digest plus initializer
identity at four locations: workflow top level, policies, registration evidence,
and nested initialization. Unknown context-prefixed publication keys are
rejected; the actual initialization.two_plane_context helper is the sole
intentional exception. Every selected array-binding record requires all four
context identity/mode/implementation/digest keys, including triplet-origin
contexts and the empty ordered-pair schedule path. The frozen schedule wrapper
continues to bind the selected-context denominator and its schedule metadata.

Early explicit refusals may precede configuration publication. Their output
separately retains the caller's requested configuration and a false presence
map; it does not fabricate a returned configuration, selected-context count,
or successful helper report. Any configuration that is present remains fully
checked. Mandatory V14 workflow implementation is checked even on refusal.

## Findings and fixes coordinated before freeze

1. The initial candidate required every present context helper to have V3
   schema/arithmetic metadata. Actual initializer V13 uses an exact three-key
   NOT_CALLED_EMPTY_DECLARED_SCHEDULE sentinel when its completed declared
   schedule has zero ordered matches. That valid triplet-only path would have
   been falsely rejected. Root fixed the candidate to accept only that exact
   typed sentinel when the schedule is complete, selected-policy completion is
   True, scheduled_matches is a typed empty list, usable_schedule is False,
   pair_lane_enabled is True, and both ordered_pair_matches and
   pair_slot_reservation are exact integer zero. This is not a broad missing
   helper allowance or an implicit disabled pair lane.

2. The initial candidate allowed a completed workflow with a missing helper
   or a noncomplete helper lacking budgets, even when a nonempty pair schedule
   was reported. Root fixed the completed path: a nonsentinel helper must be
   V3, COMPLETE, selected-policy complete, carry matching budgets, retain all
   three false authority flags, and accompany a typed nonempty schedule. A
   completed workflow without a helper now refuses. No returned measurement
   or diagnostic prefix survives failure.

3. An early-helper issue was reported to root: outside a completed
   workflow, the initial nonsentinel branch accepted an arbitrary non-COMPLETE
   helper status and contradictory authority flags when budgets are omitted.
   Real V3 helper refusals report UNAVAILABLE, selected-policy False,
   usable_proposals False, and false authority flags. Conversely, a later
   workflow refusal can legitimately retain an already COMPLETE helper with
   budgets. Root applied the requested fix: helper status is closed to COMPLETE
   or UNAVAILABLE; selected-policy completion is the exact corresponding bool;
   usable_proposals is a bool and must be False for UNAVAILABLE; all three
   authority flags are False in either state. COMPLETE still requires budgets.
   The entire revised candidate was reread. Ten prospective negative controls
   target this gap and four positives retain complete helpers after a later
   workflow refusal. Positive fixture constructors already carry all required
   state fields; no test edits were required after this candidate fix.

Source-review disposition: no remaining blocking finding within this bounded
metadata-binding scope. Ready for root's independent exact-source/PRE review,
not qualified by execution. All three fixes are in the new candidate only.

## Prospective independent controls

New tests are in `tests/test_gold11_schedule_projection_v2.py`. At this handoff
there are 20 test functions and 237 prospective outcomes by manual expansion
of explicit parametrization, not by pytest collection. No result is claimed.

The fixed fixture has two invented contact carriers, three selected registration
contexts, and either two invented ordered-pair metadata rows or an exact empty
schedule sentinel. It reuses ordinary structural constructors from the V1 and
contact test modules; it never invokes their pytest fixtures. Context literals,
budget values, implementation strings, digesting, and mutation oracles are
independent of candidate constants. All serialized valid workflow/projection
fixtures are asserted strictly below 100 KiB. Malformed byte fixtures remain
under the same bound. There is no large cohort or physical input fixture.

The controls include:

- Both successor arms and both cross-schedule combinations through actual pure
  V2 projection, frozen V1/contact projection, retained JSON serialization, and
  V2 reconstruction. The untouched inner contact projection is compared against
  a separate direct invocation on the same bytes. A spy delegates to the real
  function and checks original-byte object identity without replacing behavior.
- Complete nonempty and exact empty-sentinel paths; finite-support refusal
  without localized carriers; early absent-configuration and real helper
  refusal without budgets, across all four scan/schedule combinations. Later
  workflow refusals retaining a complete helper remain valid; contradictory
  early helper state and authority are independent negative controls.
- Independent changes at all four context publications and first/middle/last
  selected bindings, with missing/partial/wrong digests, identities, modes,
  budgets, unknown context-prefixed keys, and binding denominator changes.
- All 23 budget keys individually tested with boolean, float, and over-maximum
  values; missing/extra/negative values, chunk lower bound, query/family relation,
  every arithmetic/scope declaration, and a valid submaximal caller policy.
- No completed report with absent/noncomplete helper, wrong V3 schema, missing
  or float-valued helper budgets, nonboolean completion, positive authority,
  or an empty schedule disguised as a complete invoked helper.
- Exact sentinel key set, boolean-versus-integer aliases, disabled or absent
  joint accounting, nonzero/boolean/float ordered-match and slot counters,
  nonempty/unavailable schedule, and false completion/usable-schedule claims.
- Wrong workflow identity, historical-comparator scope, missing complete scope,
  units or carrier-unit relabeling, hash mismatch, duplicate JSON keys, NaN,
  overflowing numbers, malformed UTF-8, truncation, wrong root shape, depth,
  non-byte input, and caller input byte-cap refusal.
- Retained projection hash mismatch, changed original bytes, tampered context
  digest/configuration, inner contact prefix, schedule evidence, schema, and
  extra retained keys. Every projected rejection is reconstructed and checked
  for the exact six-key supervisor refusal, zero predictions, unknown count,
  production_use_allowed False, and cleared contact/schedule/context evidence.

The static policy test reads only context V1, V2, and V3 Python source files.
It extracts literal dataclass defaults and immutable maxima with AST literal
evaluation, verifies the exact V2 override expression structurally, and merges
the independent effective map. It does not evaluate/import backend policy
functions. Its expected map is a separate literal, so copying a changed
candidate maximum into the expected test at runtime cannot hide policy drift.

## Eventual gate closure and limits

The test import closure is the new test, tests/__init__.py, the two named older
test modules (ordinary constructors only), new schedule projection V2, frozen
schedule projection V1, contact scorer V2, mesh scorer V1,
evaluate_f1_against_manual_gold, and f1_physical_evaluation_contract_v2. Existing
NumPy/SciPy imports enter via the frozen matcher dependency; no matcher,
filesystem evaluator, CLI main, backend package, native geometry, or measurement
function is called by these new tests. The evaluator's legacy file-access
functions are not test authority and its main remains behind its main guard.
Three named context source files additionally enter the static-read closure.

The exact core qualification closure contains thirteen repository files:

1. `tests/test_gold11_schedule_projection_v2.py`
2. `tests/__init__.py`
3. `tests/test_gold11_schedule_projection_v1.py`
4. `tests/test_gold11_contact_scoring_v2.py`
5. `scripts/gold11_schedule_projection_v2.py`
6. `scripts/gold11_schedule_projection_v1.py`
7. `scripts/gold11_contact_scoring_v2.py`
8. `scripts/gold11_mesh_scoring_v1.py`
9. `scripts/evaluate_f1_against_manual_gold.py`
10. `scripts/f1_physical_evaluation_contract_v2.py`
11. `backend/cad_guided_two_plane_context_v1.py` (static source only)
12. `backend/cad_guided_two_plane_context_v2.py` (static source only)
13. `backend/cad_guided_two_plane_context_v3.py` (static source only)

The inert injected package bypasses scripts/__init__.py and all backend package
initialization. The eventual inventory must separately include this review,
root's new harness/protocol and independently prepared PRE/archive metadata,
plus pinned Python/pytest/NumPy/SciPy runtime identity. No new harness is authored
or executed by this reviewer. No existing broad test module should be collected
as an extra target: the two old test files enter only as constructor imports.
To keep that closure exact, the gate must disable pytest plugin autoload and
conftest loading, suppress cache and bytecode writes, and collect only the new
test file under the separately enforced harness. Any additional plugin,
conftest, source target, or runtime dependency invalidates this stated closure
until independently reviewed; no repository-wide pytest invocation is implied.

Before any test invocation, root must independently review exact source/tests,
freeze runtime/import closure and the fixed denominator, and obtain the required
PRE/archive for one bounded isolated gate. Suggested engineering envelope is
20 CPU seconds, 25 work seconds, 30 wall seconds, 512 MiB owned process-tree RSS,
and 1 MiB retained logs, subject to root's independent protocol decision. This
document neither invokes nor authorizes that gate. No rerun of an old attempt
is proposed. Final outcomes, quiescence, byte identities, publication inventory,
and independent POST remain separate obligations.

V2 adds a third decode of original RESULT bytes after the two frozen layers.
JSON node/depth checks occur after parser allocation. The 128 MiB input cap,
16 MiB wrapper cap, and node/depth guards are not external CPU or RSS limits;
tests under 100 KiB do not qualify maximum-size behavior. Canonical retained
comparison preserves JSON value types while allowing harmless projection JSON
whitespace whose exact retained bytes are still independently hashed.

The wrapper authenticates consistency against caller-frozen request metadata
and supplied byte hashes. It does not establish source execution, input custody,
raw-support membership, extrema correctness, schedule replay, registration,
material side, physical trim, calibrated units, unseen-family performance,
historical comparability, F1, or production authority. Backend integration and
transport/runner qualification remain separate from this small pure test gate.

## Source-only checkpoint

Reviewed candidate after all three fixes: 14,677 bytes, SHA256
`728920fe179751aa69e271a572d71acf98269a807904953ada0e38801874a3db`.
Reviewed tests: 26,090 bytes, SHA256
`61014e1500742ed40d16d7494bbeccae93e4c2ce722317e676d7123881653a76`.
Frozen schedule V1 SHA256
`a34431d167c086daa04b5e69e0c873f69df74fb27afd641148f7a91d4e899245`;
frozen contact V2 SHA256
`9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff`.
Only the new test and this new document were edited by this reviewer. Candidate
fixes were reported before root applied them. Source inspection and metadata
hashing occurred; no target execution, collection, or test pass occurred.
