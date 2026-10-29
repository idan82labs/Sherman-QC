# Residual-depth context V2 and finite-prefilter source gate

Status: SOURCE FROZEN FOR INDEPENDENT PRE, not an execution release. The exact
test denominator is241. Independent exact PRE and source-plus-PRE archive must
precede any import, collection or execution.

## Engineering reason and immutable negative

The depth-connected V1 attempt is terminal RED: six passes/one failure, zero of
four curved positives. Its retained scheduler completed256/256/252/254 ordered
matches, maximum42 per source pair. ContextV1 admits only four ordered targets
per source pair, so it refused before signed-family/slot generation or finite
queries. See the independently audited
[negative](2026-09-11_two_plane_depth_v1_connected_post_run_audit.md), SHA256
`e9ea38b68e0bee42c74e6e84666998449e20f5a2ec474761d6afb771cae55867`.
No old attempt, protocol, test or artifact is reopened. The prior69 integration
checks exercised only two matches; they did not cover this handoff cardinality.

This successor addresses that contract mismatch, not physical calibration,
optimizer speed, registration uniqueness or benchmark scoring. It is not a
larger started-experiment budget or a relaxed all-four connected oracle.

## Candidate and unchanged mathematics

ContextV1 stays byte-identical, including strict V1 budget type, four-per-pair
default and schema. A new contextV2 module reuses its immutable input record
types, numeric/work helpers, full-support bindings, canonical source/target
assignment checks, rank/rotation/translation construction, eight-slot ordering,
projection caches, witness identities and final whole-refusal semantics.
No batching, module mutation, quota refund, new pose prior or ranking is added.

Only the V2 orchestration is copied and versioned. AST comparison must permit
only its function name, typed default, explanatory docstring and one pre-input
global cardinality check to differ from the V1 body. V2 success and every early
or late refusal use the V2 schema and declared capacity profile.

Typed V2 budgets permit up to256 ordered targets on one source pair, but total
matches remain at most
`min(256, max_scan_pairs*max_ordered_cad_matches_per_scan_pair,
max_signed_families//4, max_candidate_slots//32)`.
Four signs times eight slots require32 slots per ordered match. The fixed
global ceilings remain256 matches,1024 signed families and8192 slots; distinct
source pairs remain at most64. This preflight occurs before input arrays or
support tables are owned/traversed. All other V1 caps are unchanged, including
64M scan-row visits,128M CAD-vertex visits,6144 projection-cache entries,
256MiB accounted numeric/payload work and32MiB report payload. Those are not
total Python RSS or a throughput guarantee.

InitializerV12 selects the V2 consumer only after the explicitly requested V2
scheduler passes its existing full configuration/equation/prefix/joint checks.
Before finite-query conversion, the consumer schema, complete default budget
map and V2 capacity publication must agree, with exact JSON types. The default
V1 route keeps its original consumer call and rejects V2 capacity markers.
The maintained V12 byte-binding path repeats this version/profile check before
measurement context use. Its unexpected-V2 detection includes context markers,
so a binding failure cannot preserve a usable measurement-context prefix.
Geometry gates, raw rows, free-circle fitting, transforms and all measurement
oracles remain unchanged. A context seed remains a proposal, not accepted pose,
finite support, observed count or measurement truth.

## Prospective tests and fixed denominator

Two newly authored test files are the only selected modules:

- `tests/test_cad_guided_two_plane_context_v2.py`: independent invented helper
  geometry, concentration beyond four and up to256 assignments, old V1
  equivalence/refusal, exact sign/slot accounting, strict policy/type limits,
  global257 refusal before buffers, support/identity tampering, late whole
  refusal and orchestration/math-source equivalence.
- `tests/test_cad_guided_two_plane_context_integration_v1.py`: actual tiny
  owned-array finite-prefilter handoff with more than four targets for a source
  pair, correct consumer dispatch and raw/candidate oracles; invalid returned
  schema/profile/capacity and unrequested V2 publication controls.

The independent helper file has19 test functions and179 selected outcomes;
108 outcomes are closed-range or type/range checks over all18 budget fields.
The integration file has12 functions and62 selected outcomes, including180
additional fresh field-mutation checks inside ten parameterized outcomes.
The frozen combined denominator is241 PASS and zero failures, errors, skips,
xfails or xpasses. No outcome is currently claimed passed.

The largest helper fixture runs exactly once at256 ordered matches, with8 raw
scan rows,4 scan triangles,24 CAD planes and48 CAD triangles:1024 signed
families and8192 explicit slots. Other actual depth checks use5,6 and42 matches.
The largest actual initializer fixture uses50 raw rows,64 scan triangles,8 CAD
triangles and4 CAD patches. It requires8 ordered matches on one scan pair and
256 slots for V2, versus4 matches/128 slots for V1, with a proper finite-passing
truth action in the complete passing pool. It does not require that later
48-start subsampling be monotonic. The byte-workflow cases explicitly mock
native/search/measurement and context geometry, retaining real schedule,
configuration and owned-array binding checks. They are not connected geometry.
The old test modules supply only explicitly reviewed ordinary helpers and
registered pytest fixtures; decorated fixtures are never called as ordinary
functions. No old suite or packet is selected, invoked or rerun.

## Source, process and resource custody

The new harness declares58 distinct sources:47 backend modules plus
the new harness, exact small supervisor, two new tests, three reviewed old test
dependencies, test initializer, this protocol, independent source review and
the exact connected negative POST. Two old test modules supply the ordinary
constructors/registered fixtures; the old contextV1 test is conservatively
retained in the source declaration but is neither imported nor selected.
The independent helper uses no old test fixture. Independent PRE must reconcile
the full import/source-read closure, not infer closure from this count.

The parent authenticates the unchanged standard-library supervisor buffer,
SHA256`579eb59de5538b024bb21510359dd62b50b38aa612ae7a18715e23b41bfa787b`,
and executes only its definitions and supervision function. Its old main, source
selection and tests are not invoked. Parent/worker reject any enclosing
`FullProject/__init__.py` object before attempt creation/import; the explicit
24-byte `tests/__init__.py` is frozen. Source reads are non-following regular
files with2MiB caps and stable descriptor/path identity before and after.

Frozen stop budgets are45 worker CPU seconds,55 monitored work seconds,
60 monitored parent/cleanup seconds,1GiB sampled group RSS and1MiB stdout.
Root reviewed the complete fixture inventory: the prior69-outcome integration
gate used6.64 CPU seconds; the new helper adds one bounded8192-slot case and
mostly cheap field checks. The45-second cap is proportionate for this source
gate, but is not an asserted runtime bound or permission to extend a started
attempt. Any budget failure remains terminal and requires a justified successor.
There is one worker, one numerical thread per library, plugin autoload disabled,
`-I -S -B -u`, `/dev/null` pytest config, `--noconftest`, and no pytest cache.
An inert backend package avoids application initialization. Only invented
NumPy geometry and mocked publications run: no native reader, CAD file, physical
scan, Gold, catalog, email, held-out, server or benchmark runner executes.
Interpreter/NumPy/SciPy/pytest/stdlib/OS binaries are outside repository hashes.

After exact PRE, archive all resolved source payloads plus the PRE, verify every
regular named payload and reject extras/duplicates/links/AppleDouble members.
Record hashes before one launch:

```sh
.venv/bin/python -I -S -B scripts/check_cad_guided_two_plane_context_v2.py
```

The fresh `two_plane_context_v2_source-` attempt retains only STARTED, OUTPUT
and COMPLETED, plus the separate uninspected private cache. No scientific
geometry files are emitted. No overlapping heavy worker is allowed. Keep every
outcome and ordinary failure; no positive prefix, skipped case or exit-zero-only
qualification. Source/setup/final receipt I/O lies outside monitored clocks;
sampled memory/cleanup are engineering guards, not hard-real-time guarantees.

## Decision and next boundary

Require the frozen actual PASS denominator and zero other outcomes, full output
and CPU/RSS footer, exit0, no budget/primary/cleanup/source error, confirmed empty
writer group, exact three-file inventory, all source-before/after/live/archive
bindings and initializer guards, followed by independent POST. Any failed
obligation is a terminal negative. Do not rerun, raise limits or shrink fixtures.

A GREEN source gate would qualify only this bounded helper/interface contract.
It does not prove the original four curved cases now work. Those same cases,
their unchanged tolerances, original full stage/artifact denominator and a newly
frozen connected successor still require independent PRE and one execution.
Native qualification and an audited same-input Gold11 control/challenger score
follow separately. No physical F1, historical-score superiority, calibrated
uncertainty, manufacturing count, stable datum or factory-floor release is claimed.
