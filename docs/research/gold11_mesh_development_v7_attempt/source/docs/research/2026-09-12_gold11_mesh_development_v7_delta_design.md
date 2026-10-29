# Unreleased full11 V7 delta — 2026-09-12

New source: `scripts/run_gold11_mesh_development_v7.py`. This is a source-only
draft, not a qualification, freeze, PRE, execution request, physical-input or
reference authorization. `RELEASE_AUTHORIZED=False`; timing status is
`UNRELEASED_V6_CARRYOVER_VALUES_NOT_V7_APPROVAL`. The named V7 physical protocol
is a future placeholder, not authored or read by this subtask.

## Frozen basis and exact delta inventory

Basis is unchanged `scripts/run_gold11_mesh_development_v6.py`, 166049 bytes,
SHA256 `7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80`.
V7 is a separately versioned narrow source copy, not a loaded-module monkeypatch,
an AST clone presented as V6, or a rewrite of the engine. The complete V6 source
is additionally declared as a source-only preservation basis in V7's `SOURCES`.
Its exact hash is required during both freeze and bundle verification. No V6
function or module is imported at runtime from this preservation file.

The full V6/V7 textual diff has only these classes of changes:

1. New runner, attempt, future protocol, delta-design, leaf, private-cache,
   execution-report/custody schema, private module namespace and sampled runner
   identities. Existing backend version names are untouched. `SOURCE_PROFILE`
   declares the V6 basis and the two behavioral deltas, is published in FROZEN,
   and must match during verification. Source/runtime/bundle hashes continue to
   bind the actual candidate and closure. There is no release bypass.
2. New `prestart_tree_admission()` and its one call in `execute()`, after exact
   bundle/PRE verification and before STARTED publication, private-cache creation,
   or dispatch. All original later custody checks remain.
3. A two-line `internal_coordinator()` guard after recording each returned
   outcome/receipt, before the next memory check, leaf, INFERENCE_TERMINAL or
   scoring: `source_binding_valid` must be the actual Boolean True. Otherwise it
   raises `INPUT_BINDING_NOT_ESTABLISHED_ABORT_NO_FURTHER_DISPATCH_OR_SCORING`.

`CONFIG`, numerical certificate/resource caps, `ARM_DEFINITIONS`, arm requests,
context V3 request and projection V2 binding, native V14 workflow, sampler policy,
scientific thresholds, source input/reference hashes, all11/both-arm iteration,
22-terminal reconstruction, reference71 selection, and scorer math remain
unchanged. The scorer request transport retains its literal
`GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_REQUEST_V6` schema on both producer and
consumer; SOURCE_PROFILE explicitly records this retained transport. This is
compatibility with unchanged request semantics, not a claim that V7 ran as V6.
No frozen source or file in the diagnostic's 64-file closure was edited.

## New guard semantics

Pre-start admission derives the complete expected source directory topology only
from `SOURCES`. The root must contain exactly `FROZEN.json`,
`PRE_RUN_AUDIT.json`, and `source`. For each known directory, inherited no-follow
directory opening and bounded name enumeration precede non-following metadata
checks of declared children. Missing/extra names, wrong types or symlinks,
declaration conflicts, a second-pass name difference, or changed directory
fingerprint veto admission. Unknown content is never read or recursively
discovered. The helper is release-guarded and runs inside the existing 180-second
parent-read deadline. Its names are bounded by the frozen source declaration;
it does not grant new metadata or input scope.

This is point-in-time metadata admission, not a filesystem lock or proof that no
future writer can mutate the tree. Existing exact source-byte verification,
one-use exclusive STARTED publication, leaf source checks, post-quiescence source
audit and whole-tree final inventories remain required. The original explicit
spent-marker checks are retained as an additional check after the new admission.

The input-binding guard treats an unestablished binding as an administrative
abort. False/missing/non-Boolean fields cannot authorize continued work. It does
not assert an actual hash mismatch: directory creation, STARTED publication,
memory or time refusal may happen before any input read. The original per-case
receipt and failure reason are retained unchanged. Existing coordinator abort
handling fills missing terminal positions only when writers are quiescent;
otherwise it defers fill to the outer owner and never reads/cleans live-writer
results. All22 positions remain the denominator. No scoring or reference access
follows this abort, and no favorable subset or retry is substituted.

A bound, scientifically negative or timed-out worker is different: if its exact
input binding and custody remain interpretable, the unchanged workflow may retain
that negative terminal and continue. Neither the six-slot diagnostic nor V7
requires every engine result to be positive to preserve the full denominator.

## Budget and release dependencies

All numerical limits are copied unchanged as draft values, not approved again:
3540 wall/1860 CPU seconds per inference leaf; 8 GiB sampled worker tree, 4 GiB
coordinator high-water check, 12 GiB outer tree; 1 MiB leaf log, 128 MiB result;
104400 total wall seconds, 2400 terminal reserve and 180 parent-read seconds;
180 wall/120 CPU seconds and 2 GiB for the scorer; 192 MiB scoring request and
32 MiB scorer/report result. Sampler 524288-byte stream plus 4096-byte receipt
shares the unchanged 1 MiB leaf log, leaving 520192 bytes arithmetically for other
output, not a separately enforced native quota.

22 times the 8 MiB per-case retained projection cap equals 176 MiB. A 192 MiB
request cap leaves only 16 MiB arithmetically for reference records, identities,
bindings, JSON framing and any representation differences. This is not proof of
the serialized request fitting: reconstruction and canonical serialization must
be measured/bounded together at the actual interfaces. Existing over-cap refusal
is preserved; there is no truncation, reduced case set or automatic cap increase.
Reference snapshot limits and per-object caps must not be counted as independent
proof of aggregate success. The enriched 32 MiB report and 256 KiB terminal/control
envelopes similarly need joint maximum-shape justification.

22 times 3540 is 77880 seconds; adding the 180-second scorer and 2400-second
reserve is 80460 seconds, leaving 23940 within the draft 104400-second overall
cap. That remainder must cover actual coordinator/preparation/read/retention and
publication overhead under the frozen accounting semantics; this arithmetic is
not a performance guarantee or permission to extend an expired deadline. One
heavy job remains the maximum. No consumed diagnostic cap authorizes full11.

Root must reconcile inherited maximum119-file/control-envelope blocker labels
with exact existing evidence and the new closure's schema-derived metadata/read
bounds. This draft does not independently claim that maximum finalization,
request capacity or physical timings have passed. The future protocol must bind
the actual source/runtime closure, Sep9 exact raw17 authority, unchanged71
reference hashes and scorer, full22 negative handling, limits, sampler log scope,
one-use controls, custody stop rules and independent PRE/POST requirements.

## Proportionate next review, not another broad suite

After the separate diagnostic's accepted execution evidence, independently review
this exact diff and add focused invented controls for: release and basis/profile
binding; valid root/source topology followed by unknown/missing/symlink/racing
metadata refusals before STARTED; positive source binding followed by false,
missing and mistyped controls that preserve the original reason and prohibit a
later dispatch/reference/scorer; and retained all22/scorer request and envelope
boundaries. Use inert process/I/O doubles and positive premises for each negative.
Reuse prior frozen source qualifications as separate evidence, not by rerunning
unchanged132 or claiming they qualify this new source automatically.

The diagnostic's accepted custody evidence is a prerequisite to the next release
decision, not a requirement for six positive geometry results. Any substantive
engine/resource-policy change inferred from it would require its own explicit
source delta and prospective protocol, not a silent edit here. Both fresh arms
use context V3; this remains consumed-development comparison, not historical
control replay, cached-control superiority or unseen-family accuracy evidence.

Authoring performed only named-source reads, a full textual V6/V7 diff, source
hashing and whitespace checks. No Python import, parsing, compilation, test,
native operation, freeze, attempt/control access, raw or Gold access occurred.
This source and design are handed off for later independent review; no execution
or qualification is claimed.
