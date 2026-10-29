# Context V3 diagnostic V1 — independent source review

Date: 2026-09-12. Reviewer: independent AI technical agent
`process_v2_qualification`, not the diagnostic runner/design author or the
prospective protocol author. This is a SOURCE-ONLY review, not an exact frozen
PRE, execution authority, test result, physical diagnostic POST or accuracy
claim. No target was imported, compiled, executed or collected. No process or
native API was queried, no signal was sent, and no raw input, Gold, protected
control, attempt or private-cache artifact was opened, stated or hashed.

## Decision and exact revision

GREEN FOR THE EXACT UNRELEASED SOURCE DELTA, subject to the separate focused
test gate and later release/freeze/PRE sequence. The connected source review
found one concrete reason-label defect, independently confirmed and resolved
below. No blocking finding remains in these source bytes. This is not a green
physical-run PRE or a claim that the new orchestration tests have passed.

Final reviewed runner: 57923 bytes, SHA256
`08f9e50b52f0225ca54e1920ef21fed2d0840a19941fb20b338b43e6b141a44f`.
Final reviewed protocol: 12845 bytes, SHA256
`da2db6127916ec2b714fa03c3f04cc901c0ff1263ed35db1b1c8d39d87822ec3`.
The design and provider identities below are unchanged. Final declared closure:
64 unique source paths, 1697488 bytes, canonical path-to-SHA256 map digest
`fecd0786588b1bff72389b4b295b7328c04b3c125217b1d2dc67f4eb27a36d08`.
All named Python source files parse as AST; none was compiled or executed.

The review read the complete new runner, complete design and complete prospective
protocol; the selected provider definitions and their connected call paths;
the permitted September 9 narrative and literal CASES source; and the accepted
V6 independent audit and 132-check POST. Existing backend/projection/supervisor/
sampler qualifications are separate evidence, not new execution in this review.
The complete 64-file declared source set was independently read for byte hashes
and Python AST parsing where applicable. This is not a claim that every dormant
legacy backend/scoring definition was newly reviewed line by line.

Pre-correction identities, retained to make the review delta explicit:

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| New diagnostic runner | 57918 | 15add9f689bbcbc4053c738af3e28e0d524cf1edb0a2ba5b11abd70d783b8048 |
| New diagnostic design | 19707 | a79640d7a98ba2e08737bda14004fbbe90d28ff8ff1bb9b207547a840c90e108 |
| New prospective protocol | 12693 | a8c009c37e1abb2e9604517590cfb13de6a2794c864f7e84dc26a9f709eebeaa |
| V6 provider | 166049 | 7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80 |
| September 9 review source | 25616 | 41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955 |

Pre-correction declared closure: 64 unique source paths, 1697331 bytes;
SHA256 of the canonical JSON path-to-SHA256 map, sorted keys with compact
separators, is
`d4096f04a757d497b4442a7d7beae6ab5b3f90d156a82e2bc604cec7c74ca5e7`.
This review document is not silently added to the runner's declared source set.
The later exact test/freeze protocol must bind its review evidence explicitly.

## Confirmed reporting defect and bounded repair

Root identified, and this reviewer independently traced, the original
`DIAGNOSTIC_EXACT_INPUT_HASH_BINDING_ABORT` at the coordinator's check of
`row.get("source_binding_valid") is not True`. In unchanged V6
`supervise_case`, that field starts False before destination creation, STARTED
publication, coordinator memory checking and the remaining-wall check. Both
input reads must finish before it becomes True. A pre-read reserve/memory/
publication failure therefore reaches the same diagnostic abort without any
established hash mismatch.

The implemented replacement is
`DIAGNOSTIC_INPUT_BINDING_NOT_ESTABLISHED_ABORT`, with protocol wording that
distinguishes an actual input hash mismatch from a binding not established for
another reason. This changes the diagnostic explanation, not the abort rule,
case order, numerical engine, budgets, original refusal or retained receipt.
V6 preserves the original supervisor refusal in the case record and projection;
the coordinator should not replace uncertainty with a more specific cause.
Reversing only that string in the final source reproduces the full original
runner SHA256 above. Reversing only the explanatory protocol paragraph likewise
reproduces the original protocol SHA256. The final deltas were independently
read and hashed; no numerical, budget, dispatch or failure-retention change is
hidden in this correction. The finding is resolved at source level.
The focused new test should preserve an exact original pre-read reason, assert
this generic coordinator label and prove no later leaf invocation.

## Closed provider profile and actual entrypoints

The provider is loaded as one exact-hash, bounded regular source buffer. The
bootstrap requires the release gate before source access and applies no-follow,
nonblocking open, before/after file identity checks, a byte bound and a bounded
alarm scope. Its AST allowlists contain exactly 21 constants, 55 functions and
two classes; the seven new callbacks are explicit. Constants retain provider
source order for dependencies. The selected 57 definitions are unique and
unchanged, including source locations. The application-Python AST receipt is
now explicitly versioned: `.venv/bin/python -I -S -B`, Python 3.11.15
(Clang 22.1.3). Parse the unchanged provider bytes with `ast.parse`; select its
top-level FunctionDef/ClassDef nodes whose names occur in the diagnostic's
literal V6_FUNCTIONS/V6_CLASSES; construct
`{node.name: ast.dump(node, include_attributes=True)}`; serialize using
`json.dumps(mapping, sort_keys=True, separators=(',', ':')).encode()` with no
newline. These 57 entries serialize to 778823 bytes and SHA256
`00fec424e3560003a8a48b16c262925c092f6a71373414b76e3c4e4f02defd97`.

Correction to the previously accepted 17796-byte review SHA256
`81dbb222389d192125e3453ff5c4b597c33aaea2472050169dcad4751d691bf4`:
its unspecified AST-map digest
`94d87ace0117b3eacccb9b5f88debf3f543d6ddb6be981973cdce004a363dcb7`
used shell `python3`, Xcode Python 3.9.6, whose same recipe serializes to
777540 bytes. Both recipes were independently reproduced; the AST representation
is interpreter-dependent. The application-3.11 receipt above supersedes that
unversioned receipt. Candidate/provider byte hashes, selected definition names,
source map, source-review decision and behavior are unchanged. This is a
documentation correction, not a target execution or additional test result.
Source-only symbol inspection found no missing referenced global in the
declared constants/imports/profile callbacks and builtins.

This is not full-module V6 import or a mutation of an already loaded V6 module.
The new namespace explicitly replaces ROOT/RUN, release, leaf schema, sources,
limits and state. It excludes catalog/reference globals and forbidden runner,
coordinator and scorer entrypoints. The own callbacks are `require_release`,
`publication_cap`, `leaf_sources`, `validate_leaf_binding`,
`fresh_private_cache`, `runner_command` and the six-slot `_campaign_schema`.
The profile declaration binds their names, the retained provider identities,
source paths, limits, six-slot order, no-score policy and full arm requests.

The 64 paths consist of the two runners, 50 literal provider backend sources,
artifact/supervision/stage-sampler sources, six pure projection dependencies,
the fixed input-review source, design and protocol. The backend set preserves
the actual V14 import graph plus legacy source-bound extras; retaining those
extras is not permission to execute their unrelated entrypoints. The new worker
command points to the frozen diagnostic script, not to the old V6 coordinator.
Inherited authorization observes the new callback/global bindings.

The actual inherited workflow path still constructs V14 with context V3 for
both arms, using unchanged CONFIG, certificate and numerical policy values.
Control requests V7 scan planes/schedule V1; challenger requests composite scan
planes/schedule V2. The mandatory context, scan and schedule binders remain in
the actual workflow. No per-case reference, count or case-token inference
feature is added. The engine receives the verified complete byte buffers.

## Input authority, source freeze and no-score boundary

The exact review source is parsed, not executed. Only its literal CASES JSON
expression is decoded. Shape/count/uniqueness checks establish eleven scan
hashes, six CAD hashes and seventeen distinct absolute path/hash pairs from
that pinned source declaration. `reference_id` is stripped. The selected cases
are exactly CG08506AE97D, CG4BDD1371B6 and CG4FAF273093, each control then
challenger. This is source-declared authority; this audit did not verify the
existence or bytes of any physical file.

The new freeze path is source/runtime-only and exclusively creates its own
attempt. The exact frozen bundle includes the source map, source-profile
digest, raw17 source declaration, selected three cases, two arms, six terminal
slots, limits, runtime and no-score policy. Bundle verification reconstructs
the expected object and requires canonical equality rather than trusting a
subset of fields. The qualified artifact/supervisor/sampler hashes and pure
projection map are bound before later use. Bootstrap artifact publication has
a pinned live-source fallback before the verified frozen module state exists;
there is no missing initialization prerequisite in that path.

The independent frozen PRE must bind the exact bundle/profile/order/limits/
no-score policy and report zero blockers. Current release is False and timing
is explicitly unreleased. A source gate on those bytes does not qualify a
future released flag silently: the protocol requires an explicit independent
release-only delta review before freeze and exact PRE. Neither this document
nor a previous V6 PRE grants physical access.

The only child modes are inference and the new internal coordinator. SCORER
leaf role is rejected before inherited leaf-control reads. Scoring output,
request, selected-reference and scorer-receipt publication names are vetoed.
`SCORING_INPUT.json` remains the qualified result/projection transport, not
scoring authority. The pure projection dependency graph contains dormant
scoring functions and numerical imports, but the new coordinator passes no
references and invokes no scoring function. Retained unused legacy constant
strings referring to ALL22 are not reachable all22 orchestration. The no-score
declaration stays bound through profile, PRE, coordinator context and outputs;
diagnostic completion does not release a full11 score automatically.

## Six-slot dispatch, abort and private-cache lifecycle

Coordinator dispatch binds the parent's token/hash, PID, fresh external cache
identity and frozen STARTED/bundle/PRE/profile/arm-request identities. The child
must be the own process-group leader, launched at the frozen path in its own
cache with matching cwd/HOME/TMPDIR. Terminal markers veto repeat invocation.
The exclusive STARTED consumes the attempt. There is no resumed prefix or
alternative mutable case manifest.

The real coordinator loops over the three frozen cases and two fixed arms,
calling inherited `supervise_case` serially. It aborts further dispatch on
unestablished input binding, unresolved writers, failed retention or failed
private-cache cleanup. The original input buffers are read and hash-checked
under the inherited preparation caps, with no other raw17 corroboration read.
The case-specific refusal is retained even when the coordinator must abort.
An engine scientific refusal or cleanly handled CPU stop is not itself an
instruction to drop a slot or tune the next one.

Leaf issue/claim uses the new inference-only schema and exact arm/input/request
binding while preserving the unchanged one-use parent capability checks.
Private cache identity is bound before dispatch. Result enumeration, content
retention and cleanup require the strict returned writer-quiescence receipt
and the same fresh cache identity. Missing receipt or failed retention keeps
the cache opaque and latches abort; there is no destructive TemporaryDirectory
fallback. The coordinator cache is outside the retained scientific tree and
is retained opaque rather than traversed during outer recovery.

The outer supervisor's receipt, not a guessed PID or subprocess return alone,
gates postflight source/control reads, terminal filling, retained result/log
reads and final inventory. Unknown writers produce explicit nondurable six-slot
unknowns. The outer path cannot manufacture successful retention or cleanup
from an absent receipt. The unchanged supervisor's native generation/exit
transition behavior is separate qualified evidence, not newly exercised here.

## Reconstruction and inventory are genuinely six-slot

`terminalize_case_records` derives its denominator from the supplied bundle's
cases and arms; it has no hard-coded eleven-case requirement. The own
`reconstruct_six` requires the fixed order and exactly six rows. Successful
workflow rows require the exact original-result hash, real V2 reconstruction,
source binding, clean supervisor disposition and complete custody. Failure
rows require the retained exact supervisor-refusal path. At most one complete
or partial result is retained under its explicit artifact binding; partial
bytes are not parsed as usable inference.

Closed logs are read only at the allowed post-quiescence stage. Their retained
hash/count checks bind observed closed bytes, not a hash computed at capture
time. Unknown failed-attempt capture counts remain unknown. Stage diagnostics,
receipt locations, raw resource scopes and no-score status are carried into
each row. Coordinator and outer reconstructions must agree on the evidence
digest; six terminal slots and source-profile/arm-request/no-score declarations
are checked again at the terminal boundary.

The new schema supplies the inherited inventory algorithm with 64 source files,
own controls, six case directories and their result/receipt/log/projection
slots, and six possible issued/claimed inference pairs. It adds no scorer
directory, reference-selection artifact or all22 completeness requirement.
The generic leaf-binding validator uses the new context/schema callbacks and
does not dereference private-cache contents. Unknown names are metadata-only
vetoes, not content discovery. File/directory identity, before/after tree shape,
caps and exact known bindings are checked under remaining time. The retained
generic V6 custody schema identifier is explicitly wrapped with diagnostic
semantics, not relabelled as a newly tested numerical/inventory algorithm.

COMPLETED, then FINAL_GATE, are each followed by the required final inventory
checks. An earlier marker is insufficient; a late publication/inventory error
vetoes acceptance and preserves a failure disposition. No all-positive geometry
condition is hidden in this path. Root still needs actual terminal stdout/exit,
external writer-quiescence evidence and an independent exact POST to accept a
future attempt.

## Budget and uncertainty interpretation

The fixed plan is six serial cases at 3540 wall/1860 CPU seconds each, 8GiB
sampled worker tree, 4GiB coordinator lifetime high water and 12GiB outer budget.
The 1MiB log, 128MiB result, 8MiB projection including newline, 64MiB scan and
8MiB CAD caps remain unchanged. Arithmetic is 21240 case-wall seconds plus 2400
terminal reserve plus 1560 overhead = 25200 seconds. The outer wall covers
coordinator setup as well as its loop; the internal start does not extend that
outer bound. Content work receives only remaining campaign time.

The protocol explicitly discloses at most one extra one-second emergency
administrative veto attempt after expiry, not input/result reads, inventory or
another engine case. It cannot convert an overrun into success. The practical
monitoring allowance is that disclosed 25201-second bound plus observation
latency, not an inference extension. Synchronous I/O/alarm delivery is not
claimed to be a hard real-time bound; sampled RSS is not continuous allocation
enforcement, and Darwin ru_maxrss is lifetime high water, not current RSS.

The source-only sampler is observational: 30-second cadence and 128-sample cap
cover the planned 3540-second window. Its bounded framed output shares, rather
than extends, the full native log cap. Missing/interleaved/partial observations
remain incomplete; a last observed frame is not proof of the deepest stage or
continuous native progress. No claim of scanner calibration, stable-datum
metrology, production acceptance, unseen accuracy or new F1 follows.

## Proportional next verification

The new focused source gate should exercise the real closed profile construction
and own public orchestration with inert physical/native seams. Concrete needed
coverage was sent to the test author before this review handoff:

- Release/bootstrap/profile rejection and exact callback/global wiring; source
  authority selection/stripping and bundle reconstruction drift.
- Inference-only role/publication/command policy, one-use coordinator/leaf
  dispatch identities, wrong source/runtime/profile/token/cache bindings.
- Actual six-slot order, successful and negative terminal reconstruction,
  exact projection/log bindings, retained partials and unknown receipt handling.
- Own schema with the real generic inventory algorithm: six rather than 22
  slots, forbidden scorer names, unknown entries, drift and exact final markers.
- Abort before further dispatch for binding not established, unknown writers,
  retention/cleanup failure, and late publication/inventory failure. Assert
  exact reason and callback reach, not merely that some exception occurred.
- Outer source/read/fill/inventory ordering under confirmed versus unknown
  quiescence, without opening real caches or sending process signals.

These are new-delta controls, not a request to rerun the unchanged 132 runner
checks, sampler26, projection237, prior 93 connected successes or dense fixture.
The separate five repaired-envelope controls retain their own evidence. The
diagnostic tests are being authored by another agent and have not been read,
executed or counted as passing by this review. Their final source/fixtures,
exact denominator and bounded harness need their own independent PRE.

After an accepted focused gate, release-only delta review, source freeze and
exact frozen PRE, the planned physical diagnostic can test the declared
mechanisms. Its acceptance is six interpretable durable slots with valid
custody, not six favorable engine results. A scientific refusal or engine
timeout stays a negative outcome. Any runner/custody defect remains a failure
requiring a new version, never a retry of an old terminal attempt.
