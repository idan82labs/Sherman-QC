# V7 full11 delta: independent source review — 2026-09-12

Disposition: **no concrete blocking defect found in the reviewed delta**. This
is a source-only review, not a source-test pass, physical PRE, release, freeze,
timing qualification, input authorization or score. The candidate remains
`RELEASE_AUTHORIZED=False` with unreleased carryover timing declarations. Its
future physical protocol is not qualified here. Root owns any subsequent change.

## Exact reviewed basis

| Named source | Bytes | SHA256 |
| --- | ---: | --- |
| `scripts/run_gold11_mesh_development_v6.py` | 166049 | `7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80` |
| `scripts/run_gold11_mesh_development_v7.py` | 169028 | `27737eff034891e8f7a6d09ba63f5bd17a3947c67ffd553ddfb73a931fef5cb3` |
| `docs/research/2026-09-12_gold11_mesh_development_v7_delta_design.md` | 8879 | `55e905a7bffd25847426826d498624a4ad5b6387d829e66217c03969ba2db403` |

The complete textual V6/V7 diff and design were read, together with affected
functions and the inherited admission, request, reconstruction, publication and
abort helpers needed to establish their contracts. Canonical instructions and
current-status top were read. Checks used named-source reads, SHA256 and Python
3.11 standard-library AST parsing only. No target module was imported, compiled
or executed; no test collection/run, native operation, process inspection,
attempt/control freeze, physical input, Gold or private-cache access occurred.
Only this new review document was written.

AST comparison without location attributes finds 81 V6 top-level function/class
definitions and 82 V7 definitions: one addition, `prestart_tree_admission`, no
removals, 69 unchanged inherited definitions and 12 changed definitions. The
changed set is `require_release`, `projection_module`, `freeze`, `verify_bundle`,
`owned_leaf_cache`, `stage_sampler_module`, `runner_command`,
`score_after_terminal`, `internal_coordinator`, `verify_parent_artifacts`,
`verify_campaign_inventory`, and `execute`. Full diff inspection resolves the
changes into version/source-profile identities and the two declared guards;
the definition count is not itself a behavioral proof.

## New admission and abort paths

`execute` verifies exact bundle and PRE, calls the new admission helper within
the existing 180-second preflight deadline, then binds control bytes and checks
spent markers. Only afterward can it attempt exclusive STARTED publication,
create a private coordinator cache or dispatch. Failure in the new helper has
`started_attempted=False`, so the outer preflight handler rethrows without
publishing a terminal marker or entering execution. Existing files are not
deleted or repaired.

The helper builds its namespace solely from the fixed source declarations.
Static declaration resolution gives 70 distinct, relative, non-parent-traversing
source paths, seven inspected directories and 78 declared child metadata checks:
root 3, `source` 4, `source/backend` 50, `source/docs` 1,
`source/docs/research` 2, `source/scripts` 16 and `source/tests` 2. This is a
source-declaration count, not observation of a frozen or physical directory.
The root permits exactly FROZEN.json, PRE_RUN_AUDIT.json and source. Each source
directory permits its exact declared children; an unknown/spent/missing name
vetoes the attempt. Source declarations cannot collide as both file and directory.

Inherited directory opening pins each path component with directory/no-follow
flags. Bounded enumeration returns complete sorted names or explicit one-name
overflow. Non-following child metadata must be regular-file/directory as declared.
Symlinks fail these checks. Second enumeration and directory fingerprint checks
reject changes visible during that directory's admission. Descriptor closure is
in `finally`, including refusal. No unknown entry's content is opened, nor is an
unknown directory recursively discovered. Earlier bundle verification reads only
known bound files; it is not a pre-admission read of arbitrary discovered content.

This is deliberately point-in-time metadata admission, not a lock, an atomic
whole-tree snapshot, or a promise against later mutation. Earlier exact-byte
verification, later control rebinding, no-retry STARTED creation, leaf source
verification and post-quiescence custody inventories remain necessary and
unchanged. No stronger race-freedom claim is warranted.

The coordinator records each returned outcome and receipt before requiring
`receipt.get("source_binding_valid") is True`. False, missing and truthy
non-Boolean values all abort before the next coordinator memory check, leaf,
INFERENCE_TERMINAL publication or scorer. `supervise_case` starts this field
False and sets it True only after both exact input reads complete. Thus an
early filesystem, memory or reserve failure is an *unestablished binding*, not
automatically a proven input hash mismatch. The original receipt and supervisor
failure reason remain retained; the new coordinator reason is additional.

The unchanged exception path marks `_COORDINATOR_ABORTED`, fills only missing
terminal positions when writers are quiescent, or retains 22 explicit deferred
positions when they are not. It does not replace a completed leaf or retry it.
The scoring entry point independently refuses an aborted/unquiescent coordinator
before reconstruction, reference access or scoring. A genuinely Boolean-True
binding does not imply scientific success: a cleanly retained negative or
budget-stopped leaf may still proceed through unchanged custody requirements.
The new guard therefore preserves negative evidence rather than demanding
all-positive geometry or scoring a favorable subset.

## Basis, identities and preserved scientific contracts

V7 declares the full frozen V6 provider as a source-only preservation basis.
Freeze and bundle verification both require its exact SHA; SOURCE_PROFILE is
published and required to match during verification. V7's own bytes and full
closure remain separately bound. No runtime import of this added V6 basis was
introduced, and the diagnostic's frozen V6 dependency was not edited.

Runner/attempt/protocol, leaf dispatch, private module/cache, report and custody
schema identities are consistently V7 in their changed producers and consumers.
The retained V6 scorer-request schema is explicitly documented compatibility:
both `make_scoring_request` and `scorer_arguments` still use the same literal.
This does not misidentify the runner as V6. Sampler runner-frame allowlisting
changes to the actual V7 source path; the sampler module, source SHA, log policy
and wrapper behavior otherwise stay unchanged.

AST equality independently confirms unchanged LIMITS, CONFIG, CERTIFICATE_CAPS,
ARM_DEFINITIONS, COMPARATOR, TWO_PLANE_DECLARATION, SCORING_TRANSPORT, ARMS,
RETAINED_CONTROL_POLICY, REVIEW/REVIEW_SHA, CATALOG_SHA, SELECTED_SHA,
STAGE_SAMPLER_SOURCE/SHA and STAGE_LOG_DECLARATION. The complete diff contains no
backend edit, numerical/scorer change, cap increase, schedule reduction or
reference-subset substitution. Both fresh arms retain V14/context V3 with their
distinct frozen scan/schedule requests. Full11/both-arm and unchanged71 scoring
remain the intended scope, not cached-control replay or unseen-family evidence.

## Aggregate transport and control accounting

22 × 8 MiB = 184549376 bytes (176 MiB); the 192 MiB request cap is
201326592 bytes, leaving 16777216 bytes (16 MiB) arithmetically. The per-leaf cap
includes its retained wrapper publication newline. Reconstruction revalidates
original result and wrapper bytes before taking `scorer_input`; exact supervisor
refusals use their separately validated bounded schema. The request does not
simply concatenate 22 wrapper files: it contains the reconstructed inner
objects, selected bends, family map, full arm and scan requests, 22 projection
bindings and framing. Canonical serialization is the actual accounting unit.

The request path checks all11 keys and both arms, reconstructs all22 terminal
inputs before Gold, checks the terminal outcomes binding, selects exact71, makes
the full request and invokes the consumer validator before encoding it. It then
checks the actual complete encoded length against 192 MiB **before creating or
dispatching the scorer cache**. The separate private publisher repeats the byte
type/length guard; the scorer reads at the same cap with the expected hash.
There is no prefix truncation, record dropping, retry or automatic cap increase.

This is correct fail-closed accounting, not evidence that every maximal admitted
shape fits. The catalog/selected-reference 8 MiB bounds, per-wrapper limits and
16 MiB arithmetic headroom must not be presented as independent proofs of the
aggregate. No physical reference or result bytes were read to estimate fit here.
Likewise an original scorer result near 32 MiB can exceed the same 32 MiB cap
after report enrichment; the existing complete-report encoded-length check
refuses that case. This inherited limitation is unchanged, not a newly discovered
V7 regression or an excuse to silently enlarge caps.

Publication distinguishes 8 MiB leaf receipts/projections, 32 MiB scorer/report,
source/dispatch exceptions and default 256 KiB controls. Final artifact evidence
has an explicit complete encoded 256 KiB guard. Adding one preserved source and
SOURCE_PROFILE affects FROZEN and final source/inventory envelopes; metadata
counts above do not establish their final encoded sizes. Exact future freeze/PRE
must bind the concrete controls and complete transport accounting, reusing prior
qualified finalization evidence where applicable. The inherited 119-file blocker
label alone is not a reason to rerun unchanged broad suites. These are bounded
release-accounting questions, not blocking code defects in this delta.

## Smallest useful new qualification

A focused detached-source gate can cover the changed behavior with inert
filesystem/process/reference seams; no native engine or physical inputs are
needed. The following is an illustrative 21-outcome plan, **not a frozen test
denominator or a claim that these tests exist or passed**:

| Group | Outcomes | Required reach and observation |
| --- | ---: | --- |
| Release/basis/profile | 3 | Actual unreleased refusal; exact bad basis; exact mismatched profile, with otherwise valid premises. |
| Pre-start topology | 9 | Valid tree; root extra; spent root; missing source; file symlink; directory symlink; changed second names; changed fingerprint; declaration conflict. Exercise actual helper, close descriptors, and prove refusal precedes STARTED/cache/dispatch. |
| Binding abort | 6 | Boolean True; False; missing; None; integer 1; string `true`. Run the actual coordinator branch with original negative receipt retained; non-True cases prohibit a second dispatch/references/scorer and preserve all22 terminal positions. |
| Full request/cap seam | 3 | Complete22 request reaches actual producer/consumer with bound fixtures; exactly-cap allowed; cap+1 refuses before scorer-cache creation. |

The True control should retain an honest negative workflow outcome and continue,
not require invented all-positive geometry. Exact-cap controls may use a detached
encoder length double, explicitly qualifying comparison/order only; they must not
claim a real 192 MiB serialization or maximum-shape capacity run. A small real
all22 canonical fixture can establish field accounting and denominator without
reading Gold. Any assertion that the maximal real request fits needs a separate
source-bound aggregate size argument or specifically frozen representative shape,
not another broad numerical campaign. Inspect every negative fixture for its
positive prerequisites and exact intended reason before the sole gate.

No execution is released by this review. Next: accepted six-slot diagnostic POST,
focused V7 delta qualification and exact full11 source/protocol/PRE/archive with
concrete release accounting. Reuse unchanged132 and other terminal qualifications
as separate evidence; do not rerun them or edit their frozen sources. Keep physical
accuracy negatives, administrative aborts and transport refusals distinct.
