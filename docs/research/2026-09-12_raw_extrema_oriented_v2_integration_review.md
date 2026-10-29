# Oriented raw-extrema V2 — source-only integration review

Date: 2026-09-12. Independent reviewer: `process_v2_qualification`.

## Conclusion

The smallest genuine route is a separately versioned raw-extrema V2, a context V4
owner/ledger adapter, and a closed new initializer/workflow dependency seam that
actually calls V4. A new pure context-binding projection adapter must recognize
that exact implementation. The existing contact geometry, matcher and metric
definitions need not change if their wire contracts remain unchanged.

Do not implement four wholesale copies merely to change imports. Conversely,
aliasing existing public functions or replacing one module name does not create
a working dependency seam: those functions resolve their original globals.
Prove the standalone mechanism's correctness and useful bounded work first;
defer exact workflow successor packaging until that evidence exists. This review
does not impose a new broad gate, create a runner, authorize physical access or
claim efficacy.

The inspected opt-in evaluation chain is:

`V7.workflow_entry_and_policies -> V14.inspect_cad_guided_mesh_v14 -> V13 initializer -> context V3 -> extrema V1`.

V7 is frozen evidence, not the location for the successor implementation.
Production-pipeline reachability of this chain is unestablished. The separately
inspected legacy `bend_inspection_pipeline.inspect_cad_guided_offline` calls the
older feature/measurement kernel, not this byte/native context workflow. No
default product routing change follows from adding an accelerated helper.

## Exact source basis and scope

The complete extrema V1 and context V3 sources, complete initializer V13 and
workflow V14 dispatch/publication code, complete projection V2, and the relevant
named predecessor, pure scorer and frozen V7 seams were read as source. Partial
navigation reads of legacy adapters do not constitute a broad repository audit.
No target import, compilation, execution, test, raw/Gold/control/attempt read,
private-cache access or existing-source edit occurred. This new review is the
only write. Proposed names below are architectural suggestions, not existing
qualified implementations or reservations of another author's files.

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| backend/cad_guided_raw_extrema_index_v1.py | 19626 | `668de552dea084dcf53ffb62ef4ca45bc9bfb2b12b4da8058348b3668e343121` |
| backend/cad_guided_two_plane_context_v1.py | 36075 | `9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a` |
| backend/cad_guided_two_plane_context_v2.py | 11130 | `d2bdea468ebe6eca5bea9f125a6f7cb3bbf49894a65ab6861c81fd2e7fb7b663` |
| backend/cad_guided_two_plane_context_v3.py | 25845 | `9077f434aa7baf0ce5b63c620407b091258bb41e2141bbe41465ca4074ed543d` |
| backend/cad_guided_plane_initialization_v13.py | 38651 | `79ee4de537f1dddcc1dee40d33b25fe3fe8fb667bcd4e30ecc0a811ebe2d989f` |
| backend/cad_guided_mesh_offline_v14.py | 33435 | `76378b93c44c046b773452b3d8cad400cb237ff8bc7039e258a90305bbf6c749` |
| scripts/gold11_schedule_projection_v1.py | 17231 | `a34431d167c086daa04b5e69e0c873f69df74fb27afd641148f7a91d4e899245` |
| scripts/gold11_schedule_projection_v2.py | 14677 | `728920fe179751aa69e271a572d71acf98269a807904953ada0e38801874a3db` |
| scripts/gold11_contact_scoring_v2.py | 50413 | `9143dc77636068dd41d5eb5e60bc4b461857e4e72a2b8f486935021ae628b4ff` |
| scripts/run_gold11_mesh_development_v7.py | 169180 | `e910d562e55a62de80f5e448f944c239b5bdf95ef11c6d0b6f0b83b24bfc7a52` |

These are source identities, not a new runtime/postflight claim.

## Why an index subclass alone is unsafe

1. **Lexical dispatch survives aliases.** V1 `project_raw_extrema_v1` allocates
   its module-global `_Index`; V3 `propose_two_plane_context_v3` allocates its
   `_ContextWork` and calls its `_slots`, which calls its `_projection`.
   `_ExtremaRegistry._index` explicitly constructs `_ContextIndex`. Importing or
   aliasing those functions in a new module does not redirect any of these
   globals. V13's registration entry similarly resolves its own configuration,
   initialization and publication helpers; V14 resolves `_initialize_v13` and
   its own binding function. A supposedly new entry can therefore still execute
   the frozen V1/V3 path and merely report a new name.

2. **The existing context subclass is a deliberate narrow bridge.** V3
   `_ContextIndex` does not call the standalone constructor or install its local
   cap. It shares the context's bytes-backed raw owner and one cumulative
   `index_stats` dictionary; `_charge` accepts only seven named categories and
   forwards each to the parent. A V2 method that expects new constructor fields,
   `self.cap`, a reset statistics dictionary, or an unmapped charge will not be
   compatible. Adding uncharged orientation work to avoid that rejection is not
   a valid fix. Inherited query receipts also return a fixed list of work fields;
   new counters must not disappear from per-query or refusal evidence.
   Both current budget validators use exact `type(...) is ...` checks, so a
   subclassed V2/V4 budget also requires its own validator and owned snapshot;
   it is not admitted automatically by the V1/V3 public API.

3. **Current declarations describe the AABB algorithm.** Standalone V1 reports
   one longest-root-axis stable sort, fixed leaves256, `sort_axis`, node slots,
   and `6N + 12S + S + 6S + 9E` coordinate-work components. Context V3 reports
   `19M_BUILD_PLUS_9E_QUERY`, one stable sort per distinct index, V1 arithmetic,
   fixed-leaf scope and shared-owner memory. New orientations, transformed
   bounds, extra reductions, searches or error-envelope work cannot inherit
   those strings unchanged unless their literal claims are still true and every
   extra operation is separately declared and charged.

4. **An oriented bound is not automatically the old rounded-program proof.**
   V1's sign-selected AABB corners enclose the particular staged binary64
   subtract/multiply/add/add/constant program by monotonicity in raw coordinates.
   Rotating coordinates or directions and applying the same formula does not
   establish equality of rounded programs. The new bound needs its independent
   conservative-enclosure argument, including rounding/error fallback. Keep the
   original staged raw-row evaluator and minimum-original-row tie rule if exact
   V1 output equivalence is intended. Label leaf evaluation arithmetic separately
   from the new node-bound program; an unchanged arithmetic string is not proof
   that the new pruning operation is valid.

5. **Refusal behavior is part of the API.** Query failure invalidates the index;
   registry failure invalidates the registry; public context failure clears all
   candidates/projections/families while retaining attempted counters. Reusing
   `_base`, exception-type tests or final-report helpers blindly can publish a
   V3/V1 refusal label, collapse a meaningful V2 reason into a generic exception,
   or lose constructor work. A new success path without its complete late-failure
   path is not integration.

## Constructor/work/memory obligations

The existing standalone owner copies raw points once (24N bytes), owns row
identities (8S), and charges validation before operations. Its build charges
12S root coordinates, S sort coordinates and 6S leaf-box coordinates, in addition
to row/index/sort/node work; each evaluated row adds9 coordinate units. Each
node currently owns72 bytes of numeric arrays. These are attempted abstract
charges, not a count of every CPU instruction or a process-RSS theorem.

Context V3 avoids another whole raw copy/hash. `_inputs` owns and binds the raw
buffer; each distinct index owns its support permutation and node payload.
Construction overlap with original/sorted row buffers is precharged. Parent
`_ContextWork.memory` dynamically includes all live index payload, registry
metadata and one index scratch allowance even when invoked by inherited input,
CAD projection, cache or report helpers. Its parent fixed scratch is separate
from the extra index scratch. No completed index is evicted or its work refunded.

V2/V4 must preserve that combined admission, not just pass the standalone
134217728-byte cap. The context cap remains268435456 bytes, with at most129
indices,4000000 cumulative indexed memberships,3072 queries,16000000 node visits
and256000000 extrema coordinate units; old row/index/hash/sort caps remain
independently applicable. The same query may become more expensive at build time
and cheaper at traversal time. All work still counts even if pruning is useless.

Before each new oriented operation, account its actual live numeric arrays and
bounded temporary shapes: orientation/basis data, node extents/error bounds,
transformed support scratch, selection/sort scratch, and old/new buffer overlap.
Do not replace the original shared raw owner with transformed coordinates under
the old point digest. If orientation fields enlarge per-index Python records,
reconcile fixed metadata allowances and charge encoded report/cache payload too.
The current65536-byte scratch/32-entry traversal assumption is reusable only if
the new algorithm still satisfies it. Any new work categories need initialized
statistics, a parent-charge mapping and a finite maximum or a documented charge
into existing bounded counters; they may not be free or unlimited because they
are called preprocessing. Preserve attempted-over-cap refusal before allocation
and completed-after-success counters.

Exact support alias handling must remain: bytes-backed immutable arrays; strong
alias references preventing ID reuse; support digest plus exact row equality for
a distinct alias; whole-buffer and support scopes distinguished; minimum raw-ID
ties independent of internal permutation. Projection-cache hits must still
resolve/authenticate support aliases first. A private V4 provider/index identity
must prevent accidental cross-version cache reuse. Release the registry on both
normal return and failure without deleting its captured diagnostic ledger.

## Minimal integration seam, after mechanism efficacy

Prefer a new private closed provider plus a small versioned orchestration layer.
The public caller selects a literal supported mode and an exact owned budget
type. It must not supply arbitrary factories, callbacks or report strings that
choose the implementation. The provider owns the actual V4 builder, configuration
snapshot/validator, publication constructor/binder and index descriptor contract.
Only code-selected, reviewed providers may be installed; no monkeypatching,
`__globals__`/function reconstruction, `sys.modules` replacement or mutation of
frozen V13/V14 is an integration mechanism.

The new orchestration must explicitly pass that provider at every dependent
call, not wrap an unchanged public function which still calls its old globals.
Reuse the frozen numerical helpers for input/support binding, schedule, normal
families, finite prefilter/refinement and measurement. Where those old functions
lack a dependency parameter, a small new reviewed orchestration function is
necessary; the lack of a parameter cannot be wished away. Initializer and workflow
orchestration may share one new adapter module instead of copying every old
module. Keep public implementation identities distinct (for example initializer
V14 and workflow V15) even if physically colocated. Exact naming is deferred.

The resulting minimal route is:

`new opt-in byte entry -> new closed initializer provider -> context V4/shared ledger -> extrema V2`.

The current V7 entry has no context selector: it explicitly constructs
`TwoPlaneContextBudgetsV3`, validates mode `v3`, then invokes V14. It remains
unchanged. A later evaluation adapter must actually bind/call the new entry and
serialize its full request; a helper-only synthetic call or a renamed report is
not that integration. Comparator selection is a separate prospective decision:
do not silently claim both arms changed, or silently relabel V7 as a same-source
control. Production default routing remains untouched and unestablished.

## Compatibility/publication checklist

| Seam | Minimum successor obligation | What can remain frozen |
| --- | --- | --- |
| Index public API/private context constructor | V2 schema/type; actual build/bound/work/memory descriptors; exact raw witnesses; whole refusal | Original row evaluator only if its arithmetic is literally unchanged |
| Context V4 | New `_configuration`, `_base`, owner/work/registry factory, projection dispatch and counter declarations; complete schedule/slot ledger | V1 typed plane/match objects, input/support binders, CAD projection and normal/pose arithmetic where signatures/behavior are unchanged |
| Initializer provider | New implementation/mode/context-schema declarations and exact key/budget validation; V4 builder call; `_two_plane_records` helper validation; publication and fatal clearing | Numerical triplet/prefilter/refinement helpers and V12 registration/prefilter wire shape |
| Byte workflow | New implementation and policy identity; explicit context arguments; all four top-level publication binds plus each selected context's binding; clear selected indices/support on failure | V13 workflow wire schema and unchanged native/measurement transport if their actual fields remain compatible |
| Pure projection adapter | New wrapper schema and exact context request grammar; validate workflow, policies, registration evidence, initialization, helper and selected array bindings; exact-byte reconstruction | Schedule projection V1 and contact projection/scorer V2 when their schemas/geometry semantics are unchanged |
| Evaluation/retention adapter, later | Select actual new entry, request constructor, wrapper/project/revalidate APIs and frozen closure; update report/custody implementation labels and expected schemas together | Original RESULT bytes, qualified publication/supervision algorithms, terminal retention rules, scoring formulas and null semantics |

V13 initializer `_CONTEXT_DECLARATIONS` and projection V2's independent literal
`CONTEXT_DECLARATIONS`/`CONTEXT_BUDGET_MAXIMA` currently require exact V3 metadata.
Both sides of the new contract must agree on build/bound/arithmetic and work
declarations. Add a separate node-bound/index identity if the leaf arithmetic
label alone no longer describes the new mechanism. Bind the complete declaration
and its canonical digest at workflow, policies, registration evidence and nested
initialization. For every selected measurement context, bind its exact requested
configuration again; triplet-origin and empty pair schedules do not exempt that
binding. Preserve the exact empty-schedule sentinel and early-refusal absence
rules rather than requiring completed metadata from work that never reached it.

It is unnecessary to rename every legacy wire schema. Current V14 deliberately
uses `CAD_GUIDED_MESH_OFFLINE_V13` and V13 initializer uses the V12 registration
wire contracts while separately binding implementation identity. Preserving
those shapes permits the unchanged contact scorer to remain untouched. Conversely,
passing a V4 result through existing projection V2 will reject its hard-coded
V14/V13/V3 identities; weakening that frozen validator would destroy historical
binding. A new pure adapter should delegate to unchanged schedule/contact logic
and enforce the new metadata itself. Never edit/re-encode the returned RESULT to
masquerade as V3. Inner contact projection and metric schemas may stay V2; a later
request envelope may retain its transport name only with an explicit unchanged
wire-contract declaration and new full arm-request digest.

For a future evaluation adapter, review these exact V7-source seam equivalents,
without editing V7: `SOURCES` and backend import closure; `STAGE_SAMPLE_SOURCE_PATHS`;
`SOURCE_PROFILE`, `ARM_DEFINITIONS`, comparator/context scope and `arm_request`;
`workflow_entry_and_policies`; `SCORING_SOURCES` load order/module selection;
`SCORING_TRANSPORT`; `compact_result` and `reconstruct_scoring_input`;
scorer-request validator; terminal-fill wrapper acceptance; report enrichment;
parent artifact verification and schema-derived inventory. The new numerical
files must be present in inference's allowed frozen closure, and the new pure
adapter in coordinator/parent/scorer closure. No inference access to Gold is
introduced. Bind new paths/hashes/implementation strings and actual control
lengths in a future freeze; do not inherit the number70 or its map merely because
the custody algorithm is unchanged. Actual byte caps and final vetoes remain.

## Smallest acceptance checklist; no additional broad gate

1. Finish the independent oriented-bound argument and focused invented fixtures:
   exact staged extrema/raw-ID ties, adversarial/degenerate rounding, conservative
   fallback, constructor/query limits and whole refusal. Demonstrate useful work
   reduction on a declared fixture without hiding added build/memory cost. This
   is the standalone mechanism decision, not a physical score.
2. Once worthwhile, connect V4 using the shared owner and cumulative parent
   charges. Exercise more than one support/index, a distinct exact alias, a cache
   hit, a later constructor/query refusal and registry release. A standalone
   success does not demonstrate aggregate context-cap admission.
3. Qualify only the new dependency/publication delta through the real opt-in
   entry: prove the actual V4 builder is reached, old/mismatched declarations
   fail at the intended binder, and later failures clear output/selected support.
   Use coherent invented input metadata and preserve the real scan/schedule/
   context binders; do not manufacture a successful path by bypassing them.
4. Round-trip the new projection from exact RESULT bytes with unchanged contact
   inner semantics; verify returned/refused/early-unavailable cases and selected
   bindings. No unchanged broad suite, new runner or physical invocation is
   authorized by this checklist.

Oriented extrema accelerate only the scan-extrema proposal portion. The source
still performs finite-face prefiltering and global-registration refinement after
context construction, with their own budgets and possible refusals. This change
does not itself repair those algorithms, establish a pose, observe a bend or
promise a positive geometry result. Preserve all old versions and terminal
evidence; decide the exact integrated successor only after the mechanism earns
that next step.
