# V4 proposal-only optimizer: source design, not an implementation

Status: independent source-design recommendation by `cad_support_lineage`.
Only this new document is authored. No engine, fixture, active connected
candidate or any of its91 sources changes. No project import, simulation,
test, target/native input or retained scientific artifact was accessed.
This is not an experimental PRE, speed measurement or execution release.

## Decision and precise scope

Introduce one future module, proposed name
`backend/cad_guided_proposal_optimizer_v4.py`. It owns a bounded dual-index
query session and proposal refinement only. Reuse the frozen
`cad_guided_global_registration_v2._refine` function with a new strict nominal
query callable. Do **not** call the V2 public registration entrypoint: that
would substitute PCA starts, selection and acceptance semantics
([V2 registration](../../backend/cad_guided_global_registration_v2.py#L244)).

The future initializer retains its original preparation, selected starts,
start order, both iteration budgets,48-refinement quota, inputs and scientific
policies. It substitutes the proposal callable at the two refinement calls,
then invokes unchanged V3 `_diagnose` and unchanged connected `_context_support`
on the new final pose. That is the whole algorithmic intervention. It is neither
a new approximate final-distance policy nor an alternate way to find starts.

At present initialization refuses before index/refinement construction when
there are no finite-prefiltered starts
([initializer](../../backend/cad_guided_plane_initialization_v12.py#L930));
this optimizer cannot fix that failure. It also cannot recover valid poses
outside the bounded generated/selected set or prove convergence-basin recall.

The source opportunity is to replace at most60 iterative interval queries
(30coarse+30fine) with nominal queries while retaining two final V3 diagnostic
queries and, for connected contexts, one further V3 proximity query. Early
stops may issue fewer calls. Backtracking uses fixed materialized targets and
does not issue another nearest-surface query per trial. Three versus63 interval
queries is a call-count observation, **not21x speedup**: V2 still performs two
BVH passes, finite-edge minimization and all-pair tie work.

## Source-grounded mathematical boundary

V2 chooses globally trimmed nominal distances after excluding its engineering
normal/point ambiguity flags; V3 chooses definite upper-bound support after
also excluding uncertainty flags
([V2](../../backend/cad_guided_global_registration_v2.py#L120),
[V3](../../backend/cad_guided_global_registration_v3.py#L257)). Thus a nominal
proposal changes the selected rows, correspondences and trajectory, even when
the final pose happens to be close. Neither trajectory equivalence with the
current V3 optimizer nor a nested accepted-output set is claimed.

Preserve the existing coarse weighted-Kabsch mathematics and fine Huber IRLS,
center-based fine motion caps,8-step fixed-set backtracking and1e-8 motion stop
([V2 refinement](../../backend/cad_guided_global_registration_v2.py#L131)).
The coarse update remains the existing full Kabsch update; do not silently
apply the fine-step cap there. The fixed-target objective is
`mean(Huber(||R p_i+t-q_i||))`, with weights proportional to
`min(1,delta/||p_i-q_i||)` in the fine step. It is not certified descent of
exact nearest-triangle distance. A stationary or iteration-limited pose is a
proposal, not accepted registration. Insufficient nominal refinement support
returns a valid unchanged/current pose for final checking, not a fake absence
or an automatic bypass of final qualification.

The final V3 diagnostics are recomputed on the original declared registration
rows and original refine subset, using the V3 index and exact unchanged policy
([diagnostics](../../backend/cad_guided_global_registration_v3.py#L405)).
Preserve definite/possible denominators, outward coverage/P95/RMS, uncertainty
and engineering rank. Their bounds concern represented transformed binary64
points and represented triangle hulls, not transform-rounding error, latent
scanner coordinates, native CAD or calibrated uncertainty. No epsilon is added
or removed by the proposal optimizer.

One reporting nuance needs an explicit provenance label: the final weighted
rank recomputes V3 endpoint targets, trim and Huber weights
([rank](../../backend/cad_guided_global_registration_v3.py#L430)). After nominal
V2 updates, these are **not literally the final V2 iteration's IRLS support,
targets or weights**. Keep the V3 calculation unchanged, but publish
`final_rank_weight_scope=V3_ENDPOINT_RECOMPUTED_NOT_LAST_V2_PROPOSAL_WEIGHTS`.
Do not rewrite those diagnostics as proof about the actual last proposal step.

## Smallest clean API and dependency direction

Proposed future API, names subject to implementation review:

```
prepare_proposal_optimizer_v4(
    owned_cad_triangles, owned_source_face_ids, *,
    proposal_surface_budgets: triangle_surface_v2.TriangleSurfaceBudgets,
    final_surface_budgets: triangle_surface_v3.TriangleSurfaceBudgets,
    combined_query_budgets: SurfaceRegistrationBudgetsV3,
    build_and_payload_budgets: ProposalOptimizerResourceBudgetsV4,
) -> {status, session_or_none, evidence, reasons}

session.propose(scan_subset, rotation, translation, *, diagonal, policy, coarse)
    -> (rotation, translation, iteration_count, convergence_state, last_step)
session.final_query(represented_points) -> original validated V3 query result
session.snapshot_evidence() -> bounded proposal/final/combined/build ledgers
```

The session is an in-process opaque object, not serialized index descriptors
that a caller may reconstruct and pretend are owned geometry. Its immutable
configuration and geometry are distinct from its mutable bounded work ledger.
Admission snapshots exact typed budgets against independent literal maxima;
later caller `object.__setattr__` mutation cannot change limits. Build/payload
budgets must be explicit in the first implementation, not permissive defaults.
Before V2 refinement, validate/snapshot the original policy, bounded scan subset,
finite proper input rotation/translation, diagonal and strict coarse flag using
the existing V3/caller admission contracts. V2 `_refine` alone does not provide
those V3 entry checks; its reuse must not enlarge the public input domain.

The module imports V2/V3 registration/triangle math, never an initializer or
measurement wrapper. The future initializer calls V3 `_diagnose(...,
session.final_query,...)` and its own existing `_context_support(...,
session.final_query,...)`. This prevents a circular dependency and avoids copying
their scientific gate logic into the proposal module. Existing globals are not
rebound, patched or temporarily swapped; no V2 query is disguised as V3.

For integration, use a **new explicit initializer/workflow version**, or a
separately reviewed opt-in branch whose default remains exact V3. Publish a
strict requested/returned optimizer configuration at initialization,
registration and byte workflow, and validate it before measurement. Selection
between branches must occur before dispatch, never in response to a failure.
Do not implement either integration option during this design task.

V2 `_Queries` is not reusable unchanged for the new session. It retains caller
budget objects, trusts absent work fields as zero and has no strict complete
query/schema validation
([V2 adapter](../../backend/cad_guided_global_registration_v2.py#L83)).
Use a small V4 nominal-query adapter calling the frozen V2 primitive directly,
then a bounded V2-schema/identity/telemetry validator. This is more auditable
than layering a shared allowance on a wrapper that already spent a separate
counter. V2 `_refine` already accepts a callable; no refinement-body rewrite
or monkeypatch is needed. The final adapter uses the unchanged V3 validation
contract, with allocation supplied by the same shared ledger.

## Two independently bound indexes, built once

Build both indexes once from one captured immutable triangle/face-ID snapshot,
with identical triangle order, orientation, source IDs, frame and units. Verify
their returned geometry and identities match that snapshot. Bind triangle bytes,
ordered face-ID encoding, per-index schema, exact budget snapshots and build
diagnostics separately. Byte/SHA identity is in-process custody, not independent
native-source authenticity. Existing callers must still bind native export
and CAD sampling provenance.

V2/V3 private index types are incompatible: V2 `c_u/c_v` are dimensional while
V3 uses normalized local coefficients; V3 also has per-triangle scales/guards
([V2 build](../../backend/cad_guided_triangle_surface_v2.py#L166),
[V3 build](../../backend/cad_guided_triangle_surface_v3.py#L183)). Never cast an
index, share a budget by expanding every V3 field into V2, strip an interval
field and call it nominal, or synthesize V3 uncertainty fields as false.

Build V3 first as the final geometry contract, then V2 with the residual combined
build/payload allowance; neither index is published usable until both succeed.
A late second-build refusal clears the session and usable transforms while
retaining both build attempts. No recentering, rescaling, triangle deletion,
point sampling change, native healing or fallback is part of this successor.

**Important operating-range loss:** V2 reinstates the soup-wide bound
`4096*eps*(coordinate_magnitude+maximum_edge)/minimum_quality <=
1e-5*minimum_edge`, and its query guard additionally depends on the query's
coordinate magnitude
([build guard](../../backend/cad_guided_triangle_surface_v2.py#L186),
[query guard](../../backend/cad_guided_triangle_surface_v2.py#L313)). V3's local
competitive interval-width test does not imply this V2 condition
([V3 query](../../backend/cad_guided_triangle_surface_v3.py#L488)). Therefore a
previously valid V3 soup or translated proposal may fail V4 before final checks.
That is a declared negative, not silent numerical repair. If this operating
range is unacceptable, abandon this specific design and separately review a
locally anchored nominal primitive; do not disguise one as frozen V2 reuse.

## Combined resource contract

There is one whole-registration query allowance, not old totals independently
granted to proposal and final stages. Keep V3's existing whole-registration
maxima as the ceiling for summed query calls/rows, node visits, triangle pairs,
normal pairs, certificate pairs and weighted work. Each phase additionally
obeys its own exact primitive/per-call bounds. In particular:

```
Qtotal = Qnominal + Qfinal
Rtotal = Rnominal + Rfinal
Wnominal = node_visits + 8*triangle_pairs + 3*normal_pairs
Wfinal = node_visits + triangle_box_pairs + 8*triangle_pairs
         + certificate_pairs + 3*normal_pairs
Wtotal = Wnominal + Wfinal
```

Proposal certificate count is zero because that phase performs no interval
certificates, not because an unavailable counter was guessed. Preserve both
primitive work meanings and separate completed-query counts; do not claim work
units are CPU instructions. Before each call, precharge attempted calls/rows,
allocate every work field as `min(per_call_cap, shared_remaining)`, validate
returned telemetry and debit it even on a primitive refusal. An over-allocation,
missing/malformed telemetry or reported incomplete search is fatal. Retain
known counters plus an explicit unknown-work flag when exact work is lost;
never reset that unknown to zero or expose a favorable completed prefix.

Final queries share the residual allowance too. Running out during final
diagnostics/context support invalidates the whole call. Do not silently shorten
iterations, increase limits or refill an unscheduled candidate to manufacture
room for qualification. A prospective up-front reservation of known final
call/row counts would itself need a declared policy; it is not in the minimal
variant. Unchanged call maxima already cover48*63 calls at30+30 iterations;
triangle/tie work can still exhaust the cap.

Index construction has its own **combined** nodes, triangle visits, captured
input bytes, retained-array bytes and live temporary-payload ceiling. It cannot
receive two complete old build allowances without an explicit new protocol.
The conservative initial proposal is to share the prior global build ceilings
across both builds; any resulting capacity reduction is disclosed and tested.
Keep prefilter face-index work in its existing distinct ledger, and include it
in the higher-level workflow/outer experiment report rather than hiding it.

Source-derived retained numeric-array sizes are336*T+80*B2 bytes for V2 and
520*T+80*B3 for V3, thus856*T+80*(B2+B3), before face-ID/Python objects, snapshots,
builder scratch and query arrays. AtT500000 and both declared node maxima131071,
this numeric payload alone is448,971,360B. This is not an RSS bound. Preflight
must also reserve immutable-copy peaks, overlapping live buffers, bounded query
batches and report serialization; the first implementation must derive and
freeze explicit payload limits, not use this retained-byte subtotal as proof
of4GiB safety. The external process RSS/CPU/wall limits remain separate and
unchanged until a new exact experiment protocol is reviewed.

## Failure and publication semantics

Use one fatal successor exception, proposed `_OptimizerFailureV4`, compatible
with the caller's V3 fatal-query path. Explicitly translate V2 `_SurfaceFailure`
and any adapter/primitive budget, conditioning, incomplete-search or schema
failure into it, preserving phase and underlying reason. Catch it before the
candidate `ValueError`/linear-algebra branch: V2 and V3 currently define distinct
exception classes, so an unadapted V2 failure would otherwise look like an
ordinary candidate numerical failure
([caller catches](../../backend/cad_guided_plane_initialization_v12.py#L978)).
One workable declaration is subclassing V3 `_SurfaceFailure`, plus explicit
translation at the V2 boundary; this is not a mutation of the frozen class.

Ordinary Kabsch/SVD numerical failure remains recorded against that candidate;
the unchanged complete-candidate/publication gate prevents the remaining
successful subset from being published as fully evaluated. A fatal query/build
failure stops the call, records unattempted candidates, clears every usable
transform/context index/owned measurement role handle and marks complete flags
false. Preserve attempted scalar counters, prior diagnostics and failure rows
as non-usable evidence. No V3-only fallback and no outcome-based scheduler refill.

Candidate evidence must distinguish `proposal_optimizer_mode=NOMINAL_V2_ONLY`
and its per-step nominal diagnostics from actual completed final
`COMPLETE_V3_INTERVAL_SUPPORT_POLICY` diagnostics. The latter marker may be set
only by real V3 completion; proposal stationarity must never synthesize it.
Bind each final pose to its original start/group provenance and both index
bindings. Byte workflows reject stale/missing/wrong optimizer configuration
before any measurement. Proposed new metadata is additive, bounded and typed;
it must not overwrite existing final-distance policy labels.

Connected mode must retain **all** generated contexts passing its unchanged
distance-only final criteria, even when legacy rank/uniqueness diagnostics
reject an accepted-pose interpretation
([context support](../../backend/cad_guided_plane_initialization_v12.py#L1042),
[publication](../../backend/cad_guided_plane_initialization_v12.py#L1069)).
Keep accepted registration false, transform absent, observed count unknown,
joint configuration/continuous coverage not established and raw scan-frame
measurement. Do not introduce `_select` as a prerequisite for connected mode.
Non-connected mode, if later exposed, must retain V3 `_select` and possible
competitors, not V2's old nominal score winner.

## Minimum independently authored tests before integration

1. **API and custody:** wrong index/budget types, V3-to-V2 casts, face/triangle
   reordering/mismatch, bool/int aliases, snapshot mutation, immutable arrays,
   altered configuration/digests, zero/exact/one-over build and payload budgets.
   Show both builds share one owned snapshot and exactly one build per version;
   second-build failure clears the first's usability and retains its cost.
2. **Proposal equivalence to the chosen baseline:** fixed, well-conditioned tiny
   meshes and identical supplied starts/subsets/policy; V4 proposals agree with
   direct V2 `_refine` using its own V2 query path. Compare transforms, states,
   iteration counts and nominal last-step values, excluding extra metadata.
   Do not demand equality with V3's bound-aware trajectory. Prove no PCA/start
   generator is called and no query occurs per backtracking trial.
3. **Finite geometry:** interior, beyond-edge in-plane, vertex and true-hole
   points; same-face and different-tag coplanar tessellation seams; equal-distance
   parallel sheets with compatible normals; opposite winding/fold normals;
   a near-tie triple whose incompatibility is between two nonrepresentatives.
   Keep all-pair diagnostics and32/33 tie-cap refusal. Nominal flags never become
   final correspondence or continuous-surface certificates.
4. **Range/conditioning:** ordinary translated/properly rotated copies inside
   V2's regime, a large-offset and mixed-scale soup accepted by V3 but refused
   by V2, a valid build followed by a query-guard refusal, and legal sliver/edge
   threshold boundaries. Require explicit V4 refusal, no geometry deletion,
   auto-recentering or V3 fallback. No universal rigid-coordinate invariance claim.
5. **Final isolation:** for fixed final poses, V4's two diagnostics exactly match
   direct V3 `_diagnose` under adequate budgets, and connected support matches
   the unchanged V12 function. Inject optimistic nominal evidence followed by
   V3 outside/straddling/uncertain evidence: final rejection must prevail.
   Test nominal insufficient support followed by valid final proximity without
   interpreting the proposal convergence state as acceptance.
6. **Ambiguity and observability:** distinct supported contexts remain retained;
   accepted-pose/rank failure must not suppress an otherwise valid conditional
   context. Non-connected possible-score overlap still prevents false uniqueness.
   Confirm final weighted-rank provenance is endpoint-V3, not last-step-V2.
7. **Work/fatal faults:** exact and one-over combined limits where each separate
   phase would fit; early and late nominal refusal, final-query failure after a
   previous candidate passed, malformed/refused telemetry and exceptions from
   both versioned classes. All prefixes clear, attempted work survives and no
   absent counter becomes measured zero. Final V3 certificate counts remain
   four per actually tested triangle pair on complete queries.
8. **Publication/measurement:** wrong/missing optimizer mode or index binding,
   late context-binding failure and metadata overflow all veto publication.
   Measurement receives original raw rows and owned surface-role context, never
   nominal closest points or CAD samples as measured answers. Unknown labels
   and every supported competing context remain intact.

Start with a compact pure/mock suite and a few owned-array cases (suggested
largest128triangles/128query rows, at most4 fixed starts and2+2 proposal
iterations per case), under a prospective60wall/45workerCPU/1GiB/1MiB gate.
The exact fixture list, outcomes, work/byte caps and independently authored
oracles must be enumerated before first execution; this is not approval to run
a capacity loop or tune against observed outcomes.

Only after source qualification should a separate paired connected packet use
all four unchanged curved arrays/oracles, identical supplied start ledgers,
policies and resource ceilings for current V3 versus proposalV4. Freeze output
regression criteria before running, including both L endpoint oracles and every
abstention. Record builds, nominal updates, final diagnostics and context support
time/work separately; evaluate measured total CPU/wall/RSS as well as success.
Do not compare unlike scheduler/scan-plane banks or use the current zero-start
cases as evidence that faster optimization recovered correspondence. Native,
physical all11, historical F1 and generalization remain separate later gates.

## Exact source read for this design

The full performance review and both complete registration modules and triangle
primitives were read as source; only the named initializer call/publication
seams were read. These are reference bindings, not a new frozen runtime closure:

| Source | SHA256 |
| --- | --- |
| `cad_guided_global_registration_v2.py` | `5a1b039d17b6d5bb5840ebb16b5ab3d131ca3df1b6c186746804327921627dc3` |
| `cad_guided_global_registration_v3.py` | `088c0b5e9fd940094fd44462e7e5c831299fe2c69b87418d09c362772263bae0` |
| `cad_guided_triangle_surface_v2.py` | `7b5333e18817d30905f3a1ace1cbf1e85129ad904324c5e6ccf2baff665a5cc1` |
| `cad_guided_triangle_surface_v3.py` | `8e3222bd425451e2e89b2fe0083db34a3702577018b3f557b1f026a86add9f10` |
| `cad_guided_plane_initialization_v12.py` | `558f68306eb72b9336ec96944ebc56c8ac56d6eef2bf674fac626adbf3611459` |
| `2026-09-10_proposal_optimizer_performance_review.md` | `88af493468b2ce873edcb1d55c425089d609f3d33cf2a9176cffea90a0470fb2` |
