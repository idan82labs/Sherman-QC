# Held proposal optimizer V4: contingency source review

2026-09-11. **HOLD: not qualified, integrated, benchmarked or promoted.** Consider
only if native/physical runtime becomes a demonstrated blocker, not as a new
roadmap phase. Only this document changed; no project import, test, execution,
native call, raw input or scientific artifact access occurred.

Sources read completely:

- `backend/cad_guided_proposal_optimizer_v4.py`, SHA256
  `56a18b35c3cbe52a5aefb28eaf98ea0a2a24964535f0f81e69a23334560db04d`.
- `docs/research/2026-09-10_proposal_optimizer_v4_design.md`, SHA256
  `b9b1b6d30d4401de06df8a66c4b0c1c638a1dca18970779cd636dc11fa0700cf`.

Relevant V1/V2 updates, V3 diagnostics and V12 publication were inspected;
this is not an execution-closure PRE.

## Mathematical separation

The prototype builds actual V2/V3 indexes from one captured triangle/face
snapshot, calls unchanged V2 `_refine` for proposals, and returns index-validated
V3 results from `final_query`. It does not choose starts, measure, accept a pose,
manufacture intervals or use V2 public registration/PCA.

The reused update is mathematically consistent for the row-array action
`p = x R^T + t`. Weighted Kabsch uses weighted centroids and covariance and
enforces a proper rotation, without fitted scale. In column notation its
centered increment gives `R' = D R`, `t' = D(t-c)+c+d`; thus the center moves by
`d`. Fine backtracking substitutes `D=Exp(f Log(D_full))` and `f*d`. This is a
proper rigid-action path, not a claim to use the coupled SE(3) twist exponential.
Caps use radians and the caller's common length unit, intended mm, without
calibration. Coarse full Kabsch intentionally does not inherit fine-step caps.

Fine weights are `min(1, delta/||p-q||)`; mean radial Huber loss uses the same
materialized targets/retained rows and has squared-length units. Eight fixed-set
backtracking trials need no extra queries. This is not certified nearest-surface
descent, global convergence, or proof that stationarity is a correct pose.

Nominal V2 support/ambiguity selection differs from V3 definite upper-bound and
uncertainty-excluded support. Correspondences, trim, trajectory, competing
contexts and output may all change. No pure-acceleration, equivalent-trajectory,
nested-acceptance or speedup claim is justified. Endpoint rank must be labelled
`V3_ENDPOINT_RECOMPUTED_NOT_LAST_V2_PROPOSAL_WEIGHTS`. Final V3 intervals concern
represented transformed binary64 points and triangle hulls, not scanner error,
pose-rounding enclosure, native geometry or calibrated physical uncertainty.

## Blockers before any consideration

1. **Concrete build-accounting defect.** `_build` lines 363-369 rejects a
   malformed returned diagnostics map with `_OptimizerFailureV4`. Preparation
   lines 591-593 sets unknown work/payload only when the failure is *not* that
   class. Therefore a dispatched build returning malformed telemetry can leave
   `unknown_work=False`, `unknown_payload=False` and undeclared/zero build cost.
   The session still becomes FAILED and unusable: this is false resource
   accounting, not false pose acceptance. A future fix needs explicit dispatch
   and validated-telemetry state: charge trusted counters, mark unknowns after
   unaccountable dispatch, and do not invent work for preflight refusals.
2. **No final-publication integration exists.** `final_query` is a primitive,
   not completion of V3 `_diagnose` or connected `_context_support`. A new
   explicit branch must preserve starts/order/subsets/policies, run both final
   diagnostics and context proximity on each endpoint, bind byte-publication
   provenance/configuration, and clear all usable contexts/roles on incomplete
   evaluation or fatal failure. Retain distance-supported alternatives despite
   accepted-pose rank/uniqueness failure. Proposal stationarity is not acceptance.
3. **Reduced operating range and unproved performance.** Soup-wide V2 scale/
   translation conditioning can reject a V3-valid mesh or query. Both builds
   share node/triangle-visit allowances; all proposal and final queries share
   residual limits without reserved final allowance. Whole refusal can follow
   useful proposals. No recentering, geometry deletion, fallback or quota refill.
   Dual indexes/report encoding need external RSS supervision; payload counters
   are not allocator bounds. Test retained slices/reshaped pose exports against
   weak-reference accounting.

## Smallest independent simulation gate

If the contingency is actually needed, first fix/audit the accounting defect,
then freeze one independently authored source-only suite: at most 128 triangles,
128 query/scan rows, four fixed supplied starts and two coarse plus two fine
iterations per case. Use a prospective 45 CPU / 60 parent-wall / 1 GiB / 1 MiB
output gate; enumerate outcomes, closure and caps before launch. No run is
authorized here.

- Compare V4/direct V2 on fixed finite rectangles/noncoplanar patches: known
  rigid action, no scale, centered fine caps, Huber decrease, state/iteration
  agreement. Do not require trajectory equality with V3.
- At fixed endpoints compare real V3 final diagnostics/context support to the
  direct unchanged V3/V12 functions. Include edges/holes, conflicting sheet
  ties, rank-poor but distance-supported contexts, and an optimistic nominal
  proposal whose final V3 bounds are outside or unresolved.
- Include V3-valid/V2-refused translated or mixed-scale geometry, exact/one-over
  shared build/query/final limits, late malformed build/query telemetry, final
  failure after an earlier candidate passed, input snapshot mutation and export
  lifetime checks. All failures retain honest attempts but no usable prefix.

Only a later frozen paired simulation can assess timing/output regressions on
unchanged starts/oracles. Native/physical gates remain separate. This module
cannot repair missing starts, measurement validity or unseen-family accuracy.
