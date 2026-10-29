# Proposal-only optimizer performance review

Status: source-only independent recommendation, not an implemented or qualified
candidate. The running composite-plane connected V1 attempt and all of its
frozen limits remain unchanged. This note is outside that frozen source bundle.

## Evidence and mathematical boundary

Independent reviewer: `cad_blocker_closure_review`. Main agent checked the cited
call sites in `backend/cad_guided_global_registration_v3.py` and
`backend/cad_guided_plane_initialization_v12.py`. No new experiment, physical
input access or active scientific-artifact inspection supports this review.

At the current 30 coarse plus 30 refine iteration caps, a fully iterating
candidate can issue 63 V3 interval queries: 60 proposal-optimization queries,
two final diagnostic queries, and one connected-context support query. The
surface index is already constructed once per registration. Backtracking uses
fixed materialized targets; it does not perform another surface query per
backtracking step. Every completed current V3 query uses the existing two-pass,
four-certificate-per-tested-triangle-pair accounting.

The optimizer already minimizes a materialized-target Kabsch/Huber surrogate,
not a certified exact-surface descent objective. Full interval certificates
are necessary to preserve its **current bound-aware support selection and
trajectory**, but are not necessary to generate a mere pose proposal. A
separately versioned bounded nominal-correspondence proposal optimizer could
therefore be followed by unchanged final V3 qualification. This does not
preserve convergence basins, recall, or global guarantees: final qualification
cannot recover a valid pose that was never proposed.

The connected workflow publishes conditional contexts using geometric
proximity, not an accepted/observable pose. A successor must not silently add
an accepted-pose requirement or erase supported competing contexts. Preserve
`accepted_registration=False`, unknown joint configuration, unknown continuous
pose coverage, and all generated supported contexts.

## Required invariants for a prospective successor

- Keep proposal correspondence evidence explicitly distinct from V3 interval
  evidence. Approximate distances, ties, normals and convergence are proposals
  only, never final support or measurement evidence.
- Validate every resulting rigid transform and rerun complete final V3
  diagnostics on the original declared registration rows and refine subset,
  plus the unchanged connected-context support query.
- Preserve interval support/residual gates, uncertain/ambiguous competitors,
  complete-candidate resource accounting and failure clearing. A favorable
  prefix after a numerical or budget failure is not a usable result.
- Keep all bend measurements bound to original scan rows; neither CAD-snapped
  points nor sampled CAD targets may supply measured answers or observed count.
- Initially retain the existing rigid-update and Huber mathematics, isolating
  the proposal-correspondence change rather than changing multiple mechanisms.

## Prospective qualification, not permission to execute

Only after the current connected attempt terminates and is independently
audited, decide whether this successor is justified. Freeze one optimizer
variant against the existing optimizer on identical inputs, start ledgers,
policies and caps. Predeclare endpoint/output regression limits before running.

Include all four existing curved cases and independently constructed
edge/vertex and conflicting-normal ties, partial coverage, symmetric competing
poses, near-threshold support, and tessellation/sample-alias cases. Include
deterministic failure and budget controls. Retain every candidate disposition,
conditional measurement, abstention and error. Record proposal time separately
from final qualification, plus query/work/certificate counts, CPU, wall and RSS.

The upper-bound call-count opportunity is 60 fewer interval queries out of 63,
**not a measured 21x speedup**. Query cost varies with pose, triangle/tie workload
and row counts. Prior candidate timing does not establish successor runtime.
This recommendation makes no new F1, physical measurement, historical-score,
uncertainty calibration or factory-release claim.

## Source-only reuse addendum

The independent reviewer identified a narrow prospective implementation:
`cad_guided_global_registration_v2._refine` with that same module's `_Queries`
wrapper, which calls the V2 finite-triangle primitive. Do not invoke V2's public
registration entrypoint: it would introduce different PCA starts and acceptance
semantics. Main read the existing V2 refinement/update equations; the remaining
index-construction details below are independent source-review findings, not a
new execution result.

V2 and V3 index/budget classes are incompatible. In particular their `c_u/c_v`
representations differ (dimensional versus normalized local coordinates). Build
two separate indexes once from identical owned CAD triangles and face IDs; no
casting or blind V3-budget expansion into V2. V2's additional retained numeric
arrays are approximately336 bytes per triangle plus80 per BVH node, excluding
Python objects, build work and temporary copies. Freeze separate proposal/final
ledgers and a combined resource ceiling rather than granting each stage the old
total independently.

V2 reinstates soup-wide coordinate-magnitude/minimum-edge/quality conditioning
refusals absent from V3. Mixed scale and large translation therefore require
explicit negative coverage; do not silently recenter, remove geometry or fall
back. V2 and V3 also have distinct `_SurfaceFailure` classes: translate a proposal
query failure explicitly into the successor's fatal whole-call path, preserving
the failure reason and clearing all usable publication, rather than treating it
as an ordinary candidate numerical failure.

V2 still uses two BVH passes and all-pair tie analysis. Reuse removes interval
certificate work, not all correspondence cost. No index/build/refinement change
has been made, imported, tested or authorized for the active attempt.
