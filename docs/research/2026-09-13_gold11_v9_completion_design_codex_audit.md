# V9-A third-plane completion: independent design audit

Date: 2026-09-13. Reviewer: Codex finite-start audit agent. Fable 5.1 worker
`1519fdc2` remains sole engine/test implementation owner.

## Verdict

**Defensible for bounded source implementation and independent synthetic
validation, conditional on the requirements below. Not a source acceptance,
physical PRE, correct-pose finding, or F1 improvement claim.**

The retained two-of-three pattern is a more direct reason to try third-plane
completion now than the generic cone-feasible witness counterexample in my
previous report. The two ideas address different losses: completion changes
the missing finite third correspondence after a useful seed exists; witness
repair addresses assignments that never reach the prefilter. There is no need
to force witness repair into V9-A. Preserve the predicted negative where no
two-of-three parent exists, and keep that other mechanism available later.

The proposal already gets two crucial decisions right: recompute **all three**
finite supports after translating, and keep refinement/final acceptance gates
unchanged. The main unresolved design gates are exact signed-plane mathematics,
near-nullspace handling, explicit joint budget allocation, and ranking semantics.

## Scope and inspected state

Read AGENTS.md, the newest CURRENT_STATUS checkpoint, the whole
`2026-09-13_fable51_v9_diagnosis.md`, and the relevant frozen initialization,
plane-search and pair-family seams. The earlier repository map and Astra
workflow remain applicable. The diagnosis SHA256 at this review is
`db9e154bcfff05af4679d7c2a458f182cf50529604b19843383a6b2f4822aa1a`.

The named `backend/cad_guided_plane_initialization_v15.py` was **not present**
at either source-path check. Consequently there is no V15 implementation hash
or tested closure to accept here. No imports, tests, numerical probes, physical
data, terminal attempt/control artifacts, or private caches were accessed.
Only this new review document was written; the previous review is unchanged.

## 1. What the retained evidence does and does not establish

The released diagnosis reports two finite supports with coverage 1.0 and a
third completely excluded by its assigned face box in five scans, together
with no scheduled alternative third CAD face for those same first two
correspondences. This directly motivates an alternative finite-face assignment
lane. It does not require an inference-time case identifier or Gold.

However, the statement that this establishes "the wrong finite face on the
right plane" is too strong. The triple-plane seed is built to align nominal
infinite-plane intersections (`cad_guided_plane_search_v6.py:563–568`), so
near-zero third anchor-to-plane residual is partly a construction property.
Two local supports can fit an incorrect global pose, a repeated feature, or an
incorrect first pair. The third scan patch can itself be an imperfect proposal.
The justified conclusion is narrower:

> At this generated pose, the third assigned finite face cannot support the
> patch under the declared gate, despite two other finite assignments passing.
> The present correspondence schedule never tested an alternative completion
> of that specific parent pair.

Likewise, the "feasible completions" table establishes **centroid-box proxy
survivors**, not feasible row-level support, feasible registration, or correct
poses. A centroid inside an expanded box does not imply 75% of raw rows inside
the box, and even complete box containment does not imply proximity to a
nonconvex face with holes. Conversely a centroid can be outside while more
than 75% of rows are valid and a few distant rows move the mean. Thus centroid
tests may propose/rank diagnostics, but cannot replace the exact prefilter or
reject a candidate as geometrically impossible without an additional proof.

The diagnostic 8-degree compatibility filter and 0/5 mm extra centroid
allowances are consumed-development analysis choices. They must not silently
become the V9 inference policy. The proposed runtime says it will use the
existing rowwise Wahba allowance; bind that exact calculation rather than
copying the diagnostic constants.

## 2. Correct axial completion formula and sign conventions

Use the canonical anchored equations already defined by search V6:

`n_s dot (x - a_s) = d_s`, with unit `n_s`, for the failed scan plane;
`c_j dot (y - a_j) = d_j`, with unit `c_j`, for a candidate CAD plane.

CAD face fitting currently supplies `d_j=0` and an area-centroid anchor
(`cad_guided_plane_initialization_v8.py:178–179`), but tests should retain the
general form so sign/anchor mistakes are visible.

For the unchanged parent pose `y=R*x+t`, define:

```text
w_s = a_s + d_s*n_s
y_s = R*w_s + t
m_j = s_j*c_j,                    s_j in {-1,+1}
b_j = s_j*(c_j dot a_j + d_j)
u   = normalized cross(m_1,m_2)  for the two passing CAD planes
D_j = m_j dot u
lambda_j = (b_j - m_j dot y_s) / D_j
t_j = t + lambda_j*u
```

Then `m_j dot (R*w_s+t_j) = b_j` in exact arithmetic. The numerator has units
mm, the denominator is dimensionless, lambda is mm, and `lambda*u` is a world
CAD-frame translation. Row-array application remains `x @ R.T + t_j`.

Requirements:

- Recompute the candidate's `s_j` from the new face's oriented normal and the
  rotated scan normal. Do not copy the old third face's sign blindly. Use
  `atan2(norm(cross(R*n_s,m_j)), dot(R*n_s,m_j))` and the same per-row cone-plus-
  extra allowance as search V6 (`:530–531,550–559`). Retain the chosen sign.
- Apply `s_j` to **both** the target normal and target equation right-hand
  side. Applying it to only one changes the plane. If it is applied to both,
  the sign cancels from the lambda quotient, as it should.
- Normalization must preserve both coefficients: search V6 normalizes the
  normal **and the anchored offset** (`:88–102`). For a non-unit input,
  `a_s+d_s*n_s` is not necessarily a plane witness. Use the owned canonical
  equation, or normalize correctly; do not assume an arbitrary centroid lies
  on the declared scan plane when `d_s` is nonzero.
- Flipping the sign of u must flip lambda and leave `t_j` invariant. This is
  a useful independent test of orientation bookkeeping.
- An anchor hit is not whole-plane coincidence when `R*n_s` and `m_j` differ.
  Under nominal angular mismatch it aligns one witness only. The completion
  is still a seed; the whole-support finite query must decide its usefulness.

The diagnosis states the intended shift verbally but does not freeze this
exact formula, arithmetic program, or residual checks. These must appear in
the implementation/design before source acceptance.

## 3. Rank, nullspace and near-parallel cases

For a nonparallel pair, u spans the translation nullspace of its two CAD plane
equations. Moving along u preserves their infinite-plane signed distances,
but not their finite supports. That is precisely why both previously passing
supports must be requeried after the shift.

An arbitrary nonzero cross product is insufficient for stable numerical use.
Require a declared nominal rank/conditioning policy for the pair, consistent
with or explicitly versioned from the existing pair-family checks
(`cad_guided_two_plane_context_v1.py:354–368`). If the two passing normals are
parallel or unresolved within the stipulated cones, do not invent a unique
axis. They leave a larger translation nullspace and require a different lane.

`abs(D_j)` is a normalized triple-product measure of whether the third plane
constrains the remaining axis. A positive, dimensionless, predeclared floor
should exclude near-null cases from this particular solver. Use a separate
disposition such as `AXIAL_COMPLETION_UNRESOLVED_DENOMINATOR`, not proof that
the candidate cannot match. A tiny numerator does not justify dividing by a
tiny denominator or declaring zero shift.

Distinguish these cases:

1. `D_j != 0` with sufficient margin: one nominal lambda is defined.
2. `D_j == 0`, numerator nonzero: this fixed-rotation axial family cannot put
   that witness on the candidate plane; a different rotation/family may exist.
3. Both zero: the candidate plane adds no axial constraint; infinitely many
   lambdas remain. Choosing lambda zero is a heuristic proposal at most, not
   rank-three completion.

The denominator floor is not uncertainty calibration. If `lambda=N/D`, then
small errors obey approximately `delta_lambda=delta_N/D-N*delta_D/D^2`.
Passing a floor does not bound either error without an input model. Report
lambda, denominator and postconstruction residuals; do not publish a pose
uncertainty interval from nominal rank alone. Reject/abstain on nonfinite or
out-of-domain poses before raw-array transforms and before expensive queries.

## 4. Finite faces, all-support revalidation and provenance

V14 binds three distinct scan IDs and three distinct CAD plane IDs for each
triplet, with per-support raw-row and triangle digests
(`cad_guided_plane_initialization_v14.py:534–569`). A completion must preserve
those invariants. In particular the replacement CAD face cannot be either of
the two passing faces. A new coplanar CAD face is a distinct **finite** domain,
even if its infinite plane and resulting translation are identical. Do not
merge faces or reuse the former third face's finite-query evidence.

Recompute the complete represented transformed raw rows for all three supports
at `t_j`, build/bind the replacement face's complete finite index, apply the
same box screen, then the same finite distance query where required. Neither
the two old `PASSED` flags nor a lambda of zero exempts changed bindings from
validation. If exact byte/binding cache reuse is introduced, independently
prove its key includes all pose, support and face bytes; a parent index alone
is not enough.

Parent eligibility must require a completed ordinary TRIPLET prefilter with
exactly two actual finite `PASSED` supports and one completed rejected support.
A prefix before a fatal query, a box-only possible support, or a generic
unknown row is not a parent. By design do not recursively complete completion
children; otherwise both budget and scientific meaning change.

Retain for each child: parent hypothesis index and lane; immutable original
third assignment; passing support indices; replacement face/sign; source and
target canonical equations; u, numerator, denominator, lambda and new pose;
normal-gate calculation; complete attempted/omitted/rejected status; and all
new raw/face bindings. Never change the terminal parent's rejection to pass.
If several parents/assignments generate exactly the same pose, retain every
origin when deduplicating pose groups, as V14 already does for ordinary starts
(`:664–672`). A duplicate pose does not make the finite assignments identical.

## 5. P1 if omitted from implementation: joint resource accounting

The reported V8 runs already reserve all 8,192 hypothesis records. The proposed
completion phase therefore **cannot append to that same capped ledger while
claiming unchanged joint hypothesis capacity**. Before implementation
acceptance choose one explicit new-version policy:

- Reallocate a predeclared part of the joint budget from ordinary generation to
  completion, documenting that the base proposal set changes; or
- Keep the ordinary phase unchanged and use a separately bounded additive
  completion ledger, with an explicit new combined cap, report envelope and
  runtime preflight. This is a new configuration, not unchanged V8 limits.

Either can be legitimate for a separately frozen successor. Neither is allowed
to change a started or terminal attempt. Do not adjust the allocation after
observing the new physical score.

The parent/per-parent/total completion quotas are not enough by themselves:

- Bound and charge every parent scan, CAD face-normal test, candidate record,
  sort, exact-pose key, index construction and query. Enumeration/sorting over
  every face precedes a small output quota and still costs resources.
- Original and completion phases share cumulative prefilter row/call/work,
  face-index build, memory and report accounting if those limits are claimed
  unchanged. Do not reset counters by invoking a fresh prefilter helper.
- Preflight all scheduled children with three complete support populations;
  box exclusions save actual finite work but do not turn an incomplete planned
  denominator into completion. Invalid input/resource failure must invalidate
  attempted positive prefixes as required by the declared failure policy.
- Ordinary `max_refined_start_groups` limits candidate count, not CPU cost.
  A previously all-refused scan can now launch expensive refinements, and a
  shifted start may converge more slowly. Preserve the external CPU/RSS limits
  and retain new timeouts honestly.

V14's all-support preflight and completion accounting live at `:553–574` and
`:679–690`; its lane validator currently accepts only TRIPLET and
TWO_PLANE_CONTEXT (`:537–540`). A new lane must update and test every connected
validator/publication seam deliberately. Do not make the validator permissive
to arbitrary lane strings.

## 6. P2: smallest-lambda priority is not completion coverage or uniqueness

Sorting by `abs(lambda)` is a defensible deterministic local-seed prior, but
its meaning must be stated. A small shift is not more likely correct simply
because two local supports already pass. Nearby parallel sheet skins, repeated
features and multiple disconnected coplanar faces can fill the quota with
unhelpful near-zero candidates; the only useful completion can be farther away.
For coplanar faces, many lambdas tie and face ID becomes the selection rule.
Changing CAD face numbering can then change whether the correct finite face
is even attempted.

At minimum freeze this as a nonexhaustive local completion schedule with
per-parent omitted counts and a no-uniqueness claim. Independently test a
farther valid completion behind many invalid near-zero candidates and a
coplanar-face ID permutation. A small geometry-diverse reservation, or a sound
cheap finite-support bound before consuming the expensive-query quota, is
preferable if it can stay within the declared budget. Do not replace the
ranking with centroid-box exclusion; that proxy has no row-coverage guarantee.

Two different completions can both genuinely pass finite checks. Preserve both
before the unchanged exact-group/refinement schedule. If refinement or context
quotas omit one, retain that omission and do not infer global pose uniqueness
from the surviving subset. The existing conditional-context semantics are
appropriate; completion must not strengthen them into acceptance authority.

## 7. Focused independent acceptance matrix

This is a proposed test checklist, not an executed or frozen experiment:

1. Analytic orthogonal first pair and a third plane with a known axial shift;
   nonzero scan offset/anchor, non-origin translations and arbitrary proper
   rotations. Independently compute the true transformed raw points.
2. Flip each CAD plane equation's normal and offset together, and flip u;
   represented pose should agree within the declared arithmetic contract,
   with the correct recorded sign/lambda changes.
3. Nearly parallel first pair, nearly axial-null third normal, exactly zero
   denominator with zero/nonzero numerator, and finite overflow-domain cases.
4. Two old supports pass at the parent but one falls off its finite trim after
   the completion shift. The child must fail despite the original flags.
5. Coplanar replacement face with a hole, a centroid inside the face box but
   raw coverage below 75%, and centroid outside with sufficient raw coverage.
   None may bypass or be wrongly excluded from the real finite prefilter.
6. Several near-zero wrong replacements before one farther correct one;
   same-plane face-ID permutations; repeated geometry with two genuinely
   passing completions. Test omission and ambiguity, not only recovery.
7. Parent with one/two/three passes, a failed/incomplete parent, duplicate CAD
   face reuse, stale support/pose/face hashes, and a completion child offered
   recursively. Only the declared original two-pass parents may enter.
8. Tiny independent per-parent, total, row/call, report and index-build caps;
   ensure cumulative accounting and explicit failure/omission without prefix
   salvage. Verify baseline parents and all their dispositions remain intact.

An end-to-end synthetic recovery requires a child to pass all three finite
supports and a declared downstream pose check, not merely produce lambda or
pass a centroid proxy. Keep impossible and ambiguous cases. These focused
tests can validate the new mechanism without another broad research phase;
physical recovery still requires the separately frozen audited successor.

## Snapshot and handoff

Inspected source SHA256 values:

```text
9e3ca41edd05bcc0b6fcdeb9ef0447dbacfe58df8c9458da20f9bb12c3d925bf  backend/cad_guided_plane_initialization_v14.py
aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676  backend/cad_guided_plane_search_v6.py
9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a  backend/cad_guided_two_plane_context_v1.py
```

The formula, narrow eligibility, all-support revalidation and explicit
accounting requirements are the immediate implementation handoff. I support
trying completion ahead of the earlier generic cone-witness recommendation
because of the released two-of-three evidence, without treating four
centroid-box proxy survivors as four recovered scans. The diagnosis's V9-B
material-side/arc selection claims are outside this assigned review; no
endorsement of those claims is implied by reading the whole diagnosis.
