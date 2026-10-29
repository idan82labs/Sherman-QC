# V8 finite-start audit: witness feasibility, axial sampling and proposal coverage

Date: 2026-09-13. Independent Codex source-only review for the root coordinator.
Fable 5.1 worker `1519fdc2` retains sole engine/test implementation ownership.

## Verdict and scope

**The 12 no-passing-start outcomes are not evidence that registration is
geometrically impossible.** The connected source has several explicit,
nonexhaustive proposal restrictions. Two admit simple analytic counterexamples:
an individually feasible normal alignment can fail its single chosen witness,
and a valid finite-trim axial alignment can lie between every scheduled extent
vote. The rejection-only face-box calculation is not the apparent defect.

The highest-value next step is a small independently tested **cone-feasible
normal-witness fallback**, followed by a bounded **finite-support-derived axial
proposal search**. Neither requires relaxing the existing finite-distance or
measurement policies. The source and narrative do not establish which of these
would recover any particular physical case; do not promise twelve recoveries.

Read: AGENTS.md, the newest CURRENT_STATUS checkpoint (02:39 UTC), REPO_MAP.md,
the Astra workflow, the released V8 result narrative, and the narrowly connected
source listed below. No imports, tests, geometry jobs, raw inputs, Gold,
attempt/control artifacts, or private caches were accessed. No engine/test or
frozen source was changed. This is not the physical POST or a new run protocol.

The released narrative reports six scans/twelve arm slots with zero passing
starts; across these it reports thousands of hypotheses, roughly 98–99.9% of
assigned supports box-excluded, and some individual finite supports with
coverage 1.0 but no all-support-passing hypothesis. Those are author-reported
development observations, not independently reconstructed counts in this audit.

## Connected path

`cad_guided_plane_initialization_v14.py:33–55` reuses V12, which reuses V8's
`_prepare` (`cad_guided_plane_initialization_v12.py:287`). V8 builds the owned
scan proposal bank and complete CAD-face plane declarations, then calls search
V6 (`cad_guided_plane_initialization_v8.py:140–189`). V14 adds the pair schedule
and context V4 (`:211–385`), runs the complete selected fixed-pose prefilter
(`:491–692`), and refuses before whole-part refinement when there are no start
groups (`:951–961`). Only survivors enter `_refine` (`:993–1005`).

Frozen V8 source configuration is explicit: at most 128 bank planes, two
triplets per anchor, 12 sampled CAD pairs per triplet and four unsigned
assignments per scan triplet; eight signs per unsigned triplet. The joint
prefilter caps are 8,192 hypothesis records and 1,024 unsigned assignments.
Finite prefilter distance is 2 mm and per-support minimum coverage is 0.75
(`scripts/run_gold11_mesh_development_v8.py:203–207,236–248`). These are not
manufacturing tolerances or calibrated pose-error bounds.

## Findings

### F1 — P2: one least-squares normal witness is not per-cone feasibility

`cad_guided_plane_search_v6.py:527–565` computes an unweighted proper Wahba
rotation, minimizing summed squared chord errors, then rejects the entire signed
assignment if **any** individual angle exceeds that row's allowance. This is
correctly named `NOMINAL_WAHBA_WITNESS_OUTSIDE_DECLARED_GATE`; it is not a proof
that every rotation violates the rowwise constraints. There is no constrained
or alternative-witness attempt between that rejection and loss of the start.

Analytic counterexample, before floating-point verification:

- Scan normals: `(ex, ey, ez)`; CAD normals:
  `(Rz(0.05) ex, ey, ez)`; consider the all-positive target signs.
- Scan normal-cone half angles `(0.06, 0, 0)` radians, CAD cones all zero, and
  unchanged extra alignment allowance 0.02. Row allowances are `(0.08,0.02,0.02)`.
- Wahba's optimum is `Rz(0.025)`: the planar objective is
  `cos(beta - 0.05) + cos(beta)`, maximized at `beta=0.025`, with the third
  normal unchanged. Its residuals are `(0.025,0.025,0)`, so row 1 fails.
- `R=I` has residuals `(0.05,0,0)` and satisfies all three allowances, even
  the original 0.06 cone on the changed first normal.
- The first included-angle gap 0.05 is within the 0.06 cone sum; other pair
  gaps are zero. Both nominal systems are well conditioned. Planes through
  the origin also avoid any intersection residual complication.

Thus a fallback seeking a feasible rowwise witness can recover a legitimately
admissible **proposal** without changing a threshold. Normal cones remain
stipulated, not calibrated confidence. A returned witness still needs finite
support, whole-part registration and measurement checks.

The analogous pair issue is stronger and simpler:
`cad_guided_two_plane_context_v1.py:375–411`, called unchanged by context V4,
constructs `R = frame(target) @ frame(source).T`. It matches the first normal
exactly and places the pair-angle mismatch on the second normal. Source pair
IDs are canonicalized, so renumbering otherwise identical supports can change
which normal absorbs the error. Both target orders and all sign choices do
not repair this within the same intended correspondence.

Example: scan normals `(ex,ey)`, CAD normals `(ex,Rz(-0.03)ey)`, scan cones
`(0.04,0)`, CAD cones zero. The included-angle gap 0.03 passes its 0.04 cone
allowance, but the constructed `R=I` leaves 0.03 on row 1, exceeding 0.02.
`Rz(-0.03)` leaves `(0.03,0)` and passes `(0.06,0.02)`.

For a fixed signed pair with central angle difference
`delta = theta_source - theta_target`, an extra rotation `beta` around the
target cross-product axis yields signed in-plane residuals `beta` and
`beta+delta`. A candidate interval is

`[-allowance0,+allowance0] intersect [-delta-allowance1,-delta+allowance1]`.

Select a deterministic interior witness when nonempty, reconstruct the proper
rotation, and recheck the actual `atan2` residuals. This one-dimensional witness
construction is a small proposal mechanism, not a general SO(3) feasibility
certificate. Keep antiparallel/sign and numeric boundary cases explicit.

### F2 — P2: eight extent votes do not search finite axial alignment

`cad_guided_two_plane_context_v4.py:337–359` computes, for whole buffers and
each assigned support, axial intervals `A=[a0,a1]` and `B=[b0,b1]`, then votes
at `b0-a0` and `b1-a1`. It adds the whole interval midpoint and the midpoint of
the two assigned intervals' intersection. This is at most eight scalar votes,
not an axial search. Holes, necks, nonconvex trims and partial support can make
the only useful alignment occur away from all eight.

Important correction to possible causal wording: **an empty heuristic interval
does not remove the six endpoint votes.** Only the two midpoint votes can be
`None` (`:350–358`); only those are labeled `ABSENT_EMPTY_HEURISTIC_INTERVAL`
(`:418–446`). V14 preserves them as
`AXIAL_SLOT_ABSENT_NOT_GEOMETRIC_EXCLUSION` (`:515–516`). Therefore that reason
must not be counted as proof the complete signed pair was geometrically
excluded. Available families retain at least six proposed slot records, before
exact-pose deduplication.

With exact rotation and exact complete containment, the scalar condition
`A+lambda subset B` would give `[b0-a0,b1-a1]`; that explains the endpoints.
However, the actual downstream model allows only 75% support within a 2 mm
distance gate, and neither exact containment nor full scan/CAD coverage is
assumed. These scalar intervals are consequently heuristic proposal devices,
not necessary bounds for the active support model. An overlong scan interval
can still have sufficient finite support; a short scan interval admits many
translations, not merely its endpoints and midpoint.

A bounded adversarial geometry can demonstrate the gap without any physical
case: two orthogonal planar CAD faces each have a thin long spine spanning
axial coordinate 0–100 mm, with a small inward planar island around 30–31 mm
joined by a narrow neck. Only that island is near the observed transverse
coordinates; the long spine is more than 2 mm away transversely. Scan both
islands at axial coordinates 40–41 mm, with exact central normals. The true
axial translation is -10 mm. Whole and per-face extent votes collapse to
`-40`, `59` and `9.5` mm; none reaches the island. Finite support at the true
pose is exact. Construct polygons/meshes independently and verify every vote;
this paragraph is an analytic fixture design, not a test result.

### F3 — P2: necessary support checks at one seed become a recovery bottleneck

V14 queries every raw row of every assigned support at the **unrefined** pose,
against its one declared CAD face (`:534–569,587–634`). Each support must meet
0.75 coverage, and the hypothesis survives only if all two or three pass
(`:653–660`). There is no bounded refinement or reassignment before this gate.

The arithmetic is not suspect: a seed with a wrong assignment should fail.
The architectural restriction is treating exact seed quality as a prerequisite
for even trying a refinement designed to improve seeds. Triplet translation
is `pc - R ps` (`cad_guided_plane_search_v6.py:563–568`). A small rotation error
acts about the estimated triple-plane intersection, so displacement on a
distant finite support scales approximately as `distance_to_intersection *
angular_error`. A 0.01-radian seed error at a 500 mm lever arm is approximately
5 mm, already above the 2 mm prefilter even though the normal-angle allowances
can admit that seed. The 1e-8 mm intersection backward-residual check certifies
only the solved nominal equations, not forward pose accuracy under noisy plane
normals/offsets (`cad_guided_plane_search_v6.py:335–363`).

Likewise a single failed support cannot be silently discarded as though the
three-plane pose remained equally constrained: two planes leave an axial
degree of freedom. If implementing an outlier/consensus lane, create a new
explicit lower-rank hypothesis with its nullspace and assignment provenance;
do not turn `all()` into `any()` and call it accepted registration.

The one-to-one face assignment also assumes each selected scan patch belongs
mostly to one complete CAD source face. CAD plane fitting groups by source
face identity (`cad_guided_mesh_plane_proposals_v4.py:366–396`), while scan
supports are CORE/CLOSURE/SEEDED_REGION proposals, not authenticated face IDs
(`cad_guided_plane_initialization_v8.py:144–177`). Scan oversegmentation alone
is not necessarily harmful; a patch spanning coplanar CAD face subdivisions or
incorrect closure can be. Actual relevance is unverified without retained
support-level evidence; do not merge physical CAD faces merely to pass this gate.

### F4 — P2: proposal breadth is ordinal, not geometric coverage

Several finite truncations can omit the right correspondence despite thousands
of recorded hypotheses:

- The 128-plane bank round-robins kinds, then favors normal novelty and centroid
  spread, with plane IDs as a tie-breaker (`cad_guided_plane_search_v6.py:296–319`).
- Each anchor takes only its two highest-determinant partner pairs; this
  selection precedes triplet qualification and has no fallback to the next
  qualified pair if those two fail (`:368–405`).
- Only 12 CAD pair-window ranks and four compatible triple ranks are sampled
  per triplet (`:447–524`). Fair cycling is shared across repeated **exact
  layouts**, not a completeness guarantee for any particular scan triplet.
- Pair slots are reserved only after all retained triplet records, including
  rejected signs. With 8,192 triplet records, pair capacity is zero
  (`cad_guided_two_plane_schedule_v2.py:140–154`). This is deliberate custody
  accounting, but no lane-diversity guarantee.
- V2's residual pass only revisits windows opened by its V1 breadth pass
  (`:260–278`). It prioritizes large normal-independence margins and can spend
  all remaining capacity on the first such window (`:227–246`). Thus it can
  improve CAD matches for an already prioritized scan pair while never
  examining a different scan pair needed for the correct pose.

Both CAD target orders are intentionally bundled (`schedule_v1.py:328–352`),
and every target sign is retained; no simple missing-sign bug was found.
Determinism does not imply permutation or tessellation invariance. Repeated
orthogonal faces naturally create large identical angle-index windows, making
ordinal schedules especially weak geometric discriminators. This is not
evidence of a learned model being overtrained or undertrained on a family.

The published `pair_lane_starved_by_joint_quota` flag detects **total**
starvation only: it requires no CAD windows to have been opened
(`schedule_v2.py:203`). A false flag does not rule out later-pair starvation.
The per-pair ledger remains necessary to distinguish those conditions.

## Sound checks to preserve

For an enclosing face box B, the code computes the downward-rounded infinity-
norm box distance L. Since `L <= distance(point,B) <= distance(point,face)`,
`L > 2 mm` safely proves that represented point is outside the finite gate.
The whole-support possible coverage is rounded upward before comparison
(`cad_guided_face_box_screen_v1.py:1–10,77–89,95–116`). V14 builds each box from
the complete owned finite-face index (`:400–426`) and reuses exactly the same
transformed rows for box and finite queries (`:591–600`). No false-pruning
mechanism was found in this narrow review.

Finite support uses `distance_upper <= gate`, with lower-rounded coverage;
ambiguity is counted separately, not the reason for ordinary distance-support
rejection (`global_registration_v4.py:262–271`, initializer V14 `:629–657`).
Do not blame normal-ambiguity gating for the reported box exclusions.

Transform conventions are internally consistent in the inspected path:
column action `y=R*x+t` is implemented for point rows as `x @ R.T + t`;
normals use `R*n`; pair local/world translations are reconciled through the
explicit origins and V14 binds those origins to zero. Distances/anchors/lambda
are millimetres, normal angles are radians. No degrees/mm conversion bug was
found. These statements concern represented arithmetic, not datum calibration.

## Ranked next work and bounded adversarial plan

1. **Smallest falsifiable change: alternate cone-feasible witnesses.** Keep the
   original witness and its rejection record; try the analytic pair witness
   or a bounded triplet rowwise-feasibility candidate without admitting any
   pose solely on normals. Prove recovery of the two examples above, sign/order
   equivalence under the same intended assignments, rejection of deliberately
   incompatible controls, and proper rotations. This directly targets the
   named dominant search rejection, but its physical share is unknown.
2. **Axial search based on owned finite support, not only extrema.** For a fixed
   pair family, build a bounded one-dimensional translation proposal mechanism
   from finite CAD/scan geometry. A useful rejection-only envelope for a scan
   axial coordinate s and target extent [b0,b1] is the allowed interval
   `[b0-s-g,b1-s+g]`. An interval-overlap sweep can bound the maximum possible
   coverage as lambda varies. It must not certify support or ignore holes;
   final candidates still need exact finite-surface queries. Subdivide or use
   independently defined local finite features to avoid missing narrow valid
   trim alignments. Retain unresolved intervals at budget exhaustion, not an
   empty-pose conclusion.
3. **Joint-lane allocation after the mechanism is useful.** Give finite triplet
   and pair proposals predeclared breadth under one unchanged total budget;
   spend residual capacity across scan pairs, not a single highest-margin
   pair. Preserve all attempted/rejected/omitted ledgers. Merely increasing
   thousands of uninformative signs is not a causal fix.
4. **Only if necessary, a separately declared bounded seed-repair lane.** Use
   support-error bounds/pose uncertainty or rank-aware consensus to admit
   refinement proposals that do not already satisfy final fixed-face coverage.
   Never weaken final registration or measurement acceptance. This is more
   invasive than items 1–2 and has a greater wrong-pose/CPU risk.

Proposed source-only discriminator, **not executed or frozen here**: at most
64 invented cases, 8,192 scan rows/case, 256 CAD triangles/case, one worker,
OMP threads 1, 120 s CPU / 180 s wall / 2 GiB process-tree RSS / 2 MiB report.
An independent reviewer should finalize exact fixtures and implementation
before one prospective run. Keep build, proposal, query and total CPU/work
separate, and retain every original/alternative seed and failure.

The matrix should include:

- The two asymmetric-cone examples plus impossible-angle, near-rank-deficient,
  determinant-negative sign, and exact-angle controls; reorder IDs and global
  rigid transforms without using case IDs in inference.
- Clean full rectangular faces, off-center partial scans, the connected
  spine/island trims above, holes and two nearby identical islands with genuine
  ambiguity. Include a single remote scan outlier to expose extrema sensitivity.
- Noise at a predeclared bound, differing support spans/densities and independent
  plane fitting, then independently posed tests so supplied plane normals do
  not become an answer shortcut.
- A distant triple intersection and one incorrect third assignment; verify
  recovery requires a new valid hypothesis and that unresolved axial freedom
  remains explicit.
- Repeated orthogonal CAD faces with the only correct finite match deliberately
  placed outside the old ordinal samples; ID/tessellation permutations and
  partial/total lane-starvation cases. Count matched assignments/poses, not CAD
  features or Gold lines.

Acceptance must test actual finite support and pose error against the invented
transform, not simply the number of proposals. An ambiguous geometry must stay
ambiguous. A passing synthetic discriminator establishes mechanism capability
only; it does not establish physical F1, calibration, or a new benchmark gate.

## Snapshot and limitations

SHA256 of the inspected active sources and released narrative:

```text
9e3ca41edd05bcc0b6fcdeb9ef0447dbacfe58df8c9458da20f9bb12c3d925bf  backend/cad_guided_plane_initialization_v14.py
57a7b44dedaead564d219ceb039f10bcf31ab2cd51b6ea4e2af30a34f00a1d87  backend/cad_guided_plane_initialization_v12.py
3b4220637fa19aba1e0384071f55e091705476046468327a306486078d7f7438  backend/cad_guided_plane_initialization_v8.py
aa2f9067c5e4e65129c0c8c42d39b6d980c3e33fadac6daf881e544648292676  backend/cad_guided_plane_search_v6.py
e73be23dc2284898c00a97a63a0b61d6403aab194221d743a0476dbeab6c6a25  backend/cad_guided_two_plane_context_v4.py
9a352c6bdb6029e8feeb4671e8ac8973e6dc7568dd250a9b70ab80f23353e89a  backend/cad_guided_two_plane_context_v1.py
1fcd5bb06922b64693de235cbf1e6e7df6fdd9c43795c6bced27af6d2e4b866f  backend/cad_guided_two_plane_schedule_v1.py
07b24b8e53b08a408f56ab94dc442e7db7f4e3ef6a5cd4729629860cace99977  backend/cad_guided_two_plane_schedule_v2.py
c8f66cc25f5c4cb6a6ba3c1e1beeb205540813796bb2ab30d72eb596cdf3abb7  backend/cad_guided_face_box_screen_v1.py
5b24f724efdd7dff77317d01b586577ac290ef3cee68ff3ccad7eef0cd998273  backend/cad_guided_global_registration_v4.py
0e9aa2f64f3fc14e0da137687e44c8cd0dc0712f9e8bd783726df4187b1360c5  scripts/run_gold11_mesh_development_v8.py
d16e711c887e325a289f63a4cb2c2ca3f4bfc24ca3e112ac7e8a702441cb1747  docs/research/2026-09-13_gold11_mesh_development_v8_result.md
```

No independent numerical execution was performed. The source findings and
analytic constructions are not a reconstruction of physical failing supports,
their true pose, or their measurement error. The released narrative was still
pending independent POST when read and notes an unobserved outer process exit;
this review does not repair or assess that custody limitation. Preserve V8 and
all earlier terminal attempts. No claim that historical F1 0.4580152672 was
beaten, and no new physical execution is authorized by this document.
