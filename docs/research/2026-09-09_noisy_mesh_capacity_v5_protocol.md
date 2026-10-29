# V5 noisy-mesh full-size capacity qualification — prospective protocol

Status: **SOURCE DRAFT; NOT EXECUTED.** This document fixes one invented full-size
capacity fixture and two small prerequisites before any invocation. Root owns
independent source review, exact source binding, harness selection and execution.
The first terminal outcome remains evidence, including a refusal, missing
required representative, timeout or memory stop. No fixture, noise, policy,
budget or oracle is tuned to rescue that outcome.

## Scope and source boundary

The subject is the frozen research-only extractor
`backend/cad_guided_noisy_mesh_planes_v5.py`, SHA-256
`9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75`.
The new test is `tests/test_cad_guided_noisy_mesh_capacity_v5.py`.
Neither file imports an application workflow or a native CAD/mesh kernel.
The test uses Python, NumPy, SciPy through the extractor, and pytest.

The constructor is independently adapted into the new test from the source of
`tests/test_cad_guided_mesh_plane_capacity_v4.py`. That V4 file and its protocol
are context only: they are not imported, changed or executed. V4 exact-plane
patch count/area checks are not repurposed as noisy V5 claims. The frozen V5
helper suite and its completed receipt are unchanged.

No original CAD, scan, Gold, held-out, email attachment, mesh file, native loader,
corpus identity, expected bend count or reference pose is accessed or supplied.
The released maximum input counts 863,115 raw rows and 1,701,535 triangles are
used only as integer capacity comparisons; no underlying file is inspected.
There is no dependency installation and no generated mesh/data file.

This is not the pending 108-case study, physical robustness qualification,
registration benchmark, calibrated scanner-noise model or production promotion.
It asks whether the unchanged bounded extractor can transport and analyze one
cohort-sized noisy construction while retaining its actual proposal/refusal
semantics and original-row evidence.

## Three fixed cases, two gates

The test file has exactly these cases:

1. Constructor/oracle control: `(n,h,k)=(6,2,3)`, 160 raw rows and 320 triangles.
   Independently check every row is used, unique truth coordinates, two opposite
   oriented incidences per undirected edge, Euler characteristic zero, ten
   analytic face labels, and nonzero bounded perturbations. No extractor call.
2. Small actual noisy extractor control: `(64,24,32)`, 18,304 raw rows and 36,608
   triangles. One real pure V5 call, with the same all-record oracle and all ten
   required representatives used in the large case. This is not mocked geometry.
3. The sole full-size case: `(500,250,175)`, 900,000 raw rows and 1,800,000
   triangles. Exactly one real pure V5 call, followed by the full oracle.

The first source/prerequisite gate is limited to 60 seconds wall, 45 seconds
worker CPU, 1 GiB sampled process-group RSS, and 1 MiB output. The opt-in flag is
absent, so the large case is skipped; two passes and that declared skip are the
expected successful outcome. The small positive requirement is not converted
to an acceptable abstention if it fails.

Only after independent review and a green prerequisite gate may root select a
separate named capacity route setting `SHERMAN_RUN_NOISY_MESH_CAPACITY_V5=1`.
Its fixed external envelope is 120 seconds wall, 90 seconds **worker CPU**,
4 GiB sampled process-group RSS, and 1 MiB output. The envelope includes pytest,
construction, extraction, oracle verification and output, not merely the call's
reported extraction time. It does not claim OS-level aggregate process CPU or a
certified instantaneous memory peak. The test additionally reports the worker's
lifetime `ru_maxrss`, which is not a substitute for the parent group's monitor.
Root must bind the exact harness/source/test/protocol before this separate run.
The route executes the same two small controls and the one large case; it must
not add another heavy case or a retry. Only one heavy job runs, serially.

## Invented annular-prism construction and exact denominators

The truth surface is the boundary of an axis-aligned square annular prism on a
1 mm integer lattice. The outer square is `[0,n] x [0,n]`; the square hole is
`[m,m+h] x [m,m+h]`, where `m=(n-h)/2`; the extrusion is `0 <= z <= k`.
Bottom and top annuli, four outer walls and four hole walls share the exact same
raw row IDs at all seams. Wall loops close around the real prism, not across a
fictitious periodic planar boundary. The hole is not filled.

Every face uses ordinary two-triangle unit cells. Bottom/top orientation points
outwards; the four inner walls reverse orientation into the hole. Raw rows and
triangles are allocated once and every allocated row participates in triangles.
There are no padding points, repeated disconnected copies, decimated triangles,
coordinate welding, manufactured face roots or substituted output geometry.

Let `A=(n+1)^2-(h-1)^2` be the annulus vertex count and `P=4(n+h)` the combined
boundary length. The original-row denominator is

`V = 2A + (k-1)P = 2(n+h)(n-h+2k)`.

The triangle denominator is

`F = 4(n^2-h^2) + 2kP = 2V`.

The closed connected orientable genus-one surface has `E=3F/2`, so
`V-E+F=0`. The small constructor control enumerates the actual edges and winding
independently; the full case checks all raw/triangle usage and the extractor's
shared-edge denominator against these prospective counts.

For the single large case the physical dimensions are fixed at outer
500 x 500 mm, inner 250 x 250 mm, height 175 mm, with a 125 mm annular band:

| Quantity | Exact count |
| --- | ---: |
| Bottom/top original raw rows | 378,000 |
| Intermediate wall raw rows | 522,000 |
| Total used raw rows | 900,000 |
| Triangles per bottom/top face | 375,000 |
| Triangles per outer wall | 175,000 |
| Triangles per inner wall | 87,500 |
| Total original triangles | 1,800,000 |
| Unique shared undirected raw edges | 2,700,000 |
| Extractor edge records, three per triangle | 5,400,000 |

Thus this fixture exceeds the released cohort maximum by 36,885 raw rows and
98,465 triangles, without increasing the extractor's 1,000,000-row,
2,000,000-triangle or 6,000,000-edge-record hard caps. The 2.7 million unique
edges and 5.4 million extractor edge records are different denominators.

## Noise is an exact stipulation, not calibration

For original row ID `i` and axes x/y/z, respectively, use tuples
`(a,b,q) = (5,1,32), (7,3,34), (11,5,36)` and perturb by

`delta(i) = [2*((a*i+b) mod q) - (q-1)] / 8192`.

Every numerator is odd and nonzero. Per-axis absolute numerator maxima are
31, 33 and 35. Consequently every component is strictly inside `+/-0.005 mm`,
and each row's Euclidean displacement is at most `sqrt(3275)/8192 < 0.007 mm`.
The integer truth coordinates and this dyadic perturbation are exactly
representable in binary64 at all declared coordinate magnitudes. The small
control checks integer numerator recovery and squared magnitude directly.

The extractor receives the observed coordinates, the original triangle row
indices, and the single stipulated error-ball radius `epsilon=0.05 mm`. No label
or truth normal is passed. The scalar is not fitted from residuals, inferred from
a scanner, promoted to calibrated uncertainty, or widened for any assertion.
All raw rows receive nonzero noise in all three coordinates. The full oracle
also verifies every analytic face experiences normal noise greater than 0.003
mm somewhere, so the positive is not secretly ten exact coordinate planes.

The perturbation preserves index incidence and shared seam identity. With unit
grid cells and displacement less than 0.007 mm, it is far smaller than a cell
edge; the unchanged extractor must still pass its own finite nondegenerate
triangle checks. No failure of those checks is bypassed.

## Prospective positive geometry requirements

The default 128/1024 triangle tilings, one-ring halo, 10-degree maximum normal
cone, incomplete 16-core closure schedule and every default work/metadata budget
remain unchanged. No capacity-specific policy object is supplied.

Dimensions were chosen before execution to leave wide two-dimensional interior
regions on each of the ten faces, rather than to make their total supports
count as ten extracted planes. For scale intuition, uniform scalar epsilon
gives ideal cone numerator bound `(2 epsilon)^2 = 0.01 mm^2`. An interior core
whose truth in-plane covariance has smaller eigenvalue at least 1 mm^2 is well
inside a 10-degree gate: the observed perturbation RMS is below 0.007 mm, so the
corresponding centered in-plane standard deviation remains at least 0.993 mm,
while the normal noise variance is below 0.000025 mm^2. The ideal angle is about
0.101 radians before numerical guards, below pi/18. This motivates the physical
dimensions; it is not a proof that this particular BFS tiler produces a chosen
support, nor a substitute for its reported engineering numerical checks.

Evaluator-only labels identify the ten analytic finite faces. A retained record
is **pure-face** only when every original support triangle has the same label;
all its raw truth rows must lie on that plane. The following are required:

- The full public result is `COMPLETE`, with unknown physical plane/bend counts,
  no accepted registration, no verified flatness and no production admission.
- At least one qualified **CORE**, with at least 50 distinct original raw rows,
  exists on each of the ten analytic faces. The first qualifying core by
  original core ID is the deterministic reporting representative. Missing any
  required representative is a failure, not an abstention success.
- For every qualified pure-face core, not just its representative, the true
  unoriented analytic normal lies inside the core's reported conditional cone.
  The same check applies to any qualified pure-face closure.
- For every qualified core or closure, including mixed-face supports, every
  original support row satisfies its fixed reported plane witness within
  **0.05 mm without added slack**. The reported cone does not exceed 10 degrees
  and the record explicitly says epsilon was not increased.
- Mixed-face records, rejected cores, all unique closures and all rejected
  closure reasons are retained in the disposition accounting. A whole-face or
  mixed closure need not qualify. An unscheduled or failed closure does not
  remove an independently qualified core.

The ten evaluator support classes and ten representatives are not an engine
claim of ten observed physical planes, ten parts, ten bends or ten independent
features. Shared seam rows may occur in more than one legitimate support.

## Original-row transport, work accounting and output

The oracle checks input point/triangle digests; every original raw row used;
every truth triangle on its analytic finite face; and a complete once-only base
partition of all 1.8 million triangle rows at both declared scales. It does not
infer finite support from an infinite plane or fill the annular hole.

Every core/closure's sorted immutable raw and triangle row arrays and digests
are checked. Each raw support is independently reconstructed from its original
triangle indices, chunked for large closures. Core bases remain in their halos;
closure aliases point back to their source cores. Proposal IDs equal exactly
the set of qualified records. All proposal, core, closure, qualified/rejected,
scheduled/unscheduled and retained-array counters are reconciled. Every reported
work counter must remain within its unchanged bound. Incomplete search is
reported honestly; the schedule is not expanded to reach a desired answer.

Oracle arrays are linear in original rows/triangles and use chunks of at most
65,536 for large coordinate queries. No dense all-pairs point adjacency is
constructed. There is no full-size independent edge sort added by the oracle;
the small constructor checks that combinatorics, while the large source counts
and original-row checks must agree.

The large test emits one JSON line prefixed
`NOISY_MESH_CAPACITY_V5_RECEIPT=`, bounded below 32 KiB, with exact fixture
denominators, limits, extractor status/reasons, statistics, displacement/witness
and angle maxima, all ten deterministic representative summaries, counts and
scope flags. It prints no point/triangle/lineage arrays and writes no data file.
An oracle exception after extraction emits a bounded `FAILED` receipt and is
re-raised. Earlier failure or an external resource stop remains a failing parent
receipt; absence of the compact success line is never inferred as success.

Any first negative outcome remains locked. A later altered fixture or successor
would require a newly named prospective protocol and new source review; it
cannot be described as this same capacity case passing. A positive outcome
supports only this finite noisy capacity contract. Actual scan noise, coordinate
frame robustness, physical floor testing, local native admissibility and V5
registration integration remain separate, unproven gates.
