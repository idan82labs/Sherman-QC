# V4 mesh-plane capacity construction — prospective protocol

Date: 2026-09-09. This is a separately supervised invented-array capacity test,
not a rerun of physical data or a noise/generalization result. Nothing has been
executed under this protocol when it is authored. The
[original mesh-plane protocol](2026-09-09_mesh_plane_proposals_v4_protocol.md)
already declared the exact 800,000-row / 1,600,000-triangle prospective size.
Neither count nor its 120-second wall / 90-second CPU / 4-GiB limit is reduced
or raised here. The unchanged extractor and its default policy are used.

## Why these exact counts describe a natural valid mesh

The construction is the boundary of a rectangular annular prism: outer square
side `n=500`, a centered square through-hole of side `h=300`, and height `k=150`,
all on a unit grid. It has ten planar faces: top and bottom annuli, four outer
walls and four inner walls. It is an ordinary embedded closed genus-one surface,
not a flat periodic wrap, duplicate-face padding, unused-row padding or a
degenerate abstract triangulation. Winding is outward, including into the hole
for the inner walls. All wall/top/bottom boundaries share the same raw vertex
indices; no coordinate welding or post-construction repair is performed.

One horizontal annulus contains

```
A = (n+1)^2 - (h-1)^2 = 161,600 vertices
P = 4(n+h) = 3,200 boundary vertices
V = 2A + P(k-1) = 800,000 distinct used raw vertices
```

The top and bottom together contain `4(n²-h²)=640,000` triangles. The vertical
walls contain `2Pk=960,000` triangles, giving exactly `F=1,600,000=2V`.
Every unit square is split into two nondegenerate triangles of area 1/2.
Every undirected raw edge has exactly two incident triangles, with opposite
directed winding. Thus `E=3F/2=2,400,000` and `V-E+F=0`, as expected for the
closed genus-one surface. The two small ordinary controls independently check
the same construction's edge incidence, winding, unique used vertices and its
actual extractor-to-oracle connection before the large opt-in case is eligible.

## Frozen intended output and full denominators

The extractor receives only the constructed XYZ and triangle indices, with its
unchanged default policy and work caps. Expected face definitions are generated
only in the test evaluator after extraction returns; they are not an inference
argument or a source of seed choices. Plane/patch labels are enumeration-local.

All ten faces must be proposed. Each top/bottom annulus has 320,000 triangles,
161,600 support vertices and area 160,000 mm². Each outer wall has 150,000
triangles, 75,651 vertices and area 75,000 mm². Each inner wall has 90,000
triangles, 45,451 vertices and area 45,000 mm². Summed triangle area is
800,000 mm². This particular construction is nonoverlapping; the extractor's
general area diagnostic still explicitly counts triangle multiplicity.

The top/bottom area centroid is `(250,250,z)` and each in-plane variance is
`(500²+300²)/12 = 28,333 1/3 mm²`, distinguishing the retained hole from a
filled outer square. Wall centroids and covariances use the uniform rectangle
formulas. Independent moment comparison tolerances are 5e-8 mm for centroid
and 2e-4 mm² for covariance, accounting for large floating-point sums; these
are not uncertainty bounds and do not relax the extractor's 1e-8-mm all-support
maximum-residual gate. Axis projectors, offsets, exact unit-cell areas and
complete original support IDs are also checked.

Every one of the 1,600,000 triangle rows must appear in exactly one patch and
must lie on that patch's finite expected face. Every one of the 800,000 raw
rows must appear in its exact expected face-support union. Geometric seams
are shared context, not greedily owned or deduplicated across patches:

| Raw-row patch multiplicity | Exact rows |
| --- | ---: |
| 0 | 0 |
| 1 | 792,408 |
| 2 | 7,576 |
| 3 | 16 |

The resulting support-membership and full-vertex-test counts are both 807,608.
There are `8(n+h+k)=7,600` noncoplanar crease edges, so the exact shared-edge
counter is 2,400,000 and compatible-edge counter is 2,392,400. The complete
component denominator is ten evaluated, ten proposed and zero rejected.
The test does not pass on `AVAILABLE` alone, on a partial prefix, or by dropping
unmatched components. No oracle field is added to the extractor input.

## Resource admission and custody

New test source: `tests/test_cad_guided_mesh_plane_capacity_v4.py`.
There are two ordinary small controls and one opt-in capacity case. The capacity
case requires `SHERMAN_RUN_MESH_PLANE_CAPACITY_V4=1` from its separately named
supervisor route; it is skipped in the ordinary source suite. Root must review
the complete test/source/protocol and supervise execution serially. An inherited
ambient flag must not activate this case in a different suite.

The fixed capacity case uses 120 seconds wall, 90 seconds worker CPU, 4 GiB
sampled process-group RSS and 1 MiB total output. One worker and one heavy job
are allowed; BLAS/OpenMP thread limits remain one. The harness binds itself,
these tests, this protocol, the original mesh-plane protocol and every imported
project source before execution and verifies unchanged hashes afterward. It
retains a new exclusive STARTED/output/COMPLETED receipt. A timeout, memory stop,
assertion failure or source drift remains a terminal failed attempt; no silent
retry, count reduction, fixture simplification or cap increase is permitted.

Admission stays below the existing independent extractor caps: 800,000 of
1,000,000 scan rows, 1,600,000 of 2,000,000 triangles, 4,800,000 of 6,000,000
edge records, ten of 65,536 components and ten of 4,096 output patches. The
conservative preflight membership/full-test bound is 4,800,000, below the
respective 6,000,000 / 12,000,000 caps; actual memberships are counted rather
than substituted for that preflight budget.

The constructor stores surface grids, not a 501×501×151 volume. Its permanent
XYZ and triangle arrays occupy 19,200,000 and 38,400,000 bytes. Construction
uses linear surface-index arrays; extractor storage is dominated by linear
edge/membership arrays and a bounded sparse adjacency graph, with sorting
work rather than repeated per-seed traversals. Verification scans every raw
row and every patch triangle in bounded 65,536-triangle chunks. These source
complexity observations are not a measured peak or a worst-case memory proof.
The external supervisor, not this estimate, enforces the full process group.

The compact JSON receipt is below 16 KiB and includes exact geometry/counter
denominators, separate extraction time, construction-through-full-verification
time and worker-lifetime RSS read after all lineage assertions. Parent-observed
complete-run process-group memory/time remains the authoritative resource
receipt. No raw arrays, million-row listings or generated mesh files are emitted.

## Advancement and limits

Success requires every selected assertion, all original caps and unchanged
source bindings. It establishes capacity only for this one exact, axis-aligned,
unit-grid, closed manifold construction. It is not worst-case graph behavior,
small-feature generalization, noisy-plane extraction, arbitrary triangulation
robustness, pose recovery, native CAD qualification, physical count/F1 gain or
production/floor readiness. The distinct connected initialization experiment
still needs its own frozen cases, gates, budget and independent review.
