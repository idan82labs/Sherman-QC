# V4 near-roundoff mesh-plane proposals: prospective protocol

This source-only stage connects supplied-plane initialization to actual indexed
scan triangles and import-local CAD face triangle unions. It does not reopen a
terminal V1/V2/V3 experiment. No execution, positive engine result, noisy-scanner
qualification, held-out access or physical-run approval is recorded here.

## Scope and interfaces

New source: `backend/cad_guided_mesh_plane_proposals_v4.py`.

- `extract_coplanar_scan_patches_v4(points, triangle_raw_rows, *, policy, budgets)`
  consumes finite XYZ rows and original indexed scan triangles.
- `fit_cad_face_planes_v4(triangles, *, source_face_indices,
  source_triangle_indices, policy, budgets)` fits each complete CAD face union.

Inputs are exact NumPy arrays, snapshotted into immutable little-endian bytes.
Integer source IDs, shapes, counts and operation caps are checked before indexed
geometry/sorting. Every input position is finite and at most 1e6 mm in magnitude.
Booleans, object/string arrays, array subclasses, invalid indices and repeated
vertex indices within a triangle are refused. A triangle with
`||cross|| <= 1e-12 * longest_edge_squared` is numerically outside this scope.
These are engineering guards, not certified error intervals or noise models.

The result has `patches`, every component disposition, immutable original
triangle-to-component labels, original support unions and digests. Scan supports
contain sorted unique original vertex-row IDs and original triangle rows. CAD
supports retain every triangle's source face/local-triangle identity. Equal XYZ
with different scan row IDs is never welded. Distinct CAD face IDs are not merged.
Holes and disconnected CAD subsets remain the supplied triangle union, not a
filled polygon or assumed material topology. CAD has no shared vertex-ID model
in this interface, so it does not claim topological component discovery.

`AVAILABLE` with zero proposals is allowed; it is not positive geometry evidence.
Every result states that registration has not been accepted and production use
is disallowed. A proposed plane is a fit to triangulated support, never a flange
classification or analytic native-plane assertion. Planar facets on a coarse
cylinder remain legitimate mesh-plane proposals and must not be category-filtered.

## Compression and whole-component validation

The scan builds one shared-edge graph from canonical pairs of original raw
vertex indices. More than two incident triangles on an edge refuses this narrow
batch rather than generating an unbounded incidence clique or repairing input.
Only adjacent triangles satisfying both tests are linked:

1. The sine between unoriented normals is at most `1e-8`.
2. Both opposite vertices lie within `1e-8 mm` of the neighbor's plane, using
   differences anchored at the shared edge—not world-offset subtraction.

Sparse connected components compress this graph once. A fixed whole-component
area-TLS fit follows. Every original support vertex must be within `1e-8 mm` of
that fit. Pairwise compatibility alone cannot qualify a curved chain. A whole
component failing the maximum bound stays rejected: no trim, split, re-growing
or subset salvage follows. Independently valid components remain available.

The CAD arm groups by supplied import-local face ID, fits all its triangles and
applies the same maximum-support bound. It cannot ignore a distant low-area
triangle merely because area-weighted error or vertex RMS is small.

## Independently checkable area mathematics

For a triangle, let `q_i = p_i-a` about one fixed anchor and `s=sum(q_i)`.
Uniform triangle barycentric coordinates have
`E[lambda_i²]=1/6`, `E[lambda_i lambda_j]=1/12` for `i != j`. Therefore

```
E[q]    = s/3
E[q qT] = (sum_i q_i q_iT + s sT)/12
```

Multiply both by triangle area, sum over the complete component, divide by its
total area and subtract the outer product of the mean for the covariance. The
implementation first estimates the centroid using component-local anchors, then
evaluates these moments in a second pass recentered near that centroid. It never
subtracts world-origin second moments to recover small local covariance.

The smallest covariance eigenvector supplies the unoriented TLS normal. Both
`area_centroid_mm` and `surface_covariance_mm2` are exposed for independent
analytic rectangle/retriangulation tests. This is uniform triangle-area TLS,
not equal-weight vertex or centroid TLS. Area is explicitly
`TESSELLATION_AREA_WITH_MULTIPLICITY_NOT_MATERIAL_AREA`: duplicate or overlapping
triangles contribute repeatedly; overlap is not validated or repaired.
Retriangulation invariance assumes a nonoverlapping tessellation of the same
support. Scan RMS is separately the RMS over unique original raw vertices.
CAD RMS counts triangle corners with multiplicity
because the API carries no unique native-vertex identities. Both measures are
explicitly labeled, and neither can replace the maximum-support guard.

Scan support needs 12 unique vertices; CAD needs 3 triangle corners. The ratio
of the second to largest covariance eigenvalue must be at least `1e-8`. Support
extents are full vertex ranges along the two in-plane covariance eigenvectors,
sorted descending together with their reported axes; the smaller range must be
at least `.25 mm`. These are geometric admission limits, not statistical power.
When the in-plane eigenvalue gap is at most `1e-10 * largest_eigenvalue`, the
reported axes are explicitly arbitrary within a repeated eigenspace. Such box
ranges can depend on that choice. Admission in that repeated eigenspace therefore
also requires the coordinate-invariant sufficient width bound
`2*sqrt(lambda_min_inplane) >= min_second_extent_mm`; otherwise it reports
`REPEATED_EIGENSPACE_EXTENT_UNRESOLVED`. Every in-plane directional variance is
at least `lambda_min_inplane`, and variance is at most one quarter of the squared
support range (Popoviciu's inequality). Thus the bound is sufficient for every
direction, not a tight minimum-width estimate. It can conservatively refuse a
patch with adequate actual width. The bound and admission basis remain visible
even on component rejection; displayed box axes remain arbitrary at a tie.
This real-arithmetic argument is not a certified floating-point enclosure.

For a fixed fitted unit-normal plane, its signed residual is affine over a
triangle, so the absolute residual is bounded by the largest absolute vertex
residual. The all-vertex gate therefore bounds the whole retained triangulated
patch in real arithmetic, not just sampled interior points. It is not a bound
on the unknown physical surface, native CAD approximation or scanner error.

Away from threshold and repeated-eigenspace boundaries, moment/projector and
support results should transform covariantly under rigid coordinate changes,
vertex/triangle reordering with identity remapping, winding flips and consistent
unit/policy scaling. Floating-point boundary equality is not certified. Raw
component numbers are enumeration-local, not invariant physical identities.

## Fixed resource and failure contract

Exact-type integer hard caps, which callers may only tighten:

| Item | Maximum |
| --- | ---: |
| Scan vertex rows | 1,000,000 |
| Triangles | 2,000,000 |
| Scan edge records | 6,000,000 |
| Components / CAD face groups | 65,536 |
| Qualified output patches | 4,096 |
| Support memberships | 6,000,000 |
| Full vertex residual tests | 12,000,000 |
| Triangles per working chunk | 65,536 |

Zero capacity caps are legal; a zero working chunk is invalid. The conservative
`3 * triangle_count` membership/full-test bounds and scan edge bound are checked
before allocation. Components are counted before per-component fitting. All
dispositions are computed before an output-count refusal. Global input, numeric,
nonmanifold or capacity failures clear all patches, component records and labels,
reset completion and mark remaining counters as attempted-work evidence only.
Geometric component failures retain the other proposals and all component
dispositions. No top-area selection, greedy support consumption, hidden input
thinning or noisy-fragmentation repair
is permitted. The downstream initializer's independent 12/24-plane and signed
assignment caps remain applicable; this extractor cannot bypass them.

Small pure/source verification uses a separately supervised 60 s wall /45 s CPU /
1 GiB RSS /1 MiB stdout. A distinct opt-in capacity construction at 800,000 raw
vertices and 1,600,000 triangles has prospective 120 s wall /90 s CPU /4 GiB RSS /
1 MiB stdout. The memory expectation is not a measured result. Only one heavy
job runs at once; neither budget may be raised after a terminal failure.

## Eight prospective independent groups

1. Exact flanges, a curved strip and a small orthogonal remote cap under four
   independently fixed poses: require intended planes/cap; never the whole bend
   as one plane. No oracle pose enters extraction.
2. Holes, disconnected coplanar tabs and duplicated XYZ with distinct raw row IDs:
   preserve finite unions and do not bridge by coordinates.
3. Repeated parallel inner/outer supports: preserve distinct components/face IDs,
   including smaller supports; never choose the largest area as the answer.
4. A near-coplanar chain passing local pair gates but failing the full maximum:
   retain its whole rejection and independently valid components, with no trim.
   Fixed example: strips over `x=0..12`, `y in {0,1}`, `z=4e-9*x²`.
5. CAD face with a distant low-area triangle: maximum-residual refusal despite
   low average error. Nonuniform rectangle retriangulations and a rectangle with
   a hole also check centroid, covariance and area against independent integrals.
6. Vertex/triangle permutations, winding, independent frames and unit scaling:
   compare mapped support, normal projectors and moments; disclose eigenvalue ties.
   Small repeated-eigenspace squares must not gain admission by rotation of the
   arbitrary box axes; adequate invariant-width squares must remain admitted.
7. Exact type/cap boundaries, invalid IDs, nonmanifold fans, degenerate triangles,
   deliberate noisy fragmentation and injected late numeric failures: every
   fatal batch result has zero patches; incomplete/noisy results never count as
   complete noisy-plane detection.
8. The separately audited opt-in capacity case above. It establishes only that
   fixed construction's operation/memory behavior, not worst-case noisy geometry.

After independent source/math tests, the unchanged invented connected workflow
can decide whether these proposals actually improve pose and measured output.
Only a separately frozen physical protocol may evaluate consumed development
parts. This exact-mesh slice alone cannot qualify scanner-noise robustness.

## Primary-method context, not a dependency change

[CGAL's mesh region-growing documentation](https://doc.cgal.org/latest/Shape_detection/index.html)
uses shared-edge neighbors, maximum vertex-to-plane distance and normal-angle
tests. Its choices of seed order and iterative region fitting are not inherited
as greedy ownership here. [Polylidar3D](https://pmc.ncbi.nlm.nih.gov/articles/PMC7506964/)
uses triangle-normal accumulation, segmentation and finite polygon extraction;
its smoothing/dominant-normal machinery is a broader noise-handling method, not
evidence for the present near-roundoff compression. No library was installed.
