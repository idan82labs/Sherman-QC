1. RULING
Authorize V31_DISTRIBUTED_HARMONIC_FOLD_FLUX_V1 — DHFF_V1

Implement exactly one research-only, source-only, rescue-only V16 sidecar whose primitive is a Dirichlet electrical-current field across the observed material surface, followed by a current-weighted single-turn normal-profile certificate.

Authorize:

one frozen source-free implementation and execution;

only after every source-free gate passes, one label-closed prospective run on eight newly arrived scans;

only after that prospective gate passes, the narrowly defined gold opening in §8.

V16 incumbents, candidate geometry, scores, runtime/main, and production remain closed and byte-identical. No V31b, threshold alteration, fixture replacement, alternative graph weighting, or rerun rescue is authorized. Any first post-invocation failure retires V31.

This complies with the packet’s demand for one new causal primitive rather than a repair or reinterpretation of V30. 

Pasted text

Terminal tokens

Pre-invocation fixture or harness defect: V31_DHFF_PREINVOCATION_RETIRE

First source-free mechanism failure: V31_DHFF_SOURCE_FREE_MECHANISM_RETIRE

Prospective or gold-gate failure: V31_DHFF_PROSPECTIVE_RETIRE

No new sensing capability is required for this experiment.

2. RESEARCH BASIS
Published machinery

A voltage fixed at two terminal sets on a weighted graph produces a harmonic potential on the interior; the induced current is determined by the whole resistor network, and parallel conducting routes contribute in parallel. Current-flow formulations therefore differ fundamentally from shortest- or minimum-cost-path selection: current can traverse all available routes, with lower-resistance routes receiving more flow rather than becoming the only selected paths. 
arXiv
+1

Graph Laplacians are established discretizations for geometry sampled from a manifold, including point-set constructions motivated by the correspondence between graph and Laplace–Beltrami operators. Density-normalized diffusion constructions address nonuniform sampling, while discrete differential-geometry methods provide standard normal and differential estimates on triangle meshes. 
NeurIPS Papers
+2
arXiv
+2

Normal-space or Gauss-map structure has also been used in published point-cloud feature detection, including distinguishing locally smooth samples from sharp-feature neighborhoods. 
Hal Science

Sherman-specific hypothesis

The following is not an established theorem from those papers:

At repeated, nonoverlapping axial sections of a valid V16 candidate, a scan-visible sheet fold will create a conductive observed-surface strip between two broad material-tail reservoirs, and the nodes carrying that current will exhibit one continuous, monotone normal rotation from one tail normal to the other.

The source-free execution must determine whether that hypothesis survives realistic sampling, fragmentation, opposite skins, seams, rims, and nearby surfaces.

3. WHY THIS IS NOT V30 TUNING

V30 selected up to eight minimum-cost internally node-disjoint paths at each station and then required those selected paths to possess a frozen axial midpoint spread. On the valid 90-degree fold, it found 72 physically appropriate paths, but every station failed because the optimizer repeatedly chose paths in a narrow axial band. 

Pasted text

DHFF has none of the following:

no shortest or minimum-cost paths;

no path enumeration;

no node-disjointness;

no path midpoint;

no 4s spread threshold;

no diversity term;

no path seed;

no path serialization.

Instead, each station is an isolated axial slab. A single convex Dirichlet solve places a scalar potential on every eligible node and edge simultaneously, and the resulting current is accumulated over all parallel routes. Axial distribution is established only by independent evidence in eleven disjoint stations.

Thus V30’s “all selected paths collapse to the cheapest corridor” failure cannot mechanically recur. DHFF can still fail because the surface graph is wrong or too fragmented, but that would be evidence against this new observation model, not a replay of V30’s path-selection failure.

4. CAUSAL FEATURE

A candidate is certified only when it has all four properties:

Required scan observation	Principal negatives rejected
Two independently broad, planar material-tail reservoirs extending 8s beyond the candidate transition	silhouette, free edge, scan boundary, opening, rim, thickness return
Tangent-compatible observed-surface conductance between the reservoirs	unsupported plane intersection, disconnected seam or miter, nearby panel
Distributed current support at at least 8 of 11 disjoint axial stations	isolated bridge, short welded patch, local tessellation accident
One current-carrying normal rotation, covering intermediate orientations without reversal or a second route	flat false crease, sharp disconnected miter, nonmonotone bead, corner or cycle

Additional controls:

A direct graph shortcut between two planes has no intermediate normal support and fails ANGULAR_COVERAGE.

A thickness face cannot populate all four required tail-distance bins.

A closed cycle produces current through a second normal-space route and fails OFF_ARC_CURRENT.

Duplicate-axis aliases may all certify, but exactly one unchanged ordinary NMS is responsible for instance reconciliation.

A certified rescue overlapping any V16 incumbent loses because legacy records always precede rescues.

This directly encodes the packet’s scan-observable target: two material tails, a supported transition, and a finite repeated axis, without asking how the part was manufactured. 

Pasted text

5. EXACT EXECUTABLE SPECIFICATION
5.1 Inputs

DHFF consumes only:

ScanSurfaceSample
    position_xyz          float64[3]
    source_normal_xyz     float32[3]
    parent_fragment_id    uint32
    stable_source_key     uint64
    mesh_adjacencies      optional bounded indices


V16RejectedCandidate
    candidate_id          uint64
    frozen_v16_order      uint32
    center_xyz            float64[3]
    unit_axis_xyz         float64[3]
    segment_length_L      float64
    local_scale_s         float64
    signed_turn_alpha     float64
    parent_A_id/B_id      uint32
    parent planes         (unit normal, offset, centroid)
    unchanged render segment and ordinary-NMS fields

For meshes, one surface sample is generated transiently per triangle centroid using the triangle’s unaveraged face normal. For point inputs, DHFF uses the already-emitted frozen V16 point normal and parent membership. Samples without a valid existing normal or parent membership are ignored; DHFF does not introduce a new segmentation or candidate generator.

No label, count, identity, filename, family, CAD, acquisition role, or historical result enters this schema.

5.2 Candidate eligibility

A rejected candidate is evaluated only when all predicates hold:

L≥24s
40
∘
≤α≤140
∘
∣u⋅n
A
	​

∣≤sin12
∘
,∣u⋅n
B
	​

∣≤sin12
∘

and:

parent A and B are distinct;

both parent planes and centroids exist;

the render segment is already V16-render-admissible;

the candidate was rejected by the frozen V16 selector before final NMS.

At most 128 eligible candidates are permitted per scan, in frozen V16 order. More than 128 causes CANDIDATE_WORK_CAP; no partial V31 output is emitted.

All dimensionless local coordinates are rounded to signed Q18 fixed point after division by s. Predicate statistics are rounded to Q24 before comparison.

5.3 Axial stations

Eleven station centers are fixed at:

a
j
	​

=c+(−0.40+0.08j)Lu,j=0,…,10.

Each station uses:

axial half-width: 0.75s;

maximum distance from candidate line: 24s.

Because L≥24s, adjacent station slabs are nonoverlapping.

A candidate-local frame is formed from u and the projected parent-normal bisector. Parent swap, axis reversal, and simultaneous normal reversal are canonicalized by reversing station masks where necessary and retaining the lexicographically smaller Q24 summary.

5.4 Deterministic local sampling

Within each station:

voxel side: 0.5s;

retain one sample per voxel;

winner: minimum squared distance to voxel center;

exact tie: lowest SHA-256 of quantized position, normal, and parent tag;

maximum nodes: 2,048;

maximum edges: 24,576.

A cap breach gives STATION_NODE_CAP or STATION_EDGE_CAP.

5.5 Broad tail reservoirs

For parent P∈{A,B}:

Project the station-axis point onto the frozen parent plane, producing a
P,j
	​

.

Set q
P
	​

=normalize(u×n
P
	​

).

Choose the sign of q
P
	​

 toward the frozen parent centroid.

Define tail distance:

x
P
	​

(p)=
s
(p−a
P,j
	​

)⋅q
P
	​

	​

.

A node is a tail-P seed only when:

its parent tag equals P;

5≤x
P
	​

≤13;

parent-plane residual is at most 0.75s;

normal deviation from the parent normal is at most 10
∘
.

Each of these four bins must contain at least two nodes:

[5,7), [7,9), [9,11), [11,13].

Total seeds per tail must be at least 12.

Failures are TAIL_A_BREADTH or TAIL_B_BREADTH. This is the fixed thickness-return discriminator.

5.6 Surface graph

The graph contains:

qualifying original mesh adjacencies, when present;

mutual 12-nearest-neighbor edges for both mesh and point forms.

An edge i,j is retained only if:

d
ij
	​

≤2.25s,
arccos∣n
i
	​

⋅n
j
	​

∣≤35
∘
,
max(∣
d
^
ij
	​

⋅n
i
	​

∣,∣
d
^
ij
	​

⋅n
j
	​

∣)≤0.60.

The last predicate rejects cross-gap edges that travel approximately through a sheet normal rather than along the sampled surface.

For every surviving edge:

b
ij
	​

=[1−(
2.25s
d
ij
	​

	​

)
2
]
2
,
ρ
i
	​

=
k
∑
	​

b
ik
	​

,w
ij
	​

=Q
24
	​

(
ρ
i
	​

ρ
j
	​

	​

b
ij
	​

	​

).

Zero-Q24 edges are removed.

Normals are oriented deterministically inside each connecting component by stable-key breadth-first propagation from tail-A seeds. A neighbor normal is flipped when necessary to maximize its dot product with the already oriented predecessor. Tail-B orientation must agree with the frozen candidate turn within 10
∘
, otherwise the station fails ORIENTATION_CONFLICT.

Across all components containing at least one A seed and one B seed, at least 75% of each tail’s seeds must participate. Otherwise the station fails NO_MATERIAL_CONNECTION.

5.7 Dirichlet field and electrical current

Set:

ϕ=0on tail A,ϕ=1on tail B,

and solve for every interior node:

j
∑
	​

w
ij
	​

(ϕ
i
	​

−ϕ
j
	​

)=0.

Solver:

float64 fixed-order CSR;

Jacobi-preconditioned conjugate gradient;

one numerical thread per station;

maximum 128 iterations;

relative infinity residual at most 10
−10
.

Failure gives SOLVE_FAILURE.

Edge current and node participation are:

J
ij
	​

=w
ij
	​

∣ϕ
i
	​

−ϕ
j
	​

∣,a
i
	​

=
2
1
	​

j
∑
	​

J
ij
	​

.

Let A=∑
i
	​

a
i
	​

. Require:

N
eff
	​

=
∑
i
	​

a
i
2
	​

A
2
	​

≥24

and:

∑
ij
	​

J
ij
	​

max
ij
	​

J
ij
	​

	​

≤0.12.

Failure gives LOW_CURRENT_PARTICIPATION.

5.8 Potential and normal-rotation certificate

For each k=1,…,7, the potential band

	​

ϕ
i
	​

−
8
k
	​

	​

≤
16
1
	​


must contain:

at least three current-active nodes;

at least 0.03A node-current mass.

Failure gives POTENTIAL_COVERAGE.

Project each oriented normal onto u
⊥
. Let θ
i
	​

 be its unwrapped signed angular displacement from tail A toward tail B, and let α be the frozen candidate turn.

For each of seven intermediate angular bins centered at mα/8, with half-width α/16:

at least two nodes;

at least 0.015A current mass.

Failure gives ANGULAR_COVERAGE.

Current on nodes with:

θ
i
	​

<−10
∘
orθ
i
	​

>α+10
∘

must be at most 0.08A. Otherwise: OFF_ARC_CURRENT.

For each potential band, compute the deterministic weighted median angle 
θ
ˉ
k
	​

, with endpoints 
θ
ˉ
0
	​

=0 and 
θ
ˉ
8
	​

=α.

Require:

k=0
∑
7
	​

max(0,
θ
ˉ
k
	​

−
θ
ˉ
k+1
	​

)≤5
∘

and:

k=0
∑
7
	​

∣
θ
ˉ
k+1
	​

−
θ
ˉ
k
	​

∣−α≤12
∘
.

Failure gives NORMAL_BACKTRACK.

A station satisfying every predicate is STATION_PASS.

5.9 Candidate decision

The candidate certifies only if:

at least 8 of 11 stations pass;

the longest consecutive passing run is at least 6;

at least one of stations 0–1 passes;

at least one of stations 9–10 passes.

Otherwise: AXIAL_STATION_COVERAGE.

No output line is fitted or moved. A passing candidate copies its frozen V16 render geometry exactly.

5.10 Ordering, NMS, and duplicates

Untouched V16 incumbents remain first in their existing order.

Certified rescues are appended in this deterministic order:

passing-station count, descending;

longest passing run, descending;

median normal-backtrack statistic, ascending;

median off-arc current fraction, ascending;

frozen V16 order, ascending;

candidate ID, ascending.

Then run exactly one unchanged ordinary NMS. Earlier records win conflicts:

legacy beats rescue;

the best-ranked rescue beats duplicate rescues;

there is no architecture-specific merge or second NMS.

5.11 Cache and resource contract

Per scan:

fixed header: 256 bytes;

fixed candidate record: 192 bytes;

maximum cache: 256+128×192=24,832 bytes.

Each candidate record contains only:

ID and V16 order;

status and 11-bit station mask;

11 one-byte reason codes;

quantized aggregate metrics;

a 32-byte profile hash.

No source point, vertex, face, edge, potential, path, or node list is serialized.

Execution caps:

input: at most 200,000 source points;

workers: min(32, physical_core_count);

station solve: single-threaded;

wall: 270 seconds;

process-tree RSS: 14 GiB;

no partial rescue output after any cap failure.

These remain inside the packet’s 300-second and 16-GiB limits. 

Pasted text

6. SOURCE-FREE FALSIFICATION GATES
Fixture construction

The positive fold cross-section is deliberately noncircular. Its tangent angle over an 8s transition is:

θ(z)=α(10z
3
−15z
4
+6z
5
),0≤z≤1.

It has:

straight 14s tails on both sides;

axial extent 32s;

candidate segment L=30s;

shell thickness 1.5s;

nominal sample pitch 0.5s;

deterministic normal displacement no greater than 0.04s.

The three angles are exactly 60°, 90°, and 120°.

Point forms remove connectivity. Fragmented forms remove transition observations in station slabs 1, 2, and 3 and delete an additional deterministic 10% of samples elsewhere; their expected valid mask therefore has exactly 8 passing stations with a seven-station consecutive run.

Base fixtures and frozen outcomes
Fixture	Required frozen outcome
SF01 FOLD60_MESH_C2	exactly one rescue; 11/11 stations pass
SF02 FOLD90_MESH_C2	exactly one rescue; 11/11 pass
SF03 FOLD120_MESH_C2	exactly one rescue; 11/11 pass
SF04 FOLD60_POINT_C2	exactly one rescue; 11/11 pass
SF05 FOLD90_POINT_C2	exactly one rescue; 11/11 pass
SF06 FOLD120_POINT_C2	exactly one rescue; 11/11 pass
SF07 FOLD60_FRAGMENTED	exactly one rescue; 8/11 pass; stations 1–3 NO_MATERIAL_CONNECTION
SF08 FOLD90_FRAGMENTED	exactly one rescue; 8/11; same station reasons
SF09 FOLD120_FRAGMENTED	exactly one rescue; 8/11; same station reasons
SN01 SILHOUETTE_FREE_EDGE	zero rescues; TAIL_B_BREADTH
SN02 OPENING_RIM	zero; TAIL_B_BREADTH
SN03 THICKNESS_RETURN_1P5S	zero; TAIL_B_BREADTH
SN04 DISCONNECTED_MITER90	zero; NO_MATERIAL_CONNECTION
SN05 LOCAL_ROUNDED_BRIDGE_4P5S	zero; AXIAL_STATION_COVERAGE
SN06 NONMONOTONE_BEAD_0_70_35_90	zero; NORMAL_BACKTRACK
SN07 FLAT_FALSE_CREASE12	zero; TURN_ELIGIBILITY
SN08 CORNER_CYCLE_TWO_ROUTE	zero; OFF_ARC_CURRENT
SN09 NEARBY_NONTOUCHING_PANELS_1P25S	zero; NO_MATERIAL_CONNECTION
SN10 UNSUPPORTED_PLANE_INTERSECTION	zero; NO_MATERIAL_CONNECTION
SN11 SCAN_BOUNDARY_TRUNCATION	zero; TAIL_A_BREADTH
SN12 FIVE_DUPLICATE_AXIS_ALIASES	five pre-NMS certificates; exactly one final output; four ORDINARY_NMS_SUPPRESSED

Any different count, station mask, or first-failure reason retires V31.

Metamorphic and determinism gates

Every base fixture must preserve its outcome under:

all 24 proper signed-axis permutation rotations;

translation by (17,−11,5)s;

the rotation represented by quaternion (1,2,3,4)/
30
	​

;

axis reversal;

parent A/B swap;

complete mesh-winding and normal reversal;

two fixed input point/face permutations;

uniform scale factors 1/4 and 4.

After inverse transformation and canonicalization:

result JSON must be byte-identical;

fixed cache must be byte-identical;

rendered audit pixels must be byte-identical.

The whole suite must also be byte-identical for:

one worker;

maximum workers;

three exact replays.

200,000-point stress

The stress fixture contains exactly:

200,000 samples;

128 eligible candidates;

eight positive folds;

120 fixed negatives.

Required:

exactly eight final rescues;

no cap or solver failures;

wall at most 270 seconds;

process-tree RSS below 14 GiB;

cache exactly within 24,832 bytes;

one-worker and maximum-worker byte identity.

The source-free harness, mechanism, renderer, fixtures, expected outcomes, and hashes must all be frozen before the first mechanism invocation. Any first failure is terminal, consistent with the packet’s one-shot rule. 

Pasted text

7. VISUAL GATES

Before any mechanism invocation, every physical fixture receives two manually signed 4096×4096 artifacts:

raw_native_multiview_4096.png
Original, undecimated points or triangles in perspective, axis view, both transverse views, and an opposite-skin visibility view. No DHFF graph, potential, current, or output line.

raw_station_sections_4096.png
Eleven native station cross-sections showing both tails, transition samples, gaps, thickness faces, nearby surfaces, and cycle routes. The input candidate line may be shown, but no certificate output.

After execution, every fixture receives:

station_flux_4096.png
Eleven station panels showing A/B seeds, retained graph edges, harmonic potential, edge-current thickness, passing mask, and first failing predicate.

phi_theta_certificate_4096.png
Current-weighted ϕ-versus-θ samples, potential bands, angular bins, weighted medians, off-arc mass, backtrack, and variation.

final_native_overlay_4096.png
Raw geometry with certified candidates, ordinary-NMS suppression, and the final retained line.

Manual review is mandatory for:

every positive activation;

every pre-NMS certificate;

every NMS suppression;

every fixture whose observed reason differs from expectation;

every real-scan V31/V16 delta.

A visual contradiction with the numeric certificate retires V31 as VISUAL_CAUSAL_MISMATCH; it is not permission to reinterpret the fixture or alter a predicate. This preserves the packet’s requirement that metrics alone cannot decide the mechanism. 

Pasted text

8. REAL-SCAN GATE
Label-closed selection

After source-free passage:

Canonically hash only each newly arrived scan’s raw geometry payload.

Exclude filenames, directories, metadata, acquisition role, family, and evaluator notes.

Sort by ascending SHA-256.

Freeze the lowest eight hashes before detector execution.

If fewer than eight scans are available, do not substitute historical scans and do not open gold.

Prospective execution

Run frozen V16 and V31 once on the selected eight scans.

Before any gold opening, require:

8/8 successful jobs;

one-worker/max-worker/replay byte identity;

unchanged V16 incumbent bytes;

resource compliance;

at least four V31-only final markers across at least two scans;

complete 4096×4096 multiview and flux audit for every pre-NMS certificate and final delta.

Fewer than four additions gives PROSPECTIVE_NO_ACTIVITY and retires V31. It does not justify selecting different scans.

The manual audit at this stage checks only that the stored certificate truthfully represents visible scan evidence; no evaluator labels, expected counts, or benchmark matching are consulted.

Smallest legitimate gold opening

Only after the label-closed gate passes, an independent evaluator opens complete line annotations for those same eight sealed scans. The detector is not rerun.

V31 passes this proof-of-signal gate only if all hold:

TP
added
	​

≥4,
FP
added
	​

≤1,
F1
V31
	​

−F1
V16
	​

≥0.03,

and:

exact bend count improves on at least two scans;

exact bend count worsens on zero scans;

no V16 true-positive incumbent is lost;

all added true positives are render-admissible.

Passage authorizes only a larger locked research benchmark. It does not open runtime/main or production. Failure terminally retires DHFF without threshold changes or a second prospective cohort.

9. EXPECTED FAILURE MODE

The most likely failure is recall loss on severely fragmented real folds.

A genuine fold may have enough visible geometry for human recognition but lack a tangent-compatible local graph or intermediate normal samples at four or more of the eleven disjoint stations. It will then fail as:

NO_MATERIAL_CONNECTION,

POTENTIAL_COVERAGE, or

ANGULAR_COVERAGE,

followed by candidate-level AXIAL_STATION_COVERAGE.

The decisive diagnostic is the native station_flux_4096.png: it will show broad tails but either no cross-fold current or missing current-carrying normal bins. That result would establish that a locally constructed manifold graph is still too brittle under the current scan fragmentation, despite avoiding V30’s optimizer collapse.

This is precisely the wall V31 is intended to hit quickly. If the fragmented source-free positives pass but the prospective scans fail this way, retire V31; do not enlarge the edge radius, reduce station coverage, lower the angular requirements, or add virtual completion.
