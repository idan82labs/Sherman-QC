RULING: EXECUTE_ONE

Execute one fresh source-only architecture: V56_CANONICAL_TURN_MEDIAN_LINE_RECONSTRUCTION_V1. A capability opening is not yet scientifically necessary. V52 demonstrably needs both recovery and pruning—52 TP, 27 FP, 48 FN, with a 91/100 frozen-pool ceiling—but V55 completed no scan and never measured a candidate’s transverse geometry. It falsified endpoint-schema totality, not source-only fold localization. 

Pasted markdown +2

1. Identity and one-shot contract

Full name: V56_CANONICAL_TURN_MEDIAN_LINE_RECONSTRUCTION_V1

Short name: CTMLR_V1

Precommit token: V56_CTMLR_MECHANISM_LOCKED

Terminal token: V56_CTMLR_TERMINAL_RETIRE

Operation: complete line-set replacement, with a source-certified byte-preserving hold for qualifying V52 lines.

No successor-in-place: no V56b, threshold relaxation, changed schema mapping, replacement input, partial rerun, failed-scan retry, or visual-driven modification.

V55 remains terminal: no V55 source file, cache, result, threshold table, conflict logic, or execution artifact may be imported or invoked.

Before opening any sealed source mesh, freeze one manifest containing the implementation hash, fixture hashes, all constants below, schema rules, primary/replay order, input hashes, output schema, visualizer hash, and the two tokens.

2. Insertion point and output semantics

Insert CTMLR immediately after byte-hash validation of:

the frozen source triangle mesh;

the frozen V52 prediction file;

the frozen union-pool file;

and before any final line-set assembly.

CTMLR creates a separate source-derived proposal path. Frozen rows are search seeds, not predictions to be accepted verbatim.

For every seed:

a V52 line that is directly certified as centered is emitted with its original endpoint bytes unchanged: HOLD_V52;

a V52 line that is not centered may be replaced by a newly reconstructed source-derived line: RECONSTRUCT_V52;

a pool-only seed may create a newly reconstructed line: RECONSTRUCT_POOL;

a seed without the complete certificate emits nothing.

After one duplicate-conflict collapse, the resulting set wholly replaces the V52 output. There is no immutable V52 prefix, no append stage, no expected-count prior, and no second NMS.

3. Exact inputs, population, and total schema audit
Source inputs

The detector may read only:

finite source-mesh vertices and indexed triangular faces;

frozen V52 line rows;

every row of the frozen union pool;

geometry-bearing fields listed below;

fixed row ordinals for deterministic bookkeeping.

Stored normals, candidate scores, truth matches, labels, CAD, filenames, scan IDs, part or family identities, A/B identities, expected counts, and benchmark results are not inputs.

Let:

B be the diagonal of the finite source-vertex bounding box;

h be the median length of all finite, positive, unique undirected mesh edges.

Require B>0 and 10
−7
B≤h≤0.05B.

Candidate population

For each scan, concatenate:

every frozen union-pool row;

every frozen V52 output line as a separately tagged seed.

The cap is 1,024 rows before exact-duplicate collapse. This is above the observed maximum pool size of 755 while remaining bounded. 

Pasted markdown

No row may be excluded because of candidate_source, route name, source kind, confidence, or provenance. candidate_source is logged but never branches geometry or scoring.

Supported geometry schemas
Code	Complete fields	Canonical materialization
E	finite line_start[3], line_end[3]	use the two endpoints
S	finite span_endpoints containing exactly two 3-vectors	use those two endpoints
A	finite anchor[3], finite nonzero axis_direction[3], finite span_mm > 0	symmetric segment about the anchor

For an endpoint representation with inputs x
0
	​

,x
1
	​

:

a=(x
0
	​

+x
1
	​

)/2,s=∥x
1
	​

−x
0
	​

∥,d=(x
1
	​

−x
0
	​

)/s.

For schema A:

d=
∥axis_direction∥
axis_direction
	​

,p
−
	​

=anchor−
2
span_mm
	​

d,p
+
	​

=anchor+
2
span_mm
	​

d.

Canonicalize the direction by choosing

k=arg
i∈{x,y,z}
max
	​

∣d
i
	​

∣

with ties resolved x<y<z, and negate d when d
k
	​

<0. Recompute ordered endpoints as a∓sd/2.

Numerical schema tolerances are fixed as:

ϵ
p
	​

=max(10
−7
B,10
−4
h);

ϵ
s
	​

=max(10
−7
B,10
−6
s);

ϵ
θ
	​

=10
−5
 radians.

When multiple complete encodings occur:

E and S must agree in direction and corresponding canonical endpoints within the stated tolerances;

A versus an endpoint encoding must agree in direction, span, and infinite-line location; its anchor must project inside the explicit segment expanded by ϵ
p
	​

;

fixed precedence after agreement is E, then S, then A.

Partial redundant fields are permitted only if individually consistent with the selected canonical line:

an anchor must lie on the line and within the segment;

an axis_direction must be parallel;

span_mm must equal the canonical span;

a lone endpoint must lie on the segment;

a present span_endpoints field of any length other than two is malformed.

Binding audit

Before any geometric evaluation, run the fixed parser once across all 30 primary and replay inputs and freeze schema_audit.json containing counts by schema combination and scan.

Required audit result:

100% of rows materialized;

zero malformed, unsupported, ambiguous, or inconsistent rows;

no scan above 1,024 concatenated rows;

primary and replay audit bytes identical;

all normalized values finite;

no code or constant changed after seeing the census.

Any failure immediately emits the terminal token. It is not converted into an endpoint-only population, a skipped row, or a revised schema.

This normalization is representation-preserving rather than learned: the symmetric A formula is fixed before the audit and consults neither the mesh nor any outcome.

4. Canonical ordering, duplicate handling, and cache
Input ordering

After materialization, sort seeds using IEEE-754 numeric total order on:

(d
x
	​

,d
y
	​

,d
z
	​

,a
x
	​

/B,a
y
	​

/B,a
z
	​

/B,s/B)

followed only by frozen origin-file ordinal and row ordinal. Strings, source kinds, candidate IDs, and scan identities do not affect ordering.

Exact input duplicates

Construct connected components where two seeds satisfy all three:

angular difference ≤ϵ
θ
	​

;

midpoint-to-infinite-line distance in both directions ≤ϵ
p
	​

;

canonical endpoint and span differences ≤ϵ
s
	​

.

Keep the first canonical seed and record only an integer multiplicity. Do not perform a broader near-duplicate collapse before source measurement.

Output duplicate collapse

Two certified proposals conflict only when:

direction difference is at most 1.0
∘
;

symmetric midpoint-to-line distance is at most

τ
dup
	​

=max(4h,0.003B);

overlap after projection onto their sign-aligned mean direction is at least 0.70 of the shorter span.

Use connected components. Select one winner by this lexicographic tuple:

HOLD_V52 before either reconstructed class;

larger certified-station count;

larger longitudinal coverage;

smaller median transverse-center error divided by B;

smaller median line-fit residual divided by B;

smaller median turn-IQR width divided by W;

larger median turn angle;

lower canonical seed ordinal.

Do not suppress nonparallel crossing lines or parallel lines beyond the duplicate distance.

Compact cache schema

Canonical JSON, sorted keys, shortest round-trip finite binary64 formatting, native Python scalar types, and allow_nan=false:

version
input_sha256:
  source
  v52
  pool
bbox_diagonal
median_mesh_edge
schema_counts
input_row_count
exact_duplicate_count
canonical_seed_count
candidate_rejection_counts
proposal_count
output_count
outputs[]:
  class
  source_seed_ordinal
  line_start[3]
  line_end[3]
  certified_station_count
  longitudinal_coverage
  longest_station_run
  median_turn_degrees
  median_turn_purity
  median_center_error_over_B
  median_fit_error_over_B
wall_ms
peak_rss_bytes

No source point, vertex, face, section polyline, triangle membership, support set, station-support mask, or support bitset may be serialized.

5. Deterministic geometric algorithm and constants
Fixed constants
Quantity	Frozen value
Longitudinal stations	7
Station fractions	0.10+j(0.80/6), j=0,…,6
Minimum seed span	max(12h,0.015B)
Transverse search half-width W	clamp(20h,0.012B,0.035B)
Section resampling step Δ	clamp(1.5h,0.0005B,0.002B)
Fine lag ℓ
1
	​

	clamp(4h,0.0025B,0.005B)
Middle lag ℓ
2
	​

	clamp(8h,0.005B,0.010B)
Coarse lag ℓ
3
	​

	clamp(16h,0.010B,0.020B)
Section crop radius	W+2ℓ
3
	​


Contour join tolerance	max(0.05h,10
−6
B)
Cross-scale locus tolerance ρ	max(3Δ,0.002B)
Minimum turn	15
∘

Maximum absolute turn	170
∘

Minimum peak prominence	4
∘

Minimum signed-turn purity	0.85
Maximum normalized tail RMS	0.04
Maximum turn-IQR width	0.75W
Minimum certified stations	5 of 7
Minimum station coverage	0.60 of seed span
Minimum longest station run	3
Maximum fitted-line/seed angle	7.5
∘

Line-consensus threshold	τ
f
	​

=max(3h,0.002B)
Transverse-center threshold	τ
c
	​

=max(2h,0.0015B)
Maximum median center error	max(h,0.00075B)
Intersection-segment cap	16,384 per station

clamp(x,l,u)=min(max(x,l),u).

A. Mesh-section construction

For every seed with sufficient span and every station q
j
	​

:

Define the section plane through q
j
	​

 with normal equal to the seed direction.

Build a deterministic in-plane basis:

choose the global axis least aligned with the seed direction, ties x<y<z;

project and normalize it to obtain u;

set v=d×u.

Query a deterministic triangle BVH only inside the fixed crop radius.

Classify triangle vertices relative to the plane with adaptive/exact orientation predicates. Coplanar triangles inside the crop make that station ambiguous; they are not perturbed by an arbitrary floating-point epsilon.

Use a half-open, vertex-index-keyed symbolic perturbation rule for vertex and edge hits.

Quantize contour endpoints to the join grid using round-to-nearest-even. Remove duplicate undirected segments.

Trace maximal paths without crossing graph vertices of degree greater than two. Open contours are allowed. Closed contours start at the lexicographically smallest node and take the lexicographically smaller outgoing direction.

Retain only contours whose closest point to q
j
	​

 is within W and whose arc length is at least 2ℓ
3
	​

.

Resample by linear arc-length interpolation at spacing Δ.

A station with no eligible contour is invalid for that seed, not a scan failure.

B. Multiscale turn events

For every retained contour, scale ℓ
r
	​

, and resampled arc coordinate s
i
	​

 having complete left and right support:

Fit a two-dimensional total-least-squares line to the left interval
[s
i
	​

−ℓ
r
	​

,s
i
	​

−ℓ
r
	​

/2].

Fit another to the right interval
[s
i
	​

+ℓ
r
	​

/2,s
i
	​

+ℓ
r
	​

].

Orient both tangents with increasing contour arc length.

Compute

θ
r
	​

(s
i
	​

)=arccos(clamp(t
r
−
	​

⋅t
r
+
	​

,−1,1)).

Define normalized tail error as the larger TLS orthogonal RMS divided by ℓ
r
	​

.

A scale-specific maximum is eligible when:

it is the maximum within 2Δ, with ties going to smaller arc coordinate;

θ
r
	​

≥15
∘
;

normalized tail error ≤0.04;

its turn exceeds the median turn within 2ℓ
r
	​

 by at least 4
∘
.

A persistent event must:

include an eligible middle-scale maximum;

have eligible maxima at at least two of the three scales;

have maximum-minus-minimum peak arc position ≤ρ.

Its provisional location is the turn-angle-weighted median of the participating arc positions.

Across all contours, find the greatest persistent-event median angle. Retain events with at least 0.85 of that angle; choose the one closest to q
j
	​

, then larger angle, then canonical contour order.

The station is ambiguous and invalid when a second retained event:

differs in seed distance by at most ρ; and

has at least 0.85 of the chosen event’s angle.

This deliberately rejects a seed centered between two equally plausible folds while allowing a seed clearly closer to one of them.

C. Turn-median locus

Around the selected event, over [s−ℓ
3
	​

,s+ℓ
3
	​

]:

Estimate fine-scale oriented tangents with centered chords of half-width ℓ
1
	​

/2.

Compute signed adjacent tangent changes α
i
	​

.

Use w
i
	​

=∣α
i
	​

∣ as turn mass.

Require:

15
∘
≤
i
∑
	​

w
i
	​

≤170
∘
,
∑
i
	​

w
i
	​

∣∑
i
	​

α
i
	​

∣
	​

≥0.85.

Interpolate the cumulative-turn positions s
25
	​

,s
50
	​

,s
75
	​

.

Require:

s
75
	​

−s
25
	​

≤0.75W

and at least ℓ
2
	​

 of usable low-error contour on both outer sides.

The station’s source-derived fold-locus point is c
j
	​

=x(s
50
	​

). A distributed cylinder is rejected by the prominence and concentration requirements; an S-shaped or double-turn section is rejected by signed-turn purity or ambiguity.

D. Deterministic longitudinal line reconstruction

Given all valid c
j
	​

:

Enumerate every station pair separated by at least 0.40 of the seed span.

Form the line through the pair.

Treat a locus as an inlier when its perpendicular distance is at most τ
f
	​

.

Rank pair models by:

larger inlier count;

larger station-fraction coverage;

longer consecutive run;

smaller median residual;

smaller maximum residual;

lower station-pair indices.

Require at least five inliers, coverage at least 0.60, and a run of at least three.

Refit once by three-dimensional PCA over those inliers.

Canonicalize the fitted direction.

Require fitted direction versus seed direction ≤7.5
∘
.

No random sampling and no iterative threshold update are allowed.

6. Direct transverse-localization certificate

This is the binding positive certificate and the principal scientific test.

For a proposed output line, intersect it with every certified station plane. On the exact selected section contour, compute the closest arc coordinate s
r
	​

 to the intersection r
j
	​

.

A station certifies the line only when all conditions hold:

∥r
j
	​

−x(s
50
	​

)∥≤τ
c
	​

,
s
25
	​

≤s
r
	​

≤s
75
	​

,
∣s
r
	​

−s
50
	​

∣≤0.25(s
75
	​

−s
25
	​

)+Δ.

The line must retain at least five certified stations, coverage of at least 0.60, a longest run of at least three, maximum center error ≤τ
c
	​

, and median center error ≤max(h,0.00075B).

Refit once on the surviving certified stations and recheck the full certificate. A second refit or changed inlier set is forbidden.

V52 safety hold

For a V52-origin seed:

First test the original V52 line itself against the certificate.

If it passes, emit the original V52 endpoint values and serialization unchanged.

Otherwise test the reconstructed line.

Emit the reconstructed line only if it passes.

Otherwise prune the V52 line.

For a pool-only seed, only the reconstructed line may be emitted.

For a reconstructed line, preserve longitudinal extent by projecting the seed’s two canonical endpoints onto the fitted line and using those two projected coordinates as the new endpoints. Reject a nonfinite or zero resulting span.

This certificate does not mathematically prove correspondence to locked truth. It does directly establish that the reported line crosses the measured median of localized surface turning at multiple longitudinal sections, rather than merely running parallel to an offset fold.

7. Abstention rules
Candidate-level abstention

A seed emits no proposal under exactly these reason codes:

REJECT_SHORT_SEED

REJECT_NO_SECTION

REJECT_COPLANAR_OR_BRANCH_AMBIGUITY

REJECT_NO_PERSISTENT_TURN

REJECT_MULTIPLE_TURN_EVENTS

REJECT_DISTRIBUTED_OR_IMPURE_TURN

REJECT_INSUFFICIENT_STATIONS

REJECT_LINE_CONSENSUS

REJECT_DIRECTION_DRIFT

REJECT_TRANSVERSE_CENTERING

REJECT_OUTPUT_DEGENERACY

REJECT_DUPLICATE_LOSER

Zero final lines is a valid complete scan output, not an abstention.

Scan-level abstention

There is no recoverable scan fallback. Any scan-level condition is terminal for V56:

invalid mesh indexing or nonfinite source geometry;

invalid B or h;

schema-audit failure;

more than 1,024 input rows;

section-segment overflow;

unresolved robust-predicate failure;

output-schema or serialization failure;

resource breach;

leakage;

primary/replay mismatch.

Silently restoring V52 on an unprocessable scan would make a full-replacement result scientifically uninterpretable, so it is forbidden.

8. Source-free preflight and immediate failure gates

Before locking the precommit, the implementation must pass this fixed source-free suite:

E, S, and A encodings of one line canonicalize identically.

Endpoint reversal and direction negation are invariant.

Consistent redundant encodings pass.

Inconsistent redundant encodings fail with the exact audit reason.

Nonfinite vectors, zero direction, zero/negative span, and malformed span_endpoints fail.

Exact duplicate rows collapse to one seed.

Centered sharp 90
∘
 fold returns one line.

Centered rounded 60
∘
 fold returns one line.

A single plane returns zero.

A seed offset by 0.40R returns one recentered line, not the original offset line.

One seed equidistant between two equal folds returns zero as ambiguous.

Two seeds, each near a different fold, return two lines.

A distributed cylinder returns zero.

An open boundary without a fold returns zero.

A diagonal cross-panel seed returns zero.

Parallel folds separated by 0.08B remain two outputs.

Canonical primary/replay serialization is byte-identical.

Nonfinite output serialization is rejected.

Every positive fixture must meet direction error ≤1
∘
 and transverse error ≤τ
c
	​

; every negative fixture must emit zero.

After precommit, immediately terminally retire V56 upon:

any input or implementation hash mismatch;

any audit count other than complete 100% coverage;

any truth, CAD, filename, identity, expected-count, or benchmark access before output freeze;

any line produced without the full transverse certificate;

any nonfinite value;

peak RSS ≥16 GiB;

any scan wall time ≥300 seconds;

any uncaught worker failure;

incomplete primary output;

incomplete replay;

any primary/replay byte difference;

any output or cache field outside the frozen schema.

No score is produced unless all 30 primary records and all 30 replay records freeze successfully.

9. Minimum positive retention gate

V56 is retained only if every condition below holds after the single score.

Aggregate improvement

TP ≥62;

FP ≤20;

FN ≤38;

precision ≥0.75;

recall ≥0.62;

F1 ≥0.6810055866, exactly 0.10 above V52;

exact-count accuracy ≥12/30, twice V52;

all 52 truth instances recovered by V52 remain recovered;

at least ten previously missed truth instances are recovered;

at least seven net V52 false positives are removed.

Breadth

net TP gain occurs in at least five of the 15 topology families;

no family recall is below its V52 value;

minimum family recall is strictly greater than zero.

Exact-case zero regression

For S005, S006, S007, S013, S026, and S027, the final ordered line arrays must be byte-identical to the frozen V52 arrays. A score-equivalent but geometrically changed result fails this gate. These cases were already exact and have no justified risk budget. 

Pasted markdown

Operational and visual retention

30/30 primary and replay success;

30/30 byte-identical replay;

no scan-level abstention;

P95 wall time ≤300 seconds;

peak RSS <16 GiB;

100% finite, render-admissible markers;

every mandatory visual passes;

no automatic visual failure.

A result that improves only recall, only precision, or only aggregate F1 is rejected. This run must demonstrate both recovery and safe pruning.

10. Mandatory native-resolution visual audit

Render after output and replay hashes freeze, using a separately frozen visualizer that cannot alter predictions. Visual-only section generation is not a detector invocation and cannot feed values back into the output.

Review at native scan-render resolution:

every changed scan;

all six exact V52 scans;

S011, S016, S024, and S029;

every reconstructed line;

every pruned V52 line.

For every accepted or reconstructed line, show:

at least three native-resolution 3D views with V52 and V56 distinguished;

all seven axis-normal sections;

the selected contour;

the s
25
	​

, s
50
	​

, and s
75
	​

 turn locations;

the output-line intersection;

certified versus rejected station status;

the former V52 or pool seed when geometry changed.

Mandatory positive signatures

An accepted line must visibly show:

a localized turn band with usable material tails on both sides;

the line intersection inside the turn interquartile interval;

the intersection near the turn median;

a longitudinally straight sequence of median-turn loci;

continuous placement on one physical fold across views;

no dependence on a panel boundary or isolated mesh spike.

For a pruned V52 line, sections must visibly show at least one precommitted negative condition: flat interior, distributed turn, competing folds, boundary-only turn, longitudinal inconsistency, or transverse offset without a stable reconstructed locus.

Automatic visual failures

Any one of the following rejects V56:

an accepted axis lies visibly on a flat panel interior;

an axis is outside the rounded corner’s visible turn band;

an axis follows an opening rim, free boundary, hole edge, or scan tear;

a diagonal line crosses several unrelated folds;

a line jumps between opposite skins or disconnected components;

one output bridges two separate folds;

duplicate outputs mark one fold;

a marker floats off the observed surface without the section evidence explaining a theoretical centerline;

a changed line is supported only at one end;

a certified station’s rendered intersection is outside its displayed [s
25
	​

,s
75
	​

];

any exact V52 case differs visually or bytewise;

rendering is cropped, downsampled, or omits the local transverse neighborhood.

11. Leakage, determinism, latency, memory, and reliability audits
Leakage

The inference worker receives no filename or scan identifier. It branches only on numeric source geometry, field presence, V52-origin versus pool-origin, and frozen ordinals.

candidate_source, candidate IDs, family labels, and filenames may appear in audit logs only after inference decisions are complete; they cannot enter scores, thresholds, population inclusion, or tie-breaking.

The six exact-case names and four difficult-case names may be used only by the post-freeze visual wrapper and retention scorer. Truth remains unopened until complete primary and replay output hashes are committed.

Determinism

IEEE-754 round-to-nearest; no fast-math;

adaptive/exact geometric predicates;

deterministic BVH construction;

all sets converted to sorted arrays before reduction;

fixed eigenvector sign convention;

candidate computations may run in parallel only as independent ordinal-indexed tasks;

no shared floating-point reduction;

BLAS/OpenMP internal thread count fixed to one;

results assembled strictly by canonical ordinal;

no random seed because no random operation exists.

Resource bounds

one triangle BVH per scan;

at most 1,024 seeds;

seven local section queries per seed;

only the current candidate’s contour data retained in working memory;

16,384 intersection segments maximum per station;

hard per-scan wall stop at 300 seconds;

hard RSS stop below 16 GiB.

Job reliability

No fallback or skipped scan is counted as success. Retention requires 60 completed executions—30 primary and 30 replay—and therefore gives a directly observed 100% job-success rate on this engineering lane, while making no enterprise reliability claim.

12. Why this attacks the current residual

V52’s pool already contains candidates for 91 of 100 truths, so a mechanism that safely adjudicates and relocates existing proposals can still produce a large recall gain. At the same time, the 27 inherited false positives make another append-only selector structurally incapable of the precision target. 

Pasted markdown

CTMLR attacks both defects with one causal measurement:

False positives: flat, diagonal, boundary, or floating lines cannot survive without a persistent localized transverse turn and a longitudinally coherent median-turn line.

False negatives: pool seeds near unselected or slightly misplaced bends can generate new reconstructed axes.

Mislocalized axes: a proposal may move transversely to the measured turn median rather than merely receive a higher support score.

Baseline safety: a correctly centered V52 line stays byte-identical; it is not needlessly refitted.

This is not V53: there is no mesh-adjacency/plane-lattice route agreement, bank correspondence, or cross-route population gate.

This is not V54: no longitudinal support mass or exclusive ownership is treated as localization evidence. V54’s flat-interior additions showed why direct transverse measurement is necessary. 

Pasted markdown

This is not a V55 rerun or rescue. V55 evaluated frozen candidate lines without geometry refit and never reached a completed measurement. CTMLR uses a total pre-execution representation contract, treats rows only as bounded search seeds, reconstructs a new source-derived line from median-turn loci, and may output geometry absent from every frozen row. V55 and its terminal artifacts remain untouched. 

Pasted markdown +1

The bounded limitation is explicit: CTMLR cannot recover a bend lying outside every seed’s W-tube. That limitation makes this run falsifiable rather than an implicit claim of complete candidate birth.

13. Research grounding

Established ideas. Surface feature lines are commonly formulated through localized curvature extrema or variation of surface differential quantities. Ohtake, Belyaev, and Seidel demonstrated ridge/valley extraction from dense triangle meshes using multilevel implicit fitting and curvature derivatives. 
Heriot-Watt Research Portal
 Pauly, Keiser, and Gross showed that multiscale local geometric analysis can improve line-feature detection robustness on point-sampled surfaces. 
Guibas Lab
 Demarsin and colleagues used normal variation, curvature partitioning, clustering, and graph processing to extract sharp and smooth feature lines from industrial point clouds; their sheet-metal experiment also illustrates that noise can remove otherwise detectable lines, which argues for a strict abstaining certificate rather than assuming every local maximum is real. 
CAD Journal

The hypothesize-and-verify principle for fitting a model in the presence of outlying observations is established by Fischler and Bolles’ RANSAC work. CTMLR uses the same consensus concept, but exhaustively evaluates every admissible station pair because there are only seven stations; it therefore removes random sampling and its replay variance. 
ACM Digital Library
 Adaptive exact predicates are an established way to prevent floating-point sign errors from changing combinatorial geometric decisions near degeneracies. 
Springer

Sherman-specific hypotheses. The seven station positions, schema precedence, symmetric anchor materialization, tube width, three turn scales, turn-median definition, purity and concentration thresholds, V52 safety hold, duplicate relation, and retention gates are not validated by those papers. They are frozen engineering hypotheses. The single sealed primary/replay execution, score, and native-resolution section audit constitute their test.
