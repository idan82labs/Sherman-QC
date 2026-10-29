RULING
Authorize V30_AXIAL_MULTIPATH_FOLD_CERTIFICATE_V1 (AMFC_V1) for immediate implementation and execution as a research-only, rescue-only sidecar to frozen V16.
Causal hypothesis: A genuine formed bend is a single observed surface skin that supports several independent, short, tangent-compatible paths carrying the candidate’s normal rotation from one broad material tail to the other at repeated axial stations; terminal edges, openings, thickness returns, sharp seams, and local accidental bridges cannot sustain that same distributed path family.
This is not V29b. It uses neither calibrated frames nor visibility rays. It is also not paired-skin matching, exact-mate ownership, tube-component parity, a topology hash, circle/focal fitting, profile grammar, geodesic chord deficit, or candidate reconciliation. It asks a new, narrower question of the source geometry already available:

Does this rejected V16 candidate lie inside a repeatedly observed, one-skin surface bridge that actually carries the proposed bend rotation?

The method is deliberately rescue-only. Missing or poor observations produce no rescue, not evidence that a legacy V16 line is false. That asymmetry is important for the deliberately bad scans in the prospective queue.
Published diffusion work formalizes intrinsic geometry through graph connectivity and motivates using the multiplicity of paths rather than trusting one shortest path. CMU School of Computer Science Menger’s theorem supplies the classical connection between independent paths and the size of a separating bottleneck. European Digital Mathematics Library Local-neighborhood PCA is established machinery for estimating point-cloud tangent planes and normals. ACM Digital Library
Those are the established ingredients. The Sherman-specific hypothesis is that repeated, curvature-carrying, node-disjoint cross-tail paths constitute a zero-shot bend certificate. No cited work establishes that claim; the source-free aliases and the prospective part are therefore mechanism tests, not demonstrations of an already proven method.

1. Exact architecture and authorization
Architecture: V30_AXIAL_MULTIPATH_FOLD_CERTIFICATE_V1
Primitive: CANDIDATE_CONDITIONED_TANGENT_GRAPH_NODE_DISJOINT_PATHS_V1
Authorization: Implement, statically lock, build the source-free fixtures, inspect their native-resolution raw geometry, execute the complete preregistered source-free and metamorphic schedule, and—only after every gate passes—consume exactly one prospective unseen real scan.
No historical scan, locked benchmark, gold, label, CAD, drawing, expected count, or V29 artifact may be opened during implementation or source-free execution.
AMFC is a candidate rescue certificate, not a new candidate generator and not a V16 veto.

2. Exact insertion point and output behavior
Insertion is:


Run frozen V16 candidate construction and frozen V16 candidate selection through creation of:


the complete pre-NMS candidate records;


their frozen candidate frames;


the frozen V16 legacy accept/reject bit and rank.




Apply AMFC only to legacy-rejected candidates.


Form:
Cunion​=CV16 accepted​∥CAMFC rescued​
with every legacy candidate first and byte-unchanged.


Run exactly one existing frozen ordinary V16 NMS.


Emit the resulting research output.


AMFC may not:


remove, modify, shorten, rotate, or re-score a V16-accepted line;


create a line not already present as a V16 pre-NMS candidate;


invoke a preliminary, candidate-local, pairwise, or second NMS;


alter the frozen ordinary NMS;


allow a rescued candidate to precede a legacy candidate.


Rescues are ordered by the fixed tuple:
(−Nvalid stations​,−Nvalid paths​,Qmedian turn error​,V16 rank,candidate key)
where all quantitative values are quantized as specified below. Ties use only the frozen V16 candidate key—not filename, scan identity, part family, or arrival metadata.

3. Allowed and prohibited inputs
Allowed inside the AMFC worker


Raw source coordinates.


Triangle indices when a source mesh exists.


Frozen V16 source-spacing scalar s.


Frozen V16 pre-NMS candidate key and rank.


Candidate line L=(c,a).


The two frozen parent-plane equations
πj​: nj⊤​x=hj​,j∈{0,1}.


Frozen parent support centroids or frozen outward tail directions.


Frozen candidate axial support interval [u0​,u1​].


The frozen V16 accept/reject bit.


Deterministically computed local normals and in-memory spatial indices.


Source point or vertex indices only as transient array addresses; they may not enter ranking, output identity, or serialized caches.


Prohibited
The worker input schema must make it impossible to supply:


filename, directory, part number, family, customer, or identity;


content hash, arrival order, or prospective-selection key;


expected bend count;


good-scan or bad-scan status;


deviation status;


labels, gold, manual selections, or adjudication;


historical or locked-benchmark outcomes;


V16 performance feedback;


benchmark feedback of any kind;


CAD, STEP, drawing, MBD, Revit, nominal dimensions, or tolerances;


acquisition poses, intrinsics, calibrated range frames, visibility rays, or V29 intermediates;


color, intensity, texture, or reflectivity;


production/runtime/main outputs or feedback.


Prospective content hashes may be used outside the worker to select the real part. They are never passed to V16 or AMFC.
The worker runs with an allow-listed binary input schema in a temporary research directory with no network, no historical-data mount, and no evaluator or annotation mount.

4. Frozen deterministic mechanism
4.1 Candidate validation and scale
Use the frozen V16 spacing scalar s. AMFC may not estimate a replacement scan scale.
A candidate is AMFC-eligible only when:
s>0,u1​−u0​≥64s,
all frame values are finite, and:
∣∥a∥−1∣≤10−4,∣∥nj​∥−1∣≤10−4,
∣a⊤nj​∣≤sin3∘.
The acute angle between the unoriented parent normals is:
α=arccos(clamp(∣n0⊤​n1​∣,0,1)).
Because unoriented normals admit both the acute and supplementary surface turn, the permitted target set is:
T={α, π−α}∩[15∘,165∘].
An empty T makes the candidate ineligible.
The two outward in-plane tail vectors are taken from the frozen frame. Where the frame contains centroids but not signed tail vectors, derive only the sign:
t~j​=∥nj​×a∥nj​×a​,
and choose tj​∈{t~j​,−t~j​} pointing toward the frozen parent support centroid. No source search or optimization may re-estimate the frame.
4.2 Point normals
Compute normals once per scan and reuse them for all candidates.
For every source point or mesh vertex:


obtain the 16 nearest distinct neighbors within radius 3s;


require at least eight;


compute the 3×3 covariance in float64;


solve it with a fixed 12-sweep Jacobi eigensolver;


take the eigenvector of the smallest eigenvalue as the unoriented normal νi​;


accept it as surface-like only if
λ0​+λ1​+λ2​λ0​​≤0.12.


No BLAS eigensolver, stochastic normal orientation, learning, smoothing iteration, or adaptive neighborhood is allowed.
4.3 Axial stations and local source window
Use exactly nine stations:
zk​=u0​+8s+k8u1​−u0​−16s​,k=0,…,8.
For each station, collect valid-normal source nodes satisfying:
∣(pi​−c)⊤a−zk​∣≤6s
and
​(I−aa⊤)(pi​−c)​≤32s.
The source window is capped at 8,192 nodes per station.
If the window exceeds that count:


voxelize candidate-local coordinates at 0.75s;


retain the lexicographically smallest quantized-coordinate record in each voxel;


if more than 8,192 remain, emit STATION_NODE_CAP.


A cap on any eligible candidate is a resource-contract failure; it may not be treated as ordinary rejection.
Node identity uses candidate-local coordinates quantized at s/1024, followed by exact IEEE coordinate bits only to break a true quantization tie. Original point order is not a semantic key.
4.4 Broad tail seeds
Let the projection of c onto parent plane j be:
qj​=c+(hj​−nj⊤​c)nj​.
For each local node define:
xij​=(pi​−qj​)⊤tj​,dij​=∣nj⊤​pi​−hj​∣.
Node i belongs to tail seed set Aj​ when:
4s≤xij​≤12s,
dij​≤1.25s,
∣νi⊤​nj​∣≥cos25∘.
A tail seed set is valid only when it contains at least 24 nodes, spans at least 8s axially within the station slab, and spans at least 4s in xij​.
This deliberately excludes narrow thickness walls from serving as a second material tail.
4.5 Tangent-compatible source graph
Construct an undirected graph from:


source mesh edges whose endpoints survive the local selection; and


reciprocal 12-nearest-neighbor edges inside the station window.


Every edge must independently satisfy the same geometric compatibility tests.
For d=pj​−pi​ and ℓ=∥d∥, retain the edge only when:
0.2s≤ℓ≤2.75s,
∣d⊤a∣≤2.25s,
ρij​=ℓmax(∣d⊤νi​∣, ∣d⊤νj​∣)​≤0.30,
and
βij​=arccos(∣νi⊤​νj​∣)≤32∘.
Thus, spatial proximity alone cannot bridge two nearby layers or a sharp miter.
The fixed integer edge cost is:
wij​=1+round(4096sℓ​+1024ρij​+512πβij​​).
The station graph is capped at 98,304 undirected edges. Reaching the cap is a hard resource failure, not a negative classification.
4.6 Independent path extraction
Split every non-seed node into an in-node and out-node with unit capacity. Add a super-source connected to A0​ and a super-sink connected from A1​.
Compute at most eight minimum-cost internally node-disjoint paths using successive shortest augmenting paths with:


integer costs only;


fixed Johnson potentials;


Dijkstra ties resolved by node key, then predecessor key;


no randomized queue ordering;


no approximate solver.


A path is geometrically valid only if all of the following hold:


It contains 5–64 graph edges.


Its source-space length G satisfies:
G≤40s.


For endpoint chord length D:
G/max(D,s)≤1.65.


Its first and last normals align with the corresponding parent normals within 15∘, using absolute dot products.


It contains at least three internal nodes belonging to neither tail seed set.


Its total axial excursion is at most 8s.


Orient the path normals only for turn integration:


orient the first normal toward n0​;


flip each following normal when necessary so that its dot product with the previous oriented normal is nonnegative.


For consecutive oriented normals define:
δr​=atan2(a⊤(νr​×νr+1​),νr⊤​νr+1​).
Then:
T=​r∑​δr​​,V=r∑​∣δr​∣,M=max(V,10−9)T​.
The path must satisfy:
θ∈Tmin​∣T−θ∣≤12∘
and
M≥0.72.
The second condition rejects an S-shaped or back-turning bead that happens to have the correct net endpoint angle.
4.7 Station and candidate verdicts
A station is valid when:


both broad tail seed sets are valid;


at least four of the eight extracted paths are valid;


the accepted path midpoints span at least 4s axially.


A candidate receives AMFC_PASS only when:


both tail sets are valid in at least 7 of 9 stations;


at least 6 of 9 stations are valid;


at least four valid stations are consecutive;


at least 24 paths are valid in total;


the median target-turn error over all valid paths is at most 8∘;


no cap, NaN, overflow, solver, or frame error occurred.


Every other candidate receives no rescue. Diagnostic reason codes are:


FRAME_INVALID


SPAN_SHORT


TAIL_MISSING


GRAPH_DISCONNECTED


PATH_BOTTLENECK


TURN_TOO_SMALL


TURN_MISMATCH


TURN_NONMONOTONE


AXIAL_COVERAGE


RESOURCE_CAP


Insufficient evidence is never converted into a V16 deletion.
4.8 Bounds, caches, and identity


Maximum eligible rejected candidates per scan: 256.


More than 256 is RESOURCE_CAP; do not truncate to a favorable subset.


Maximum concurrent candidate workers:
min(physical CPU cores,16).


Candidate output is sorted after all workers complete; worker count cannot affect bytes.


Per-candidate serialized record: at most 512 bytes.


Complete V30 summary sidecar: at most 132 KiB.


No point coordinates, vertex lists, graph edges, path nodes, face lists, or source-support arrays may be serialized.


The visual renderer consumes paths from ephemeral memory during execution.


V30_candidate_id is:
SHA256("V30_AMFC_V1"∥frozen V16 candidate key).


The ID is a replay/tie key only. It may not enter the pass predicate.

5. Smallest source-free causal and hard-negative suite
Fixtures are direct analytic 3D surfaces. They do not use cameras, rays, depth images, z-buffers, or synthetic sensor rendering.
The base rounded-fold cross-section is:
q(t)=⎩⎨⎧​(t,0),(Rsinϕ, R(1−cosϕ)),(Rsinθ+τcosθ, R(1−cosθ)+τsinθ),​−W≤t≤0,0≤ϕ≤θ,0≤τ≤W.​
Extrude it along the axial coordinate y:
P(y,⋅)=(qx​, y, qz​).
Sampling deletion represents an unobserved part of a valid surface; it does not create contradictory coordinates or multiple depths at a pixel.
FixturePhysical constructionRequired resultSF01_FOLD90_MESH90∘, R=6s, W=16s, axial length 112s, triangle mesh; candidate axis perturbed 0.5∘, line offset 0.5sExactly one AMFC_PASSSF02_FOLD60_POINT60∘, R=4s, point-only analytic sampling, deterministic density modulation, normal displacement bounded by 0.08sExactly one AMFC_PASSSF03_FOLD120_FRAGMENTED120∘, R=10s, mesh with deterministic missing patches; no gap wider than 2s transversely and no more than two adjacent station slabs affectedExactly one AMFC_PASSSF04_ROUNDED_OPENING_RIMBroad planar sheet ending in a rounded slot/opening rim; no second broad material tailNo rescue; TAIL_MISSINGSF05_DISCONNECTED_MITER_SEAMTwo broad planes forming a 90∘ miter but represented as separate physical components with a 0.4s seam gapNo rescue; GRAPH_DISCONNECTEDSF06_THICKNESS_RETURNBroad plane, rounded edge, and only a 2.5s-wide return/thickness faceNo rescue; TAIL_MISSINGSF07_LOCAL_TAB_BRIDGETwo broad tails connected by a smooth fold only over two of nine station regionsNo rescue; AXIAL_COVERAGESF08_NONMONOTONE_WELD_BEADG1-continuous extruded cross-section with signed turns +70∘,−50∘,+70∘; net 90∘, total variation 190∘No rescue; TURN_NONMONOTONESF09_FLAT_FALSE_CREASEOne continuous flat panel with a syntactically valid 90∘ candidate frameNo rescue; TURN_TOO_SMALL
A separate SF10_RESOURCE_STRESS contains 200,000 analytic points and 192 spatially separated partial-bridge candidate assemblies. Each exercises graph construction and path extraction but has only two valid station regions. Required output: zero rescues and no cap.
Do not construct a geometrically identical smooth fold and smooth seam with opposite semantic labels. Such a pair would be an observational nonidentifiability statement, not a valid geometry-detector fixture.

6. Static precommit, harness validity, and immediate execution
Gate 0 — static precommit
Before the mechanism lock, pass all of these checks:


Research-only target; no runtime/main or production imports or writes.


Worker input matches the allow-listed schema and contains no metadata fields.


No evaluator, manual-reference, benchmark, CAD, range-frame, or V29 dependency.


Exactly one call site to the frozen ordinary NMS.


Legacy records precede every rescue and are not mutated.


All constants, caps, predicate order, quantizers, and reason-code precedence are literal and hashed.


No RNG, adaptive threshold, learned weight, environment-dependent threshold, or clock input.


Fixed float64/Jacobi behavior; compiler fast-math disabled.


Every dynamic structure is bounded by the declared caps.


Cache schema cannot serialize raw source geometry or path membership.


Then freeze:


mechanism source hash;


constants hash;


compiler and flags hash;


input/output schema hash;


frozen V16/NMS dependency hashes;


exact execution schedule.


Emit:
V30_AMFC_MECHANISM_LOCKED
Gate 1 — fixture validity before detector invocation
The fixture builder, serializer, renderer, and expected-result evaluator are separately versioned from the mechanism.
Before invoking AMFC:


check all coordinates and frames are finite;


verify analytic surface residuals to 10−9s;


verify expected component counts;


verify positive tail widths, angles, and axial spans;


verify the nonmonotone fixture’s signed turn sequence;


reject zero-area triangles and unintended self-intersections;


serialize, reload, and compare fixture hashes;


produce a 4096px native-resolution raw-geometry contact sheet for every fixture;


manually sign that each fixture depicts its declared physical construction.


Maintain an append-only AMFC invocation counter.
A fixture-construction correction is allowed only when:


the invocation counter is zero;


the mechanism and all mechanism hashes remain unchanged;


the diff is confined to the separately versioned fixture/harness tree;


the expected causal distinction and required result are unchanged;


the harness version is incremented;


all analytic and raw-visual checks are repeated.


Emit the nonterminal token:
V30_AMFC_HARNESS_INVALID_PREINVOKE
and correct the harness.
Once AMFC has run on any fixture, no harness correction is authorized inside V30. A post-invocation fixture defect is terminal, because it is no longer possible to separate changed test construction from a mechanism rerun.
Immediate execution
After Gate 1 passes:


Execute SF01–SF09 in fixed lexical order.


Execute the complete preregistered metamorphic set.


Execute three same-input replays.


Execute the resource stress five times.


Produce and inspect all visual artifacts.


Only after all pass, select and consume the prospective real scan.


No source-free result may be used to alter a constant, graph rule, fixture, pass condition, or reason-code precedence.

7. Metamorphic, determinism, resource, and 4096px visual gates
Metamorphic schedule
For every SF01–SF09 fixture, run:


candidate-axis reversal;


parent-plane swap;


point/vertex permutation;


face permutation and triangle winding reversal;


one fixed rigid transform: Euler rotations 37∘,−23∘,61∘ plus translation;


uniform scales 0.5 and 2.0;


one composite variant containing parent swap, axis reversal, permutation, the rigid transform, and scale 2.0.


Required:


identical rescue/no-rescue classification;


identical reason code;


identical station and path counts for axis reversal, parent swap, and permutation;


transformed rescued lines return to the base line within 10−6s positional error and 10−7 radians angular error after inverse transformation;


quantized turn diagnostics differ by no more than one quantization unit under rigid transform or scale;


no legacy-order or NMS difference.


Determinism


Three identical runs with one worker: byte-identical JSON and summary cache.


Three identical runs with maximum workers: byte-identical JSON and summary cache.


One-worker and maximum-worker outputs: byte-identical.


Rendered PNGs use fixed metadata-free encoding and must also be byte-identical on exact replay.


Resources
On SF10_RESOURCE_STRESS:


V30 incremental P95 over five runs: ≤180 seconds.


Peak process-tree RSS: <8 GiB.


Summary cache: ≤132 KiB.


No station node or edge cap reached.


No failed job or solver error.


On the prospective real scan:


frozen V16 plus V30 wall time: ≤300 seconds;


peak process-tree RSS: <16 GiB;


successful completion without truncation.


Native-resolution visuals
Every base fixture, every composite metamorphic fixture, and the stress fixture receives a 4096×4096 four-view artifact containing:


every source point or source triangle, without LOD or decimation;


candidate line and axial stations;


the two tail seed sets;


retained graph edges in the candidate corridor;


every accepted path;


final pass or rejection reason.


The inspector must verify on positives that paths remain on one physical surface, cross the curved transition, and do not jump through space or another layer.
On negatives, the inspector must verify that the stated rejection reflects the visible physical failure—not an accidental graph-construction bug.
Every artifact is listed in VISUAL_MANIFEST.json and requires a signed decision. No fund or retire token may be issued while a relevant visual row is unsigned. A numerical pass with a visual shortcut or physically wrong path is a failure.

8. Exactly one prospective unseen real-part test
Selection without replacement
After all source-free gates pass:


Construct an eligibility manifest of newly received scans that:


have never been opened by Sherman research or manual bend inspection;


have never been run through V16 or any later architecture;


are not byte-identical to any prior scan;


have passed only the neutral supported-format ingestion check.




The manifest may contain only:


neutral arrival sequence;


raw content SHA-256;


format;


neutral ingestion-success bit.




Do not expose filenames, part identifiers, customer, family, or good/bad/deviation status.


Select the row minimizing:
SHA256("V30_AMFC_PROSPECTIVE_1"∥raw-content-SHA256).


Assign the neutral ID V30-P001.


Remove every other queue entry from the execution environment.


There is no replacement for:


an uninformative part;


a V16-perfect part;


a zero-bend part;


a bad or incomplete scan;


a resource failure;


an AMFC no-activation;


a visual ambiguity;


an unfavorable result.


Blinded comparison
Run and seal:


frozen V16 output;


V30 output;


all timing and resource records;


output hashes.


Only after sealing, create a 4096px native multiview render without predictions. A predesignated inspector, blinded to both outputs, filename, identity, and good/bad status, records the visible bend-line reference.
Seal that reference before showing any prediction overlay.
Then:


compare both outputs with the already frozen line-instance matcher and tolerances;


do not introduce a new matching tolerance;


render the full V16/V30 comparison;


render a native-resolution close-up for every V30-only surviving line;


manually verify every V30 path certificate and every V30-only line.


The manual reference and comparison results remain outside the detector environment and cannot be fed back into V30.

9. Exact fund/retire criteria and tokens
Source-free continuation criteria
Proceed to the real part only if:


SF01–SF03 each produce exactly one rescue;


SF04–SF09 and SF10 produce zero rescues;


every required diagnostic reason matches;


every metamorphic gate passes;


all exact replays pass;


all resource gates pass;


every visual artifact is signed as physically correct.


Any failure is terminal. There is no threshold change, fixture substitution after invocation, V30b, or mechanism rerun.
Prospective funding criterion
Issue the fund token only if all of the following are true:


Every frozen V16 post-NMS line is byte-identical and remains present in V30.


At least one V30-only candidate survives the sole NMS.


Every V30-only surviving line matches a blinded manual bend under the frozen evaluator.


At least one V30-only line corrects a V16 false negative.


No V30-only line is a false positive or duplicate.


Therefore:
TP30​>TP16​,FP30​=FP16​,FN30​<FN16​,F130​>F116​.


The full-scan and every candidate-level visual are signed as genuine formed bends with on-surface paths and no shortcut.


Runtime and resource contracts pass.


The atomic unit of improvement is one correctly recovered line instance with zero regression. That is the minimum attributable prospective gain.
Fund token:
V30_AMFC_PROSPECTIVE_CAUSAL_GAIN_FUND_LOCKED_VALIDATION
This funds only a later locked multi-part validation. It does not open runtime/main or production.
Retirement conditions
ConditionTerminal tokenFrozen architecture cannot satisfy the static isolation, boundedness, or one-NMS contractV30_AMFC_STATIC_CONTRACT_RETIREAny positive fails, any hard negative rescues, or required reason is wrongV30_AMFC_SOURCE_FREE_MECHANISM_RETIREA fixture defect is discovered after AMFC invocationV30_AMFC_POSTINVOKE_HARNESS_INVALID_RETIREAny metamorphic or replay requirement failsV30_AMFC_METAMORPHIC_OR_REPLAY_RETIREAny node, edge, candidate, memory, time, cache, or job-success contract failsV30_AMFC_RESOURCE_CONTRACT_RETIREMetrics pass but native visual inspection reveals a shortcut, wrong surface, alias, or physically false certificateV30_AMFC_VISUAL_CAUSALITY_RETIREProspective output adds no surviving line, the chosen part is uninformative, or V16 is already perfectV30_AMFC_PROSPECTIVE_NO_ATTRIBUTABLE_GAIN_RETIREAny prospective addition is false/duplicate, any legacy line changes, or F1 does not strictly improveV30_AMFC_PROSPECTIVE_REGRESSION_RETIRE
After the prospective scan is consumed, no rerun, replacement scan, adjusted predicate, different ordering, changed candidate cap, or modified fixture is authorized.

10. Capability opening
No capability opening is scientifically necessary now.
AMFC can be implemented and falsified using the raw meshes/point clouds and frozen V16 pre-NMS frames already available. V29’s missing calibrated-range inputs are irrelevant to this mechanism.
There remains a real identifiability boundary: a perfectly smooth joined seam and a formed bend can, in principle, produce identical observed geometry. AMFC does not claim to infer manufacturing history from identical surfaces. Should such observationally identical but oppositely labeled cases become the demonstrated blocking class, the smallest justified opening would be one co-registered material-sensitive channel—such as scanner intensity or native texture—not V29 range metadata and not part identity or expected counts. That opening is not warranted before AMFC is executed.
