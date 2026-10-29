1. RULING: PERSISTENT_TUBE_COMPONENT_PARITY_V1 (PTCP_V1)
Authorize exactly one source-only intervention.
PTCP_V1 replaces PSRO’s brittle question—

“Does a fixed ray at this station reciprocally hit the expected opposite owner?”

—with a topological one—

“As small geometric gaps are closed, is there a persistent interval in which the candidate tube contains exactly two distinct, line-long surface sheets, each continuing from parent arm A to parent arm B, before those two sheets merge at a resolved thickness scale?”

It does not complete PSRO owners, reuse PSRO rays, soften its station masks, consume V17/V18 fields, or infer an expected candidate count. It adds only a candidate-local H0​ offset filtration over native mesh patches.

2. RESEARCH BASIS
Alpha shapes and related offset constructions provide a scale-indexed description of sampled geometry: as the offset radius grows, disconnected pieces join and the topology changes at discrete critical scales. ISTA Publications+1
Persistent homology formalizes the distinction between short-lived topological events and features that survive over a nontrivial interval; persistence-diagram stability gives the relevant justification for using the lifetime of a connectivity state rather than one exact surface hit. Springer Link+1
Local-homology and stratification-learning research applies multiscale topology to noisy sampled spaces specifically to distinguish different local structures. Related work shows that local-homology persistence can be approximated using simpler proximity complexes. SciCom Institute+1
Established machinery: offset filtrations, persistent connected components, local topological signatures, nerves/proximity complexes, and deterministic union-find evaluation.
Sherman-specific hypothesis: within a candidate-aligned tube, an independent interior sheet-metal fold produces two persistent A-to-B spanning surface components—one per skin. Mesh fragmentation controls their birth scale; physical sheet thickness controls their merge/death scale. A genuine fold should therefore have a substantial interval between those two scales.
Published work does not establish this as a bend detector. The two-spanning-component predicate, its arm-port construction, and its interpretation as material ownership are the falsifiable Sherman contribution.

3. CAUSAL FEATURE
For candidate c, let:


gb​ be the smallest geometric gap at which fragmented native patches assemble into exactly two admissible A-to-B surface components.


gd​ be the first larger gap at which those two components directly merge.


Pc​=1−gb​/gd​ be their normalized persistence.


The causal interpretation is:


Fragmented or locally disconnected fold: missing triangles raise gb​, but the two skins remain separate until approximately gd​, so the state survives when gb​≪gd​.


Face-orientation corruption: irrelevant to component construction; all normal predicates use abs(dot(...)).


Silhouette or single observed skin: at most one A-to-B spanning component.


Opening rim or thickness wall: the two skins are already joined by the wall, so they form one component at zero or near-zero gap; there is no two-component persistence interval.


Cycle-closing boundary: produces an early skin-to-skin connection or an additional substantial A-to-B component.


Ordinary nearby panel pairing: normally gives components that touch only one candidate arm, components too far apart to die within the thin-sheet range, or more than two substantial A-to-B components when both solids’ skins are visible.


Miter or seam evidence: disconnected panel halves need a seam-closing gap comparable to the inter-skin gap, making gb​/gd​ too large; a bead, cap, or welded closure instead connects or branches the components too early.


The architecture does not claim to distinguish two geometries that are genuinely identical in the observed tube—for example, a perfectly ground full-penetration seam whose two visible surfaces reproduce exactly the same two nested folded sheets. That is the explicit observational alias in Section 6.

4. EXACT SPECIFICATION
4.1 Eligible pool and insertion point
PTCP_V1 receives every frozen V11 candidate satisfying all of the following:


provenance is exactly scan_mesh_adjacency;


the frozen V11 crop is finite and renderable;


it has two distinct near-side transition parents A and B;


it was rejected by the frozen V11/V16 selector;


it is not already an emitted frozen V16 marker.


It reads no V17 or V18 cache, mask, owner, ray, mate, or decision field.
There is no scan-level candidate cap. Every eligible candidate is processed serially. A per-candidate structural-cap hit is an architecture execution failure, not a semantic candidate rejection.
PTCP survivors are appended only after frozen V16 has completed. All frozen incumbents retain their original order and bytes. The combined stream then uses the unchanged V11 line duplicate/NMS predicate and unchanged render-admissibility test.
4.2 Candidate frame and bounded tube
Reuse the frozen V11 native mesh spacing s, candidate midpoint o, unit line direction d, and crop length L.
For each parent X∈{A,B}, let nX​ be its signless plane normal and μX​ its source-only support centroid. Define the transverse arm direction:
rX​=normalize(nX​×d)
and choose its sign so that:
rX​⋅(μX​−o)>0.
No opposite owner is constructed.
Freeze:
R=min(0.30L, 80s).
Hard reject the candidate when:
R<20s.
Use only the central 80% of the frozen crop:
−0.40L≤(x−o)⋅d≤0.40L.
The tube is a convex 16-sided prism:


two axial cap half-spaces;


16 transverse half-spaces at angles kπ/8;


circumradius R.


Native triangles are clipped against this prism by deterministic half-space polygon clipping. Each resulting convex polygon is fan-triangulated from its lexicographically smallest candidate-frame vertex. No clipping face is introduced; the prism boundary never becomes artificial material.
4.3 Initial components
Each clipped fan triangle stores only transiently:


three candidate-frame vertices;


area and centroid;


original native triangle key;


signless geometric normal;


axial bin;


A-port/B-port membership.


At gap zero, fan triangles are connected when they:


share clipped geometry inherited from the same native triangle;


share a native vertex or edge whose clipped portions touch inside the prism; or


have quantized geometric distance zero.


Initial component identity is based on a geometry-derived 128-bit patch key, not input face order. Thus face and vertex permutations cannot change component ordering.
4.4 Arm ports
Divide the axial interval into eight equal bins.
A fan triangle contributes to port X when its centroid satisfies:
5R/8≤r≤R
and
∥y∥rX​⋅y​≥cos(π/8),y=x−o−d(d⋅(x−o)).
Within each component and port, a bin is occupied only when accumulated triangle area in that bin is at least:
2s2.
For an 8-bit mask M, define span(M) as the number of bins from its first set bit through its last set bit, inclusive.
A component is A-to-B spanning exactly when:
popcount(MA​)≥4,span(MA​)≥6,
and
popcount(MB​)≥4,span(MB​)≥6.
This requires line-long continuation but does not require an exact endpoint or center hit.
4.5 Offset-connectivity filtration
Freeze:
Δ=s/1024,gmax​=R/4.
Build a transient spatial hash over the initial components and generate every component pair whose AABBs can be within gmax​. Compute the exact minimum triangle-to-triangle distance dij​, then conservatively quantize:
qij​=⌈Δnextafter(dij​,+∞)​⌉.
Discard pairs with qij​Δ>gmax​.
Sort proximity edges by:


qij​;


smaller component key;


larger component key.


All edges with equal qij​ are unioned as one batch. Port masks, areas, axial ranges, and canonical component keys are aggregated in ascending component-key order with compensated floating-point summation.
Only persistent H0​ is computed. No higher-dimensional complex and no reconstructed mesh are produced.
4.6 Unique two-sheet interval
After every distance batch, identify roots satisfying the A-to-B spanning predicate.
A maximal filtration interval is provisionally admissible when it contains exactly two such roots, S0​ and S1​.
Let:


gb​ be the first gap value of that maximal interval;


gd​ be the first subsequent batch at which S0​ and S1​ become the same root.


The candidate rejects unless all of the following hold:


Unique topology
Exactly one maximal two-root interval satisfies all subsequent gates. Zero or multiple qualifying intervals reject.


Direct death
Immediately before the gd​ batch, S0​ and S1​ are distinct, and that batch contains at least one edge directly connecting those two pre-batch roots. A merge mediated only through a third root rejects.


Resolved merge scale
3s≤gd​≤R/5.


Fragmentation-to-thickness separation
gb​≤gd​/4.


Absolute interval width
gd​−gb​≥2s.


Equivalently, normalized persistence must satisfy:
Pc​=1−gb​/gd​≥0.75.


No substantial third sheet
At gb​, for every root C∈/{S0​,S1​}, at least one of these must hold:
Aport​(C)<0.10Aport,total​
or
Bport​(C)<0.10Bport,total​.
A third root carrying at least 10% of both arm-port areas rejects.


Distributed inter-sheet separation
Consider all stored proximity edges between descendants of S0​ and S1​ with:
dij​≤1.25gd​.
For closest points x0​,x1​, triangle normals n0​,n1​, and
v=normalize(x1​−x0​),
an edge is an offset-valid inter-sheet edge exactly when:
abs(n0​⋅n1​)≥cos(25∘)
and
min(abs(n0​⋅v),abs(n1​⋅v))≥cos(35∘).
Offset-valid edges must occupy:
popcount(Maxial​)≥5,
popcount(MAcross​)≥3,
popcount(MBcross​)≥3.
These are coverage masks over proximity-edge locations, not ray stations.


A candidate survives only if every gate passes. There is no continuous score threshold and no fallback path.
4.7 Ranking and duplicate behavior
For surviving candidates, define the deterministic rank tuple:


descending Pc​;


descending popcount(M_axial);


descending popcount(M_A_cross) + popcount(M_B_cross);


ascending gd​/R;


frozen V11 source order;


ascending canonical candidate ID.


Duplicate processing is:


all frozen V16 incumbents first, unchanged;


PTCP candidates in the order above;


exact frozen V11 geometric duplicate/NMS predicate;


incumbent always wins;


among PTCP duplicates, the first ranked candidate wins;


suppressed candidates retain a sidecar record with DUPLICATE_NMS, but emit no marker.


The emitted line, crop, direction, endpoints, and render marker are exactly the frozen V11 candidate geometry.
4.8 Fixed sidecar schema
Use a new 64-byte scan header and one fixed 224-byte record per eligible candidate.
GroupExact contentsBytesIdentitycandidate ID, 128-bit canonical key, frozen source rank, status/failure bits32Counts and topologynative triangle count, clipped fan count, initial component count, proximity-edge count, birth/death quantized values, two root IDs32Scalar featuress,R,L,gb​,gd​,Pc​,gd​/R, arm angle, offset-valid fraction, port-exclusivity fraction40Maskstwo A masks, two B masks, axial cross mask, A/B cross masks, offset-valid mask16Component summariescomponent areas, A/B port areas, axial minima/maxima, death-pair patch IDs48Determinism hashes128-bit topology hash and 128-bit construction/code-hash prefix32Reservedzero-filled24
Thus a three-candidate U02 or C017 cache is exactly:
64+3(224)=736 bytes.
No points, vertices, triangle coordinates, face indices, adjacency lists, component memberships, or proximity edges are serialized.
4.9 Structural and resource bounds
Process one candidate at a time. Freeze these per-candidate maxima:


clipped fan triangles: 65,536;


initial components: 16,384;


proximity edges: 262,144;


PTCP-owned transient allocation: 256 MiB.


A limit breach is PTCP_STRUCTURAL_CAP_FAILURE; it fails the source gate or tranche. It must not silently reject the candidate and continue.
The native mesh remains read-only. PTCP may build one transient candidate-local spatial hash/BVH but no second persistent mesh copy.
Topology construction is scalar deterministic CPU code. Equal-distance unions are batched; no unordered parallel reduction is permitted.

5. FALSIFICATION GATES
5.1 Source-free constructions
All tests run once with the frozen constants above.
Mandatory positives


Fragmented 90-degree double skin


sampling scale s=1;


axial length 320s;


arm reach 96s;


inner/outer bend radii 12s and 20s;


thickness 8s;


deterministic disconnected deletion patches of diameter at most 1.5s, totalling 20% of each skin;


raw ownership split into at least 11 fragments per skin;


50% hash-selected face-orientation reversal.


Required: exactly one retained candidate.


Fragmented 60-degree cropped double skin


radii 21s and 27s;


thickness 6s;


20% axial crop;


the same deletion, ownership-fragmentation, and orientation corruption.


Required: exactly one retained candidate.


For both positives:


gb​≤gd​/4;


both arm cross masks pass;


no source-free duplicate;


emitted geometry equals the frozen synthetic candidate.


Mandatory negatives
Each must emit zero:


single observed L-skin;


terminal skin-to-skin thickness wall;


opening rim with a complete wall;


terminal/opening wall with deterministic sub-gd​/4 holes;


two independent flat nearby panels;


two nested closed L-shells, exposing four through-going skins;


lap or T seam with a weld-bead bridge;


cycle-closing box corner.


Metamorphic requirements


three rigid transforms;


uniform scales 0.5 and 2.0;


complete face-orientation reversal;


vertex and face permutation;


changed CPU worker scheduling.


Decision bits, masks, quantized birth/death values, canonical topology hash, and sidecar bytes must remain identical after inverse normalization. A record must remain exactly 224 bytes.
Any source-free failure terminates PTCP_V1 before U02.
5.2 U02–U05 closed-label gate
Run U02 first.
Required:


all three eligible lines reject;


SMA1, SMA2, and SMA6 emit no marker;


sidecar size exactly 736 bytes;


frozen V16 output, marker order, and replay hashes remain byte-identical;


no structural-cap or nonfinite failure.


U03–U05 must remain on the frozen route unless they independently contain eligible V11-rejected adjacency candidates. Empty PTCP sidecars are exactly 64 bytes. U02–U05 must remain 15 TP / 0 FP / 0 FN.
Any U02 append terminates PTCP_V1. C017 is not opened.
5.3 C017 source-only activation gate
Gold remains closed.
Process all three C017 candidates, with no candidate-count condition other than processing all three.
Required:


at least one candidate passes every PTCP predicate;


no structural-cap failure;


two deterministic replays produce byte-identical 736-byte sidecars;


every activation and rejection receives the native audit below.


Zero activations is a terminal PTCP architecture stop. No constant, port width, persistence ratio, area threshold, or coverage mask may be changed.
5.4 Mandatory pre-gold native audit
For every U02 and C017 candidate, produce:


native_context_4096.png
Four native-resolution views showing the frozen line, the R-prism, both arm ports, and surrounding context out to 4R.


tube_components_zero_4096.png
Exact zero-gap clipped components.


tube_components_birth_4096.png
Components immediately after the gb​ batch, with S0​, S1​, and all substantial extra roots distinguishable.


tube_components_predeath_4096.png
State immediately before gd​, including direct death edges and the eight axial coverage bins.


persistent_component_barcode.png
A discrete H0 timeline marking the two-sheet interval, gb​, gd​, gd​/4, and R/5.


transverse_native_slices_4096.png
Eight transverse slices, one per axial bin, using native triangles rather than reconstructed surfaces.


The audit must be completed and frozen before any label join. No gold, expected count, or target identity may appear in these artifacts.
5.5 One-shot 35-scan gate
Only after C017 activation and audit, run the locked 35-scan development set once.
PTCP survives only if all are true:


frozen V16 markers are byte-identical;


incremental TP count is at least 4;


recovered TPs cover at least two part families;


at least one recovered TP belongs to a family whose frozen V16 recall was below 0.80;


incremental precision is at least 0.90;


incremental FP count is at most 1;


exact-count cases increase by at least 2;


U02–U05 remain 15 TP / 0 FP / 0 FN;


every new marker is render-admissible;


overall render admissibility remains at least 99%;


no architecture cap, timeout, nonfinite value, or incomplete job occurs.


Failure is terminal. The gold result may not alter PTCP constants or construction.
5.6 Fresh family-held-out gate
On the preregistered fresh held-out tranche:


at least two new TPs across at least two scans;


incremental precision at least 0.90;


exact-count performance does not decline and improves on at least one scan;


100% of PTCP markers render admissibly;


no source inference failure;


no family identity or held-out outcome enters the predicate.


A source-only tranche with no activations does not count as positive evidence; the architecture stops for lack of demonstrated generalization.
5.7 Determinism and operational gates
Across source-free, U02, C017, development, and held-out runs:


process-tree peak RSS below 16 GiB;


total pipeline P95 no greater than 300 seconds;


all jobs exit successfully;


no per-candidate structural cap is reached;


two independent source-inference replays are byte-identical;


cache size is always exactly 64 + 224 × eligible_candidate_count.



6. EXPECTED FAILURE MODE
The most likely false positive is a nested independent L-shaped panel or perfectly flush fused seam whose only observed surfaces inside the tube are exactly two thin, parallel A-to-B sheets. If its other skins, closure walls, weld bead, or part separation lie outside the tube or are absent from the scan, its local offset topology is observationally identical to a folded sheet:


two A-to-B roots;


small gb​;


resolved thin gd​;


distributed normal-offset support.


The exposing artifact is the combined native_context_4096.png plus transverse_native_slices_4096.png, extended to 4R: it should reveal an additional skin, independent continuation, seam bead, closure, or separate solid outside the strict predicate tube.
If the wider native scan contains no distinguishing geometry, this is not a threshold failure. It is a scan-only identifiability limit. PTCP must then stop rather than add another completion radius, port tolerance, or ownership score. The smallest scientifically necessary future capability would be acquisition-derived free-space/view-ray evidence establishing whether the missing closure surface was observed absent or merely unobserved.
Authorize PERSISTENT_TUBE_COMPONENT_PARITY_V1 exactly as frozen above.
