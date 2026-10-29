RULING
Authorize exactly one isolated, one-shot source-free successor:
V32_AXIAL_VORONOI_HARMONIC_FOLD_MEASURE_V1
Short name: AVHFM_V1
V32 is a research-only, rescue-only V16 sidecar. It replaces V31’s eleven independent point-sampled station decisions and 11-bit mask with:


one candidate-wide harmonic-current solve over the same observed-surface graph; and


a phase-free axial length measure obtained by projecting current-supported local fold certificates onto source-derived axial Voronoi cells.


V32 does not rerun V31, shift V31’s station lattice, vote over several phases, lower 8/11-type coverage requirements, change a nuisance threshold, refit candidates, or open any real-scan, gold, benchmark, runtime/main, or production lane.
The authorized action is implementation plus one frozen source-free execution in the exact order below.

1. DIAGNOSIS
Most likely classification: implementation-contract defect
More precisely, this is a defect in V31’s discrete axial observation contract, not evidence of an ordinary coding error.
The evidence is unusually diagnostic:


The harmonic-current/normal-turn mechanism produced a valid certificate at 7 of 11 locations on the first clean positive.


The four failures were non-geometric coverage failures: three ANGULAR_COVERAGE and one POTENTIAL_COVERAGE.


Manual 4096px review found the same continuous fold, broad tails, and opposite skin at all eleven locations.


There was no visible material gap, boundary, seam, thickness interruption, or fold-shape change at the failed locations.


The 2.4s station advance walks through the phase of the 0.5s source lattice: 2.4/0.5=4.8. The resulting five-phase cycle is fully capable of producing noncontiguous membership and angular-bin changes without any change in the continuous surface.


This is not most likely a fixture defect. The physical fixture had already passed analytic validity and native-resolution review, and the post-run views confirmed that the supposedly deficient stations contained the intended material.
It is also not yet a failure of harmonic fold flux as a causal primitive. A mechanism that is physically wrong would more naturally fail consistently, produce the wrong current route, reverse or diffuse the normal turn, or activate a nuisance. Instead, the primitive repeatedly succeeded and the mask aggregation withheld the result according to where the continuous fold landed relative to a discrete station/sampling phase.
That distinction does not reopen V31. V31 remains terminal. Its architecture includes the phase-sensitive station contract, and that architecture failed exactly as executed.
My confidence in the phase diagnosis is high but not absolute because V31 stopped before its resampling-phase metamorphics. V32 therefore puts phase variants immediately after the first positive; the hypothesis is killed within the first five physical invocations if wrong.

2. RESEARCH BASIS
Solving a graph Dirichlet problem and interpreting conductance-weighted potential differences as electrical current is established mathematical machinery, not Sherman novelty. Harmonic functions, electrical networks, and graph Laplacians are canonically linked; electrical flows are computed from Laplacian linear systems. arXiv+1
Using Voronoi or finite-volume cells to integrate quantities on sampled surfaces is also established discrete differential-geometry machinery. Meyer et al. derive discrete surface operators using mixed finite-element/finite-volume constructions and averaging Voronoi cells. Springer Link+1
The general reason for replacing a point decision with a neighborhood integral is similarly well supported: integral geometric quantities are often more stable under sampling irregularity than pointwise differential estimates. the University of Bath's research portal+1 A related signal-processing precedent is translation-invariant estimation, where dependence on one decimated sampling phase is removed rather than repaired by choosing a favorable shift. Springer Link+1
Sherman-specific and unproven is the following narrow claim:

A genuine scan-visible sheet-metal fold produces a tail-to-tail harmonic current whose current-weighted normal rotation satisfies the frozen V31 local nuisance predicates over a large, nearly continuous axial length; free edges, rims, returns, seams, local bridges, beads, cycles, and nearby panels do not.

V32 is the smallest executable test of that claim because it removes only the exogenous eleven-station phase variable. It does not introduce learning, model selection, part identity, a new candidate generator, or another geometric fitting family.

3. WHY THIS IS NOT V31 TUNING
The architectural replacement is exact:
11 independent station solves+11-bit count/run gate​
is retired and replaced by
1 candidate-wide solve+source-derived axial length measure​.
There are:


no new station offsets;


no best-phase selection;


no majority over rerun V31 masks;


no changed V31 local angular, potential, current, tail, layer, boundary, or normal-turn thresholds;


no acceptance of the observed 7/11 mask;


no threshold derived from SF01’s outcome.


V32 has no station mask field at all. The unit of longitudinal evidence is physical normalized length, not the number of favorable lattice samples.

4. COMPLETE EXECUTABLE LOCK
4.1 Frozen parent boundary
The following components are inherited byte-for-byte from the frozen V31 lock and bound into the V32 mechanism hash:


frozen V16 source sampler and maximum 200,000-source-sample representation;


source graph construction and positive base conductances;


source-normal construction and orientation;


V16 candidate records, candidate ordinals, anchors, axes, finite extents, and original rank values;


V31 candidate-frame construction;


V31 tail-electrode predicates;


every station-independent V31 nuisance predicate;


V31 potential-bin, angular-bin, current-floor, current-balance, tail-coverage, opposite-skin, layer-count, boundary/rim, and current-weighted normal-turn constants;


legacy V16 records and their ordering;


ordinary NMS implementation and NMS constants.


The V32 lock must contain the SHA-256 of the canonical frozen V31 parent lock and the hashes of the exact imported functions. A mismatch fails before invocation with V32_PARENT_CONTRACT_MISMATCH.
V32 must not call the V31 runner or produce a V31 result. It may call only parent pure primitives whose hashes are embedded in the V32 lock.

4.2 Candidate inputs
Each V32 candidate receives only:
C=(ordinal, c, a, e0​, e1​, F0​, F1​, r16​)
where:


c∈R3: frozen candidate anchor;


a∈R3: frozen candidate axis;


e0​,e1​: frozen finite endpoints;


F0​,F1​: the two source-derived V31 tail reference frames and their electrode-membership predicates;


r16​: frozen original V16 rank tuple;


ordinal: original candidate-pool ordinal.


The source input is:
S=(X,N,G,s)
with:


X={xi​}i=0N−1​, N≤200,000;


frozen oriented unit normals ni​;


frozen undirected observed-surface graph G=(V,E,c0);


frozen local scale s>0.


No candidate may receive a label, expected count, fixture ID, filename, family name, CAD value, gold value, benchmark statistic, or visual adjudication.
V32 processes only candidates rejected by the frozen V16 selector. Legacy-kept candidates do not enter V32.
Hard input caps
QuantityExact capSource samples200,000V32 candidate records128Candidate-local vertices32,768Candidate-local undirected edges262,144Axial event coordinates65,536
A cap breach is a run-level terminal resource failure. It is not an ordinary negative classification and must not cause truncation or top-K selection.

4.3 Canonical candidate frame
Normalize a in float64. Orient it so that the first coordinate whose absolute value is at least 2−30 is positive.
Project the two frozen tail reference normals onto a⊥:
nˉh​=∥nh​−(nh​⋅a)a∥nh​−(nh​⋅a)a​,h∈{0,1}.
Swap F0​,F1​ when necessary so that
a⋅(nˉ0​×nˉ1​)>0.
Define the candidate turn:
Θ=atan2(a⋅(nˉ0​×nˉ1​),nˉ0​⋅nˉ1​).
Require:
6π​≤Θ≤65π​.
Failure is V32_FRAME_INVALID.
No candidate line or tail frame is refitted after this operation.

4.4 Candidate-wide graph
For each source vertex define:
zi​=(xi​−c)⋅a
and
ρi​=∥xi​−c−zi​a∥.
Two axial domains are fixed:


audited interval


A=[−12s,+12s],∣A∣=24s;


solve interval with boundary padding


A+=[−13.2s,+13.2s].
The candidate-wide graph GC​ contains the inherited source-graph vertices satisfying:
zi​∈A+
and the exact frozen V31 transverse-corridor predicate. If the parent predicate requires an explicit numerical outer limit, the limit remains the V31 value; it may not be changed for V32.
Every inherited source edge whose endpoints survive is retained with its unchanged V31 positive conductance cij0​. V32 does not add Euclidean shortcuts, k-nearest-neighbor edges, cross-component edges, synthetic faces, paths, or station links.
The graph is ordered by ascending canonical source-vertex ID. Edges are stored as:
(min(i,j),max(i,j))
and lexicographically sorted.

4.5 Global electrodes
Apply the exact frozen V31 tail predicates over the whole padded interval rather than inside eleven station slices:
B0​={i∈VC​:F0​(i)=1},B1​={i∈VC​:F1​(i)=1}.
Require, for each Bh​:


at least 16 vertices;


nonzero support in all four six-s quarters of A;


source-derived axial Voronoi coverage of at least 15/16 of A;


no uncovered axial gap longer than 1.2s.


The electrode Voronoi construction is the same one defined in §4.9, using electrode-vertex zi​ values and the same 0.75s half-cell cap.
Failures are, in order:


V32_ELECTRODE_0_COVERAGE;


V32_ELECTRODE_1_COVERAGE.


The graph must contain exactly one connected component that touches both electrodes and contains at least 15/16 of each electrode’s vertices. Otherwise fail V32_ELECTRODE_COMPONENT.
This prevents a candidate-wide field from silently combining separate folds or unrelated layers.

4.6 Candidate-wide harmonic current
Set:
ui​=0,i∈B0​;ui​=1,i∈B1​.
For every remaining vertex solve:
j:(i,j)∈EC​∑​cij0​(ui​−uj​)=0.
Equivalently:
LII​uI​=−LIB​uB​.
Exact solver contract


float64 only;


CSR rows and columns in ascending canonical order;


no fast-math;


no unordered sparse accumulation;


Jacobi-preconditioned conjugate gradient;


initial interior potential ui​=0.5;


maximum 4,096 iterations;


fixed pairwise binary-tree reductions;


stopping condition:


∥r∥∞​≤2−36max(1,∥b∥∞​);


after convergence, each ui​ is rounded half-to-even to Q40:


ui​=2−40roundeven​(240ui​).
Failure to converge is V32_SOLVER_NONCONVERGENCE.
For an edge e=(i,j), orient it from lower to higher potential and define:
Ie​=cij0​(uj​−ui​).
Edges with:
∣uj​−ui​∣<2−24
are assigned zero current.
No potential, current vector, graph vertex list, or edge list may be serialized into the cache.

4.7 Normal-turn coordinate
For each valid projected normal:
ni​=∥ni​−(ni​⋅a)a∥ni​−(ni​⋅a)a​.
Define angular progress:
qi​=ΘunwrapΘ​[atan2(a⋅(nˉ0​×ni​),nˉ0​⋅ni​)]​.
unwrap selects the unique representative nearest the closed interval [0,Θ]. Store qi​ in signed Q24.
For every current-carrying edge store transiently:
ze​=2zi​+zj​​,pe​=2ui​+uj​​,qe​=2qi​+qj​​,Δqe​=qj​−qi​.
These values are never retained after candidate adjudication.

4.8 Local current windows
Quantize normalized edge-midpoint coordinates using round-half-even:
Ze​=roundeven​(220sze​​).
The sorted unique Ze​ values inside A are the axial events. There is no uniform station lattice.
At each event Zk​, define a local window:
Wk​={e:∣ze​−zk​∣≤1.2s}.
All window sums are generated with sorted prefix arrays; a new harmonic solve is not performed per event.
Calculate axial current leakage:
Λk​=∑e∈Wk​​∣Ie​∣∑e∈Wk​​∣Ie​∣∥xj​−xi​∥∣(xj​−xi​)⋅a∣​​.
The window passes axial locality only when:
Λk​≤31​.
Then apply the exact V31 station-local adjudicator to Wk​, using the candidate-wide potentials and currents but retaining all frozen V31 thresholds and reason order. The extracted local adjudicator must be statically proven equivalent to V31 on the original eleven V31 station graphs when supplied the original V31 station potentials.
The local reason order is therefore unchanged:


current floor/balance;


tail/electrode coverage;


potential coverage;


angular coverage;


current-weighted normal-turn;


all remaining frozen V31 local nuisance predicates.


V32 adds AXIAL_LEAKAGE immediately after current floor and before potential coverage.
No local predicate may be omitted merely because the global field already connected the electrodes.

4.9 Axial Voronoi cells
Let the sorted event coordinates in physical units be:
z1​<z2​<⋯<zm​.
For event k, construct the clipped cell:
ℓk​=max[−12s,zk​−0.75s,2zk−1​+zk​​],
rk​=min[+12s,zk​+0.75s,2zk​+zk+1​​].
For k=1, omit the previous-midpoint term; for k=m, omit the next-midpoint term.
Cells are half-open [ℓk​,rk​), except that the final cell may include +12s. Empty cells are discarded.
The 0.75s cap is mandatory: one isolated current event may cover at most 1.5s. Sparse evidence therefore cannot claim a long empty axial region.
All boundaries are represented in signed Q21 units of s, allowing exact half-midpoints without floating-point comparisons.

4.10 Nested axial measures
For every local gate stage g, form the union Ug​ of cells whose event passed all predicates through g. Merge touching intervals exactly.
Define:
Mg​=μ(Ug​),
where μ is interval length.
For each nested local stage require:
Mg​≥1613​∣A∣=19.5s.
The first stage whose measure is below 19.5s supplies the candidate’s failure token. This prevents a mixture of different local failures from being hidden by one aggregate pass rate.
For the final local-PASS union U⋆​, require all of:
Total accepted length
μ(U⋆​)≥19.5s.
Maximum uncovered interior gap
Gmax​≤1.2s.
Longest continuous accepted run
Rmax​≥53​∣A∣=14.4s.
Endpoint persistence
Within each endpoint interval
[−12s,−9.6s]and[9.6s,12s],
accepted length must be at least:
1.8s=43​(2.4s).
These are physical-length conditions. There is no station count, station mask, or longest station run.

4.11 Candidate decision
The candidate is accepted if and only if:


all parent and input gates pass;


electrode and component gates pass;


the global solve converges;


every nested measure gate passes;


final total measure, maximum gap, longest run, and both endpoint gates pass.


The emitted bend geometry is exactly the original V16 candidate geometry. V32 may not move, extend, trim, rotate, or refit the line.
The sidecar score used only for ordering V32 rescues is the integer tuple:


descending final accepted Q21 length;


descending longest accepted Q21 run;


ascending maximum Q21 gap;


descending frozen V16 rank;


ascending candidate SHA-256.


No weighted scalar score is introduced.

5. INSERTION, DUPLICATES, AND NMS


Frozen V16 incumbents are emitted byte-identically in their existing order.


Accepted V32 rescues are appended in the order above.


Exactly one unchanged ordinary NMS is invoked.


A V32 rescue can never precede, veto, modify, or replace a V16 incumbent.


Two V32 aliases both exist pre-NMS; the first deterministic V32 order wins.


No second NMS, special duplicate merger, scan-level cap, expected-count logic, or candidate geometry averaging is permitted.


The duplicate fixture must produce:


two relevant pre-NMS lines;


exactly one final line;


the ordinary NMS suppression token on the later alias.



6. QUANTIZATION AND ORDER RULES
QuantityRepresentationCoordinates and parent geometryinherited float64Normalized axial coordinatessigned Q20Cell boundaries and lengthssigned Q21PotentialsQ40Normal progresssigned Q24Dimensionless decision metricsQ24HashingSHA-256RoundingIEEE round-half-to-evenCandidate orderfrozen ordinal, then specified integer tupleVertex ordercanonical inherited source IDEdge orderlexicographic endpoint pairInterval orderstart, end, reason code
Threshold comparisons operate on integers after quantization. Equality passes only where the specification uses ≤ or ≥ explicitly.
Current-mass normalization uses a fixed pairwise reduction. When normalized edge masses must sum to 240, residual integer units are assigned by descending fractional remainder, with lexicographic edge order as the tie-breaker.
Each candidate is single-threaded internally. Parallelism is only across candidates or fixture variants.
The worker count is:
W=min(∣sched_getaffinity(0)∣,candidate count).
OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, and equivalent nested-library thread counts are fixed to one. Results are gathered by frozen candidate ordinal. One-worker and W-worker decision payloads must be byte-identical.

7. FAILURE-TOKEN PRECEDENCE
The first applicable token is final:


V32_PARENT_CONTRACT_MISMATCH


V32_FORBIDDEN_INPUT


V32_INVALID_NUMERIC_INPUT


V32_SOURCE_CAP_BREACH


V32_CANDIDATE_CAP_BREACH


V32_FRAME_INVALID


V32_GRAPH_CAP_BREACH


V32_ELECTRODE_0_COVERAGE


V32_ELECTRODE_1_COVERAGE


V32_ELECTRODE_COMPONENT


V32_SOLVER_NONCONVERGENCE


V32_NO_CURRENT_EVENTS


V32_CURRENT_MEASURE


V32_AXIAL_LEAKAGE_MEASURE


V32_TAIL_COVERAGE_MEASURE


V32_POTENTIAL_COVERAGE_MEASURE


V32_ANGULAR_COVERAGE_MEASURE


V32_NORMAL_TURN_MEASURE


remaining inherited V31 local reasons in frozen V31 order, suffixed _MEASURE


V32_AXIAL_MAX_GAP


V32_AXIAL_LONGEST_RUN


V32_AXIAL_ENDPOINT_MEASURE


V32_ACCEPT


NMS suppression is recorded separately and is not an architecture failure.

8. CACHE LAYOUT AND CAP
Use a canonical little-endian binary cache.
Header: exactly 256 bytes
It contains:


8-byte magic: V32AVHF1;


schema version;


byte-order marker;


record size;


source-sample count;


candidate count;


source SHA-256;


candidate-pool SHA-256;


V31 parent-lock SHA-256;


V32 lock SHA-256;


result-payload SHA-256;


reserved bytes, all zero.


Candidate record: exactly 1,024 bytes
It contains only bounded summaries:


candidate SHA-256 and ordinal;


status and failure-token code;


accepted/NMS flags;


graph vertex, edge, event, and solver-iteration counts;


untouched anchor, axis, and finite endpoints;


s, Θ, total current;


measure after every local stage;


final measure, longest run, maximum gap, and endpoint measures;


worst accepted axial leakage;


counts of local reason classes;


at most 16 interval witnesses;


semantic record SHA-256;


reserved zero padding.


Witness retention is fixed:


eight longest accepted intervals;


eight longest rejected intervals;


descending length, then ascending start;


a truncation flag when additional intervals exist.


The cache must contain no source point, source index list, graph vertex list, graph edge list, potential vector, normal vector, face list, path, image, filename, fixture ID, or identity field.
With the 128-candidate cap:
256+128(1024)=131,328 bytes.
Any larger cache is a terminal cache-contract failure.

9. SOURCE-FREE FIXTURES
9.1 Reuse of the 21 V31 physical fixtures
All twenty-one V31 fixture source files, construction parameters, continuous-surface equations, expected physical roles, and existing raw visual hashes remain frozen.
They are not replaced merely because V31 failed SF01.
The required positive classes include:
ClassExpected V32 resultSF01_FOLD60_MESH_C2pre-NMS 1, final 1clean 90° point formpre-NMS 1, final 1clean 120° formpre-NMS 1, final 1frozen fragmented positive formsexactly their precommitted positive resultfrozen cropped-but-materially-supported positive formsexactly their precommitted positive result
The required adversarial classes remain:


silhouette/free edge;


scan boundary;


opening/rim;


thickness return;


disconnected seam or miter;


local bridge;


nonmonotone bead;


flat false crease;


corner/cycle;


nearby disconnected panel;


cylinder or continuous nonlocalized curvature;


insufficient tails;


short axial support;


cap/wall;


three-layer or extra-skin alias;


opposite-skin-only structure;


duplicate-axis aliases.


Every nuisance fixture must retain its frozen zero-final-line expectation, except the duplicate fixture, which must exercise ordinary NMS as described above.

9.2 Four mandatory phase variants
Add four fixtures derived from the exact continuous SF01 equations. Only the origin of the analytic axial sampling lattice changes:
IDAxial sampling offsetSF22_FOLD60_MESH_C2_PHASE_01+0.1sSF23_FOLD60_MESH_C2_PHASE_02+0.2sSF24_FOLD60_MESH_C2_PHASE_03+0.3sSF25_FOLD60_MESH_C2_PHASE_04+0.4s
The continuous fold, candidate line, normals, thickness, tails, opposite skin, cropping domain, and nuisance-free status do not change.
For SF01 and all four variants require:


pre-NMS 1;


final 1;


identical canonical candidate geometry;


direction error at most 0.10∘;


line distance at most 0.20s;


final axial accepted measure at least 15/16 of A, namely 22.5s;


maximum uncovered gap at most 0.5s;


no phase-related failure token.


Failure of any one of these five fixtures terminates V32 immediately with V32_PHASE_INVARIANCE_STOP.
This is the first direct test of the diagnosis.

9.3 Two new global-current adversaries
A candidate-wide solve creates one important new risk: current may route axially around a local absence and then cross somewhere else. Two additional fixtures are mandatory.
SF26_AXIAL_BYPASS_BRIDGE
The two broad tail electrodes are connected only by a route that runs at least 6s substantially parallel to the candidate axis before crossing. No local cross-section contains a complete fold.
Expected:


pre-NMS 0;


final 0;


token V32_AXIAL_LEAKAGE_MEASURE.


SF27_STAGGERED_HALF_FOLDS
One axial half contains only the early potential/angular portion of the transition and the other half contains only the late portion. Global pooling could exhibit all potential and angular bins, but no local window contains a complete tail-to-tail fold.
Expected:


pre-NMS 0;


final 0;


token V32_POTENTIAL_COVERAGE_MEASURE or the earlier inherited local-coverage token fixed in the pre-invocation expected manifest.


The expected token must be fixed before invocation by running the pure analytic truth generator, not by observing V32.
These two fixtures ensure that replacing stations does not weaken local nuisance rejection.

10. METAMORPHIC GATES
The following transformations are predeclared.
Exact decision-payload byte identity
Required for:


source-point permutation using three SHA-256-derived permutations;


mesh-face permutation;


edge-input permutation;


candidate-pool permutation followed by restoration to frozen ordinal;


candidate-axis reversal;


tail-parent swap;


global normal reversal with the corresponding frozen orientation normalization;


one worker versus maximum workers;


repeated execution on identical input.


Canonical inverse-transform byte identity
Apply, inverse-transform outputs, then canonicalize:


90∘ rotations about x,y,z;


120∘ rotation about (1,1,1);


translations (7s,−11s,13s) and (−17s,5s,3s);


uniform scales 1/2, 2, and 4.


The canonical semantic result must be byte-identical after inverse transformation. Envelope hashes that deliberately bind transformed source bytes are excluded from semantic comparison but must themselves be deterministic.
Resampling-phase metamorphic
The five SF01 sampling phases are not expected to be byte-identical because their source coordinates differ. They must nevertheless meet the identical activation, geometry, measure, gap, and error gates in §9.2.
Any metamorphic failure terminates V32. No canonicalization patch is permitted after invocation.

11. REPLAY, STRESS, AND RESOURCE GATES
Replay
After all physical and metamorphic fixtures pass:


execute the complete fixture suite again with one worker;


execute it again with maximum workers.


The canonical output trees, binary caches, decision payloads, failure-token vectors, and rendered diagnostic metadata must be byte-identical to their corresponding first executions.
Stress
Use the exact 200,000-source-sample, 128-candidate stress fixture. Its V32 expected manifest is fixed before invocation as:


128 candidate records;


no cap breach;


no solver failure;


no missing record;


pre-NMS 0;


final 0;


cache size exactly 131,328 bytes when all 128 candidate slots are present;


one-worker/max-worker byte identity.


Resource requirements:
GateLimitPeak process-tree RSS<16 GiBFull stress wall time≤300 sJob success100% for the source-free invocationNaN/Inf count0Unordered-reduction warnings0Serialized source indices/geometry0 bytes
All affinity-visible CPU cores are used through candidate-level parallelism when more than one candidate is available.

12. VISUAL CONTRACT
Pre-invocation
For each of the 27 physical fixtures, render and manually inspect at native 4096×4096 resolution:


raw_multiview.png
Source geometry only, four fixed views, with boundaries and disconnected components visible.


axial_unwrapped_input.png
Candidate axis horizontal; source support, two electrode sets, opposite skin, axial event density, and solve/audit interval boundaries visible. No detector decision is shown.


All 54 images receive individual SHA-256 hashes and signed visual dispositions before architecture_execution_count may change from zero.
For the stress fixture, render eight 4096px pages containing sixteen candidate regions per page.
Post-invocation
For every physical fixture, positive or negative, render:


harmonic_current_multiview.png
Electrode memberships, quantized potential, current-bearing edges, and detected detours.


axial_measure_audit.png
Event coordinates, Voronoi cells, local reason at every event, nested accepted measures, merged intervals, maximum gap, longest run, and endpoint gates.


final_overlay.png
Raw geometry, unchanged V16 candidate line, V32 pre-NMS rescue, ordinary-NMS outcome, and any incumbent conflict.


A numerical pass is terminally invalid when the native imagery shows current traveling through a rim, thickness face, unrelated panel, cap, seam, layer hop, or axial detour inconsistent with a local material fold.
A numerical failure on a visibly valid continuous fold is also terminal. Manual review may diagnose the wall but may not override or amend the output.

13. ONE-SHOT EXECUTION ORDER
After static checks, fixture hashing, and all pre-run visual signoffs:


SF01_FOLD60_MESH_C2;


the four SF01 phase variants in increasing offset;


remaining positive fixtures in ascending frozen fixture ID;


all inherited adversarial fixtures in ascending frozen fixture ID;


SF26_AXIAL_BYPASS_BRIDGE;


SF27_STAGGERED_HALF_FOLDS;


metamorphic bundles;


one-worker replay;


maximum-worker replay;


200,000/128 stress in one-worker and maximum-worker modes.


The first mismatch stops the process immediately. Unreached gates remain explicitly marked NOT_RUN_AFTER_TERMINAL_FAILURE.
The sole full-pass token is:
V32_AVHFM_SOURCE_FREE_PASS

14. TERMINAL STOP RULES
Before the first mechanism invocation, a demonstrable fixture-generator, expected-manifest, hash, or renderer defect may be corrected only while:
architecture_execution_count == 0
The affected artifacts and all dependent hashes must then be regenerated and reinspected.
After the first V32 mechanism invocation, any of the following retires V32 permanently:


any positive mismatch;


any phase-variant mismatch;


any adversarial activation;


any unexpected local reason;


any visual causal mismatch;


any metamorphic mismatch;


any replay mismatch;


any cap, resource, cache, or solver failure;


any forbidden-data access;


any alteration to the frozen parent contract;


any output other than exactly one ordinary NMS.


The terminal token is:
V32_AVHFM_TERMINAL_RETIRE
There is no V32b, threshold change, larger cell cap, lower axial measure, additional phase set, replacement positive, altered current solver, modified electrode predicate, or rerun rescue.
A source-free pass does not promote V32, open gold, or alter V16. It preserves V32 as a frozen research survivor eligible for a separately governed label-closed prospective scan action.

Bottom-line ruling
Proceed with V32_AVHFM_V1.
The evidence does not justify a harmonic-flux family STOP yet. It justifies removing the failed exogenous station lattice exactly once. The decisive experiment is front-loaded: the original SF01 plus its four 0.1s-spaced sampling phases. If those five do not all pass before any nuisance fixture is reached, retire V32 and treat the phase-free harmonic-flux hypothesis as unsupported rather than iterating again.
