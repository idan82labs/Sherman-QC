EXECUTE_ONE — V73_INTEGRAL_TRANSVERSE_COVARIANCE_GROWTH_VETO_V1
V72 remains terminal. Do not modify, import, wrap, rerun, or reinterpret CASL or Amendment A1. V73 is a new research-only module with a new namespace, implementation hash, fixture pack, runner, and precommit manifest. It consumes the same frozen source-derived candidate stream that V72 would have received and emits only an independent keep/veto mask.
1. Scientific mechanism
Use one conservative null-model test:

Veto a candidate only when candidate-centered transverse covariance is persistently compatible with one flat or smoothly curved surface across the complete scale range.

A localized physical bend should interrupt that smooth scale law:


A flat cross-section has rank-one transverse covariance.


A broad smooth roll has transverse nonplanarity that grows approximately quadratically with neighborhood radius.


A sharp bend has dimensionless transverse covariance that approaches a nonzero scale plateau.


A rounded bend grows at small scales but saturates once the neighborhood exceeds its fillet radius.


A return, hem, opposing sheet, partial bend, nearby second bend, occlusion, or complex junction normally produces a nonquadratic or nonpersistent trajectory.


V73 is therefore a high-confidence veto, not a positive bend recognizer. Any unsupported, ambiguous, non-smooth, localized, or cap-affected candidate is retained.
This is materially different from V72:


no annular secants;


no global radial trim;


no removed tail;


no tail-faithfulness assertion;


no candidate ownership or exclusive source assignment;


no cross-candidate comparison;


no degree, π, trigonometric threshold, or six-degree binary64 constant;


no graph or Q0​ construction.


2. Frozen inputs and insertion point
For one scan, V73 accepts:
P=(p0​,…,pN−1​),pi​∈R3,
and a frozen candidate sequence
C=(c0​,…,cM−1​).
Each candidate supplies only:
c=(o,a,e0​,e1​,s),
where:


o∈R3 is any point on the candidate line;


a∈R3∖{0} is its axis;


e0​,e1​∈R3 are its finite segment endpoints;


s>0 is the exact unchanged source-derived spacing scalar already carried or used by the frozen upstream candidate construction.


No new spacing estimator is permitted. If the upstream stream uses one scan-level spacing, every candidate receives that identical value.
V73 is inserted after the unchanged candidate geometry is frozen and before unchanged downstream result assembly. It may only change a candidate from retained to vetoed. It may not alter its anchor, axis, endpoints, score, ordering fields, or duplicate handling.
Candidate score, confidence, source family, filename, part identity, expected count, gold, labels, CAD, provenance, benchmark results, and existing downstream ownership information are neither read nor serialized.
3. Binary64 and canonical geometry contract
All arithmetic is IEEE-754 binary64, round-to-nearest ties-to-even, with:


one arithmetic thread;


floating-point contraction and FMA disabled;


subnormals preserved;


no extended-precision intermediates;


no BLAS, eigensolver, GPU, or nondeterministic parallel reduction;


all comparisons evaluated in the written order.


NaN and infinity are invalid inputs and cause an atomic scan abort before candidate work.
3.1 Canonical source order
Sort source records once by the IEEE total-order bit keys of:
(xi​,yi​,zi​).
Exact duplicate points are indistinguishable and need no ordinal tie-break. All candidate-local accumulation follows this canonical source order.
3.2 Canonical candidate segment
Normalize a.
Let r be the first coordinate in x,y,z order attaining
max(∣ax​∣,∣ay​∣,∣az​∣).
If ar​<0, replace a←−a.
Define the closest line point to the origin:
oc​=o−(o⋅a)a.
Project both supplied endpoints:
t0​=(e0​−oc​)⋅a,t1​=(e1​−oc​)⋅a,
and set
t−​=min(t0​,t1​),t+​=max(t0​,t1​).
Normalize longitudinal coordinates:
α=st−​​,β=st+​​,L=β−α.
3.3 Deterministic transverse frame
Choose q∈{x,y,z} minimizing ∣aq​∣, with ties resolved x<y<z. Let Eq​ be the corresponding Cartesian basis vector.
u=1−aq2​​Eq​−aq​a​,v=a×u.
No source-derived normal, candidate score, adjacent candidate, or ownership information participates in this frame.
4. Candidate-local compact source measure
There is no global source deletion. Every candidate tests every raw source record independently.
For candidate c, transform each point into dimensionless coordinates:
ξi​=s(pi​−oc​)⋅u​,ηi​=s(pi​−oc​)⋅v​,λi​=s(pi​−oc​)⋅a​.
Only points satisfying the fixed expanded-tube predicate enter the candidate-local compact measure:
α−4≤λi​≤β+4,ξi2​+ηi2​≤1352.
Assign each included record to the candidate-local cell
ki​=(⌊ξi​⌋,⌊ηi​⌋,⌊λi​⌋).
For each occupied cell, accumulate its source records in canonical source order and compute the centroid
qg​=(ξˉ​g​,ηˉ​g​,λˉg​).
Every occupied cell subsequently has unit mass, irrespective of its raw multiplicity. Raw multiplicity is retained only as an internal accounting integer and is not used in the covariance or decision.
Thus:


density duplication within a cell cannot manufacture evidence;


all points contribute to their cell centroid;


no cell or point is exclusively assigned to a candidate;


the same source record may legitimately participate in any number of candidate-local measures;


adding or removing another candidate cannot alter an existing candidate’s measure.


Candidate-local cells are discarded before processing the next candidate and are never serialized.
5. Stations, scales, and covariance statistic
A candidate with
L<40
is immediately retained as KEEP_ABSTAIN_SHORT; no candidate-local source scan is performed.
For L≥40, use exactly nine longitudinal stations:
λj​=α+4+j8L−8​,j=0,…,8.
The axial half-width is exactly
H=4.
Use exactly eight transverse radii:
(m0​,…,m7​)=(8,12,18,27,40,60,90,135).
For cell centroid qg​ and station j, define
ζgj​=λˉg​−λj​,ρg2​=ξˉ​g2​+ηˉ​g2​.
The cell belongs to station j, scale k exactly when
∣ζgj​∣≤4andρg2​≤mk2​.
Both boundaries are inclusive.
Implementation shall first assign a qualifying cell to the smallest radial shell containing it, then prefix-sum shells in increasing k to obtain the eight nested neighborhoods.
Because L≥40, adjacent station separation is at least 4. A cell can therefore update at most three stations, including exact-boundary coincidences.
5.1 Population covariance
For station j and scale k, let Qjk​ be the included unit-mass cell centroids expressed as
(ξ,η,ζ).
Use the population covariance:
Cjk​=njk​1​q∈Qjk​∑​qq⊤−qˉ​jk​qˉ​jk⊤​.
All sums use Neumaier-compensated binary64 accumulation in sorted cell-key order.
Define the transverse block quantities
Tjk​=Cξξ​+Cηη​,
Djkraw​=Cξξ​Cηη​−Cξη2​.
Let
Gjk​=Tjk2​.
Apply only this numerical PSD clamp:


if Draw<0 and
−Draw≤2−40G,
set D=0;


if 4Draw>G and
4Draw−G≤2−40G,
set D=G/4;


otherwise the station is ineligible.


Then define
Fjk​=4Djk​,Ejk​=Gjk​Fjk​​.
E∈[0,1] is the dimensionless transverse covariance energy. No eigenvalue computation is used.
5.2 Axis-decoupling requirement
Define
Ajk​=Cζζ​Tjk​Cξζ2​+Cηζ2​​.
The comparison is performed without division:
16(Cξζ2​+Cηζ2​)≤Cζζ​Tjk​.
This must hold at all eight scales.
5.3 Station eligibility
Station j is eligible only if all of the following hold:
nj0​≥24,nj7​≥96;
at the outer scale,
nj7−​≥24,nj7+​≥24,
where n− counts ζ<0, n+ counts ζ>0, and ζ=0 counts toward neither half;
and, at every scale:
Tjk​>0,Cζζ​>0,
the PSD clamp is valid, and the axis-decoupling inequality passes.
An ineligible station cannot support a veto.
6. Exact smooth-surface null model
An eligible station is SMOOTHLIKE under exactly one unified null model with two limiting regimes.
6.1 Flat limiting regime
The station is FLAT when, for every k=0,…,7,
Ejk​≤2−13.
Implement this as
213Fjk​≤Gjk​.
6.2 Smooth-curvature growth regime
The station is SMOOTH_GROWTH only when all conditions below hold.
Nonzero inner response:
Ej0​≥2−24,
implemented as
224Fj0​≥Gj0​.
No corner-like outer isotropy:
Ej7​≤21​,
implemented as
2Fj7​≤Gj7​.
Strict growth at every adjacent scale:
Ej,k+1​≥89​Ejk​,k=0,…,6,
implemented without division as
8Fj,k+1​Gjk​≥9Fjk​Gj,k+1​.
Let the scale-normalized quantity be
Zjk​=mk2​Ejk​​.
The complete eight-scale trajectory must remain within a factor of two:
kmax​Zjk​≤2kmin​Zjk​.
Implement this for every ordered pair p,q∈{0,…,7} as
Fjp​Gjq​mq2​≤2Fjq​Gjp​mp2​.
Finally,
SMOOTHLIKEj​=FLATj​∨SMOOTH_GROWTHj​.
These are not independently selectable modes. They are the zero-curvature and nonzero-curvature regimes of the same smooth-surface null hypothesis.
7. Candidate decision
For an evaluated candidate, let
S=#{j:SMOOTHLIKEj​}.
Let R be the longest contiguous run of SMOOTHLIKE stations in station order.
Emit:
VETO_SMOOTH_COVARIANCE⟺S≥8 ∧ R≥7​
Otherwise retain the candidate.
The only candidate result states are:


VETO_SMOOTH_COVARIANCE;


KEEP_FEATURE_INCOMPATIBLE;


KEEP_ABSTAIN_SHORT;


KEEP_ABSTAIN_SUPPORT.


Unknown, cap-adjacent, locally unsupported, complex, or non-smooth evidence can never veto.
No candidate decision may depend on another candidate’s existence or decision.
8. Hard caps and atomicity
ResourceFrozen capRaw source records200,000Candidates64Candidate-local occupied cells200,000 per candidateCandidate-source point tests12,800,000 per scanCandidate-local cell finalizations12,800,000 per scanCell-to-station updates38,400,000 per scanStation-scale metric tuples4,608 per scanResult candidate records64V73 scratch allocation536,870,912 bytesCanonical result1,048,576 bytesPeak process RSSless than 17,179,869,184 bytesWall timeat most 300 seconds per scan execution
There is no source trimming. N=200,001 causes SOURCE_CAP before candidate work.
All input schemas, finite values, source count, candidate count, candidate geometry, and scale values are validated before the first decision is committed.
Any hard-cap breach, invalid arithmetic state, output overflow, or serialization failure produces one scan-level abort artifact and zero partial candidate decisions. Result publication uses temporary-file write, canonical-byte verification, and atomic rename.
9. Compact no-source serialization
Canonical JSON is UTF-8, ASCII key order, compact separators, no NaN, and exactly one final newline.
All binary64 diagnostics are serialized as float.hex() strings, never decimal JSON numbers.
Each candidate record may contain only:


canonical candidate ordinal;


decision token;


9-bit eligibility mask;


9-bit flat mask;


9-bit smooth-growth mask;


9-bit smoothlike mask;


longest smoothlike run;


candidate metric SHA-256.


The metric digest covers the ordered support counts, F, G, and coupling-comparison operands for all 9×8 station-scale entries.
Forbidden serialized fields include:


source coordinates or normals;


source indices;


cell keys, centroids, memberships, or multiplicities;


candidate-local point lists;


filenames, paths, identities, family names, expected counts, labels, CAD references, baseline scores, and provenance.


The top level may additionally contain architecture token, status, source count, candidate count, work counters, timings, RSS, implementation hash, constants hash, input manifest hash, and result hash.
10. Mandatory 32-case source-free lock
The fixture generator must be independent of the V73 implementation. It may share neither predicate code nor generated constants.
Unless overridden below, use:
s=1,a=(1,0,0),o=(0,0,0),[t−​,t+​]=[−96,96],
X={−100,−99,…,100},Y={−160,−159,…,160}.
For a cross-sectional function f, define the extrusion
Pf​={(x,y,f(x,y)):x∈X, y∈Y}.
IDFrozen constructionRequired resultG01f=0all 9 FLAT; candidate vetoedG02f=0, y∈{0,…,160} onlyall 9 FLAT; perimeter candidate vetoedG03f=y2/4096all 9 SMOOTH_GROWTH; vetoedG04G03 with every record duplicated seven times before the source cap is applied to this smaller fixturebyte-identical candidate decision and masks to G03G05(f=(7/128)yG06(f=yG07f=y2+642​−64retainedG08f=y2+162​−16retainedG09U-return: lower arm (y,0),y=−160…0; bridge (0,z),z=1…15; upper arm (y,16),y=0…160; candidate anchor (0,0,8)retainedG10union of (f=yG11(f=(7/128)yG12G06 restricted to x=−100…4, candidate interval [−96,0]retained; no endpoint abstentionG13G06 with candidate interval [−16,16]KEEP_ABSTAIN_SHORTG14union of folds centered at y=−12 and y=12: ((x,d+q,qG15Run each G14 candidate alone and then togethereach candidate’s decision and metric digest identical in all runsG16Duplicate one G14 candidate exactlyboth duplicate records identical except canonical ordinalG17f=(−1)y/64all 9 FLAT; vetoedG18(f=(7/128)yG19G05 with every point at y≡0(mod4) duplicated five timesidentical decision and masks to G05G20Reverse source-record order for G03byte-identical resultG21Permute G07 source records by fixed mapping i↦(65537i+17)modNbyte-identical resultG22Apply signed coordinate permutation (x,y,z)↦(z,−x,y)+(1024,−2048,4096) to G03 and its candidateidentical masks and decisionG23Multiply every coordinate, endpoint, anchor, and s in G07 by 8identical masks and decisionG24Shift candidate anchor by 37a, adjust endpoints geometrically, then reverse the supplied axisidentical masks, decision, and canonical geometry digestG25Independent radial membership oracle with points at nextafter(8,−∞), 8, and nextafter(8,+∞)first two included at scale 0; third excludedG26Independent axial membership oracle at nextafter(4,−∞), 4, and nextafter(4,+∞)first two included; third excludedG27Independent covariance predicate values immediately below, at, and above every dyadic comparison boundary, including the 2−40 PSD clampexact inclusive comparator behaviorG28200,000 finite unique source records and one short candidateaccepted; no source trim; source count unchangedG29G28 plus one finite recordatomic SOURCE_CAP; zero candidate recordsG30valid small source with 64 duplicate candidatesaccepted; 64 complete records; exact work accountingG31G30 plus one candidateatomic CANDIDATE_CAP; zero candidate recordsG32canonical serialization, forbidden-key scan, 1 MiB result cap, primary/replay execution from cleared process statebyte-identical artifacts; no forbidden source-bearing field
The constants fixture must independently assert:


radii exactly 8,12,18,27,40,60,90,135;


H=4;


length floor 40;


support 24,96,24,24;


dyadics exactly 0x1p-13, 0x1p-24, 0x1p-40, 0x1p-4, and 0x1p-1;


growth integers 9 and 8;


normalized spread factor 2;


no production AST occurrence of pi, DEG, radians, sin, cos, tan, an angle-valued hexadecimal literal, or any import from V72/CASL.


Source-free execution rule
Freeze and hash:


V73 implementation;


independent fixture generator and oracle;


single-invocation runner;


constants manifest;


serialization schema;


precommit manifest.


Execute the complete suite exactly once. One top-level invocation must contain a primary run and a cleared-state internal replay.
Proceed only on:


primary 32/32;


replay 32/32;


byte-identical canonical artifacts;


all caps and accounting exact.


Any failure terminally retires V73. No amendment, correction, threshold adjustment, fixture replacement, or source-free rerun is authorized. U01–U04 remain unopened.
11. One-shot U01–U04 protocol
After a clean source-free lock, execute exactly one top-level U01–U04 invocation containing primary and cleared-state replay. Use the unchanged sealed four-scan manifest and unchanged frozen 48-candidate stream.
Gold, expected bend counts, labels, identities, filenames, CAD, benchmark feedback, and the reserved scan remain closed.
All of the following are mandatory:


Exactly 48 candidate records in primary and 48 in replay.


Canonical primary/replay bytes identical.


No scan-level abort.


At least 36 candidates evaluated rather than short/support-abstained.


At least one and at most 24 total vetoes.


For each scan i, with frozen input candidate count Ci​,
Vi​≤⌊2Ci​​⌋.


Zero candidate additions, geometry changes, endpoint changes, reordering, or cross-candidate decision dependencies.


Every source/candidate/work/result cap respected.


Every primary and replay scan completes within 300 seconds and below 16 GiB peak RSS.


Canonical output remains below 1 MiB and passes the forbidden-field audit.


Zero vetoes means the architecture is non-actionable and terminal. Exceeding the bounded deletion gates means the architecture is overbroad and terminal. No rescue run is permitted.
12. Mandatory native-resolution visual adjudication
The reserved physical scan remains sealed until visual adjudication is complete and signed.
Render from the original raw source records, never from candidate-local cells and never from a decimated cloud.
For every vetoed candidate, produce:


unchanged full-part native-resolution view with all pre-V73 candidates;


corresponding post-V73 view;


view along the candidate axis;


nine transverse station panels using the exact ∣ζ∣≤4 slabs;


a joint overlay with every other candidate satisfying both
lineDistance≤135s
and
∣a⋅a′∣≥15/16.


A veto passes visual adjudication only when the source visibly forms one flat or smoothly varying surface through the candidate at at least seven contiguous station panels and no localized crease, fillet center, return, hem, or second bend is visible at any remaining panel.
Any one of these signatures retires V73:


the vetoed line coincides with a visible sharp or rounded crease at two nonadjacent stations;


it follows a visible bend for at least one quarter of its supported segment;


it removes a return or hem center;


it removes one member of two nearby distinct bends;


the apparent smoothness is caused by occlusion or missing support rather than a visibly smooth surface;


the line is visually ambiguous rather than unambiguously false;


the native source contradicts the compact-cell interpretation.


Every changed result and every qualifying nearby retained candidate must be inspected. Ambiguity is failure, not permission to tune.
13. Reserved-scan opening
Only if the source-free lock, all U01–U04 operational gates, and every native-resolution visual gate pass does this ruling authorize opening the reserved physical scan.
The reserved scan must use the identical V73 implementation, constants, runner, and serialization bytes in one primary-plus-replay invocation. No threshold, cap, scale, fixture, candidate-stream, or rendering change is permitted.
Gold and expected count remain closed. Perform native-resolution inspection of every changed candidate before any scoring or promotion claim.
Any determinism, cap, runtime, serialization, or visible false-veto failure terminally retires V73. A pass is prospective engineering evidence only; it does not open runtime/main or production and does not establish the final 30-scan/15-family promotion claim.
14. Why this is the bounded successor
V72’s terminal evidence does not establish source-only impossibility. One failure came from applying a candidate-independent removal to overlapping candidate queries and then requiring cross-candidate tail faithfulness; the other was an adjacent-expression binary64 contradiction. V73 eliminates both premises rather than correcting either V72 assertion.
The exact causal target is the remaining class of candidate lines supported by a flat face, perimeter, or broad smooth roll. V73 asks whether the complete multiscale transverse measure remains consistent with a smooth surface. A genuine localized crease protects itself by breaking the required all-scale growth law. Partial, nearby, hem, opposing, and uncertain cases also default to retention, so the mechanism cannot improve apparent precision merely by broadly reducing prediction count.
Integral invariants were developed because differential curvature estimates are noise-sensitive, while neighborhood-integrated descriptors can provide more stable and useful multiscale behavior. UT Austin Computer Science+1
Mérigot, Ovsjanikov, and Guibas showed that covariance-based integral measures can encode curvature in smooth regions and the direction and angle structure of sharp features, with robustness to noisy point-cloud sampling. Guibas Lab
Pauly, Keiser, and Gross used covariance analysis of local point neighborhoods across scale and observed that increasing neighborhood size attenuates high-frequency variation. CGL ETHZ
The particular statistic E=4det(B)/tr(B)2, the all-scale smooth-growth null model, the dyadic gates, and the conservative veto rule are system-specific inferences, not claims copied from those papers.
