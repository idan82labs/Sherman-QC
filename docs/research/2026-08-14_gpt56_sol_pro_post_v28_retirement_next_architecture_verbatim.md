RULING

AUTHORIZE exactly one bounded action:

V29_CALIBRATED_RANGE_CONTINUITY_OWNERSHIP_V1
abbreviation: CRCO_V1

This is a minimal scan-acquisition capability opening, not another reconstructed-surface detector. V29 may use calibrated raw range-frame topology—depth pixels, validity, intrinsics, and sensor pose—to adjudicate whether an already-existing geometric bend candidate lies on:

a depth-continuous interior surface;

a true range jump such as a silhouette, free edge, opening, or rim; or

an unresolved/occluded region.

V29 shall neither fit a new axis nor pair, merge, complete, or extend fragments. It operates only on geometric-core candidates already produced before selection. V27 and V28 are not invoked, copied, repaired, or renamed; the V29 branch starts from the frozen V22 pre-NMS candidate path.

This is the smallest defensible opening because the reconstructed mesh has discarded acquisition facts that distinguish “no surface because this is a physical termination” from “no surface because this view did not observe it.” V21 correctly forbids guessing that distinction from byte-identical permitted surface observations. Structured range data retains line-of-sight, visibility, and range-discontinuity information that an unorganized point set or fused mesh can lose. 
Stanford Graphics
+2
IPB Uni Bonn
+2

No further surface-only architecture is authorized in this tranche. I am not issuing a global research STOP because this one additional observation is both narrowly scoped and causally identifiable.

The action must not be represented as a likely route directly to the target. The 35-case set contains 176 true lines. Reaching 90% recall requires at least 159 TP, which is 70 more than V27. At 159 TP, 90% precision permits at most 17 FP, roughly 80 fewer than V27. V29 is therefore a falsification of one potentially material error mechanism, not a promotion experiment.

RESEARCH BASIS
Established published machinery

Curless and Levoy’s range-image fusion work treats scanner observations as structured measurements with directional line-of-sight uncertainty. It distinguishes observed empty space from never-observed space, integrates multiple aligned range images, and emphasizes order-independent processing. It also explicitly notes that converting acquisition data into unorganized points discards useful structure and reliability information. That is the central technical justification for opening raw range provenance rather than inventing another surface-only score. 
Stanford Graphics
+1

The free/occupied/unknown distinction is standard in ray-based 3D mapping. OctoMap, for example, represents occupied, free, and unknown regions separately and uses a compact hierarchical representation. V29 does not need to implement OctoMap, but it adopts the same epistemic rule: an invalid or absent measurement is not evidence of free space. 
IPB Uni Bonn
+1

Range-image research has long distinguished surface boundaries and depth-jump edges, including jumps arising from self-occlusion and inter-object occlusion. Hoover et al.’s comparative range-segmentation work treats jump-edge structure as a measurable property of the range image rather than merely a property of the reconstructed mesh. Viewpoint-dependent visibility can also be computed directly from point observations when the viewpoint is known, as shown by Katz, Tal, and Basri. 
ResearchGate
+3
Informatics Homepages
+3
Informatics Homepages
+3

Robust geometric predicates and explicit total ordering are established numerical tools, but reproducible behavior does not follow automatically from floating-point arithmetic or parallel reductions. V29’s ordering, quantization, and tie rules below are therefore binding engineering requirements, not optional implementation details. 
Springer
+1

Sherman-specific hypothesis

The published literature does not establish that calibrated range continuity will improve Sherman bend-line F1, nor does it supply the station counts, tolerances, or ownership predicates below.

The narrow Sherman hypothesis is:

A material subset of V22 false positives and false negatives occurs after the pipeline collapses acquisition-continuity information into a fragmented surface representation. A candidate that is geometrically plausible but ownership-ambiguous can sometimes be identified as an interior fold or a physical range jump from multiple calibrated raw views.

The exact 24-station construction, four-keyframe selection, 16/24 decision gates, 5/7 flank tests, and V29 rescue/veto restrictions are preregistered Sherman hypotheses. Failure of any one of them retires V29; the literature cannot be invoked to excuse a failed implementation or fixture.

CAUSAL SCOPE

V29 changes the permitted observation from:

reconstructed scan surface only

to:

reconstructed scan surface + bounded calibrated raw range topology

It remains scan-only. It does not open CAD, drawings, labels, expected counts, filenames, family identity, benchmark feedback, RGB, intensity, or manufacturing records.

V29 may claim	V29 may not claim
A candidate is depth-continuous across both proposed flanks in at least two sufficiently different raw views.	A depth-continuous corner was manufactured by bending rather than welding, joining, machining, or another process.
A candidate coincides with a repeatable first-return depth jump to a valid farther surface.	A missing or invalid range pixel represents free space.
A previously ownership-rejected geometric-core candidate has source evidence supporting interior continuation.	A candidate failing curvature, axis, span, or other geometric-core tests should be rescued.
A selected boundary-risk candidate has source evidence supporting a physical termination or opening.	An ordinary interior candidate without the frozen boundary-risk predicate should be removed.
The mechanism may improve precision through certified vetoes and recall through certified ownership rescues.	One prospective scan demonstrates benchmark F1, family generalization, exact-count performance, or the 90/90 target.

The binding equivalence rule is:

Identical surface bytes and identical calibrated range-frame bytes must produce byte-identical inference.

Identical surface bytes with different raw range observations may produce different inference, but only through the frozen continuity/jump predicates.

Hidden manufacturing semantics that do not alter either permitted observation must never alter inference.

A mandatory source-free semantic-twin fixture enforces the final condition.

EXACT SPECIFICATION
1. Capability and input contract

The existing bounded geometric detector input remains at approximately 200,000 points.

V29 additionally receives read-only access to raw range frames containing only:

first-return depth;

valid/invalid status;

calibrated intrinsics or equivalent range-lattice calibration;

sensor-to-scan rigid pose;

integer pixel coordinates.

RGB, reflectance, intensity, timestamps, filenames, scanner serial numbers, operator IDs, part IDs, and acquisition comments are excluded.

Exactly four range keyframes are selected before candidate inspection:

Canonicalize every available frame’s depth and validity encoding.

Construct a content hash from those bytes and a global-rigid-invariant relative-pose signature.

Select the unordered frame pair with maximum optical-axis separation; break ties by sorted content hashes.

Select the third and fourth frames by maximin optical-axis separation from the already selected set; break ties by content hash.

Exact duplicate frame observations are collapsed.

A continuity or jump certificate requires two contributing frames whose station-to-camera viewing directions differ by at least 12°.

Fewer than four valid calibrated frames is a precommit capability failure. There is no single-view fallback.

At most 262,144 unique raw pixels may be queried during one V29 job. Pixel coordinates are deduplicated and sorted before access. Exceeding the cap is a structural failure; no truncation or cap increase is allowed.

Invalid pixels are always UNKNOWN. They can never constitute background, free space, a jump, continuity, or occlusion evidence.

2. Candidate input

V29 is inserted inside select_unified:

bounded candidate generation
→ frozen geometric-core predicates
→ CRCO_V1 ownership adjudication
→ deterministic total sort
→ existing ordinary bounded-complexity NMS
→ build_result serialization

V29 must execute before the first and only final NMS. Nothing may add or alter candidates after that NMS.

V29 receives no source-point or face lists in candidate records. Each bounded record contains only:

canonical line endpoints;

support interval;

two unordered proposed flank tangent vectors;

frozen geometric-core pass bits;

frozen boundary-risk bit;

frozen rejection reason;

span and scalar quality summaries;

content-derived candidate hash.

Maximum candidate records: 4,096. Exceeding the cap is terminal.

A rejected row is eligible for V29 rescue only when:

every frozen geometric-core predicate passed;

its sole rejection reason is BOUNDARY_OWNERSHIP_AMBIGUOUS;

no curvature, axis, span, continuation-direction, resource, or duplicate predicate failed.

A selected legacy row is eligible for V29 veto only when its frozen boundary_risk is true. That bit is true only when a mesh/open boundary lies within 8s of the candidate at at least 8 of the 24 stations.

V29 never changes a line’s location, direction, endpoints, or span. A rescued line must reuse the geometric-core bytes exactly.

3. Station construction

Let the canonically ordered candidate endpoints be a and b, and let s be the already-frozen detector length scale.

Construct exactly 24 stations:

x
k
	​

=a+
24
k+0.5
	​

(b−a),k=0,…,23.

For each station and each of the two unordered flank tangents t_j, construct seven expected surface samples:

q
k,j,m
	​

=x
k
	​

+mst
j
	​

,m∈{2,3,4,5,6,7,8}.

For each selected keyframe:

Project the station, the 14 flank samples, and the pixel path joining the two inner flank samples.

Query the exact native-frame pixels required by those projections.

Back-project valid depths into the scan frame.

Quantize all comparison coordinates to s/1024 before classification.

For an expected sample q, the observed first-return sample is:

MATCH when its Euclidean distance from q is at most 1.5s;

BEHIND when it is at least 4s farther from the camera than q;

IN_FRONT when it is at least 4s closer to the camera than q;

UNKNOWN otherwise.

A frame/station is:

CONTINUOUS

at least 5/7 samples on each flank are MATCH;

the candidate is not classified IN_FRONT at its projected center;

a four-connected valid-depth path exists between the two inner flank projections;

every adjacent pair on that path has a back-projected 3D gap no greater than 2s.

JUMP

one flank has at least 5/7 MATCH;

the other flank has at least 5/7 BEHIND;

no qualifying four-connected depth path crosses the candidate;

the candidate center is not behind a closer surface.

The two flank assignments are tested symmetrically. Side numbering never affects the result.

OCCLUDED

at least 5/7 relevant samples are IN_FRONT;

neither CONTINUOUS nor JUMP passed.

All other outcomes are UNKNOWN.

4. Cross-view station state

For each station:

C when at least two views separated by 12° classify it CONTINUOUS;

J when at least two such views classify it JUMP and no view classifies it CONTINUOUS;

O when at least two such views classify it OCCLUDED and no view classifies it CONTINUOUS or JUMP;

U otherwise.

Contradictory evidence does not get ranked or averaged. It becomes U.

5. Candidate decision

A rescue is emitted only when:

the candidate satisfies the rescue eligibility contract;

at least 16/24 stations are C;

zero stations are J;

the longest consecutive run of C stations is at least 8.

A terminal veto is emitted only when:

the candidate is already selected by a legacy lane;

its frozen boundary_risk is true;

at least 16/24 stations are J;

zero stations are C;

the longest consecutive run of J stations is at least 8.

Every other candidate is unchanged.

There is no score interpolation, soft confidence, threshold sweep, “near pass,” or scan-level quota.

6. Deterministic ordering and canonicalization

All input arrays are sorted before use. No operation may depend on source iteration order.

Candidate endpoints are sorted by their quantized coordinate tuple. Endpoint reversal must therefore produce the same station sequence after canonicalization.

Flank vectors are treated as an unordered pair. Their canonical order is the lexicographic order of their quantized direction bytes after endpoint canonicalization.

Candidate identity is:

SHA-256(
  canonical endpoints
  || canonical unordered flanks
  || geometric-core certificate bytes
  || support interval
)

It may not include an array index, filename, insertion sequence, memory address, frame sequence number, or V27/V28 candidate ID.

Frame identity is content-derived and relative-pose-derived, not list-position-derived.

V29 is single-threaded. It performs no parallel floating-point reduction. All branch predicates operate on quantized values or robust orientation/comparison predicates. NaNs, infinities, signed zero, and invalid calibration values are rejected before inference.

7. Ranking, NMS, and ownership

Legacy candidate order and legacy rank bytes remain frozen.

The total pre-NMS order is:

non-vetoed legacy candidates, preserving their exact frozen relative order;

V29 rescue candidates sorted by:

continuous_station_count descending
longest_continuous_run descending
unknown_station_count ascending
span descending
canonical_candidate_hash ascending

A V29 rescue candidate can never suppress a non-vetoed legacy candidate in a duplicate collision. The legacy lane always wins.

A veto is candidate-local: it removes only the matching geometric-core hash. It does not suppress a nearby candidate with a different core hash.

A vetoed core hash may not be reintroduced by the V29 rescue lane.

The existing ordinary NMS executes after these decisions. Its duplicate geometry and thresholds remain byte-for-byte frozen. build_result then only sorts and serializes.

8. Resource and cache bounds
Item	Hard bound
Geometric input points	existing approximately 200,000 cap
Candidate records	4,096
Selected keyframes	exactly 4
Queried raw range pixels	262,144
V29 incremental allocated memory	512 MiB
Full process-tree peak RSS	less than 16 GiB
V29 incremental P95	at most 30 s
Full inference P95	at most 300 s
Persistent V29 cache	at most 16 MiB

The persistent cache may contain frame hashes, calibration hashes, station-state counts, candidate hashes, and final line records. It may not contain depth images, pixel patches, point coordinates, source vertex lists, source face lists, projected point lists, or raw ray arrays.

9. Prohibited inputs and behavior

The worker may not access:

any of the historical 35 LOFO cases;

the 26 recent source pools;

U04 or its drawing-confirmed mechanism;

V27 or V28 outputs;

labels, drawings, CAD, expected counts, filenames, family names, good/bad status, or benchmark results;

RGB or intensity;

production feedback;

runtime/main.

No constant may differ between source-free and real-scan execution.

FAST FALSIFICATION GATES

The gates execute in the order below. Failure stops the action immediately; later fixtures or real scans remain unopened.

Gate 0 — static precommit audit

Before fixture generation or detector execution, freeze:

source hash;

executable hash;

fixture-generator hash;

manifest hash;

all numerical constants;

raw range schema;

keyframe selection;

candidate record schema;

total-order implementation;

NMS insertion point;

legacy-wins duplicate rule;

resource caps;

renderer hash.

The audit must prove:

CRCO_V1 is called before the final NMS.

No candidate is added after that NMS.

A V29 rescue cannot suppress a non-vetoed legacy candidate.

Candidate and frame keys contain no input indices.

Endpoint reversal and flank swapping are canonicalized.

Invalid depth never contributes to a jump or continuity certificate.

V27 and V28 are unreachable in the execution graph.

The four-keyframe selection is independent of frame list order.

All structural caps fail closed rather than truncate.

No historical or real input is reachable from the source-free runner.

Any failure produces:

V29_CRCO_PRECOMMIT_CONTRACT_FAILURE_RETIRE

No patch or audit rerun is authorized.

Gate 1 — source-free fixtures

All fixtures and hidden expected outputs are frozen before the first execution. Hidden semantic labels are never supplied to the worker.

Fixture	Required result
SF01_OCCLUDED_90_FOLD	Fused surface candidate is ownership-rejected; two raw views show genuine depth continuity. Exactly one V29 rescue. Rescued line geometry is byte-identical to the geometric-core line. Direction error ≤0.10°, line distance ≤0.20s.
SF02_FREE_EDGE_CAUSAL_TWIN	Surface bytes and geometric-core candidate bytes are identical to SF01. Raw views instead show a valid farther background beyond a physical free edge. No rescue; the selected boundary-risk candidate is vetoed.
SF03_60_DEG_SCALE_CROP	Different scale, 60° fold, axial crop, two continuous views. Exactly one rescue with the same geometric tolerances.
SF04_FULLY_OBSERVED_CLEAN_FOLD	Ordinary legacy line remains byte-identical. V29 emits neither rescue nor veto.
HN01_THROUGH_HOLE_RIM	Valid background visible through the opening. Zero final V29 lines; any boundary-risk candidate is vetoed.
HN02_UNRELATED_PANEL_OCCLUSION	Foreground panel obscures the missing side but no range-continuous path crosses the proposed fold. No rescue.
HN03_FORMED_VS_WELDED_SEMANTIC_TWINS	Geometry and range frames are byte-identical while hidden manufacturing labels differ. Inference and cache bytes must be identical.
HN04_REFLECTIVE_INVALID_DROPOUT	Missing side consists of invalid pixels rather than valid farther returns. All relevant stations remain unknown; no rescue or veto.
HN05_CAP_WALL_CYCLE_ALIAS	Cap, thickness wall, cycle-closing boundary, and terminal silhouette variants. No rescue; any jump-certified boundary-risk line is vetoed.
HN06_CONFLICTING_VIEWS	One view appears continuous and another appears as a jump. Conflict becomes unknown; no action.

SF01 and SF02 are the decisive causal pair. The surface representation is deliberately identical; only the newly opened acquisition evidence differs.

Any missed positive, extra line, failed veto, semantic-twin divergence, or conflict resolution other than UNKNOWN retires V29.

Gate 2 — permutation, endpoint, and rigid-transform invariance

The activation fixtures must be executed under all of the following fixed variants:

complete point-record reversal;

frame-record reversal;

pixel-record reversal within every frame;

content-key permutation;

candidate endpoint reversal;

flank-vector swap;

endpoint reversal plus flank swap.

Inference bytes, station certificates, candidate hashes, and cache bytes must be identical.

Apply four fixed proper rigid transforms to all points, lines, sensor poses, and calibration frames:

identity;

cyclic axis rotation plus translation (13s, -7s, 5s);

37° rotation about normalized (1,2,3) plus a non-axis-aligned translation;

121° rotation about normalized (-2,1,4) plus a non-axis-aligned translation.

After inverse-transforming and recanonicalizing outputs, semantic output bytes and station-state bytes must match the identity execution exactly.

Any mismatch produces:

V29_CRCO_METAMORPHIC_FAILURE_RETIRE

Gate 3 — clean-process replay

Every source-free fixture is executed twice in separate clean processes.

Required:

byte-identical inference output;

byte-identical cache;

byte-identical manifest;

identical selected keyframe hashes;

identical queried-pixel set;

identical peak-cap status;

identical visual raster pixels after stripping image metadata.

A third run is prohibited unless both preregistered runs passed and the third run is part of the already-frozen test manifest. There is no diagnostic rerun after failure.

Gate 4 — resource gate

Required across the source-free suite:

full process-tree RSS below 16 GiB;

V29 incremental allocation no greater than 512 MiB;

V29 incremental P95 no greater than 30 s;

full inference P95 no greater than 300 s;

cache no greater than 16 MiB;

no candidate or queried-pixel cap reached;

100% job completion.

Any limit violation is terminal, even when geometric outputs are correct.

Gate 5 — mandatory 4096px native visual audit

Before any real scan is selected, generate native-resolution visual artifacts for every fixture that produces a rescue or veto.

Each fixture requires four 4096px-long-edge artifacts:

Native surface view: full-resolution source surface, geometric-core candidate, final line, and any veto mark. No decimated proxy.

Range-keyframe view: all four selected raw range frames with candidate stations and both proposed flank projections.

Continuity evidence view: the queried pixels, back-projected samples, valid farther returns, invalid pixels, and connected depth paths.

Station matrix: all 24 stations with per-frame and cross-view C/J/O/U states.

The audit fails when any of the following is visible:

a rescue lacks genuine depth continuity across both flanks;

a jump is based on invalid or missing pixels rather than a valid farther return;

evidence belongs to an unrelated panel;

a rescue lies on a silhouette, opening, rim, cap, thickness wall, or terminal edge;

a veto removes an evidently depth-continuous interior fold;

projected samples are displaced from the native source geometry;

the line is duplicated or ordinary NMS ownership is wrong.

A visual failure overrides numerical fixture success.

REAL-SCAN GATE

Exactly one prospective routine-workflow scan may be used.

Selection

After all source-free gates pass:

Publish the final code, executable, manifest, and renderer hashes and an immutable time T0.

Consider the next ten routine scan acquisitions entering the ordinary queue after T0.

Select the first acquisition whose schema-only validator confirms:

valid raw calibrated range frames;

at least four unique frames;

finite calibration;

bounded geometric input;

no parse corruption.

Eligibility may not inspect candidate count, V29 activation, geometry type, expected bends, family, filename, or human opinion.

Strip filenames and operational metadata and assign a content hash.

Freeze the complete selected input bundle before inference.

If none of the next ten routine acquisitions satisfies the capability schema, retire with:

V29_CRCO_CAPABILITY_NOT_OPERATIONAL_RETIRE

Once an input is selected, there is no replacement for zero activation, difficult geometry, unfavorable visuals, crash, ambiguity, or failed replay.

No intentionally constructed part, rescan, known troublesome family, known positive, or previously inspected scan is eligible.

Execution

Run the selected scan once, followed by exactly one clean-process replay.

The following are terminal:

zero V29 rescues and zero V29 vetoes;

output or cache replay mismatch;

a structural cap;

a resource failure;

a crash;

a calibration inconsistency.

Zero activation is not evidence of safety; it is an uninformative prospective replication and retires the mechanism.

Source-only visual adjudication

Two reviewers independently inspect only:

the native full-resolution scan surface;

the four raw depth keyframes;

V29 station projections;

valid/invalid depth status;

the candidate lines and final markers.

They receive no CAD, drawing, gold count, part name, family, historical prediction, V16/V22/V27 result, or benchmark match.

For every V29 decision, each reviewer records yes/no answers to these fixed questions:

Does the displayed raw depth evidence agree with the emitted CONTINUOUS or JUMP certificate?

For a rescue, is there an actual source-visible continuous surface path across both proposed flanks?

For a veto, is the missing side supported by a valid farther first return rather than invalid or absent depth?

Is the evidence owned by the candidate surface rather than an unrelated foreground or background panel?

Is the final marker geometrically located on the claimed interior fold band or physical termination?

Did legacy-wins duplicate ownership and final NMS behave as specified?

Every V29 decision must receive unanimous affirmative answers. Reviewer disagreement is a failure, not an invitation for discussion, threshold modification, or a replacement scan.

The real scan requires the same four classes of 4096px artifacts as the source-free fixtures, plus a full-scan overview showing every legacy line, rescue, and veto.

This gate may establish only that the range-continuity mechanism activated prospectively and was source-visually correct on one scan. It does not produce a precision, recall, exact-count, or family-generalization estimate.

RETAIN/RETIRE RULE
Retain

V29 receives the token:

V29_CRCO_ONE_PROSPECTIVE_MECHANISM_SURVIVAL_RESEARCH_ONLY

only when all of the following are true:

the static audit passes;

every source-free positive and hard negative passes;

the surface-identical causal twins behave as specified;

the manufacturing-semantic twins remain byte-identical;

all permutation, endpoint, flank, and rigid-transform tests pass;

both clean-process executions are byte-identical;

every resource and cache bound passes;

every 4096px source-free visual passes;

exactly one prospective input is selected by the frozen rule;

that scan produces at least one V29 decision;

replay is byte-identical;

every prospective decision passes unanimous source-only visual adjudication.

Retention means only that calibrated range continuity is a surviving research mechanism. It does not alter V16, promote V27, revive V28, modify production, or establish benchmark improvement.

Retire

Any failure at any gate produces:

V29_CRCO_TERMINAL_RETIRE

with a frozen reason code.

Retirement is mandatory for:

unavailable or invalid range provenance;

fewer than four valid raw range frames;

nondeterministic frame or candidate ordering;

input-order dependence;

endpoint or flank reversal dependence;

rigid-transform mismatch;

ordinary NMS occurring before V29;

a V29 addition suppressing a non-vetoed legacy candidate;

post-NMS candidate creation;

a source-free miss or alias activation;

semantic-twin divergence;

invalid pixels being interpreted as free space;

cap truncation;

any resource exceedance;

zero prospective activation;

prospective replay mismatch;

any visually false rescue or veto;

reviewer disagreement.

There is no V29b, threshold change, alternate keyframe count, lowered station count, relaxed view separation, replacement input, historical replay, benchmark opening, or post-hoc rescue under this authorization.
