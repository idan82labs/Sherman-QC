**Ruling
DECISION: OPEN_ONE_NECESSARY_CAPABILITY**
**Capability opened:** CIBEC_V1 — CANONICAL_INSPECTION_BEND_EVENT_CONTRACT_V1
Do **not** fund a V10 geometry rule yet. Do **not** change any V9 threshold. Do **not** expose detector developers or inference code to the new physical scans.
The smallest necessary opening is a manufacturing-authoritative, scan-grounded definition of:
	1	what constitutes one inspection bend event;
	2	which single line represents that event;
	3	when multiple CAD surfaces belong to the same event; and
	4	how hems, beads, lips, rolls, interrupted bends, compound transitions, and partial visibility are counted.
The package manifest verifies, and the five contact sheets support this ruling.

**1. Dominant causal diagnosis**
The remaining failure is primarily an **identity and representation failure**, not a registration, determinism, latency, or threshold failure.
V9 operates at the level of paired cylindrical B-Rep primitives. Its two-normal gate determines that measured scan geometry contains a local surface transition near a proposed axis. It does **not** establish that the transition is:
	•	one inspection event rather than one constituent surface of a larger event;
	•	represented by the same line convention as the trusted truth;
	•	distinct from another nearby transition belonging to the same forming operation.
This produces two coupled error mechanisms.
**Dominant FP mechanism: valid primitives are emitted as separate inspection identities**
A lip, bead, hem, rolled edge, or compound rounded transition may contain several real cylindrical patches and several physically defensible axes. Multiple candidates can therefore pass both proximity and two-normal verification even though the inspection ontology counts the entire region once—or does not count some constituent transitions at all.
The V9 normal test can verify that each candidate lies near a real transition, but it cannot decide the **equivalence relation**
[ {\text{CAD surface primitives}}\longrightarrow {\text{inspection bend events}}. ]
Consequently, tightening the two-normal threshold would mostly trade recall for precision without resolving the many-to-one identity problem.
**Dominant FN/localization mechanism: the emitted axis is not necessarily the scored line**
The trusted line may represent a virtual sharp, tangency line, neutral axis, flange intersection, or manually selected inspection locus. A native cylinder axis can be physically meaningful while lying more than the 4 mm localization tolerance from that convention.
40500000 is decisive evidence: the CAD lane produces approximately the trusted number of visibly plausible formed transitions, yet the lines do not localize to the trusted set. 47959001 similarly produces five visually plausible lines against five trusted lines while scoring zero matches. That pattern cannot be explained by candidate count alone.
Partial 38172000 scans add genuine visibility-related FNs, but solving partial support first would increase the number of ambiguously represented candidates without fixing their identities.
B-Rep adjacency and feature labeling are appropriate mechanisms for grouping faces, but the sheet-metal literature also distinguishes geometric feature recognition from manufacturing-operation identity. Published work notes that grouping geometric bends into one folding operation can depend on process sequence and shop-floor intent rather than being recoverable uniquely from folded geometry. Feature taxonomies are likewise application-specific and must be defined before classification or consolidation rules are meaningful. (<u>[ResearchGate](researchgate.net/publication/392250597_Bending_Simulation_Framework_for_Rapid_Feasibility_Checks_of_Sheet_Metal_Parts)</u>)
**Inference for Sherman QC:** another virtual-sharp, clustering, or topology-merging rule would silently choose an ontology and then appear to succeed or fail according to an already consumed annotation convention. That is not a clean scientific test.

**2. The single capability: CIBEC_V1**
CIBEC_V1 is an evaluation and truth-definition capability. It is **not an inference feature** and may never be provided to either detector at runtime.
It produces two cryptographically separated artifacts:
	1	SCAN_BEND_EVENTS_V1.json The primary scan-grounded truth, authored without CAD, model output, existing gold, expected counts, filenames, or family identity.
	2	CAD_EVENT_CORRESPONDENCE_V1.json A post-lock map from native STEP faces and geometric primitives to the already frozen scan event IDs. This artifact can diagnose and consolidate the CAD-assisted lane but cannot modify the primary scan truth.
**2.1 Required inputs
Generic contract inputs**
	•	CIBEC_V1_CHARTER.yaml
	•	a signed source-free synthetic fixture pack;
	•	the exact six-view renderer already used for the V9 audit, pinned by SHA-256;
	•	a deterministic 3D line-annotation and serialization tool.
**Per-case primary annotation inputs**
	•	raw scan points;
	•	units metadata;
	•	the six fixed native-resolution views;
	•	a random 128-bit case identifier.
The primary annotators must not receive:
	•	CAD or rendered CAD overlays;
	•	part or family names;
	•	filenames;
	•	expected bend counts;
	•	existing trusted lines;
	•	A0, incumbent, V7, V8, V9, or multisource predictions;
	•	metric tables;
	•	previous annotator results.
**Post-lock CAD correspondence inputs**
Only after the primary scan truth is hashed and immutable:
	•	native STEP;
	•	OCP face topology;
	•	the existing V9 raw-scan-to-CAD transform;
	•	overlay trust/abstention state;
	•	the frozen primary event lines.
No second registration system is authorized.

**2.2 Mandatory charter decisions**
Before any physical case is shown, one manufacturing/inspection authority must author and a second authority must countersign one value for each of the following fields:
**Charter field
Required decision**
COUNTING_UNIT
Exact definition of one inspection event
CANONICAL_LINE
Exact line convention used to represent the event
INNER_OUTER_RULE
Whether paired inner/outer surfaces are one event
COMPOUND_RADIUS_RULE
When adjacent rounded patches merge
RETURN_LIP_RULE
Counting and marker rule
OPEN_HEM_RULE
Counting and marker rule
CLOSED_HEM_RULE
Counting and marker rule
BEAD_RULE
Counting and marker rule
ROLLED_EDGE_RULE
Counting and marker rule
INTERRUPTION_RULE
Maximum interruption, if any, that remains one event
PARTIAL_VISIBILITY_RULE
Required observed evidence and endpoint behavior
NONLINE_EVENT_RULE
Whether a legitimate inspection event can lack one line
ENDPOINT_RULE
How the visible event span is delimited
No value may be selected after examining physical scores or current prediction overlays.
A rule such as “use whichever representation matches the operator line” is invalid. Each rule must be executable from the scan evidence and generic feature class alone.

**2.3 Coordinate frame and fixed constants**
The following constants are frozen:
**Item
Frozen value**
Primary coordinate frame
Raw scan frame
Units
millimetres
Minimum render-admissible line span
8.0 mm
Annotation-agreement direction tolerance
5.0°
Annotation-agreement transverse tolerance
4.0 mm
Annotation-agreement shorter-span overlap
0.60
Endpoint extrapolation beyond observed support
0.0 mm
Primary annotators
2
Adjudicators
1
Fixed rendered views
6
Serialized coordinate quantum
(10^{-6}) mm
Non-finite coordinate tolerance
none; hard failure
Permitted primary event geometry
one finite straight segment
Source points in compact outputs
prohibited
The 5°/4 mm/60% matcher is reused solely to measure agreement and preserve comparability with the present scoring convention. It is not used to modify an annotation.
**Endpoint canonicalization**
For endpoints (p_0,p_1), define
[ d=\frac{p_1-p_0}{|p_1-p_0|}. ]
Let (k) be the lowest-index coordinate among those having maximum absolute magnitude in (d), with coordinate priority (x<y<z). If (d_k<0), swap the endpoints.
Coordinates are rounded half-to-even to (10^{-6}) mm before serialization.
**Event ordering and IDs**
After endpoint canonicalization, events are sorted by:
	1	midpoint (x,y,z);
	2	direction (d_x,d_y,d_z);
	3	segment length;
	4	charter class code.
All comparisons use the serialized (10^{-6}) mm values. IDs are then assigned sequentially as E0001, E0002, and so on.
This ordering is the sole tie-breaker. Annotation creation order and user-interface order must not affect output bytes.

**2.4 Primary annotation operation**
Each annotator independently performs the following:
	1	Load the raw scan under its random case ID.
	2	Apply CIBEC_V1_CHARTER.
	3	Create exactly one segment for every observable event satisfying the charter.
	4	Crop the segment to the observed contiguous physical span. No endpoint may be extrapolated from nominal CAD geometry.
	5	Assign one charter class code and visibility state.
	6	Record the fixed views in which the event was visually verified.
	7	Canonicalize, sort, and serialize.
Events not observable in the scan are not silently inferred from nominal part structure.
When an observable physical feature cannot be assigned one unambiguous event and one line under the frozen charter, the annotator must emit:
ONTOLOGY_UNRESOLVED
rather than selecting a convenient line.
When a required inspection event is inherently non-linear under the charter, the case must emit:
NONLINE_EVENT_CONFLICT
That condition is a failure of the universal bend-line objective, not an event to be omitted from scoring.

**2.5 Agreement and adjudication**
The two primary event sets are compared with the existing frozen 5°/4 mm/60% line matcher.
The matcher implementation and its SHA-256 are pinned before execution. Maximum-cardinality matching and all existing deterministic tie-breaking are retained unmodified.
The adjudicator receives:
	•	the raw scan;
	•	the frozen charter;
	•	both annotations;
	•	the matcher’s disagreement report.
The adjudicator still may not see CAD, model outputs, existing trusted truth, scores, filenames, or expected counts.
Every disagreement must be resolved by citing a specific charter rule. The adjudicator may select one annotation, replace both with a new line, merge events, split events, or issue an unresolved status—but may not modify the charter after seeing the case.
The final primary truth is then hashed and made immutable.

**2.6 Post-lock STEP topology correspondence**
Only after the primary hash is frozen, two independent CAD mappers receive:
	•	the frozen primary events;
	•	native STEP;
	•	OCP faces, edges, and adjacency;
	•	V9’s existing overlay transform and trust state.
For each event, the mapper identifies the complete native STEP face set representing that inspection event.
The operation is deliberately many-to-one:
	•	one STEP face may belong to at most one event;
	•	one event may own any positive number of STEP faces;
	•	several paired-cylinder primitives may map to one event;
	•	a scan event with no defensible STEP representation is SCAN_ONLY_EVENT;
	•	a CAD feature absent from observable scan truth is CAD_ONLY_UNOBSERVED;
	•	ambiguous ownership is CAD_MAPPING_UNRESOLVED.
The mapper may use:
	•	face adjacency;
	•	analytic surface type;
	•	coaxiality;
	•	shared boundary edges;
	•	tangency/continuity relationships;
	•	the existing registered overlay.
The mapper may not create, remove, move, split, or merge primary scan event lines.
Face indices are sorted numerically. Event records use the frozen primary event order. Two otherwise identical correspondence choices are resolved by the lower minimum face index, then lower maximum face index, then lexicographic sorted-face tuple.
**Registration abstention**
When the existing overlay is unavailable or untrusted:
{
  "overlay_status": "ABSTAIN",
  "reason": "OVERLAY_UNAVAILABLE_OR_UNTRUSTED",
  "event_correspondence": []
}
The primary scan truth remains valid. No CAD correspondence is fabricated.

**2.7 Output schemas
Primary scan truth**
{
  "schema": "SCAN_BEND_EVENTS_V1",
  "ontology_sha256": "<64 hex>",
  "renderer_sha256": "<64 hex>",
  "case_id": "<random 128-bit identifier>",
  "scan_sha256": "<64 hex>",
  "units": "mm",
  "coordinate_frame": "RAW_SCAN",
  "events": [
    {
      "event_id": "E0001",
      "class_code": "<charter code>",
      "line_start_mm": [0.0, 0.0, 0.0],
      "line_end_mm": [0.0, 0.0, 0.0],
      "length_mm": 0.0,
      "visibility": "FULL|PARTIAL",
      "render_admissible": true,
      "verified_view_ids": ["V1", "V2"],
      "adjudication_rule_id": "<charter rule>"
    }
  ],
  "case_status": "RESOLVED|ONTOLOGY_UNRESOLVED|NONLINE_EVENT_CONFLICT"
}
**Post-lock correspondence**
{
  "schema": "CAD_EVENT_CORRESPONDENCE_V1",
  "primary_truth_sha256": "<64 hex>",
  "step_sha256": "<64 hex>",
  "overlay_transform_sha256": "<64 hex>",
  "overlay_status": "TRUSTED|ABSTAIN",
  "events": [
    {
      "event_id": "E0001",
      "step_face_indices": [12, 14, 18],
      "v9_primitive_ids": ["B0004", "B0005"],
      "relation": "ONE_TO_ONE|MANY_TO_ONE|SCAN_ONLY_EVENT",
      "mapping_status": "RESOLVED|CAD_MAPPING_UNRESOLVED"
    }
  ],
  "cad_only_unobserved": [
    {
      "step_face_indices": [31, 32],
      "reason_code": "<charter code>"
    }
  ]
}
No raw points, thumbnails, existing predictions, scores, expected counts, filenames, or family identities may appear in either compact artifact.

**3. Source-free prerequisite suite**
The manufacturing authorities must sign the expected event identity and marker for every fixture **before** the annotation software is accepted.
The suite must contain exactly these 14 synthetic cases:
**ID
Synthetic condition
Required falsification target**
S01
Single 90° bend, positive native axis
One stable event and line
S02
S01 with negative native cylinder direction
Byte-identical canonical event after rigid equivalence
S03
One bend with paired inner/outer cylinders
Many CAD faces map to one event
S04
One transition represented by three tangent cylindrical radii
Charter’s compound-radius rule applied exactly
S05
Two distinct parallel bends separated by 6 mm
They remain distinct
S06
Two coaxial bends with non-overlapping spans
Charter’s interruption rule applied exactly
S07
Return lip
Exact charter count and marker
S08
Open hem
Exact charter count and marker
S09
Closed hem
Exact charter count and marker
S10
Raised bead
Exact charter count and marker
S11
Rolled edge
Exact charter count and marker
S12
Compound rounded transition
Exact charter count and marker
S13
Partial scan exposing only one contiguous portion
No unobserved endpoint extrapolation
S14
Valid scan truth with absent/untrusted overlay
Primary truth retained; CAD map abstains
The suite must additionally perform:
	•	translation invariance at (+1000,-2000,+3000) mm;
	•	rotations of 90°, 180°, and 270° about each Cartesian axis;
	•	endpoint reversal;
	•	face-index permutation;
	•	input record permutation;
	•	duplicate-face ownership injection;
	•	NaN, infinity, zero-length line, non-mm units, and malformed-transform rejection;
	•	ten repeated serializations per fixture.
**Prerequisite pass gate**
	•	14/14 exact expected semantic outputs;
	•	14/14 byte-identical ten-run replay;
	•	100% rejection of malformed fixtures;
	•	zero duplicate face ownership;
	•	zero source-point serialization;
	•	zero unresolved outcomes.
Any failure retires the implementation before physical annotation.

**4. Locked physical execution and custody protocol
4.1 Algorithmic execution status**
The physical qualification is **development-only**.
Exactly the existing 13 development-consumed cases may be used for:
	•	annotation reproducibility;
	•	ontology adjudication;
	•	CAD-face-to-event correspondence;
	•	visual qualification of the contract.
No detector thresholds may be changed, and no new detector score is used for selection during this action.
**4.2 New physical scans**
New scans may be opened **only inside an independent annotation custody lane**, and only after CIBEC_V1 passes all source-free and 13-case qualification gates.
They may not be opened by:
	•	detector developers;
	•	model-selection personnel;
	•	inference processes;
	•	A0, incumbent, V9, or multisource pipelines;
	•	anyone with access to current prediction overlays during annotation.
Before any new geometry is opened, a custodian must verify from metadata alone that the pre-enrolled collection can reach at least:
	•	30 scans;
	•	15 families.
If the inventory cannot satisfy both counts, return:
CORPUS_CAPACITY_ABSTAIN
and open none of the new scans.
When sufficient inventory exists:
	1	Freeze the complete case list and family-stratification manifest.
	2	Replace all names with random case IDs.
	3	Hash every raw scan before annotation.
	4	Annotate every case using the same two-annotator-plus-adjudicator process.
	5	Freeze and encrypt the primary truth.
	6	Keep scans, truth, event counts, family mapping, and CAD correspondence inaccessible to detector development.
	7	Run **no inference on the new corpus in this action**.
This creates an independently annotated, sequestered promotion corpus without consuming it for the present architecture decision.

**5. Qualification gates
5.1 Numerical annotation gates on the 13 consumed cases**
All of the following are mandatory before adjudication unless otherwise stated:
**Gate
Required value**
Exact event-count agreement
at least 12/13 cases
Inter-annotator line precision
≥ 0.90
Inter-annotator line recall
≥ 0.90
Inter-annotator line F1
≥ 0.90
Median matched-line transverse disagreement
≤ 1.5 mm
P95 matched-line transverse disagreement
≤ 3.0 mm
P95 matched-line angle disagreement
≤ 2.0°
Exact class-code agreement
≥ 0.90
Post-adjudication unresolved cases
0
Post-adjudication non-line conflicts
0
Final duplicate event identities
0
Final duplicate STEP-face ownership
0
For the two independent CAD correspondence maps, compute a pairwise grouping classification over all unordered pairs of native STEP bend primitives:
	•	positive when both primitives map to the same inspection event;
	•	negative otherwise.
Required grouping precision, recall, and F1 are each at least 0.90.
The registration-abstaining 10839000 case is excluded only from CAD correspondence agreement. Its primary scan annotation remains mandatory.
**5.2 Visual gates**
Every one of the 13 final annotations must receive native-resolution approval from the adjudicator in all applicable fixed views.
The audit must explicitly verify:
	•	no single physical event is represented more than once;
	•	no two distinct events are merged;
	•	every line follows the frozen canonical-line convention;
	•	partial-scan endpoints terminate at observed evidence;
	•	no marker crosses unsupported empty interior unless the charter explicitly defines a virtual line and the same rule was fixed synthetically;
	•	40500000, 47959001, and 15651004 each receive a written charter-rule justification;
	•	every lip, hem, bead, rolled section, and compound transition is either consistently represented or causes the capability to fail.
Visual verdict must be **PASS**, not “pass with limitations.”
**5.3 Determinism and reliability gates**
	•	Primary and replay JSON artifacts byte-identical.
	•	CAD correspondence primary and replay byte-identical.
	•	10/10 repeated exports identical on every synthetic fixture.
	•	Physical annotation-tool job success: 1.000.
	•	Schema validation success: 1.000.
	•	Hash-manifest verification: 1.000.
	•	Leakage audit: zero prohibited fields or assets in the primary workspace.
**5.4 Latency and memory gates**
Human annotation time is recorded but is not substituted for detector latency.
For all machine operations excluding human interaction:
	•	per-case annotation export P95 ≤ 5 seconds;
	•	per-case OCP correspondence-load and export P95 ≤ 30 seconds;
	•	peak RSS < 2 GiB;
	•	no source-point arrays in compact outputs;
	•	no modification to the existing production inference limit of 300 seconds or 16 GiB.
A failure of these tool limits retires the implementation, not the ontology.

**6. Retain and retire logic
Immediate status
Retain**
The **0.4580 incumbent—scan plus scan-supported legacy CAD—remains the retained balanced CAD-assisted operator-support lane**.
Also retain as infrastructure:
	•	the corrected V8 signed-axis implementation;
	•	existing automatic overlay and exported transform;
	•	OCP native topology loading;
	•	controlled overlay abstention;
	•	deterministic replay and compact evidence contracts.
**Retire from autonomous contention**
	•	V9 targeted-only;
	•	V9 fusion;
	•	the three-source high-recall union;
	•	any interpretation of a normal-verified topology axis as an autonomous inspection event.
The three-source union may remain available only as an explicitly selected human “show all candidates” review tier. It is not the default answer and cannot drive automatic disposition.
**Relation to the 0.4580 score**
CIBEC_V1 does not challenge the incumbent’s F1. It defines the missing target capability.
Therefore:
	•	no current score is recomputed;
	•	no historical truth is silently rewritten;
	•	no V9 score is rescued by re-annotation;
	•	the incumbent stays retained at 0.4580 on the existing consumed slice;
	•	no architecture is promoted during this action.
**Capability retain condition**
Retain CIBEC_V1 only if every prerequisite, numerical, visual, determinism, custody, and failure gate passes exactly once.
**Capability failure condition**
Issue an immediate **FORMAL_STOP on the universal scalar bend-line objective** if any of the following occurs:
	•	fewer than 12/13 cases have independent exact-count agreement;
	•	inter-annotator line precision or recall is below 0.90;
	•	any physical case remains ONTOLOGY_UNRESOLVED;
	•	any legitimate required event remains NONLINE_EVENT_CONFLICT;
	•	CAD primitive grouping agreement is below 0.90 precision or recall;
	•	the charter must be changed after physical exposure to resolve a case;
	•	a model output, expected count, CAD representation, filename, or prior truth reaches a primary annotator.
After such a failure, no further geometry clustering, virtual-sharp substitution, threshold search, or topology rescue is authorized under the current bend-line claim.

**7. Lane classification and separation**
CIBEC_V1 belongs to **both lanes as evaluation infrastructure only; it belongs to neither lane as inference input**.
The partition is strict:
**Scan-only lane**
May later be evaluated only against SCAN_BEND_EVENTS_V1.
It may never receive:
	•	CAD correspondence;
	•	STEP face IDs;
	•	CAD-derived event grouping;
	•	overlay transforms;
	•	native feature classes;
	•	expected event counts.
Its zero-shot scan-only claim is therefore unchanged.
**CAD-assisted operator-support lane**
May use CAD_EVENT_CORRESPONDENCE_V1 to:
	•	group several native faces or axes into one displayed event;
	•	explain which CAD surfaces support a suggestion;
	•	expose CAD-only unobserved features;
	•	present registration abstention.
It remains explicitly outside the scan-only promotion claim.
The primary scan truth is frozen before any CAD mapping, and the CAD artifact is structurally incapable of changing it. This prevents CAD-assisted semantics from contaminating scan-only truth or model selection.

**8. Why no bounded geometry intervention is authorized**
A bounded “virtual sharp,” flange-intersection, face-adjacency clustering, or cylinder-family consolidation rule is technically implementable. It is not yet scientifically identifiable.
Such a rule would have to assume, without an authoritative contract:
	•	which surfaces constitute one event;
	•	which line convention is correct;
	•	whether a return, bead, or hem is counted;
	•	where a partial event begins and ends;
	•	when two adjacent transitions merge.
Those are precisely the variables producing the strongest current contradictions. The same-count/zero-localization cases show that improving enumeration or support alone cannot distinguish a geometry failure from a truth-convention failure.
CIBEC_V1 is therefore the smallest necessary opening. It does not relax a gate, add a sensor, expose held-out truth to inference, or create a benchmark-feedback loop. It makes the target falsifiable before another geometry mechanism is funded.
**Authorized work is limited to CIBEC_V1. No V10 detector intervention, threshold change, truth rewrite, or new-corpus inference is authorized.**

