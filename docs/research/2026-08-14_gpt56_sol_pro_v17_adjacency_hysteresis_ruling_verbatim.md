1. RULING

AUTHORIZE_MESH_ADJACENCY_AXIAL_HYSTERESIS_RECALL_V17

Authorize one append-only, post-selector recall primitive for the native full-resolution V11 lane. It may restore a rejected scan_mesh_adjacency candidate only when weak local transition evidence is backed by a long, well-localized, nearly continuous, two-sided axial feature.

No generator, feature computation, route decision, incumbent selection, or bounded-pool lane changes.

2. RESEARCH BASIS

Pauly, Keiser, and Gross extract line-type features from point samples by combining local geometric classification with hysteresis and graph-level line formation; their multiscale treatment is specifically intended to make feature detection more robust to noise. The relevant adaptation here is not their graph construction, but their principle that a locally weaker feature response can be retained when it belongs to a sustained line structure. 
Eurographics Digital Library

Demarsin et al. similarly begin with sharp-feature candidates derived from estimated normals and then recover feature lines through graph structure. V17 adapts the distinction between an isolated normal discontinuity and a coherent feature line, while reusing the already-emitted V11 normal-angle, side-support, coherence, and axial-run fields rather than constructing another graph. 
ScienceDirect

Canny’s established criteria separate detection, localization, and the requirement for a single response to one edge; the paper also reports improved detection when support is extended along the edge. V17 adapts those ideas as an axial extent-versus-localization certificate and incumbent-first duplicate suppression. It does not port the 2D detector itself. 
ScienceDirect
+1

The inequalities below are Sherman-specific engineering hypotheses, not constants reported by those papers. They are authorized as one preregistered configuration only: no sweep and no adjustment after either regression result.

3. EXACT PREDICATE
Insertion point

Implement inside the existing V11 native full-resolution selector, not inside V16 routing:

Run V11 unchanged through its complete current selector and obtain:

the final incumbent accepted list S;

the original rejected candidate records R.

Apply the predicate below only to rejected raw scan_mesh_adjacency records.

Suppress eligible rescues against S and against earlier rescues.

Append surviving rescues to S.

Continue through the existing V11 output formatting, hashing, and serialization.

V16 remains route ownership only. Bounded-small, bounded-complex, and V6 paths must never enter this branch.

Eligibility

For rejected candidate c, define:

Q(c)=
p90_line_distance_mm
span_mm
	​

.

Use the serialized binary64 source-feature values directly. Do not recompute them from points. Missing, NaN, infinite, or nonpositive p90_line_distance_mm means immediate rejection.

ELIGIBLE(c) iff all conditions hold:


candidate_source == "scan_mesh_adjacency"


c was rejected by the unchanged current V11 final selector


finite_support_fraction >= 0.90


finite_support_longest_run_fraction >= 0.90


endpoint_support_fraction >= (2.0 / 3.0)


0.0 < p90_line_distance_mm <= 3.0


span_mm >= 10.0 * p90_line_distance_mm


two_sided_transition_score >= 0.25


two_sided_normal_angle_deg >= 45.0


two_sided_support_balance >= 0.50


two_sided_min_support_fraction >= 0.20


two_sided_normal_coherence >= 0.50

Evaluate the conjunction in that order, with ordinary binary64 comparisons and no epsilon branches.

The operative new recall mechanism is the conditional hysteresis:

two_sided_transition_score >= 0.25 is only a weak local transition floor;

the candidate is accepted only when axial support is continuous and its supported length is at least ten times its 90th-percentile localization error.

No confidence, learned selector probability, case percentile, candidate count, source count, expected-count logic, part identity, or rank among the scan’s candidates may enter eligibility.

On the supplied C017 source values:

SMA1 has Q=32.59/1.787≈18.24;

SMA3 has Q=32.60/1.389≈23.47.

Both therefore clear the authorized scale-separation gate, as well as the other source-only gates. Their gold alignment is irrelevant to predicate execution and is used only later for development scoring.

Duplicate relation and NMS

Do not define a second angle/offset/overlap relation and do not import the 5-degree/4-mm evaluation match contract.

Use the exact final V11 duplicate predicate currently called by the full-mesh NMS, byte-for-byte and with all of its existing constants unchanged. Hash that function or enclosing source region before and after implementation; any change invalidates the experiment.

NMS is incumbent-first:

Pre-seed the retained set with every V11 incumbent in its existing order.

Sort eligible rescues by the following key, descending except for the final ID:

Q
two_sided_transition_score
finite_support_longest_run_fraction
endpoint_support_fraction
two_sided_normal_coherence
two_sided_min_support_fraction
two_sided_support_balance
span_mm
candidate_id ascending as the final tie-break

Traverse once. Reject a rescue if the frozen V11 duplicate predicate returns true against any incumbent or previously retained rescue.

Otherwise append it unchanged.

Incumbents always win a duplicate conflict, regardless of their ranking values.

Apply no top-k limit and no per-scan rescue-count cap.

The candidate’s anchor, axis, endpoints, span, and support record remain byte-identical to the emitted source candidate. V17 is a selection-only intervention.

4. WHY IT MAY GENERALIZE

The mechanism distinguishes extended line evidence from a locally plausible adjacency response:

span_mm / p90_line_distance_mm measures how far the feature persists relative to its own localization uncertainty. It is dimensionless and invariant to rigid transformations and uniform unit scaling.

The finite-run and endpoint gates require support over the candidate’s axial extent, rather than merely near its midpoint.

The normal-angle, balance, minimum-side, and coherence gates require an actual two-sided surface transition rather than a one-sided ridge response.

The transition threshold is intentionally weak because it is conditioned on the stronger axial certificate. This is the hysteresis component.

Incumbent-first NMS preserves V11’s current precision decisions and prevents a newly eligible adjacency representation from reviving an existing alias.

This is materially different from lowering the V11 score threshold. The intervention uses a conjunction of localization, persistence, and two-sided geometry and cannot rescue a short candidate merely because it has a high normal jump or classifier score.

It is also consistent with the earlier archived diagnosis that candidate-local evidence could retain corner residuals and that trustworthy selection needed maximal axis support, endpoint behavior, and explicit duplicate geometry. 

01_prior_professor_response

The C017 values give meaningful, but not razor-thin, margin on the principal new quantity: approximately 18.24 and 23.47 against a gate of 10. The weaker SMA3 transition value of 0.312 is accepted only because its line is continuous, accurately localized, two-sided, and more than twenty-three localization widths long.

5. FALSIFICATION GATES
Development regression

Run exactly once on the frozen 35-scan corpus after implementation tests pass. All of the following are mandatory:

Every previously matched V16 TP remains present with identical geometry. No scan may lose a previously matched TP.

Every newly appended rescue must score as a TP. No added rescue may be an FP.

C017 must become exactly:

2 TP;

0 FP;

0 FN.

Pooled development result must satisfy:

TP at least 134;

FP no more than 60;

FN no more than 42;

F1 at least 0.724324.

Minimum-family recall and minimum-family F1 must both become strictly greater than 0.

Exact cases must be at least 5/35.

No previously exact case may cease to be exact.

Failure of any condition is architecture STOP for this primitive. In particular, recovering both C017 axes while introducing even one new development FP fails the intervention.

Locked U02–U05 regression

The complete serialized V17 outputs for U02, U03, U04, and U05 must be byte-identical to their frozen V16 outputs.

This deliberately requires V17 to be a no-op on U02’s already-clean full-mesh result. The aggregate must remain exactly 15 TP / 0 FP / 0 FN. U03–U05 must remain routed to V15 without invoking the new predicate.

Fresh held-out confirmation

After the development result passes:

Freeze code, constants, dependency lock, source hashes, and validator hashes.

Precommit an ordered queue of untouched native full-resolution scans by raw-source SHA-256.

Run source-only inference in that fixed order and select the first scan on which V17 appends at least one rescue.

Open truth only for that scan.

The confirmation passes only when:

at least one rescue fired;

every rescue is a TP;

zero rescue FPs occur;

every incumbent TP remains;

the native-resolution audit confirms the counted status of every added axis.

If the precommitted queue contains no activation, the result is non-confirmatory and this intervention remains stopped rather than being promoted on a safety-only no-op.

Determinism and contracts

Mandatory:

ordinary execution and a candidate-order-permuted execution serialize byte-identically;

the independent validator reconstructs the same eligibility decisions, ranking, incumbent-first NMS, and final bytes;

candidate-generation artifacts and feature-schema hashes remain unchanged;

source_point_serialization remains false;

no source coordinates, point subsets, or new support arrays enter caches;

output axes use the existing row schema only;

process-tree peak RSS remains below 16 GiB;

end-to-end P95 remains at or below 300 seconds.

No failed gate may be followed by changing 10.0, 0.25, any continuity or normal threshold, NMS priority, or duplicate semantics. A revised predicate would require a new architecture identity and new authorization.

6. EXPECTED FAILURE MODE

The most likely false positive is a long, accurately scanned exterior perimeter, opening boundary, thickness edge, or welded corner seam. Such an object can have:

high axial extent-to-error ratio;

excellent finite and endpoint support;

a substantial normal discontinuity;

apparently two-sided coherent support;

while still not being a counted bend.

This predicate has no new endpoint-topology or panel-ownership feature, so it cannot prove that the line belongs to an independent fold rather than closing a corner or tracing an opening.

The mandatory visual audit is one native-resolution endpoint-ownership overlay package for every appended axis:

four orthographic views;

rescued axis and both endpoints shown without shortening;

the two incident mesh surfaces colored separately;

existing V11 axes shown simultaneously.

The audit fails if the rescued line terminates on the outer silhouette or an opening, follows a thickness face, closes a cycle between already-explained wall/base bends, or occupies a terminal corner instead of a visible material fold. One such rescue is sufficient for terminal STOP; it must not trigger a boundary-specific exception or threshold rescue.

Sources
