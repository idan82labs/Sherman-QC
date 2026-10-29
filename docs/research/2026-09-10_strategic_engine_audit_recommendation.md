# Strategic engine audit and conditional recommendation

Date: 2026-09-10. Root synthesis with independent retained-result review by
`production_architecture_review` and independent source/math review by
`cad_support_lineage`. This is an AI technical audit, not human manufacturing
approval. No new inference, scorer execution or physical access in this review.
Requested browser GPT Astra6Pro consultation is still PENDING model choice;
these conclusions must not be attributed to that consultant.

## Bottom line

There is a justified case for changing the registration/evidence architecture,
but not yet for an unbounded rewrite or a claim that a novel method is necessary.
The immediate failure is a new engine path that cannot deliver a usable pose,
not demonstrated learned undertraining or inaccurate bend measurements.
The historically useful hybrid and the newer conservative pipeline are different
systems. Component qualifications on the newer path have not translated into
physical output. We should measure connected output and bottleneck removal,
not use passing-test counts as the primary indicator of roadmap progress.

The most promising conditional direction is **multi-hypothesis finite-surface
registration with separate proximity, correspondence, pose and measurement
eligibility**. Existing code already names these concepts honestly; the issue
is operational coupling of their gates, not simply missing status labels.
No gate should be removed merely to get lines. Change its mathematical evidence
contract only when independently constructed tests establish the replacement.

## Family evidence: what actually explains the numbers

The independent reviewer checked retained `SCORER_REPORT.json` SHA256
`6307f41ef3677a822e69969ba8da03b3ce5664375fbfc6ec642f52de6cb956b3`.
The table below is arithmetic from the already-scored report, not a rerun.

| Family | Cases / refs | UNION TP / FP / FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
|10839000|3 /18|4 /13 /14|.235294|.222222|.228571|
|37361005|1 /4|2 /1 /2|.666667|.500000|.571429|
|38172000|4 /34|18 /10 /16|.642857|.529412|.580645|
|40500000|1 /7|3 /3 /4|.500000|.428571|.461538|
|49001000|1 /5|1 /2 /4|.333333|.200000|.250000|
|49024000|1 /3|2 /1 /1|.666667|.666667|.666667|
|Total|11 /71|30 /30 /41|.500000|.422535|.458015|

Family38172000 supplies18/30TP, but within-family case F1 spans .142857 to
.761905. Its CG1E0BBA2933 and CGF41F49066A alone supply15/30TP. Modern V3
CPU-kills the first and completes finite screening with zero starts on the
second. This is not a failure confined to historically difficult families.
Family10839000 supplies13/30FP, yet its CGE4B0549E41 has2TP/0FP/2FN; V3 stops
that case at membership admission. A family label is not a causal explanation.

Fusion also has unequal effects. Relative to A0, UNION adds net11TP and16FP:
family38172000 adds7TP/6FP,10839000 adds2TP/8FP,40500000 adds0TP/1FP.
These aggregate differences do not prove individual row attribution. Agreement
filtering is not a general remedy: intersection retains only10of UNION's30TP.

Four families have only one case; all11cases are consumed development data.
There is no authorized evidence of learned training curves, model-weight
comparisons or unseen-family evaluation proving overtraining. There IS direct
evidence of an under-representative engineering validation matrix: earlier
connected .005mm/tangent tests did not represent physical .05mm/free-circle;
large capacity controls did not exercise accepted overlapping closures. Repeated
adaptation to familiar synthetic constructions can itself overfit development
tests, but its magnitude is not measured here.

## Failure factors, ranked by evidence and leverage

| Factor | Evidence | Interpretation |
| --- | --- | --- |
| No jointly supported pose |6/11cases,45/71refs;18,541poses,0starts|Highest currently localized physical blocker. Wrong/missing hypotheses versus false rejection still unresolved.|
| Unobserved CPU bottleneck |3/11cases,19/71refs terminate at150s with no RESULT|Need bounded stage/work telemetry. Do not attribute time to a stage from another case or infer host RAM shortage.|
| Thin-part proposal limitation |Exact physical-config construction0/2; only two accepted normal directions|Three-plane initialization cannot cover this admitted geometry; a two-plane seed alone does not solve downstream pose observability.|
| Coupled proximity/ambiguity gates |Source excludes uncertain closest-point correspondences from finite support; new noisy50-row positive fails|Specific seam/tessellation hypothesis is plausible, not established without query evidence.|
| Native feature coverage |Three control cases contain15refs but zero admitted ROIs|An overlapping downstream ceiling, not the cause of every zero-output case. Cannot achieve .90recall while these15remain unsupported.|
| Expanded membership load |1/11case;19.52M of20M memberships already charged|Storage, logical work and decode work need separate design; deduplication already exists.|
| Evidence publication |44extra V3partials omitted from manifest|Blocks formal release separately; does not explain geometric zero output.|

The latest initializer negative remains terminal40PASS/1FAIL. Its traceback
does not retain the decisive per-query evidence. Future diagnostic runs should
retain bounded stage evidence from the FIRST call, including a failing positive,
so diagnosis does not require repeating an opaque test. Pure helper success and
recording-test success must stay separate from actual positive output.

## Recommended architecture questions and rejection tests

### 1. Finite surface support independent of a unique closest triangle

A complete finite-surface distance bound U(p,F)<=gate can establish proximity to
the represented finite triangle union F even when more than one closest-point/
normal interpretation remains possible. It does not establish physical-face
identity. The current [support consumer](../../backend/cad_guided_global_registration_v3.py)
intersects definite proximity with ambiguity exclusions before usable support;
the distance-only `_support_masks` function itself does not apply ambiguity.
[Triangle tie logic](../../backend/cad_guided_triangle_surface_v3.py) lines558–565
can mark uncertain point coincidence even for equal oriented normals. An internal
tessellation diagonal should not by itself change membership in an unchanged
represented union; different curved-surface approximations need not be invariant.

Propose a separately typed proposal-support contract retaining set-valued
ambiguity. Do not pick an arbitrary nearest point for optimization, suppress
flags, merge disconnected sheets, fill holes or relax final measurement gates.
First falsifier: fixed rectangle geometry, off-plane noisy observations, opposite
diagonals/subdivision/face-ID permutations, then holes, folds, duplicate sheets
and genuinely separated near surfaces. Compare membership and uncertainty
semantics, not just the number of passing rows. The pending unchanged50-row
diagnostic must determine whether this mechanism explains the actual failure.

### 2. Preserve residual pose families and evaluate real finite information

For two nonparallel normals n1,n2, let a=n1×n2. A point-to-plane Jacobian row
J=[p×n,n] satisfies J[0,a]=n·a=0 when n lies in span(n1,n2). Such rows alone
cannot constrain axial translation.
Current V3 uses three face-interior rank checks; boundaries and uncertain regions
are excluded. V11 adds finitely many axial seeds but does not prove the family
exhausted or uniquely resolved. Merely adding seeds cannot repair a structurally
deficient normal-only rank test. Actual full-raw queries may include end-cap or
other normals absent from the proposal bank, and the finite objective may carry
boundary information; neither possibility is ruled out by this two-normal proof.

Propose a bounded re-queried finite-surface objective/coverage profile along the
unresolved family, plus scan-supported manufactured boundary/hole evidence where
available. Test isolated minima versus plateaus and separated equal modes. A
scan crop edge is not a part edge; partial coverage/trim policies can leave a
plateau even when CAD is finite. Do not use fixed point-to-point correspondences
as artificial rank, and do not call an isolated best-fit minimum a datum.
Falsifiers: complete/cropped extrusions, repetitive features, asymmetric holes,
nearly parallel normals, unequal sampling, and true geometry deviations.

### 3. Measurement-specific identifiability, not blanket rescue

For a retained feasible pose set T, a measured quantity q can only be reported
as stable if its scan-derived range across the admissible hypotheses is within
its declared model/error limit. A radius or included angle may remain stable
under axial uncertainty while endpoints, bend identity, signed displacement and
finite ROI membership do not. This is a design hypothesis, not a calibrated
confidence interval. All relevant hypotheses/support sets must be represented;
eight axial samples alone do not bound a continuous family.

Falsifiers: constant versus varying cross-section, short/partial arcs, repeated
bends, reversed material orientation, shifted ROI support and missing bends.
Retain the raw rows behind every quantity. Withhold uncertain line localization,
observed count and conformance even if an invariant radius is available. This
direction may improve useful metrology without improving Gold line F1; report
those outcomes separately rather than using partial measurements as fake lines.

### Alternative coarse solver: challenger, not presumed replacement

A robust feature-correspondence proposal based on
[TEASER's truncated least-squares formulation](https://arxiv.org/abs/2001.07715)
is a reasonable comparison candidate if independently derived geometric
correspondences exist. It does not resolve repeated sheet-metal features,
wrong correspondences, unit assumptions or metrology truth automatically.
[Go-ICP](https://arxiv.org/abs/1605.03344) illustrates bounded global search;
optimality for its stated objective is not physical correctness, and whole-part
search cost must be qualified. These are research anchors, not installed solvers
or transferred performance claims. Compare a minimal proposal adapter against
the current lane before committing to a broad architecture rewrite.

[Bonnabel et al.](https://doi.org/10.1109/ACC.2016.7526532) specifically caution
that ICP rematching affects covariance. Any finite-boundary or nuisance-pose
design needs its own assumptions and uncertainty analysis; changing the
Jacobian to obtain full rank does not create information.

## Decisive next sequence, conditional on consultation and independent PRE

1. Complete the requested model consultation using this evidence, not an
   unsupported claim of overtraining. Ask for a ranked alternative and explicit
   tests that would reject it. Independently check the mathematics/references.
2. Execute a NEW frozen bounded diagnostic of the original noisy failure with
   full stage retention and unchanged inputs/policies. No fix or score attached
   to this diagnostic. Current draft is unreviewed/unexecuted.
3. Freeze one narrow semantic repair if the mechanism is demonstrated, plus
   independent retessellation/genuine-ambiguity falsifiers. In parallel design
   the smallest bounded alternative proposal comparison; no physical access.
4. Demand a connected positive under EXACT .05mm/free-circle configuration,
   independently posed/noisy/partial/ambiguous and genuinely deviated cases.
   Report pose coverage, measured outputs, error, abstentions and cost, not a
   total of unrelated unit tests. If this gate fails, retain it and reassess the
   mechanism before expanding implementation.
5. Qualify completed runner custody and stage telemetry, then independently
   freeze a successor all11same-input comparison. Preserve the cached UNION
   control and71refs/scorer unchanged; include every failure. Diagnose new lines
   by geometric strata, not family-ID filters. Integrate only proven results.

Proportionate prospective engineering budget for the existing stage-diagnostic
draft is60wall/45CPU/1GiB,1MiBstdout and2×16MiB artifacts. This review does NOT
release it or other runs. Other experiments require their own finite limits and
PRE before execution. Keep one heavy job at a time and repository RSS limits.

The cached score is still .4580152672. A slight numerical gain is not the locked
destination:64TP and<=7FP at64TP are needed for .90P/R, minimum-family recall
must reach .80, and exact observed count .95requires11/11on this cohort. Missing
physical measurement truth, uncertainty calibration and datum evidence remain
separate limitations. No unseen-family or factory-floor release follows.

## Evidence links

- [Cached result and provenance limits](2026-09-10_gold11_cached_control_v2_result.md)
- [V3 dispositions and separate custody failure](2026-09-10_gold11_mesh_development_v3_result.md)
- [Earlier complete failure-factor audit](2026-09-10_gold11_failure_factor_audit.md)
- [V11 retained negative POST](2026-09-10_two_plane_v11_initializer_post_run_audit.md)
- [Consultation evidence and questions](2026-09-10_strategic_engine_consultation_brief.md)

Runtime source and frozen attempts were not changed by this audit. Source-only
V4 runner and diagnostic drafts are paused, unqualified, and not promoted.

Independent synthesis review by `cad_support_lineage` is GREEN-with-limitations
for these two documents only. It checked mathematics and inference boundaries,
not the physical artifacts or external browser identity; its three precision
corrections on represented unions, normal-only rank and consumer attribution
are incorporated. No experiment, redesign or promotion is released.

## Subsequent evidence narrows recommendation1

The [V11 stage diagnostic](2026-09-10_two_plane_v11_stage_diagnostic_result.md)
has now executed under a new exact freeze without changing the engine. It
confirms that the correct pose was generated in the noisy50-row construction,
and rejected only after uncertainty exclusion reduced25/25distance-supported
rows to20/25and21/25on its two faces. Shared-diagonal closest-point uncertainty
is observed; the initial5/5row guess refines to5/4. The different selected-bank
and missing-correct-proposal hypotheses are ruled out for this control.

This strengthens the case for reviewing proximity versus correspondence
semantics. It still does not qualify a replacement, full registration, native
measurements or physical gain. Earlier hypothetical wording retains chronology;
the new result and its exact audit state supersede only this mechanism's
evidence level. The requested Astra consultation remains pending model choice.
