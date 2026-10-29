# Strategic CAD-assisted engine audit: consultation brief

Date: 2026-09-10. Status: evidence brief and open design questions, NOT a
consultant response, executed experiment, approved redesign or promotion.

## Request and boundaries

The user requests an independent strategic review: explain the success/failure
patterns, distinguish family bias from mathematical/architectural limitations,
and recommend a material change if justified. Requested consultant: GPT Astra 6
Pro through browser or Codex side chat. Browser inspection found Work's explicit
GPT-6 Astra model with six reasoning levels through Ultra, but no Pro-labelled
option. Chat's explicit model menu lists Latest, GPT-5.6 Sol and GPT-5.5. Codex
tools list gpt-6-astra, not Pro; native UI access was denied by the computer-use
surface and was not bypassed. No consultation has yet been sent or received.
The user was asked whether the verified Astra Ultra option is acceptable.

Only the consumed Gold11 cohort is authorized: eleven scans, six CADs and the
unchanged71reference lines bound by September9 hashes. No new email parts,
held-outs, unrelated data, hardware, external deployment, people or paid API.
Review uses retained released results and source, not new inference. Preserve
all terminal attempts and negatives. Gold stays outside inference, registration,
datum and carrier selection. CAD supplies correspondence/nominals, never the
measured answer or observed count. Common-mm/whole-part best fit are provisional,
not scanner calibration or verified manufacturing datums.

## The evidence the recommendation must explain

1. Recovered cached UNION is30TP/30FP/41FN, F1=60/131=.4580152671755725.
   Scan-only A0 is19TP/14FP/52FN; gated CAD20/24/51; intersection10/7/61.
   UNION gains net11TP and16FP over A0. Simple agreement filtering loses20of
   UNION's30TP. Numerical retained-byte reproduction is independently audited;
   original runtime/invocation/reference-history attestation remains incomplete.
   This is a nonzero cached control, not fresh historical-engine execution.
2. Fresh physical V3 compares newer V8/V10 paths, not the historical hybrid.
   Both emit zero lines on all11cases:0TP/0FP/71FN, recall0, undefined P/F1.
   Challenger's earliest failures partition the cohort:6cases/45refs complete
   finite screening but yield no start;3/19 CPU-killed at150s with internal
   stages UNKNOWN;1/4 exceeds expanded membership admission;1/3 has no scheduled
   triplet pose. Measurement accuracy was not reached, so these are not proved
   radius/angle/localization errors or evidence of absent physical bends.
3. Six completed searches propose18,541poses and test55,623assigned supports:
   54,951box exclusions,672full queries,3,340,144certificate pairs, zero starts.
   These aggregate counters are independently verified. They do not show the
   correct physical pose was proposed or incorrectly rejected. A root-only
   finer histogram reports17,918poses with0passing supports,610with1,13with2,
   none with3; this finer result has not been independently recomputed.
4. A native constructed thin part, with the EXACT physical .05mm/free-circle
   configuration, has0/2positive outcomes. Clean/noisy banks have27planes and
   reject48/48triplets. The clean bank has only two distinct normal directions;
   its narrow third-direction cap has conservative normal cone17.83degrees,
   exceeding10degrees. Geometry is present but rejected as a usable normal.
   Earlier connected positives used .005mm/tangent-constrained measurement,
   so passing them did not qualify this operating range.
5. A new two-plane/context lane qualifies pure context110checks and scheduler68.
   But its first connected initializer gate is40PASS/1FAIL: the original broad
   clean50-row control produces a start; the noisy <=.002mm counterpart does
   not, despite a .05mm declared bound. No timeout or46-file drift. The original
   failed gate retained only a traceback, not full per-query evidence; cause
   below completed-prefilter/no-start remains UNKNOWN. Workflow/native gates
   remain unexecuted and blocked. Pure helper passes are not output progress.
6. Three native-control cases contain15refs but no admitted ROIs under global
   solid-validity policy. This overlaps earlier failures. If those15remain
   unsupported, recall ceiling56/71=.7887 and zero-FP F1 ceiling112/127=.8819
   are below the locked target. Blindly ignoring shell validity can invert
   material side and signed geometry; local admission needs explicit premises.
7. The membership failure charges19,518,222 of20M expanded memberships, with
   837,407raw rows and1,666,083triangles. Deduplication already precedes the
   charge. A900k capacity pass rejected all16closures: it did not qualify
   accepted overlapping closure load. Host128GBRAM does not explain or fix this
   logical-work cap. CPU-killed stages need retained profiling, not guesses.
8. V3 also has a separate publication-custody defect:44extra partial files
   outside a116-file manifest; full160 exceeds its128scientific-entry cap.
   Canonical negative records are intact; formal release is blocked. This
   defect is not the cause of zero geometry. A helper114PASS qualifies only its
   owned-directory scope; the new V4 runner draft is unfinished and untested.

Sources: [cached control](2026-09-10_gold11_cached_control_v2_result.md),
[physical V3](2026-09-10_gold11_mesh_development_v3_result.md),
[failure-factor audit](2026-09-10_gold11_failure_factor_audit.md),
[exact-config native](2026-09-10_mesh_physical_config_constructed_v9_result.md),
[V11 negative POST](2026-09-10_two_plane_v11_initializer_post_run_audit.md).

## Architecture questions for the consultant

Please challenge the current design, not merely endorse its incremental plan.
Separate proved facts, source-derived hypotheses and missing evidence. You have
this brief, not the repository or physical geometry; do not imply direct access.

1. Is the major problem a brittle, serial all-or-nothing registration gate?
   Distinguish proposal generation, finite-distance screening, correspondence
   ambiguity, global pose observability, per-bend measurement identifiability,
   datum validity and final localized line eligibility. Which should be separate
   typed outputs rather than one acceptance bit?
2. For two nonparallel normals n1,n2, translation along a=n1×n2 is unobservable
   in infinite-plane point-to-plane residuals whose normals are in their span.
   With J=[p×n,n], J[0,a]=0 for those rows. Other normals in full-raw queries
   may add information even when absent from the proposal bank. Current
   V3 rank uses only unambiguous face-interior rows, excluding region-uncertain
   boundaries. Adding two-plane pose seeds does not itself resolve this rank
   condition. Can genuine finite boundaries/holes supply scan-supported axial
   evidence? How must occlusion/acquisition cut edges be distinguished from
   manufactured boundaries? Do NOT substitute fixed point-to-point matches as
   an artificial full-rank certificate or infer a manufacturing datum.
3. Could a set of admissible poses support a correct local radius or included
   angle even when global pose is ambiguous? Define a measurement-specific
   identifiability test (e.g. invariance/range across feasible poses), and explain
   why finite endpoints, bend identity, signed location or Gold line matching
   may still need to abstain. This is not permission to count nominal CAD bends.
4. Source hypothesis for the noisy50-row failure: coplanar triangles sharing an
   internal diagonal may receive closest-point-ambiguity-uncertain flags, which
   V3 excludes from finite support. Each5×5rectangle has5diagonal rows; losing
   those would leave20/25=.8 against .95coverage. Actual flags are unretained,
   so this is unconfirmed. Should support query semantics be invariant to
   tessellation when the represented bounded triangle union is unchanged
   (not arbitrary changes in a curved-surface approximation)? How can a
   topology-aware finite surface union preserve holes, real creases, duplicate
   sheets and material-side ambiguity while eliminating artificial seams?
5. Compare three candidate directions: (a) repair/qualify current proposal lane;
   (b) a different coarse robust registration proposal using geometric features,
   followed by independent finite/raw validation; (c) partial-pose hypotheses
   with measurement-specific eligibility and scan-supported local topology.
   Rank expected leverage, integration risk and falsifying tests. Explain when
   TEASER-style robust correspondence or bounded global search would be useful
   and when their cost/assumptions make them poor choices here. Do not assume
   their published performance transfers to sheet metal or this noise model.
6. Is any learned overtraining diagnosis supported? Current failures precede
   measurement and involve explicit geometric policies. Family score disparity
   alone cannot establish training provenance. Identify which mechanical strata
   (normal diversity, symmetry, finite extent, topology, coverage, scan density,
   closure overlap, trim complexity) would make a better diagnostic matrix.
7. Recommend a minimal decisive experiment ladder with fixed candidates,
   ablations, full failure retention and output/coverage/runtime metrics. Avoid
   months of component qualification without connected positive evidence.
   Freeze before execution and require independent PRE/POST at integration;
   do not rerun spent attempts or change scorer/thresholds to improve a score.

The target is a genuine improvement over reproducible cached .4580152672, then
P/R/localizedF1>=.90, minimum-family recall>=.80 and exact observed count>=.95.
With71refs, >=.90recall needs64TP; at64TP, >=.90precision allows at most7FP.
On11cases >=.95exact count needs11/11, but emitted count is not observed truth.
Physical measurement truth/calibrated uncertainty are absent and separate.

## Research anchors already checked, not proof of a fix

- [Yang, Shi and Carlone: TEASER](https://arxiv.org/abs/2001.07715) develops
  robust point correspondence registration using truncated least squares and
  decoupled estimation. Its problem formulation motivates an alternative
  proposal path; no benchmark result or certification transfers automatically.
- [Yang et al.: Go-ICP](https://arxiv.org/abs/1605.03344) uses branch-and-bound
  over rigid motion with an ICP objective. Global optimality of a stated
  objective is distinct from correct physical correspondence or a valid datum.
- [Bonnabel, Barczyk and Goulette](https://doi.org/10.1109/ACC.2016.7526532)
  caution that ICP rematching matters to covariance and distinguish
  point-to-point from point-to-plane analysis. An inverse Hessian is not by
  itself calibrated uncertainty for this engine's bounded/model-mismatch case.

No implementation or new physical evaluation follows merely from this brief.

## Subsequent diagnostic evidence — 2026-09-10

The [newly frozen stage diagnostic](2026-09-10_two_plane_v11_stage_diagnostic_result.md)
now fills the missing evidence in evidence item5 and architecture question4. The original
clean/noisy positives reproduce PASS/FAIL, without changing candidate or inputs.
In BOTHcases the same bank[4,1] supplies two complete25-row supports and eight
identical correct-pose aliases24–31 (action error0mm), among64hypotheses.
At that correct pose all25rows per face pass the distance upper-bound test.
The noisy case loses5face22rows and4face11rows solely to
`closest_point_ambiguity_uncertain`, yielding20/25and21/25accepted support below
.95. Those rows project to shared triangulation diagonals, including corners;
normal ambiguity and definite closest-point ambiguity are false. Clean has no
such closest-point uncertainty, despite the same19region-uncertain rows perface.
The source consumes this uncertainty as exclusion from finite support.

Thus missing a correct proposal is ruled out for this constructed failure.
The seam/uncertainty exclusion is now observed, not merely a source hypothesis.
It remains unproved that removing/splitting this gate is safe, that any
retessellation test would pass, that a full pose/measurement becomes eligible,
or that physicalV3has the same cause. Do not generalize this finding to Gold11
or call the diagnostic a repaired positive. Detailed bindings and audit status
are in the result link; no consultation message has yet been sent.

## Connected contract constraints from the subsequent source/math review

The independently reviewed
[proximity/correspondence contract](2026-09-10_proximity_correspondence_contract_review.md)
narrows a safe proposal-only interpretation. The same ambiguity exclusions
remain in refinement and final support/rank, and the workflow binds proposal and
final evidence to one shared old policy identifier. Therefore a prefilter-only
repair is not a connected-output solution; a real split needs separately bound
stage policies. None has been implemented by this review.

The current distance intervals concern stored transformed coordinates, not
latent scanner-error balls. Transfer requires separately justified point/
transform bounds, and native/physical geometry needs its own premises. Exact
same-union distance is tessellation-invariant, but near-threshold interval
dispositions and work-limit completion need not be. Near-hole proximity can be
valid while correspondence/material identity remains unresolved. These are
conditions for a falsifiable design, not reasons to erase uncertainty or widen
the old gates. Ask the consultant to address the connected path, not simply
approve replacing the initializer's row mask.
