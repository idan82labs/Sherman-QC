# User-directed Fable 5.1 implementation handoff

The user explicitly requests: "Ask fable to patch the issues until we get a
clean f1 run" and "Fable 5.1 - through Claude code, use max effort in workflow."
You are the requested Claude Code implementation owner. Use exactly
`claude-fable-5-1`, effort `max`, without silent model fallback. Existing Claude
Max subscription only; do not enable extra usage, change billing, buy credits,
switch to a paid API key, install/subscribe to another service or incur new
paid-service costs. Stop and report a subscription/access blocker if reached.

## Objective and working contract

Own the Sherman QC CAD-assisted offline-engine repair workflow until a clean,
audited full-cohort Gold11 run yields an honest numeric localized F1, with
before/after results. Do not stop at a plan, source-only patch, passing small
tests or "rerun ready". Iterate implementation, focused synthetic verification,
independent technical review, connected workflow validation and separately
frozen development benchmarks. Preserve failed approaches. A clean run means
the evaluator completes on all eleven selected cases in both paired arms,
retaining every failure/abstention and verified inputs/source/scorer bindings;
it does not mean suppressing unsupported cases or forcing positive predictions.
Do not fabricate a numeric F1 when the unchanged scorer says undefined.

Work directly in `/Users/idantal/ShermanQC/FullProject`, the authoritative Git
root, preserving the existing dirty/untracked source. Do not use the parent
wrapper's backend/scripts or bootstrap/update helpers. Do not silently move to
a clean worktree that omits the substantial uncommitted research work. If the
background-agent workflow requires a worktree, report that before editing.
Read AGENTS.md, CLAUDE.md, newest docs/CURRENT_STATUS.md, docs/REPO_MAP.md,
docs/research/2026-09-07_production_generalization_roadmap.md, and
docs/research/2026-09-09_gold11_input_review_checkpoint.md first. Follow the
locked roadmap, not stale resume commands in older checkpoint sections.

Codex is the coordinating/auditing owner and will not edit engine source in
parallel with you. Earlier Codex subagents have handed off and are idle. Use
one editor per file. Independent reviews must be by a reviewer who did not
author the candidate, with concrete source/math findings and a durable decision.
Use separate reviewer agents at max effort if available; do not self-label your
own unreviewed patch independently green. Record actual model/effort and any
fallback or limitation. Keep concise dated checkpoints and exact run commands.

## Standing user authority and strict exclusions

Necessary local engine/adapter/evaluator/test/docs changes are authorized.
Repeated development evaluations are authorized ONLY on the exact already-
consumed September9 hash-bound Gold11 cohort: eleven scans, six CAD files,
unchanged71 reference lines. Bind to the approved raw hashes/reference snapshot,
not a mutable manifest. Relevant retained predictions, source/configuration
records and derived artifacts/proposed datum sidecars are in scope.

Every existing terminal attempt remains terminal. Do not reopen, resume,
overwrite, re-score, disguise a retry, relax a started limit or edit its frozen
source bundle. Preserve original inputs, Gold, historical results and R0.
A new physical successor needs a documented engineering reason, a separately
versioned candidate/protocol freezing source/configuration/inputs/references/
scorer/budgets/failure handling, and independent PRE BEFORE execution. Standing
authority replaces per-run user permission; it does not replace these audits.

No email retrieval, new parts, held-outs, unrelated datasets, hardware/robot
operation, external deployment, shared-infrastructure changes, contacting people,
git push, destructive git cleanup, dependency upgrades or new paid services.
Do not read credentials or .env. No factory-floor release. Zero-shot optimization
and cosmetic UI remain deferred. One heavy geometry job at a time, process-tree
RSS below16GiB and all stricter frozen limits; numerical threads1 for macOS
Open3D. The host's M5 Max/128GB is not permission to enlarge a started cap.

Gold stays outside inference, pose/datum/carrier selection and tuning inference
to case identities. CAD may guide correspondence and supply nominal values,
never the measured answer or observed count. Scan-derived finite bend geometry
and deviations must retain honest unavailable/uncertain states. Common-mm and
whole-part best-fit assumptions are provisional, not calibration/stable-datum
metrology. Physical angle/radius/deviation truth is missing; report it separately.
Do not invent tolerances, manufacturing approval or historical provenance.

## Verified physical failure: Gold11 V7

Read the released result and connected diagnosis, not an unchanged rerun:

- docs/research/2026-09-12_gold11_mesh_development_v7_result.md
- docs/research/2026-09-12_gold11_mesh_development_v7_scientific_diagnosis.md
- docs/research/2026-09-12_gold11_mesh_development_v7_post_run_audit.md
- docs/research/2026-09-12_gold11_mesh_development_v7_scoring_post_run_audit.md

V7 finished once, outer exit0 after about103minutes. All22 case/arm slots were
retained:21 registration refusals, one CPU-killed worker. No workflow reached
bend measurement. Each arm TP0/FP0/FN71, recall0, precision/F1null under the
frozen no-predictions policy. Physical observed count unavailable11/11 per arm.
There is no new numeric F1 and no demonstrated output improvement.

Immediate causes over22:10 raw-extrema coordinate-work limits;4 finite proposal
searches without a passing start;2 scan-plane triangle-membership limits;
2 competitive-triangle certificate width failures;2 near-tie storage limits;
1 completed residual rejection;1 CPU-terminated geometry-unknown outcome.
Five also report INVALID_NATIVE_SOLID. One partial supported pose was invalidated
by a later fatal query; do not salvage that prefix after the fact. These are
deterministic geometry/capacity failures, not evidence of over/undertraining.

Historical F1 0.4580152672 still lacks reconstructed comparability. Establish
it only from evidence; otherwise use a reproducible fresh same-input paired
control/challenger and explicitly do NOT claim it beats the historical score.
Longer-term locked offline targets: exact observed count accuracy>=95%,
precision/recall/localizedF1>=90%, minimum-family recall>=80%, independently
constructed measurement/deviation correctness under declared error models.

## Latest completed engineering experiment: do not promote it

Read docs/research/2026-09-12_raw_extrema_index_v2_result.md and its independent
POST/efficacy diagnosis. Root accepted both. The standalone sheared V2 index
passed16/16 correctness tests but FAILED its frozen efficacy decision: total
common work11,434,038→19,118,892; four-case instrumented build+queryCPU
0.098855→1.624711s (16.44x). Both required resolved/folded comparisons regressed.
Fewer evaluated rows were not enough to pay for construction and scalar bounds.
The earlier sheared primitive40 test is also terminal/passed, not speed evidence.

V2 source862b6926..., test799f3295..., harness e925765b..., protocol63bda4d9...
and its entire ten-file frozen closure must remain unchanged. Sole V2 attempt:
docs/research/cad_engine_v1_synthetic_checks/raw_extrema_index_v2-492swhwk.
Its exact independent POST is
docs/research/2026-09-12_raw_extrema_index_v2_source_post_run_audit.md,
SHA256670f62e50ee9b4c4865d8e21ceb57eec452126b69aa56948db5217c0200345dd.
No integration or physical rerun is justified by this rejected mechanism.

Do not blindly implement an expensive global reorder either. The source-only
Morton contingency and cost-adaptive design notes are NOT qualified protocols:
docs/research/2026-09-12_raw_extrema_spatial_order_contingency.md and
docs/research/2026-09-12_raw_extrema_cost_adaptive_design.md. A proposed80S key
build cannot amortize on the cheap folded12-query workload. Diagnose total
costs, not only leaf visits. These synthetic fixtures are now consumed development.

## Concrete next investigation, not a mandatory architecture

Current connected path: V7 runner -> mesh offline V14 -> plane initialization
V13 -> two-plane context V3 -> raw extrema V1. Preserve audit/scorer boundaries.
Inspect the actual named imports before proposing integration. The legacy
operator engine does not automatically use this research path.

Two-plane context V3 queries the WHOLE part first, then assigned supports. Its
raw extrema tree uses one longest-axis sort. Do not assume whole inputs are
planar. The source also computes `direction=rotation.T@u` and
`constant=dot(u,t0)` for each signed CAD assignment and caches exact represented
direction/origin/constant tuples. In the ideal frame algebra, assignments share
the source-pair axial direction up to sign, and minimum-norm t0 has zero axial
component. Floating implementations need not share bytes. Never snap these
coefficients or substitute a nominal answer to manufacture cache hits.

Potential alternative: reuse certified prior-query per-node enclosures with
a mathematically defensible bound on the DIFFERENCE of the original rounded
programs, recomputing actual original-row witnesses. Exact-affine Lipschitz
alone is insufficient; account old and new rounding errors, subnormals, origin
and constant changes, ties, charged cache construction and cheap-query fallback.
This is only a source-level hypothesis. You may choose a better justified
mechanism after audit. Use primary research when it materially helps, and
independently review critical math, transforms, units and uncertainty.

## Opaque caches: never inspect, stat, traverse or delete

- /private/tmp/gold11-v7-coordinator-u6nn_hg0
- /private/tmp/gold11-diagnostic-v1-coordinator-q3zs7zol
- /var/folders/yp/yrd9ss4s73zdt3d4_2221h580000gn/T/native-context-v3-tests-8f5x3_z_
- /var/folders/yp/yrd9ss4s73zdt3d4_2221h580000gn/T/native-context-v3-tests-ojapmm6u

Earlier opaque caches remain opaque too. No benchmark or geometry process was
live at Codex handoff; check current command identities before any heavy work.

## Deliver and continue

First acknowledge actual model/effort and source ownership, then inspect and
implement. Maintain docs/research/2026-09-12_fable51_progress.md as your concise
status/commands/audit ledger and update CURRENT_STATUS.md only with verified
milestones. Keep all attempts immutable. Ask for external authority only when
genuinely required, not another "proceed" after engineering steps. Do not create
endless readiness suites or require every CAD case to succeed before a
defensible all-case development score. Pause only for a real outside-scope
blocker, subscription limit, missing external authority or completed clean-run
deliverable. End with exact reproducible commands, audited full-case metrics,
failure accounting and honest remaining roadmap/measurement limitations.
