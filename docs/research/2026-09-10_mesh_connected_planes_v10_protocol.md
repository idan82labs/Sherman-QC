# V10 unchanged native matrix: rejection-only finite-face prefilter

Date:2026-09-10. Prospective source and invented-native qualification. Physical
scans/CAD, cached predictions and Gold are outside this protocol. No execution
until independent exact pre-run release of the full source/test/harness closure.

## Engineering reason and prerequisites

The [V9 native attempt](2026-09-10_mesh_connected_planes_v9_result.md) is terminal
under240s workerCPU: only two of four complete case artifacts survived. In those
two records60,812,796 initialization certificate pairs versus4,301,876 refinement
pairs identify expensive initialization work; this is not a measured CPU profile.
V10 introduces a separately tested complete-face box rejection screen to avoid
full finite-triangle queries only when a complete support is already provably
ineligible. No hypothesis, support, case or original geometry/error gate is removed.

Prerequisites: face-box helper122PASS; initializer86PASS and mocked workflow54PASS
under their [integration freeze](2026-09-10_face_box_v10_pre_run_audit.md), each
with independent retained post-GREEN. The latter are not native qualification.
The V9 physical-configuration diagnostic0/2 remains separate: the original
thin-part cap directions fail its .05mm noise-cone policy before any pose/query.
V10 does not claim to repair that cause or calibrate scanner uncertainty.

## Candidate and unchanged four cases

Candidate is `inspect_cad_guided_mesh_v10` using the V10 box-aware initializer and
unchanged V3 bound-aware finite-surface registration. R0 here is the common-wrapper
extractor control, not a reconstruction of the frozen historical pipeline.

The original asymmetric1mm thick R5/R6 strip,20/29mm flanges and30mm extrusion
are retained. Native STEP and analytic scan tessellations are independent. The
four ordered cases remain:

1. V1 independently oblique nominal: exactly one scan-derived measured bend.
2. V1 independently oblique missing arc: completed geometric abstention, never
   observed absence or a resource/numerical failure counted as a success.
3. R0 common-wrapper independently oblique nominal: exactly one measured bend.
4. V1 nominal with the original bounded3D row-ID modular noise: exactly one
   measured bend; represented noisy-minus-clean displacement norm<=.001mm.

Position-error bound remains .005mm, stipulated and uncalibrated. The original
.999 rounding reserve, all points/faces, signed pose, geometry and case order
remain unchanged. Native worker AST is compared to V8 with only V10 API/phase
names changed. CAD cannot supply measured geometry; known pose is used only after
inference for evaluation. Gold is not loaded.

Original clean maxima: radius2e-5mm, absolute sweep2e-5degrees,
generator/endpoints.02mm and whole-raw-row pose action.20mm. Original noisy
maxima: radius.01mm, absolute sweep.1degrees, endpoints.05mm and pose.20mm.
The independently preserved nominal oracle still rejects every nominal abstention.

All policies remain as V9/V8:32768 CAD samples, at most3000 registration rows,
1024 coarse/fine samples,20+20 iterations, .5mm correspondence,.05mm P95,
.95 scan/.30 CAD-sample coverage and tangent measurement mode. Internal quotas,
uncertainty/rank/ambiguity handling and1.024B initializer/128M surface/8B
registration certificate-pair caps remain unchanged. Box rows add charged work
inside the same cumulative work limit; no started or predecessor cap is raised.

## Completion evidence and allowed accounting adaptation

Every used box is bound to an eagerly constructed immutable V3 whole-face index
after exact triangle-union checks; all face vertices, not patch rectangles or
selected islands, determine its extrema. Complete raw support is materialized
once in the same represented transform buffer. Every row is screened; exclusion
requires an upward possible-inlier fraction strictly below minimum coverage.
An inconclusive box receives unchanged full V3 queries on every support row.
Other supports/hypotheses remain evaluated and retained after an exclusion.

The native result verifies policy/completion, face/row/digest bindings, exact
support denominator, rational upper-fraction inequality and row/call/support
partitions. Excluded supports have no finite query or actual inlier claim.
V9's geometric completion AST changes only two query-accounting assertions to
allow this explicit strict complete exclusion. Original schedule, group-origin,
refinement, selected-pose and measurement oracles remain exact.

Missing-arc prefilter abstention needs a nonempty completed schedule and, for
every rejected proposal, either a strict whole-support box upper exclusion or
the original definite finite-distance witness. Uncertainty, no generated poses,
incomplete execution, exhausted budgets or inconclusive boxes cannot pass.

Independent test source has57prospective outcomes:45predecessor outcomes plus
two all-box/mixed-query positives and10 faulty box-binding/completion/accounting
refusals. Its source gate expects56PASS and exactly one declared native skip.
The native gate expects57PASS plus four complete passing case artifacts.

## Freeze, execution, resources and terminal handling

Root read the full1058-line test. An independent reviewer must verify the exact
test, candidate, dependency closure, this protocol and additive harness routes.
Freeze source/native gates together; no common-file edits between their runs.
The V9 test is an additional AST-only source dependency, never a native rerun.

First run exactly once:
`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite mesh_connected_planes_v10_source`

Source gate:60s monitored wall,45s workerCPU,1GiB sampled process-groupRSS,
1MiB output. Native opt-in is absent. After independent retained post-GREEN,
run exactly once:
`.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite mesh_connected_planes_v10`

Native gate:300s monitored wall,240s workerCPU,4GiB sampled process-groupRSS,
1MiB output. These are exactly the V9 limits. The harness supplies only the new
V10 native opt-in and fresh parent-owned artifact directory; private HOME/TMPDIR,
threads1, one heavy job at a time. Four exclusive case slots, each<=16MiB and
total<=64MiB, retain complete scientific records. Do not overwrite or reopen a
terminal attempt; failure needs a separately justified frozen successor.

Retain STARTED, OUTPUT, COMPLETED and all full/partial case artifacts, all source
maps, errors, omissions and resource dispositions. Any unexpected skip, failing
assertion, missing artifact, source drift or resource failure makes the matrix
negative. Partial passing cases do not make a four-case pass. Existing supervisor
sampled-RSS/total-parent-wall/cleanup limitations are not strengthened by wording.

## Advancement and claim boundary

Independent post-audit must confirm all57outcomes, four artifacts and unchanged
original geometry/error oracles under the declared cap. A complete pass is
constructed-engine completion progress relative to V9's terminal partial matrix,
not a controlled per-operation speedup ratio or physical F1 improvement.

Then proceed to a NEW independently frozen all11 consumed-development evaluation
against the recovered retained prediction control, retaining unsupported cases
and all71 references. Do not wait for universal CAD support or count synthetic
F1 as physical F1. Historical invocation/reference attestation remains incomplete;
cached reconstruction is not historical replay. These gates do not authorize
default engine promotion, observed-count accuracy claims, calibrated metrology,
unseen-family generalization or factory-floor release.
