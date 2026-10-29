# CAD mesh-engine V1 working checkpoint — September9

The autonomous goal is active. The first new physical diagnostic has completed
with no score gain; see the [retained result](2026-09-09_gold11_mesh_development_v1_result.md).
Both arms emitted0 lines; all71 false negatives per arm remain and P/F1 are
undefined. All22 attempts completed in160.483460s, no budget stop/source drift.
Independent post-run custody audit is complete, valid diagnostic/no gain/no
promotion. The [finite-surface successor checkpoint](2026-09-09_surface_registration_v2_mechanism_checkpoint.md)
records the physical diagnosis and the completed six-test constructed stage trace.
Frozen R0, approved raw files, Gold and prior terminal
attempts remain unchanged; held-outs/email and default product activation remain
out of scope. Work continues through the full roadmap, not stopping at these
engineering milestones.

Verified focused checks: mesh input138, provisional registration47, full-raw
measurement73, connected workflow53 pass. Workflow's first53-case invocation
retained38 failures caused by an invalid float count in the test fixture;
correcting3.0 to strict integer3 yielded53/53 without engine/criteria changes.

CAD sampler first108 checks retained106pass/two numerical failures. Source fixes
use exact bounded integer digit reversal with one division, and anchored affine
triangle interpolation. Original tests plus30 exact-rational/numerical controls
now pass138/138. Actual native sampling four-case test initially2/4: box/cylinder
pass; moved/reversed reused-box meshes report native status128, initially
misclassified as error. Diagnostic-only replay retained that exact flag. Official
OCCT names128 `IMeshData_Reused`; the independently reviewed fix admits only0/128
and preserves all geometry checks, rejecting every other status bit. Final163
focused tests and4/4 constructed native cases pass. Reversed/moved boxes share
the parent's cache; this is not full coverage or a general metrology certificate.

Native finite-adapter16-case run initially13/16. Independent review found two
construction-frame test mistakes: native signed-axis spans had been treated as
worldZ. Correct endpoint transform, same fixtures/gaps and downstream kernel
yield15/16; hole/notch actual scan fits now pass. Ellipse initially remained a failure.
A bounded three-phase tolerance probe proves unchanged Boolean ellipse creation
adds BSpline trim-edge tolerances5e-6mm and vertex tolerances5.000001e-6mm;
base-before/after remains~1e-7mm. The implemented local treatment preserves the
strict2e-6mm analytic surface/contact gate. Higher edge/vertex tolerances expand
exclusion regions, never certify generators. Complete edge bounds explicitly
include located vertex tolerance balls. Final original16/16 and nine independently
constructed uncertainty controls pass, including the unchanged ellipse and its
posed downstream scan measurement. A first9-case run retained one wrapper-chain
oracle failure; deriving the actual input ownership path repaired that test only.

The first actual STEP-to-scan registration/measurement connection is1/3, not green.
Both nominal arms recover pose within0.00104838mm but fail the unchanged95% scan
coverage gate against8192 finite CAD samples (actual91.2064%). The retained
nearest-sample distance p95 is0.549465mm. Distance to a fixed set is1-Lipschitz,
so even correcting the entire pose error leaves p95>=0.548417mm>0.5mm: this
specific sample set cannot satisfy that gate at the true pose. The missing-arc
case abstains but its best pose shifts inner flange planes onto outer planes;
that is not a verified acquisition-side solution. A separately versioned32768
sample successor retains all three cases and every original threshold/error
ceiling. The original failed protocol and outputs remain unchanged.

That dense successor also returns1/3 (four source-invariance controls pass).
Nominal best candidate20 now has scan coverage0.948401, plane p950.429645mm,
and zero fit-subset observability ratio, after both20-iteration limits. This
differs from the accurate sparse-run pose: increasing samples changes nearest
correspondences, PCA inputs and trimming/optimization behavior, not just final
coverage. It is a retained failed resolution hypothesis, not a repaired workflow.
Independent retained-output arithmetic puts dense nominal maximum/RMS pose-action
error at1.502403/0.887232mm; no generated pose meets the0.20mm oracle ceiling.
All final inlier support is rank6 (ratio0.0316467), but the actual696-row trimmed
fit subset from857 optimizer rows is rank5 even before Huber weighting. Exact
which-row/cap exclusion was not retained in that receipt. The separately frozen
stage trace now retains next-selection cap eviction, without establishing that
trimming alone explains the wrong pose or the physical registration failures.
No further threshold/budget changes or replay occurred. The fixed physical draft
uses different declared registration/free-circle policies, so neither constructed
matrix is proof of its success; the first physical run remains a diagnostic with
no automatic promotion and known connected-accuracy limitations.

Every new synthetic execution has an exclusive directory under
`docs/research/cad_engine_v1_synthetic_checks/`, with STARTED/OUTPUT/COMPLETED.
Notable terminal receipts:

- native original diagnostic: `native_features_v1_constructed-zcxnme57`13/16.
- corrected-frame native: `native_features_v1_constructed-uy4cw94n`15/16.
- ellipse tolerance-only: `native_tolerance_probe_v1-6v8w067q`,0.695s/361MB.
- sampler numerical negative: `cad_sampling_v1-4z9bmoeq`106/108.
- sampler numerical repair: `cad_sampling_v1-lwbbp7ej`138/138.
- native sampler negative: `cad_sampling_native_v1-r2br4hku`2/4.
- native sampler flags diagnostic: `cad_sampling_native_v1-e1vppljy`2/4.
- native sampler repair: `cad_sampling_native_v1-0lc6ptuc`4/4.
- final sampler focused checks: `cad_sampling_v1-8pheugvq`163/163.
- workflow fixture negative: `mesh_offline_v1-rfwsxzsm`15/53.
- workflow fixture repair: `mesh_offline_v1-4ojwjdei`53/53.
- post-inference scorer: `gold11_mesh_scoring_v1-mqjm5flf`47/47.
- per-case scorer reporting repair: `gold11_mesh_scoring_v1-08auo_vl`49/49; parent0.742715s, peak85999616B.
- native local uncertainty first matrix: `native_trim_uncertainty_constructed_v1-6y_o1_bd`8/9.
- native adapter final: `native_features_v1_constructed-ises_nyh`16/16; parent1.253266s, peak387055616B.
- native uncertainty final: `native_trim_uncertainty_constructed_v1-czgc6_od`9/9; parent1.180316s, peak368115712B.
- actual connected negative: `mesh_connected_native_v1-jkz08n45`1/3; parent5.403510s, peak380633088B.
- dense connected negative: `mesh_connected_native_dense_v1-j5p3bnlr`1/3; parent16.225065s, peak391299072B.

All these checks obeyed their frozen budgets without a budget stop. Their source
and synthetic scope does not demonstrate physical F1/generalization gain.

Independent pre-run review closed the per-case reporting and selection-rule
findings after49 focused tests. The protocol now requires defined-both-arm F1
gain and no regression in precision, recall, minimum-family recall or emitted
count equality; undefined required metrics mean gain unestablished. This is not
automatic promotion. Dispatch authentication and complete-run RSS repairs passed
60 source checks; the separately enabled guarded runtime smoke passed61 total
checks. The smoke constructs only an invented gp_Pnt, no mesh/STEP/inference. Receipts
`gold11_mesh_runner_v1-ptn141_o` and `gold11_mesh_guard_smoke_v1-xl2fzf3s` retain
source100b9dc7dfcb5dfe6a3f4bdd77b03e375ff878278efdf9524d0582299a856837,
parent0.414694/0.849296s and sampled peak46415872/355991552B, no stop/drift.

The new source-only freeze completed once at
`docs/research/gold11_mesh_development_v1_attempt/`; canonical bundle hash
`facecee17870701a2aa3eb9812099ee07cd745152abac17b5773649fa550d33a`.
It subsequently received an exact independent GREEN audit and executed once.
Frozen protocol/source bytes must not change. Its known connected negatives
block promotion even if a diagnostic line score improves. All-abstention and
undefined scores are valid terminal outcomes, not reasons to reopen the run.

Independent post-run verification confirms all22 result/worker/terminal hashes,
scoring projections,23 one-use leaf dispatches (22 inference plus1 scorer), fixed
order/limits and all15 source bindings. Twenty attempts were refused by
registration; two arms of one scan accepted registration but had no native
features. Their stale phase label is a successor reporting fix, not a reason
to alter the retained results.

Next active work: source-audited fixed-candidate registration stage trace and
retained physical candidate/gate diagnosis, followed by a justified separately
versioned mechanism successor. Historical incumbent
provenance remains unestablished, so any new score is explicitly a same-input
common-wrapper control/challenger result, not proof the old score was beaten.
