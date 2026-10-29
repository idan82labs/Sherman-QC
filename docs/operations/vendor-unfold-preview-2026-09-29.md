# Vendor CAD visual unfolding — 2026-09-29

Scope: same five explicitly approved vendor-demo CADs. No scan, Gold, email,
held-out, benchmark or catalog database input is accessed by fixture generation.
Existing raw CADs, folded PLYs and candidate lines remain unchanged.

## Frozen engineering limits before CAD execution

One serial worker per CAD; 60 seconds wall time, 4 GiB RSS, 40 MB result envelope.
Native parser limits: 5 MB source, 1,024 faces, 32 candidates, 200,000 triangles,
16,384 edge uses, 1,000,000 edge identity and overlap comparisons. OMP threads: 1.
Timeout, invalid topology, mismatch or unsupported geometry produces an empty
UNAVAILABLE envelope, never a partial flat pattern. Do not increase started-run limits.

Only exact source hashes in `scripts/build_vendor_unfold_demo.py` are authorized;
the generator rejects mutable fixture hash substitutions. Independently audited
synthetic geometry must pass before physical CAD-only preview generation.

## Meaning and acceptance

This develops a single connected physical CAD skin, preserving planar trims and
cylindrical arc length. It is not a neutral-axis blank, bend compensation, cut
pattern, tool setup or certified metrology. Units remain assumed millimetres.
Bend direction is unknown unless independently established; do not infer it from
arbitrary face-normal orientation. Tree topology and sweeps <=175 degrees only.

Acceptance: source-only math/native tests, independent geometry audit, retained
five-CAD dispositions, frontend type/lint/build, folded/flat ID/order/group and
touch regression, folded-only reference dimensions, and all-page PDF visual QA.

Execution/checkpoint results will be appended after verification.

Independent pre-CAD audit: GREEN_FOR_BOUNDED_PINNED_CAD_VISUAL_PREVIEW_GENERATION.
Exact backend SHA-256 `07b0596e706f5bd1e2134d1e07d252749805d427d0b1067bae0ca4c887acfa98`;
supervisor SHA-256 `51b3c249fc0bb0dc889a5ae4d98c6a394672ae869182f6213599b67fe17d6ff1`.
34 synthetic backend checks and 5 supervisor checks passed. Independent checks:
96 chart cases, 3,600 seam/composition cases, 5,000 overlap pairs; native ordinary
solid and through-hole accepted, sharp-return counterexample refused.
This is an engineering gate only, not manufacturing or accuracy qualification.

Frozen run command: `OMP_NUM_THREADS=1 .venv/bin/python scripts/build_vendor_unfold_demo.py`.

V1 execution completed: all five inputs returned `UNSUPPORTED_NATIVE_TRIM_CURVE`
before tessellation/development. No time/RSS cap was raised. Negative dispositions
are retained in `sites/vendor-tagging-demo/public/vendor-corpus/unfold-generation-v1.json`.
Successor engineering reason: support bounded native trim curves on intrinsically
developable planes/cylinders and prove curved sheet-boundary caps without losing
source-face completeness. New candidate requires synthetic tests and independent
audit before the same fixed-budget five-CAD successor run.

Retained V1 archive: `artifacts/vendor-unfold-preview/v1-source.tar.gz`
SHA-256 `7c4e5d53ac9a37710dba64df38d08e15c3a7b78683781e2b8ac054cecfdef5a7`.
Its unchanged extractor dependency is retained separately as
`v1-source-companion.tar.gz`; extractor SHA-256
`f089f71466e0f9bba47139c06efdf8bf651e7707fb05c78ca2195f06146b1314`.
Environment: Python 3.11.15, OCP 7.9.3.1, NumPy 2.4.6.

V2 engineering is in progress; no V2 CAD run or publication has occurred yet.
Independent review closed preflight native-curve mutation, unconditioned rational
weight comparison and circle-angle/length risks. Remaining gate: certify genuinely
equivalent spline boundaries under independently rounded parameterizations without
accepting differently shaped or warped material. Old budgets remain unchanged;
new degree/pole/refinement limits must be frozen before successor execution.

Integration checks already passed: maintained API response source-isolated tests,
frontend TypeScript/scoped lint, 18 malformed-preview refusals and capture metadata
preservation, existing folded five-part/group/save/reload browser regression, touch
orbit/tap/pinch/pan and general-reference/PDF regression. Flat real-CAD UI/PDF QA
and public deployment remain pending a supported audited preview.

V2 PRE-RUN GATE: GREEN_FOR_BOUNDED_PINNED_CAD_VISUAL_PREVIEW_V2_GENERATION.
Frozen engine `f316f56530e2fd6e44f21aa28a9871f43c795714c4b15b652139ee2cfdca461a`;
tests `d401a13c3b28730830d081eadece0d97591d56b69dc1e888744acde6256c231e`;
supervisor unchanged `51b3c249fc0bb0dc889a5ae4d98c6a394672ae869182f6213599b67fe17d6ff1`.
Additional strict bounds: curve degree8,256 poles/curve,32,768 total curve poles,
1,024 poles/surface,512 refined curve poles and512 knot insertions per comparison.
No original wall/RSS/byte/face/triangle/overlap budget is increased.
Independent native ordinary, spline-cutout and rounded-slot positives passed;
valid sharp return remains refused. 400 independent spline-pair probes corroborate
the analytic own-native-knot positive-rational Bezier error certificate.

V2 command: `OMP_NUM_THREADS=1 .venv/bin/python scripts/build_vendor_unfold_demo.py --engine-sha256 f316f56530e2fd6e44f21aa28a9871f43c795714c4b15b652139ee2cfdca461a --supervisor-sha256 51b3c249fc0bb0dc889a5ae4d98c6a394672ae869182f6213599b67fe17d6ff1 --generation unfold-generation-v2`.
Generator refuses an existing generation record and source/fixture hash mutation.

V2 execution is terminal and COMPLETE. All five cases abstained, with no flat
geometry emitted and no limit raised:

- 37361005: `UNKNOWN_HOLE_WALL_BOUNDARY`.
- 49001000: `UNPROVEN_NONPLANAR_BSPLINE_SURFACE`.
- 42288005: `NO_CONNECTED_SKIN_COVERS_ALL_CAD_BENDS`.
- 38172000: `UNKNOWN_HOLE_WALL_BOUNDARY`.
- 10839000: `PARENT_FLANGE_EXTENDS_INTO_DEVELOPED_BEND`.

Report: `sites/vendor-tagging-demo/public/vendor-corpus/unfold-generation-v2.json`.
Frozen source archive: `artifacts/vendor-unfold-preview/v2-source.tar.gz`, SHA-256
`458466a75e573df4f0eecf6c8b0b0f7ffac1f69e98de916cabf36af05cf2ab64`.
Final focused source-only receipt: 146 passed in 0.93 seconds, using
`pytest --noconftest` on the six named unfolding/worker/API/generation/moves/
measurements contract modules. These are engineering checks, not evidence that
a physical unfolded view is available. Public Site remains version 8.

Next diagnostic hypothesis: the 37361005 refusal must be localized to either
the native wall-axis/span test, thickness slab, opposite-skin plane, or translated
curve proof. A new diagnostic-only module/runner will preserve the V2 acceptance
mechanism and read only this pinned CAD, once, after independent source review.
Budget remains 60 seconds / 4 GiB RSS / 40 MB result / one serial native child.
The diagnostic will retain a new terminal record and never mark partial geometry
AVAILABLE. A successor preview requires a justified geometric mechanism, new
frozen source/protocol and independent pre-run gate; V2 is not reopened.

Diagnostic V1 frozen source: module
`fd59acd6ec5860cbc8332108dd84257d153a90cc4844c9842aa47871e91ba911`, runner
`cc97d3cdabf24e350d20a6f1f840fb6c4794ff1645b28a2fd7c14d206abf92b4`.
Additional diagnostic-only work caps: 128 wall records, 256 opposite-rail
attempts, 128 rail-pair certificates, 256 explicit directional bound calls,
32 reported knots/curve, 1 MB JSON. Combined 17 synthetic checks passed in
0.49 seconds, including an independently authored native spline-hole fixture.
No physical diagnostic has launched at this checkpoint; independent gate pending.

Diagnostic V1 independent gate: GREEN for the exact module/runner hashes above.
Six independent in-memory boundary cases agree with the frozen proof: positive,
curve drift, wrong span, axis skew, wrong plane separation and out-of-slab wall.
Strict JSON and no-geometry guards passed. This gate permits only the pinned
37361005 boundary diagnostic, not a V2 preview retry or manufacturing claim.

Diagnostic V1 completed once, exit 0, under unchanged caps. Terminal trace:
`artifacts/vendor-unfold-preview/37361005-diagnostic-v1.json` (88 native faces,
4 candidate bends, 24 wall records, 28 rail-pair checks). Four circular holes
have a full-circle boundary on source face 51 and two half-circle edges on
opposite face 49. Native walls 56–59 pass axis, 1.5 mm assumed thickness,
opposite plane and slab checks. Only one-to-one curve-locus matching fails.

V3 engineering reason: topology segmentation is not geometry difference. Add a
bounded, exact native closed-circle rail-set proof for cylinder hole walls,
certifying carrier agreement and complete circular coverage on both skins.
Missing arcs, real gaps, duplicate/overlapping arcs, axis/radius/center drift and
partial walls must remain refused. No tolerance, data or resource cap increase.
V3 remains source-only until independent pre-run audit; same five pinned CADs
and all failures/abstentions will be retained in a new V3 generation record.

V3 frozen candidate: engine
`8412778cee16d97eb70454cdc76904be939125c9fcacf4159d18c49e150f8f07`,
supervisor unchanged `51b3c249fc0bb0dc889a5ae4d98c6a394672ae869182f6213599b67fe17d6ff1`,
generator `63ccf72a308cee2781c39ebd3a0aa1f572da8c96d4e3f55b42ff818ab317eae7`.
New circle-set test module:
`756fade69c376de1e1583c384ac3de6577ac70a2c0348691bf40d75b9290c6ee`.
New stricter cap: 32 circular rails per side. No old cap or GEOM_TOL changes.
181 focused synthetic/integration checks passed in 0.95 seconds.

Independent V3 audit: GREEN_FOR_BOUNDED_V3_FIVE_PINNED_CAD_PREVIEW, conditional
on exact source archive verification. Independent 1,000 random circle-set cases,
13 adversarial cases, 16 in-memory OCP edge/adaptor cases, final 33 circle-set
tests and 37 pure existing regression tests passed. Fallback gathers every
shared rail and charges whole-carrier, closure, gap, overlap, roundoff and
translation errors to one unchanged physical error budget. No physical input
was read by author or reviewer.

Archive created before physical access:
`artifacts/vendor-unfold-preview/v3-source-8412778c.tar.gz`, SHA-256
`028a1b773e8ee0e73fbf72fa5a0061dea43f7c4e7ccde3e7cdb4b8c30de65135`.
Frozen V3 command:
`OMP_NUM_THREADS=1 .venv/bin/python scripts/build_vendor_unfold_demo.py --engine-sha256 8412778cee16d97eb70454cdc76904be939125c9fcacf4159d18c49e150f8f07 --supervisor-sha256 51b3c249fc0bb0dc889a5ae4d98c6a394672ae869182f6213599b67fe17d6ff1 --generation unfold-generation-v3`.
Decision: retain all five outcomes; expose only complete source-bound previews.
At least one real supported preview is required before flat UI/PDF acceptance
and public publication. Negative V1/V2 attempts remain terminal and preserved.

V3 archive condition independently CLOSED: whole archive hash and all eleven
source payloads match exact reviewed bytes. Archive contains 22 regular members:
eleven sources plus eleven expected 163-byte macOS AppleDouble sidecars; no
unexpected paths or links. Final pre-run gate remains GREEN with zero blockers.

V3 execution is terminal and COMPLETE, exit 0, all five cases retained:

- 37361005: AVAILABLE, all 4 bends, 1,600 flat triangles.
- 49001000: UNAVAILABLE, `UNPROVEN_NONPLANAR_BSPLINE_SURFACE`.
- 42288005: UNAVAILABLE, `NO_CONNECTED_SKIN_COVERS_ALL_CAD_BENDS`.
- 38172000: UNAVAILABLE, `PARENT_FLANGE_EXTENDS_INTO_DEVELOPED_BEND`.
- 10839000: UNAVAILABLE, `PARENT_FLANGE_EXTENDS_INTO_DEVELOPED_BEND`.

The segmentation blocker is closed on 37361005; 38172000 now reaches a later
geometric refusal. No cap was raised and no partial geometry was published.
Report: `sites/vendor-tagging-demo/public/vendor-corpus/unfold-generation-v3.json`.
This is one supported CAD preview out of five, not an all-family capability or
a production blank. Flat browser/PDF QA and publication are now in progress.

Final integration QA passed on the generated real 37361005 preview:

- Frontend TypeScript, scoped ESLint and static production build passed.
- Five-part availability/fallback browser check passed; all four flat markers
  persist with original IDs. Confirmed order, shared move color/selection and
  numbering survived flat -> folded -> flat. Restoring an unmapped manual mark
  immediately disabled flat mode and showed its folded marker. Zero page errors.
- Flat touch orbit/pitch, marker-origin drag without accidental selection,
  stationary selection, pinch zoom and two-finger pan passed. Measured marker
  orbit 23.58 px, marker-origin orbit 211.42 px, pinch span 149.10 -> 387.66 px,
  pan 85.39 px. These are interaction checks, not geometry accuracy scores.
- Folded reference point selection, CAD-coordinate distance, pair association,
  negative-target refusal and save/reload passed. References are not copied to
  flat coordinates or rendered as flat measurements.
- Four-page A4 PDF exported with folded view, flat view, bend table and folded
  reference table. Every page was rendered and visually inspected; labels,
  groups, Hebrew fonts, disclaimers and tables are readable without clipping.

UI test harness corrections were retained: count markers by `data-bend-marker`
not optional group metadata; account for the group caption beside the step;
wait for the asynchronously remounted folded marker. These were test failures,
not physical-run retries. Sandbox browser launch required its normal approved
local-browser permission. No source tolerance or dataset changed for UI QA.

Checked illustrative worksheet copied without replacing earlier outputs to
`vendor-unfolded-example-37361005.pdf` in the task output folder and
`Sherman-QC-unfolded-37361005-2026-09-29.pdf` in iCloud Drive. All three local
copies SHA-256 `5e4da07fb9a69621692431207a462a153bdb5793fd9ee97aed300923841e29ce`.
iCloud upload propagation is not independently verified. The worksheet's
operator-entered example target/notes are illustrative, not manufacturing truth.

Publication completed: public Site version 9, source commit
`3c08f7198ea2bed9031ec9eaa13896e9344b1bd0`, saved version
`appgprj_6ab51472f1648191bc205999a4dadde0~appgver_f7a995ea97b48191910658a89880af4e`.
Archive content hash `sha256:1bb0c4e5e334ef7aec374dd63608ef77b46c61f3a19cd9715d824d4b6b41a8ef`.
Native deployment `appgdep_6abb7cf68bb881918afc011fc8c79b6f` returned
SUCCEEDED at 2026-09-29T08:55:35Z with URL
`https://sherman-vendor-bend-demo.idan82labs.chatgpt.site`.
Public audience preserved; no complete platform/API, raw STEP or scan was deployed.

Supported operator path: choose 37361005 -> פרוס -> select/tag bends -> optional
מהלך grouping -> confirm order -> PDF. Folded export adds the flat view as a
separate sheet when the complete preview is available. Other four parts remain
folded-only. Manual geometry corrections have no proved flat mapping yet.
Future safe work: finite-width parent-flange validation for complex trims,
complete split-face skin graphs and certified developable native spline surfaces,
each with new source tests and an independent gate. Do not reopen V1/V2/V3 or
claim a neutral-axis/manufacturing blank from this visual preview.

## Preview-first app verification after manufacturing-candidate work

The user deferred manufacturing qualification and requested an in-app preview
and visual folded/unfolded integrity checks. The existing visual development is
already integrated in the maintained vendor page/API and public Site version 9.
It must not be described as the new FreeCAD compensated backend: that candidate
remains offline, and no compensation profile or manufacturing release is implied.

Fresh live-browser verification of version 9 passed. Older open tabs still had
the previous interface; a fresh tab exposed the current `מקופל` / `פרוס` controls.
For the existing published 37361005 derived preview, all four bend markers and
the flat outline/cutouts rendered. An isolated, unsaved example ordered bends
B3, B1, B2, B4 and grouped B3/B1 as מהלך 1. Flat -> folded -> rotated/pitched
folded -> flat preserved all four bend identities and the exact 1/2/3/4 order;
both group bends were highlighted together with matching color/group labels in
both views. No test plan was persisted, and existing user tabs were untouched.

At a 1024 x 768 tablet landscape viewport, the controls, outline, selected group
and labels remained visible with document width 1024 (no horizontal overflow).
The normal desktop viewport also had no document overflow. This turn verified
visible orbit/pitch controls; it did not claim a new touch-pinch or native
geometry accuracy experiment. The earlier exact-source touch/native checks
remain separately recorded above.

All four other published parts were navigated through the demo's part selector:
49001000, 42288005, 38172000 and 10839000 all had disabled flat controls,
explicit unavailable notices and folded view retained. Returning to 37361005
restored its available flat preview. The checked fresh tab had zero captured
console errors. No raw physical CAD, scan, Gold, email or new dataset was read;
no native generation was retried, no source tolerance changed and no new Site
version was needed for this already-integrated path.

Handoff: preview-first is usable for **one of the five** current demo parts.
Broader real-part support and the separate compensated backend integration are
future engineering work, not implied by these UI checks. Manufacturing remains
deferred at the user's request. The fresh deliverable tab opens on the existing
37361005 flat view without an example order saved into the app.
