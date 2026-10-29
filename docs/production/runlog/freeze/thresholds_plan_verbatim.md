<!-- E4b pre-A1 freeze: plan §5 (accuracy gates) and §10 (binding amendments), verbatim.
     source: docs/production/PRODUCTION_LINE_PLAN.md (sha256 097ae68ba67d9a7d96c73deece1ecfadd8e49b577f1a984b2d0b2214ebb27513)
     §5: lines 271-336
     §10: lines 416-503
     Thresholds never change mid-run (amendment 3). -->

## 5. Accuracy gates (must pass; "improve over current" is explicit)

All physical numbers below come from **development-consumed** local parts. They are reported in
the engineering ledger, labelled consumed development, and are never quoted as unseen-family
accuracy. Physical truth for flange length does not exist, so physical flange/angle numbers are
**nominal-relative on known-good complete parts** (bounded by real process variation).

**G1. Synthetic math qualification** (E2 suite, ≥ 200 cases across ≥ 7 part families, seeds fixed,
a realistic virtual scanner, not an idealised one):

Complete parts, full visibility:

| Characteristic | Bias | P95 abs error |
|---|---|---|
| Flange length | ≤ 0.05 mm | ≤ 0.20 mm |
| Angle | ≤ 0.10° | ≤ 0.30° |
| Bend-line position | — | ≤ 0.30 mm |

Other requirements:
- With realistic occlusion, coverage must reach ≥ 90% of characteristics MEASURED.
- **Zero false PASS:** no out-of-tolerance injected error ≥ tol + U may be decided PASS.
- Incomplete parts:
  - per-bend formed/unformed state correct on ≥ 98% of unambiguous cases;
  - symmetric cases yield AMBIGUOUS, never a confident wrong ID;
  - injected backgauge errors ≥ 0.5 mm detected;
  - no NOT_YET_FORMED bend is ever FAIL.
- More occlusion must raise UNKNOWN, not error.
- Runtime P95 ≤ 45 s and RSS ≤ 4 GiB at ~1M scan points.

**G2. Physical dev, complete scans** (local parts in §9 that compile; recipes auto-built from CAD
and marked `DEV_AUTO_CONFIRMED`):
- (a) **Line localization beats the current best.** Engine OML surface lines vs Gold11
  references, scored with the existing 5°/4 mm/0.6 matcher with the locus convention stated:
  **line-F1 ≥ 0.80** (current best 0.458) and precision ≥ 0.90.
- (b) Coverage: ≥ 90% of planned bends get a MEASURED angle and ≥ 70% of planned flange sides get a
  MEASURED length.
- (c) Nominal-relative angle on known-good parts: MAE ≤ 1.0° and no worse than the legacy runtime
  control arm on the same scans. Report per part.
- (d) Nominal-relative flange length: P95 |Δ| ≤ 1.0 mm on known-good parts. Larger deviations must
  be shown visually to be real (a vision reviewer confirms the scan edge really is there).
- (e) Runtime P95 ≤ 45 s per scan (legacy control 46 s); zero crashes; every failure is an
  explicit UNKNOWN or INVALID.

**G3. Physical dev, incomplete scans** (staged scans: 10839000 full + partials, 23553000_B FULL +
SCAN_1/2, 28870000_A FULL + SCAN_1, 38172000 SCAN_1/2, 40500000 SCAN1/2, and the 11979003
BEFORE BENDING blank as stage 0):
- The per-bend formed/unformed state matches the visual verdict for **every** bend.
- No unformed bend is reported as a defect.
- The blank is decoded as stage 0 with all bends NOT_YET_FORMED.
- Formed bends meet the G2(b–d) thresholds.

**G4. Visual verification on every part (hard gate).**
- For each dev part and scan, E4 renders at least four views (iso plus three orthographic) of the
  registered scan, with each planned bend's measured OML line drawn and labelled with its sequence
  number and `bend_uid` short code. It adds a flange-length dimension overlay and a heatmap view.
- Independent vision-reviewer agents inspect **every** image and record a per-bend verdict
  (lands on the physical bend: yes/no/occluded; label matches the recipe bend: yes/no).
- Gate: **100%** of MEASURED bends land on the physical bend, **0** mis-tagged bends, and every
  UNKNOWN is visually justified (it really is occluded or missing).
- Any failure goes back into the accuracy loop.
- Synthetic parts get the same review on a sample of at least 20 cases, including the incomplete
  ones.

**G5. No regression.** All existing focused tests touched by any change still pass. The legacy
runtime path stays byte-for-byte unchanged unless a change is explicitly listed.

## 10. Binding amendments from the independent sanity check (2026-10-17)

An independent max-effort plan reviewer returned **APPROVE_WITH_AMENDMENTS**. Every REQUIRED
amendment below is binding and **overrides** any conflicting text above or in the survey
architect plan. The OPTIONAL items are adopted as stretch goals.

Corrections by the integrator:
- `ezdxf 1.4.5` was already installed into `.venv` before the review. No further pip installs.
- Schedule consequence of amendment 1: a **wave 0** (C01 contracts + `ENGINE_IO.md`, and the G5
  baseline) runs before wave 1.
- E4's scorer, truth loader, overlay renderer and scanner model card move to wave 2. The G3 truth
  table, Gold transform recovery and legacy control arm are in wave 3, before A1.


### Required amendments

1. **§4 reconciliation (binding before wave 1).** "C03 is retired. E2 replaces its generator and E3 replaces its measurement. Every C02/C05/C06/C11/C12 reference to C03, STL CAD fixtures or the legacy worker is retargeted. Production recipes require STEP, and tests use E2 STEP fixtures. The runner's engine step is `python -m domains.production.engine.inspect` (hard stop 180 s, 4 GiB), followed by C06 assets and PDF. The legacy worker runs only in E4's control arm. Before wave 1, C01 publishes `contracts/production/ENGINE_IO.md` with:
   - the CompiledPart schema and the E3 command-line interface;
   - the E3 output directory (result JSON, registered scan, per-point deviations, per-stage display mesh, bend polylines), all in one declared frame;
   - a vocabulary map from IN/OUT/NOT_MEASURED/UNFORMED to PASS/FAIL/REVIEW/UNKNOWN/NOT_YET_FORMED, with REVIEW shown amber;
   - compile reconciliation blockers shown in the wizard (MARK_NOT_IN_FLANGE_GRAPH, UNMARKED_BEND)."

2. **§4 ownership.** "E1 owns `domains/production/engine/__init__.py`. Entry points are separate (`…engine.compile`, `…engine.inspect`); there is no shared cli.py. Real-filename parsing moves from C02 to C01 `sku.py` plus the C05 resolver. It covers `BEFORE BENDING  SKU_REV.ply`, `SKU_REV_FULL_SCAN 1.ply`, `SKU_REV_SCAN_2 1.ply`, `(1)` copies and double spaces. The SKU token may appear anywhere in the name; a revision that differs from the recipe goes to quarantine as REVISION_MISMATCH. E4 owns the scorer and scorecards (`domains/production/engine/eval/**`); C12's evaluator and scorecard become thin wrappers or are dropped. C11's licence to fix other chunks excludes `domains/production/engine/**`. R1 owns additive edits to `docs/CURRENT_STATUS.md`. No pip installs: OCP 7.9.3.1.1 and ezdxf 1.4.5 are already present."

3. **§4 schedule.**
   - "E4's scorer, truth loader and overlay renderer move to wave 2.
   - Synthetic testing is tiered. A regression tier (about 50 cases, about 200k points, at most 20 min) runs on every A1 change. The full G1 suite runs at milestones and on the final config. Runtime and memory gates use a fixed subset at about 1M points.
   - Wave 4 has a single writer: only A1 edits `domains/production/engine/**`, and engine defects found by V1 or R1 queue to A1.
   - V1 runs after the engine is frozen. Any later engine change re-runs G1–G4 on the affected cases.
   - A1 has a stated iteration and wall-clock budget. When the budget runs out, the report gives each gate PASS or FAIL with its numbers. Thresholds never change mid-run."

4. **§5 freeze before A1 starts (hashes recorded in the ledger).**
   - "(a) All thresholds.
   - (b) An E2 scanner model card: noise, point spacing, edge flare, grazing and specular dropout, and duplicate-skin offset, measured on at least 3 authorized real scans.
   - (c) DEV seeds for iteration and QUAL seeds used only on the final run. G1 is gated on QUAL.
   - (d) The DEV_AUTO_CONFIRMED recipes: every V10/native bend, both flange sides REQUIRED, exclusions only through a pre-written per-feature UNSUPPORTED rule. Compile failures stay in the denominators, and at least 80% of authorized STEPs must compile.
   - (e) The G2 and G3 case lists. 'Known-good' means the stable and secondary controls in ANGLE_TRUTH_CONTROL_SET_20260404.
   - (f) A DEV-CHECK subset of about a third of the part families, including one staged family. A1 never sees it; it runs once at the end, and 'gates met' requires G2 and G3 to hold on it as well.
   - One engine config hash must pass every gate. E3 never receives filename tokens or metadata counts; renamed copies with opaque names must give identical output."

5. **G1 integrity.**
   - "False PASS means a PASS while the true |dev| > tol, judged on E2 truth, not on the engine's U. Required: zero. False FAIL (FAIL while the true |dev| ≤ tol): at most 0.5%.
   - U calibration: |measured − truth| ≤ U on at least 95% of MEASURED characteristics. At full visibility, median U ≤ 0.15 mm and ≤ 0.25°.
   - Decisiveness: at least 90% PASS or FAIL (not REVIEW) when the true |dev| ≤ 0.5·tol or ≥ 1.5·tol.
   - The slope of measured against injected deviation lies in [0.95, 1.05].
   - 'Unambiguous' and 'symmetric' are defined by E2 truth plus an E1 compile-time symmetry test (the stage model registers onto itself within 0.3 mm RMS). An engine AMBIGUOUS or UNKNOWN on a truth-unambiguous case counts as an error.
   - Wrong-SKU and mirrored parts are never PASS. A missing or extra bend in a FINAL scan never yields ALL_MEASURED_IN_TOLERANCE.
   - Name the seven families (the plan lists only six). Each includes stage-0 and stage-1 cases, acute bends, Ri ≥ 4t, flanges ≤ 12 mm and flanges ≥ 300 mm.
   - E2 truth is computed analytically from the parameters, and E2 imports nothing from E1 or E3."

6. **G2.**
   - "(a) Before scoring, E4 recovers each Gold case's raw-to-reference transform from the time the labels were drawn: either the stored alignment, or the aligned scan matched vertex-for-vertex to the raw scan, with residual ≤ 0.01 mm. Scoring happens in the raw-scan frame. Cases with no recoverable transform are listed and excluded before any scoring. One line locus is declared in advance (the visible-surface apex); other loci are diagnostics only. Only bends whose BEND_LINE_POSITION is MEASURED from fitted scan planes emit lines; NOT_YET_FORMED bends emit none. The result is labelled 'recipe-guided vs blind union, not like-for-like'. Add an exact-part rate that must beat the legacy arm on the same scans.
   - (c) Compare only bends that both arms measure, and require engine coverage at least equal to legacy coverage.
   - (d) Nothing is excluded from the P95. Every |Δ| > 1 mm gets a zoomed section crop, reported separately.
   - (f) New repeatability gate: on 15651004 ×3, 42288005 ×2 and 47959001 (PLY and STL), P95 |Δ| between repeats ≤ 0.3 mm and ≤ 0.3°, and better than legacy."

7. **G3.** "The per-bend visual truth table is built before any engine output is seen. Two independent reviewers compare the registered raw scan with the numbered recipe CAD and record FORMED, NOT_FORMED or NOT_DETERMINABLE. A third reviewer resolves disagreements, then the table is frozen. An engine UNKNOWN matches only NOT_DETERMINABLE. Where `expected_completed` exists, the FORMED count must equal it (used for scoring only, never as an engine input)."

8. **G4 protocol.**
   - "Renders: E4 produces the four global views plus one close-up per bend at ≥ 8 px/mm with a 10 mm scale bar. The measured line is drawn solid on the outer apex, on the material. The OML sharp is dashed and the nominal is grey dashed. Points are coloured by segment, with a flange-length arrow to the fitted edge. Next to each close-up is a recipe-CAD render from the same camera, labelled from the recipe.
   - LANDS: the solid line lies within max(2 mm, Ri + t) of the crease over at least 80% of its length, and is parallel to it. TAG_OK: its label matches the CAD render's label at that spot.
   - Reviewers: two fresh agents at max effort, who did not write the engine and see no metrics. A third breaks ties. Verdicts go to `g4_verdicts.jsonl` (image sha256, bend_uid, verdict, reason).
   - Canaries: at least 10% of close-ups are secretly corrupted (line shifted 8–15 mm, or two labels swapped). The review counts only if every canary is flagged.
   - AMBIGUOUS is justified only by E1's symmetry test.
   - The legacy arm is rendered the same way, and its LANDS/TAG_OK rate is reported as the visual baseline."

9. **§3.2 engine design.**
   - "Coarse registration does not depend on the stage. It uses the datum plane plus its free edges and holes, never boundaries that are bend lines, because an unformed bend leaves a flat continuation there.
   - Fine registration uses an articulated ICP over the datum and all fitted flanges. If the in-plane pose is degenerate, mark only bend-line position and heatmap as REGISTRATION_UNTRUSTED; angle and flange length do not depend on pose.
   - The θ search runs from flat to nominal sweep + 25°, seeded in the band next to the hinge (within Ri + 2t + 5 mm). A ±25° window around nominal misses flanges that are not yet formed."

10. **§6 end-to-end additions.**
    - "Servers run from a temporary app copy (C04 `--profile app`), never from the dev repo. Every screenshot is paired with a DOM or API assertion.
    - Drop a synthetic scan with known injected errors (+1.5 mm flange on B2, +2° on B3). The table must show those bends as FAIL or REVIEW, with actuals within 0.2 mm / 0.3° of truth. Also drop the real 10839000 FULL scan.
    - Add screenshots before and after orbit, zoom and touch, and of the in-progress state. Add per-bend close-ups in the station's 3D view, judged by the G4 LANDS/TAG_OK criteria. Check that the WebGL canvas is not blank.
    - Show the station and the wizard in Hebrew right-to-left. The PDF's Hebrew must have correct right-to-left order, for example via PyMuPDF `insert_htmlbox`. The vision reviewer reads back known Hebrew strings and confirms they are not mirrored.
    - Check that the release created a `benchmark_references` row with per-bend nominals."

11. **G5.** "Before any edit, record a baseline of tsc, lint, the `check-vendor-*.mjs` scripts and the backend vendor-setup tests, plus a sha256 manifest of `backend/**`, `domains/bend/**` and `scripts/**`. Gate: no new failures, and the manifest is unchanged except for listed files."

### Optional (stretch)
- Where a symmetry maps bends onto bends with identical nominals and tolerances, report values as "ID up to symmetry" instead of UNKNOWN.
- Define a declared-stage `.json` sidecar, or explicitly defer STAGE_MISMATCH.
- Inject partially formed bends (30–70% of sweep) and order swaps, with detection-rate gates.
- Wrap V10, which imports the `scripts` v7/v9 modules, or make it explicit in the engine bundle's import list.
- Report G2(c) per part with histograms. Real springback can exceed 1° on good parts, and a gate measured against nominal rewards pulling values toward nominal; the G1 slope check is the guard.
- If WebGL fails in headless Chromium, use `--use-angle=swiftshader --enable-unsafe-swiftshader` or headed mode.

