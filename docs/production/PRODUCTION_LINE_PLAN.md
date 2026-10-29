# Sherman QC production line: plan of record

Date: 2026-10-17 (overnight autonomous run). The owner is away and has pre-authorized this
plan once an independent sanity-check agent confirms it meets two conditions: clear accuracy
gates that are verified visually, and an end-to-end flow verified visually in the app.

Planning evidence (read-only surveys, 2026-10-17) is under `docs/production/planning/`:
- `2026-10-17_subsystem_surveys.json`: nine subsystem surveys with file:line citations.
- `2026-10-17_survey_architect_plan.json`: the infrastructure architecture and contracts
  (chunks C01..C12). This plan **amends** it as described below.
- `2026-10-17_f1_reader_reports.json`, `2026-10-17_f1_synthesis.json`,
  `2026-10-17_f1_synthesis_critique.json`: the F1 deep dive and its adversarial critique.

## 1. What the owner asked for

1. **One streamlined onboarding flow ("tagging pipeline").** It takes the SKU/name, the folded CAD
   (STEP), the PDF drawing, the **unfolded/flat part** (DXF or flat STEP), and a manual
   bend-order tag that includes מהלך (machine-move) groups. The output is an immutable, versioned
   PartRecipe, and the same release registers a benchmark reference.
2. **Ingest.** A ZEISS robotic scan cell exports a mesh (PLY, STL also accepted) into a watched
   folder. The filename carries the SKU. Real local examples include `23553000_B_FULL_SCAN 1.ply`,
   `38172000_01_SCAN_2 1.ply` and `BEFORE BENDING  11979003_C.ply`. ZEISS sells no product
   named "RoboScan 7000"; the closest are the ATOS ScanBox Series 5/7 and ScanCobot. The watcher
   is therefore generic: it accepts a file once its size is stable, a temp-file rename, or an
   optional `.done` sidecar.
3. **Engine.** Using the recipe, measure each planned bend on the scan and report expected
   vs actual for **flange length first**, then angle and bend-line position, with tolerance
   verdicts and honest unknown states. It must handle **two scan types**:
   - (A) complete parts;
   - (B) incomplete parts, scanned mid-sequence with only some bends formed, while the
     reference CAD is the finished part.
   The engine must also tolerate incomplete *capture* (occluded views).
4. **Live 16:9 station UI** for the press-brake operator. It shows a 3D view with toggles for CAD,
   scan, heatmap and bend labels, plus orbit/zoom and click-a-bend focus. Next to it are the
   per-bend expected/actual values and an overall state. It updates automatically when a scan
   lands.
5. **Exportable detailed PDF report** per scan, in Hebrew and English. Heebo is OFL-licensed and
   may be bundled.
6. **Engine accuracy keeps improving**, measured honestly. Jev (TypeSafe System One) gets **data
   prep and an offline harness only**; it never sits on the measurement path.
7. Production readiness: durability, auth, observability, tests, and deployment on a shop-floor
   PC.

## 2. Ground truth from the surveys (what is actually true today)

- **"F1" is a line-localization metric.** It uses a 5°/4 mm/60% Hungarian matcher, and the research
  program is named after it. No F1 number has ever measured angle, flange length or position.
  - The best audited physical result is the scan + CAD conservative union on Gold11, which is
    development-consumed: **line-F1 0.458** (30 TP / 30 FP / 41 FN, exact count 1/11).
  - Scan-only on the same set: 0.365.
  - Scan-only V51: 0.634 on development data, but **0.197 prospective** (2/28 exact).
  - The heavy cad_guided V1–V11 (v18) chain emitted **0 lines** on physical Gold11 and takes
    10–20 minutes per case.
- **Active runtime** (`backend/bend_analysis_worker.py` → `run_progressive_bend_inspection`):
  - Alignment chain: PCA, then auto-scale, then flip-ICP, then Tukey ICP, then a RANSAC datum
    step.
  - Bend handling: a RANSAC/dihedral detector, then a greedy per-CAD-bend matcher. It runs blind:
    no recipe, no order, no stage.
  - Blind exact bend count is 3/12 at P95 46 s. All output is stamped RESEARCH ONLY.
  - Its only "flange" number is an arg-max tip-to-tip distance that has never been evaluated.
  - The structured-count promotion is a **silent no-op** on this machine, because `latest.json`
    points into `/Users/idant`.
  - Alignment hard-fails below fitness 0.55, which kills partly bent parts.
  - It reuses degrees as millimetres in two places: `qc_engine.py:1820-1823` and
    `bend_analysis_worker.py:237-243`.
- **Nothing reads the Vendor Setup draft.** Marks, sequence, machine_moves, completed_steps and
  general_measurements are written and never consumed. `completed_steps` is a single per-SKU
  integer counted in bends, not a per-scan declaration. Bend IDs are uuid4 values that are
  regenerated on every CAD publish.
- **CAD extractor V10** (boundary chains) matches 120/127 known bends against the AI-reviewed
  partial reference, in about 1.5 s per CAD. On a random email STEP it scored 12/12. It is not
  wired into upload: upload uses `vendor_native_step_upload_v1`. Unfold works on 6 of 20 corpus
  parts.
- **CAD-guided "no gain" diagnosis.** CAD was used as a blind line generator. It projected the
  **cylinder axis (centre of curvature)** rather than the sheet surface, which puts the line about
  3.7–4.7 mm from surface references. The 3 mm support gate then rejects large radii. Over-gating
  killed the heavy rebuild. The instrument (11 scans, AI-assisted labels) is too small to
  resolve a 0.09 shift.
- **Physical truth.** There is none for flange length, angle or position. Only nominal-relative
  checks on known-good parts, and repeatability, are possible until shop measurements exist.
- **Local development material** (consumed; never quote it as unseen accuracy): see §9.

## 3. Architecture (amended)

The infrastructure in the survey architect plan is adopted: processes, storage under
`SHERMAN_LINE_DATA_ROOT` outside the repo, the SQLite line store, the SMB-safe polling hotfolder,
the supervised runner, device tokens, route-policy middleware, the station feed with ETag and SSE,
the engine bundle, canonical JSON, and IDs.

**Amendment A1: a new production engine replaces "legacy worker + postprocess" on the critical
path.** The legacy worker stays available as an optional **control arm** for comparison only.

The principle: the CAD plus the operator-confirmed recipe decide *what* to inspect, once per SKU.
Each scan only answers *how far* each planned characteristic deviates, or reports UNKNOWN. Blind
bend discovery is removed from the critical path.

### 3.1 Onboarding compile: `domains/production/engine/compile/`

Runs once per recipe release, offline. All CAD work is precomputed here.

- **Input.** The folded STEP, the confirmed Vendor Setup draft and the optional flat (DXF/STEP).
- **Flange graph** (via OCP):
  - planar panels on both skins, with thickness `t` measured where both skins exist;
  - bends as cylindrical/bend regions with inner radius `Ri`, `t`, sweep, axis, and the parent and
    child panels;
  - a tree rooted at the datum panel. The root is the panel that never moves in the sequence; if
    several qualify, the largest.
  - Bend inventory comes from extractor V10 or from the native-features code
    (`backend/cad_guided_native_features.py` already computes `finite_flange_lengths_mm`). It is
    reconciled with the operator's confirmed marks.
  - Unsupported geometry (rolled, hem, spline wall, stepped boss) is declared **per feature**, not
    per part.
- **Loci.** Per bend, the compile stores the cylinder axis **and** the surface loci: inner and outer
  tangent lines, inner and outer virtual sharp (IML/OML), and the inner/outer apex.
  **The cylinder axis is never published as the bend line.**
- **Nominals with explicit conventions:**
  - angle: included angle, 180° = flat;
  - flange length per side: OML virtual sharp to the far boundary, where the boundary is either
    FREE_EDGE or ADJACENT_BEND (OML to OML);
  - bend-line position: OML in the datum frame;
  - general measurements (operator-tagged in Vendor Setup) as first-class toleranced
    characteristics.
- **Stable bend identity.** `bend_uid` is a hash of CAD sha plus canonical geometry, so it
  survives draft edits. Vendor Setup marks bind to it. A new CAD file is a new recipe version.
- **Stage models S_k:**
  - Method A (always available): rotate each not-yet-formed child subtree back by its sweep about
    the bend axis, and replace the bend with a flat strip of neutral length `α(Ri + K·t)`.
  - Method B (when a flat is bound): fold the flat.
  - K comes from the bound real flat when one exists; otherwise from shop K (setting, default
    0.33) with an explicit uncertainty.
  - Stages are defined **per bend** with the order as a soft prior, not only at מהלך boundaries.
    This represents splits inside a group, skips and out-of-order bends.
- **Flat binding.**
  - DXF parsing (ezdxf): outer and inner loops, plus bend lines found by layer, linetype or
    annotation (angle / UP / DOWN).
  - Flat STEP: a planar solid with no bend annotations, so bend lines are inferred geometrically
    or confirmed by the operator.
  - Bind flat bend lines to CAD `bend_uid` by panel-tree graph match, then a 2D rigid/mirror fit
    against the CAD development, then operator confirmation in the UI.
  - QA check: developed lengths from the flat compared with folded-CAD lengths under K. A
    mismatch flags the recipe.
- **Output.** A `CompiledPart` artifact (canonical JSON plus a face-labelled sample cloud `.npz`
  per stage, at 0.5 mm, with KD-tree ready). It is hash-bound to the recipe.

### 3.2 Per-scan engine: `domains/production/engine/inspect/`

Budget: P95 ≤ 45 s, hard stop 180 s → UNKNOWN(TIMEOUT), RSS ≤ 4 GiB, CPU only.

1. **Ingest.**
   - Mesh or point PLY/STL, in mm. **No continuous auto-scale**; only an explicit 25.4 or 1000
     factor is accepted, and a suspected unit mismatch marks the scan INVALID.
   - Two clouds: a voxel cloud (about 0.5–1 mm) for registration and a full-resolution KD-tree for
     measurement.
2. **Datum registration.**
   - Coarse: a stored per-SKU fixture pose if one exists. Otherwise a global feature match
     (FPFH + RANSAC, or plane-triplet hypotheses) of the scan against the datum panel plus its
     rigid neighbourhood. The datum is chosen so it is present at every stage.
   - Fine: trimmed or Tukey point-to-plane ICP on the datum island.
   - Keep the top-K poses, including symmetric and mirror candidates, and the margin between them.
     If the margin is ambiguous, state AMBIGUOUS_POSE. Never silently assign bend IDs on a
     symmetric part.
3. **Articulated fit.**
   - Walk the flange tree from the datum. For each child flange, find its points by
     **region-growing a plane** seeded inside the nominal corridor, with a wide angular search
     (±25°). Do not use a distance-to-nominal gate, so defective flanges are measured rather than
     dropped.
   - Fit a hinge-constrained rigid motion with θ free, then refine the plane.
   - This handles under- and over-bent parts and stage differences in the same pass.
4. **Per-bend state decode.** For each bend, decide FORMED, NOT_YET_FORMED, PARTIALLY_FORMED,
   UNEXPECTED_FORMED or UNKNOWN. The evidence is the dihedral between the fitted child and parent
   planes versus nominal and versus flat, plus coverage. The tagged order and מהלך groups act as
   a soft prior.
   - A declared stage (optional, per scan) that disagrees with the inference gives
     STAGE_MISMATCH and goes to review. The declaration is never trusted or overridden silently.
5. **Section metrology** for each formed bend, at 5–11 stations along the visible length:
   - Angle: the included angle between the fitted flange planes.
   - Bend-line position: the OML virtual sharp. When only one skin is visible, use that surface ±
     the measured or nominal `t`, and carry the `t` uncertainty in U.
   - Flange length per side:
     - FREE_EDGE: a robust far-edge boundary fit inside the flange plane.
     - ADJACENT_BEND: OML to OML.
     - Incomplete parts: the **backgauge length** from the just-formed bend's OML across
       still-flat material to the far boundary, with a stage-specific nominal from S_k. This is
       the most actionable mid-sequence check, so it is a toleranced characteristic.
   - Radius: optional diagnostic.
   - Aggregate stations by median, and report min/max to expose crowning and taper.
   - Uncertainty U combines station MAD, fit residual and a pose-perturbation term.
6. **Decisions (guard band):**
   - PASS only if |dev| + U ≤ tol.
   - FAIL only if |dev| − U > tol.
   - Otherwise REVIEW.
   - UNKNOWN never becomes PASS or FAIL, and the nominal is never copied into "actual".
   - Each UNKNOWN carries a reason: OCCLUDED, EDGE_NOT_SEEN, LOW_DENSITY, AMBIGUOUS_SEGMENTATION,
     REGISTRATION_UNTRUSTED, AMBIGUOUS_POSE, TIMEOUT or UNSUPPORTED_FEATURE.
   - Part summary: ALL_MEASURED_IN_TOLERANCE only when every required characteristic is MEASURED
     and PASS. The authority stays OPERATOR_ASSIST.
7. **Heatmap.** Signed point-to-surface deviation against S_k̂ under the datum registration
   (whole-part form view), plus a per-flange articulated residual. The legend is signed and
   matches the colours.
8. **Outputs.** `ShopfloorInspectionResultV1` with per-bend rows in sequence order, characteristic
   values with expected/actual/Δ/U/tol/decision/state/reason, stage decision, and validity. It also
   writes decimated, index-aligned viewer assets (CAD mesh, scan points, deviations, bend-line
   polylines with labels) and the PDF.

### 3.3 Jev (TypeSafe), offline only: `research/jev/`

- Exports coordinate-free, identity-free candidate states (descriptors with units, a neighbour
  summary, nearest planned bend) for one Noul: "Is this candidate a real press-brake bend line?"
  - Criteria list the documented confusers: outline, hole wall, rolled band, step, duplicate skin,
    tongue silhouette, artifact.
  - An optional Choice question picks the confuser type.
- Labels come from human labels and recipe correspondence. Splits are grouped by family, with no
  held-out data touched. The client is dry-run by default; a live call requires
  `TYPESAFE_API_KEY` plus an explicit `--live` flag, and is **not run** in this session.
- Three hooks are documented for later: onboarding mark review; unexplained creases in the
  residual lane, sent to REVIEW; and borderline formed/unformed calls, sent to REVIEW.

## 4. Chunks, ownership, waves

Rule: one owning agent per file. The only existing files allowed to change are the ones listed
explicitly, and each has a single owner. Changes there are additive and minimal: no reformatting,
no removal of user lines. Every chunk ships its own focused tests, run with
`PYT := env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:backend
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider`
on named `tests/test_prod_*.py` files only.

Infrastructure chunk IDs and specs come from the survey architect plan. The engine chunks E1–E4
and the amendments are new.

**Wave 1** (no cross-dependencies; contracts are duplicated where needed and checked later by
parity tests):

| ID | Chunk | Effort | Notes / amendments |
|---|---|---|---|
| C01 | Contracts + recipe domain | high | Adds `flat` (optional) to PartRecipeV1; per-bend states FORMED, NOT_YET_FORMED, PARTIALLY_FORMED, UNEXPECTED_FORMED, UNKNOWN; characteristic kinds FLANGE_LENGTH (side A/B), INCLUDED_ANGLE, BEND_LINE_POSITION, BACKGAUGE_LENGTH, GENERAL_MEASUREMENT; `inspection_stage` FINAL or IN_PROCESS (+ optional declared_stage); a StageDecision block; a CompiledPart reference. Decision states PASS/FAIL/REVIEW/UNKNOWN with guard band. |
| C02 | Line store, hotfolder, supervised runner | high | Adds STL ingest; SKU parsing must handle the real local name shapes (`SKU_REV_FULL_SCAN 1.ply`, `SKU_REV_SCAN_2 1.ply`, `BEFORE BENDING  SKU_REV.ply`, `(1)` copies). |
| C04 | Appliance foundation | high | As planned. |
| E1 | **Onboarding compile**: flange graph, loci, nominals, stage models, flat parse/bind, K | **max** | New `domains/production/engine/compile/**`. Uses OCP and ezdxf (installed into `.venv`; recorded in new `requirements.production.txt`). |
| E2 | **Synthetic sheet-metal suite** + virtual ZEISS scanner | **max** | `domains/production/engine/synthetic/**`. Parametric multi-bend parts (L, U, Z, hat, channel with return, a two-direction "box-ish" part with a tab, holes and notches) are built as **real STEP BReps** through OCP, so they exercise the compile path. Exact truth per bend for any stage k, injected errors (angle, flange, bend-line shift, springback per group, order swap, missing/extra bend, mirror, wrong SKU), and the matching flat DXF. The virtual scanner raycasts from robot viewpoints, with self-occlusion, noise, dropout, edge flare, outliers and a random pose, and writes a PLY mesh/points file plus ZEISS-like filenames. |

**Wave 2:**

| ID | Chunk | Effort | Notes / amendments |
|---|---|---|---|
| E3 | **Per-scan engine** (§3.2) | **max** | `domains/production/engine/inspect/**`, plus CLI `python -m domains.production.engine.cli inspect`. Consumes E1 CompiledPart; tested on E2. |
| C05 | Recipe API + release gate + flat intake backend | high | Adds flat upload/parse/bind routes and calls E1 compile at release; the release fails with explicit codes when compile fails. |
| C06 | Inspection adapter + viewer assets + detailed PDF | high | Wraps E3 (not the legacy worker); PDF in Hebrew and English with bundled Heebo (OFL); includes per-bend views, a stage section and a general-measurements table. |
| C07 | Line API | high | As planned. |
| C08 | 16:9 station UI | high | Polished, high-end industrial UI (frontend-design guidance); Hebrew and English (RTL); touch plus mouse; layers CAD/scan/heatmap/labels/stage ghost; click row → focus bend; worst-first jump. |
| C09 | Onboarding wizard UI + **flat upload & unfolded tagging view** | high | Owns `VendorSetupPage.tsx` minimal additive edits: an uploaded-flat source for the unfolded view, and a link back to the wizard. |
| C10 | Jev data prep + offline harness | high | As planned; no network. |

**Wave 3:**

| ID | Chunk | Effort | Notes / amendments |
|---|---|---|---|
| C11 | Integration + E2E | high | Owns `apps/api/routes/__init__.py`, `frontend/react/src/App.tsx`, Layout nav, server/service entrypoints, wiring, parity tests, fast E2E and engine E2E. |
| C12 | Deploy package + docs + benchmark evaluator/scorecard | high | Plus `docs/operations/zeiss-cell-integration-spec.md`. |
| E4 | **Evaluation + visual verification harness** | **max** | Per-bend scorer, synthetic scorecard, a real-part dev runner (serial, one heavy job at a time), the legacy-runtime control arm, overlay renderer, Gold11 line-metric adapter (surface locus). |

**Wave 4 (loops):**

- **A1 Accuracy loop (max):** E3 owner + E4 runner + vision reviewers, iterated until all §5 gates
  pass or the iteration budget runs out. Every change is rerun on the full synthetic suite to
  catch regressions.
- **V1 App E2E visual verification:** a Playwright run of the full flow (§6) with a screenshot
  review by vision agents. The fix loop goes back to the owning chunk.
- **R1 Final review:** an adversarial code review per area (correctness, security, honesty
  semantics, UX), with fixes, a re-verify, and a final checkpoint in `docs/CURRENT_STATUS.md`.

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

## 6. End-to-end flow (must work and be verified visually in the app)

The flow runs on real local servers, with a temp `SHERMAN_LINE_DATA_ROOT`, Playwright driving
Chromium at 1920×1080, and screenshots at each step reviewed by a vision agent.

1. An admin logs in and opens **Production → Parts → New part**. They enter the SKU and name and
   upload:
   - the STEP (once for a synthetic E2 part, once for a real dev part, e.g. 10839000_01.stp);
   - the PDF drawing;
   - the **flat DXF** (the synthetic part's generated flat).
2. The wizard shows the CAD with extracted bends. The operator confirms the bends, tags the
   **order**, groups a **מהלך**, and toggles **folded/unfolded**, where the unfolded view comes
   from the uploaded flat. Tolerances are set, the expectations table shows nominal flange
   lengths and angles, and the part is **Released** as `SKU@v1`.
3. The test copies a scan PLY into the station inbox, simulating ZEISS: write to temp, rename, use
   a ZEISS-like filename.
4. The 16:9 **station screen** updates by itself within 5 s of the job finishing, without a manual
   reload. It shows:
   - the verdict banner;
   - the 3D part with numbered bends;
   - the flange-first table with expected/actual/Δ/tol/state.
5. Screenshots: heatmap on/off; CAD-only, scan-only and both layers; labels off; click bend row 2
   (camera focuses); worst-first jump. Each state gets a screenshot.
6. An incomplete-scan drop (a synthetic stage-k scan) shows NOT_YET_FORMED bends greyed, the
   stage decision, and backgauge lengths.
7. A wrong-SKU file drop ends in **Quarantine** with a reason. The admin resolves it by assigning
   the SKU, and the job runs.
8. **Export PDF** downloads; the test renders page images and a vision agent checks them (Hebrew
   and English).
9. Killing the line service mid-job and restarting it recovers the job. A duplicate drop is
   idempotent.

**Gate:** every step passes, and the vision review signs off every screenshot with a short
critique. Polish issues are fixed or listed.

## 7. Process rules for the run

- Engine chunks (E1–E4, A1) and engine reviews run at **max** reasoning effort.
- One heavy geometry job at a time on real scans (serial runner); synthetic tests are light. Use
  `OMP_NUM_THREADS=1`; process-tree RSS < 16 GiB.
- Do not modify user-modified engine files (`backend/*`, `domains/bend/**`) except where a chunk
  lists them. The new engine lives in new packages.
- Never `npm run build` into the served `frontend/dist`; build to `$TMPDIR`.
- No commits or pushes (owner rule). A source backup was taken before the run:
  `~/ShermanQC-backups/FullProject-src-2026-10-17-pre-overnight.tgz`.
- Secrets: the TypeSafe key lives only in the gitignored `.env.local`. It is never printed,
  logged or committed.

## 8. Deferred (explicitly out of tonight's scope)

- Live Jev calls.
- ERP/MES and barcode integration.
- TLS/SSO.
- Moving legacy upload jobs onto the durable queue.
- Postgres migration.
- GPU acceleration.
- Multi-cell scale-out.
- Physical truth campaign: the protocol is written, but execution needs the shop.

## 9. Data custody for this run

Authorized (development-consumed, local):
- `../source-assets/02_CAD_QC_3D/**` (STEP/PLY/STL);
- the Gold11 development cases and their line references, located through
  `scripts/evaluate_f1_against_manual_gold.py`;
- `data/bend_corpus/parts/*/metadata.json` stage keys.

**Forbidden:**
- HOLDOUT10;
- sealed truth;
- opaque N001–N014;
- EHO4 and any email attachment;
- authorization/one-shot control artifacts;
- any held-out manifest.

Results on authorized material are engineering-ledger development evidence only.

---

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

## 11. Approval

**APPROVED, 2026-10-17, auto-approval as pre-authorized by the owner.** The independent sanity
checker confirmed that clear, visually verified accuracy gates exist and demand improvement
over the current engine (amendments 4–8). It also confirmed a concrete E2E app flow verified
visually (§6 + amendment 10). Implementation proceeds in waves 0–4 under §7 and §10.
