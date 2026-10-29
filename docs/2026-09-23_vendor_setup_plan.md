# Vendor setup: implementation plan and technical review

Owner: primary agent (planning and review). Implementation: GPT-6 Sol agent.
Scope: a usable, catalog-connected pre-scan setup demo. No benchmark execution,
new physical inputs, hardware operation, or factory release is part of this work.

## Operator workflow

1. Create a SKU or open an existing catalog part. Show part number and revision.
2. Attach CAD and an optional PDF drawing. Report upload and extraction separately.
3. Review candidate bend marks in CAD coordinates. Select a mark to see nominal
   data; flag/exclude a false mark; add a missing mark. Preserve the original
   candidate and correction provenance.
4. Arrange the bend sequence using stable bend identities and accessible up/down
   controls. A bend identifier must not change when its operation number changes.
5. Save a draft tied to the CAD and drawing content hashes and revision. On reload, detect source
   changes and require review again. Save conflicts must not silently overwrite
   another editor's draft.
6. Select the planned completed operation count and preview the relevant bends.
   This is a preparation aid, not an observed count or a measured scan result.

The screen should prioritize the model and its bend list. Source attachments,
part metadata and advanced matching settings are secondary. Follow the existing
dark UI and React/Tailwind stack. Show loading, errors, empty geometry and stale
drafts explicitly. Keep original CAD candidates and manual annotations separate.

## Source findings

- The active HTTP application is `apps/api/main.py`, with explicit route/service
  binding. Adding a legacy decorator alone does not expose a new endpoint.
- CAD upload in `backend/_server_legacy_impl.py` calls the existing extractor but
  discards bend endpoints when populating the catalog. This prevents reliable
  marking of persisted candidates and must be repaired.
- `PartBendSpec` stores coordinates as JSON strings, whereas the frontend declares
  arrays. Normalize at the boundary and reject malformed/non-finite geometry.
- The upload client declares a `part` response that the CAD upload does not
  currently return. Make extraction outcome and catalog refresh explicit.
- Existing catalog specs can have no geometry. A table entry is not a located
  bend. Show unavailable geometry, or request a new preprocessing operation.
- The shared viewer accepts raw-reference coordinates and STL/PLY meshes. STEP
  bytes require conversion before they can be displayed as a mesh. A diagram of
  isolated bend lines must be labelled as such, not presented as a CAD surface.
- The initial PDF patch is incomplete: it needs active routing, authorization,
  bounded input handling and generated storage names. Avoid caller-controlled
  filesystem paths and overwriting an earlier attachment during upload.

## What will not work reliably

### Whole-part alignment of a partly bent scan to finished CAD

A rigid transform preserves pairwise distances. Folding one flange relative to
another changes cross-flange distances, so one rigid transform generally cannot
align both the intermediate and finished states. Increasing the registration
distance threshold cannot correct that model mismatch.

For example, take two flange points 100 mm from the same ideal hinge, on
opposite sides of a flat sheet: `a = (100, 0, 0)` and `b = (-100, 0, 0)`.
Their separation is 200 mm. After rotating the second flange by 90 degrees,
`b' = (0, 0, 100)`, and the separation is `sqrt(20000) = 141.421... mm`.
No rotation and translation can map both point pairs exactly because those
distances differ. This is a geometric counterexample, not a physical test.

Recommended next geometry step: represent planar regions as a graph connected
by finite bend features. Use local rigid fits for visible flange patches and
constrain adjacent fits by their shared bend. Order narrows the candidate stage;
it does not prescribe the measured angle. For a simplified hinge model, a point
rotated about a unit axis u through c follows

`x' = c + R(u, theta) (x - c)`.

For a chain, compose transforms in graph order, transforming downstream axes
consistently. Here theta is relative to a declared reference state. Never replace
an unknown angle with the nominal value and report it as measured.

This hinge approximation is useful for flange correspondence, but is not an
exact forming model: bend zones have finite radius and thickness, and the
neutral-axis location and developed length need additional assumptions. Hems,
rolled features, non-developable surfaces, closed cycles and multi-hit forming
need explicit support or abstention. No automatic unfolding guarantee follows
from a finished STEP model alone.

### Matching by height and width alone

Dimensions must be defined in local feature coordinates; camera-axis bounding
box height/width changes with pose. Partial coverage also shortens visible
extent. Repeated equal-length bends can be geometrically indistinguishable.

Use length and flange widths as candidate filters alongside topology, local
normals, adjacent bends, holes/edges when visible, support and competing poses.
Keep multiple hypotheses when evidence is ambiguous. Ask the operator to choose
a feature or acquire more coverage when no unique correspondence is supported.

The user's request for high tolerances is interpreted as a configurable broad
correspondence search window. It must remain separate from drawing-defined
acceptance tolerances; neither has an invented production default in this demo.

### Operation number as measurement truth

One tool operation can form multiple segments, and a bend may require repeated
hits. A simple one-bend-per-step list is a demo constraint. A later operation
model should map each operation to one or more stable feature IDs and allow
repeated features. Planned completion and scan observation are distinct fields.
Unseen or not-yet-formed geometry cannot be labelled a defect from the preset.

### Nominal and measured bend data

CAD supplies nominal angle/radius and intended finite support. Manual markings
are user assertions. Actual angle, radius, endpoints and signed deviations
require scan support. Signed angle also requires a declared normal orientation,
reference side, and angle convention; an unsigned angle between two normals is
not sufficient. The setup screen must not manufacture a measurement or PASS.

## Research context for the later matching engine

[Li et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Category-Level_Articulated_Object_Pose_Estimation_CVPR_2020_paper.html)
models articulated pose through part coordinates and joint parameters.
[Hagenow et al., human-in-the-loop articulated registration](https://arxiv.org/abs/2203.06018)
studies user corrections alongside pose/articulation fitting. These support
investigating part-level registration and explicit user constraints. They do
not establish accuracy for sheet-metal bends or validate this application's
engine. The graph/hinge proposal above is an engineering inference to test.

## Acceptance checks for this delivery

- New page is reachable from catalog and by URL; synthetic demo is clearly
  identified and never silently saved as a catalog part.
- Candidate selection, manual marking/exclusion, sequence edits, stage selection
  and draft save/reload work. Missing positions remain unavailable.
- Draft binding checks CAD/drawing hashes and revision; changing sources invalidates review.
- Inputs validate finite coordinates, nonzero lines, unique IDs, valid sequence
  membership and valid operation count. Saving uses revision conflict checks.
- PDF/CAD operations report upload/extraction errors without declaring a part
  defect; new routes enforce authorization and bounded upload handling.
- Isolated synthetic persistence/contract checks, frontend type/build and scoped
  lint, and responsive browser checks. No live catalog mutation during tests.

## Follow-up gate for real partial-scan inspection

Before connecting this preset to measured inspection output, construct synthetic
flange graphs with independently posed partial scans at multiple bend stages.
Include symmetry, repeated dimensions, reversed normals, occlusion, noise,
missing support, incorrect operator stage and genuine angular deviations. Check
feature association independently of measurement, retain ambiguity, and verify
that nominal values cannot leak into measured values. Then define a separately
authorized physical evaluation; UI completion is not matching-engine validation.

## Delivery checkpoint — 2026-09-23

Implemented by GPT-6 Sol and reviewed by the primary agent:

- Catalog entry: `/parts/vendor-setup`; per-part entry:
  `/parts/vendor-setup/:partId`; isolated synthetic UI: `/parts/vendor-setup/demo`.
- Catalog CAD/PDF upload, retained extraction endpoints, derived PLY preview,
  surface-picked missing and corrected bend marks, reversible exclusion, ordered
  bend list, nominal details and planned inspection stage.
- Server-persisted drafts with source/candidate-generation binding and a
  transaction-protected optimistic revision. Location correction replaces the
  original in the same operation slot. Original CAD candidates remain retained.
- New setup/upload/preview routes require an active operator/admin account.
  Existing unrelated catalog routes retain their pre-existing access model.

Verification:

- Sol reports six isolated backend tests passing (temporary synthetic SQLite
  databases and stub legacy context), Python compilation, scoped frontend ESLint,
  frontend TypeScript/Vite build and diff whitespace checks. Build retains
  existing chunk-size/dynamic-import warnings.
- Primary-agent Playwright check: synthetic 3D rendering, bend reordering,
  planned stage selection, remove/restore, actual surface-picked manual line,
  and a location correction that preserves its operation position and stage;
  no page errors or document overflow at 1440px and 390px widths.
- Primary-agent catalog UI check using an intercepted in-memory API: authenticated
  PLY preview, ordered save, stage restored after reload, stale drawing blocks
  saving, and explicit source review preserves the optimistic revision.
- No production database, real CAD, scan, benchmark or hardware was exercised.

Reproduction of isolated backend checks from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=.:backend .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider backend/tests/test_vendor_setup_catalog.py backend/tests/test_vendor_setup_service.py -q
npm --prefix frontend/react run build
```

Known delivery limits: the synthetic demo is transient and does not save to the
catalog; catalog drafts do persist. Operations are one bend per step. Source
units are explicitly assumed to be millimetres. Existing CADs without a preview
need re-upload/preprocessing. The current legacy extraction call is synchronous;
large-model background processing, time limits, job cancellation and durable
progress should precede broader production use. Preview generation is capped at
one million vertices and 500,000 faces. No physical extraction accuracy or
partial-scan matching accuracy was established by these UI/contract checks.

## Touch workspace revision

User feedback: enlarge the view and make controls easy to operate on a touch
screen. The measured initial desktop canvas was approximately 668 by 440 CSS
pixels at a 1440px viewport; several bend controls were only 24–32px high.
The revision targets a dominant canvas, an expanded editing workspace,
48-by-48px minimum controls, directly tappable model labels, explicit zoom and
stage increment/decrement controls, and protection against camera drags being
interpreted as marking taps. Touch behavior is checked with synthetic data at
desktop, tablet landscape/portrait and phone dimensions.

Subsequent user clarification: desktop is the primary layout. Keep the CAD view
and a compact bend inspector side by side on desktop, with smaller navigation
chrome and details expanded for the selected bend. Retain large tap targets;
stacking on smaller screens is a responsive fallback, not the desktop default.

Desktop-first verification: the 1440px viewport now gives the CAD canvas
930 by 680 CSS pixels (about twice the original area), beside a fixed-width
bend inspector. The route-specific navigation is an 80px rail. Synthetic
Playwright checks passed at widths 1920, 1440, 1024, 768 and 390: no document
overflow or page errors, and visible feature controls meet the 48px minimum.
Touch-emulated model selection, expanded-workspace Escape, zoom/reset,
stage increment/decrement and camera-drag rejection passed. Desktop and narrow
screenshots were visually reviewed. Sol reports frontend build and scoped
ESLint passing. These are browser-emulated interaction checks, not physical
touchscreen or factory validation.

## Bend-review usability revision

User requested distinct bend colors, high-contrast selection of the entire bend,
explicit confirmation, easy rotation, and angle/height/width in every bend row.
Height was clarified as flange depth from the bend to an adjacent outer edge.

The current catalog contract carries finite line endpoints, angle and radius,
not adjacent-face boundaries. Therefore this revision highlights the full
retained bend span, not an inferred curved CAD face. Width means bend span
(Euclidean endpoint distance, in assumed mm); flange depth is unavailable until
an explicit adjacent face and outer-edge reference are retained and validated.
Manual spans are operator-marked geometry, not extracted CAD measurements.
Operator confirmation reviews the marking only; it is separate from planned
completed bending operations and is not a QC acceptance or measurement result.

Delivered with stable per-mark colors, solid neutral CAD presentation, full-span
colored overlays and selected white halo, direct line/badge/list selection,
explicit rotate controls, and guided confirm-next/undo with saved pending or
confirmed marking status. Removed and unlocated marks cannot be confirmed by
the service. UI corrections and removal/restoration clear marking confirmation.

Verification: Sol reports frontend build, scoped ESLint and seven isolated
backend tests passing. Primary-agent mocked-catalog browser checks passed
confirmation save/reload, undo, removal reset, badge and direct colored-line
selection, actual rotation of projected geometry, and no overflow at 1440,
1200, 768 and 390px. The first screenshot exposed transparent CAD drawing over
opaque overlay lines; corrected by opt-in transparent-queue overlays with
disabled depth writing and explicit render order. Final screenshot visually
confirms complete colored spans and a selected white halo across the surface.
Touch regression passed at 1920, 1440, 1024, 768 and 390px: 48px controls,
no overflow/page errors, marker selection, camera-drag rejection while marking,
zoom/reset, expanded-workspace Escape and planned-stage controls. No physical
parts or production database were accessed; APIs were intercepted in browser
checks. Flange-depth extraction and full curved-face highlighting remain
unsupported by the retained geometry contract, not silently approximated.

## Synthetic UI recording

Created `frontend/react/scripts/record-vendor-demo.mjs` to reproduce a 1920×1080
scripted-operator recording inside the real vendor screen. It constructs an
invented eight-bend stepped panel (sharp planar folds, not a radius model),
intercepts all API requests, seeds seven candidate marks, rotates the part,
selects and confirms marks, adds the omitted eighth marking with two actual
surface clicks, and ends with eight markings reviewed. A persistent on-screen
badge identifies the synthetic scripted demo. Captions and cursor are recording
overlays only; application code is unchanged by this recording task.

Run with Vite on port 3017:
`node frontend/react/scripts/record-vendor-demo.mjs`.
Optional `--preview` generates an initial screenshot only; `DEMO_VIDEO_DIR`
selects the media destination. Default video:
`/tmp/sherman-vendor-video/vendor-bend-marking.webm` (51.32 s, 1920×1080,
25 fps, approximately 4.9 MB). Script assertions reached 8/8 marking reviews
with no page errors. Initial/final screenshots and a decoded video frame at
26 seconds were visually checked, including the manually added eighth span.
This demonstrates UI interactions only, not automated bend-extraction accuracy.

## 2026-09-24 real-CAD appearance and video update

The user subsequently authorized a demo using one previously consumed corpus
part that the importer can process and that has at least four extracted bends.
The selected source is `37361005_01.stp` (part `37361005`, Gold11 family
`CG4BDD1371B6`), bound to reviewed SHA-256
`81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e`.
`scripts/build_vendor_real_part_visual.py` rejects a hash mismatch and exports
only a derived preview mesh and UI fixture. This run yielded 15,470 vertices,
31,004 triangles, and four native STEP-topology candidate axes. It did not read
a scan, Gold labels, held-out data, or a live catalog database. The four marks
are CAD candidates, not observed bends or QC measurements.

The vendor CAD view now has opt-in contour edges, stronger key/fill/rim lighting,
and a subtle procedural brushed-aluminum appearance. An `Aluminum look` control
switches to a plain neutral view; the effect is visualization only and does not
assert the part's actual alloy or surface finish. The real-part recording uses
the actual derived mesh and four candidate axes, rotates to expose flange
height, shows the appearance comparison, and scripts selection and marking
review. Those confirmations demonstrate the UI, not human approval of bend
truth. The on-screen bend `Height` stays unavailable because the retained
catalog contract lacks validated adjacent-face/outer-edge boundaries.

Reproduction: run `OMP_NUM_THREADS=1 .venv/bin/python -B
scripts/build_vendor_real_part_visual.py`, serve the React app on port 3017,
then run `node frontend/react/scripts/record-vendor-real-part.mjs`. The local
fixture is under `/tmp/sherman-vendor-real-demo/`; no backend API writes occur
because the recorder intercepts its requests. A 1280x720 H.264 MP4 with
`yuv420p` and `faststart` was made for sharing. The real recording was visually
checked at five stages (initial lit CAD, untextured comparison, lower-angle
rotation, selected-line highlight, and 4/4 reviewed), and the final MP4 decoded
without errors. Frontend build, scoped lint, script syntax/compile, and
`git diff --check` passed. This is a realistic visual demo, not evidence of
measured bend accuracy or production readiness.

## 2026-09-24 spin-and-select recording revision

At the user's request, the real-CAD recorder now makes three visible mouse-drag
orbit sweeps before resetting to a view that exposes flange height. It then
clicks B1 and B3 in the bend list and B2 and B4 on their model badges. Each
click changes the selected bend, lights its entire retained line in white,
and holds that state before a separate confirmation pass. Separating selection
from `Confirm & next` matters: that control auto-selects the following bend,
which would otherwise make later clicks appear to trigger a highlight that had
already occurred. All four selected-state frames and three orbit frames were
visually inspected; the scripted UI assertions verified matching pressed
states in the list and on the model, 4/4 final marking reviews, and no page
errors. The earlier real-part MP4 is retained; the revision is exported under
the distinct name `Sherman-QC-Real-Part-Spin-Select-HD.mp4`.

## 2026-09-24 CAD flange-surface visual preview

The user clarified that selecting a bend should light up its created flange
*area*, not just its axis. `backend/vendor_flange_preview.py` now provides a
bounded, optional CAD-only association. It verifies the source SHA-256, uses
native STEP inner/outer cylinder axes and import-local edge/face incidence,
checks the cylinder axis and radii against the independent extractor, and
requires a scale-separated planar skin pair adjacent to the bend. The two
selected planes must be parallel, separated by the bend's sheet thickness,
and nonparallel to the base web. Candidate-to-axis matching is unique and
sub-millimetre; duplicate or ambiguous matches abstain. File, face,
candidate, mesh and overlay triangle caps apply. These are conservative visual
eligibility checks, not verified sheet-metal manufacturing-event topology or
flange-depth metrology.

The vendor setup API returns eligible triangle overlays as optional CAD-frame
geometry. The feature layer associates them by stable candidate ID; the shared
viewer only renders the selected surface in that bend's color, beneath its
existing bright full-span line. If a surface is unavailable, the screen says
so and keeps line-only selection rather than lighting a guessed patch. This
does not alter saved drafts, candidate extraction, scan inference, or QC
outcomes. Because the currently maintained upload candidate extractor may
produce lines that do not agree closely with native STEP axes, not every
catalog STEP will receive a flange overlay. There is no claim of corpus-wide
coverage.

For the hash-checked `37361005_01.stp` visual demo, the four accepted source
face pairs were B1 `[49, 51]`, B2 `[18, 20]`, B3 `[8, 10]`, B4 `[0, 7]`
(zero-based import-local ordinals). The new recording keeps the full-part spin,
clicks all four bends, and turns further for the otherwise edge-on B2 and B4
flanges. The cyan, pink, green and amber selected flange areas were visually
checked in source frames; a decoded 720p H.264 frame also shows the B2 face.
The shareable revision is `Sherman-QC-Real-Part-Flange-Surface-HD.mp4`.
Focused source-free service/association tests (9), frontend build and scoped
lint, exact reviewed-CAD fixture generation, scripted browser assertions (4/4
reviewed, no page errors), MP4 decode, and `git diff --check` passed.

## 2026-09-24 CAD-nominal flange depth

The earlier `Height unavailable` statements describe the preceding UI state;
this revision adds a narrow, optional CAD-nominal depth to the eligible native
STEP flange previews. It does not add a scan measurement. The datum is the
straight edge shared by a flange skin and its bend cylinder, not the neutral
mid-bend candidate line. The opposite edge must be a single straight rail on
the planar flange face's *outer wire*, nearly parallel to and covering most of
the tangent edge. Each skin is measured independently as the perpendicular
distance between those parallel rails. Both skins must agree within the
greater of 0.1 assumed mm or 1%. Holes are excluded by outer-wire ownership;
nonuniform, short/trimmed, tapered, or multiply plausible rails abstain.

On the hash-bound, previously consumed `37361005_01.stp` demo source, this
produced B1 17.7, B2 12.7, B3 17.7, and B4 7.7 in assumed millimetres. The
four values were checked against native seam and opposite-edge coordinates;
this is one source-only CAD example, not a corpus coverage estimate or physical
flange-height validation. The vendor bend list labels each value `Flange depth
(CAD)` and makes unavailable cases explicit. Unit declaration is still
unverified, and no tolerance, scan, observed-bend count or QC verdict is
inferred from these numbers.

Verification on this revision: 16 focused synthetic backend/service tests,
frontend TypeScript production build and scoped ESLint passed. The reviewed
real-part fixture yielded four surfaces with the four depths above. The
scripted 1920×1080 browser recording asserted each displayed depth, selected
all four flanges and completed four marking reviews without page errors;
initial and selected-B2 frames were visually checked. A distinct 1280×720
H.264 `yuv420p`/faststart video, `Sherman-QC-Real-Part-Flange-Depth-HD.mp4`,
was written to the existing iCloud `Sherman QC` folder and decoded without
errors. The older videos were preserved.

## 2026-09-24 operator order and Hebrew vendor flow

Scope clarified with the user: this delivery covers the vendor pre-scan
journey first, not every Sherman QC screen. The English/Hebrew switch is
persistent for this journey. Hebrew sets `lang=he` and RTL direction for the
route and its visible navigation; CAD/SKU identifiers and numeric values are
kept LTR. Vendor instructions, form labels, state messages, known validation
errors, accessible control names, model rotation/zoom labels and the compact
navigation rail have Hebrew copy. Other application routes remain unchanged.

The worker now has a prominent `Operator bending order` panel. CAD list order
is explicitly an unconfirmed draft and model/list badges show `?` until the
worker starts a fresh first-to-last tagging pass. Each click on a bend's model
marker/line or list card assigns the next number; duplicate clicks do not
assign twice. Undo, restart and cancel keep the saved order unchanged. Apply
requires all active bends exactly once, then marks the order confirmed. A
reordered plan resets any previous completed-stage count to zero. Changing an
active bend or its geometry invalidates order confirmation and also resets
that stage count. Marking confirmation remains a separate operator action;
none of this is a QC result or verified physical bending order.

The saved draft now carries `operator_sequence_confirmed`. For a nonempty
active bend set the backend rejects saves without it, so an initial CAD order
cannot silently become an operator plan. Existing legacy drafts lacking the
flag load as unconfirmed and require review before resaving; zero-bend drafts
remain possible. The source-only service tests include rejection of an
unconfirmed order.

Browser verification on the hash-bound, previously consumed `37361005_01.stp`
derived fixture: English-to-Hebrew switch, desktop RTL, 390px mobile RTL with
no horizontal overflow, duplicate-tap rejection, undo, a non-default
`B2 → B4 → B1 → B3` operator sequence, explicit apply, mocked save and reload
with Hebrew/order retained. A further source-free interaction check removed a
bend after saving and observed the expected loss of order confirmation,
completed-stage reset, and disabled Save. APIs were intercepted; no live
catalog, scan, Gold, held-out part or factory system was touched. Browser page
errors: none. The UI/UX pass drove explicit status text, RTL-safe identifiers,
touch-sized controls and a non-color-only order state.

Verification on the final revision: focused backend/service tests `17 passed`,
frontend TypeScript production build, scoped ESLint and `git diff --check`
passed. The scripted real-part Hebrew demonstration was recorded at 1920×1080,
encoded to a 41-second H.264 `yuv420p` faststart MP4 (4.2 MB), visually checked
at the initial, mid-tagging and saved stages, decoded without errors, and
copied byte-identically into the existing iCloud `Sherman QC` folder as
`Sherman-QC-Hebrew-Bend-Order-FullHD.mp4`. The video demonstrates UI behavior,
not human approval, physical bend truth or scan measurement accuracy.
