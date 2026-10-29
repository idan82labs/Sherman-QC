# Vendor group visibility, reference measurements and PDF — 2026-09-29

Follow-on folded/unfolded visual-development work is recorded separately in
[the unfolding checkpoint](vendor-unfold-preview-2026-09-29.md). It does not
reinterpret the folded reference measurements documented below as flat dimensions.

## Delivered source state

- Explicit machine moves retain a shared color and a `מהלך N` badge on every member's 3D marker. Available CAD flange patches are lightly tinted persistently; selecting a move intensifies all its member lines, markers and flange patches. Ungrouped bends remain ungrouped and do not reuse the authored moves' palette slots.
- General references use two exact operator-picked mesh anchors. Bend-pair mode first associates two distinct active located bends, then asks for exact measuring points. Point mode accepts two points anywhere on the preview. The association is operator-authored, not a proof that the chosen anchors belong to particular flange boundaries.
- The draft stores both CAD-coordinate endpoints, the Euclidean anchor distance, an optional operator-entered drawing dimension, and a note describing how to measure. Dashed `R1`, `R2`, ... references remain visible on the model. No midpoint, nearest-edge, surface-path distance, dimensional tolerance, scan result or machine offset is inferred.
- API and static adapter validate finite, distinct endpoints, numerical distance consistency, positive optional targets, unique reference IDs and valid bend associations. Old drafts default to no references. Removing/correcting a bend removes its associated references rather than retaining stale geometry.
- Browser PDF export uses the current view and editable draft, not necessarily saved data. A4 landscape sheets include a marked view, move legend, bend schedule, reference IDs/endpoints/dimensions/notes, and any authored bend notes. Empty notes do not add a blank page. Export is browser-local and adds no service/dependency.
- Canonical seams: `VendorSetupPage.tsx`, `vendorSetupCopy.ts`, `vendorSetupPdf.ts`, shared `ThreeViewer.tsx`, shared annotation types, `types/index.ts`, `vendorDemoStaticApi.ts`, and `apps/api/services/vendor_setup_service.py`.

## Verification on this revision

- `npx tsc --noEmit`: passed.
- Scoped ESLint on all changed TypeScript seams: passed with zero warnings.
- Static Sites build: passed. Existing bundle-size and old Browserslist notices remain; no dependency updates were made.
- `pytest --noconftest` for general-measurement and machine-move contracts: 91 passed on isolated synthetic fixtures.
- Five-part local browser regression: passed bend confirmation/numbering, explicit optional groups, grouped selection, save/reload, no-move persistence and old singleton-group normalization; no page errors.
- Tablet browser gesture regression: passed one-finger orbit (including marker-origin drag), stationary tap, pinch zoom and two-finger pan; no page errors.
- New `scripts/check-vendor-general-measurements.mjs`: passed persistent two-group colors/badges after deselection, duplicate bend selection protection, no anchor placement during drag, two exact raycast anchors, rejection of a negative target, bend-pair reference save/reload, arbitrary-point reference with null target, and actual PDF download. No page errors.
- Independent PDF seam checks: strict TypeScript and ESLint; synthetic Hebrew and 200-bend/100-move pagination stress checks; invalid image/input-cap rejection. Those checks preceded the integrating agent's small R-number and empty-note-page changes.
- Final real-demo PDF: 3 A4 landscape pages, about 1.7 MB, PDF 1.4, no JavaScript/forms/encryption. All three final rendered pages were visually inspected for Hebrew, geometry, labels, tables and clipping.

## Published artifact

- Public Site: `https://sherman-vendor-bend-demo.idan82labs.chatgpt.site/#/parts/vendor-setup/corpus-37361005`.
- Version 8; exact pushed Site source `eafe70f1c95134c93ef9cca9b6f84a9d0be2a3d0`.
- Saved version ID: `appgprj_6ab51472f1648191bc205999a4dadde0~appgver_696332ac8e688191953e9241fe387825`.
- Deployment ID: `appgdep_6abb529ea4c081918e90331817bdde10`; native deployment status `succeeded`. Public audience unchanged.
- Example PDF: `/Users/idantal/.codex/visualizations/2026/07/12/019f585f-68cd-7851-a588-9986c8cfb86e/vendor-setup-example.pdf`. It contains deliberately labeled illustrative operator annotations, not a supplied manufacturing drawing.

## Boundaries and next action

CAD units remain assumed millimetres and unverified. Point picking is not edge snapping or calibrated metrology. The printed view is not to scale and the PDF is not a released manufacturing drawing or QC result. Operator-entered target dimensions are not tolerances. Public-demo saving is browser-session-only; the maintained API stores the same extended draft contract locally when used in the full platform. No API server, raw scan, new CAD, held-out, email input or QC benchmark was accessed or launched in this task. Only the five already-published CAD-derived fixtures were used for UI checks.

Next useful refinement, if requested: edge/vertex snapping and typed measurement-direction semantics (straight-line versus projected gap), followed by explicit verified-unit handling. Do not infer bend compensation or machine feasibility without the corresponding machine, tooling and material model.
