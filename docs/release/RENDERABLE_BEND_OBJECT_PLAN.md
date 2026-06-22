# Renderable Bend Object Plan

## Goal

Turn zero-shot bend-support strips into stable, renderable bend objects that can:

- show each detected bend directly on the 3D scan,
- let operators click a bend mark or local callout,
- open a bend card with bend ID and scan-derived data,
- stay visually aligned with the geometry,
- remain explicitly non-CAD-authoritative in scan-only mode.

This tranche is not about improving bend count by itself. It is about making scan-only structural output inspectable and actionable in the product.

## Current State

The repo now has a working end-to-end renderable bend object path:

- zero-shot builds `bend_support_strips`
- zero-shot derives `renderable_bend_objects`
- `ScanStructuralHypothesis` serializes those objects
- the bend inspection page falls back to scan-only annotations when CAD bend geometry is unavailable
- the viewer shows all bend marks on the part, a selected local callout, and a persistent bend card
- a public demo route exists for direct visual verification

What is already good enough:

- bend IDs are visible on the 3D part
- a bend chip or in-view callout click updates the selected bend card
- selected-bend highlighting is visually stronger than non-selected bends
- scan-only overlay mode no longer floods the screen with all-bends callout cards

What is still not fully mature:

- bend ownership is still derived from strip heuristics, not explicit surface assignment
- exact strip boundaries are still approximate
- current viewer validation is smoke-grade, not a full screenshot-diff suite
- scan-only bend cards are descriptive overlays, not trusted metrology claims

## Existing Repo Seams

### Backend

- `backend/zero_shot_bend_profile.py`
  - emits `bend_support_strips`
  - emits `renderable_bend_objects`
- `backend/zero_shot_bend_overlay.py`
  - converts strips into render-ready bend objects
  - renders static overview/focus artifacts
- `domains/bend/services/part_profiling.py`
  - serializes scan structural payloads into job results

### Frontend

- `frontend/react/src/services/api.ts`
  - types the scan structural payload
- `frontend/react/src/pages/BendInspection.tsx`
  - adapts scan-only bend objects into viewer annotations
- `frontend/react/src/components/ThreeViewer/index.tsx`
  - renders bend lines, markers, local callouts, and click-to-focus behavior
- `frontend/react/src/pages/ScanOverlayDemo.tsx`
  - demo route for isolated scan-only UX verification

### Validation

- `tests/test_zero_shot_bend_overlay.py`
  - backend contract smoke tests
- `docs/release/BEND_VALIDATION_RUNBOOK.md`
  - repo-wide promotion contract
- `docs/release/BEND_STRIP_TASK_BOARD.md`
  - tracked tranche status

## Product Contract

Renderable bend objects should expose:

- `bend_id`
- `label`
- `status`
- `anchor`
- `cad_line` when available
- `detected_line`
- `detail_segments`
- `metadata`

The frontend should map that into a stable viewer annotation contract with:

- bend line on geometry
- clickable bend marker
- selected local callout
- persistent bend card
- focus point / camera target

The product default should be:

- all bend marks visible
- one selected local callout
- one persistent bend card

Not:

- every bend exploding into floating cards at once

## Phase Decomposition

### E1: Contract Hardening

Owner: Volta
Scope:

- keep `renderable_bend_objects` stable in zero-shot payloads
- keep backend overlay manifest generation aligned with the same object
- document required vs optional fields

Exit gate:

- backend payload shape stable
- backend tests green

### E2: Viewer Integration

Owner: James
Scope:

- adapt scan-only bend objects into `ThreeViewer`
- keep scan-only mode distinct from CAD-authoritative mode
- support click on bend chips, callouts, and 3D bend marks
- ensure selected bend visually dominates the scene

Exit gate:

- selected bend card updates from viewer interaction
- overlay remains aligned on demo fixtures

### E3: Visual Verification

Owner: James + Locke
Scope:

- keep a dedicated visual smoke verifier for the scan-only demo route
- capture overview and selected-bend screenshots
- use UX critique to drive visual cleanup until score is acceptable

Exit gate:

- live overlay UX score at least `9/10`
- screenshots captured from the live app, not only static backend renders

### E4: Validation and Promotion

Owner: Mendel
Scope:

- define renderable-bend-object supplemental gates
- require backend contract test plus live demo verification
- ensure protected dev/holdout zero-shot slices remain green

Exit gate:

- no regression in protected zero-shot slices
- renderable object smoke passes
- validation docs reference the tranche

## Specific Subagent Decomposition

### Main rollout

- own cross-cutting contract
- decide when renderable bend objects are good enough to freeze
- decide when professor input becomes necessary

### Volta

- backend renderable object shape
- zero-shot strip-to-renderable adapter
- backend static overlay artifact seam

### James

- viewer behavior
- selection model
- overlay alignment
- bend card presentation

### Mendel

- validation contract
- task board updates
- promotion/readiness gates for this tranche

### Locke

- UX critique on live screenshots
- visual approval target

## Current Acceptance Criteria

This tranche is considered complete only if all of the following hold:

1. zero-shot emits renderable bend objects for scan-only results
2. the bend inspection page can render scan-only bend overlays without CAD bend geometry
3. the viewer shows each bend with a visible on-part mark
4. selecting a bend updates a persistent bend card
5. the selected bend has a local callout near the bend, not a global clutter stack
6. the live overlay is visually reviewed and scores at least `9/10`
7. protected zero-shot development and holdout slices remain green

## What Still Needs the Professor

Not this adapter/viewer tranche.

Professor input becomes necessary when we want renderable bend objects to be backed by stronger physical ownership:

- true support assignment on the surface
- stricter strip boundaries
- duplicate vs split as inference instead of heuristics
- confidence over competing decompositions

That is the point where renderable bend objects need to evolve from “good UI-facing structural objects” into “principled physical bend objects.”
