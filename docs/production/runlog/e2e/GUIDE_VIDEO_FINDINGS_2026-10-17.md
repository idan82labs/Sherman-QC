# App findings from the guide-video run (2026-10-17, synthetic part SYN-4ED9FC3569)

Queue for V1/R1 fixes (UI/API; not engine unless noted):
1. Release stays disabled with DRAFT_STALE when the Vendor Setup draft is saved before the drawing upload (draft records the drawing hash at save time). Needs a clear hint and/or a one-click refresh, or exclude the drawing hash from draft staleness.
2. Flat "propose" button is disabled until a Vendor Setup draft exists, and the wizard step gives no reason.
3. Release checklist shows COMPILE_PREVIEW_UNAVAILABLE ("This server has no pre-release compile preview") — informational, but it reads like an error to an operator; reword or hide in production profile.
4. Injected +2.0 deg on B3 is measured as +2.1 deg (engine, within U; keep for the A1 calibration check).
5. The bend tagging in the video was saved via API (API_ASSIST), not clicked live; V1 must automate the Vendor Setup click-through.
6. Measurement (GENERAL_MEASUREMENT measured vs CAD) is not demonstrable end to end in the station UI yet.

## From the v3 re-shoot (2026-10-18)
7. Wizard "Next" from the Drawing step skips the optional Unfolded-part step (should go to step 4; optional ≠ skipped).
8. Uploaded STEP/PDF file cards show the stored hashed name (e.g. `f6aef….step`) instead of the original filename (DXF keeps its name).
9. Flow order friction: "Propose binding" needs a saved bend setup, so the natural story (upload flat → see unfolded) can't happen before tagging; consider letting the binding be proposed from CAD bend candidates before the operator setup exists, or say so on the upload step.
