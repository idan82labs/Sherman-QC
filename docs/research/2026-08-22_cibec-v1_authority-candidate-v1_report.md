# CIBEC_V1 authority candidate V1 — source-free review report

Date: 2026-08-22

## Outcome

A complete but explicitly non-authoritative production-floor interpretation of the 13 CIBEC charter fields and exact S01–S14 event markers is ready for two manufacturing authorities to accept, edit, or reject.

The review payload SHA-256 is `653fa43170663b8ea9f06f1ae053bdb093ba6aa8345b262a2223897c5ec095ee`.

No physical scan, detector prediction, existing truth, expected physical count, filename identity, or family identity was opened. The pack is source-free evaluation infrastructure and cannot influence inference.

## Proposed convention

The proposal defines one inspection event as one observable contiguous forming-axis feature. It collapses paired inner/outer skins, splits curvature-sign reversals and disconnected axial spans, crops endpoints to observed support, and assigns named one-event conventions to open hems, closed hems, beads, and rolled edges.

Controversial consequences are deliberately explicit:

- S04: three compound-radius events.
- S05: four distinct bend axes; the close 6 mm pair remains separate.
- S06: two events because the axial spans are disconnected.
- S07: three return-lip bend axes.
- S08/S09: one named hem event each.
- S10: one bead-crown event.
- S11: one rolled-edge crown event.
- S12: two opposing-curvature events.
- S13: one partial event cropped to X=17–58 mm.
- S14: scan event retained while CAD correspondence abstains.

## Native-resolution visual audit

All 14 marker-review images and seven contact sheets were inspected at native resolution.

- S01/S02: marker sits at the rounded-turn midpoint and mirrors correctly.
- S03: one marker lies midway between inner and outer skins.
- S04/S05/S07/S12: every proposed split is visually separable and numbered.
- S06: the axial panel clearly shows two segments with an unsupported gap.
- S08/S09: open and closed hem markers are distinct and visible.
- S10/S11: bead and rolled-edge crown markers are on the proposed feature axes.
- S13: the line terminates at the actual shorter observed span.
- S14: the scan marker remains present; CAD abstention is encoded in the answer-key metadata.

Visual ruling: `PASS_FOR_AUTHORITY_REVIEW_ONLY`. This is not the professor-required source-free semantic qualification PASS because the proposal remains unsigned.

## Mechanical verification

`27 passed in 5.27s` across the candidate builder, two-authority hash validator, fail-closed signed materializer, complete source-free qualification dry run, neutral pack, CIBEC contract, sealed preflight, and the independently pinned shorter-span annotation-agreement matcher.

The unsigned validator correctly returns exit code 2 with only these remaining blockers:

- `MANUFACTURING_AUTHORITY_NAME_REQUIRED`
- `COUNTERSIGNING_AUTHORITY_NAME_REQUIRED`

Any edit to the charter or marker key invalidates the payload hash and requires a fresh two-authority acceptance.

The post-signature path is now automated. A temporary two-authority test payload successfully materialized an exactly ordered signed charter and passed all 14 semantic fixtures, ten-run byte replay, S02 rigid equivalence, malformed-input rejection, many-faces-to-one-event CAD mapping, duplicate face-ownership rejection, and S14 CAD abstention. The real unsigned payload still refuses materialization and qualification with exit code 2; no fake authority artifact was retained.

The audit-discovered matcher ambiguity is resolved without changing historical scores: CIBEC inter-annotator agreement now uses the ruling's angle ≤5°, transverse offset ≤4 mm, and shorter-span overlap ≥0.60 rule in `research/cibec_v1/agreement.py`. The legacy evaluator retains its historical dual-span rule. The CIBEC matcher SHA is pinned independently by the source-free preflight.

## Generalization Progress Indicator

Grade: `C_NO_GENERALIZATION_EVIDENCE`.

Direction: `EVALUATION_AUTHORITY_FRICTION_REDUCED_WITHOUT_ACCURACY_CHANGE`.

The retained CAD-assisted operator-support F1 remains 0.458015. No detector run or physical qualification was authorized by this action.
