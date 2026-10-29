# Gold11 V11 terminal zero-output audit — 2026-09-14

Reviewer: Codex, read-only post-run reconciliation after writer quiescence.
The frozen V11 attempt and its source archive were not changed. No raw input,
new part, held-out, cache or scorer rerun was used for this audit.

## Observed result

The single waiter-launched V11 attempt ran once from 2026-09-14 02:31:41 UTC
to 09:09:41 UTC and returned exit code 0. It left 22 durable terminal case/arm
records. Both control and challenger emitted zero predictions against the same
71 reference lines: TP=0, FP=0, FN=71, recall=0, precision=null and F1=null.
The result is `COMPUTED_DEVELOPMENT_DIAGNOSTIC`; historical F1
`0.4580152672` remains non-comparable and no historical-score claim is made.

## Failure funnel

The 22 retained records partition exactly as follows (11 cases × 2 arms):

| retained terminal reason | records | interpretation |
| --- | ---: | --- |
| `NO_GEOMETRICALLY_SUPPORTED_MEASUREMENT_CONTEXTS` | 11 | finite registration candidates existed, but none survived the connected measurement-context support gates |
| `NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS` | 4 | initialization produced no finite noisy plane start |
| `NATIVE_FEATURES_UNAVAILABLE` | 2 | native topology/feature extraction stopped before registration |
| `NO_VALID_EXISTING_SUPPORT_COVER_MEASUREMENTS` | 3 | contexts completed, but the cover consumer received zero valid measurement records |

All 22 top-level records are `UNAVAILABLE` and all have zero predictions. The
zero is therefore upstream of scoring; it is not a scorer arithmetic or
precision/recall bug.

The five context-bearing records are the important diagnostic boundary:
`CG1E0BBA2933` (both arms) and `CG4BDD1371B6` control completed context
publication, then produced zero support groups. Their published flange
diagnostics repeatedly show `FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL` or
`FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY`. For example, the retained
CG1E0 control window has role-0 RMS residual 0.326662 mm (gate 0.03 mm) and
role-1 RMS residual 0.025673 mm with the normal/thickness policy rejecting the
selected population. In the retained CG4BDD control windows, one role can pass
the residual check, but the opposite role is weak/not planar; the existing
whole-component support-cover rule consequently emits no group. This is an
honest abstention under the declared policy, but it explains why the intended
measurement kernel never reaches a published bend line.

## Controlled recovery evidence

To test whether the downstream line geometry was viable independently of the
failing cover gate, the separately versioned candidate
`V11_TOPOLOGY_AXIS_RAW_SCAN_SUPPORT_RECOVERY_V1` was run on the same consumed
CG4BDD/37361005 bytes. It uses the retained completed V11 context pose, native
STEP finite bend axes only as proposals, and raw scan support to trim finite
endpoints. Gold is opened only after inference for unchanged matching.

The metric-complete reproduction is
`docs/research/gold11_v11_single_scan_luna_max_v2/CG4BDD1371B6/RESULT.json`:

| TP | FP | FN | precision | recall | F1 | bend-count capture |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0 | 0 | 1.0 | 1.0 | 1.0 | 100% |

The visual artifact contains four finite coloured scan-supported lines in
oblique and plan views. Its result hash is
`6cac4d3f4cc3945750418f869c628d218ec0daf13a4768e38f07b3380aad3c27`; the
visual hash is recorded in `CHECKSUMS.json` and matches the earlier V1 visual.
The wrapper's four focused tests pass, and a final source/input/result/visual
hash audit returned `AUDIT_OK`.

## Engineering conclusion

The principal mechanism blocking the full V11 score is the strict connected
measurement path: registration may produce conditional contexts, but the
flange residual/normal gates and whole-component exact-support-cover consumer
can reduce them to zero published groups. Native-axis recovery proves that the
same scan/CAD pair contains line-support evidence, but it does not prove a
calibrated bend radius, manufacturing event identity or a valid whole-part
measurement.

The justified next engine seam is a controlled, explicit axis-support fallback
behind an opt-in research path. It must require an accepted context pose with a
declared pose bound, a proper-rigid CAD→scan transform, finite native topology
axis, contiguous raw support and duplicate/trim/coverage checks. It must publish
`CONDITIONAL_CONTACT_LINE` with scan-derived endpoints/support only, keep radius
and deviation unknown unless independently fitted, and preserve all abstention
reasons. Promotion requires a separately frozen multi-part replay with
independent PRE/POST and adversarial wrong-pose, partial-coverage, duplicate,
unsupported-trim and deformed-flange controls. The one-scan 1.0 F1 is consumed
development evidence, not unseen generalization, production qualification or
floor-release authorization.
