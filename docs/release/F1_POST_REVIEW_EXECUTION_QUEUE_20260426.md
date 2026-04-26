# F1 Post-Review Execution Queue - 2026-04-26

## Purpose

This queue defines the next engineering work after PR review starts. It assumes the current PR remains draft until reviewers accept the two-slice boundary:

1. offline F1 research lane;
2. product diagnostics / scan overlay.

## Immediate Review Responses

### R1: Keep F1 Offline

If a reviewer asks whether F1 affects runtime counts, point to:

- `docs/release/F1_PR_SELF_AUDIT_20260426.md`
- `docs/release/F1_MERGE_READINESS_20260426.md`

Required invariant:

- no runtime/domain/frontend caller of `run_patch_graph_latent_decomposition(...)`;
- raw F1 is not an authoritative runtime count;
- accepted known-count output stays control-gated.

### R2: Review Large Branch By Slice

Recommended order:

1. Review offline F1 algorithm/research scripts and tests.
2. Review render/export semantics.
3. Review product diagnostics / scan overlay.

Do not merge product diagnostics as an implied F1 runtime promotion.

## Next Algorithmic Work After Review

### A1: Fix Raw F1 Undercount On Protected Anchors

Target parts:

- `47959001`
- `10839000`

Current status:

- accepted output is protected by control guard;
- raw F1 undercounts and is not promotable.

Likely work:

- inspect owned-support loss and missed interface births;
- compare raw support regions against zero-shot/control strips;
- add targeted birth/split repair only if it does not broaden known10.

Acceptance:

- raw F1 no longer undercounts the protected anchor;
- accepted known10 remains exact/range-safe;
- no broadened accepted cases.

### A2: Validate Raw F1 Singleton Promotion Candidates

Target parts:

- `48991006`
- `49024000`

Current status:

- raw F1 singleton candidate;
- manual visual review still required.

Required artifacts:

- 1080p overview with numbered bend marks;
- one selected-bend card/contact sheet for disputed bends;
- sidecar JSON with raw count, accepted candidate count, route, and owned support metadata.

Acceptance:

- manual count confirms raw F1 exact count;
- markers sit on real bend supports;
- no clustered/floating markers;
- candidate remains within control range or has explicit reviewed evidence.

### A3: Expand Strict Visual Marker Audit

Current audit:

- focused 4-case audit passes `4/4`, `0` issues.

Next audit:

- run strict marker audit on the full known10 render pack;
- include rounded/raised/non-conventional parts as separate labeled category;
- preserve render-only suppression as visualization-only.

Acceptance:

- no floating markers;
- no repeated cluster markers on one bend;
- raw solver counts unchanged by visual cleanup.

### A4: Partial-Scan Observed-Bend Lane

Current stance:

- partial scans target visible observed bends, not inferred full-part total.

Next work:

- rerun partial validation with `scan_limited=true`;
- render every partial case;
- escalate only unclear or contradictory ground truth.

Acceptance:

- no partial scan uses whole-part total as exact observed truth;
- candidate output is range/abstain unless visible observed count is defensible;
- manual labels are stored as observed-bend truth, not total-bend truth.

### A5: Rounded / Raised / Non-Conventional Feature Policy

Problem:

- rounded parts and raised features can look like bend supports but may not be conventional bends.

Next work:

- separate feature classes: `conventional_bend`, `rolled_section`, `raised_feature`, `non_counting_transition`;
- add policy flag deciding whether each class counts for the customer-facing target;
- add examples from manual review to the special-case slice.

Acceptance:

- rounded/raised cases do not corrupt conventional bend accuracy;
- render cards explicitly state feature class and countability.

## Professor Escalation Triggers

Escalate only if one of these happens:

- raw F1 remains stably wrong after interface birth/split repair on `47959001` or `10839000`;
- rounded/raised feature taxonomy cannot be made consistent with customer count semantics;
- partial-scan observed-bend semantics conflict with expected production output;
- exact-promotion evidence needs a principled confidence rule beyond current control guards.

## Do Not Do Yet

- Do not wire F1 into runtime APIs as authoritative count.
- Do not remove control guard for protected anchors.
- Do not auto-promote `48991006` or `49024000` without manual render review.
- Do not let render-only marker suppression change solver/count outputs.
