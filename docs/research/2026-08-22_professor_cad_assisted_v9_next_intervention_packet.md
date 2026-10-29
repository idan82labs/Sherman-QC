# Sherman QC F1 — CAD-assisted production V9 next-intervention packet

Date: 2026-08-22  
Requested reviewer: GPT-5.6 Pro extended thinking  
Decision requested: one bounded falsifiable next intervention, or a formal capability-opening STOP

## Research objective and lane separation

The locked research objective remains a trustworthy zero-shot scan-only bend detector. CAD-assisted inference is a separate operator-assist lane and may not contaminate the scan-only promotion claim, model selection, thresholds, or held-out truth.

Production target gates remain:

- at least 30 scans across at least 15 known or near-neighbor families;
- exact-count accuracy >= 0.90;
- bend-line precision >= 0.90;
- bend-line recall >= 0.90;
- minimum family recall >= 0.80;
- render-admissible marker accuracy >= 0.99;
- deterministic replay = 1.00;
- P95 latency <= 300 seconds;
- peak RSS < 16 GiB;
- job success >= 0.995;
- no identity, filename, expected-count, CAD, gold, or benchmark-feedback leakage into the scan-only lane;
- no part-specific tuning.

The current 11-scan matched physical slice is development-consumed. It can falsify mechanisms but cannot prove promotion or generalization.

## Frozen matched-slice results

| Lane | Precision | Recall | F1 | TP | FP | FN | Exact count | Minimum family recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scan-only A0 | 0.5758 | 0.2676 | 0.3654 | 19 | 14 | 52 | 0/11 | 0.1111 |
| Retained incumbent: scan + scan-supported legacy CAD | 0.5000 | 0.4225 | **0.4580** | 30 | 30 | 41 | 1/11 | 0.2000 |
| Native STEP V7 targeted-only, before axis fix | 0.3171 | 0.1831 | 0.2321 | 13 | 28 | 58 | 2/11 | 0.0000 |
| Native STEP V7 fusion | 0.4333 | 0.3662 | 0.3969 | 26 | 34 | 45 | 1/11 | 0.2000 |
| Native STEP V8 targeted-only, corrected axis | 0.3571 | 0.3521 | 0.3546 | 25 | 45 | 46 | 2/11 | 0.0000 |
| Native STEP V8 fusion | 0.4000 | 0.4507 | 0.4238 | 32 | 48 | 39 | 2/11 | 0.2000 |
| Native STEP V9 targeted-only, two-normal gate | 0.3898 | 0.3239 | 0.3538 | 23 | 36 | 48 | 0/11 | 0.0000 |
| Native STEP V9 fusion | 0.4429 | 0.4366 | 0.4397 | 31 | 39 | 40 | 2/11 | 0.2000 |
| Three-source high-recall review lane | 0.4138 | **0.5070** | 0.4557 | **36** | 51 | **35** | 2/11 | 0.2000 |

Full 13-scan V9 targeted-only performance: precision 0.3582, recall 0.3117, F1 0.3333, 24 TP, 43 FP, 53 FN, exact-count accuracy 1/13, minimum family recall 0.

## Implemented CAD-assisted architecture

1. Reuse the existing automatic CAD/scan overlay; do not implement a second registration system.
2. Export the exact raw-scan-to-CAD homogeneous transform used by the overlay.
3. Force overlay execution for all CAD-assisted cases; safely abstain if the overlay is unavailable or below trust threshold.
4. Read native STEP B-Rep topology through OCP instead of tessellating STEP into mesh crease candidates.
5. Define nominal candidates as paired coaxial inner/outer partial cylinders with:
   - angular sweep 15–200 degrees;
   - axial span >= 8 mm;
   - coaxial offset <= 0.5 mm;
   - axis-angle difference <= 0.5 degrees;
   - common-span fraction >= 0.80;
   - radius separation 0.2–10 mm.
6. Project each axis into raw scan coordinates through the inverse overlay transform.
7. Require the pre-existing scan-frame support gate: 201 samples, <= 3 mm point distance, >= 10% support, >= 8 mm longest contiguous supported span; crop to that span.
8. Suppress geometric duplicates at 5 degrees, 4 mm, and 60% shorter-span overlap.
9. V9 additionally requires a two-surface scan-normal signature in a 10 mm corridor: >= 40 points, >= 15 degrees normal turn, and >= 15% support in the smaller unoriented normal mode.

No thresholds were fitted to gold scores.

## Corrected mathematical defect

V7 canonicalized a cylinder axis direction before evaluating STEP's signed V parameters. Cylinders with a negative native direction were mirrored. V8 evaluates the V parameters in the native axis direction and canonicalizes only endpoint ordering. Two explicit unit tests cover positive and negative directions.

This correction materially improved targeted-only matched F1 from 0.2321 to 0.3546.

## Determinism, resources, and reliability

- Blind E2E inference completed 13/13 cases.
- Blind contract rate: 1.00.
- Forced-overlay P95 latency: 79.709 seconds.
- One case safely abstained because the automatic overlay was unavailable or untrusted.
- Primary and replay V9 standalone and fusion metric tables are byte-identical.
- 117 relevant tests passed; 1 skipped.
- No source points are serialized into compact evidence outputs.

## Native visual adjudication

All 13 V9 cases were inspected through five contact sheets containing six native-resolution views each.

Positive findings:

- V8 fixes visibly mirrored axes.
- Native STEP axes are more manufacturing-relevant than mesh silhouette edges.
- The V9 two-normal gate removes some weak proximity-only candidates.
- Several simple tray and bracket bends are clean and operator-interpretable.

Remaining failure modes:

- Inner lips, beads, hems, rolled sections, and compound rounded features produce multiple physically real CAD axes for one inspection-level bend identity.
- V9 precision remains 0.3898 alone and 0.4429 in fusion.
- The high-recall lane adds 6 TP over the incumbent but 21 additional FP.
- `40500000` remains the strongest ontology conflict: CAD yields multiple visibly plausible formed transitions, but their identity does not agree cleanly with the trusted line set.
- Partial `38172000` scans remain under-covered or multiply represented.
- One `10839000` scan correctly abstains at registration.

Visual ruling: PASS WITH LIMITATIONS for operator suggestions; FAIL for autonomous part disposition.

## New physical evidence state

New physical scans exist but are not yet an independently annotated locked 30-scan/15-family promotion corpus. Do not tune on them opportunistically. If you authorize using them, specify a pre-exposure annotation and custody protocol and the exact frozen execution lane.

## Reviewer task

Choose exactly one next action that maximizes credible progress toward production-grade bend inspection while preserving the scan-only promotion claim and the CAD-assisted lane separation.

Your ruling must contain:

1. `DECISION`: one of `FUND_ONE_BOUNDED_INTERVENTION`, `OPEN_ONE_NECESSARY_CAPABILITY`, or `FORMAL_STOP`.
2. A precise causal diagnosis of the dominant remaining FP/FN mechanism.
3. One implementable intervention only. Define inputs, coordinate frame, topology/geometry operations, every fixed constant, deterministic tie-breaking, abstention behavior, and output schema.
4. A source-free or synthetic prerequisite suite that can falsify the implementation before physical scoring.
5. A single locked physical execution protocol, including whether it is development-only or may open new scans.
6. Required numerical, visual, determinism, latency, memory, and failure gates.
7. Explicit retain/retire logic relative to the 0.4580 incumbent.
8. A statement on whether the intervention belongs to scan-only inference, CAD-assisted operator support, or both. It must not silently weaken the zero-shot scan-only claim.
9. If no bounded geometry intervention can plausibly bridge the current gap, identify the smallest scientifically necessary capability opening instead of proposing another threshold search.

No rescue tuning, post-hoc threshold adjustment, identity routing, expected-count conditioning, or benchmark-feedback loop is authorized.

## Evidence paths in the attached package

- `2026-08-22_f1_cad_assisted_production_readiness_v9.md`
- `2026-08-22_f1_cad_assisted_zero_shot_e2e_result.md`
- `2026-08-22_f1_step_topology_targeted_e2e_result.md`
- `2026-08-22_cad_assisted_production_v9_gpi.json`
- `2026-08-22_cad_assisted_production_v9_visual_audit.json`
- `FUSION_GOLD11_DIAGNOSTIC_V9.json`
- `MULTISOURCE_GOLD11_DIAGNOSTIC_V9.json`
- `VERIFIER_AUDIT_V9.json`
- `LINE_COMPARISON_V9.json`
- `contact_sheet_01.png` through `contact_sheet_05.png`
- source and evidence SHA-256 manifest
