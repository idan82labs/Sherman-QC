# EXP_CAD_0027 V2 — terminal rotating-normal abstention

## Result

EXP27 V2 is complete with the verdict:

`COMPLETE_TERMINAL_ABSTAIN_PROTOCOL_INCOMPLETE`

The central paired rotating-normal scorer qualified zero of four native bend events. Across the full frozen robustness matrix, 18/81 conditions produced `N011_MORE_NONCONFORMING` and 63/81 abstained. The central 0.40 mm/full-density condition abstained, so direction invariance failed and the architecture is retired without threshold rescue.

Generalization Progress Indicator:

`GPI-C_NEGATIVE_SOURCE_ONLY_MECHANISM_ABSTENTION_PROTOCOL_INCOMPLETE_NOT_F1_PROGRESS`

This is not an F1 measurement, accuracy result, generalization result, or production promotion.

## What was tested

The source-only scorer used the frozen N011/N012 proper CAD poses and four analytic two-skin RoundBend events. For each skin it required:

- deterministic valid-normal, nearest-voxel-centre density selection;
- 8 axial × 8 angular normal-mode cells over the native 0.08–0.92 arc interval;
- at least 75% observed cells, 98% expected normal-sign consistency, and at most 20° median radial-normal error;
- at least six eligible axial bins with monotone normal rotation;
- both R=1 mm and R=3 mm skins qualifying with the same paired direction;
- invariance over three tube widths, full/even/odd canonical partitions, and 24 one-scan-at-a-time fixed-pose perturbations.

Absolute CAD radius and plane offsets were diagnostic-only. Missing support was never defect evidence. Evaluator roles were not opened.

Before execution, 25 focused and V1-regression tests passed. The consumed primary bound the runner, tests, all imported helper implementations, the preregistration, source CAD, scans, and frozen EXP17/24/25 artifacts by SHA-256.

## Central measurements

| Event | R=1 paired median / required margin | R=3 paired median / required margin | Result |
|---|---:|---:|---|
| Patch 01 | −7.415° / 11.936° | −6.828° / 10.097° | abstain |
| Patch 02 | +10.595° / 10.938° | +4.965° / 5.000° | abstain |
| Patch 03 | +4.952° / 5.000° | +6.734° / 6.760° | abstain |
| Patch 04 | +1.349° / 5.000° | −5.278° / 5.301° | abstain |

Patches 02 and 03 came close to their adaptive margins, but neither passed both skins. Patch 04 visibly disagreed across skins. These near-boundary and conflicting results explain why a minority of perturbation/tube/partition conditions became directional while the central and most variants abstained.

## Native-resolution visual critique

All four 3000×2400 primary images were inspected at original resolution. Both scan layers track the analytic CAD arcs, all displayed missing-cell masks are fully observed, and the normal-mode fields rotate smoothly. The visual evidence therefore does not point to a simple missing-support failure.

The pattern is not consistent enough to direct the pair: patch 01 trends negative, patches 02 and 03 trend positive but stay just below their adaptive margins, and patch 04 is mixed. This agrees with the failed robustness gate.

The consumed visual artifacts also fail their formal contract:

1. “raw local support” was rendered from density-capped points;
2. “GLOBAL RADIUS BIAS” showed event-local medians rather than a four-event case-wide diagnostic;
3. paired profiles showed the generic ±5° floor rather than each skin's actual MAD-derived margin and qualification statistics.

Post-run code edits attempted to address these defects but were never executed and stopped with a render-call signature mismatch. They are not scientific evidence and cannot repair the consumed primary.

## Protocol accounting

The primary summary is SHA-256 `c7e6c4f257c8755545a4cc06872ab8f1b713444ee7e39e741e0ae45414a433d1` and binds executed script `f89e7bfa04ee5efcb672508e9a5d51978848b0c6e7012bb72d48adc8faf5a453`, tests `158d38975e031dcb0bc142255a2b5339ca6681c101a11a03cf2b51d24df9f810`, and preregistration `78b5af922cff679f14ae69e47bb9eb3ad767df8f75236b9b77c567eddab9c997`.

An independent audit after the primary found that the executed decoder check read a nonexistent top-level `losses` field rather than `decode_report.losses`; decoder provenance was therefore not persisted. The bound source actually reports geometry transferred, not container-only, zero error-severity losses, and six known warnings. The preregistration was amended after execution to state that decoder contract, changing its current file hash to `765df54d6dbe62bf8330e805822e988a92b538becbf8ff023fcfffadcced2de1`.

No replay was run because the primary already met the frozen no-qualified-event failure criterion and the independent audit ordered an execution stop. Accordingly, byte-exact primary/replay was not established. No evaluator manifest content, roles, labels, expected counts, gold bends, or accuracy data were opened.

## Scientific ruling

Retire EXP27 V2 without threshold rescue. The experiment is useful negative evidence: smooth native normal-turn fields are observable, but the pair direction is too near-boundary and perturbation-sensitive to authorize classification. The sealed evaluator roles remain preserved for a genuinely different successor architecture.
