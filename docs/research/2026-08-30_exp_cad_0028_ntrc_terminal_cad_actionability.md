# EXP_CAD_0028 NTRC-V1 — terminal CAD-actionability result

## Outcome

`NORMAL_TURN_RIGID_CONSENSUS_V1` is terminally stopped before physical-scan execution.

The frozen source-only selector succeeded on `42637000_B`: six active native `RoundBend` features and six independently recovered unique inner/outer cylinder pairs agreed, with the expected 45/45/130/178/178/210 mm span inventory. The next mandatory CAD-only gate failed. Exactly 12 of the 24 required topology-owned tangent panels have only 1.0 or 2.5 mm of finite inward extent. The frozen guard contract needs at least 3.0 mm for the 1t inward exclusion plus 1t non-event trim/hole erosion at t=1.5 mm.

The architecture therefore cannot furnish all four finite guards for any event without changing the preregistered measurement. No erosion, support, fit, margin, or pose threshold was relaxed.

## Scientific disposition

- Final status: `TERMINAL_SOURCE_ONLY_ARCHITECTURE_INFEASIBILITY_NO_SCAN_EXECUTION`
- GPI: `GPI-C_NEGATIVE_SOURCE_ONLY_ARCHITECTURE_INFEASIBILITY_NOT_F1_PROGRESS`
- Generalization or F1 evidence: none
- N009/N010 point-cloud contents loaded: no
- Evaluator manifest, roles, labels, gold, expected counts, and quality mapping opened: no
- Pair direction produced: no
- Replay or native scan render required: no; the prerequisite failed before the scan lane opened
- Production promotion: not authorized

This is a useful architecture-level negative result: a four-plane rigid nuisance correction is not geometrically supportable on this six-bend CAD under the frozen finite-panel safety erosion. It does not say the scans lack bend information, and it does not justify reducing the erosion to rescue NTRC-V1.

## Preserved engineering result

The unexecuted prototype remains useful as reusable research infrastructure. It includes the exact six-pair selector, connected coplanar panel reconstruction, deterministic raw partition and voxel aggregation, an identifiable five-parameter guard-only Huber fit with axial translation fixed to zero, a globally oriented cylinder-normal turn statistic, within-case two-skin consensus, scan-swap antisymmetry, and the exact 39-row robustness contract.

Twenty-one preflight tests pass, including the authentic CAD-only six-pair/actionability regression. The source execution CLI remains disabled.

## Bound artifacts

- Preregistration SHA-256: `10b96fa48b05bed6848ce837839c945cc02054c91c8c83691f44ba50bf0d7b84`
- Runner/core SHA-256: `1d3c9f7e1837a17bc41a894637cb3d3052d94a33b769308b7cfbd7a0e51cb5a3`
- Test SHA-256: `18e4c0d59735f3af5a6b96f44bec55aca29ee4040e08d8001e5269086cb01e8e`
- CAD guard preflight SHA-256: `8577f81f398681a9c2558314abc5e2229d69540e3eb9e283e2bdfb6fc3255064`
- Preflight binding SHA-256: `f4a8d74c251b4453a5833942f2fcfdf8bd1dee154af195a6ac5c7fba9dff86bd`

The exact failed guard rows are recorded in `output/research/autoresearch-loop/exp-cad-0028-ntrc-v1-preflight/CAD_GUARD_ACTIONABILITY_PREFLIGHT.json`.
