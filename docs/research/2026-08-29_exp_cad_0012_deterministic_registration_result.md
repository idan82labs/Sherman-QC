# EXP_CAD_0012 - deterministic CAD reference surface qualification

## Decision

**Partial infrastructure improvement; strict qualification failed.**

Generalization Progress Indicator: **GPI-C_INFRASTRUCTURE_PARTIAL_NONDETERMINISTIC**.

The deterministic area-stratified CAD surface sampler eliminated stochastic reference construction. Across independent primary and replay processes, all 13 reference point hashes and all 13 reference-normal hashes matched exactly. Datum selection and bend-local reversal event sets also matched in all 13 cases.

The complete preregistered gate nevertheless failed. Raw semantic payloads were byte-identical in 0/13 cases. Selected transforms differed only at approximately `2e-13`, far below scanner resolution, but rejected datum candidates remained numerically unconstrained. One zero-fitness rejected candidate differed by approximately 288 mm even though both runs made the same rejection decision.

## Results

| Qualification | Result |
|---|---:|
| Reference point/normal hashes identical | 13 / 13 |
| Datum selection identical | 13 / 13 |
| Bend-local reversal event sets identical | 13 / 13 |
| Semantic payload bytes identical | 0 / 13 |
| Maximum global-transform delta | `1.39e-13` |
| Maximum selected datum-transform delta | `1.94e-13` |
| Maximum rejected datum-candidate delta | `287.825` mm |
| Peak RSS | 1.143 GiB |

The sampler is scientifically useful but cannot yet be called a complete deterministic registration substrate.

## Interpretation

There are now two distinct issues:

1. **Decision determinism is effectively solved for this tranche.** Reference geometry, selected/rejected routing, and bend-local event decisions were stable.
2. **Persisted semantic determinism is not solved.** The artifact includes raw intermediate transforms and floating-point metrics. Selected transforms carry physically irrelevant sub-picometer noise; rejected zero-fitness candidates can carry arbitrary transforms that should never cross a trust boundary.

Rounding the existing result after seeing it would manufacture a pass. The next experiment must preregister a physical canonicalization contract before execution: rejected candidates emit no transform, while selected transform and metric values are serialized at fixed `1e-9` mm precision. Selection logic and bend-local thresholds must remain unchanged.

No F1, generalization, or production promotion claim follows from this infrastructure experiment.

