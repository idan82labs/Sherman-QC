# Original thin-part diagnostic at physical V2 configuration

Date:2026-09-10. Prospective new constructed diagnostic, not physical-data access
or advancement of the failed connected V9 gate. That gate stays terminal under
its unchanged240s CPU limit. This experiment asks whether the original thin-part
nominal geometry can produce output under the exact physical-run configuration.
It is a different operating-condition diagnostic, not a rescued V9 rerun.

## Fixed question, source and two cases

Test `tests/test_cad_guided_mesh_physical_config_constructed_v9.py` executes the
actual V9 byte workflow on the original independently constructed native STEP
and mesh PLY: asymmetric1mm-thick R5/R6 strip,20/29mm flanges,30mm extrusion,
original independently posed2752 rows and5080 triangles. Two nominal V1 cases:
unchanged clean scan and the original bounded3D-noise scan (<=.001mm displacement).
No missing-arc/r0 case, thickening, extra rows, changed pose or easier geometry.
No true pose, reference geometry, row labels or fitted answer reaches inference.

The complete CONFIG expression is copied from the frozen physical V2 runner,
compared by AST without importing/executing that program. This includes epsilon
.05mm, free_circle fit, requested2048/8192 rows,30+30 iterations,2mm correspondence,
.5mmP95,.75/.25 coverage and every existing work bound. Only additive V9 types
and explicit certificate caps(1.024billion initializer,128million surface query,
8billion registration) differ. Constructor snapshots must match every field.
Do not replace the exact configuration with synthetic defaults.

The post-inference free-circle oracle is the original nominal oracle with exactly
two fit-model/role string substitutions. Original geometry/error calculations and
values remain unchanged: clean radius2e-5mm,sweep2e-5degrees,both endpoint errors
.02mm,pose.20mm; noisy radius.01mm,sweep.1degree,both endpoints.05mm,pose.20mm.
Actual free-circle measurements, raw-row support and selected pose are required
for a positive case. CAD nominal radius never supplies the measured answer.

## Separate completion and geometry ledgers

Both cases remain selected. Each full returned workflow, failure, partial ledger,
binding and error is retained in its exclusive case JSON before any summary.
Record separately:

- Whether inference returned and a complete bounded diagnostic record was saved.
- Whether the scheduled numerical work completed; missing fields remain unknown.
- Whether one required nominal prediction exists and passes the original oracle.

A nominal refusal is FAILED_NOMINAL_NO_PREDICTIONS, never a successful missing-arc
sentinel. Wrong count/geometry remains failed output. The native test asserts
diagnostic recording only; pytest green must NEVER be called positive geometry.
`positive_geometry_matrix_passed` is a separate all-two requirement and every
case's output/failure must appear in the result narrative regardless of test exit.
Malformed return, missing artifact, fixture error, drift or outer stop fails even
the recording contract. Preserve the negative; no retry or criterion/cap change.

The .8mm-wide third-normal cap may lose plane admission at epsilon.05. That is
a source-based hypothesis to test, not a reason to thicken the fixture or lower
the10degree proposal gate.2752 rows cannot exercise8192 effective refinement rows;
this limitation persists even if both cases produce correct geometry.

## Audit, freeze, budgets and serial execution

Root reads the full source and independently freezes the complete dependency
closure; a reviewer independent of test author/root checks exact hashes, actual
dispatch, fixture/CONFIG/oracle invariants, failure semantics and source count.
No source/protocol/common-harness changes between the two started gates.

Source expected21PASS plus exactly one declared native skip.60s monitored wall,
45s workerCPU,1GiB sampled process-groupRSS,1MiB stdout. After independent exact
pre-run GREEN, root executes only this route once:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite mesh_physical_config_constructed_v9_source
```

After retained source post-audit and unchanged-closure native release, expected
22PASS for recording (not necessarily geometry); one actual two-case native
diagnostic with300s monitored wall,240s workerCPU,4GiB sampled groupRSS,1MiB stdout:

```bash
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite mesh_physical_config_constructed_v9
```

Two complete artifacts <=16MiB each/32MiB total. Fresh retained ARTIFACTS directory,
exclusive STARTED/OUTPUT/COMPLETED, complete before/after source hashes, isolated
HOME/TMPDIR and numerical threads1. Only this native route enables
SHERMAN_RUN_INVENTED_MESH_PHYSICAL_CONFIG_CONSTRUCTED_V9 and supplies
SHERMAN_MESH_PHYSICAL_CONFIG_CONSTRUCTED_V9_ARTIFACT_DIRECTORY.
One heavy geometry/native job at a time. No physical scan/CAD, Gold, prediction
payload, held-out/email, services, network or application startup.

The shared supervisor's monitored-wall interval is not a guaranteed total-parent
deadline: initial/final hashing/receipt IO and unbounded legacy reaping/cleanup
remain outside the interval. RSS is sampled, not OS isolation. Worker CPU signals
remain resource failures even if parent budget_stop is null. Report these facts
and actual exits; absence of a parent stop is not evidence of successful work.
The previous supervisor21091254 is preserved byte-identically before these routes.

Independent retained-only post-audit must classify diagnostic completion AND
positive geometry separately. This does not qualify V9 capacity, the physical
cohort, stable datums, scanner calibration, historical comparability, F1/count
targets, production or floor release. Use the outcome to diagnose the operating
range; continue separately versioned computational recovery and a newly frozen
all11 development comparison rather than an unlimited synthetic prerequisite loop.
