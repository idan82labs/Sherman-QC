# Noisy mesh-plane V5: first pure-gate pre-run audit

Date: 2026-09-09. Decision recorded before first execution: **GREEN for the
named pure/synthetic gate only**. This is not a native, physical-input,
registration, measurement, benchmark or promotion authorization.

## Exact source closure

The integrating root independently read the implementation, protocol, mathematical
notes, harness route and complete final 623-line test source. Rehashed bytes:

| File | SHA-256 |
| --- | --- |
| `backend/cad_guided_noisy_mesh_planes_v5.py` | `9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75` |
| `docs/research/2026-09-09_noisy_mesh_planes_v5_protocol.md` | `039b6a3f46e170947039ee32a8ce871f768aea781b369b244fc0e96a9965b71b` |
| `docs/research/2026-09-09_noisy_plane_registration_design_notes.md` | `03e014332e4a9072f631dd449c6c6f55710fba175dbe19af6e7349eed7562e52` |
| `tests/test_cad_guided_noisy_mesh_planes_v5.py` | `56c12020f3ae77dfafb7525b94d9ab823a9e2f630b98537ef1bac1434f715580` |
| `scripts/check_cad_guided_measurement.py` | `61a04fddcd84f79856a24871f8ccb05c9593c3a0f819eb6c03c85e3cc46996c2` |

`/root/cad_measurement_integration_review` authored the new extractor and protocol.
`/root/cad_support_lineage` independently authored and reviewed its tests without
executing them. `/root/astra_repo_guidance_review` independently reviewed the
conditional mathematics, final source repairs and additive harness route. Root
reviewed their findings and all final test controls. No physical inputs, Gold,
native library, application server or prior predictions are test fixtures.

## Findings closed before first execution

1. Core-count capacity now stops partition construction before the next block
   would be allocated, rather than after constructing all tiny components.
2. Owned point/triangle/error-array dimensions and owned triangle index bounds
   are rechecked before indexing. A caller-array mutation between validation
   and snapshotting cannot make negative indices authoritative.
3. Ordinary late computation exceptions fail closed and clear every usable
   proposal/partition/completion flag. Process-control exceptions remain visible.
4. Heterogeneous position-error energy, covariance/eigensystem guards and the
   computed-normal angular guard are retained separately. They are conditional
   engineering bounds, not calibrated confidence intervals or formal floating-
   point certificates.
5. The new harness route adds only the named pure test, five-file binding,
   60/45/1-GiB budget, isolated namespace and retained receipt. It does not enable
   native execution. Previous successful native harness bytes were preserved
   before editing the live harness.

## Frozen oracle and failure handling

Four required noisy-grid recoveries use the independently specified rational
proper rotation, three scales and translated frames in the test source. Original
raw-row and triangle identities, heterogeneous error declarations, two complete
fixed partitions and immutable retained support are checked. The required
132-triangle failed-closure case must retain its valid local core, rather than
discarding it or publishing a failed whole closure. These fixtures and expected
outcomes are fixed before execution.

The test source includes independent combinatorial/work accounting, noisy true-
plane cone checks, a TLS feasible-plane/witness-failure counterexample, strict vs
touching vs disjoint strip intersections, separately possible but jointly failing
closure points, computed-normal perturbation, near/absent eigengaps, negative
cone numerator, exact-axis versus rotated zero-error behavior, holes, duplicated
XYZ with distinct raw identities, disconnected small components, unscheduled
cores, ownership mutation and early/late failure controls. Large array values
remain in owned arrays rather than expanded JSON.

Every failed required assertion fails this gate. Retain the full output and
terminal receipt. Do not change a started gate's limits, retune the four positive
fixtures or failed-closure oracle, reinterpret a failure as qualification, or
silently retry the same terminal attempt. A repair requires diagnosis and a
separately recorded successor decision. The prospective 108-case study and
large-capacity qualification are not included or represented as executed.

## One permitted execution

From the Git root, with no other heavy geometry job:

```sh
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite noisy_mesh_planes_v5
```

Frozen parent wall limit: 60 seconds. Worker CPU: 45 seconds. Sampled process-
group RSS: 1 GiB. Combined output: 1 MiB. Preserve the harness's new exclusive
attempt directory and before/after five-file source maps. Any source drift,
resource stop, supervision failure or test failure blocks advancement.

A pass would qualify only this invented-array proposal contract under declared
uncalibrated position errors. It would not establish realistic scanner error
bounds, physical plane/bend count, registration accuracy, Gold11 F1 or factory-
floor readiness. The next integration gate remains bounded correspondence and
connected scan-supported measurement before a separately frozen all-case paired
Gold11 development run. Unsupported CAD cases stay retained abstentions.
