# Plane-search V5: first pure-gate pre-run audit

Date: 2026-09-09. Recorded before first execution. **GREEN for the named pure
invented-plane gate only**; no native, physical, registration or promotion claim.

## Frozen closure

| File | SHA-256 |
| --- | --- |
| `backend/cad_guided_plane_search_v5.py` | `13898d4d736ba99be951ad228e19d47d8848082001786240a0b8a212d6ab9b98` |
| `docs/research/2026-09-09_plane_search_v5_protocol.md` | `2b0983d9dc6612147d5b1ecc45303f44f819bd2f7f14ed5e30c3b366af4257b0` |
| `tests/test_cad_guided_plane_search_v5.py` | `add36429382e4a5ce50775ef9fd61b068838a083e466493452dcd3f881944c22` |
| `scripts/check_cad_guided_measurement.py` | `b11c0185acfe8a766241333d702fc32efabbd6b847b6d871a062b116dc086c90` |

Author `/root/cad_measurement_integration_review` returned stable source after
repair. Independent reviewer `/root/astra_repo_guidance_review` fully read the
source/protocol, derived the rigid-transform conditions and reviewed the exact
repair bytes above as source-GREEN. Independent test author `/root/noise_plane_math`
read the repaired source and supplied the complete 582-line test file without
execution. Root read the complete source, protocol and all final test fixtures,
and inspected the five additive harness memberships. There is no application
or native import in this test dependency closure.

The successfully executed noisy-extraction harness was byte-preserved before
the new route in `source_snapshots/noisy_mesh_planes_v5_61a04fdd/`; its hash is
`61a04fddcd84f79856a24871f8ccb05c9593c3a0f819eb6c03c85e3cc46996c2`.
It is an archival source copy, not a relocated runnable entrypoint.

## Independent findings closed before first execution

Late failures previously bypassed export of attempted CAD participation, leaving
false zero counters/NOT_REACHED labels. The source now publishes the accumulated
usage on success or ordinary failure. An active triplet is EVALUATING before
its join and explicitly INTERRUPTED after fatal failure; unstarted triplets
remain separate. Every usable pose is invalidated on that failure.

Nominal Wahba rejection is now labeled a witness failure, not feasibility failure
over heterogeneous cones. The fixed counterexample admits identity under the
declared cones but rejects the nominal least-squares rotation. Unavailable nominal
intersections also have neutral labels; a numerical solve-residual failure is
not proof of geometric impossibility. No search quota or geometric threshold
was changed by these source repairs.

## Frozen checks and scope

The 77 prospective controls include three actual asymmetric triad poses/scales
using an independent rational rotation; exact signed proper-rotation alternatives;
anchor reparameterization, joint coefficient scaling, sign-and-offset reversal,
unmatched partial centroids and independent frame changes. Imperfect nominal
congruence must preserve intrinsic-intersection origin covariance. A deliberately
perturbed solve must retain its nonzero residual and refuse the witness.

The heterogeneous-cone and reflected-unoriented-triad controls preserve known
limitations rather than demanding an invalid all-reflections rejection. A
30-CAD-plane scalar oracle reconstructs each searched interval/mask decision:
435 pair records, 300 indexed-window entries, 12 searched pairs, 288 unsearched
entries, 240 compatible unsigned occurrences and 236 unscheduled occurrences.
These are not claimed exhaustive correspondence or pose coverage.

All fourteen hard-budget maxima, nine tiny exact/one-below preflights, zero
quotas, typed snapshots, late SVD participation and whole-report overflow are
checked. Tests must not be narrowed, retuned or reinterpreted after execution.
Retain the first output and terminal receipt, including any failure. A repair
requires a diagnosed, separately recorded successor decision.

## One serial execution

```sh
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite plane_search_v5
```

Limits are60s parent wall,45s worker CPU,1GiB sampled process-group RSS and1MiB
combined output. No other geometry worker runs concurrently. Before/after hashes
must match the four-file closure. Any stop, source drift or test failure blocks
advancement. No runtime result is claimed by this pre-run document.

A pass qualifies declared-plane candidate search only. Actual noisy mesh support,
finite assigned faces, whole-part registration and scan-derived measurements
still require connected validation. No physical F1, calibrated uncertainty,
observed count, global uniqueness or factory-floor qualification follows.
