# EXP_CAD_0035 terminal source-free memory-differential nonqualification

## Verdict

`EXP_CAD_0035` is complete and terminal with status:

`COMPLETE_TERMINAL_PRE_PHYSICAL_SOURCE_FREE_MEMORY_DIFFERENTIAL_NONQUALIFICATION_NO_RERUN`

The corresponding progress indicator is:

`GPI-C_NEGATIVE_SOURCE_FREE_MEMORY_DIFFERENTIAL_NONQUALIFICATION_NOT_F1_PROGRESS`

This is a source-free, exact-scale systems result. It is not an accuracy, F1,
physical-generalization, or production-promotion result.

## Frozen gate result

The preregistered six-trial schedule completed exactly once in the order
legacy, sequential, legacy, sequential, legacy, sequential. All six fresh
children returned zero, stayed below their applicable wall/RSS ceilings, and
published valid canonical evidence. The sole failed gate was the required
comparative median process-tree RSS reduction.

| Quantity | Exact result |
|---|---:|
| Legacy peaks | 945,586,176; 946,176,000; 945,963,008 bytes |
| Legacy median | 945,963,008 bytes |
| Sequential peaks | 943,947,776; 944,209,920; 944,570,368 bytes |
| Sequential median | 944,209,920 bytes |
| Observed median saving | 1,753,088 bytes (1.671875 MiB) |
| Required saving | 33,554,432 bytes (32 MiB) |
| Gate shortfall | 31,801,344 bytes (30.328125 MiB) |
| Fraction of required gate achieved | 5.224609375% |
| EXP34 physical RSS overage | 21,741,568 bytes (20.734375 MiB) |
| Fraction of EXP34 overage covered | 8.063300678% |
| Shortfall even to EXP34 break-even | 19,988,480 bytes (19.0625 MiB) |

The failure class is `MEDIAN_RSS_DELTA_BELOW_PREREGISTERED_MINIMUM`, and the
gate result is `FAIL_COMPARATIVE_MEDIAN_RSS_DELTA_GATE`.

## What passed

- Exact-size Open3D boundary evidence passed before measured trial 1.
- The frozen EXP34 legacy wrapper ran exactly once per legacy child; the EXP35
  filtered wrapper ran exactly once per sequential child.
- EXP31 reports, pair actionability, retained array bytes, nested EXP30
  semantic/visual payloads, and EXP32 counter projections were identical.
- Scientific projection SHA-256 was `74acd30e9729f84e3aea84735bc0750a59bd81c6971f88c2f7666cfc08406160`
  for all six children.
- Nested payload projection SHA-256 was
  `d6e89faea925b756802bcc6dc95e65132f9170283818cf2c7b3e6a90c405cf72`
  for all six children.
- Runtime fingerprint SHA-256 was
  `012fb70ae932f394037465d2c79d73ac9db66f8d5911b0a375b93439008f99ad`;
  the exact loaded-module map SHA-256 was
  `972478e73688009b6387de36866f30542297b1e43eee52c3e6dc3a2ee8f4afcf`.
- The consumed code/fixture map SHA-256 was
  `7974966789078ec6606b9ed2e4da179f743b4e1ffd97eeab277afd084c7492df`.
- Legacy result bytes were identical across all three legacy trials, and
  sequential result bytes were identical across all three sequential trials.
- The exact source-free regression surface passed 349/349 tests, with frozen
  test-ID-list SHA-256
  `39a1a9c33e7a48289a6c038310d05c08a78bac6c882806d2f889ae7e7135255c`.

The maximum observed child wall time was 5.418802707921714 seconds. No trial
timed out, exceeded its applicable RSS ceiling, encountered a sampler failure,
or returned nonzero.

## Bound source-free artifacts

- Preregistration SHA-256:
  `462274a567cd0ed84b041b8e0576f93ceb1ce6b4df9abe75e63c15ec1f268cb2`
- Fixture manifest SHA-256:
  `c1435f585052220b89946874d1362d042c3c5a1ed09f32d11c097006ce9dd2cf`
- Boundary N009/N010 SHA-256:
  `1e9988770b591c108dcf3906da541c39abb041380ed0e651e831ee29e5094344`,
  `5c3ab0e5f61ebd71dd1076a0844147c3a6e3f39c6dfd867d808f74291c5f0aa5`
- Maximal-retention N009/N010 SHA-256:
  `071c1ffc28d22050f515a65bf6aae2c9673e0bb6c5c3d3fa39d207b92f04c5ff`,
  `77053d60072f72ff9c6eaa9657be3589ceef5e42a9c0f002695d3f91247ca58e`
- Terminal qualification aggregate SHA-256:
  `99fa8eaaaac64bdd19f34c9358df60ccdb16da1c03c4a4bded98cfa1c7ef906d`

The fixture tree contains exactly three directories and five files. The
qualification tree contains exactly seven directories and thirteen files.
Neither tree contains a symlink, special file, hidden residue, or extra entry.

## Interpretation and terminality

The sequential lifetime assertions are true, but the process RSS allocator
high-water behavior erased nearly all of the expected working-set advantage.
The mechanism therefore did not provide the preregistered margin needed to
justify another physical one-shot after EXP34 exceeded the 4 GiB cap.

The canonical qualification root has been consumed. The frozen protocol
forbids retry, replacement, fixture reduction, cap extension, or same-identity
rescue. No EXP35 environment/test attestation, prelock approval, immutable
physical lock, detached physical approval, or physical attempt may be created.

Physical scan hash calls, physical point-cloud loads, physical byte-open
operations, and evaluator accesses were all zero. No physical scan path or
evaluator input was supplied to any qualification child.

