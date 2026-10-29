# V72 CASL Amendment A1 terminal retirement

## Disposition

`V72_CASL_TERMINAL_RETIRE`

The single top-level lock invocation authorized by `V72_CASL_SOURCE_FREE_LOCK_AMENDMENT_A1` completed on 2026-08-16. The primary suite and its internal replay were byte-identical at **41 passed, 2 failed, 0 errors, 0 skipped**. Amendment A1 requires exact 43/43 primary and replay and states that any deviation terminally retires V72 with no fixture correction, threshold edit, implementation rescue, or replacement lock run. V72 is therefore retired.

No U01-U04 input, native-resolution visual artifact, reserved physical scan, gold, label, CAD, expected count, identity, filename, runtime/main, production, or benchmark-feedback artifact was opened.

## Immutable inputs

- Original V72 ruling SHA-256: `31ea25b5c572f0611620aedc9a95aa212a77bf8b545a418a3b63429513386b8b`
- Amendment A1 SHA-256: `d7ec349d5aec67c9e8a1b2e4a73ce3be82b7275c487b598d2ac409963712df5a`
- Production CASL implementation SHA-256 before and after execution: `b0da73bd3fc59fdab968d1bbe87d27f0a4dc215abca48e199589b81f6f01071a`
- Original failed fixture snapshot SHA-256: `8d7ddece77b4f56b17883e8b8f0ae436ce5a38bc301572bdb7ec9e33e8233147`
- Original failed JUnit SHA-256: `e609c9f6221448e7017d667a49470be8cd7ca41a43b1b9adc2185fc7fbe4cde8`
- Amended fixture SHA-256: `6fcc73d6ec998fc96a31de1106c60d8b004d0beb0e1b251821fb1fde78c3a2d1`
- Single-invocation lock runner SHA-256: `edc0656ef56437333bfcc40537fa47a4d6d506d66e72cfd1a94903ed99d8654d`
- Precommit manifest SHA-256: `892c8c7f2c81c64cacaf209218daa5ae8ecf4d52915e04b92d4e87939dd22bf4`

## Authorized invocation

```text
.venv/bin/python scripts/run_f1_v72_casl_source_free_lock_amendment_a1.py --output-root output/research/f1-v72-casl-amendment-a1-lock-20260816 --precommit output/research/f1-v72-casl-amendment-a1-precommit.json
```

This command was invoked exactly once. It executed the 43 ordered primary cases and then the 43 ordered internal-replay cases.

## Result hashes

- Summary SHA-256: `7c3f3c7d34b504557972ceb430c23804069d443ac786b12652bc4ab0067d0f56`
- Primary canonical result SHA-256: `8e611f3adec86e64a9de6126ab3fe5f67fdf0f33e8b04c4e7a7361c03e7b7496`
- Replay canonical result SHA-256: `8e611f3adec86e64a9de6126ab3fe5f67fdf0f33e8b04c4e7a7361c03e7b7496`
- Primary JUnit SHA-256: `80afb2d006870409fc6500033be3771679c59a1f54f20d8b4b0bb3bf4aa3362a`
- Replay JUnit SHA-256: `108c0fb0a059ebcc2f37ac27cea027e9812b556d693248f83e3f7f9d81702965`

The canonical primary and replay artifacts are byte-identical. JUnit hashes differ only because JUnit contains per-run timing metadata.

## Deterministic failures

### G10 removed-tail query contradiction

The amended G10 source contained exactly 199,826 records, satisfying the real source cap. The mandatory removed-tail audit then found **1,512 removed analytic tail records** inside the unchanged station-annulus queries of candidate `G10-B`.

This is a structural contradiction, not a threshold miss: G10 places two rounded bends 24 units apart in one source-only point set. CASL has no candidate-specific source ownership. Tail points removed from one bend can therefore lie inside the other candidate's 72.75957614183426-unit outer query radius. Amendment A1 explicitly states that any nonzero removed-tail membership is a lock failure and may not be cured by another endpoint edit.

### Six-degree production-expression contradiction

Amendment A1 supplied the boundary `0x1.acee9f37bebd5p-4`, equal to binary64 evaluation of `(6*pi)/180`:

```text
0.10471975511965977
```

The frozen production implementation evaluates `6.0 * DEG` where `DEG = pi / 180`, producing the adjacent binary64 value:

```text
0x1.acee9f37bebd6p-4
0.10471975511965978
```

The mandatory equality assertion therefore failed before the paired direction-vector predicate. Correcting the expression, constant, fixture, or comparator is forbidden after this terminal run.

## Consequence

V72 may not be rerun, rescued, used for U01-U04, visually adjudicated, or opened on the reserved physical scan. The next research action must be a newly selected source-only successor architecture or an explicitly authorized smallest capability opening, with a fresh precommit and native-resolution visual gate.
