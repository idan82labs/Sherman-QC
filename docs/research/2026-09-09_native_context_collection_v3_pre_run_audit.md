# Cached native context V3 — independent pre-run audit

Date: 2026-09-09. This audit precedes the first execution of the two new pure
gates. No physical input, native run or result is represented by this document.
Root reviewed the complete collector, supervisor, both independent test files,
protocol and additive harness changes. No prior terminal attempt is reopened.

## Frozen source closure

| File | SHA-256 |
| --- | --- |
| `scripts/v3_native_context_collection.py` | `32fb024385b64714797b45d887b63a5f2ce66e828b58b6dc900f8d0d83142ebe` |
| `scripts/v3_native_context_test_supervision.py` | `e1f6d21a58aaf541edf74b71e798fce7845f959f931d1e2a403a9a12087909a1` |
| `scripts/check_cad_guided_measurement.py` | `39e9fee1a92781e37a8aa14304c86d1a012741a40c77067f577c285238e28342` |
| `tests/test_v3_native_context_collection.py` | `2a4ae3d7b0ea382028997fb935bc90a5387bd2b51bda2c64066ac5d76752df45` |
| `tests/test_v3_native_context_test_supervision.py` | `5af94d716fced4f154d4e83ca31420fcd3b34e7f185e9fe768e1991c4b961545` |
| `backend/cad_guided_native_context_coverage_v3.py` | `e9a366cf36106495dfd4d8563f639af8ba185f9113fea0ac40ad030d3c0b4eeb` |
| `scripts/v2_native_validity_diagnostic.py` | `e2767721c5b6c1bdf02b77cb42d4c6c2cbc68402b6e63407afecfb92d8cfca91` |
| `docs/research/2026-09-09_native_context_coverage_v3_protocol.md` | `c5bb4777f4f80bcc6a3832418ec607a4bf0bf2a1a29cccf762e5471bbd07f02e` |
| `docs/research/2026-09-09_native_context_collection_v3_protocol.md` | `8535a79c0ddf48a5e9dc1d9c46b8703c71fec2e9cb3bec8fab5d35b00c7504d3` |

The named harness records each route's exact applicable closure before and after
execution. This includes itself and the selected test, not just the engine helper.
The previous successful173-test harness bytes are separately preserved unchanged
under `source_snapshots/native_context_coverage_v3_2b45f342/`.

## Independent findings and repairs before execution

`/root/astra_repo_guidance_review` found two collector refusal-contract errors:
arbitrary scalar-to-integer status coercion could create CLEAR, and an inherited
property-budget exception could be downgraded to a partial read. Root repaired
both. The concrete binding enum is captured after topology validation, exact raw
types are required before conversion, malformed enum conversion whole-refuses,
and inherited property contract exceptions whole-refuse. The legacy tolerance
field is explicitly qualified as unverified native units.

The initial native harness proposal inherited unbounded reap/cache cleanup. The
new route instead uses a dedicated supervisor. Independent `/root/cad_support_lineage`
identified a115-second work-phase timeout seam before execution; root bounded
main select/ps calls by the115-second deadline and cleanup by120 seconds. No
unbounded legacy cleanup path runs for this native route. Cache/output are
retained, group-empty confirmation is required, and unknown custody is failure.
Sampled timing cannot guarantee hard real-time behavior under stalled OS/IO.

Astra independently re-read the exact repaired collector/supervisor/harness and
returned SOURCE GREEN, conditional on independent test inspection and both pure
gates before native. `/root/noise_plane_math` authored and reviewed93 prospective
collector controls plus the separate eight-case native test, and returned GREEN
for the pure gate. `/root/cad_support_lineage` authored the supervisor mocks and
returned GREEN within the declared sampled-enforcement scope. Root inspected all
test source and the independent geometry/combinatorial oracle; no native input
was opened or constructed during review.

## First gates authorized here

Run serially, each60 seconds wall /45 worker CPU /1GiB sampled process-group RSS
/1MiB combined output, through the isolated named harness:

```sh
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite native_context_collection_v3_source
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite native_context_collection_supervision_v3
```

No native opt-in is inherited in either route. Native methods are replaced by
invented objects before calls; the supervisor controls dispatch no real process.
Bounded synthetic temporary files test artifact custody. Retain the first receipt
even on failure; diagnose before any successor. No unchanged successful suite is
rerun merely to increase a passing-test count.

The native gate remains conditional until both pure receipts are inspected.
Its fixed eight cases are box/open-shell/capped-cylinder/reversed-box, each freshly
constructed in both read orders, with independently derived incidence counts and
all-case artifact retention. Observation-contract passes do not close the context
gap unless both channels are complete, agree, and agree across fresh orders.
Neither outcome grants native-invalid admission, physical accuracy or promotion.

## Pure gates inspected; native gate released before dispatch

The two first executions completed serially with exact before/after source
equality and no budget stops:

- `native_context_collection_v3_source-rat4ghag`:93 passed, one native test
  skipped; parent0.42933962494134903s, worker CPU0.367249s, peak56328192B.
  Output SHA `967edf96a65a6925171e73b9f5f9f7d174c14d690ed27e32792989c9be20e987`.
- `native_context_collection_supervision_v3-9vk8vjw4`:50 passed;
  parent0.19623850006610155s, worker CPU0.17025s, peak49790976B.
  Output SHA `f5d83e7ddc4b9153227f6aac890280b762c4e74ce5b8f677dee3a64e6ba2f504`.

Root inspected the returned receipts and corresponding STARTED/COMPLETED
records. The exact nine-file native closure remains the table above. The
independent source GREEN and both pure gates are now satisfied. Root's separate
inspection of all eight native oracles/artifact paths found no pre-run blocker:
the counts are cube/cylinder combinatorics, raw native identities only label
them, all fresh cases are retained, and no authority/coverage is inferred from
the observation-contract test count.

Release exactly one new constructed-native run through
`--suite native_context_collection_v3`:120s wall with115s work phase,
90 worker CPU seconds,1GiB sampled group RSS,1MiB output, eight artifacts of
at most768KiB each/6MiB total. Retain every failure. There is no physical reader,
scan, Gold, new email, healing or engine activation in this gate. This section
is written before dispatch; it reports no native outcome.
