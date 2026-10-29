# Cached native context V3: contract passed, coverage gap remains

Date: 2026-09-09. Independent post-run reviewer:
`/root/astra_repo_guidance_review`. Scope: retained JSON/text evidence, named
source and bounded primary-source diagnosis only. No native construction,
collector/test rerun, physical input, cache population, healing or process signal
was performed during this audit.

## Decision

The terminal [native attempt](cad_engine_v1_synthetic_checks/native_context_collection_v3-1i3cpnja/COMPLETED.json)
is a valid **8/8 constructed observation-contract pass**, but **0/8 cases have
complete context coverage in either channel**. `context_gap_closed_on_constructs`
is false. Keyed and iterator observations agree within every case, and the four
fresh read-order pairs agree after the frozen geometric normalization.

This is a useful negative diagnostic, not a native-validity admission gate or an
engine-accuracy result. Missing evidence remains MISSING; global native true does
not replace it. Local/global/production admission flags remain false, and the
observed bend count remains null.

## Exact terminal custody

The [pre-run audit](2026-09-09_native_context_collection_v3_pre_run_audit.md)
records the nine-file closure, prior pure gates and release before native dispatch.
Its audited bytes here have SHA-256
`e29fdba8029f118eefef2255af28011e4d4e4e46112e979a95eea26ff37636e6`.
The native STARTED and COMPLETED records contain identical nine-entry source
maps, matching that table. The collector, supervisor, two test files, two bound
helpers and two protocols were independently rehashed and still match.

The executed harness was
`39e9fee1a92781e37a8aa14304c86d1a012741a40c77067f577c285238e28342`.
Its exact bytes are preserved in
[the archival snapshot](source_snapshots/native_context_collection_v3_39e9fee1/check_cad_guided_measurement.py).
After native completion, the live harness acquired only the separately requested
V5 pure-suite test/binding/budget/namespace/receipt memberships; the inspected
live hash was `61a04fddcd84f79856a24871f8ccb05c9593c3a0f819eb6c03c85e3cc46996c2`.
That is not in-run drift. The archival script's relative ROOT assumes its
original location; the snapshot is evidence, not a directly runnable copy.

Control-file SHA-256 values:

- [STARTED.json](cad_engine_v1_synthetic_checks/native_context_collection_v3-1i3cpnja/STARTED.json):
  `be45dc12d77d46e44377a0aa2e1a4440dac94db1082a4cd3489908ac34300d63`
- [COMPLETED.json](cad_engine_v1_synthetic_checks/native_context_collection_v3-1i3cpnja/COMPLETED.json):
  `ffe9f4ffab95a24c549c42fbe0dcf78cf3499c2959f70f619bb183676900ff13`
- [OUTPUT.txt](cad_engine_v1_synthetic_checks/native_context_collection_v3-1i3cpnja/OUTPUT.txt), 3,797 bytes:
  `fcf71fe0f48dd0bc52b7a06eeea7ee8170aa7608ed8064d1e052dc8b5269336f`

Exactly eight artifact files were present. Every byte count and independently
computed hash matched its OUTPUT manifest entry; aggregate size was 163,068 bytes,
below the frozen 6 MiB aggregate and 768 KiB per-case limits.

| Artifact | Fresh case / read order | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `case_00.json` | box / keyed first | 24,647 | `af72f5c80e9d76dbbab8adc8c36db288dee90bdf35fe00c10c1b7d2e0039085a` |
| `case_01.json` | box / iterator first | 24,647 | `5136b9ea097be16a768db4ef7b732684456f905c0f55457b796ef2d1a17a3954` |
| `case_02.json` | open shell / keyed first | 22,909 | `4864ce5da6cb8a48ef296e4b0eba9d2940a9973e8950f08d3e4a8c5958868a6d` |
| `case_03.json` | open shell / iterator first | 22,909 | `39f59a90afc81780dc6d5da809b63932cdbf98d41b3d9190c46b2ca333f8e452` |
| `case_04.json` | capped cylinder / keyed first | 9,322 | `c9d98acede68cdc4fc06f1a0b03f3bec8d61caa2d17ee23002d37a81618a0ed6` |
| `case_05.json` | capped cylinder / iterator first | 9,322 | `d39f466ed62082149f7e24b27361cecf1e3b1468aae017e4a0020c27b80e588d` |
| `case_06.json` | reversed box / keyed first | 24,656 | `5bff9eb10660d400cb22458cedff6baab9f5e5dfac755a11280c638ef1bfe76e` |
| `case_07.json` | reversed box / iterator first | 24,656 | `dcf626f0e574c6acac4a8b4ff9405116f5c8de62fba20e92815f11a384a2a047` |

Files are under the attempt's `ARTIFACTS/` directory. Each retains its full report,
case/order, completed phase, contract disposition and semantic observation.
No case was skipped, replaced, retried or represented by an encoding-failure
fallback. The frozen test writes and verifies the same exported report that its
oracle checks. The 94 pytest passes mean 93 pure controls plus one eight-case
native matrix test; they are not 94 native geometry successes.

Measured parent wall time was 3.470438916236162 s, worker CPU 2.756593 s,
sampled group peak RSS 323,862,528 B and worker-reported peak 323,829,760 B.
Exit code was zero, with no budget stop, primary error, cleanup error or source
drift. Work stopped at 3.45263420837 s, the child was reaped at
3.452636457979679 s and group-empty confirmation occurred at
3.470380458049476 s. The receipt's retired group identity is historical evidence
only; no signal is authorized by this note. The receipt records a retained private
cache; this audit did not open it.
These are sampled process-group/worker counters, not total host memory, hard
real-time guarantees or a maximum-cap performance qualification. Frozen limits
remained 115 s work / 120 s total monitoring-and-cleanup deadline, 90 worker CPU
seconds, 1 GiB sampled group RSS and 1 MiB combined output.

## Denominator and missing-evidence audit

All eight reports were parsed. An independent ancestor walk over every retained
occurrence row reproduced the required typed pair set, and every pair had exactly
one membership query. Entity occurrence counts matched the retained tree.
The frozen test's independent cube/cylinder combinatorics were also inspected:
box context counts are 24 vertex-edge + 24 vertex-face + 24 edge-face + 6 wire-face
and 1 shell-solid = 79; removing one face gives 24 + 20 + 20 + 5 + 1 = 70;
the capped cylinder gives 4 + 4 + 5 + 3 + 1 = 17. These denominators are not
derived from whichever cache entries happened to exist.

The following counts hold separately for **each of the two fresh read orders**.
Observed/missing counts are per channel; the channels match exactly.

| Case | Unique entities / occurrences | Expected contexts | Observed contexts | Missing contexts | Native global | Nonzero context statuses |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| box | 34 / 86 | 79 | 63 | 16 | true | 0 |
| five-face open shell | 32 / 72 | 70 | 56 | 14 | false | 1 |
| capped cylinder | 13 / 26 | 17 | 13 | 4 | true | 0 |
| reversed box | 34 / 86 | 79 | 63 | 16 | true | 0 |

Missing pair patterns:

- Box and reversed box: 12 edge-face and 4 vertex-face pairs per order.
- Open shell: 8 edge-face, 2 vertex-face and 4 vertex-edge pairs per order.
- Cylinder: 2 edge-face and 2 vertex-face pairs per order. Specifically, each
  circular rim edge and its vertex lack the corresponding planar cap context;
  the recorded missing pairs are `EBOTTOM->FBOTTOM`, `ETOP->FTOP`,
  `V0->FBOTTOM` and `V1->FTOP`.

Across eight cases there are 490 expected pair queries, 390 observed pairs per
channel and 100 missing pairs per channel. Every missing pair is explicitly
`ABSENT / NOT_CALLED` in the keyed query ledger and absent from the iterator.
There are no empty returned status lists, differing statuses, unexpected pairs,
keyed-only pairs or iterator-only pairs. Every observed context has one status.
The open shell retains `BRepCheck_NotClosed` (code 28) for shell-solid in both
channels, with the solid's subtree validity false; it is not rewritten to CLEAR.

All common self rows have `[0]`, complete metadata and no read errors. Their
counts per order are 34, 32, 13 and 34. Global collector status-value counters
are respectively 160, 144, 39 and 160: self plus both observed channels.
Membership and getter counters exactly match the independently counted queries
and observations. Coverage summaries remain incomplete solely for missing
context evidence; completed native reads do not imply completed coverage.
The reported tolerance value `1e-7` retains its explicit unverified-native-unit
qualification and is not scanner calibration.

## Bounded primary diagnosis and next source seam

This result rules out an **iterator-export-only explanation** for these eight
cases: the keyed membership API also says those exact contexts are absent.
Changing iterator serialization alone cannot recover a missing keyed entry.
The gap also occurs on constructs whose native global result is true, and it
persists in both tested read orders. These observations do not establish the
underlying installed-kernel or binding cause.

In the primary [OCCT 7.9.3 Result interface](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Result.hxx),
membership reads the cached map, keyed status access reads an existing map entry,
and iterator status reads the current entry. The
[iterator implementation](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Result.cxx)
skips the self key, not a declared class of incident edge/face contexts.

The primary [Analyzer source](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Analyzer.cxx)
contains context-calculation dispatch for the five relation classes used by the
denominator. Its later validity checks inspect cached results and move iterator
cursors. The [Edge context routine](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Edge.cxx)
and [Vertex context routine](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Vertex.cxx)
bind a context entry before their geometry checks. Therefore ordinary geometry
rejection is not, by itself, a demonstrated explanation for these absent entries.
These upstream sources describe intended behavior; this audit did not establish
byte-level equivalence to the installed native binaries or a unique root cause.

The smallest justified source successor is a **passive cache-lifecycle diagnostic**,
not an admission change: bind the actual runtime/build identity and read back
analyzer options, then record the same complete context denominator immediately
after one analyzer is constructed and after the common global/subtree/property
observations. Preserve both cached-read channels, all fixed cases, the original
missing results and every stage failure. This would distinguish constructor/cache
state from an observation-sequence effect before choosing a repair. It requires
new source, an independent protocol/oracle/budget audit and a separately retained
attempt; none is authorized or executed by this results note.

Do not relax the denominator, substitute global validity, manufacture NoError,
populate the frozen analyzer's cache, heal geometry, change the installed kernel
or retry this terminal run to make the gap disappear. Any future explicit native
context calculation must be a separately declared computation with distinct
provenance, not relabeled as an observation of the original cache. No physical
accuracy, local-feature admission or automatic promotion follows from this run.
