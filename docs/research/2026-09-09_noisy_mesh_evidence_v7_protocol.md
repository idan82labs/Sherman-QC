# V7 exact float64 metadata normalization — prospective protocol

Status: **SOURCE ONLY; NOT EXECUTED.** This is a narrowly versioned repair of the
V6 metadata boundary. Root owns independent test authorship, Astra source review,
source binding and every execution gate. V5/V6 source, tests, protocols and
terminal outcomes remain unchanged. No native or physical input is accessed.

## Retained V6 negative and identified mismatch

Root reported `noisy_mesh_evidence_v6-7ltfj07_`: 78 passing and 13 failing checks.
Actual V6 extraction attempts refused with `UNSUPPORTED_METADATA_VALUE` before
their first core was appended; the paired V5 calls on the same arrays completed.
That is a failed V6 gate, not positive geometry or capacity evidence. It is not
rerun or retrospectively reclassified by this successor.

The source-visible mismatch is narrow. Frozen V5 `_analyze_support` obtains
`u = np.finfo(float).eps`; downstream numerical-guard values can therefore have
exact type `np.float64`. V5's ordinary JSON serialization represents those values
as JSON binary64 numbers. V6 `_scalar_size` deliberately accepts only exact
builtin scalars, so it rejects this actual V5-generated numeric type.

The intermediate Python tree returned by V5 `_metadata` still contains these
NumPy float64 objects. Consequently V6's claim of preserving that intermediate
tree's exact Python types was not the correct wire-level contract. V7 explicitly
targets **V5's actual serialized JSON metadata tree**, with array descriptors
and tuples-as-lists, rather than that intermediate object graph.

## Exact repair and API

The new file is `backend/cad_guided_noisy_mesh_planes_v7.py`, copied from the
frozen V6 implementation, SHA-256
`1f5fe2fd447c445efd8c0e12f9947bfe484126fd15b2fe1be1149f2d5baf2f72`.
The unchanged V5 dependency is SHA-256
`9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75`.
In `_Codec.encode`, immediately after its existing
single `work.visit(depth)` call and before container/scalar dispatch, it applies
only this normalization:

```python
if type(value) is np.float64:
    value = float(value)
```

This changes only the local encoder variable. The exact finite IEEE binary64
value, including negative zero, is preserved. There is no float32 conversion,
decimal formatting/rounding, fitting, tolerance change or array rewrite. All
internal raw record values retain their existing types and bits. Array descriptor
hashes continue to bind the same array bytes. The numeric wrapper consumes the
same one node visit as a builtin numeric value; no recursive normalization or
extra visit is introduced.

Public names are `extract_noisy_scan_plane_proposals_v7`, `encode_metadata_v7`
and `decode_metadata_v7`. The private export seam remains `_metadata`. Extractor
schema becomes `noisy_mesh_planes_v7`; its logical metadata description becomes
`V5_JSON_WIRE_METADATA_WITH_ARRAY_DESCRIPTORS`. The wire grammar and format stay
`noisy_mesh_metadata_schema_v6`: this is scalar normalization into the existing
grammar, not a new tagged representation.

Only exact `np.float64` receives the encoder conversion. There is no generic
NumPy scalar `.item()` conversion, `np.generic` dispatch, `isinstance` expansion,
or user-defined numeric-subclass hook. Float32, long double, complex, object
scalars, integer subclasses and arbitrary float64 subclasses remain unsupported.
Nonfinite float64 values normalize to builtin nonfinite values and then fail the
unchanged finite-value guard; they are not repaired into a number.

The decoder remains unchanged except for its public function name and clarified
docstring. It accepts only the strict builtin JSON tree. An in-memory envelope
containing `np.float64` instead of a builtin float still refuses; the encoder is
the only normalization boundary. JSON parsing or this decoder does not reconstruct
original NumPy arrays, numeric wrapper types or source authenticity.

## Everything else remains fixed

V7 imports the same frozen V5 computation helpers. The numerical equations,
original-row/triangle partitions and supports, schedules, closure alternatives,
failure behavior and unknown/production flags are unchanged. No geometric
computation from V5 is copied and modified. The inherited V5 policy/budget classes
and validation stay in use, including default 128/1024-triangle tiling, one-ring
halos, caller-stipulated epsilon and the incomplete 16-core closure schedule.

All V6 codec limits remain fixed:

| Quantity | Limit |
| --- | ---: |
| Encoded metadata bytes | 67,108,864 |
| Expanded compact V5 JSON-equivalent bytes | 268,435,456 |
| Schemas / keys per schema / key characters | 256 / 128 / 128 |
| Scalar string characters / integer magnitude bits | 65,536 / 4,096 |
| Logical recursion depth | 64 |
| Cumulative encoder node/key visits | 64,000,000 |
| Cumulative array bytes hashed | 4,294,967,296 |

The same numeric 64 MiB encoded cap is still applied to the schema-table
representation, not V5's larger repeated-key representation. This is not a claim
of unchanged V5 expanded-JSON admission. There is no cap increase.

One shared interner remains in incremental admission; each appended record and
each newly introduced schema are charged before publication. Final export uses
a fresh deterministic schema table. Exactly three final passes remain charged
to the same extraction work ledger, with predicted traversal/hash totals verified
against actual totals and byte counters required to converge. All late-mutated
fields remain included. A scalar, budget or final-accounting failure clears the
whole candidate output, not a successful prefix. No rejected record is dropped.

## Independent prospective verification

Root will separately author a V7 test successor while preserving the failed V6
suite. Required controls before execution are:

- Exact finite float64 encoder values, including both zero signs, subnormal and
  normal extremes, are represented as builtin floats with identical binary64
  values. Nonfinite values still refuse, and unsupported scalar/subclass types
  remain rejected.
- The decoder remains strict and rejects in-memory non-builtin numeric values.
- The logical oracle models V5's real JSON serialization: exact float64 values
  become builtin floats while arrays become their unchanged descriptors. It does
  not falsely require the intermediate NumPy wrapper type to survive JSON.
- The internal-geometry comparator separately requires the actual V5/V7 record
  scalar types and binary64 bytes to be unchanged, along with every original
  array, partition, support, analysis, candidate and closure/schedule result.
- Every V6 grammar, hard-cap, whole-refusal, mutation, shared-interner and final
  counter control remains in scope. The new scalar normalization must not alter
  work counts relative to the equivalent builtin scalar or enlarge any bound.

The first gate remains root-owned serial supervision: 60 seconds wall,
45 seconds worker CPU, 1 GiB sampled process-group RSS and 1 MiB output. No test,
import, native library or physical input is executed during this source task.
The full-size capacity case is not enabled by this document. Any later capacity
successor requires its separate exact frozen review and the unchanged
120-second wall / 90-second worker CPU / 4 GiB / 1 MiB envelope, one heavy job.

Passing this repair would establish only the corrected serialization boundary
and the specified paired invented checks. It would not erase V6's failed gate,
qualify full-size capacity, calibrate noise, improve physical count/F1, activate
product defaults or authorize production-floor acceptance.
