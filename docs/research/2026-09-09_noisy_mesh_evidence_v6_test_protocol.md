# V6 noisy-mesh evidence: independent first-test protocol

Date: 2026-09-09. **Prospective source-only gate; not executed.** Root owns the
exact source-closure review, frozen hashes, isolated harness and sole first run.
This document neither authorizes a capacity run nor accesses physical inputs.

## Narrow question and source boundary

The independent test is
[`tests/test_cad_guided_noisy_mesh_evidence_v6.py`](../../tests/test_cad_guided_noisy_mesh_evidence_v6.py),
against the separately versioned
[`cad_guided_noisy_mesh_planes_v6.py`](../../backend/cad_guided_noisy_mesh_planes_v6.py).
The [V6 implementation protocol](2026-09-09_noisy_mesh_evidence_v6_protocol.md)
defines the grammar and resource units. The frozen
[V5 source](../../backend/cad_guided_noisy_mesh_planes_v5.py),
[V5 tests](../../tests/test_cad_guided_noisy_mesh_planes_v5.py) and
[V5 geometry protocol](2026-09-09_noisy_mesh_planes_v5_protocol.md) remain unchanged.
The V5 source SHA-256 is
`9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75`.
Root will bind this exact test, both implementation sources, this protocol,
the implementation protocol and the actual isolated harness before execution.

The question is whether the schema-table codec preserves all represented V5
metadata and whether the unchanged small geometry produces identical internal
evidence. The numerical 67,108,864-byte limit charges a **different encoded
representation** from V5 expanded JSON. Success cannot rescue or reinterpret
the terminal V5 full-size representation failure.

The test imports only standard Python modules, NumPy, pytest and the two named
geometry modules (which use SciPy sparse operations). It has no native imports,
file readers, data paths, process launch, retained-artifact reads, CAD features,
Gold, incoming email or physical input. Fixture arrays are constructed in memory.
The private fault-injection controls affect only V6 codec seams. V5 geometry,
its policy, source and original test fixtures are not monkeypatched or edited.

## Fixed resource envelope and selection

One serial pure/mock/small-geometry job: **60 seconds total wall, 45 seconds
worker CPU, 1 GiB sampled process-group RSS, 1 MiB total output**. Root's isolated
harness supplies the enforcement, source binding and durable terminal receipt.
No native/capacity opt-in is enabled. No heavy worker runs in parallel.

The source enumerates **91 prospective pytest controls**. This is a selection
count, not a passing result. All selected failures stay failures; no cap increase,
retry-driven fixture reduction, assertion weakening or geometry retuning follows
from a failed first run. No full reports are printed by the tests. Assertion
output is still subject to the supervisor's fixed output cap.

Large codec caps are tested using independently seeded private counters next to
their fixed limits, not 64 million allocated nodes, 4 GiB arrays or 256 MiB JSON.
The largest string control contains 65,537 Unicode characters; schema controls
use at most 257 one-key dictionaries. Geometry has at most 162 raw vertices and
256 triangles in one cloud. These controls do not qualify maximum-cap memory,
time, noisy scan capacity or an adversarial operating-system boundary.

## Frozen codec oracles

The independent compact-JSON oracle uses `ensure_ascii=True`, `allow_nan=False`
and separators `,`/`:`. It is independent of V6's streaming size routine.
The logical-tree oracle converts exact arrays to `{shape,dtype,sha256}` using
their complete C-order bytes, and tuples to lists, matching the declared V5
metadata boundary. It does not reconstruct or authenticate original arrays.

Recursive equality checks dictionary insertion order, list order and exact
Python primitive type. Floats are compared by packed binary64 bits, preserving
negative zero; `False`, integer zero and floating zero cannot compare as the
same type. Tag-like literal values and lists, escaped Unicode, false/null,
different ordered schemas and all original array descriptors are retained.
Repeated invented records must encode to fewer bytes than expanded JSON while
all 100 records and their values roundtrip exactly; no compression-ratio target
is selected from outcomes.

Malformed cases cover envelope shape and exact outer key/format types, malformed
schema tables, duplicate keys/schemas, unused schemas, schema-ID type/range,
wrong arity/tags, bare untagged containers, nonfinite/unsupported primitives,
container/array subclasses and noncontiguous/object/complex arrays. A four-key
outer envelope must refuse before the traversal ledger is constructed.

Exact encoded-byte boundaries and one-below refusals are checked in both codec
directions. Additional controls cover the fixed limits for schema count, keys,
key length, scalar-string length, integer bit length and depth; cyclic encoding
must reach the bounded depth refusal. Private near-limit controls verify node,
expanded-byte and repeated-array-hash accounting without allocating the maximum
data volume. These are engineering guard tests, not security certification.

## Incremental and final evidence accounting

Two same-schema records must charge the first schema once and the second record
only for its node; a third record exceeding the declared total is never appended.
The expanded logical size still counts every repeated key. Fresh export must
include late closure dispositions, added cone fields and every source-core alias
without inheriting obsolete admission-only schemas.

An actual small extraction is instrumented only at V6 `_append` and `_envelope`.
An independent recursive denominator counts every logical node and key visit,
the extra array-to-descriptor visit, and every repeated array's full bytes.
Every append and each of exactly three final passes must consume those exact
counts using one shared work ledger. Final encoded and expanded byte fields
must equal independent actual JSON sizes; fresh export must reproduce them.

An actual one-byte append cap and an actual final cap equal to the observed
incremental admission sum are required whole refusals. The latter cap is derived
by a fixed predeclared rule, not tuned for recovery: the complete small fixture
must have more than 256 bytes of final-tree overhead, all incremental records
must fit, and final export must fail while clearing two cores, one closure and
three qualified proposals. This characterizes representation boundaries only.
Separate injected size/structural drift requires exactly three final passes and
whole refusal on nonconvergence, never an unbounded stabilization loop.

## Fixed geometry and evidence comparison

Every paired case runs the unchanged V5 extractor and the V6 successor on the
**same array objects, position-error stipulations, policies and budgets**.
Comparison removes only these declared representation/resource fields:

- Top level: `schema_version`, and V6-only `metadata_encoding`.
- Statistics: `metadata_bytes`, plus V6-only `metadata_expanded_bytes`,
  `metadata_node_visits`, `metadata_array_bytes_hashed`.

Every other internal field, dictionary/list/tuple order, scalar type/float bit,
array dtype/shape and complete array bytes must be identical. Both internal
arrays must remain immutable. This is stronger than comparing only descriptor
hashes, counts or successful proposals. A separate export comparison requires
decoded V6 metadata to equal the entire V5 metadata interpretation of that V6
payload, without removing any field.

Four required positive cases reproduce the frozen V5 9-by-9 grid, 128 triangles,
alternating local z-noise of magnitude 0.01 and original per-row error bound
`0.05*scale`. The four frame tuples are unchanged:

| Rotation | Translation | Scale |
| --- | --- | ---: |
| Identity | `(0,0,0)` | 1 |
| R | `(3,-4,5)` | 1 |
| R | `(37,-21,9)` | 0.25 |
| transpose(R) | `(-9,5,11)` | 4 |

Here `R=[[1,8,4],[8,1,-4],[-4,4,-7]]/9`, the frozen proper rotation.
All four must retain two full-support CORE proposals and one shared CLOSURE,
source-core aliases `[0,1]`, all 81 raw IDs and 128 triangle IDs, and the frozen
geometry/work counts (including three fits, 243 full-support row tests and
9,112 retained array bytes). Every witness must satisfy the **original** error
bound without added tolerance and contain the independently known normal in its
reported conditional cone. An abstention fails a required positive.

Five additional unchanged small controls preserve: a finite grid hole with raw
row 40 unreferenced; two coincident-coordinate components with distinct raw IDs;
a qualified core whose entire 132-triangle/85-row closure is rejected; rotated
zero-error numerical abstention; and a zero-closure quota that keeps qualified
cores while explicitly omitting their closures. The failed-closure and zero-error
cases do not become positive recovery credit. No area filter, welding, residual
trimming, altered epsilon, native label or expected bend count enters inference.

## Decision

Passing qualifies only this bounded representation and paired small-geometry
contract. It does not establish full-size noisy extraction, connected alignment,
native completeness, physical count/F1, manufacturing conformance or product
promotion. Any full-size V6 capacity successor and connected/physical workflow
requires its own separately reviewed source closure, fixed fixtures, scoring,
budgets and terminal receipt. Frozen V5 failures remain in the evidence ledger.
