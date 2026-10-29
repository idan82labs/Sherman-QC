# V6 lossless noisy-mesh metadata representation — prospective protocol

Status: **SOURCE DRAFT, NOT EXECUTED.** This is a separately versioned engineering
successor, not a retry of the terminal V5 capacity case. Root owns independent
source/test authorship review, exact source binding, execution gates and later
integration. No native or physical input is accessed by this source task.

## Retained negative and narrow hypothesis

Root reported the terminal full-size V5 attempt
`noisy_mesh_capacity_v5-ksogonyt`, output SHA-256
`957cfa8b576e8d54a05f5566d6c768fda6996e5c08f06ddecd998e56c896f5a0`:
22,452 cores had been retained and 13,847 qualified before the 67,108,864-byte
metadata admission cap was reached; no closure evaluation began. Parent wall
was 6.583641 seconds, worker CPU 6.545417 seconds, sampled process-group peak
979,861,504 bytes and worker lifetime peak 982,007,808 bytes, with no reported
outer budget stop or source drift. The outcome is a
capacity failure, not a geometric positive. Its source, fixture, protocol and
receipt remain unchanged; this document relies on the released aggregate and
does not reopen the artifacts or underlying physical data.

The new hypothesis is only that repeated dictionary field names can be represented
once in a shared schema table, retaining every field and value while consuming
fewer encoded metadata bytes. **The numerical 64 MiB cap is unchanged, but the
encoded representation being charged is changed. This is not unchanged V5
expanded-JSON admission.** No rejection, candidate, uncertainty term, raw index,
float precision or late-added closure field is dropped to meet the cap.

There is no claim that the successor must finish the full-size case, produce ten
required face representatives, qualify a closure, improve registration or advance
physical count/F1. Geometry and other independent work caps may still refuse.

## Source and API

The new source is `backend/cad_guided_noisy_mesh_planes_v6.py`. It imports the
unchanged frozen `backend/cad_guided_noisy_mesh_planes_v5.py`, SHA-256
`9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75`.
It reuses V5 configuration validation, owned input snapshots, original-raw-edge
adjacency, BFS partitions, support extraction, TLS/witness/cone mathematics,
closure scheduling, immutable lineage storage, digests and whole-result failure
handling. Only the main orchestration is copied where necessary to substitute
metadata admission/finalization calls. V5 itself is not edited or monkeypatched.

The separate public extractor is:

`extract_noisy_scan_plane_proposals_v6(points, triangle_raw_rows, *,
position_error_bounds_mm, policy=NoisyMeshPlanePolicyV5(),
budgets=NoisyMeshPlaneBudgetsV5())`.

The policy/budget classes are explicitly the existing V5 types, validated and
snapshotted by the existing V5 configuration helper. All their defaults and
maximum values stay unchanged. The internal output retains the ordinary core,
closure, analysis and proposal dictionaries and immutable original-row arrays.
It adds a V6 schema version, a metadata-encoding description and separate
metadata-work/expanded-size counters. Existing numerical geometry fields are
not reinterpreted as calibrated, physical or production-approved evidence.

`encode_metadata_v6(value, *, max_bytes=67108864)` and its exact export seam
`_metadata(value, *, max_bytes=67108864)` return a JSON-compatible envelope.
`decode_metadata_v6(envelope, *, max_bytes=67108864)` reconstructs the V5 JSON
metadata tree or raises on malformed/unsupported/over-budget input. A caller may
only reduce `max_bytes`; booleans and values outside `[0,67108864]` are invalid.
These functions perform no file I/O or native calls.

## Exact represented object and grammar

The logical object is precisely the tree returned by V5 `_metadata`: dictionaries
with ordered string keys, lists, null, booleans, integers, finite floats and
strings. Original tuples become lists as they already do in V5 metadata. An
original NumPy array becomes its V5 `{shape, dtype, sha256}` descriptor. The
decoder does **not** reconstruct or authenticate original array contents. Array
values remain available separately in the immutable internal extractor payload.

An envelope has exactly `format`, `schemas` and `root`. Its format is
`noisy_mesh_metadata_schema_v6`. `schemas` is a list of ordered key lists. Logical
nodes have the following unambiguous grammar:

- Dictionary: `["D", schema_id, [encoded value, ...]]` in schema key order.
- List: `["L", [encoded value, ...]]` in original item order.
- Primitive: the original JSON primitive, without a wrapper or rounding.

The two node tags cannot collide with an ordinary logical list because every
logical list is tagged. Dictionary key order, record order, literal strings,
false, null, integer/float distinction, float values and array descriptor hashes
are preserved. Negative floating zero is not rounded to integer zero. Arrays
and tuples have only the intentionally narrower V5 metadata interpretation.

Schemas are interned by exact ordered key tuple, not by an unordered key set.
The decoder requires valid integer (not boolean) schema IDs, exact value arity,
unique keys, unique schema definitions, no unused schemas, valid tags and finite
supported primitives. Dictionary-like and array subclasses are not a plugin
dispatch mechanism. Supported metadata arrays are C-contiguous numeric/bool
NumPy arrays; the actual V5 immutable lineage arrays meet this restriction.

## Prospective bounds and exact size accounting

The following codec-only hard limits are additional conservative engineering
guards, not increased V5 geometry budgets:

| Quantity | Fixed limit |
| --- | ---: |
| Compact encoded envelope / admission metadata | 67,108,864 bytes |
| Compact expanded V5 JSON metadata equivalent | 268,435,456 bytes |
| Unique schemas per table | 256 |
| Keys per dictionary/schema | 128 |
| Characters per key | 128 |
| Characters per scalar string | 65,536 |
| Magnitude bits per integer scalar | 4,096 |
| Logical recursion depth | 64 |
| Cumulative node/key traversal visits | 64,000,000 |
| Cumulative array bytes hashed for descriptors | 4,294,967,296 bytes |

Compact JSON uses `ensure_ascii=True`, `allow_nan=False` and separators `,`/`:`.
Encoded size includes every tag, value, separator, the complete schema table and
the outer envelope. Size counting streams encoder fragments; it does not build
a complete serialized string. Strings and integers are bounded before scalar
serialization; the integer guard prevents unbounded decimal conversion work on
a single malformed diagnostic value. Expanded size is computed recursively from exact JSON scalar/key byte
lengths plus container punctuation, without constructing the expanded JSON
string. It charges each repeated logical key at every dictionary occurrence,
even though encoded keys are interned. It includes all eventual late mutations.

The expanded-size limit bounds the logical diagnostic representation separately
from its compressed bytes. Decoder traversal has its own 64-million-visit
invocation limit and the same encoded/expanded/depth/schema/string restrictions.
These are bounded Python engineering operations, not an OS memory certificate;
external resource guards remain necessary for any execution.

## Incremental admission versus deterministic final export

One schema interner and one work ledger are shared by **all incremental append
calls in an extraction**. Before a record is appended, its tagged node bytes and
every newly introduced schema's bytes are charged. Already introduced schemas
are not recharged per record. The logical expanded admission tree is also
counted across records. Any overflow aborts the whole extraction and invalidates
all cores, closures, partitions and proposals via V5 failure handling. A prefix
of admitted evidence cannot be published as a completed result.

Incremental charges are an admission accounting sum, not a substitute for the
final serialized envelope. Core closure dispositions, `source_core_ids`, cone
diagnostics, completion flags and other post-admission fields still require the
final whole-tree check. Newly introduced schemas from pre-mutation record forms
must not leave hidden state in reproducible exported metadata.

Therefore final/export encoding builds a **fresh deterministic table** from the
final tree. Each final pass charges the same extraction work ledger used during
admission. Independent `_metadata(result)` later starts its own bounded
invocation and reproduces the final encoded envelope and byte count without
needing any hidden incremental interner history.

Exactly three final encoding passes are scheduled. The first measures the now
structurally stable tree's traversal/hash work. It predicts total work after all
three passes and publishes that fixed total before the remaining passes. Size
counters are updated from the prior pass. After the third pass the implementation
requires both encoded/expanded size fields to equal the actual final sizes and
the predicted work fields to equal actual cumulative traversal/hash work. Any
nonconvergence or unexpected structural change whole-refuses; there is no
unbounded fixed-point loop, omitted pass charge or counter update after accepting
a stale encoded envelope. Exceptions retain attempted metadata work counters.

## Independent checks required before any capacity successor

Root will independently author and audit focused source/pure controls before
execution. The intended first gate is at most 60 seconds wall, 45 seconds worker
CPU, 1 GiB sampled process-group RSS and 1 MiB output, with no physical/native
inputs and no capacity run enabled. Required coverage is:

- Exact roundtrip to V5 metadata over nested scalar/list/dict/array-descriptor
  trees, including false/null, strings equal to tags, key order, signed zero,
  integer versus float values, and varying schemas.
- One shared incremental schema table and pre-append charging; independent exact
  compact JSON lengths for both encoded and expanded trees; measurable reduction
  on repeated invented analysis records without any dropped/rounded value.
- All hard boundary/refusal classes, malformed tags/IDs/arity/duplicates,
  nonfinite values, cyclic/deep trees and whole-clearing early/late overflow.
- Exact small-geometry equivalence to V5 for partitions, every original support,
  analysis values, proposal IDs and schedule/closure outcomes; only explicitly
  versioned metadata/accounting fields may differ.
- Late closure-field/alias mutations survive encode/decode, independent final
  byte reproduction is exact, and all three final passes share work accounting.

A full-size V6 capacity case requires its own independently reviewed and frozen
successor gate. The original V5 fixture/noise/truth/oracle may be paired as
consumed invented development evidence, but no V6 capacity test or run is
authorized by this document alone. The later external envelope stays
120 seconds wall, 90 seconds worker CPU, 4 GiB sampled process-group RSS and
1 MiB output, one heavy job serially. No automatic retry, truncation, cap increase,
source drift or post-outcome oracle weakening is permitted.

All original V5 results remain as recorded. Default product routing, physical
input custody, uncertainty, unknown observed counts and production gates are
unchanged. Source existence, codec roundtrip success or reduced metadata bytes
alone are not evidence of a complete capacity or connected-engine result.
