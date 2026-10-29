# Cached native-context collection V3: shape-only protocol

Date: 2026-09-09. Status: source implementation awaiting independent source
review and tests. This protocol does not authorize execution or physical access.
Root owns exact closure review, frozen controls and serial bounded execution.

## Question and non-goals

Can the installed binding expose the cached status for every occurrence-derived
required native context, and do keyed reads agree with iterator reads in both
orders? The retained V2 diagnostic remains terminal and unchanged. Its reported
traversal completion did not establish the incident-context denominator.

The pure V3 denominator helper passed its independently authored 173-test gate
(`native_context_coverage_v3-0ly2yyyq`, as reported by root). That qualifies its
supplied-array bookkeeping only. It does not validate this new collector, native
binding behavior, CAD geometry or local feature admission.

This successor has no original-CAD reader, bytes/path entrypoint, command-line
interface, scan input, inference, healing, native cache population or production
activation. A caller supplies a constructed native shape under its own audited
custody. Coordinates and units are not established by this shape-only API.

## Exact API and source closure

```python
collect_cached_native_contexts_v3(
    shape, *, allow_ocp=False, read_order="KEYED_THEN_ITERATOR"
)
```

Only `KEYED_THEN_ITERATOR` and `ITERATOR_THEN_KEYED` are supported. Invalid opt-in
or order refuses before dependency loading or native calls. Imports of the new
script are inert standard-library imports. Native imports are lazy.
After topology validation, capture the exact native enum type from the same
binding's `BRepCheck_NoError`. Only that concrete type may reach legacy enum
conversion. Coercible scalars, another enum and conversion failures whole-refuse;
a native list iterator that throws remains an explicit incomplete read. Inherited
property schema/budget exceptions also whole-refuse, unlike ordinary getter errors.

The collector loads and executes only the verified bytes of these two named
inert source dependencies, at most 64 KiB each, without importing the application
or backend package entrypoint:

| Source | Required SHA-256 |
| --- | --- |
| `scripts/v2_native_validity_diagnostic.py` | `e2767721c5b6c1bdf02b77cb42d4c6c2cbc68402b6e63407afecfb92d8cfca91` |
| `backend/cad_guided_native_context_coverage_v3.py` | `e9a366cf36106495dfd4d8563f639af8ba185f9113fea0ac40ad030d3c0b4eeb` |

It reuses only `_topology`, `_owner_solid`, `_properties`, `_status_codes` and
their supporting definitions from the diagnostic. It does not call its physical
reader, public physical entrypoint or `_inspect_imported_shape`. No frozen file
is edited. These source digests are not runtime/binary or native-shape attestation.
The controlled dependency loader temporarily registers each module for standard
dataclass decoration and restores the prior registration after loading.

## Sequence and cached-read semantics

1. Export the bounded occurrence tree and native identity map with `_topology`.
2. Run the pure validator on that tree with no details and unknown native
   validity. Refuse malformed/over-budget topology and an oversized self-detail
   denominator before creating any analyzer. Use its complete sorted expected
   context pairs; do not derive expected coverage from observed cache entries.
3. Resolve the single solid owner, then construct exactly one
   `BRepCheck_Analyzer(solid, True, False, True)`. Record its global validity and
   common self/subtree/property observations. No second analyzer is used to
   obtain a favorable result.
4. Read both channels in the requested order. For every expected pair, record a
   membership query outcome. A present entry permits keyed `StatusOnShape(owner)`;
   an absent entry never calls the getter. Independently enumerate the iterator
   channel and its zero-argument `StatusOnShape()` values.
5. Validate both channels separately against the same denominator and shared
   self observations. Compare their pair sets and sorted status-code sets.

The upstream [OCCT result interface](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Result.hxx)
distinguishes cached membership/keyed access from `InContext`. Its
[iterator implementation](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Result.cxx)
skips the self key. This establishes the intended API semantics, not an
attestation of the installed binding. The collector never calls `InContext`,
`Minimum`, `Blind`, status setters, shape repair, meshing or cache-filling helpers.

Absent cache and empty status lists are missing evidence, not native errors.
A thrown membership/getter/iterator read is incomplete collection. Native
nonzero status codes remain native ERROR evidence under the pure validator;
they are not manufactured-part defect claims. Global native false is preserved.

## Output and failure contract

The returned plain dict has schema `NATIVE_CONTEXT_COLLECTION_V3` and:

- `entities`, `occurrences`, `owner_path`: one topology representation;
- `self_details`: common self/property/subtree rows stored once, without contexts;
- `keyed_observations` and `iterator_observations`: separate lists of
  `[native_id, context_id, status_codes]`, retaining each channel's values;
- `cache_queries`: every expected pair as
  `[native_id, context_id, membership_outcome, getter_outcome]`;
- membership outcomes `PRESENT`, `ABSENT`, `ERROR` or `RESULT_UNAVAILABLE`;
  getter outcomes `READ`, `EMPTY`, `ERROR` or `NOT_CALLED`;
- `channel_details`: per-channel `[native_id, complete, error_or_null]` rows;
- `coverage`: separate compact keyed/iterator validator summaries, including
  status, reasons, coverage-complete flag, counts and false admission flags;
- `comparison`: matching count, keyed-only, iterator-only, differing-status,
  missing-both and unexpected-iterator pairs;
- native observation/options/count, source bindings, bounded enum names/counters
  and always-false local/global/production admission flags. Observed bend count
  remains null; feature and physical-generalization claims remain false.

The collector temporarily assembles complete validator-compatible detail rows
for each channel but does not copy the full validator output rows into the final
report. No channel is selected as authoritative. Channel disagreement remains
visible even if one channel has complete coverage.
The inherited `native_tolerance_mm` field name is explicitly qualified: its
values are in unverified native units for this shape-only API. It does not
establish millimetres, scanner calibration or physical measurement uncertainty.

`COLLECTED` means the native read sequence completed, not that coverage is
complete or the channels agree. `PARTIAL_COLLECTION` retains bounded observed
records with incomplete per-entity flags after native getter/traversal failures.
Coverage, read completion, topology completion and channel agreement are separate
fields. A completed absent-cache query does not become a completed coverage claim.

Malformed topology/status records, duplicate or outside-topology iterator pairs,
native flag contradictions rejected by the pure validator, source mismatch and
budget exhaustion return `UNAVAILABLE` with all evaluated arrays/counts cleared.
An unexpected but known iterator owner also makes that channel fail validation
and clears the entire evaluated report. No successful prefix remains. A recorded
global native observation and attempted analyzer count may survive refusal.

## Prospective fixed budgets

Topology inherits 4,096 occurrence rows, depth 32, 2,048 unique shapes and the
frozen per-type caps. Self details are capped at 1,024 before the analyzer starts.
The pure validator additionally bounds ancestor work at 131,072 steps.

Each channel has at most 4,096 context records; iterator contexts are additionally
capped at 64 per entity. Expected pairs deduplicate repeated incidence; observed
duplicates refuse. Keyed membership and getter calls each cannot exceed 4,096.
Iterator status-getter attempts also cannot exceed 4,096, including failures.
Each status list has at most 32 codes, and the total status-value cap is 32,768
across self plus both channels, including repeated values across channels.
Every visited status value consumes this budget even if a later list read fails.
All limits are fixed, with no caller override or retry-driven widening.

Serialized output is capped at 768 KiB; overflow whole-refuses, rather than
truncating evidence. Bounded topology/context arrays can require more temporary
memory than the final serialized cap. These are application bounds, not an OS
sandbox or maximum-cap performance qualification.

Root's prospective pure/mock gate is 60 s wall / 45 s CPU / 1 GiB process-tree
RSS / 1 MiB output. A separately frozen native gate is 120 s total wall / 90 s
CPU / 1 GiB process-tree RSS / 1 MiB output, with work and cleanup time separated
by a five-second cleanup reserve within the 120-second wall limit. Native work
uses the single pytest worker; neither collector nor fixture may spawn children.
One heavy native worker only. No execution is
authorized until its independent exact-source/runtime gate is complete.

The additive `scripts/v3_native_context_test_supervision.py` owns only this native
test route. It stops work at 115 seconds, gives process-table queries and child
reaping bounded timeouts within an absolute 120-second deadline, confirms the
owned group empty on both normal and killed exit, and classifies late completion
or unknown custody as failure. It retains the private cache instead of invoking
unbounded directory cleanup. Native worker stdout is retained under its byte cap,
not forwarded during monitoring. Timings and custody errors remain in the receipt.
These are sampled deadline enforcement and post-run failure checks, not a hard
real-time guarantee against stalled OS/filesystem operations. Source hashing and
receipt publication are bounded-size I/O with no real-time guarantee. A durable
STARTED remains if later publication fails. No historical route is changed.

The native test writes all eight reports to a fresh parent-owned artifact
directory, passed only through `SHERMAN_NATIVE_CONTEXT_COLLECTION_V3_ARTIFACT_DIRECTORY`.
Each exclusive-create JSON is capped at 768 KiB; the aggregate cap is 6 MiB.
The test prints only a compact manifest (at most 16 KiB) with all case dispositions,
paths, byte counts and SHA-256 values, inside the supervisor's 1 MiB total output
cap. Encode complete reports before writing, fsync and verify retained bytes.
No caller-supplied artifact path is inherited. Failed cases remain selected and
retained; neither serialization failure nor a skipped case is a passing outcome.
The CPU limits are worker-process limits enforced by the isolated harness; parent
monitoring CPU is not included in that counter. Process-group RSS is sampled.

Root's isolated commands after independent source/budget gates are:

```sh
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite native_context_collection_v3_source
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite native_context_collection_supervision_v3
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite native_context_collection_v3
```

Both pure commands must pass before the separately audited native gate. Native
opt-in is set by the exact named worker route only; the pure route cannot inherit
that opt-in. Both routes retain before/after source hashes and unique receipts.

## Fixed independent checks, not yet executed

Pure/mock controls must independently enumerate a shared-edge and seam-edge
tree; verify denominator-before-analyzer ordering, exactly one analyzer and both
read orders; and test absence, empty lists, read exceptions, disagreement,
duplicates/extraneous IDs, whole-refusal late caps, source binding and output
overflow. Assert no forbidden cache-mutating calls. Verify both coverage results
and every expected membership/getter outcome, not only aggregate test success.

The proposed native matrix is four fresh invented shape types in each read
order: a box, the existing five-face open-shell fixture, a capped cylinder with
seam, and a reversed box. Eight cases total, one analyzer per fresh construction.
Independently enumerate expected incidences and preserve seam multiplicity;
compare sorted semantics across orders without treating import-local IDs as
persistent between separately constructed shapes. No fixture or read-order
change may erase a failing first receipt. This matrix is not a physical rerun.

Independent fixed denominators are: box and reversed box 34 unique shapes /
86 occurrence rows / 79 expected typed context pairs; five-face open shell
32 / 72 / 70; capped cylinder with seam 13 / 26 / 17. Geometry-based native
identity classification must map these predeclared incidences without deriving
the expected pair set from the collector's tree or observed cache. Box/cylinder
global native validity is expected true and open shell false. Reversed-box
validity is retained and compared across orders without presupposing its sign.

Observation-contract success and closing the context-evidence gap are separate
outcomes. A complete native read with correctly retained missing cache or channel
disagreement may pass the collector contract; it cannot count as complete context
evidence. The manifest must expose each case's two coverage-complete flags and
agreement, plus `context_gap_closed_on_constructs`, true only when all eight pass
the contract AND have complete agreeing channels AND all four fresh read-order
pairs agree after geometry-based semantic normalization. Read-order disagreement
remains a retained diagnostic outcome, not a context-gap closure. No native partial/unavailable
collection is a passing expected-call contract outcome.

Passing would establish bounded cached-context observation on these constructs,
not complete physical-native validity causes, local geometry admission or engine
accuracy. Any later connected collector or physical use needs its own audited
source/interface and scope; no native-invalid bypass or healing is implied.
