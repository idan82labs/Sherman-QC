# Native context coverage V3: pure denominator protocol

Date: 2026-09-09. Status: source implementation, awaiting independent review
and invented-only tests. No check or native job is authorized by this document
alone. Root owns review, test freeze and serial execution.

## Question and boundary

The [retained single-CAD audit](2026-09-09_single_cad_validity_result.md) found
468 missing incident-context witnesses despite the earlier exporter's traversal
completion flag. Retained invented boxes exhibit the same edge-context gap.
The current question is only whether a pure validator correctly accounts for
the required context denominator of a supplied typed occurrence tree.

Implementation: `backend/cad_guided_native_context_coverage_v3.py`.
It imports only standard-library dataclass/immutable-tuple support and `math`; there is no file,
network, application, native-geometry, process or clock access. It does not read
the physical diagnostic, frozen controls, source files or any corpus itself.
All input arrays and native observations are supplied by the caller.

This is not an engine-accuracy gain, a connected native collector, a local
admission policy or a promotion. Frozen V1/R0, the terminal diagnostic, original
inputs and existing engine modules remain unchanged.

## API and input contract

```python
validate_native_context_coverage_v3(
    entities, occurrences, entity_details,
    *, native_valid, source_report_sha256=None,
    limits=NativeContextCoverageLimitsV3(),
)
```

The first three arguments are plain lists using the diagnostic's structural
record layout. They must remain stable during the call. Array ordering is not
identity: entity/detail arrays may be permuted; IDs remain import-local.
Returned rows contain new primitive containers, not input aliases.

Entities contain exactly `native_id`, `type`, `type_index`, `occurrence_count`
and `location_is_identity`. Entity IDs are unique and contiguous `0..n-1`;
per-type indices are also unique/contiguous. The declared occurrence counts must
equal the complete occurrence array. Location flags are booleans, not verified
transform evidence.

Each occurrence is `[native_id, parent_occurrence_index, orientation_enum]`.
There is exactly one root at row zero. Other parents precede the child;
orientation is an integer in `0..3`, never a boolean. The supported immediate
child hierarchy is:

```text
COMPOUND -> any supported native type, including nested COMPOUND
COMPSOLID -> SOLID -> SHELL -> FACE -> WIRE -> EDGE -> VERTEX
```

Nonstandard direct-child arrangements are refused as unsupported, not declared
invalid native geometry. A repeated native identity must retain the same
immediate-child identity multiset, including multiplicity. Child order and
cumulative orientation are intentionally excluded from that signature: reversing
an edge use can reverse its vertex orientations. Repeated seam-edge uses remain
occurrences; only the expected context-pair denominator is deduplicated. A native
identity recursively contained in itself is refused.

Details require `native_id`, `self_status_codes`, `contexts`, `detail_complete`
and `errors`; `subtree_valid` is needed for a clear usable record. Only the
existing bounded scalar diagnostic metadata fields are additionally accepted:
native tolerance/value-validity flag, geometry-type name, degenerated,
same-parameter and same-range flags. Those fields are schema-checked, not used as
geometric qualification. Compounds/compsolids do not have self-detail records in
this contract. Missing other entity details remain explicit missing evidence.

Each context is `[context_native_id, status_codes]`. Only expected context pairs
are admitted. Duplicate exported details/pairs, unknown IDs, malformed records or
conflicting identity metadata invalidate the whole result. Status codes are
unique integers `0..255`; zero alone is clear, every nonzero code is an error,
and a mixed zero/error list is contradictory. Unknown positive codes remain
errors, not a permissive whitelist. An empty status list is missing evidence.

`native_valid` must be a boolean or `None`. It is preserved as a caller-reported
observation, including false on validation failure, never recomputed or promoted.
A true global observation contradicting exported error/subtree-false evidence
causes whole-result refusal; so does self-error with a true own subtree result.
An edge may legitimately have true subtree status and an error in a face context.

`source_report_sha256` is optional lower-case 64-hex text. It is recorded only as
caller-declared provenance: no bytes are read/hashed and no native identity,
authenticity or execution authority is established by the string.

## Required denominator and dispositions

Walk the bounded ancestor chain of every supplied occurrence. Require exactly
these typed relations, deduplicating each `(entity_id, context_id)` pair:

| Entity | Required owning-context type |
| --- | --- |
| VERTEX | EDGE and FACE |
| EDGE | FACE |
| WIRE | FACE |
| SHELL | SOLID |

Do not invent every generic ancestor relation; in particular FACE-to-SHELL,
EDGE-to-SOLID and container contexts are not part of this contract. This matches
the declared analyzer's relevant context traversal, not a universal topology
theorem. The upstream
[OCCT analyzer implementation](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Analyzer.cxx)
is API context, not installed-binary attestation.

Every required self/context row receives `disposition` equal to `CLEAR`, `ERROR`
or `MISSING`. A known nonzero status is retained as ERROR even if other details
are incomplete. A zero status is CLEAR only for a usable detail record; absent
or empty evidence is MISSING, not a native error. Missing data never becomes
NoError. False/incomplete detail flags cannot be overridden by their favorable
status prefix.

Output is a plain dict with:

- `status`: `COVERAGE_COMPLETE`, `COVERAGE_INCOMPLETE` or `UNAVAILABLE`;
- `coverage_complete`, `reasons`, caller provenance and `native_valid`;
- `self_dispositions`: sorted rows with native ID/type, disposition, copied
  status codes and optional subtree observation;
- `context_dispositions`: sorted expected-pair rows with `native_id`,
  `context_id`, typed `relation`, disposition and copied status codes;
- `counts`: entities, occurrences, required self records, observed details,
  expected/observed context pairs, CLEAR/ERROR/MISSING self/context counts,
  incomplete details, false subtrees, status values and ancestor work;
- always-false local/global/production admission flags, null observed bend count
  and no feature claims.

Complete coverage can contain negative statuses and does not imply native
validity. Even an empty context denominator cannot grant admission. Malformed or
over-budget input returns `UNAVAILABLE`, clears both row arrays and sets counts
to `None`, preserving no successful evaluated prefix. Missing evidence instead
remains explicit in the complete denominator with `COVERAGE_INCOMPLETE`.

## Fixed work/memory scope

`NativeContextCoverageLimitsV3` is frozen. Each field may only be lowered.
The hard maxima are independent immutable literal values, not the public default
instance: bypassing a dataclass freeze cannot raise them. Each call snapshots the
caller's limit values, validates that snapshot, and uses only the private
immutable integer snapshot thereafter. Later mutation of the caller's instance
cannot change the admitted call's budgets. This is not protection against
arbitrary Python module/code mutation.

| Field | Maximum/default |
| --- | ---: |
| `max_entities` | 2048 |
| `max_occurrences` | 4096 |
| `max_depth` | 32 |
| `max_details` | 1024 |
| `max_expected_context_pairs` | 4096 |
| `max_observed_context_pairs` | 4096 |
| `max_statuses_per_list` | 32 |
| `max_total_status_values` | 32768 |
| `max_ancestor_steps` | 131072 |

Per-type entity caps additionally remain32 compounds,16 compsolids,8 solids,
64 shells,256 faces,512 wires,1024 edges and1024 vertices. Depth counts edges from
the root, whose depth is zero. Every ancestor step consumes work, including
irrelevant type relations; only expected-pair count deduplicates. Exported
duplicate pairs are rejected, not used to evade the observed-record cap.
Bounded lists are snapshotted and output size is bounded by these counts. This
is not a maximum-cap performance qualification or an OS resource sandbox.

## Prospective independent invented checks

Root/Astra authors the independent tests after reading the complete source.
Use explicit isolated imports without application startup or native libraries.
Freeze the exact suite/source closure before one serial60 s wall /45 s CPU /
1 GiB RSS /1 MiB output run. This is not permission to run now.

Predeclared strata:

1. A symbolic two-face/shared-edge occurrence tree with independently enumerated
   expected pairs. Preserve multiplicity in occurrences, but count its shared
   edge/vertex-to-face pairs once each. All-clear coverage with unknown or false
   native validity must leave every admission flag false.
2. Remove the second incident-face context, remove vertex-to-face contexts,
   remove an entity detail, empty a status list, and mark a detail incomplete.
   Assert the exact missing-pair/self denominator and no admission. A nonzero
   context on one incident face must remain an error even when the edge's own
   self/subtree statuses are clear.
3. Reverse entity/detail/context ordering, reorder occurrence branches while
   remapping parent indices, and reverse cumulative edge/vertex orientations.
   Compare sorted semantic results; never alter native IDs to create equality.
   Include repeated seam-edge occurrences and inconsistent repeated child
   identity/multiplicity counterexamples.
4. Reject duplicate IDs/type indices/details/exported pairs/status codes, ID
   gaps, wrong occurrence counts, multiple roots, forward/cyclic parent indices,
   recursive native identities, invalid type transitions, invalid orientation,
   unknown/extraneous context pairs, mixed NoError/error codes, wrong primitive
   types including bool-as-int, contradictory native observations and malformed
   provenance/metadata. Put faults late and require wholly cleared evaluated
   output, not an earlier success prefix.
5. For each of the nine budget fields, cover an independently counted exact
   boundary and a one-over or lowered-cap refusal. Refuse raised/noninteger
   limits. Include global status/context budgets distributed across several
   details, not only one oversized row. Bypass the public default dataclass's
   freeze with `object.__setattr__` and require refusal against the independent
   literal caps; restore the fixture afterward. Mutate a caller-owned limit
   object after snapshot validation and require the admitted budgets to remain
   unchanged. Keep fixtures bounded and invented.
6. Mutate every supplied container after return; results must not change.
   Verify module import and every public path have no file/native/process access,
   and that positive/negative/missing/malformed outputs retain the fixed
   no-admission/unknown-observed-count boundary.

Passing these tests would establish bookkeeping semantics only. Any test failure
is retained; source repair requires another exact review before execution.

## Future collector interface, not implemented here

A successor collector can supply these same arrays plus its own byte-bound
provenance. For every required incidence it should compare iterator enumeration
with keyed cached `IsStatusOnShape(owner)` and `StatusOnShape(owner)` reads,
recording missing cache/getter failure separately rather than calling `InContext`
or healing to generate a favorable entry. It must retain source/import identity,
orientation, self statuses and all incident contexts before proposing local use.
The [OCCT result interface](https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/V7_9_3/src/BRepCheck/BRepCheck_Result.hxx)
exposes those distinct cached queries.

Native collection needs its own complete source/runtime review, constructed
matrix and bounded gate. Local feature admission additionally needs finite trims,
tangency, orientation/material-side justification and all consumed dependencies;
this pure validator supplies none of those. No physical rerun, keyed native call,
source-file reader or production activation is included in this step.
