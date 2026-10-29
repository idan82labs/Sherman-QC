# Context V3 engine integration — new source boundary

Date: 2026-09-12. Status: NEW / SOURCE-ONLY / UNQUALIFIED INTEGRATION.

Authorized new implementation files are
`backend/cad_guided_plane_initialization_v13.py` and
`backend/cad_guided_mesh_offline_v14.py`. This document is the third authorized
edit. All frozen sources, historical attempts, scorer rules and scientific
inputs remain unchanged. No target import, syntax execution, test collection,
geometry execution, native operation or physical input access was performed by
this author. Root separately reports the context-helper76 gate and its accepted
POST; that is not qualification of this new engine connection.

## New comparator, not a historical control

The declared successor comparison is
`FRESH_V14_CONTEXT_V3_V7_SCHEDULE_V1_VS_COMPOSITE_SCHEDULE_V2_NATIVE_V1`.

| Successor arm | Scan proposals | Scheduler | Context proposer | Native adapter |
| --- | --- | --- | --- | --- |
| control | v7 | v1 | v3 | v1 |
| challenger | composite_v1 | v2 | v3 | v1 |

Both arms use contextV3 under an explicit new comparison. This is not the V5
paired control, a continuation of V5, or a replay of a terminal attempt. V14
declares the comparator and immutable mechanism tuples; the future V6 runner
must independently freeze complete effective budgets, policies and source bytes.
The adapter is not an authority to launch either arm or to mutate a frozen run.

The paired mechanism difference remains scan proposal plus schedule choice.
Native adapter, finite-distance support/admission, connected-context measurement,
free-circle fitting and scoring rules are not changed here. ContextV3's staged
scan arithmetic is explicitly different from legacy NumPy-matmul projection;
the change is not claimed bitwise equivalent or a measured end-to-end speedup.
No support-cover union planner is integrated by this task.

## Smallest source-only integration seam

Frozen initializerV12 binds its context implementation to schedule mode. Frozen
workflowV13 locally imports that initializer and aliases a V12 array binder that
checks the same coupling. Merely aliasing those functions into a new module would
not change their global lookups; swapping a global or forging a V2 helper report
would hide the actual implementation. Neither technique is used.

InitializerV13 instead owns only the new context configuration/validation and
the narrow orchestration requiring new plumbing: `_two_plane_records`, `_prepare`,
`_initialize`, public prefilter and registration wrappers. It directly reuses
frozen triplet preparation, `_prefilter`, assigned-face query machinery, numerical
refinement, context support admission, context publication, schedule validation,
input row selection and bounded report envelope. The exact
`PlaneInitializationBudgetsV12` class is re-exported as that same type; it is not
silently subclassed or given larger limits.

WorkflowV14 owns V13's byte orchestration with the explicit new initializer,
plus a narrowly versioned array binder. It reuses scan/CAD preparation, native
features, CAD sampling, composite support preparation, selected registration
row validation and support-cover measurement. No dynamic function cloning,
`sys.modules` rewriting, global monkeypatching, source execution wrapper or
caller-provided provider callable participates in this integration.

## Closed public context configuration

The new public prefilter, registrar and byte workflow accept:

- `two_plane_context_mode="v3"`;
- `two_plane_context_budgets=None`, meaning a newly constructed default
  `TwoPlaneContextBudgetsV3`, or an exact instance of that type.

Only the exact string `v3` is accepted, and the pair lane must be explicitly
enabled. Neither schedule version chooses a context implementation now. The
validated budget is copied into a new frozen dataclass before work; the original
caller object is not the execution configuration. Both scheduling modes retain
their old configuration rules, including scheduleV1 with budgetsNone.

The exact context configuration object has twelve keys:

1. `mode`: `v3`.
2. `implementation`: `RAW_EXTREMA_CONTEXT_V3_STAGED_SCAN_PROJECTION`.
3. `schema_version`: `two_plane_context_v3`.
4. `budgets`: the complete23-field integer budget snapshot.
5. `scan_projection_arithmetic`:
   `STAGED_BINARY64_SUB_MUL_ADD01_ADD2_CONSTANT_POSITIVE_ZERO_V1`.
6. `scan_projection_bound_scope`:
   `ENCLOSURE_OF_STAGED_ROUNDED_PROGRAM_NOT_EXACT_REAL_INTERVAL`.
7. `cad_projection_arithmetic`: `UNCHANGED_V1_NUMPY_MATMUL_PROJECTION`.
8. `scan_projection_chunk_scope`: `FIXED_EXTREMA_LEAF256_NOT_CHUNK_ROWS`.
9. `chunk_rows_scope`: `INPUT_VALIDATION_AND_CAD_PROJECTION_ONLY`.
10. `extrema_guard_status`:
    `NEW_PROSPECTIVE_ENGINEERING_LIMITS_NOT_V1_V2_INVARIANTS`.
11. `extrema_membership_scope`:
    `SUM_OF_DISTINCT_INDEX_SUPPORT_SIZES_INCLUDING_WHOLE_SCAN`.
12. `extrema_memory_scope`:
    `SHARED_RAW_OWNER_PLUS_ALL_LIVE_INDEX_PAYLOAD_UNDER_PARENT_CAP_NOT_RSS`.

Budget keys/maxima are the eighteen V1 keys with
`max_ordered_cad_matches_per_scan_pair=256`, plus the five V3 guards:
`max_extrema_indexes=129`, `max_extrema_indexed_rows=4000000`,
`max_extrema_node_visits=16000000`, `max_extrema_coordinate_visits=256000000`,
`max_extrema_queries=3072`. Exact integer values must be nonnegative and no
greater than the corresponding maximum, `chunk_rows>=3`, and
`max_extrema_queries<=3*max_signed_families`. Booleans/floats, missing/extra keys,
wrong implementation or altered arithmetic/scope strings are rejected.

The canonical configuration digest is SHA256 of
`json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()`.
Python's default `ensure_ascii=True` applies; declaration strings are ASCII.
Only the bounded validated configuration is encoded, never the whole workflow.
Snapshots are reconstructed independently so caller/returned dictionaries do
not supply the expected configuration by alias.

## Honest transport schemas and mandatory implementation binding

The unchanged measurement consumer requires the V12 registration transport
schema, and the unchanged contact projector requires the V13 workflow transport
schema. Those wire schemas remain exactly as named. They are not rewritten after
the fact and are not being used as claims of historical implementation identity.
No schema, original result byte stream, digest or source identity is disguised.

The new byte envelope separately publishes
`implementation="CAD_GUIDED_MESH_OFFLINE_V14_CONTEXT_V3"`, including its bounded
workflow-refusal envelope. Its normal envelope explicitly states
`wire_schema_scope="V13_TRANSPORT_WITH_EXPLICIT_V14_IMPLEMENTATION_NOT_HISTORICAL_EXECUTION"`
and the both-arms context request scope. A future new projection wrapper must
require this implementation identity; old V13 wire acceptance alone cannot
qualify a V14 result.

Every usable publication carries:

- `initialization_implementation` equal to
  `CAD_GUIDED_PLANE_INITIALIZATION_V13_CONTEXT_V3`;
- the exact `two_plane_context_configuration`;
- `two_plane_context_configuration_sha256` recomputed from that configuration.

These fields are published on workflow top level, workflow `policies`,
registration `evidence`, and the nested `initialization`. Each selected
`registration_array_binding` also carries initialization implementation,
`two_plane_context_mode="v3"`,
`two_plane_context_implementation="RAW_EXTREMA_CONTEXT_V3_STAGED_SCAN_PROJECTION"`
and the same configuration SHA. The byte adapter binds every parent before
measurement, then every selected context's array/pose/support lineage.

These checks are outside the nonempty pair-context branch: a context whose
selected origin is a triplet still must bind the V3 request, as must a run whose
pair schedule is empty. For a nonempty schedule, the helper's actual V3 schema,
complete budget map, capacity and arithmetic/scope declarations are also checked.
The original schedule, family, sign, slot, candidate and raw-buffer witness
checks remain in force before any proposed pose is consumed.

Frozen V12's `_has_two_plane_schedule_v2_publication` treats any context capacity
record as a V2-scheduler marker. That implication is no longer valid when both
schedulers use V3. The new workflow's scheduleV1 binder therefore checks actual
schedule/configuration markers directly. It does not erase context metadata or
pretend V3 is V1. ScheduleV2's full equation/configuration binder is reused
unchanged. Context validation remains independent and mandatory.

## Failure, budget and custody semantics

The V3 proposer receives the entire existing declared schedule once. There is
no batching, prefix acceptance, omitted pair/sign/slot, resource refill, fallback
to V1/V2, retry with smaller data, or changed finite-distance criterion. Every
nonexcluded original support row still reaches the reused finite prefilter.
The registry's shared raw owner and cumulative extrema work/memory accounting
remain the helper's responsibility; the connection does not raise those limits.

A context-helper failure propagates through whole initialization refusal. The
triplet ledger may remain as attempted diagnostic evidence, never a successful
fallback. A registration failure clears usable context indices in both scheduler
modes and releases the owned support transport on a refused result. The workflow
builds the selected binding list atomically before measurement; any context
binding failure clears the list and all usable context indices, nulls the owned
support transport and reaches the existing zero-prediction refusal path. There
is no first/middle/last-context success prefix.

Configuration fields appear only after the corresponding configuration stage
has succeeded. Admission/input failures before that stage may lack them. The
unchanged canonical initialization detailed-report-refusal envelope deliberately
contains no added fields: no out-of-budget metadata is appended to it. It cannot
become an accepted context, because the required implementation/configuration
binding and complete initialization checks fail. The byte top-level V14 identity
still identifies the adapter that returned the refusal. A future projection
must validate any present metadata and must require all publications for a
completed result; it must not fabricate not-reached configuration evidence.

Metadata strings and hashes are consistency evidence, not signatures, raw-data
authentication or proof that named source bytes ran. The future source freeze,
worker authorization and exact-result reconstruction provide the separate
execution/byte custody boundary. This adapter changes none of those controls.

## Independent verification and remaining work

Independent new tests should use invented tiny meshes for the actual public
initializer-to-proposer-to-finite-prefilter seam in both schedule modes. Suggested
50-row two-sheet fixture expectations are a prospective denominator, not a test
result: two ordered matches, eight signed families,64 slot records and all25
original rows per proposed support. No native decoding, refinement or measured
bend claim is needed for this first connection check.

Mocked byte/native boundaries should still exercise real context configuration,
schedule and selected array binders. Cover complete and empty pair schedules,
triplet-origin contexts, caller mutation, every configuration/arithmetic/budget
field, each publication location, every selected binding, input digest/pose
tampering, exact helper refusal, missing/extra fields and whole-list clearing.
Freeze-preservation and lack of dynamic injection require source checks too.

The integration author did not execute these tests. Independent review/tests,
an exact source protocol/PRE/archive, then one bounded synthetic connection gate
and its POST remain required. Native end-to-end qualification and any Gold11V6
physical successor remain separate gates. Root owns the future pure projection
wrapper and runner; neither is authored or connected by these files.

The new source does not resolve the known connected support-cover/registration
barriers by assertion, prove physical accuracy, or produce F1. Preserve all
existing residual thresholds, input/scoring custody and terminal dispositions.
