# V11 original thin-part diagnostic at the unchanged physical configuration

Date: 2026-09-10. Prospective, source-only protocol. No native execution, physical
input access, qualification or promotion is implied by this document.

## Question and frozen predecessor

Can the V11 triplet-plus-two-plane/context path produce the original required
measured output on the exact two thin-part cases that failed in the V9
physical-configuration diagnostic? The terminal V9 attempt remains unchanged.
This is a separately versioned mechanism comparison, not a retry or relaxed
operating condition.

The test is `tests/test_cad_guided_mesh_physical_config_constructed_v11.py`.
Its complete CONFIG expression must have identical AST to both
`tests/test_cad_guided_mesh_physical_config_constructed_v9.py` and the frozen
`scripts/run_gold11_mesh_development_v2.py`. The physical runner is read as
source only, never imported or invoked. Neither its inputs nor old result
artifacts are accessed by this test.

The two cases, in their original order, are:

1. `v1_physical_config_clean_nominal`: unchanged clean analytic raw mesh.
2. `v1_physical_config_bounded_3d_noise_nominal`: the same mesh with the unchanged
   original deterministic row-ID 3D perturbation.

Both use the original independently created asymmetric R5/R6, 1mm-thick native
strip, unequal 20/29mm flanges, 30mm extrusion, 2752 raw vertices and 5080 mesh
triangles. The original .8mm-wide remote cap sampling is not thickened. The
original native shape/STEP writer, analytic scan, quaternion pose, PLY encoding
and represented-noise helper are reused unchanged. CAD triangulation does not
generate scan observations. No pose, Gold answer, feature identity, fitted
radius or correspondence reaches inference. The known pose is used only to
construct the raw input and evaluate the returned pose/lines afterward.

## Unchanged configuration and numerical criteria

The complete original physical CONFIG remains fixed: epsilon .05mm,
`free_circle`, 32768 CAD samples, .1mm/.2rad tessellation, requested 2048/8192
registration rows, 30+30 iterations, .85 trim, 2mm finite correspondence,
.5mm P95, .75/.25 scan/CAD coverage, .005 observability ratio, all measurement
and support-window fields, and every existing internal work/size bound.
The .005mm easier four-case synthetic configuration is not used.

The same V3 certificate budgets already explicit in the V9 diagnostic remain:
initializer 1,024,000,000 pairs, per-query surface 128,000,000 pairs, cumulative
registration 8,000,000,000 pairs. Constructor snapshots must match every CONFIG
field plus the exact V11 type names, native adapter `v1` and these additive
certificate fields. No requested/effective-row confusion is allowed: all 2752
raw rows remain available, but this cannot exercise 8192 effective refinement
rows. The retained ledger reports the actual coarse/refined row counts.

The original free-circle oracle is retained verbatim. It is the original
nominal assertion AST with only the previously declared fit-model and tangency
role string replacements. Its noisy invocation changes only the detached
`CRITERIA` dictionary, never version strings or execution-check globals.

| Criterion | Clean nominal | Original noisy nominal |
| --- | ---: | ---: |
| Radius absolute error | 2e-5 mm | .01 mm |
| Absolute sweep error | 2e-5 degrees | .1 degree |
| Sampled generator endpoint error | .02 mm | .05 mm |
| Sampled axis endpoint error | .02 mm | .05 mm |
| Full-row pose-action maximum error | .20 mm | .20 mm |

The noisy input keeps the fixed `.999` rounding reserve and hashes the
represented displacement `noisy_raw - clean_raw`; its maximum Euclidean norm
must remain <=.001mm. These are stipulated simulator bounds and engineering
error criteria, not scanner calibration or manufacturing tolerances. No
criterion changes after first execution, including for a nominal abstention.

## Additive V11 lane and exact budget contract

Only the public V11 workflow/initializer dispatch and the additional pair lane
change the inference mechanism. The actual call explicitly sets
`two_plane_enabled=True`. No implementation function or pose is monkeypatched.
The public V11 API uses the existing helper/scheduler defaults; the test freezes
their full literal `PAIR_DEFAULTS` dictionaries, checks each against the
dataclass default, checks the two calls have no budget override, and records
both dictionaries in every case and the compact receipt.

Scheduler defaults admit at most 128 scan-bank equations and 2048 CAD equations,
8128 scan-pair records, 64 selected scan pairs, 520192 scan selection tests,
2096128 CAD pairs, 44018688 sort-work proxy units, 134152192 window pair tests,
64 exact layouts, 128MiB exact layout keys, 256MiB accounted payload, 32MiB report,
and 65536-row chunks. Every selected unordered CAD pair retains both orders.
The scheduler is label-independent and bounded, not exhaustive or guaranteed
invariant to bank/ID ordering. Repeated exact layout exposure is not global
correspondence recovery.

Context defaults admit at most 1M scan rows, 2M scan triangles, 500000 CAD
triangles, 128/2048 scan/CAD planes, 64 scan pairs and four ordered CAD matches
per pair; 1024 signed families and 8192 candidate slots; 64M scan-row visits,
128M CAD-vertex visits, 64M index visits, 256M index-sort work, 1GiB hashed bytes,
6144 projection-cache entries, 256MiB accounted payload, 32MiB report and
65536-row chunks. Accounted payload is not process RSS or a runtime guarantee.

Triplet records reserve the joint quota first. Each scheduled ordered pair
reserves one unsigned assignment and all 32 sign/slot records; no rejected sign,
absent midpoint or exact duplicate refunds capacity. The joint limits remain
8192 records and 1024 unsigned identities. Pair-lane starvation is retained, not
hidden by refilling or selecting a favorable prefix. All passing origins share
one unchanged at-most-48 group refinement schedule.

The post-inference checker verifies all triplet records, pair match orders,
four signs per match, eight exact slot identities per signed family, absent and
rejected slots, complete candidate-origin unions, input digests and count
reservations. Proposed triplet/pair rows must have exactly three/two actual
assigned supports. No synthetic third correspondence is allowed. All generated
proposed signs must be finite-screened under the unchanged support policy.

The frozen V10 post-inference box checker is called directly, without altering
its globals. It requires complete bound-face digests, all original assigned
support rows, strict conservative box exclusions and separate full finite-query
completion counters for nonexcluded supports. Box exclusion is not finite-query
completion. V11-specific schema, joint-lane and full group-partition checks are
implemented explicitly in this new test. Refinement must preserve all selected
candidate records, zero numerical failures, exact group origins and the actually
selected transform. Existing uncertainty, observability and ambiguity gates
remain binding; two planes alone do not constrain axial translation or establish
a unique global pose.

## Completion, positive output and failure retention

Three distinct ledgers are mandatory:

- Diagnostic completion: both original cases returned a schema-valid bounded
  report and their complete records were retained under unchanged source hashes.
- Numerical completion: scheduled proposal, support and selected refinement
  work completed with truthful denominators. Empty schedules/refusals are
  explicit and do not count as completed refinement.
- Positive geometry: exactly one required actual full-raw measured output,
  conditional valid selected registration, complete measurement computation,
  and all original clean/noisy error ceilings passed.

A nominal zero-output result is `FAILED_NOMINAL_NO_PREDICTIONS`, never a passing
missing-arc sentinel. Numerical work can complete while both positives fail.
The native pytest case asserts diagnostic recording only, exactly as the V9
diagnostic did. A green pytest exit must not be called positive engine progress:
`positive_geometry_matrix_passed` must independently be true for both cases.
Wrong geometry/count, missing metadata, incomplete queries, selected-pose
substitution, resource/solver/input/report failure or lost origin cannot pass
positive output. A missing or malformed returned stage stays UNKNOWN; it is not
assumed never to have run. Full returned workflow records are retained even when
the positive or numerical oracle fails.

Mechanism attribution is an additional, separate ledger. After geometry passes,
`two_plane_origin_recovery_passed` requires a genuine two-support pair origin in
the selected refined group. `two_plane_origin_recovery_matrix_passed` requires
both original geometry positives and both such origins. A correct triplet-only
result still counts as correct output, but cannot be claimed as pair-lane
recovery. Neither attribution nor a proposed pair seed relaxes a geometry gate.

The source controls include explicit report-only fabricated ledgers and mocked
recording dependencies. Those are recorder tests, not geometric/inference
positives. Actual native cases use neither such ledgers nor mocked runtime.

## Prospective qualification, custody and resources

Prospective test count: 55 pure/source/mock outcomes plus one native skip;
native mode 56 pytest outcomes, with the separate two-case output ledger.
Root must independently read/freeze the complete dependency closure and this
protocol after prerequisite helper, scheduler and focused V11 integration gates.
The test author also authored the scheduler dependency; therefore this file is
not an independent approval of that runtime. Root and a different reviewer must
audit the scheduler and these oracles before release. No candidate import,
test/native execution or physical access occurred during test authoring.

The proposed source route is `mesh_physical_config_constructed_v11_source`,
60s monitored wall, 45s worker CPU, 1GiB sampled process-group RSS, 1MiB stdout.
Only after independent exact source pre/post-GREEN and unchanged closure may root
release one native route `mesh_physical_config_constructed_v11`: 300s monitored
wall, 240s worker CPU, 4GiB sampled process-group RSS, 1MiB stdout. These native
limits are identical to the prior diagnostic, not increased.

Exactly two detailed JSON artifact slots, <=16MiB each and <=32MiB total, are
written exclusively in root's fresh retained `ARTIFACTS` directory before a
positive summary. The worker writer and two fixed slots enforce artifact limits;
this is not an independent parent filesystem quota. Partial artifacts remain
negative evidence. Source/receipt/hash closure and complete output inventory
must be checked independently; source bytes cannot change between started gates.
Only the native route may set
`SHERMAN_RUN_INVENTED_MESH_PHYSICAL_CONFIG_CONSTRUCTED_V11=1` and
`SHERMAN_MESH_PHYSICAL_CONFIG_CONSTRUCTED_V11_ARTIFACT_DIRECTORY`.

Root freezes the actual supervisor route and its phase/deadline/cleanup semantics
before execution. Monitored wall is not automatically a total-parent deadline;
sampling does not establish a hard OS memory bound. CPU signals, absent artifacts
and partial execution are failures even if an outer `budget_stop` is null.
One heavy job runs serially, with isolated environment and numerical threads one.
No services, network, application startup, physical/raw CAD or scan, Gold,
cached predictions, held-out or incoming-email data enter this test.

Even a 2/2 positive qualifies only this exact constructed thin-part path. It does
not qualify broad noise, arbitrary partial/occluded scans, missing-bend detection,
multi-bend grouping/count, native invalid-solid recovery, calibrated bands,
historical comparability, F1 improvement, Gold11 execution or floor release.
The next output comparison requires its own source/custody/authority gate.
