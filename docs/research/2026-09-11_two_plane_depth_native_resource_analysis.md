# Depth-native resource analysis — source-only, provisional

## Disposition

**The 32,768/284 = 115.3803 CAD-point ratio is not a registration-time multiplier.**
The current optimizer queries transformed scan rows against finite CAD triangles;
it does not query all CAD samples on every iteration. Native sampling adds real
work, including repeated byte-binding checks, but native tessellation and the
selected finite search remain separate, potentially larger uncertainties.

Leave the native CPU/work/parent time constants `None` for now. The evidence
below does not establish a proportionate numeric completion budget for native
L-R. A future finite engineering ceiling must follow the corrected connected
terminal result and independent GREEN POST, then a new exact native PRE. This
note grants no run authority, changes no source or limit, and does not promote
the held optimizer prototype.

This review read named source/protocol and released narrative reports only. No
target imports, collection, tests, geometry/native calls, active scientific
artifacts, raw/control data or experiment replay were performed. The existing
97-source connected freeze is unchanged. The native draft is still 39 outcomes:
38 pure/mock checks and one actual L-R, not 39 native registrations.

## Dimensions and source routing

Use distinct dimensions:

- `S <= 650`: unique selected L-R scan rows; its source fixture has 650 raw rows
  and 1,176 scan triangles. Coarse/refine requests are 2,048/8,192, so neither
  increases this fixture's actual selected row count.
- `M = 32,768`: generated native triangle-interior samples; `M_c <= M` after
  canonical position deduplication. The array fixture instead supplies 284
  unique CAD vertices as initialization points and 564 CAD triangles.
- `T`: actual native tessellation triangles, not `M`; `F` is its face/plane
  structure. Neither is established by an unexecuted native draft.
- `H <= 8,192`: joint hypothesis records; `K <= 48`: selected whole-part
  refinement groups; `C <= K`: published conditional measurement contexts.

The preserved native policy is 0.1 mm linear / 0.2 radian angular meshing, fixed
32,768 samples, and the original registration/measurement policies. Millimetres
remain a provisional common-unit assumption, not scanner calibration.

| Stage | Work-driving inputs | Effect of increasing CAD sample count alone |
| --- | --- | --- |
| STEP construction/import and native feature extraction | Native topology, faces, edges, feature windows | None: these receive a shape, not the sampled point cloud |
| Native tessellation/export | Shape, meshing policy, mesh nodes and `T` | None: meshing runs before sample allocation |
| Area-distributed sample construction and incidence binding | `T`, `M` | Direct: CDF allocation, barycentrics, arrays, hashes and incidence checks |
| Scan plane preparation | Original scan positions/topology/error bounds | None |
| CAD plane extraction, equation search, two-plane context generation | Native CAD triangles/faces/planes, scan proposals and fixed quotas | None directly; a changed native mesh can change the resulting search |
| Assigned-face screening and finite prefilter | Hypotheses, owned scan support rows, complete assigned-face triangles | None directly |
| V3 coarse/refine optimizer | Selected scan rows, candidate poses and whole-CAD triangle index | None directly |
| Final candidate reverse coverage | `M_c` query points against a tree on transformed scan rows | One sample-sensitive KD query per diagnosed candidate |
| Byte/context publication binding | `C`, CAD points/triangles and report structures | Repeated CAD-point canonicalization and hashes |
| Full-raw support-cover measurement/publication | Scan rows, native feature windows, contexts and support groups | No sampled-CAD-point fitting; native feature/context changes may change work |

This is a routing distinction, not an equivalence claim. Interior area-distributed
samples and vertex samples measure different empirical reverse coverage. With
the unchanged distance/coverage gates, that can alter conditional-context
eligibility. Native triangulation also changes finite support geometry and plane
proposals. Same nominal strip construction does not imply the same candidates,
optimization trajectory, coverage, result or native runtime.

Source anchors: `cad_guided_mesh_offline_v13.inspect_cad_guided_mesh_v13` passes
sampled points and triangles separately; initialization V12 passes only scan
and CAD-triangle inputs to `_initialize`. Its `_prepare_triplets` uses V8
`_prepare`; CAD plane fitting consumes complete face triangles. V3 `_refine`
uses finite surface targets. V3 `_diagnose` alone queries
`cKDTree(transformed_scan).query(canonical_cad, k=1, workers=1)` for reverse
coverage. V12 `_context_support` reuses that scalar but performs its own finite
surface query on selected scan rows.

## Source-derived operation bounds

These bounds concern admitted calls/rows under this fixed small fixture, not
CPU instructions, time, resident memory or guaranteed successful completion.

### Whole-part refinement and diagnostics

Each candidate has at most 30 coarse and 30 refine queries. Diagnostics add two
queries and connected context support adds one. Therefore:

```text
whole-part finite calls <= K * (30 + 30 + 2 + 1) <= 3,024
whole-part finite query rows <= 48 * 63 * 650 = 1,965,600
reverse KD query calls <= K <= 48
native reverse KD query rows <= 48 * 32,768 = 1,572,864
array reverse KD query rows at the same K <= 48 * 284 = 13,632
```

The eight-step refine backtracking loop evaluates fixed materialized targets;
it does **not** add eight finite-surface queries. Early convergence/refusal can
reduce work. These bounds exclude initialization's separate face queries,
index construction, byte bindings, measurements, recorder and oracles.

One whole-part triangle index is built, then reused across candidates. Query
cost still depends on triangle/node traversal, near ties and interval work,
not just query-row count. The actual finite query ledger retains its independent
caps: 4,096 calls; 64,000,000 rows; 2,000,000,000 node visits and triangle pairs;
8,000,000,000 certificate pairs; 500,000,000 normal pairs; and
20,000,000,000 weighted work units. Each query is also constrained by its smaller
surface budget. These are refusal limits, not proof that the native case fits.

V3 validates the declared work identity:

```text
work = node_visits + triangle_box_pairs + 8*triangle_pairs
       + certificate_pairs + 3*normal_pairs
```

A complete query has two completed passes and four certificate pairs per
triangle pair. No factor in that identity charges KD-tree work, JSON traversal,
NumPy sorting, native meshing or operating-system work. It cannot be converted
to CPU seconds without appropriate measurements.

### Initialization and finite-face prefilter

Each scheduled hypothesis has two or three assigned scan supports. With at
most 8,192 records, each support a unique subset of the 650 original rows, and
the fixed chunk size larger than 650, there are at most 24,576 support calls
and 15,974,400 row tests **per** box-screen / finite-query pass. Box exclusion
can eliminate a whole support's finite query but cannot grant support or remove
selected rows from a required query. This is a loose mixed-lane bound, not a
claim that all records are proposed or all supports contain 650 rows.

The separate policy still caps prefilter calls/rows at 32,768/64,000,000 and
weighted work at 4,000,000,000; all original triangle, certificate, normal and
index-build caps also remain. Assigned-face indices are eagerly built over
their complete retained triangle memberships. Native face tessellation and
proposal multiplicity can change this cost even though `M` never enters it.
Increasing depth is not merely accelerating an unchanged trajectory.

### Sampling, validation and repeated binding

`sample_triangle_interiors_v1` checks all `T` triangles, constructs their area
CDF, and issues `M` `searchsorted` targets. Fixed Halton indices 1..32,768 require
16 base-2 plus 10 base-3 whole-array digit iterations. Barycentric construction,
position/normal generation and hashes are sample-sensitive. `_surface_binding`
then checks every sample's triangle/face/local identity and recomputes its
represented affine position and triangle normal. This is incidence validation,
not metrology calibration.

The workflow's context list invokes V12's binder once per published context;
that binder calls V8's `_registration_binding`. Every invocation repeats
`np.unique(cad["points"], axis=0)`, canonical/sample hashes and triangle hashes.
Together with registration's initial canonicalization, there are at most 49
CAD-point canonicalizations: at most 1,605,632 input rows across 49 separate
sorts. The prepared scan-plane token does not cache these CAD checks. KD query
implementation work and sort work lack corresponding runtime counters here;
the 115.3803 ratio describes sample rows for a fixed candidate count only.

The six retained sample arrays contain 96 bytes per sample: points, normals,
barycentric weights, and three integer identity arrays. At 32,768 this is exactly
3,145,728 B (3 MiB) of array payload. A gathered `M x 3 x 3` triangle buffer alone
is another 2,359,296 B (2.25 MiB). These are identified buffers, **not** total or
peak RSS: copies, temporaries, full CAD triangles, native allocations, BVHs and
Python reports coexist. The point-count increase alone is not evidence for
raising the draft 4 GiB process-group ceiling.

## Native uncertainty and timing evidence

Native meshing is invoked once, with relative and parallel flags false, before
sampling. Source admission bounds topology to 32,768 occurrences/depth 32 and
1,024 faces; export caps mesh-node occurrences at 1,000,000 and triangles at
500,000. The Python export loops obey those post-mesh counts. They do not bound
the OCP mesher's internal time/allocation before it returns. STEP import and
analytic feature extraction also have native work not represented by the
finite-query ledger. An external supervisor is necessary, not optional.

Released historical observations are informative but inadequate for a native
L-R estimate. The composite connected POST dated 2026-09-10 records registration
CPU of 210.721639 s for U-R (11 starts), 872.147698 s for U-A (48 starts), and
122.282361 / 80.586032 s for L-R/L-A (zero starts). The complete older packet used
1,292.328019 CPU seconds and 582,025,216 B worker high-water RSS. Its L cases
never performed the missing whole-part refinement path; dividing their time by
sample counts would be meaningless. Those are historical engineering reports
under another schedule, not qualification of the current candidate or native
completion predictions.

During this review root separately released active **OUTPUT-only** progress:
current U-R shared registration 1,047.316910 CPU / 1,048.125053125 wall seconds;
V2 measurement 1.697320 CPU seconds with its original oracle provisionally
passing. This is a relayed partial observation, not independently inspected
scientific evidence or a terminal four-case result. No current start/triangle
counts are inferred. It supplies no completed current L-R or native timing.

## Prospective budget decision

The source supports keeping the existing draft 4 GiB sampled owned-group RSS,
1 MiB stdout and 129 MiB result transport / 138 MiB + 4096 B aggregate transport.
The result allowance is the separately checked 1 MiB recorder envelope plus the
unchanged 128 MiB workflow cap; it is not a memory guarantee or extra engine
work allowance. CPU/work/parent timing must remain unset at this checkpoint.

After a terminal corrected connected GREEN POST, review the actual L-R and
other per-stage times together with selected/attempted starts, actual iterations,
box exclusions, completed finite queries and work ledgers, report sizes and RSS.
That can anchor an **engineering** time allowance for the unchanged native
fixture, with explicit uncertainty for import, meshing, reverse coverage and
binding overhead. It cannot certify that OCP will complete inside any selected
time. No valid coefficient presently converts either `M` or weighted query
units to seconds; copying the four-case 6,000-CPU budget, dividing it by four,
or multiplying it by 115.3803 would not be an evidence-derived native estimate.

The native recorder presently separates fixture and whole-workflow timing, not
every internal stage. Do not claim a future aggregate workflow time measures
the sampling increment alone. This review requests no instrumentation change,
extra benchmark, reduced sampling, oracle relaxation or optimizer substitution.
Any frozen ceiling remains a terminal refusal policy: budget exhaustion leaves
the one native case unsuccessful/unknown, not an absent bend, and does not
authorize a retry. Final source closure, synchronized literals, exact PRE,
nonoverlap and root release remain mandatory.

## Reviewed source anchors

Named routing/math sources: `backend/cad_guided_mesh_offline_v13.py`,
`cad_guided_mesh_offline_v12.py`, `cad_guided_mesh_offline_v8.py`,
`cad_guided_mesh_offline_v2.py`, `cad_guided_cad_surface_v2.py`,
`cad_guided_cad_sampling_v1.py`, `cad_guided_global_registration_v1.py`,
`cad_guided_global_registration_v3.py`, `cad_guided_plane_initialization_v8.py`,
`cad_guided_plane_initialization_v12.py`, and the relevant native-feature source
signature/count declarations. Fixed fixture/policy sources:
`tests/test_cad_guided_two_plane_depth_native_v1.py`,
`tests/test_cad_guided_connected_v12_native_bytes.py`,
`tests/test_cad_guided_connected_v12.py`; draft native harness/protocol;
`2026-09-11_two_plane_depth_v1_native_source_review.md`; and released
`2026-09-10_composite_plane_v1_connected_post_run_audit.md` / source-only
`2026-09-10_proposal_optimizer_performance_review.md`. This is not a new complete
source-custody ledger or an exact native PRE.

Latest draft identities independently checked after root's transport-only edit:

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `tests/test_cad_guided_two_plane_depth_native_v1.py` | 48117 | `ed2acb8dde5881e1f36697bec056b672b2a48e25612559612feb88d38fd12974` |
| `scripts/check_cad_guided_two_plane_depth_native_v1.py` | 10773 | `f9b145c95f1586f8b72726570813975d0808bb269ee39acd341e305ecb6ea6d9` |
| `docs/research/2026-09-11_two_plane_depth_v1_native_protocol.md` | 15885 | `013dfc4c2f80455eac2925e66da859b6b820a57605aed9ac87dcd8b9a3da5e48` |

The final inventory open now uses `O_RDONLY | O_NOFOLLOW | O_NONBLOCK` before
checking regular-file type; a substituted FIFO cannot block in that open.
This does not bound every synchronous OS call or alter engine work. Times are
still `None`, the 39 outcomes are unchanged, and final PRE must review both this
edit and the updated prospective 104-source inheritance, including the then
available corrected connected POST. No future POST hash is inferred here.
