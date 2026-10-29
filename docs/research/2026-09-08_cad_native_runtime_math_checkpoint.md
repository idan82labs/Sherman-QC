# Installed native provenance seams and exact bend-axis mathematics

Later continuation: [engine-output scorecard and invented native-file reader receipt](2026-09-08_engine_output_scorecard.md)
adds bounded STEP/BREP/IGES reader checks. Statements below about unexercised
readers describe this earlier checkpoint; physical metrics and root grouping
remain unqualified in both.

Date: 2026-09-08. Continuation of locked Priority 3, following the user's request
to proceed carefully, use the available compute and double-check critical math.
This checkpoint adds actual installed-library evidence to the
[earlier guarded-double/source receipt](2026-09-08_cad_native_support_provenance_checkpoint.md).
It does not assert full native application or physical inspection qualification.

## Outcome and implemented change

Read-only package discovery found OCP and Open3D in the existing application
environment, but no OCC package. The importer previously required PythonOCC to
reach native owner/edge capture. [The importer](../../backend/cad_import.py) now
contains an adapter for the existing OCP binding: explicit native face/edge casts,
static `_s` API adapters, and orientation-preserving value copies. No package was
installed or upgraded. Provenance schemas and detector policies are unchanged.

Final review caught a promotion boundary: automatically preferring OCP would
change existing callers from gmsh/unavailable to an unqualified new backend.
The final importer therefore requires strict keyword-only **`allow_ocp=True`**
to admit OCP. Default `False` never even probes OCP and preserves the existing
OCC-then-gmsh route. A selected native binding/file failure is not retried through
another backend. The active pipeline is NOT opted in. File-reader and connected
qualification must precede any later default-activation decision.

The first native worker completed successfully. All 88 in-memory OCP cases, all
115 Open3D/source arithmetic checks and all 5,120 exact rational math cases passed
their predeclared criteria. No fixture, cutoff, cap or epsilon was adjusted after
results. There was no failed native attempt. After adding the opt-in gate, the
OCP-only matrix was revalidated against that changed source. Open3D and the exact
math matrix were not rerun. The OCP revalidation is a regression receipt, not an
additional 88 independent cases.

This advances the installed-native API seam from mocked to exercised. It does
NOT close file-reader/units/source-frame verification, full native pipeline
integration, root-role nomination, native finite-trim validation, manufacturing
event semantics, observed scan evidence or a physical milestone exit.

## Fixed native CAD comparison

[OCP fixture source](../../scripts/cad_native_ocp_checks.py) constructs every
shape in memory with native builders; it compiles only the named importer
methods and never imports the application or calls a file reader.

Frozen matrix:

- Three boxes: dimensions `(10,20,30)`, `(1,1,1)`, `(.001,10,100)`.
- Four proper rigid poses, three positive scales `1/8,1,8`, forward/reversed
  orientation: **72 cases**.
- Two differently located instances of the SAME native box, including a rotated
  and translated common parent: **12 cases**.
- The SAME unlocated box referenced through two distinct nested compounds:
  **four ambiguous-occurrence controls**.

For every case, capture-enabled and capture-disabled **OCP** tessellation points,
triangles, normals and V1 face ranges are exactly equal. This is not OCP-versus-
gmsh or OCP-versus-PythonOCC equivalence. Every face range is nonempty, contiguous
and contains its triangle indices. All emitted vertices inverse-transform into
a declared box and onto one of its boundary planes. Independent analytic corner
extrema check cumulative transforms; child translations are rotated by the
parent, not simply added in world coordinates. The bound is the predeclared
`128 * binary64_epsilon * max(1, abs(constructed corners))`, not a physical tolerance.

Each positive solid has six faces and twelve owner-local edges, each used once
FORWARD and once REVERSED, with no degenerate flag or cross-owner edge identity.
The ambiguous controls must clear the entire native map while retaining mesh
geometry and twelve import-local face ranges.

Receipt: **84 captured + four expected ambiguity abstentions = 88 paired cases**;
**624 face occurrences** checked; maximum analytic corner error
`9.094947017729282e-13` in invented coordinate units. Installed package reports
`OCP.__version__ = 7.9.3.1`; this is a binding version stamp, not a source-byte or
CAD-file kernel-version certificate.

Native curved-face seams, degenerate edges, open-shell/loose-face paths and
worst-case identity-comparison capacity are NOT newly qualified by these boxes.
Their previous guarded-double tests remain a separate source evidence ledger.

## Fixed native Open3D comparison

[Open3D checks](../../scripts/cad_native_open3d_checks.py) use the actual installed
Open3D **0.19.0** with exact-source detector classes, not a service/app import.
Frozen matrix: four invented strata × four proper poses × three scales × two
duplicate densities = **96 paired cases**, all passed:

| Stratum | Passed / attempted |
| --- | --- |
| Perpendicular flange grids | 24 / 24 |
| Sampled quarter-cylinder | 24 / 24 |
| Disconnected planar patches | 24 / 24 |
| Exact/nextafter voxel-boundary neighbors | 24 / 24 |

Two additional cap controls passed:

- **200,000 raw rows**: 10,000 distinct grid positions repeated twenty times;
  9,600 processed rows retained with **192,000 original contributors**, and
  **8,000 explicitly rejected original rows**. Exact complete partition and
  centroids; paired case plus independent arithmetic checks took **0.406517 s**.
- **200,001 raw rows**: lineage entirely UNAVAILABLE with
  `RAW_VERTEX_INPUT_INVALID_OR_OVER_BUDGET`; legacy geometry remains identical
  and nonempty (10,000 retained processed rows). **0.190091 s**. The cap limits
  additive provenance, not legacy geometry work.

The other seventeen checks: four exact rational rotations, eight actual-native-
trace controls (untouched positive and seven corrupted negatives), four numerical
source guards, and one nonempty native-preprocessing/stipulated-proposal transport
case. All passed; total **115 / 115**, zero unexpected abstentions/failures.
The transport case preserves **six original finite candidates through three
passes while legacy deduplication selects one**. Proposal/RANSAC geometry is
explicitly stipulated, not tested as native detection accuracy. This is not a
fully native CAD-file-to-inspection run.

All 98 paired geometry cases retain exactly the same positions and normals.
Every COMPLETE case has an exact retained/rejected original-row partition and
zero NumPy-centroid disagreement. Maximum binary64 rotation orthogonality error
is `1.1102230246251565e-16`.

### A deliberately retained numerical distinction

Fraction arithmetic on the actual binary64 coordinates, rounded grid bound and
voxel-size operands differs from floating subtraction/division/floor on **375 raw
row-instances across the paired cases**, by at most **one voxel cell**. These are
boundary roundoff diagnostics, not 375 physical point errors or failed lineage
records. For example, `nextafter(.125,0) + .125` rounds to `.25`, although the exact
rational sum of those float operands is below `.25`.

The lineage contract checks the actual native floating computation, not an exact
real-arithmetic tessellation or rotation-invariant physical voxel assignment.
This distinction and the first-failure field (`null`) are retained in the result.
The 200,000-row fixture is genuinely nonempty at the declared cap but has 10,000
distinct positions; it is not worst-case 200,000-distinct-row capacity proof.

## Independently checked critical mathematics

The [exact arithmetic simulation](../../scripts/cad_root_axis_math_checks.py)
uses Fraction throughout, with no numerical tolerance. For two unit plane
normals `n1,n2`, `d=n1·n2`, radius `R>0`, and signs `s1,s2 ∈ {-1,+1}`, the
cross-sectional cylinder-center displacement `c` is measured from the sharp
flange-plane intersection, with plane constants translated away and `c·a=0`.
It solves `n_i·c=s_i R` independently by 2×2 Cramer's rule. The separate
Gram-matrix derivation gives:

```text
||c||² = [s1 R, s2 R] [[1,d],[d,1]]⁻¹ [s1 R,s2 R]ᵀ
       = 2 R² (1-s1 s2 d)/(1-d²)
       = 2 R²/(1+s1 s2 d),           |d| < 1.
```

For perpendicular flange planes, the center axis is displaced **√2 R** from
their sharp intersection line. These are not interchangeable bend-axis meanings.
All four sign alternatives remain distinct; no inner/outer branch is guessed.

The tangent contact generator is `c - s_i R n_i + t a`, where `a` is the unit
cross-axis. Its points lie on both the plane and cylinder. Infinite tangency
does **not** identify the allowed finite `t` interval or the occupied trimmed
surface. Rigid translations cancel exactly; positive scales multiply radius,
displacement and finite parameters consistently.

Frozen **5,120 cases**: ten rational half-angle parameters × four signs × four
radii × four exact proper frames × four scales × two separately stipulated
finite intervals. Includes 512 near-parallel, 512 near-antiparallel, 512 right-
angle and 3,584 other oblique cases. Translations include `10^12`; scales span
`1/1000` to `1000`. The normal-system squared condition number, equivalently the
Gram condition number, reaches exactly `10^12`; `κ₂(A)` itself is `10^6`.

Receipt: **169,563 exact identity checks passed**, including 30,720 plane-membership
checks and 30,720 cylinder-membership checks (two generators × three axial
parameters × 5,120 cases), and 10,240 outside-interval controls demonstrating
that infinite tangency cannot certify finite support. Math-stage wall **1.466637 s**.
The fixed 8,192-case/512-bit guards remain unchanged. These are exact constructed
identities, not a floating numerical implementation, calibration or metrology proof.

## Execution and resource receipt

Root inspected named source/tests and native package startup paths before execution;
independent review checked native invariants, proper transforms and the bounded
worker. Neither an application module nor physical CAD/scan/held-out/email,
manifest, control/authorization or one-shot artifact was accessed. No service,
database or credential initialization occurred.

First focused native-free regression command:

```sh
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_selected_cad_attribution.py tests/test_cad_import_provenance.py tests/test_cad_geometry_loader_provenance.py tests/test_cad_native_topology_capture.py tests/test_cad_support_lineage.py tests/test_cad_provenance_handoff.py tests/test_cad_provenance_end_to_end.py tests/test_cad_ocp_binding_adapter.py tests/test_cad_native_open3d_checks_source.py tests/test_cad_root_axis_math_checks.py -k 'not one_complete_exact_matrix' -q
```

**386 passed, one intentionally deselected in 4.57 s**; external wall **4.68 s**;
peak RSS **93,487,104 bytes**. The full-matrix pytest wrapper was deselected so
the 5,120-case matrix ran only once, in the following worker. Its extra wrapper
assertions are not represented as executed pytest coverage.

First combined native worker command:

```sh
.venv/bin/python -I -S scripts/check_cad_native_provenance.py
```

Predeclared limits: **300 s wall, 240 s CPU, 2 GiB process-tree RSS**, 1 MiB output.
One child process, isolated temporary HOME/cache, explicit one-thread BLAS/OpenMP
settings, no `.pth`/site startup or inherited credentials, and Python guards
against process/network calls and repository reads outside named source/package
paths. Reviewed native builders use memory-only inputs. The Python audit hook is
defense-in-depth, NOT an operating-system sandbox. OCCT's legacy parallel-meshing
setting was retained; OMP limits do not assert all native thread pools are serial.

Observed: parent wall **10.978789 s**; worker wall **10.790497 s**; worker CPU
**5.347597 s**; worker high-water RSS **574,980,096 bytes (548.3 MiB)**; sampled
process-group peak **573,800,448 bytes**. Exit **0**, no resource stop. RSS is
sampled with group termination on a breach, not hard memory admission. This
small invented workload is not scanner-scale production latency qualification.

After the first native run, independent review identified a nonblocking output-cap
hardening: the crossing chunk had been forwarded before the cap check. Root
changed the parent to truncate before forwarding and added six pure byte-budget
regressions: **6 passed in 0.03 s**. The successful native run did not hit the
output cap. The initial executed parent hash below is preserved rather than
replaced retrospectively.

### Final explicit-opt-in revision

After inspecting the gate changes, root ran only the affected importer/loader/
topology and binding regressions, using the same isolated pytest flags as above:

```sh
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_cad_import_provenance.py tests/test_cad_geometry_loader_provenance.py tests/test_cad_native_topology_capture.py tests/test_cad_ocp_binding_adapter.py -q
.venv/bin/python -I -S scripts/check_cad_native_provenance.py --ocp-only
```

Final gate source: **183 passed in 3.19 s**, external wall **3.27 s**, peak RSS
**74,252,288 bytes**. The fixed OCP-only worker again passes the same **88 cases,
84 captures/four expected ambiguity abstentions, 624 face occurrences**, with
the identical corner-error maximum. Parent wall **0.591710 s**, worker wall
**0.512903 s**, CPU **0.837872 s**, worker/sampled-group RSS **312,623,104 bytes**;
exit 0, no stop, same limits. The retained legacy mesher can use native threads;
CPU may exceed wall time. This warm OCP-only measurement is not comparable to
the first combined worker's startup or a production latency result.

The only fixture-source change was explicit `allow_ocp=True` in both paired
constructors. The parent gained the fixed OCP-only selector and output-cap
hardening. No source or fixture affecting these executed results changed afterward.

## Research grounding and next decision

- [OCCT topology/geometry and location model](https://occt3d.com/dev/doc/overview/html/occt_user_guides__modeling_data.html)
  distinguishes native incidence from geometric surfaces and trimmed faces.
- [OCP native shape API declarations](https://github.com/CadQuery/OCP-stubs/blob/master/OCP-stubs/TopoDS/__init__.pyi)
  support explicit Face/Edge casts and orientation-preserving value returns.
- [Open3D PointCloud APIs](https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)
  expose trace indices and outlier selection; installed behavior, not documentation
  alone, was checked above.
- [OCCT cylinder parameterization](https://occt3d.com/dev/doc/refman/html/class_geom___cylindrical_surface.html)
  separates axial lines and circular cross-sections. The independent derivation
  above grounds the flange-intersection/cylinder-axis distinction.
- [Tseng and Joshi, 1994](https://www.sciencedirect.com/science/article/pii/0010448594900183)
  is background on feature interpretation. Our equal-geometry/different-process
  label control only shows that geometry omits process identity; it is not a
  proof that every invented solid is manufacturable by either process.

Retain the verified native seam and exact-math evidence. Before grouping, nominate
an explicitly bounded native transition hypothesis with transformed analytic
surface parameters, correct axis semantics, direct local adjacency, tangency,
finite trims and source/frame/unit binding. Incidence or shared flange context
alone is insufficient: a remote common flange can bridge separate tabs, and
the same incidence can embed a chamfer, cylinder or spline. A periodic tube seam
is not automatically a bend boundary. Keep unsupported/ambiguous cases unknown.

Do not promote CONTEXT flange faces to root support. Grouping inputs stay absent,
`grouping_eligible=False`, observed state UNKNOWN, and the research grouping
function remains uncalled. Native file readers and a fully native connected
pipeline still require invented-input verification; no physical run follows from
this receipt. After these prerequisites, continue finite scan-supported mapping
with registration uncertainty. M2 capture/calibration and M4 operational essentials
remain parallel priorities, not replaced by more simulations. M3's physical exit
is open; M5/M6 have not begun. C15's retained failures, terminal EXP10/EXP35,
physical/release gates and the paused goal are unchanged.

## Source identity at the first combined native execution

```text
backend/cad_import.py
e6383d696f91c46824477f009e8a5e35434a47a0fffc84ef07c886896f1f479b
backend/feature_detection/bend_detector.py
4080c7f397ead65a77d81ec32d6f07ada6ec20cbaa7c8e2bbd8c2ecdf61f5dd2
backend/cad_provenance.py
313b9cbb4109c8f80c3f1f900a26c9793529aedbc6cd296ff4285b6728e13e10
backend/bend_inspection_pipeline.py
4054b5e9096aa95ca96c396970e331d45a01abf709e43eed057cd772f0982a69
scripts/check_cad_native_provenance.py (executed, before output-cap hardening)
34550e2ca54f80c175c04ac828e22125cd10e3bb67722257b7bc58fb0e25db76
scripts/cad_native_ocp_checks.py
dffbb8e6b25295393130718e3442a1be7645c62175a1819d9700b7a44389ec75
scripts/cad_native_open3d_checks.py
fa0b0bb09f097a30603ef437f9130bb394b9e613d66d2c08648e464d0d53629c
scripts/cad_root_axis_math_checks.py
71868897f103bb276f5fe92fc5a1aa8c95ac835008c178e9856a679f5697f7a0
tests/test_cad_ocp_binding_adapter.py
d10ca37fe3617aaaec0f33e7635727c57e1f04c25cae2a9ed17627d25f7e4903
tests/test_cad_native_open3d_checks_source.py
04c68a7b7ecf557e344f33afa2ed71036a3ce96bfa87793b422be3eeb801077a
tests/test_cad_root_axis_math_checks.py
5742c257e1fb69f9e0db3d1fc348dfd82887caec89d319caab1250df4f0eee29
```

Final opt-in revision/revalidation changed only the following receipt identities
(other source/check hashes above remain unchanged):

```text
backend/cad_import.py
3d06d2b1601f14bd4f5e489d7835dfeee7005c2231408efcf4a9d9711605dd2f
tests/test_cad_ocp_binding_adapter.py
ffa1cb48daa7f5d511ee5dfbc1551c28c045fbdc98063bc48c0c7176536fd106
scripts/check_cad_native_provenance.py
308e2543de73ee91541813eaa5ca18ea29e1c8b795a48de5c513bf16bbcffc76
scripts/cad_native_ocp_checks.py
aca1fe337082ee84a86324879ed4d487c694fe16d10f4c25dc7486e6621fb72b
tests/test_cad_native_worker_output_budget.py
97e3484eda7c630650124a6f780ea6e6dee24057763232a94cc72a97bee1ed64
```
