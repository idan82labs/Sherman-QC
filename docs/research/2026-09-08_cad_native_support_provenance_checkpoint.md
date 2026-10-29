# Native CAD incidence and pre-contraction support provenance

Later continuation: [installed-native seam and exact-math verification](2026-09-08_cad_native_runtime_math_checkpoint.md)
adds actual in-memory OCP/Open3D execution. The no-native-execution statements and
345-check receipt below describe this earlier source-only checkpoint, not the
later work. Root-role/manufacturing authority remains unverified in both.

Date: 2026-09-08. User-authorized closure of the remaining provenance-loss defect
under [locked Priority 3](2026-09-07_production_generalization_roadmap.md#locked-execution-baseline--user-confirmed-2026-09-07).
Only source, invented in-memory geometry and guarded native-library doubles are
in scope. No physical CAD/scan, email, held-out, manifest, control or one-shot
artifact is opened. No native package, application, service or worker is run.

## What is being closed

The [prior repair](2026-09-07_cad_attribution_provenance_checkpoint.md) closed
caller-set reuse and unlimited-distance attribution, but preserved only local
face ranges. Native owner/edge incidence still disappeared at import, and original
support/candidates disappeared during preprocessing and geometric contractions.

This repair retains those records through the active CAD path. It does NOT
replace the existing geometric detector or its selected output, authorize native
root grouping, or validate manufacturing-event semantics. In particular, a
legacy selected specification may still span multiple original candidates; the
new ledger preserves their separate original finite segments and support instead
of treating the selected envelope as their identity.

## Native occurrence capture

[CAD import](../../backend/cad_import.py) now carries a separate optional fifth
`CADMesh.native_topology` field. The existing three/four-argument constructors,
mesh arrays, normals and V1 tessellation range field remain compatible.

The source traverses native child occurrences with cumulative location and
orientation. Each face belongs to its nearest solid occurrence, otherwise shell,
otherwise that loose face; a compound or shared coordinate is not an owner.
Native edge identity is scoped to that owner, with each oriented use and its
degeneracy flag retained, including seams. Faces must match the exact retained
tessellation enumeration one-to-one, including orientation. A repeated coincident
native face is ambiguous and clears the entire native map rather than receiving
an invented occurrence assignment. Different native tokens can share coordinates.

Fixed additive budgets: 4,096 faces, 32,768 edge uses, 65,536 traversed nodes,
depth 64 and 1,000,000 native identity comparisons. These are simultaneous upper
bounds: the comparison cap can be reached before the face cap. An API failure,
ambiguous identity or exceeded budget leaves native mapping UNAVAILABLE, without
changing the existing geometry result. This is not native-scale qualification.

The retained identity is import-local, not a persistent CAD/XCAF assembly label,
verified source-byte identity, manifold certificate or physical bend event.
Gmsh still lacks this native incidence path and remains unavailable for it.

## Original support through preprocessing and contraction

The [primary detector](../../backend/feature_detection/bend_detector.py) has an
additive bounded lineage mode; the default scan API and selected geometry stay
unchanged. The active CAD path opts in. The original voxel result remains the
geometry input. An additional native trace is accepted only when point order,
coordinates, normals, voxel-cell memberships and raw-row centroids agree exactly.
No proximity search, coordinate welding or reordered best match repairs a trace.
Outlier indices must reproduce the exact retained point/normal rows.

Every original vertex is accounted for as a contributor to a retained processed
row or an explicitly rejected preprocessing row. Each segmentation pass records
original plane supports, rejected rows and unassigned rows. Coplanar merges retain
their constituent source-plane pieces. Every pre-dedup bend gets an immutable
invocation-local identity and copied original finite endpoints with both flange
memberships. Confidence deduplication, multiscale deduplication and rolled merging
retain all these records while preserving legacy selection and geometry.

Fixed bookkeeping budgets: 200,000 raw vertices, 4,096 original candidates and
4,000,000 support references, including original plane pieces. They do not cap or
qualify the legacy geometry work. Any late failure clears the entire lineage,
including surviving candidates and original/merged plane objects. Temporary native
clouds and the trace matrix are released before outlier/normal work; unsupported
native versions or exact-order/numeric disagreements explicitly lose provenance.

## Active handoff and its authority boundary

[The loader and pipeline](../../backend/bend_inspection_pipeline.py) now request
both face ranges and native incidence, pass provenance into the primary CAD
extractor, and retain its full pre-contraction ledger even if a legacy extraction
strategy is later selected. Existing loader defaults still return three values;
the prior face-only option remains unchanged. The new native option requires the
explicit face-provenance option.

The pure [handoff](../../backend/cad_provenance.py) checks native schema/occurrences,
complete original-row partitions, per-pass segment identities, exact candidate
support bindings and complete face-range coverage. One voxel cannot split across
two segmentation categories. Original row IDs map to their emitted native face
blocks, not the closest coordinates. Each pre-coplanar piece remains separately
available. Missing or altered selected-strategy lineage cannot inherit the primary
ledger or retain a successful selected prefix.

The full ledger remains in `BendInspectionRunDetails`; ordinary report serialization
exports only a compact, non-authoritative summary. Source topology/frame/units,
physical geometry and event verification stay false/unknown. All flange faces are
explicitly CONTEXT. Root-role nomination is NOT supplied, grouping eligibility
stays false, and the research grouping function is not invoked. The new transport
must not be fed to it as stipulated COMPLETE native root topology.

## API grounding

The occurrence traversal and identity semantics follow the official
[OCCT iterator](https://occt3d.com/dev/doc/refman/html/class_topo_d_s___iterator.html),
[shape identity](https://occt3d.com/dev/doc/refman/html/class_topo_d_s___shape.html)
and [edge degeneracy](https://occt3d.com/dev/doc/refman/html/class_b_rep___tool.html)
references. PythonOCC exposes the shape copy constructor in its
[wrapper declaration](https://github.com/tpaviot/pythonocc-core/blob/master/src/SWIG_files/wrapper/TopoDS.i).
These references explain implementation choices; they do not verify the installed
native runtime or a source model.

The exact grid and index-trace checks are grounded in
[Open3D v0.19.0 source](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/geometry/PointCloud.cpp)
and its [point-cloud API](https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html).
The third trace return is the complete original-row membership; the eight-column
matrix is not substituted for it. Equality is checked at runtime rather than
assuming all installed versions share ordering or arithmetic.

## Acceptance frozen before first execution

Run only the named, reviewed synthetic tests with application imports, native
packages and filesystem geometry readers guarded out. Include the three prior
attribution/importer/loader regressions because their connected source changed.
Do not run a corpus, historical geometry suites, C15, EXP10 or EXP35.

Required positive controls: native shared edges within one owner, distinct owner
occurrences despite coincident geometry, original-row transport including rejected
and unassigned rows, pre-coplanar pieces, complete original finite candidates after
both early dedup and rolled merging, exact-source importer-to-loader-to-extractor
handoff, and unchanged legacy geometry. Required negatives: ambiguous/missing
native identity, malformed/late/over-budget trace, voxel group splitting, invalid
source/selected ledger bindings and attempted authority escalation. No failure is
rescued by changing limits, deleting a fixture or returning a successful prefix.

Root review before execution already corrected default native-cloud lifetime,
merged-plane invalidation, missing legacy detector metadata compatibility,
malformed selected-record comparisons and a voxel-partition verification gap.
These were pretest source-review repairs, not observed test results.

## Verification receipt

Root reviewed the new test files, their guarded harnesses and the source
definitions they execute. The three prior test files retain their recorded
identities. **The first and only focused run passed 345 checks in 4.00 s.**
External wall time was 4.11 s; maximum process RSS was 92,979,200 bytes (88.7 MiB).
No source or test changed after this passing execution. Exact command:

```sh
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_selected_cad_attribution.py tests/test_cad_import_provenance.py tests/test_cad_geometry_loader_provenance.py tests/test_cad_native_topology_capture.py tests/test_cad_support_lineage.py tests/test_cad_provenance_handoff.py tests/test_cad_provenance_end_to_end.py -q
```

The six connected checks execute exact importer/capture, loader, extraction
adapter, primary extractor, lineage stages, contractions, handoff and compact
report definitions. Four positive combinations use 12 invented raw vertices,
four native face doubles, two owner occurrences and two original candidates:

| Connected construction | Existing selected geometry | New retained evidence |
| --- | --- | --- |
| Axes offset 0 and 1; distinct voxel cells | Early dedup selects one specification | Both original finite candidates, all 12 original rows and distinct owners survive; preprocessing trace is exercised. |
| Axes offset 0 and 11; distinct voxel cells | Dedup retains two, rolled merge selects one envelope | The two separate original segments survive unchanged; the envelope is not substituted for them. |
| Same two offset cases, coincident face coordinates | Same legacy contractions | Native occurrences stay distinct; preprocessing is explicitly disabled in these two controls rather than inventing an impossible identity voxel trace. |
| Nonboolean native identity result or final trace row reuses row zero | Legacy geometry equals provenance-disabled baseline | Entire handoff is UNAVAILABLE; the failed trace leaves no earlier successful candidate prefix. |

The connected fixture's coplanar traversal is an explicit no-merge control.
Separate focused tests exercise an actual coplanar contraction, multiscale
contractions, rejected/unassigned rows, missing faces, seams/degenerate edge uses,
late invalidation and native temporary-object lifetime. Cap tests exercise
failure branches using small invented fixtures; they do not qualify worst-case
capacity. RANSAC/proposal geometry and native APIs are stipulated doubles, not
physical detection or native-library behavior. This is not full-worker replay,
native peak-memory headroom or inspection-latency evidence.

Executed source identities (SHA-256):

```text
backend/bend_inspection_pipeline.py
4054b5e9096aa95ca96c396970e331d45a01abf709e43eed057cd772f0982a69
backend/cad_import.py
b745c335421487753b85b4d663e3621c9e213b94254fbd2198d675d35e421a89
backend/feature_detection/bend_detector.py
4080c7f397ead65a77d81ec32d6f07ada6ec20cbaa7c8e2bbd8c2ecdf61f5dd2
backend/cad_provenance.py
313b9cbb4109c8f80c3f1f900a26c9793529aedbc6cd296ff4285b6728e13e10
tests/test_selected_cad_attribution.py
9b6f3ba7c28f202e7e27cbbb05d5cb138444497c8cc9c85c1d529df8f59a7052
tests/test_cad_import_provenance.py
685907dfbb9301ad31f6c09146dee497e6c47ce200713eff1d3639195ef657cf
tests/test_cad_geometry_loader_provenance.py
d6646a0b41c7c9c9fc8d112919d16c5b51de9ace7b8f8b10fb78459a51f06132
tests/test_cad_native_topology_capture.py
516fd16e56f586764831744c85fc9fbb91edcb371762a0c8088f81473b8485d9
tests/test_cad_support_lineage.py
e4fb05816fa88c7bb68d107bd36cf1563e90f63e44e4394702508577909e729d
tests/test_cad_provenance_handoff.py
02d68fbf855e300c122fb2f6199bb505b12f77ac8ca852d81e26e2d55d579454
tests/test_cad_provenance_end_to_end.py
88367c91b16c348265b53460dccc28d523f7d09e0ad25adbc48f88d0a45b2aea
```

The research grouping source retains SHA-256
`cb9fdd88de78b30f32985bd6ec06082db0cb19be5522b99a6e7424abacf6ae74`;
it was not changed, executed or given a runtime caller by this repair.
Final static verification passed: all 11 executed source/test identities remain
unchanged and parse as full Python modules; all 11 changed files have clean
trailing whitespace and final newlines; all six explicitly whitelisted local
checkpoint links resolve. The scoped `git diff --check` also passed. No protected
data or control artifacts were inspected by these checks.

## Next decision

Decision: green to retain this additive source-level closure of provenance LOSS
within its declared budgets. It is NOT green for verified native-root grouping
or physical/floor use. The original reuse and distance regressions still pass.

Keep engineering closure separate from native and physical evidence. The next
Priority 3 work is explicit local root-role
nomination, verified native execution and finite scan-supported mapping with
registration uncertainty. Persistent source/frame/unit binding and manufacturing
interpretation remain prerequisites for qualified expected-event identity.
M2 capture/reference/calibration and M4 durable recovery/trusted release remain
parallel priorities. No research/floor promotion follows from this checkpoint.
All physical gates, C15's retained failures, terminal EXP10/EXP35 and the paused
long-running goal remain unchanged.
