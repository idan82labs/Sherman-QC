# CAD attribution repair and tessellation provenance handoff

Date: 2026-09-07. User-authorized continuation after the connected-module audit,
under [locked Priority 3](2026-09-07_production_generalization_roadmap.md#locked-execution-baseline--user-confirmed-2026-09-07).
Source and invented in-memory geometry only; no native package, physical input,
held-out, email, control, one-shot artifact, service or production job execution.

Follow-on: the [2026-09-08 native incidence and original-support repair](2026-09-08_cad_native_support_provenance_checkpoint.md)
records the user-authorized closure of the remaining source provenance loss.
The original receipt below is retained unchanged as historical evidence; its
native/root/physical limitations are not retrospectively waived.

## Audit disposition and scope

The [previous grouping checkpoint](2026-09-07_cad_root_grouping_checkpoint.md)
remains valid for its stipulated-topology contract. Its 3/12 to 12/12 partition
comparison is not a runtime detector benchmark. The audit nevertheless blocked
integration because the connected runtime could reuse extracted surface IDs,
accept arbitrarily distant attributions and discard native provenance upstream.

The audit executed only three exact AST-extracted pipeline helper bodies against
invented objects. Two selected candidates reused one extracted bend, both copying
surface IDs (7,8), while reporting matched_count=2 and leaving the caller's used
set empty. A separate selected candidate 1,000,000 mm away was accepted. No native
pipeline was imported or executed. These were existing connected-path defects,
not regressions from the unused research grouping module.

This bounded repair changes existing runtime source attribution and adds an
optional traceability handoff. It does NOT promote the research grouping engine,
run a physical evaluation, restore native ownership, or complete Priority 3.

## Attribution mechanism and compatibility changes

The [pipeline](../../backend/bend_inspection_pipeline.py) now builds the complete
selected-to-extracted admissibility graph and accepts only isolated edges: both
the selected row and extracted column must have degree one. Many-to-one and
one-to-many components remain unattributed; no best-score ranking or iterative
removal rescues another edge. Independent isolated edges can still be retained.
Duplicate extracted bend identities invalidate their attributions even when the
duplicate is geometrically distant. The retained single-spec helper preserves
the caller's used set and checks original ambiguity before consumed indices.

The fixed, pretest engineering envelope requires:

- Center separation and unoriented finite endpoint-pair bottleneck below 45 mm.
- Bend-angle difference below 35 degrees, with inputs within 0 to 180 degrees.
- Unoriented axis and corresponding normal dot products above 0.97.
- Explicit finite, nondegenerate endpoint/direction/normal records, extracted
  center consistent with its finite endpoints, and each axis compatible with
  the intersection direction of its flange planes.
- Exactly one direct/swapped normal correspondence; only its reordered local
  mesh-surface IDs may transfer. Surface IDs must be distinct nonnegative integers.

The 45/35 upper limits reuse the nearby runtime assignment envelope; 0.97 also
appears in the existing CAD direction screening. Their use here is a NEW safety
policy, not calibrated registration uncertainty, validated metrology or verified
correspondence. Separate roundoff margins reject threshold equality and
unsupported numerical ranges. Pair work is capped at 256 candidates per side,
at most 65,536 checks. Over-cap attribution is wholly unavailable; this cap does
not bound native parsing or the selected output list's size.

Payloads start from copied selected geometry, never an extracted `to_dict()`.
Only surface IDs and explicit attribution diagnostics are added. Extraction
confidence, success, observed-state and native-identity claims cannot leak into
the selected payload. Unattributed records retain distinct negative surface IDs,
which do not enable mesh-surface classification. The attribution itself says
UNKNOWN observed state and unverified native topology.

Compatibility impact: malformed selected geometry or missing flange normals now
raises `ValueError` before any batch attribution. The prior helper invented
default normals or could pass nonfinite geometry into measurement. No result
prefix is returned on invalid selected input. A missing or malformed EXTRACTED
alternative instead leaves a valid selected record unattributed. Ambiguity can
reduce the number of mesh-guided measurements; no physical recall improvement
is claimed. Other existing measurement fallbacks remain in place.

Fundamental counterexample: if the only supplied alternative is a wrong,
disconnected but identically oriented bend translated 10 mm, it can pass these
guards. Uniqueness is conditional on supplied alternatives, not native root
identity. Native ownership and finite scan evidence must still be established
separately; the repair does not solve that problem by choosing tighter thresholds.

## What provenance is now preserved

The [importer](../../backend/cad_import.py) adds an optional fourth `CADMesh` field,
`tessellation_provenance`, without changing existing three-argument construction
or `CADImportResult.to_dict()`. The OCC source path records one import-local face
occurrence per enumerated face, with contiguous half-open vertex and triangle
ranges. Missing and empty triangulations remain explicit records. This gives
triangle-to-face-occurrence bookkeeping without a triangle-sized side array.

The existing vertex coordinates, triangle indices, normals, face order and
location application are unchanged. No coordinate welding or native identity
inference occurs. A 4,096-face record cap or malformed native count/local triangle
index clears the ENTIRE mapping, not a successful prefix; original geometry
construction and exception behavior remain unchanged. Thus an unavailable
sidecar does not itself repair an invalid legacy mesh. Gmsh explicitly reports
mapping UNAVAILABLE because its current extraction discards native face entities.

`load_cad_geometry(..., include_provenance=True)` returns the existing three values
plus a copied sidecar. Default callers still receive three values. The optional
handoff checks schema/counts, contiguous complete range bookkeeping and record
caps; it drops extra authority flags, fixes V1 scopes and leaves all ownership,
edge, source-byte, frame and unit verification UNKNOWN or false. Missing or
inconsistent sidecars, direct meshes and point-cloud fallbacks are unavailable.
These are bookkeeping checks, not verification of CAD source bytes or geometry.

Import-local face ordinals are NOT persistent B-rep IDs or assembly occurrence
identities. No owner/edge incidence, oriented topology, physical root nomination,
manufacturing convention, trusted frame or verified unit basis is supplied to
the research grouping engine. Existing default runtime callers still do not
consume this optional sidecar. Preprocessing and geometric deduplication continue
to lose support lineage, so a hook only before rolled merging remains too late.

## Fixed verification target before first execution

Run only the three new, fully source-reviewed test files. Their harnesses compile
explicit pipeline/importer AST definitions, with invented objects and guarded
native-library doubles. They do not import the application or read a CAD file.
No historical suite is represented as rerun.

Acceptance: the original reuse and million-mm counterexamples fail safely;
independent unique pairs and swapped normal ordering remain usable; ambiguity,
duplicate identities, malformed numbers and budget overflow cannot authorize
surface attribution; selected geometry is preserved and copied. Importer tests
must retain exact invented geometry and preserve source-face ranges including
missing faces; malformed indices or cap overflow clear the mapping. Loader tests
must preserve default tuple compatibility and opt-in source-to-source handoff,
while preventing authority elevation. Tests and limits are not adjusted to rescue
an observed failure. The first result and any subsequent repair are recorded below.

## Verification receipt and next decision

Root fully reviewed the three test files, their source-extraction harnesses and
every executed source definition. Independent review covered the compatibility
graph, finite arithmetic, surface ordering, importer bookkeeping and loader
authority boundary. **The first and only focused run passed 172 checks in 1.67 s.**
External wall time was 1.75 s and maximum process RSS was 74,350,592 bytes (70.9 MiB).
No source or test changed after that passing execution. The exact command:

```sh
/usr/bin/time -l env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider tests/test_selected_cad_attribution.py tests/test_cad_import_provenance.py tests/test_cad_geometry_loader_provenance.py -q
```

| Previously unsafe behavior / fixed control | Verified current result |
| --- | --- |
| Initially empty used set lost; one source reused | Caller set becomes `{0}`; second use returns no match. |
| Two selected candidates claim one source | Both remain unattributed; no greedy first claim survives. |
| Only source is 1,000,000 mm away | No attribution, no consumed index, no copied source IDs. |
| Two disjoint uniquely admissible pairs | Both retained across input-order permutations. |
| One conflicted graph plus independent singleton | Conflict remains unresolved; independent pair retained. |
| Flange order reversed in extraction | Local surface IDs swap; selected normals and all finite geometry stay unchanged. |
| Duplicate extracted identity at distant locations | Duplicate rows unresolved despite locally unique geometry. |
| Invalid selected geometry late in batch | Raises before any pair work or returned measurement payload. |
| Valid OCC-source bookkeeping through exact loader source | Six invented vertices, two triangles and three face records remain; triangle origins are `[0,2]`, preserving the missing middle face. |
| Missing/malformed/over-cap mapping | All face records cleared; no partial provenance claim. |
| Caller supplies verified/native flags or persistent-ID scopes | Extra authority fields dropped; canonical authority remains unknown and import-local. |

The positive importer fixture also verifies three triangles map to source face
occurrences `[0,0,3]` across four enumerated faces, including absent and empty
triangulations. Coincident face coordinates remain separate vertex blocks. Mocked
node-index errors that cross into another face clear the new mapping without
silently changing legacy geometry. Existing globally out-of-range exceptions
remain exceptions. Gmsh's mocked missing-node behavior remains explicitly without
native mapping. No actual native-kernel behavior is established by these doubles.

These checks close the demonstrated source attribution defects and the first
triangle-to-face-occurrence transport seam. They do not measure physical accuracy,
native CAD compatibility, full-worker behavior, peak native memory, registration
or complete inspection latency. Older geometry suites were not rerun. The prior
grouping, dense-radius, window-ingestion and primary detector sources retain their
recorded hashes; the research grouping function still has no runtime caller.

Executed source identities:

```text
backend/bend_inspection_pipeline.py
e182ff63552fb73a55b851d7fe7d8e8382897456619cdda4f08833139eaea11a
backend/cad_import.py
a9374cdc9204a61a1c867fb4ace575a83df0fe01168de875b97cf00b9b4dcdc6
tests/test_selected_cad_attribution.py
9b6f3ba7c28f202e7e27cbbb05d5cb138444497c8cc9c85c1d529df8f59a7052
tests/test_cad_import_provenance.py
685907dfbb9301ad31f6c09146dee497e6c47ce200713eff1d3639195ef657cf
tests/test_cad_geometry_loader_provenance.py
d6646a0b41c7c9c9fc8d112919d16c5b51de9ace7b8f8b10fb78459a51f06132
```

Final static verification passed: all five executed Python files parse as full
modules and retain the SHA-256 identities above; all eight touched files have
clean trailing whitespace and final newlines; the five explicitly whitelisted
local link targets exist. The scoped `git diff --check` also passed. No protected
data or historical control artifacts were inspected by these checks.

Decision: green for retaining this bounded safety repair and continuing source-only
Priority 3 work; NOT green for verified native-root grouping or physical/floor use.
Next preserve owner occurrences/native edge identity and original support lineage
BEFORE preprocessing/deduplication, nominate root-role support explicitly, then
bind finite observed evidence and registration uncertainty without reusing scan
observations. Do not pass generic flange/context faces or these import-local
ordinals as complete native root topology. Do not use ordinal grouping IDs as
persistent event identities.

Roadmap position: M0's bounded checkpoint is delivered; M1 native-runtime evidence
is open; M2 capture/reference/calibration and M4 durable recovery/trusted release
remain incomplete; M3 has another source-level improvement but its physical and
grouped generalization exit is open. M5/M6 have not begun. Parallel M2/M4 work
remains required. C15's retained failures remain unwaived, terminal EXP10/EXP35 and
all physical/promotion gates are unchanged, and the long-running goal stays paused.
