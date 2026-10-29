# Gold11 bounded input/frame review — protocol V1

Date: 2026-09-09. Authority: the user's “proceed” directly answers the request
to open only already-consumed Gold11 CAD/scans and unchanged reference geometry
for a bounded datum/frame compatibility review. This is **not a benchmark run**,
an old terminal-control retry, access to new email parts, or production authority.

## Frozen question and allowed access

Determine actual byte/schema compatibility with the frozen R0 input contract,
source unit declarations, native CAD frame/planar-surface inventory, and unchanged
reference representation/raw-scan plausibility. Do not fit any registration,
select datums using Gold, generate bend predictions, score F1, tune a threshold,
change references, or select a favorable subset. All eleven cases remain reported.

The fixed case IDs are CG08506AE97D, CG191DEC7F81, CG1E0BBA2933,
CG4BDD1371B6, CG4FAF273093, CG64C412CC73, CG65CD575B1C,
CG84C8FA12D2, CGE4B0549E41, CGEBA9A869AC, CGF41F49066A.
The six CAD families are 10839000, 37361005, 38172000, 40500000,
49001000 and 49024000. Paths and SHA-256 declarations are taken exactly from
the previously released `cad_guided_gold13_physical_20260822.json` slice and
copied into the fixed review script before any raw input access. Excluded slice
entries CG60B2F9102E and CGC60DACC70E are not admitted.

One named reference container is admitted solely to select these eleven records:
`data/bend_corpus/annotations/bend_line_gold_catalog_v1.json`. Selection is by
the frozen scan hashes and reference IDs, before geometry inspection. Unselected
records are neither emitted nor geometrically processed; no referenced paths are
followed. Whole-container reading/hashing for extraction and identity does not
make its other records available for research. Selected references must join
uniquely, retain 71 total lines, and retain their original geometry. A current
container/selected-record hash is a new snapshot binding, not proof of equality
with an unavailable historical byte-level reference snapshot.

No enumeration/discovery, original label-source followups, application startup,
database/model imports, uploads, email, held-outs or existing attempt artifacts.
Only the exact 17 raw paths (11 PLY/six STEP) can be opened. CAD snapshots are
copied into an owned temporary directory only after their declared hash matches.
No input or reference file is written. R0 source/config/bundle stays unchanged.

## Bounds and execution rule

The review has separate, predeclared diagnostic limits: 64 MiB/PLY,
8 MiB/STEP, 8 MiB/reference container, 64 KiB/PLY header, 32 scalar vertex
properties, one million raw vertices, two million standard mesh faces,
16 vertices per face and eight million face indices in total. This does not
raise R0's 200,000-row/vertex-only cap. Header/count compatibility is reported
separately; R0's actual loader is not executed by the reviewer.
Unsupported geometry retains its case disposition, never silently strips faces,
invalid rows, or changes row order. Duplicate CAD is read once by frozen identity.

Native inventory: at most 32 transfer roots, 8,192 topology nodes/edge/vertex
occurrences, depth 32 and 2,048 faces; no tessellation, feature extraction or
repair. Native failures/overflows remain unavailable, not zero geometry.

One sequential worker: 300 s wall, 240 s CPU, 2 GiB sampled process-group RSS,
1 MiB combined output. BLAS/OMP thread declarations are one; native kernel
thread behavior is not independently attested. The parent monitors and kills
the process group on budget/monitor failure. Sanitized environment, no site
startup/.pth, no network/subprocesses from the worker. Python audit guards are
defense in depth, not an OS sandbox or interception of every C++ read.

Source-only parser tests and a separately invented native fixture must pass;
an independent source/protocol audit must greenlight the exact runner before
physical access. The parent binds the three review scripts and this protocol
before/after execution, and creates a fixed exclusive attempt directory before
raw access. Once that attempt starts it is terminal even on failure: no retry,
cap increase or subset replacement under this protocol. Raw files are bounded
regular-file snapshots with hash and before/after descriptor consistency checks.
Source drift or source/reference hash mismatch prevents a clean review result.

## Fixed descriptive calculations and interpretation

1. Validate PLY encoding/schema, all declared vertex/face rows and byte length;
   preserve all raw vertex rows. Report count, AABB, centroid, coordinate extrema,
   and principal RMS lengths. These are descriptive, not chosen inference input.
2. Read STEP source length-unit names and record kernel conversion separately.
   Explicitly request system length unit 1.0 for this review after loading the
   STEP model. Native units are not scan export units or physical calibration.
   Record root/shape/face-type inventory and analytic bounding box; planar
   normals/centroids/areas are potential datum geometry, not actual stable datums.
3. For each selected reference preserve the historical endpoint precedence:
   `span_endpoints`, then `line_start/line_end`, then anchor/axis/span. Check all
   coexisting representations for agreement, rejecting contradictions rather
   than choosing a visually better line. Record explicit frame/unit/carrier
   fields at both case and line level if present (at most 1,024 JSON characters
   per explicit field; oversized fields get an explicit unavailable marker).
   Report same-key case/line differences without assuming aliases or silently
   resolving them. Do not infer a manufacturing convention from proximity.
4. Sample 201 equally spaced positions on each unchanged reference line; query
   nearest distance to the **complete** valid raw PLY vertex cloud with one KD
   tree worker. Report per-line min/median/p95/max distances, endpoints and span.
   No fit, threshold pass/fail, predicted bends or F1. These distances are in
   raw coordinate units unless scan unit evidence supports more. Near-surface
   agreement is a frame sanity observation, not proof of correct carrier/event.
5. Report CAD and scan AABBs side by side, plus the scan's vertex centroid;
   this is not a whole-solid CAD centroid comparison. Do not fit rotation,
   reflection, scale or translation. Size/overlap alone cannot establish
   registration. No Gold point ever participates in pose or datum selection.

Expected output: byte/schema admission matrix, explicit source/kernel/scan unit
status, native geometry inventory, reference-integrity/plausibility record,
unresolved datum/frame/carrier facts, resources, and a concrete next engineering
decision. No physical accuracy, count/F1 improvement, manufacturing signoff or
floor-readiness claim is possible from this review alone.

The unit API interpretation follows the official [OCCT STEP reader reference](https://occt3d.com/dev/doc/refman/html/class_s_t_e_p_control___reader.html)
and was checked against the installed OCP method documentation. It distinguishes
file-unit reporting from the transfer's system length unit.
