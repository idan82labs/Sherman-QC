# Consumed Gold11 input review: concrete rerun blockers

Date: 2026-09-09. The user approved the exact bounded raw-input/reference review.
It completed once under the independently reviewed [V1 protocol](2026-09-09_gold11_input_review_protocol.md).
**The files are intact, but R0 is not compatible with these inputs. No engine,
registration or benchmark was executed; no improvement over historical F1 .458
has yet been demonstrated.** New email parts and held-outs were not accessed.

## What is now established

- All eleven scan files and six distinct CAD files match their previously
  declared SHA-256 values. Every raw PLY body and mesh-face index list validated
  under the separate diagnostic limits; no rows were removed or reordered.
- All eleven selected reference records joined by scan hash, reference ID,
  scan path and family. All 71 lines are finite, nonzero direct start/end
  segments. No conflicting coexisting endpoint representations were present.
  There are no per-line frame/unit/carrier declarations to contradict their
  case-level metadata. Original inputs, labels and authority templates remain
  untouched. The reference container hash is unchanged during this review.
- All six STEP files imported as one top-level solid for inventory, with one
  root transferred. All report source length `MILLIMETRE`; the review explicitly
  set and queried transfer system length unit 1.0. This is file/reader evidence,
  not scan calibration or full native-validity/feature qualification.
- The complete review took 36.917479 s parent wall, 36.708488 s worker CPU,
  594,755,584 B peak worker/sampled group RSS, and 464,290 B output. Exit zero,
  no budget stop, all four review source/protocol hashes unchanged.

Machine-readable [compact receipt](2026-09-09_gold11_input_review_summary.json),
full [review output](gold11_input_review_v1_attempt/OUTPUT.txt), and
[parent completion receipt](gold11_input_review_v1_attempt/COMPLETED.json).

The reference-container snapshot is
`4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36`;
selected-record snapshot is
`d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc`.
These newly observed hashes do **not** establish equality with an unavailable
historical byte-level Gold snapshot. The 71-line denominator alone cannot prove
historical geometry equality either.

## Actual incompatibilities, not speculative prerequisites

### 1. All eleven PLYs contain mesh faces

Every input is binary little-endian PLY with XYZ/normals and a nonempty triangle
face list, exported by Geomagic. R0's reader accepts only vertex-only PLYs;
therefore **0/11 match its declared element schema**. Ten also exceed its
200,000-row limit. The smallest input passes the row limit but still has faces.
This is a source-contract comparison with inspected headers, not an executed
R0 rejection result.

| Case | Raw vertices | Mesh faces | Gold lines |
| --- | ---: | ---: | ---: |
| CG08506AE97D | 817,867 | 1,623,601 | 6 |
| CG191DEC7F81 | 601,144 | 1,192,006 | 5 |
| CG1E0BBA2933 | 738,364 | 1,470,056 | 10 |
| CG4BDD1371B6 | 581,263 | 1,158,804 | 4 |
| CG4FAF273093 | 46,528 | 92,990 | 3 |
| CG64C412CC73 | 357,851 | 712,686 | 7 |
| CG65CD575B1C | 715,498 | 1,429,620 | 6 |
| CG84C8FA12D2 | 863,115 | 1,701,535 | 8 |
| CGE4B0549E41 | 837,407 | 1,666,083 | 4 |
| CGEBA9A869AC | 618,228 | 1,233,721 | 8 |
| CGF41F49066A | 751,464 | 1,494,766 | 10 |

The diagnostic parser validates mesh lists but deliberately uses only unchanged
vertices for this review. It has **not** been connected/promoted to engine input.
The existing row-lineage planner remains unconnected; its rank sampling does
not guarantee preservation of small bends or spatial coverage.

### 2. Native CAD scope excludes three families before bend fitting

| CAD family | Faces | Topology occurrences | Cases using it | Above R0 face cap 128 |
| --- | ---: | ---: | ---: | --- |
| 10839000 | 164 | 2,702 | 3 | Yes |
| 49001000 | 114 | 2,026 | 1 | No |
| 38172000 | 164 | 2,740 | 4 | Yes |
| 37361005 | 88 | 1,592 | 1 | No |
| 49024000 | 74 | 1,396 | 1 | No |
| 40500000 | 370 | 6,964 | 1 | Yes |

Thus eight cases exceed the current face cap. Family 40500000 also exceeds
R0's 4,096 topology-node bound. These are actual imported occurrence counts;
native validity, inner/outer classification and supported bend extraction were
not run. Increasing a cap alone would not qualify real trim geometry: the
current extractor additionally requires rectangular inner cylinders, 20–160
degree sweep and two rectangular adjacent tangent flanges. Hole walls,
compound trims and unsupported candidates must not silently become bend events
or be selectively discarded to improve a score.

### 3. Units, pose and datum stability remain different questions

The only PLY comment is `Exported by Geomagic`; no unit declaration or datum/frame
metadata appears in its header. STEP length units are now documented, but scan
millimetres are not established by file metadata. FileUnits also reported the
odd solid-angle name `metre` for these six files; that literal result is retained,
not reinterpreted as a coherent full unit graph. Length transfer state is the
specific checked field. The official [OCCT reader documentation](https://occt3d.com/dev/doc/refman/html/class_s_t_e_p_control___reader.html)
distinguishes source-unit names from system transfer units.

Ten of eleven CAD/scan AABBs are numerically disjoint under identity coordinates;
the remaining overlap is not registration evidence. No transform, scale,
reflection or reference-driven alignment was fitted. This is an identity-frame
sanity observation with unverified scan units, not calibrated physical distance.

All CADs contain several plane-normal directions (smallest normalized normal-Gram
singular value .204545–.298150). That establishes available CAD normal diversity,
**not** six-DOF scan correspondence or stable manufacturing datums. Hole walls and
deformable flanges can contribute normals. No stable scan-row/CAD datum pairing
is supplied by the reviewed records. The `DATUM_CONTRACT_ABSENT` disposition
means absent in the admitted records, not a search proving none exists anywhere.

### 4. Gold is structurally readable, but carrier intent is still unresolved

Nine cases declare `reference_mesh_raw`; the two consumed X/V scans declare
`scan_raw`. No selected record supplies physical unit or carrier convention.
All 71 direct segments were preserved and compared descriptively to their full
raw vertex clouds using 201 fixed samples per line. No threshold, match score,
registration or fit was applied.

Nearest-vertex distances are not distances to a continuous surface and not
metrology uncertainty. Per-line p95 values range .185889–13.257485 raw units;
maximum sampled distance is 13.525759 raw units. CG65CD575B1C and CGEBA9A869AC
contain the largest discrepancies. This does not establish which references,
carriers or coverage assumptions are wrong, and does not justify dropping or
relocating lines. The unchanged finite Gold convention cannot automatically be
replaced by the new measured cylinder-center axis convention.

## Verification and independent gate

Final source/synthetic checks: **110 parser/reference tests +16 review controls
=126 distinct passing tests**, plus an invented STEP/PLY native inventory check.
The controls and invented check were rerun after final boundary fixes; earlier
12/15-control and invented passes are not counted as independent evidence.
Final controls .17 s pytest/.262696 s parent, peak73,613,312 B. Final invented
native check .424702 s parent/.357916 s CPU, peak315,441,152 B, no budget stop.

Before any raw access, independent AI source review found and closed missing
per-line declarations and redundant unbounded declaration aliases. Root also
repaired macOS temp-path alias handling and partial STEP-transfer admission
before the first invented run. The final physical-review GREEN was scoped to
the parent command and runner hash
`41a79da155cf9955bf231fd3374baa5a9e4c79c1a0e6002af6f6a1dab4592955`.
It is an engineering audit, not a human manufacturing signature. The frozen
protocol and helper hashes are retained in the attempt receipt.

Independent post-run review of only the two new receipts confirmed the identity,
schema/cap, unit and reference counts and their stated limits. Final checks:
six named Python files parse, four local receipt links resolve, scoped whitespace
checks pass, and all four frozen review source/protocol hashes still match.
Full output SHA-256:
`0d1c5abcc534bd6d377296575941b0cab0dcf7bf8dec8b4bdd3f6c036a709c3f`.

## Next engineering decision and stop boundary

The next useful work is **a successor input/native-geometry adapter**, not a
rerun of unchanged R0:

1. Connect validated mesh-PLY vertex extraction to byte-bound raw-row lineage
   with an explicit, tested sampling/coverage policy. Preserve all datum rows
   and separate inspection rows; malformed mesh data must remain unavailable.
2. Define a bounded topology policy and feature handling for realistic CADs,
   with independent multibend/holes/trim tests. Preserve R0 exactly; a larger
   candidate budget is a new development configuration, not an R0 retry.
3. Connect an explicit scan-unit and label-independent registration/datum mode,
   then an audited finite-carrier bridge. Distinguish provisional whole-part
   best-fit comparisons from stable-datum metrology; neither Gold geometry nor
   a desired score may choose pose, datum rows or carrier convention.
4. Freeze that candidate, input mapping, unchanged reference snapshot, scorer,
   all-case failure handling and budgets before requesting the actual consumed
   development benchmark. Do not wait for final accuracy targets or universal
   geometry support before a defensible all-case diagnostic.

This review attempt is **complete and terminal**. No repeat, cap increase,
reference changes, new raw accesses or benchmark are authorized by its command.
Source/synthetic engineering may continue under the existing roadmap; the next
physical diagnostic needs its separately scoped execution gate. The historical
score stays .4580152672, R0's physical score remains null, the paused goal stays
paused, and zero-shot/UI/floor work remain deferred. This is positive proof of
input-readiness diagnosis, not positive proof of engine accuracy improvement.
