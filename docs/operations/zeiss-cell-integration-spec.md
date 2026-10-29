# ZEISS scan cell: integration specification for the Sherman QC line

Status: **draft for the cell integrator**, 2026-10-17. Each question in section 3 shows the
default the Sherman QC watcher assumes today and the fallback if the integrator cannot meet it.
Nothing below has been confirmed on the real cell yet. The answers must be recorded in section 8
before the first production scan.

Counterparts: the hotfolder (`apps/production_line/ingest/hotfolder.py`), the probe
(`ingest/ply_probe.py`), the filename parser (`domains/production/recipe/sku.py`) and the engine
ingest (`domains/production/engine/inspect/scan_io.py`, ENGINE_IO §4).

## 1. Which cell

**ZEISS sells no product called "RoboScan 7000".** The request named it, but no such model
exists. The closest current ZEISS automated optical cells are:

- **ZEISS ATOS ScanBox Series 5**: a robot cell for small and medium parts with many upgrade
  options.
- **ZEISS ATOS ScanBox Series 7**: an 8-axis kinematic cell for large parts such as body side
  panels.
- **ZEISS ScanCobot**: a collaborative-robot station for parts up to about Ø 500 mm and 50 kg,
  with an ATOS Q sensor.

All of them are programmed and run from **ZEISS INSPECT**. Measurement sequences run in the
Virtual Measuring Room (VMR) and are started by operators from the **Kiosk Interface**. INSPECT
can save meshes as STL, G3D, JT Open, ASCII or PLY. STL export needs a polygonized mesh. INSPECT
also has a Python add-on API, with which projects can be scripted to export meshes
(`gom.script.sys.export_stl` and the corresponding PLY export; macros recorded in the add-on
editor). Sources: zeiss.com ATOS ScanBox and ScanCobot product pages; ZEISS INSPECT "Mesh editing"
page; ZEISS Quality Forum threads 3727, 24203, 1177.

**Integrator action Q0:** confirm the exact cell model, sensor (e.g. ATOS Q 12M, ATOS 5) and
ZEISS INSPECT version and licence (Kiosk Interface, VMR, add-on/Python scripting).

## 2. The contract in one paragraph

After each measurement, ZEISS INSPECT exports one **binary PLY in millimetres**:

- **raw / fixture frame** (no best-fit to CAD needed);
- polygonized mesh or point cloud, **not thinned** below about 0.5 mm point spacing;
- written to the station's export share under a **temporary name**, then **renamed** to
  `<SKU>_REV<revision>_SN<serial>_<yyyymmddThhmmss>.ply`.

The share is mounted on the line PC as that station's inbox. The line takes it from there:
- it waits for a stable size;
- it parses the SKU, revision and serial from the name, for the operator only (the engine never
  sees the name);
- it archives the file, runs the inspection and moves the file out of the share.

## 3. Questions, defaults and fallbacks

| # | Question | Default assumed by the watcher | Fallback if not answered or not possible |
|---|---|---|---|
| Q1 | ZEISS INSPECT version, and whether the Kiosk Interface / VMR can run a post-measurement export step | INSPECT 2023 or later, with a scripted (add-on) export step at the end of each kiosk sequence | The operator exports manually from INSPECT, using the same name template, to the same share. The line behaves identically; only throughput changes. |
| Q2 | File format | **Binary little-endian PLY** (`binary_little_endian 1.0`), `vertex` with float `x y z`; extra vertex properties (normals, colour, quality) are ignored | **Binary STL** is accepted (triangle soup; duplicate vertices merged). ASCII PLY/STL is accepted but slower and 3-5× larger. G3D, JT, ASCII point lists, OBJ and ZIP are **not** accepted (UNSUPPORTED_FORMAT). |
| Q3 | Units | **Millimetres**. The engine applies no auto-scale. | An inch or metre export is not rescaled silently: the engine reports UNIT_MISMATCH_SUSPECTED (INVALID_INSPECTION). The fixed factors 25.4 / 1000 exist in the engine CLI but are not wired per station today. Export mm. |
| Q4 | Mesh or points | Either. A polygonized mesh is preferred (its vertices are the measured points); faces are counted, not used. | A point-cloud PLY with no faces is fine. |
| Q5 | Raw or pre-aligned frame | **Raw scanner / fixture frame.** The engine registers on the recipe's datum panel by itself. It does not depend on the export pose and never trusts a ZEISS alignment. | A pre-aligned (best-fit to CAD) export is accepted too; the engine re-registers anyway. Do **not** use a local best-fit that warps or crops the mesh. Do not apply a "cut to CAD" that removes material outside the nominal: that hides real deviations and missing flanges. |
| Q6 | Polygonization and thinning | Polygonize with **no curvature-based thinning on flat areas**: about 0.3-0.6 mm point spacing everywhere, 0.5 mm target. Keep hole and edge detail ("preserve edges"). Do not smooth beyond INSPECT's default noise reduction. | If thinning is required for file size, keep max point distance ≤ 0.8 mm on flanges. Sparser exports lead to UNKNOWN(LOW_DENSITY / EDGE_NOT_SEEN), never to a wrong value. Keep exports ≤ about 2 M points: the engine budget is 45 s P95 and 180 s hard at about 1 M points, 4 GiB RSS. |
| Q7 | Export share (SavePath) and layout | One SMB share on the cell PC (or a NAS), with **one directory per station**: `\\<cell-pc>\ScanExport\<station-id>\`, e.g. `\\zeiss-pc\ScanExport\cell-1\`. Only finished exports and their optional sidecars go there. The line account gets read, write and delete (accepted files are **moved** out). | Read-only share: set the station's `source_policy` to `copy_leave`. The line then copies and remembers each file (path, size, mtime, sha), and ZEISS-side housekeeping must delete old files. Subdirectories per day: set `"recursive": true`. |
| Q8 | Filename template | `<SKU>_REV<revision>_SN<serial>_<yyyymmddThhmmss>.ply`, e.g. `23553000_REVB_SN0042_20261017T101500.ply`. The SKU is the catalogue part number; the revision equals the drawing revision in the recipe; the serial is the part or traveller id. | The minimal survey template `<SKU>_<serial>_<yyyymmddThhmmss>.ply` also parses (revision NOT_IN_NAME, so no revision check). The existing manual names (`23553000_B_FULL_SCAN 1.ply`, `38172000_01_SCAN_2 1.ply`, `BEFORE BENDING  11979003_C.ply`, ` (1)` copies) still parse. A site regex `SHERMAN_LINE_FILENAME_REGEX` with a named group `(?P<sku>...)` can replace the parser. An unknown or ambiguous SKU goes to **quarantine** (UNMATCHED_SKU / AMBIGUOUS_SKU), and so does a different revision (REVISION_MISMATCH). Nothing is guessed. |
| Q9 | Completion signal | **Write to a temporary name, then rename** to the final `.ply`. Temporary names that are always ignored: `*.tmp`, `*.part`, `*.partial`, `*.crdownload`, `*~` and any dotfile. The watcher additionally requires the size and mtime to be unchanged for 3 polls (1 s apart), at least 2 s since the last write, and at least 1 KiB. | **`.done` sidecar**: write `<file>.ply`, then an empty `<file>.ply.done`, and set the station's `"sidecar_mode": "required"`. The sidecar may carry JSON `{"sha256": "...", "size_bytes": N, "declared_stage": K}`; a sha or size that disagrees quarantines the scan (SIDECAR_INVALID). With neither: size stability alone, and a writer pausing for more than 60 s mid-file risks PLY_TRUNCATED. |
| Q10 | Kiosk add-on for automatic export | The kiosk sequence ends with an INSPECT add-on, a Python script on the ZEISS PC, that does the following: polygonizes with the Q6 settings; exports binary PLY in mm in the Q5 frame to `<share>\<station>\.<name>.ply.tmp`; renames it to `<name>.ply`; with `sidecar_mode=required`, writes `<name>.ply.done`. The SKU, revision and serial come from the kiosk's part selection or barcode fields. | Manual export by the operator from INSPECT (Q1 fallback). The name template and share are the same, so the line needs no change when the add-on is added later. |
| Q11 | Incomplete parts (scans mid-sequence) | The same template. The stage is inferred by the engine from the geometry. Optional: the kiosk asks "bends done so far" and writes it as `declared_stage` in the `.done` JSON. | Without a declaration the engine decides the stage itself. A declaration that disagrees with the geometry gives STAGE_MISMATCH (REVIEW) and is never trusted silently. |
| Q12 | One file per part and setup merge | One merged export per physical part per measurement (all robot views merged and aligned by INSPECT into one mesh or point cloud). | Separate per-view files are **not** supported. Merge in INSPECT. |
| Q13 | Fixture and table in the scan | The fixture and the table are cut away in INSPECT (selection or "cut to bounding volume", **not** "cut to CAD") before export. | A fixture plane left in the export can make the part look like the WRONG_PART (WRONG_PART_SUSPECTED) or block its registration. Clamps hiding a datum edge lower confidence (see the E3 notes). |

## 4. Network and access

- The line PC mounts the share with SMB 3. On Linux: `cifs`, `vers=3.0`, `actimeo=1`, with a
  dedicated account (runbook section 5). On Windows: a UNC path, with the service account granted
  on the share.
- The cell PC never connects to the line PC, and the line PC never writes anything but deletions
  into the share. The line's API listens on 127.0.0.1:9000 only.
- Clocks: the timestamp in the name is informational. The line records its own `received_at`
  (UTC). Keep both PCs on NTP anyway, because operators compare them.

## 5. What the line does with a file

1. Detect the file and wait until it is ready (Q9).
2. Probe the bytes:
   - format, header, completeness (a PLY body shorter than its header declares is truncated),
     and vertex and face counts;
   - empty, unsupported or invalid files are quarantined after a 60 s grace.
3. Compute the sha256 and register the scan (SCAN_RECEIVED). Identical bytes dropped again give
   SCAN_DUPLICATE and no second job.
4. Resolve the name to the SKU's active released recipe (Q8). Unmatched names are quarantined
   and can be assigned by an operator.
5. Move the file to `archive/scans/YYYY/MM/<sha256>.ply`. The engine sees only an opaque
   `input/scan.ply`.
6. Run the inspection (≤ 180 s), render the viewer assets and the bilingual PDF, and update the
   station screen within seconds.

## 6. Density and size guidance (from the engine budget)

| Part size | Suggested spacing | Typical points | Notes |
|---|---|---|---|
| ≤ 300 mm | 0.3-0.5 mm | 0.2-1 M | default |
| 300-1000 mm | 0.5 mm | 0.5-2 M | keep ≤ 2 M |
| > 1000 mm | 0.6-0.8 mm | ≤ 2 M | flange edges must still be resolved |

The scanner model card (E4, amendment 4b) measured a median nearest-neighbour spacing of about
0.46 mm on the development exports.

## 7. Acceptance test with the integrator

1. Export the same physical part three times with the add-on, then once more manually with a
   wrong SKU.
2. The line must:
   - accept the three exports and produce three jobs;
   - quarantine the wrong SKU with UNMATCHED_SKU;
   - leave nothing in the share (with `move`).
3. Kill the network mid-export (pull the cable for 30 s). The partial file must not be ingested
   before the rename.
4. Export once in inches on purpose. The result must be INVALID_INSPECTION
   (UNIT_MISMATCH_SUSPECTED), never a measurement.
5. Check `GET /api/line/health/ready`: `hotfolder:<station>` stays `ok` throughout.

## 8. Answers (to be filled in with the integrator)

| # | Answer | Date / who |
|---|---|---|
| Q0 | | |
| Q1 | | |
| Q2-Q4 | | |
| Q5 | | |
| Q6 | | |
| Q7 | | |
| Q8 | | |
| Q9 | | |
| Q10 | | |
| Q11-Q13 | | |
