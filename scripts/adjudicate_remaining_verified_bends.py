#!/usr/bin/env python3
"""Build provisional human bend axes for the nine remaining verified scans.

The choices in this file are manual geometric adjudications made from six-view
renders, high-resolution local crops, and scan cross-sections.  They are kept
separate from the detector output so the resulting labels can be rerendered
and visually rejected before they are copied into a Bend Label Studio job.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / "source-assets"
SCANS = ASSETS / "02_CAD_QC_3D"
EXPERIMENTS = ROOT / "output" / "experiments" / "f1-10of13"


def _points(path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path))
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(points):
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)
    return points[np.isfinite(points).all(axis=1)]


def _stable_pca(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = _points(path)
    center = np.median(points, axis=0)
    sample = points - center
    _values, vectors = np.linalg.eigh(sample.T @ sample / max(1, len(sample)))
    basis = vectors[:, ::-1]
    for column in range(3):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
    return center, basis


def _raw_from_pca(endpoints: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.asarray(endpoints, dtype=np.float64) @ basis.T + center


def _pca_from_raw(endpoints: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(endpoints, dtype=np.float64) - center) @ basis


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _owned(payload: dict, bend_id: str) -> np.ndarray:
    record = next(item for item in payload["owned_bend_regions"] if item["bend_id"] == bend_id)
    endpoints = record.get("span_endpoints")
    if not endpoints:
        endpoints = [record["line_start"], record["line_end"]]
    return np.asarray(endpoints, dtype=np.float64)


def _hypothesis(payload: dict, bend_id: str, source_kind: str) -> np.ndarray:
    record = next(
        item
        for item in payload["bend_hypotheses"]
        if item["bend_id"] == bend_id and item["source_kind"] == source_kind
    )
    anchor = np.asarray(record["anchor"], dtype=np.float64)
    direction = np.asarray(record["axis_direction"], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    half_span = direction * (float(record["span_mm"]) / 2.0)
    return np.asarray([anchor - half_span, anchor + half_span])


def _line_at_y(line: np.ndarray, y_values: tuple[float, float]) -> np.ndarray:
    delta = line[1] - line[0]
    result = []
    for y in y_values:
        t = (y - line[0, 1]) / delta[1]
        result.append(line[0] + t * delta)
    return np.asarray(result)


def _line_at_x(line: np.ndarray, x_values: tuple[float, float]) -> np.ndarray:
    delta = line[1] - line[0]
    result = []
    for x in x_values:
        t = (x - line[0, 0]) / delta[0]
        result.append(line[0] + t * delta)
    return np.asarray(result)


def _case_01() -> tuple[Path, list[np.ndarray]]:
    scan = SCANS / "10839000_01_full_scan.ply"
    result = _payload(
        EXPERIMENTS / "full-zero-shot-current" / "10839000"
        / "10839000_01_full_scan.ply--1c826530ab64" / "decomposition_result.json"
    )
    center, basis = _stable_pca(scan)
    pca = {key: _pca_from_raw(_owned(result, key), center, basis) for key in (f"OB{i}" for i in range(1, 9))}

    # Four side-rail roots. OB4 was truncated by F1, so extend it over OB2's
    # visually complete longitudinal span.
    axes = [pca["OB1"], pca["OB2"], pca["OB3"], _line_at_y(pca["OB4"], (pca["OB2"][0, 1], pca["OB2"][1, 1]))]

    # The remaining F1 regions are cross-section fragments at the four corners.
    # Pair equal-height endpoints across the frame to recover the four actual
    # horizontal bend axes.
    for left_id, right_id in (("OB5", "OB7"), ("OB6", "OB8")):
        left = pca[left_id][np.argsort(pca[left_id][:, 2])]
        right = pca[right_id][np.argsort(pca[right_id][:, 2])]
        axes.extend((np.asarray([left[0], right[0]]), np.asarray([left[1], right[1]])))
    return scan, [_raw_from_pca(axis, center, basis) for axis in axes]


def _case_02() -> tuple[Path, list[np.ndarray]]:
    scan = SCANS / "10839000" / "10839000_01_scan_1.ply"
    result = _payload(
        EXPERIMENTS / "full-zero-shot-current" / "10839000"
        / "10839000_01_scan_1.ply--98b00841230a" / "decomposition_result.json"
    )
    center, basis = _stable_pca(scan)
    pca = {key: _pca_from_raw(_owned(result, key), center, basis) for key in ("OB1", "OB2", "OB3", "OB4")}

    top_points = np.vstack((pca["OB1"], pca["OB4"]))
    bottom_points = np.vstack((pca["OB2"], pca["OB3"]))
    top = np.asarray([top_points[np.argmin(top_points[:, 0])], top_points[np.argmax(top_points[:, 0])]])
    bottom = np.asarray([bottom_points[np.argmin(bottom_points[:, 0])], bottom_points[np.argmax(bottom_points[:, 0])]])
    axes = [top, bottom, np.asarray([top[0], bottom[0]]), np.asarray([top[1], bottom[1]])]
    return scan, [_raw_from_pca(axis, center, basis) for axis in axes]


def _case_05() -> tuple[Path, list[np.ndarray]]:
    scan = SCANS / "37361005_01.ply"
    result = _payload(EXPERIMENTS / "full-zero-shot-current" / "37361005" / "decomposition_result.json")
    return scan, [_owned(result, f"OB{index}") for index in range(1, 5)]


def _case_06() -> tuple[Path, list[np.ndarray]]:
    scan = SCANS / "38172000_01_SCAN_1.ply"
    result = _payload(EXPERIMENTS / "plane-lattice-probe" / "06_38172000" / "plane_lattice_manifest.json")
    center, basis = _stable_pca(scan)
    raw = {item["bend_id"]: np.asarray(item["span_endpoints"], dtype=np.float64) for item in result["owned_bend_regions"]}
    pca = {key: _pca_from_raw(value, center, basis) for key, value in raw.items()}
    axes = []
    for inner_id, outer_id in (("PLI_RP11_RP2", "PLI_RP1_RP3"), ("PLI_RP2_RP8", "PLI_RP1_RP4")):
        inner = pca[inner_id]
        outer = pca[outer_id]
        outer_return = outer.copy()
        outer_return[:, 2] += 14.0
        # The cross-section contains one inner plate transition and a two-bend
        # outer return.  The z=-14 inner-wall edge is a free edge, not a fold.
        axes.extend((inner, outer, outer_return))
    return scan, [_raw_from_pca(axis, center, basis) for axis in axes]


def _case_07() -> tuple[Path, list[np.ndarray]]:
    scan = SCANS / "38172000_01_SCAN_2 1.ply"
    result = _payload(EXPERIMENTS / "structural-render-v2" / "07_38172000" / "38172000" / "decomposition_result.json")
    center, basis = _stable_pca(scan)
    pca = {key: _pca_from_raw(_owned(result, key), center, basis) for key in (f"OB{i}" for i in range(1, 9))}
    axes = [pca["OB1"], pca["OB2"]]
    # OB4/OB5 lie midway up the two inner walls.  The center cross-section
    # resolves the two distinct junctions seven millimetres to either side.
    end_x = float(np.mean(pca["OB6"][:, 0]))
    for inner_id, outer_id in (("OB4", "OB1"), ("OB5", "OB2")):
        # F1 truncates both inner-wall regions near the formed end.  The scan
        # shows uninterrupted rails through the end-flange root, so extend the
        # fitted line over that full physical span before separating the two
        # vertically distinct junctions.
        inner = _line_at_x(pca[inner_id], (float(np.min(pca[outer_id][:, 0])), end_x))
        lower = inner.copy()
        upper = inner.copy()
        lower[:, 2] -= 7.0
        upper[:, 2] += 7.0
        axes.extend((lower, upper))
    axes.append(pca["OB6"])
    # OB7/OB8 are vertical corner fragments; only their upper endpoints define
    # the real return bend across the formed end flange.
    high_7 = pca["OB7"][np.argmax(pca["OB7"][:, 2])]
    high_8 = pca["OB8"][np.argmax(pca["OB8"][:, 2])]
    axes.append(np.asarray([high_7, high_8]))
    return scan, [_raw_from_pca(axis, center, basis) for axis in axes]


def _case_08() -> tuple[Path, list[np.ndarray]]:
    scan = SCANS / "40500000_0_SCAN2 1.ply"
    result = _payload(EXPERIMENTS / "plane-lattice-probe" / "08_40500000" / "plane_lattice_manifest.json")
    center, basis = _stable_pca(scan)
    raw = {item["bend_id"]: np.asarray(item["span_endpoints"], dtype=np.float64) for item in result["owned_bend_regions"]}
    axes = [
        _pca_from_raw(raw["PLI_RP1_RP2"], center, basis),
        _pca_from_raw(raw["PLI_RP1_RP3"], center, basis),
        # Four wall-end corner folds resolved by high-dihedral vertical clusters.
        np.asarray([[-90.0, 63.0, -4.6], [-90.0, 63.0, 40.4]]),
        np.asarray([[-92.9, -57.9, -4.2], [-92.9, -57.9, 38.9]]),
        np.asarray([[90.5, -61.3, -3.5], [90.5, -61.3, 41.6]]),
        np.asarray([[92.5, 66.6, -4.0], [92.5, 66.6, 40.8]]),
        # The +PC1 wall has a real top return; the -PC1 edge is a free outline.
        _pca_from_raw(raw["PLI_RP2_RP4"], center, basis),
    ]
    return scan, [_raw_from_pca(axis, center, basis) for axis in axes]


def _case_10() -> tuple[Path, list[np.ndarray]]:
    scan = SCANS / "49001000_A-PRELIM.ply"
    # The STEP resolves five paired bend cylinders: one 400 mm front flange,
    # two tall end walls, and two smaller inner tabs.  Their axes were aligned
    # to the scan and offset onto the observed bend surfaces.  Every resulting
    # line has sub-millimetre median/p90 scan distance.
    axes = [
        np.asarray([[-33.424, -66.283, 242.807], [-285.081, -376.090, 269.038]]),
        np.asarray([[-340.909, -287.798, 309.126], [-364.389, -246.464, 329.952]]),
        np.asarray([[-171.434, -12.045, 312.006], [-127.978, -28.787, 289.098]]),
        np.asarray([[-290.480, -373.313, 264.218], [-312.633, -334.317, 283.867]]),
        np.asarray([[-77.895, -47.862, 262.064], [-36.896, -63.658, 240.452]]),
    ]
    return scan, axes


def _case_11() -> tuple[Path, list[np.ndarray]]:
    scan = SCANS / "49024000_A.ply"
    # The clean STEP contains exactly three paired cylindrical bend faces
    # (inner radius 3 mm, outer radius 6 mm).  Their axes were rigidly aligned
    # to the scan's matching STL frame, then shifted to the inner-surface arc
    # midlines so markers sit on observed scan points rather than floating on
    # the cylinders' center axes.  Median scan distance is 0.14/0.23/0.18 mm.
    axes = [
        np.asarray([[-214.487, -126.078, 283.985], [-234.120, -143.164, 272.978]]),
        np.asarray([[-212.201, -122.472, 274.308], [-227.344, -135.651, 265.818]]),
        np.asarray([[-224.555, -118.025, 256.230], [-211.013, -106.239, 263.823]]),
    ]
    return scan, axes


def _case_13() -> tuple[Path, list[np.ndarray]]:
    scan = ASSETS / "wetransfer_ysherman-asc_2026-04-20_0802" / "ysherman.ply"
    result = _payload(
        EXPERIMENTS / "corridor-completion-integrated-v2" / "13_ysherman"
        / "ysherman" / "decomposition_result.json"
    )
    axes = [_owned(result, f"OB{index}") for index in range(1, 13) if index != 7]
    # OB7 follows the tongue silhouette.  B12 is the transverse, scan-supported
    # tongue-root bend visible in both oblique and side views.
    axes.append(_hypothesis(result, "B12", "zero_shot_strip_seed"))
    return scan, axes


CASES = {
    "01_10839000": ("bend_label_01_10839000", _case_01),
    "02_10839000": ("bend_label_02_10839000", _case_02),
    "05_37361005": ("bend_label_05_37361005", _case_05),
    "06_38172000": ("bend_label_06_38172000", _case_06),
    "07_38172000": ("bend_label_07_38172000", _case_07),
    "08_40500000": ("bend_label_08_40500000", _case_08),
    "10_49001000": ("bend_label_10_49001000", _case_10),
    "11_49024000": ("bend_label_11_49024000", _case_11),
    "13_ysherman": ("bend_label_13_ysherman", _case_13),
}


def main() -> None:
    provisional_root = ROOT / "output" / "manual-bend-label-review" / "provisional"
    provisional_root.mkdir(parents=True, exist_ok=True)
    index = []
    for case_id, (job_id, builder) in CASES.items():
        scan, axes = builder()
        payload = {
            "schema_version": 1,
            "kind": "sherman_qc_bend_instance_labels",
            "job_id": job_id,
            "coordinate_frame": "reference_mesh_raw",
            "review_status": "in_progress",
            "manual_bends": [
                {
                    "bend_id": f"M{index:02d}",
                    "semantic": "conventional_bend",
                    "source": "human_two_point",
                    "line_start": [float(value) for value in axis[0]],
                    "line_end": [float(value) for value in axis[1]],
                }
                for index, axis in enumerate(axes, 1)
            ],
            "marker_reviews": [],
        }
        output_path = provisional_root / f"{case_id}.json"
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        index.append({"case_id": case_id, "job_id": job_id, "scan": str(scan), "labels": str(output_path), "bend_count": len(axes)})
    (provisional_root / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
