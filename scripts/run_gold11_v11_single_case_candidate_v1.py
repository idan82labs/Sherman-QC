#!/usr/bin/env python3
"""Run the versioned V11 single-part topology-support recovery candidate.

The frozen Gold11 V11 attempt is never reopened.  This successor consumes the
retained, terminal V11 context pose for CG4BDD1371B6, revalidates the current
scan/CAD bytes, and adds only the new recovery seam: native STEP bend axes are
projected through that pose and accepted only when the raw scan supports their
finite span.  Gold is opened only after inference, for scoring.  The result is
research-only and is not a production or floor-release decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.cad_guided_mesh_input_v1 import prepare_mesh_scan_v1
from backend.cad_guided_v11_topology_recovery_v1 import recover_scan_supported_axes_v1
from scripts.evaluate_f1_against_manual_gold import Thresholds, match_lines
from scripts.run_gold11_mesh_development_v11 import workflow_entry_and_policies


CASE_TOKEN = "CG4BDD1371B6"
SOURCE_ASSETS = ROOT.parent / "source-assets/02_CAD_QC_3D"
SCAN_PATH = SOURCE_ASSETS / "37361005_01.ply"
CAD_PATH = SOURCE_ASSETS / "37361005_01.stp"
GOLD_PATH = ROOT / "output/bend_label_05_37361005/bend_instance_labels.json"
RETAINED_V11_RESULT = ROOT / (
    "docs/research/gold11_mesh_development_v11_attempt/cases/"
    f"{CASE_TOKEN}/control/RESULT.json"
)
DEFAULT_OUTPUT = ROOT / "docs/research/gold11_v11_single_case_candidate_v1" / CASE_TOKEN

# These are the reviewed raw-file bindings for this consumed Gold11 case.  A
# changed byte is a hard stop; no mutable manifest can expand the input set.
SCAN_SHA256 = "d1732de180230bdbf0caa6eaf454a33071c206060aea276dce67a9abcf13b038"
CAD_SHA256 = "81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scan_to_cad_from_retained_v11(result: dict[str, Any]) -> np.ndarray:
    evidence = result.get("registration", {}).get("evidence", {})
    indices = evidence.get("measurement_context_indices")
    candidates = evidence.get("candidates")
    if not isinstance(indices, list) or not indices or not isinstance(candidates, list):
        raise ValueError("RETAINED_V11_CONTEXT_POSE_UNAVAILABLE")
    index = indices[0]
    candidate = candidates[index]
    rotation = np.asarray(candidate.get("rotation"), dtype=np.float64)
    translation = np.asarray(candidate.get("translation"), dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("RETAINED_V11_CONTEXT_POSE_SHAPE")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("RETAINED_V11_CONTEXT_POSE_NONFINITE")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    # V11's rigid-transform consumer is bound to the selected context.  This
    # check avoids silently using a scale/shear matrix in the fallback.
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("RETAINED_V11_CONTEXT_POSE_NOT_RIGID")
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError("RETAINED_V11_CONTEXT_POSE_REFLECTION")
    return matrix


def _render(path: Path, points: np.ndarray, predictions: list[dict[str, Any]]) -> None:
    # A deterministic decimation keeps the visual artifact responsive while
    # retaining the complete raw scan for all geometry calculations.
    limit = 30_000
    if len(points) > limit:
        rows = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
        cloud = points[rows]
    else:
        cloud = points
    fig = plt.figure(figsize=(15, 9), dpi=150)
    axes = (fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d"))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, max(1, len(predictions))))
    for panel, (elev, azim, title) in zip(
        axes,
        ((22, 36, "V11 candidate · oblique"), (72, 8, "V11 candidate · plan view")),
    ):
        panel.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=0.30, c="#8a96a3", alpha=0.20, linewidths=0)
        for index, (prediction, color) in enumerate(zip(predictions, colors)):
            line = np.asarray([prediction["line_start"], prediction["line_end"]], dtype=np.float64)
            panel.plot(line[:, 0], line[:, 1], line[:, 2], color=color, linewidth=2.8,
                       label=f"B{index + 1}  {prediction['bend_data']['line_span_mm']:.1f} mm")
            direction = line[1] - line[0]
            length_sq = float(direction @ direction)
            parameter = np.clip((points - line[0]) @ direction / length_sq, 0.0, 1.0)
            nearest = line[0] + parameter[:, None] * direction
            distance = np.linalg.norm(points - nearest, axis=1)
            support_rows = np.flatnonzero(distance <= 3.0)
            if len(support_rows) > 2500:
                support_rows = support_rows[np.linspace(0, len(support_rows) - 1, 2500, dtype=np.int64)]
            if len(support_rows):
                support_cloud = points[support_rows]
                panel.scatter(support_cloud[:, 0], support_cloud[:, 1], support_cloud[:, 2],
                              s=3.0, color=color, alpha=0.55, linewidths=0)
                # Re-plot after the support cloud so the finite marker remains
                # legible with Matplotlib's 3-D painter ordering.
                panel.plot(line[:, 0], line[:, 1], line[:, 2], color=color, linewidth=3.2)
            midpoint = line.mean(axis=0)
            panel.text(*midpoint, f"  B{index + 1}", color=color, fontsize=8)
        panel.set_title(title, pad=12, fontsize=12, fontweight="bold")
        panel.set_xlabel("X (mm)")
        panel.set_ylabel("Y (mm)")
        panel.set_zlabel("Z (mm)")
        panel.view_init(elev=elev, azim=azim)
        panel.legend(loc="upper left", fontsize=8, framealpha=0.9)
        # Equal scale makes line placement visually meaningful.
        lo, hi = np.min(cloud, axis=0), np.max(cloud, axis=0)
        center, radius = (lo + hi) / 2.0, float(np.max(hi - lo) / 2.0)
        panel.set_xlim(center[0] - radius, center[0] + radius)
        panel.set_ylim(center[1] - radius, center[1] + radius)
        panel.set_zlim(center[2] - radius, center[2] + radius)
    fig.suptitle("Sherman QC · Gold11 V11 single-part recovery (research-only)", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.015, "Colored lines are finite raw-scan-supported targets; CAD radii/sweeps remain nominal reference metadata.",
             ha="center", fontsize=9, color="#46515c")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def run(output: Path) -> dict[str, Any]:
    if _sha256(SCAN_PATH) != SCAN_SHA256 or _sha256(CAD_PATH) != CAD_SHA256:
        raise ValueError("FROZEN_CASE_BYTE_HASH_MISMATCH")
    retained = json.loads(RETAINED_V11_RESULT.read_text(encoding="utf-8"))
    if retained.get("status") != "UNAVAILABLE" or retained.get("predictions") != []:
        raise ValueError("RETAINED_V11_CONTROL_NOT_THE_ZERO_OUTPUT_CASE")

    # Constructor qualification is side-effect free and binds the candidate to
    # the maintained V11 policy family without launching the terminal 11-case
    # attempt again.
    _, policies = workflow_entry_and_policies("control")
    scan_bytes = SCAN_PATH.read_bytes()
    scan = prepare_mesh_scan_v1(
        scan_bytes=scan_bytes,
        max_engine_rows=policies["workflow_policy"].max_registration_scan_rows,
    )
    if scan.get("status") != "AVAILABLE":
        raise ValueError("V11_SCAN_PREPARATION_UNAVAILABLE")
    matrix = _scan_to_cad_from_retained_v11(retained)
    recovered = recover_scan_supported_axes_v1(scan["raw_points"], CAD_PATH, matrix)

    # Gold is deliberately opened only at this post-inference scoring seam.
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))["manual_bends"]
    score = match_lines(gold, recovered["predictions"], Thresholds())
    gold_count = int(score["gold_count"])
    capture = score["true_positive"] / gold_count if gold_count else None
    result = {
        "schema": "GOLD11_V11_SINGLE_CASE_CANDIDATE_RESULT_V1",
        "candidate_implementation": recovered["implementation"],
        "case_token": CASE_TOKEN,
        "part_key": "37361005",
        "reference_id": "corpus/05_37361005",
        "status": recovered["status"],
        "v11_control": {
            "terminal_attempt_reopened": False,
            "retained_result": str(RETAINED_V11_RESULT.relative_to(ROOT)),
            "retained_status": retained["status"],
            "retained_prediction_count": len(retained.get("predictions", [])),
            "context_pose_source": "RETAINED_V11_COMPLETED_CONTEXT_INDEX_0",
        },
        "input_bindings": {
            "scan_path": str(SCAN_PATH),
            "scan_sha256": SCAN_SHA256,
            "cad_path": str(CAD_PATH),
            "cad_sha256": CAD_SHA256,
            "gold_path": str(GOLD_PATH),
            "gold_used_for_inference": False,
            "gold_used_for_scoring": True,
        },
        "source_bindings": {
            "candidate_script_sha256": _sha256(Path(__file__).resolve()),
            "recovery_module_sha256": _sha256((ROOT / "backend/cad_guided_v11_topology_recovery_v1.py").resolve()),
            "retained_v11_result_sha256": _sha256(RETAINED_V11_RESULT),
        },
        "v11_measurement_control": {
            "status": "UNAVAILABLE",
            "reason": "FLANGE_PLANE_GATE_ZERO_SUPPORT_RECORDS_IN_TERMINAL_V11_CONTROL",
            "patched_stage": "NATIVE_STEP_AXIS_RAW_SCAN_SUPPORT_RECOVERY_AFTER_V11_MEASUREMENT_EMPTY",
        },
        "recovery": recovered,
        "score": {
            **score,
            "bend_count_capture": capture,
            "target_80_percent_met": bool(capture is not None and capture >= 0.80),
            "f1": (2.0 * score["true_positive"] /
                   (2.0 * score["true_positive"] + score["false_positive"] + score["false_negative"]))
            if (2.0 * score["true_positive"] + score["false_positive"] + score["false_negative"]) else 0.0,
            "matcher": "UNCHANGED_MANUAL_GOLD_ONE_TO_ONE_MINIMUM_COST",
        },
        "production_use_allowed": False,
        "floor_release_allowed": False,
        "uncertainty_status": "UNCALIBRATED_SCAN_SUPPORT_DISTANCES",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESULT.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _render(output / "VISUAL.png", scan["raw_points"], recovered["predictions"])
    (output / "VISUAL_METADATA.json").write_text(json.dumps({
        "schema": "GOLD11_V11_SINGLE_CASE_VISUAL_METADATA_V1",
        "path": str((output / "VISUAL.png").resolve()),
        "raw_scan_points_rendered": min(30_000, len(scan["raw_points"])),
        "raw_scan_points_used_for_geometry": len(scan["raw_points"]),
        "prediction_count": len(recovered["predictions"]),
        "semantics": "VISUAL_REVIEW_ARTIFACT_NOT_METROLOGY_CERTIFICATE",
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "CHECKSUMS.json").write_text(json.dumps({
        "schema": "GOLD11_V11_SINGLE_CASE_CANDIDATE_CHECKSUMS_V1",
        "result_sha256": _sha256(output / "RESULT.json"),
        "visual_sha256": _sha256(output / "VISUAL.png"),
        "visual_metadata_sha256": _sha256(output / "VISUAL_METADATA.json"),
        "source_bindings": result["source_bindings"],
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(json.dumps({
        "status": result["status"],
        "predictions": len(result["recovery"]["predictions"]),
        "true_positive": result["score"]["true_positive"],
        "gold_count": result["score"]["gold_count"],
        "bend_count_capture": result["score"]["bend_count_capture"],
        "f1": result["score"]["f1"],
        "target_80_percent_met": result["score"]["target_80_percent_met"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
