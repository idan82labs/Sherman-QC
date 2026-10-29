"""Prepare a local vendor UI demo from one previously consumed CAD only.

No scan, Gold reference, catalog database, or benchmark artifact is read.
The fixed hash binds the source to the reviewed Gold11 CAD input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import trimesh

from backend.bend_inspection_pipeline import load_cad_geometry
from backend.vendor_flange_preview import build_flange_previews
from scripts.extract_step_topology_bend_axes import extract_step_bend_axes

CAD = ROOT.parent / "source-assets/02_CAD_QC_3D/37361005_01.stp"
APPROVED_SHA256 = "81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/sherman-vendor-real-demo"))
    args = parser.parse_args()
    if hashlib.sha256(CAD.read_bytes()).hexdigest() != APPROVED_SHA256:
        raise ValueError("Reviewed CAD hash mismatch; refusing demo preparation")
    vertices, triangles, source = load_cad_geometry(str(CAD), cad_import_deflection=0.05)
    if triangles is None or not len(triangles):
        raise ValueError("CAD did not yield a mesh")
    if len(vertices) > 1_000_000 or len(triangles) > 500_000:
        raise ValueError("CAD preview exceeds the bounded vendor preview contract")
    topology = extract_step_bend_axes(CAD)
    axes = topology["bend_axes"]
    if len(axes) < 4:
        raise ValueError(f"Fewer than four native bend axes: {len(axes)}")
    marks = []
    for index, axis in enumerate(axes, start=1):
        marks.append({
            "feature_id": f"native-step-{index}",
            "label": f"B{index}",
            "source": "cad_candidate",
            "status": "active",
            "review_status": "pending",
            "issue": None,
            "note": "",
            "line_start": axis["canonical_line_start"],
            "line_end": axis["canonical_line_end"],
            "target_angle": round(float(np.median(axis["angular_sweeps_deg"])), 1),
            "target_radius": None,
        })
    flange_surfaces = build_flange_previews(CAD, APPROVED_SHA256, marks)
    if {surface["feature_id"] for surface in flange_surfaces} != {mark["feature_id"] for mark in marks}:
        raise ValueError("The reviewed demo requires a verified flange preview for every bend")
    args.output.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=triangles, process=False).export(
        args.output / "real-part-preview.ply"
    )
    (args.output / "real-part-demo.json").write_text(json.dumps({
        "part_number": "37361005",
        "cad_sha256": APPROVED_SHA256,
        "geometry_source": source,
        "vertices": len(vertices),
        "triangles": len(triangles),
        "candidate_source": "native_STEP_topology",
        "candidates": marks,
        "flange_surfaces": flange_surfaces,
        "flange_surface_status": "AVAILABLE",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "vertices": len(vertices), "triangles": len(triangles), "bend_axes": len(marks), "flange_surfaces": len(flange_surfaces)}))


if __name__ == "__main__":
    main()
