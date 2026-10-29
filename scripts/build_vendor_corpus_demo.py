"""Build a bounded, CAD-only local vendor demo from five existing corpus STEP files.

No scans, Gold labels, held-outs, catalog DB, or benchmark artifacts are read.
Outputs are derived UI fixtures in /tmp, not production frontend assets.
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
from backend.vendor_flange_preview import FlangePreviewUnavailable, build_flange_previews
from scripts.extract_step_topology_bend_axes import extract_step_bend_axes


CAD_ROOT = ROOT.parent / "source-assets/02_CAD_QC_3D"
SOURCES = (
    ("37361005", "37361005_01.stp", "81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e"),
    ("49001000", "49001000_A-PRELIM.stp", "d8c1042f1673258976fa2b9ef812930c1be81ea3a60ab3f6357f966f7f452fd5"),
    ("42288005", "42288005_01.stp", "7afb000cdb1f18eddb4d270c5b8c2c5854d822d67adfd3cfebc5e7ad9e126837"),
    ("38172000", "38172000_01.stp", "4d512b5c565ab0e9e4d02e73fcaafd164ae82bcd977eba69198c8b0356f5644f"),
    ("10839000", "10839000_01.stp", "ea3e4820dff568916164c79417925ead79f307fba8aff11c7b3aeda7375e0b37"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/sherman-vendor-corpus-demo"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for part_number, filename, expected_hash in SOURCES:
        cad = CAD_ROOT / filename
        raw = cad.read_bytes()
        if len(raw) > 5_000_000 or hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"CAD source binding failed: {part_number}")
        vertices, triangles, geometry_source = load_cad_geometry(str(cad), cad_import_deflection=0.05)
        if triangles is None or not len(triangles) or len(vertices) > 1_000_000 or len(triangles) > 500_000:
            raise ValueError(f"CAD preview exceeds bounds: {part_number}")
        axes = extract_step_bend_axes(cad)["bend_axes"]
        if not 4 <= len(axes) <= 32:
            raise ValueError(f"Expected 4–32 native candidate axes: {part_number}, got {len(axes)}")
        marks = [{
            "feature_id": f"native-step-{index}", "label": f"B{index}",
            "source": "cad_candidate", "status": "active", "review_status": "pending",
            "issue": None, "note": "", "line_start": axis["canonical_line_start"],
            "line_end": axis["canonical_line_end"],
            "target_angle": round(float(np.median(axis["angular_sweeps_deg"])), 1),
            "target_radius": None,
        } for index, axis in enumerate(axes, start=1)]
        try:
            surfaces = build_flange_previews(cad, expected_hash, marks)
            surface_status = "AVAILABLE" if surfaces else "UNAVAILABLE"
        except FlangePreviewUnavailable:
            surfaces = []
            surface_status = "UNAVAILABLE"
        mesh_path = output / f"{part_number}.ply"
        trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=triangles, process=False).export(mesh_path)
        fixture = {
            "part_number": part_number, "cad_sha256": expected_hash,
            "geometry_source": geometry_source, "candidates": marks,
            "candidate_generation": hashlib.sha256(json.dumps(marks, sort_keys=True).encode()).hexdigest(),
            "flange_surfaces": surfaces, "flange_surface_status": surface_status,
            "vertices": len(vertices), "triangles": len(triangles),
        }
        (output / f"{part_number}.json").write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
        manifest.append({
            "id": f"corpus-{part_number}", "part_number": part_number,
            "cad_sha256": expected_hash, "candidate_count": len(marks),
            "flange_surface_count": len(surfaces), "mesh_bytes": mesh_path.stat().st_size,
            "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        })
        print(json.dumps(manifest[-1]), flush=True)
    (output / "manifest.json").write_text(json.dumps({"schema_version": 1, "parts": manifest}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
