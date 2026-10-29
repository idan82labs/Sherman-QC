"""Append bounded visual developments to the same five approved Site fixtures."""
from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.vendor_unfold_worker import build_unfold_preview_bounded

SOURCES = (
    ("37361005", "37361005_01.stp", "81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e"),
    ("49001000", "49001000_A-PRELIM.stp", "d8c1042f1673258976fa2b9ef812930c1be81ea3a60ab3f6357f966f7f452fd5"),
    ("42288005", "42288005_01.stp", "7afb000cdb1f18eddb4d270c5b8c2c5854d822d67adfd3cfebc5e7ad9e126837"),
    ("38172000", "38172000_01.stp", "4d512b5c565ab0e9e4d02e73fcaafd164ae82bcd977eba69198c8b0356f5644f"),
    ("10839000", "10839000_01.stp", "ea3e4820dff568916164c79417925ead79f307fba8aff11c7b3aeda7375e0b37"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument("--supervisor-sha256", required=True)
    parser.add_argument("--generation", required=True, choices=["unfold-generation-v2", "unfold-generation-v3"])
    args = parser.parse_args()
    fixture_root = ROOT / "sites/vendor-tagging-demo/public/vendor-corpus"
    cad_root = ROOT.parent / "source-assets/02_CAD_QC_3D"
    def verify_engine():
        for file, expected in (("backend/vendor_unfold_preview.py", args.engine_sha256),
                               ("backend/vendor_unfold_worker.py", args.supervisor_sha256),
                               ("scripts/extract_step_topology_bend_axes.py", "f089f71466e0f9bba47139c06efdf8bf651e7707fb05c78ca2195f06146b1314")):
            if hashlib.sha256((ROOT / file).read_bytes()).hexdigest() != expected:
                raise ValueError("Audited source binding failed")
    verify_engine()
    report_path = fixture_root / f"{args.generation}.json"
    if report_path.exists():
        raise ValueError("Existing generation is terminal; do not overwrite or disguise retries")
    report = {"generation": args.generation, "status": "RUNNING", "engine_sha256": args.engine_sha256,
        "supervisor_sha256": args.supervisor_sha256, "limits": {"wall_seconds": 60, "rss_gib": 4, "source_bytes": 5_000_000, "result_bytes": 40_000_000},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dispositions": [{"part": number, "status": "NOT_STARTED", "cad_sha256": expected} for number, _, expected in SOURCES],
        "measurement_claim": "NONE", "manufacturing_ready": False}
    def retain():
        report_path.write_text(json.dumps(report, allow_nan=False, separators=(",", ":")))
    with report_path.open("x") as stream:
        json.dump(report, stream, allow_nan=False, separators=(",", ":"))
    try:
        for index, (number, filename, expected_hash) in enumerate(SOURCES):
            verify_engine()
            fixture_path = fixture_root / f"{number}.json"
            fixture = json.loads(fixture_path.read_text())
            if fixture["part_number"] != number or fixture["cad_sha256"] != expected_hash:
                raise ValueError("Pinned fixture/CAD identity mismatch")
            preview = build_unfold_preview_bounded(cad_root / filename, expected_hash, fixture["candidates"])
            preview["diagnostics"]["generation"] = args.generation
            fixture["unfold_preview"] = preview
            # Derived JSON fixture only; existing source, PLY, candidates are retained.
            fixture_path.write_text(json.dumps(fixture, allow_nan=False, separators=(",", ":")))
            result = {"part": number, "cad_sha256": expected_hash, "status": preview["status"], "reason": preview["reason"],
                      "bends": len(preview["bend_lines"]), "triangles": len(preview["triangles"]) // 3}
            report["dispositions"][index] = result
            retain()
            print(json.dumps(result), flush=True)
        report["status"] = "COMPLETE"
    except BaseException:
        report["status"] = "FAILED"
        raise
    finally:
        retain()


if __name__ == "__main__":
    main()
