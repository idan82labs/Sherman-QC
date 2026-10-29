#!/usr/bin/env python3
"""Apply the locked geometry-only F1 source router without opening gold labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_zero_shot_source_router import route_f1_zero_shot_sources  # noqa: E402
from backend.f1_corridor_plane_snap import snap_corridor_axes_by_plane_pareto  # noqa: E402
from backend.f1_interface_birth_consensus import promote_repeated_interface_births  # noqa: E402
from backend.f1_interface_birth_stitch import promote_stitched_interface_births  # noqa: E402
from backend.f1_parallel_skin_plane_snap import snap_stitches_to_supported_skin_plane  # noqa: E402
from backend.f1_transition_boundary_refit import promote_supported_transition_boundary  # noqa: E402
from backend.f1_axial_support_split import promote_axial_support_lobes  # noqa: E402
from backend.f1_dominant_topology_replacement import replace_complete_dominant_topology  # noqa: E402


BASE = ROOT / "output/research/f1-corpus-evidence-20260729-v1"


def _zero_predictions(overlay_path: Path, result_payload: dict) -> list[dict]:
    manifest = json.loads(overlay_path.read_text(encoding="utf-8"))
    strip_families = {
        str(strip.get("strip_id")): str(strip.get("strip_family") or "")
        for strip in result_payload.get("bend_support_strips") or []
    }
    predictions = []
    for bend in manifest.get("bends") or []:
        line = bend.get("detected_line") or bend.get("cad_line") or {}
        if line.get("display_start") is None or line.get("display_end") is None:
            continue
        predictions.append(
            {
                "bend_id": str(bend.get("bend_id") or f"ZS{len(predictions)+1}"),
                "span_endpoints": [line["display_start"], line["display_end"]],
                "source_kind": "zero_shot_support_strip",
                "strip_family": strip_families.get(str(bend.get("bend_id") or ""), ""),
            }
        )
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", type=Path, default=BASE / "patch_graph_fresh/inference")
    parser.add_argument("--zero-summary", type=Path, default=BASE / "inference/inference_summary.json")
    parser.add_argument("--output-dir", type=Path, default=BASE / "routed_v1")
    args = parser.parse_args()
    patch_dirs = sorted((args.patch_root / "isolated_cases").glob("[0-9][0-9]_*"))
    zero_summary = json.loads(args.zero_summary.read_text(encoding="utf-8"))
    if len(patch_dirs) != len(zero_summary["result_map"]):
        raise ValueError("patch/zero case-count mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_map = {}
    decisions = []
    for index, patch_dir in enumerate(patch_dirs, start=1):
        token = f"C{index:02d}"
        patch_paths = sorted(patch_dir.glob("result/**/decomposition_result.json"))
        if len(patch_paths) != 1:
            raise ValueError(f"expected one patch result for {token}, found {len(patch_paths)}")
        patch_payload = json.loads(patch_paths[0].read_text(encoding="utf-8"))
        zero_paths = zero_summary["result_map"][token]
        zero_payload = json.loads(Path(zero_paths["zero_shot_result"]).read_text(encoding="utf-8"))
        zero_selected = _zero_predictions(Path(zero_paths["overlay_manifest"]), zero_payload)
        snapped_patch = snap_corridor_axes_by_plane_pareto(
            patch_payload.get("owned_bend_regions") or [],
            patch_payload.get("bend_hypotheses") or [],
            patch_payload.get("flange_hypotheses") or [],
        )
        decision = route_f1_zero_shot_sources(
            snapped_patch["selected"], zero_selected, zero_payload
        )
        birth_consensus = promote_repeated_interface_births(
            decision["selected"], patch_payload.get("bend_hypotheses") or []
        )
        birth_stitch = promote_stitched_interface_births(
            birth_consensus["selected"], patch_payload.get("bend_hypotheses") or []
        )
        skin_snap = snap_stitches_to_supported_skin_plane(
            birth_stitch["selected"], patch_payload.get("flange_hypotheses") or []
        )
        transition_boundary = promote_supported_transition_boundary(
            skin_snap["selected"],
            patch_payload.get("bend_hypotheses") or [],
            patch_payload.get("flange_hypotheses") or [],
            patch_payload.get("atom_graph") or {},
        )
        axial_split = promote_axial_support_lobes(
            transition_boundary["selected"],
            patch_payload.get("bend_hypotheses") or [],
            patch_payload.get("atom_graph") or {},
        )
        topology_replacement = replace_complete_dominant_topology(axial_split["selected"])
        case_dir = args.output_dir / token
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path = case_dir / "decomposition_result.json"
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "algorithm": "f1_geometry_only_source_router_v1",
                    "inference_contract": {
                        "uses_part_identifier": False,
                        "uses_expected_count": False,
                        "uses_manual_truth": False,
                    },
                    "source_route": {key: value for key, value in decision.items() if key not in {"selected", "formed_feature_selected"}},
                    "corridor_plane_snap": {key: value for key, value in snapped_patch.items() if key != "selected"},
                    "interface_birth_consensus": {key: value for key, value in birth_consensus.items() if key not in {"selected", "promoted"}},
                    "promoted_interface_birth_regions": birth_consensus["promoted"],
                    "interface_birth_stitch": {key: value for key, value in birth_stitch.items() if key not in {"selected", "promoted"}},
                    "promoted_interface_birth_stitches": birth_stitch["promoted"],
                    "parallel_skin_plane_snap": {key: value for key, value in skin_snap.items() if key != "selected"},
                    "transition_boundary_refit": {key: value for key, value in transition_boundary.items() if key not in {"selected", "promoted"}},
                    "promoted_transition_boundaries": transition_boundary["promoted"],
                    "axial_support_split": {key: value for key, value in axial_split.items() if key not in {"selected", "promoted"}},
                    "promoted_axial_support_lobes": axial_split["promoted"],
                    "dominant_topology_replacement": {key: value for key, value in topology_replacement.items() if key != "selected"},
                    "owned_bend_regions": topology_replacement["selected"],
                    "formed_feature_bend_regions": decision["formed_feature_selected"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        case_map[token] = str(result_path)
        decisions.append({"case_token": token, **{key: value for key, value in decision.items() if key not in {"selected", "formed_feature_selected"}}, "formed_feature_count": len(decision["formed_feature_selected"])})
    (args.output_dir / "case_map.json").write_text(json.dumps(case_map, indent=2) + "\n")
    (args.output_dir / "routing_decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
    print(json.dumps({"case_count": len(case_map), "routes": {row["case_token"]: row["route"] for row in decisions}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
