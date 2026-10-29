#!/usr/bin/env python3
"""Run frozen V63 LTCO once on the visually accepted V5R3B lane."""

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

from backend.f1_cylindrical_band_completion import complete_cylindrical_bands
from backend.f1_localized_turn_corridor_ownership_v63 import (
    CONFIGURATION_SHA256,
    PROFILE,
    MeshSectionIndex,
    StationArtifact,
    apply_v63,
)
from scripts.build_f1_scan_only_candidate_pool import _load_geometry_views


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V5R3_CORPUS_CONDITIONED_LTCO_PROSPECTIVE"
EXPECTED_CONFIGURATION_SHA256 = "778bda993566612fbe671390106962c97b58bbb92c5a8f49c0c6e07cdd1501c4"
EXPECTED_CORE_SHA256 = "f35dbc2941bff52b3191bcda75b17313ab47fc26450b6e3fbeae3a5af1f8251c"
VISUAL_PRECOMMIT = ROOT / "docs/research/2026-08-15_f1_v5r3_corpus_conditioned_visual_precommit.md"
EXPECTED_VISUAL_PRECOMMIT_GENERATOR_SHA256 = "59e79628e96d3d2c01ac822a228d575ef8f71f6c71e16b402c2351e3091d8d74"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _restore_hypotheses(pool: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in pool.get("candidates") or ():
        if str(row.get("candidate_source") or "") != "bend_hypothesis":
            continue
        restored = dict(row)
        source_id = str(restored.get("source_candidate_id") or "")
        restored["candidate_id"] = source_id
        restored["bend_id"] = source_id
        restored.pop("candidate_source", None)
        restored.pop("source_candidate_id", None)
        result.append(restored)
    return result


def _rebuild_transient_support(
    pool: dict[str, Any], points: np.ndarray, normals: np.ndarray
) -> dict[str, tuple[int, ...]]:
    support: dict[str, tuple[int, ...]] = {}
    rebuilt = complete_cylindrical_bands(
        points,
        normals,
        _restore_hypotheses(pool),
        point_budget=200_000,
        candidate_budget=512,
        corridor_radius_mm=30.0,
        transient_support_sink=support,
    )
    pool_rows = {
        str(row.get("source_candidate_id") or ""): row
        for row in pool.get("candidates") or ()
        if str(row.get("candidate_source") or "") == "cylindrical_wide"
    }
    rebuilt_rows = {str(row["candidate_id"]): row for row in rebuilt["candidates"]}
    if set(pool_rows) != set(rebuilt_rows):
        raise RuntimeError("TRANSIENT_WIDE_CANDIDATE_ID_MISMATCH")
    keys = (
        "seed_ids", "anchor", "axis_direction", "span_mm", "span_endpoints",
        "radius_mm", "transverse_offset_mm", "vote_count", "support_count",
        "normal_spread_deg", "score", "consensus_count", "radius_consensus_count",
    )
    for candidate_id in sorted(rebuilt_rows):
        left = {key: rebuilt_rows[candidate_id].get(key) for key in keys}
        right = {key: pool_rows[candidate_id].get(key) for key in keys}
        if left != right:
            raise RuntimeError(f"TRANSIENT_WIDE_CANDIDATE_SEMANTIC_MISMATCH:{candidate_id}")
    if set(support) != set(rebuilt_rows):
        raise RuntimeError("TRANSIENT_WIDE_SUPPORT_KEY_MISMATCH")
    return support


def _render_profiles(case_id: str, artifacts: list[StationArtifact], output: Path) -> list[dict[str, Any]]:
    rows = []
    by_candidate: dict[str, list[StationArtifact]] = {}
    for artifact in artifacts:
        by_candidate.setdefault(artifact.candidate_id, []).append(artifact)
    for ordinal, candidate_id in enumerate(sorted(by_candidate), 1):
        figure, axes = plt.subplots(3, 3, figsize=(15, 11), dpi=140)
        for axis, artifact in zip(axes.ravel(), sorted(by_candidate[candidate_id], key=lambda row: row.station_index)):
            axis.set_title(
                f"station {artifact.station_index + 1} | {'PASS' if artifact.passed else artifact.reason}",
                fontsize=8,
            )
            if artifact.u is None:
                axis.text(0.5, 0.5, artifact.reason, ha="center", va="center", transform=axis.transAxes)
            else:
                axis.plot(artifact.u, np.degrees(artifact.theta_fine), color="#ff7f0e", lw=1.5, label="fine")
                axis.plot(artifact.u, np.degrees(artifact.theta_coarse), color="#1f77b4", lw=1.3, label="coarse")
                if artifact.fine_interval is not None:
                    axis.axvspan(*artifact.fine_interval, color="#ff7f0e", alpha=0.12)
                if artifact.coarse_interval is not None:
                    axis.axvspan(*artifact.coarse_interval, color="#1f77b4", alpha=0.10)
                axis.axvline(0.0, color="black", lw=0.7, alpha=0.6)
                axis.set_xlabel("transverse arclength u (mm)", fontsize=7)
                axis.set_ylabel("unwrapped normal angle (deg)", fontsize=7)
                axis.tick_params(labelsize=6)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="upper right")
        figure.suptitle(f"{case_id} | {candidate_id} | V63 transient routed profiles")
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        path = output / f"candidate_{ordinal:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        plt.close(figure)
        rows.append({"candidate_id": candidate_id, "path": str(path), "sha256": _sha(path)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--v51-root", type=Path, required=True)
    parser.add_argument("--v62-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("exact V5R3 prospective token required")
    if CONFIGURATION_SHA256 != EXPECTED_CONFIGURATION_SHA256:
        raise RuntimeError("V63_CONFIGURATION_HASH_MISMATCH")
    if _sha(ROOT / "backend/f1_localized_turn_corridor_ownership_v63.py") != EXPECTED_CORE_SHA256:
        raise RuntimeError("V63_CORE_HASH_MISMATCH")
    if not VISUAL_PRECOMMIT.is_file() or EXPECTED_VISUAL_PRECOMMIT_GENERATOR_SHA256 not in VISUAL_PRECOMMIT.read_text(encoding="utf-8"):
        raise RuntimeError("MISSING_VISUAL_PRECOMMIT")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("V63 output root is not empty")
    corpus_root = args.corpus_root.resolve()
    public = _read(corpus_root / "public_manifest.json")
    if public.get("token") != TOKEN or public.get("confirmatory_eligible") is not False:
        raise ValueError("V5R3 public manifest contract mismatch")
    case_ids = tuple(str(row["opaque_case_id"]) for row in public["cases"])
    records = []
    for pass_name in ("primary", "replay"):
        for row in public["cases"]:
            case_id = str(row["opaque_case_id"])
            source_path = Path(str(row["bundle_path"])) / "source.ply"
            if _sha(source_path) != row["source_sha256"]:
                raise RuntimeError(f"{case_id}:SOURCE_HASH_MISMATCH")
            v62_path = args.v62_root / pass_name / case_id / "semantic_result.json"
            pool_path = args.v51_root / pass_name / case_id / "pool" / f"{case_id}.json"
            pool = _read(pool_path)
            points, normals, _, _ = _load_geometry_views(source_path, plane_voxel_mm=1.8)
            support = _rebuild_transient_support(pool, points, normals)
            mesh = MeshSectionIndex.from_path(source_path)
            result, artifacts = apply_v63(
                _read(v62_path), pool, mesh, points, normals, support
            )
            result_path = output_root / pass_name / case_id / "semantic_result.json"
            _write(result_path, result)
            visuals = []
            if pass_name == "primary":
                visuals = _render_profiles(case_id, artifacts, output_root / "transient-profile-visuals" / case_id)
            records.append({
                "case_id": case_id, "pass": pass_name,
                "source_sha256": row["source_sha256"], "v62_sha256": _sha(v62_path),
                "pool_sha256": _sha(pool_path), "v63_sha256": _sha(result_path),
                "v52_origin_count": result["audit"]["v52_origin_count"],
                "v52_retained_count": result["audit"]["v52_retained_count"],
                "v52_deleted_count": result["audit"]["v52_deleted_count"],
                "candidate_count": result["candidate_count"], "profile_visuals": visuals,
                "transient_source_support_serialized": False,
            })
            # Profile and support arrays must not survive this iteration except
            # as the already-rendered scalar/line plots.
            del artifacts, support, points, normals, mesh
    primary = {row["case_id"]: row for row in records if row["pass"] == "primary"}
    replay = {row["case_id"]: row for row in records if row["pass"] == "replay"}
    identity = {case_id: primary[case_id]["v63_sha256"] == replay[case_id]["v63_sha256"] for case_id in case_ids}
    if not all(identity.values()):
        raise RuntimeError("V63_PRIMARY_REPLAY_MISMATCH")
    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v63_ltco_v5r3_prospective_execution",
        "profile": PROFILE,
        "configuration_sha256": CONFIGURATION_SHA256,
        "token": TOKEN,
        "case_ids": list(case_ids),
        "case_count": len(case_ids),
        "passes": ["primary", "replay"],
        "semantic_replay_identity": identity,
        "records": records,
        "source_only": True,
        "research_only": True,
        "truth_opened": False,
        "source_point_serialization": False,
        "visual_gate_passed_before_inference": True,
        "confirmatory_eligible": False,
    }
    _write(output_root / "execution_manifest.json", manifest)
    print(json.dumps({
        "case_count": len(case_ids), "replay_identity": True,
        "v52_origin": sum(row["v52_origin_count"] for row in primary.values()),
        "v52_deleted": sum(row["v52_deleted_count"] for row in primary.values()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
