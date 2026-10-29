#!/usr/bin/env python3
"""Execute frozen V51 once on a sealed opaque 30-case corpus, then replay.

The script contains no scorer and cannot read the truth/family stores under the
frozen Seatbelt profile. Runtime fields are kept out of the semantic result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOKEN = "V51_NEW_UNTOUCHED_CORPUS_BLIND_ONCE_LOCKED"
CASE_IDS = tuple(f"O{index:03d}" for index in range(1, 31))
SANDBOX = ROOT / "research/f1_v51_new_untouched_corpus_detector.sb"
PYTHON = ROOT / ".venv/bin/python"
MAX_RSS_PATTERN = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.MULTILINE)
THREAD_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "LC_ALL": "C",
    "TZ": "UTC",
}
SEMANTIC_KEYS = (
    "schema_version",
    "kind",
    "profile",
    "v49_retained",
    "candidate_count",
    "predictions",
    "single_axis_parallel_return_recovery_audit",
    "expected_count_used",
    "labels_used",
    "cad_used",
    "part_identity_used",
    "source_only",
    "research_only",
    "v51_result_sha256",
)
FORBIDDEN_CACHE_KEYS = {
    "source_points",
    "source_cloud",
    "vertices",
    "faces",
    "triangles_raw",
    "raw_points",
    "point_cloud",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if compact
        else json.dumps(payload, sort_keys=True, indent=2)
    )
    path.write_text(text + "\n", encoding="utf-8")


def _append_event(payload: dict[str, Any]) -> None:
    events = ROOT / "output/research/f1-v51-new-untouched-corpus-20260814/invocation_events.jsonl"
    record = {
        "token": TOKEN,
        "event_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "research_only": True,
        **payload,
    }
    encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with events.open("ab", buffering=0) as handle:
        handle.write(encoded)
        os.fsync(handle.fileno())


def _recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(str(key) for key in value)
        for child in value.values():
            keys.update(_recursive_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_recursive_keys(child))
    return keys


def canonical_semantic(raw: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in SEMANTIC_KEYS if key not in raw]
    if missing:
        raise ValueError(f"raw V51 result is missing semantic keys: {missing}")
    result = {key: raw[key] for key in SEMANTIC_KEYS}
    if result["kind"] != "sherman_qc_f1_v51_single_axis_parallel_return_recovery_result":
        raise ValueError("unexpected V51 result kind")
    if result["source_only"] is not True or result["research_only"] is not True:
        raise ValueError("V51 result violated source-only/research-only contract")
    if any(result[key] is not False for key in ("expected_count_used", "labels_used", "cad_used", "part_identity_used")):
        raise ValueError("V51 result reported forbidden inference input")
    return result


def validate_opaque_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("token") != TOKEN or payload.get("schema_version") != 1:
        raise ValueError("opaque manifest token/schema mismatch")
    cases = list(payload.get("cases") or [])
    if [str(row.get("case_id")) for row in cases] != list(CASE_IDS):
        raise ValueError("opaque manifest must contain O001..O030 in exact order")
    for row in cases:
        case_id = str(row["case_id"])
        scan = Path(str(row["scan_path"])).resolve()
        if scan.name != f"{case_id}.ply":
            raise ValueError(f"{case_id}: scan filename is not opaque")
        if _sha(scan) != row.get("scan_sha256"):
            raise ValueError(f"{case_id}: scan hash mismatch")
        if set(row) != {"case_id", "scan_path", "scan_sha256"}:
            raise ValueError(f"{case_id}: opaque manifest contains extra metadata")
    return cases


def _run(stage: str, command: list[str], case_dir: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update(THREAD_ENV)
    wrapped = ["/usr/bin/sandbox-exec", "-f", str(SANDBOX), "/usr/bin/time", "-l", *command]
    started = perf_counter()
    completed = subprocess.run(
        wrapped,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=360,
    )
    elapsed = perf_counter() - started
    (case_dir / "logs").mkdir(parents=True, exist_ok=True)
    (case_dir / "logs" / f"{stage}.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (case_dir / "logs" / f"{stage}.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    match = MAX_RSS_PATTERN.search(completed.stderr or "")
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with {completed.returncode}: {(completed.stderr or completed.stdout)[-2000:]}")
    return {
        "stage": stage,
        "elapsed_seconds": elapsed,
        "maximum_resident_set_bytes": int(match.group(1)) if match else None,
        "returncode": completed.returncode,
    }


def _run_case(case: dict[str, Any], root: Path, pass_name: str, v9: Path, v4: Path) -> dict[str, Any]:
    case_id = str(case["case_id"])
    scan = Path(str(case["scan_path"])).resolve()
    case_dir = root / pass_name / case_id
    base_root = case_dir / "base"
    result_root = base_root / "isolated_cases" / case_id / "result"
    slice_path = case_dir / "slice.json"
    _append_event(
        {
            "event": "DETECTOR_CASE_INVOCATION_STARTED",
            "case_id": case_id,
            "pass": pass_name,
            "detector_invocation_count": 1,
            "scan_sha256": _sha(scan),
        }
    )
    _write_json(
        slice_path,
        {
            "schema_version": 1,
            "target": "opaque_v51_new_untouched_corpus",
            "metric_eligibility": "none_until_independent_drawing_adjudication",
            "entries": [{"part_key": case_id, "scan_path": str(scan), "audit_tags": ["inference_only"]}],
        },
    )
    stages = []
    stages.append(_run("base", [str(PYTHON), str(ROOT / "scripts/run_patch_graph_f1_prototype.py"), "--slice-json", str(slice_path), "--output-dir", str(result_root), "--cache-dir", str(case_dir / "cache"), "--route-mode", "apply", "--strict-blind-inference", "--no-render"], case_dir))
    pool_dir = case_dir / "pool"
    stages.append(_run("pool", [str(PYTHON), str(ROOT / "scripts/build_f1_scan_only_candidate_pool.py"), "--prediction-root", str(base_root), "--output-dir", str(pool_dir), "--case-id", case_id], case_dir))
    pool = pool_dir / f"{case_id}.json"
    v16 = case_dir / "v16.json"
    stages.append(_run("v16", [str(PYTHON), str(ROOT / "scripts/select_f1_mesh_route_v16.py"), "--input", str(pool), "--v9-model", str(v9), "--v4-model", str(v4), "--output", str(v16)], case_dir))
    v43 = case_dir / "v43.json"
    stages.append(_run("v43", [str(PYTHON), str(ROOT / "scripts/run_f1_v43_v16_real_scan_compatibility.py"), "--pool", str(pool), "--retained", str(v16), "--scan", str(scan), "--output", str(v43)], case_dir))
    v44 = case_dir / "v44.json"
    stages.append(_run("v44", [str(PYTHON), str(ROOT / "scripts/run_f1_v44_from_v43.py"), "--v43", str(v43), "--output", str(v44)], case_dir))
    v50 = case_dir / "v50.json"
    stages.append(_run("v50", [str(PYTHON), str(ROOT / "scripts/run_f1_v50_from_v44.py"), "--v44", str(v44), "--pool", str(pool), "--output", str(v50)], case_dir))
    v47 = case_dir / "v47.json"
    stages.append(_run("v47", [str(PYTHON), str(ROOT / "scripts/run_f1_v47_from_v45.py"), "--v45", str(v50), "--pool", str(pool), "--output", str(v47)], case_dir))
    v48 = case_dir / "v48.json"
    stages.append(_run("v48", [str(PYTHON), str(ROOT / "scripts/run_f1_v48_from_v47.py"), "--v47", str(v47), "--output", str(v48)], case_dir))
    v49 = case_dir / "v49.json"
    stages.append(_run("v49", [str(PYTHON), str(ROOT / "scripts/run_f1_v49_from_v48.py"), "--v48", str(v48), "--output", str(v49)], case_dir))
    raw_v51 = case_dir / "v51.raw.json"
    stages.append(_run("v51", [str(PYTHON), str(ROOT / "scripts/run_f1_v51_from_v49.py"), "--v49", str(v49), "--pool", str(pool), "--output", str(raw_v51)], case_dir))
    semantic = canonical_semantic(json.loads(raw_v51.read_text(encoding="utf-8")))
    semantic_path = case_dir / "semantic_result.json"
    _write_json(semantic_path, semantic, compact=True)
    render_path = case_dir / "prediction_only.png"
    stages.append(_run("prediction_render", [str(PYTHON), str(ROOT / "scripts/render_f1_v51_new_corpus_prediction_only.py"), "--scan", str(scan), "--prediction", str(semantic_path), "--case-id", case_id, "--output", str(render_path)], case_dir))
    pool_payload = json.loads(pool.read_text(encoding="utf-8"))
    forbidden = sorted(FORBIDDEN_CACHE_KEYS.intersection(_recursive_keys(pool_payload)))
    if forbidden:
        raise ValueError(f"{case_id}: forbidden cache keys: {forbidden}")
    if int(pool_payload.get("working_point_budget") or 0) != 200000:
        raise ValueError(f"{case_id}: point-budget contract mismatch")
    return {
        "case_id": case_id,
        "pass": pass_name,
        "scan_sha256": _sha(scan),
        "candidate_pool_sha256": _sha(pool),
        "semantic_result_sha256": _sha(semantic_path),
        "prediction_render_sha256": _sha(render_path),
        "point_budget": 200000,
        "forbidden_cache_keys": forbidden,
        "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in stages),
        "peak_stage_rss_bytes": max(int(row["maximum_resident_set_bytes"] or 0) for row in stages),
        "stages": stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opaque-manifest", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("explicit frozen execution token required")
    cases = validate_opaque_manifest(args.opaque_manifest.resolve())
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("one-shot output root is not empty")
    records = []
    for pass_name in ("primary", "replay"):
        for case in cases:
            records.append(_run_case(case, output_root, pass_name, args.v9_model.resolve(), args.v4_model.resolve()))
    primary = {row["case_id"]: row for row in records if row["pass"] == "primary"}
    replay = {row["case_id"]: row for row in records if row["pass"] == "replay"}
    identity = {case_id: primary[case_id]["semantic_result_sha256"] == replay[case_id]["semantic_result_sha256"] for case_id in CASE_IDS}
    if not all(identity.values()):
        raise RuntimeError("primary/replay semantic bytes differ")
    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_v51_new_untouched_corpus_blind_execution",
        "token": TOKEN,
        "truth_opened": False,
        "scoring_opened": False,
        "case_count": 30,
        "primary_success_count": 30,
        "replay_success_count": 30,
        "semantic_replay_identity": identity,
        "records": records,
    }
    _write_json(output_root / "sealed_execution_manifest.json", manifest, compact=True)
    print(json.dumps({"case_count": 30, "replay_identity": True, "manifest": str(output_root / "sealed_execution_manifest.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
