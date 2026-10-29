#!/usr/bin/env python3
"""Run frozen V51/V52 on the visually accepted real-scan robustness lane.

This execution is development-only.  It deliberately cannot read the
evaluator-only pseudo-truth and may not be used as locked or independent F1
evidence.  Primary/replay semantic bytes must match before prediction-only
native-resolution audits are rendered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.f1_cylindrical_consensus_recovery_v52 import compose_v52
import scripts.run_f1_v51_calibrated_raw_synthetic_v1 as v51_wrapper
from scripts.render_f1_source_only_prediction_audit import render as render_source_only_audit


TOKEN = "REAL_SCAN_AFFINE_RESAMPLE_ROBUSTNESS_V1_DEVELOPMENT_ONLY"
CASE_IDS = tuple(f"R{index:03d}" for index in range(1, 9))
PUBLIC_MANIFEST_SHA256 = "d168b46852d9ee19e72b48eed9f0dc3069bab8385b5c0b52229591925d881b31"
VISUAL_DOCUMENT_SHA256 = "f9b161a711f9fdaea8ad50d88d86ce502bc5ca020ac89426368431ca91d1df60"
GENERATOR_SHA256 = "d14ada2bfa256284c651876ff1f4742f35ef618a4d7ca31fbc4ec5786ef76ecb"
EVALUATOR_MANIFEST_SHA256 = "f6ff50099f5da870fb3a58a4cbe5557ce088f0a89a330f23a2b53267e331ab38"
V9_MODEL_SHA256 = "ac48c8b9ed48d352ca5a125137a6128a73f793cef012c967eb3a998e460a39ae"
V4_MODEL_SHA256 = "412946c78a63d8550f9abf1d476c94dcf01b7ed1215cfef28053a6dfd7c966f1"
SANDBOX = ROOT / "research/f1_real_scan_affine_resample_robustness_detector.sb"
VISUAL_DOCUMENT = ROOT / "docs/research/2026-08-15_f1_real_scan_affine_resample_robustness_v1_visual_precommit.md"
GENERATOR = ROOT / "scripts/generate_f1_real_scan_affine_resample_robustness_v1.py"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def validate_inputs(
    public_manifest_path: Path,
    precommit_path: Path,
    v9_model: Path,
    v4_model: Path,
    *,
    expected_public_sha256: str = PUBLIC_MANIFEST_SHA256,
    expected_visual_document_sha256: str = VISUAL_DOCUMENT_SHA256,
    expected_generator_sha256: str = GENERATOR_SHA256,
    expected_v9_sha256: str = V9_MODEL_SHA256,
    expected_v4_sha256: str = V4_MODEL_SHA256,
    visual_document: Path = VISUAL_DOCUMENT,
    generator: Path = GENERATOR,
) -> list[dict[str, str]]:
    public_manifest_path = public_manifest_path.resolve()
    precommit_path = precommit_path.resolve()
    if _sha(public_manifest_path) != expected_public_sha256:
        raise ValueError("public manifest hash mismatch")
    if _sha(visual_document.resolve()) != expected_visual_document_sha256:
        raise ValueError("visual precommit document hash mismatch")
    if _sha(generator.resolve()) != expected_generator_sha256:
        raise ValueError("generator hash mismatch")
    if _sha(v9_model.resolve()) != expected_v9_sha256:
        raise ValueError("V9 model hash mismatch")
    if _sha(v4_model.resolve()) != expected_v4_sha256:
        raise ValueError("V4 model hash mismatch")

    public = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    if public.get("schema_version") != 1 or public.get("source_only") is not True:
        raise ValueError("public manifest schema/source-only mismatch")
    if public.get("research_only") is not True or public.get("locked_benchmark_eligible") is not False:
        raise ValueError("lane must remain research-only and benchmark-ineligible")
    if public.get("detector_execution_authorized") is not False:
        raise ValueError("generation manifest must remain pre-authorization")
    rows = list(public.get("cases") or [])
    if [str(row.get("opaque_case_id")) for row in rows] != list(CASE_IDS):
        raise ValueError("public manifest case ordering mismatch")

    seal = json.loads(precommit_path.read_text(encoding="utf-8"))
    required_seal = {
        "decision": "ACCEPT_DEVELOPMENT_ROBUSTNESS_EXECUTION_ONLY",
        "public_manifest_sha256": expected_public_sha256,
        "visual_precommit_document_sha256": expected_visual_document_sha256,
        "generator_sha256": expected_generator_sha256,
        "evaluator_manifest_sha256": EVALUATOR_MANIFEST_SHA256,
        "v9_model_sha256": expected_v9_sha256,
        "v4_model_sha256": expected_v4_sha256,
        "locked_benchmark_eligible": False,
        "independent_f1_eligible": False,
        "human_gold": False,
    }
    for key, expected in required_seal.items():
        if seal.get(key) != expected:
            raise ValueError(f"visual precommit seal mismatch: {key}")
    if list(seal.get("authorized_case_ids") or []) != list(CASE_IDS):
        raise ValueError("visual precommit did not authorize exact case set")
    authorization = seal.get("authorized_execution") or {}
    if authorization.get("v51") is not True or authorization.get("v52") is not True:
        raise ValueError("V51/V52 execution is not authorized")
    if list(authorization.get("passes") or []) != ["primary", "replay"]:
        raise ValueError("primary/replay authorization mismatch")
    if authorization.get("post_hoc_tuning") is not False:
        raise ValueError("post-hoc tuning must remain forbidden")

    root = public_manifest_path.parent
    cases: list[dict[str, str]] = []
    for row in rows:
        case_id = str(row["opaque_case_id"])
        scan = (root / str(row["source_file"])).resolve()
        audit = (root / str(row["source_only_audit_file"])).resolve()
        if scan.name != "source.ply" or scan.parent.name != case_id:
            raise ValueError(f"{case_id}: non-opaque source layout")
        if _sha(scan) != row.get("source_sha256"):
            raise ValueError(f"{case_id}: source hash mismatch")
        if _sha(audit) != row.get("source_only_audit_sha256"):
            raise ValueError(f"{case_id}: pre-execution visual audit hash mismatch")
        cases.append({"case_id": case_id, "scan_path": str(scan), "scan_sha256": str(row["source_sha256"])})
    return cases


def _configure_v51(event_path: Path) -> None:
    v51_wrapper.TOKEN = TOKEN
    v51_wrapper.CASE_IDS = CASE_IDS
    v51_wrapper.SANDBOX = SANDBOX
    v51_wrapper._configure_frozen(event_path)


def _compose_v52(v51_root: Path, v52_root: Path) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    records: list[dict[str, Any]] = []
    for pass_name in ("primary", "replay"):
        for case_id in CASE_IDS:
            case_root = v51_root / pass_name / case_id
            v51_path = case_root / "semantic_result.json"
            pool_path = case_root / "pool" / f"{case_id}.json"
            result = compose_v52(
                json.loads(v51_path.read_text(encoding="utf-8")),
                json.loads(pool_path.read_text(encoding="utf-8")),
            )
            output = v52_root / pass_name / case_id / "semantic_result.json"
            _write(output, result)
            records.append({
                "case_id": case_id,
                "pass": pass_name,
                "v51_sha256": _sha(v51_path),
                "pool_sha256": _sha(pool_path),
                "v52_sha256": _sha(output),
                "recovered_candidate_count": result["cylindrical_consensus_recovery_audit"]["recovered_candidate_count"],
            })
    primary = {row["case_id"]: row for row in records if row["pass"] == "primary"}
    replay = {row["case_id"]: row for row in records if row["pass"] == "replay"}
    identity = {case_id: primary[case_id]["v52_sha256"] == replay[case_id]["v52_sha256"] for case_id in CASE_IDS}
    if not all(identity.values()):
        raise RuntimeError("V52 primary/replay semantic bytes differ")
    return records, identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--visual-precommit", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("explicit development robustness token required")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("execution output root is not empty")
    cases = validate_inputs(
        args.public_manifest, args.visual_precommit, args.v9_model.resolve(), args.v4_model.resolve()
    )
    v51_root, v52_root = output_root / "v51", output_root / "v52"
    _configure_v51(output_root / "invocation_events.jsonl")
    v51_records: list[dict[str, Any]] = []
    for pass_name in ("primary", "replay"):
        for case in cases:
            v51_records.append(v51_wrapper.frozen._run_case(
                case, v51_root, pass_name, args.v9_model.resolve(), args.v4_model.resolve()
            ))
    v51_primary = {row["case_id"]: row for row in v51_records if row["pass"] == "primary"}
    v51_replay = {row["case_id"]: row for row in v51_records if row["pass"] == "replay"}
    v51_identity = {
        case_id: v51_primary[case_id]["semantic_result_sha256"] == v51_replay[case_id]["semantic_result_sha256"]
        for case_id in CASE_IDS
    }
    if not all(v51_identity.values()):
        raise RuntimeError("V51 primary/replay semantic bytes differ")

    v52_records, v52_identity = _compose_v52(v51_root, v52_root)
    audit_records = []
    scan_by_case = {row["case_id"]: Path(row["scan_path"]) for row in cases}
    for case_id in CASE_IDS:
        output = output_root / "prediction-only-audits" / f"{case_id}.png"
        metadata = render_source_only_audit(
            geometry=scan_by_case[case_id],
            prediction_path=v52_root / "primary" / case_id / "semantic_result.json",
            output=output,
            max_points=250_000,
        )
        metadata_path = output.with_suffix(".json")
        _write(metadata_path, metadata)
        audit_records.append({
            "case_id": case_id,
            "prediction_only_audit_sha256": _sha(output),
            "prediction_only_metadata_sha256": _sha(metadata_path),
        })

    manifest = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_real_scan_affine_resample_v51_v52_execution_v1",
        "token": TOKEN,
        "evidence_class": "development_only_non_confirmatory",
        "locked_benchmark_eligible": False,
        "independent_f1_eligible": False,
        "truth_opened_by_detector": False,
        "pseudo_truth_opened_before_visual_audit": False,
        "post_hoc_tuning_authorized": False,
        "case_ids": list(CASE_IDS),
        "v51_semantic_replay_identity": v51_identity,
        "v52_semantic_replay_identity": v52_identity,
        "v51_records": v51_records,
        "v52_records": v52_records,
        "prediction_only_audits": audit_records,
    }
    _write(output_root / "execution_manifest.json", manifest)
    print(json.dumps({
        "case_count": len(CASE_IDS),
        "v51_replay_identity": True,
        "v52_replay_identity": True,
        "v52_primary_recovered": sum(row["recovered_candidate_count"] for row in v52_records if row["pass"] == "primary"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
