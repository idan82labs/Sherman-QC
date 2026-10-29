from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_f1_real_scan_affine_resample_v51_v52_v1 import CASE_IDS, validate_inputs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    document = tmp_path / "precommit.md"
    generator = tmp_path / "generator.py"
    v9 = tmp_path / "v9.joblib"
    v4 = tmp_path / "v4.joblib"
    for path, content in ((document, b"visual"), (generator, b"generator"), (v9, b"v9"), (v4, b"v4")):
        path.write_bytes(content)
    rows = []
    for case_id in CASE_IDS:
        source = tmp_path / "public" / case_id / "source.ply"
        audit = tmp_path / "audits" / f"{case_id}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        audit.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(case_id.encode())
        audit.write_bytes((case_id + "audit").encode())
        rows.append({
            "opaque_case_id": case_id,
            "source_file": str(source.relative_to(tmp_path)),
            "source_sha256": _sha(source),
            "source_only_audit_file": str(audit.relative_to(tmp_path)),
            "source_only_audit_sha256": _sha(audit),
        })
    public = tmp_path / "public_manifest.json"
    public.write_text(json.dumps({
        "schema_version": 1,
        "source_only": True,
        "research_only": True,
        "locked_benchmark_eligible": False,
        "detector_execution_authorized": False,
        "cases": rows,
    }))
    seal = tmp_path / "VISUAL_PRECOMMIT.json"
    seal.write_text(json.dumps({
        "decision": "ACCEPT_DEVELOPMENT_ROBUSTNESS_EXECUTION_ONLY",
        "public_manifest_sha256": _sha(public),
        "visual_precommit_document_sha256": _sha(document),
        "generator_sha256": _sha(generator),
        "evaluator_manifest_sha256": "f6ff50099f5da870fb3a58a4cbe5557ce088f0a89a330f23a2b53267e331ab38",
        "v9_model_sha256": _sha(v9),
        "v4_model_sha256": _sha(v4),
        "locked_benchmark_eligible": False,
        "independent_f1_eligible": False,
        "human_gold": False,
        "authorized_case_ids": list(CASE_IDS),
        "authorized_execution": {"v51": True, "v52": True, "passes": ["primary", "replay"], "post_hoc_tuning": False},
    }))
    return public, seal, v9, v4, document, generator


def test_validate_inputs_accepts_exact_development_contract(tmp_path: Path) -> None:
    public, seal, v9, v4, document, generator = _fixture(tmp_path)
    cases = validate_inputs(
        public, seal, v9, v4,
        expected_public_sha256=_sha(public),
        expected_visual_document_sha256=_sha(document),
        expected_generator_sha256=_sha(generator),
        expected_v9_sha256=_sha(v9),
        expected_v4_sha256=_sha(v4),
        visual_document=document,
        generator=generator,
    )
    assert [row["case_id"] for row in cases] == list(CASE_IDS)


def test_validate_inputs_rejects_source_tamper(tmp_path: Path) -> None:
    public, seal, v9, v4, document, generator = _fixture(tmp_path)
    (tmp_path / "public" / "R003" / "source.ply").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="R003: source hash mismatch"):
        validate_inputs(
            public, seal, v9, v4,
            expected_public_sha256=_sha(public),
            expected_visual_document_sha256=_sha(document),
            expected_generator_sha256=_sha(generator),
            expected_v9_sha256=_sha(v9),
            expected_v4_sha256=_sha(v4),
            visual_document=document,
            generator=generator,
        )
