from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.run_f1_v51_new_untouched_corpus_blind_once import (
    CASE_IDS,
    SEMANTIC_KEYS,
    TOKEN,
    canonical_semantic,
    validate_opaque_manifest,
)


def test_serializer_excludes_runtime_and_inherited_diagnostics() -> None:
    raw = {key: None for key in SEMANTIC_KEYS}
    raw.update(
        {
            "schema_version": 1,
            "kind": "sherman_qc_f1_v51_single_axis_parallel_return_recovery_result",
            "profile": "F1_V51_SINGLE_AXIS_PARALLEL_RETURN_RECOVERY_V1",
            "candidate_count": 0,
            "predictions": [],
            "single_axis_parallel_return_recovery_audit": {},
            "expected_count_used": False,
            "labels_used": False,
            "cad_used": False,
            "part_identity_used": False,
            "source_only": True,
            "research_only": True,
            "v49_retained": True,
            "v51_result_sha256": "a" * 64,
            "elapsed_seconds": 9.5,
            "scan_path": "forbidden.ply",
            "legacy_audit": {"nondeterministic": True},
        }
    )
    result = canonical_semantic(raw)
    assert tuple(result) == SEMANTIC_KEYS
    assert "elapsed_seconds" not in result
    assert "scan_path" not in result
    assert "legacy_audit" not in result


def test_opaque_manifest_accepts_only_exact_order_and_fields(tmp_path: Path) -> None:
    cases = []
    for case_id in CASE_IDS:
        path = tmp_path / f"{case_id}.ply"
        path.write_bytes((case_id + "\n").encode())
        cases.append(
            {
                "case_id": case_id,
                "scan_path": str(path),
                "scan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "token": TOKEN, "cases": cases}))
    assert len(validate_opaque_manifest(manifest)) == 30
