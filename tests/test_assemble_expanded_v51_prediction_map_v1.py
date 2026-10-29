import hashlib
import json
from pathlib import Path

import pytest

from scripts.assemble_expanded_v51_prediction_map_v1 import assemble_prediction_map


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution(tmp_path: Path, name: str, count: int, *, abstain: set[int] | None = None):
    root = tmp_path / name
    abstain = abstain or set()
    records = []
    identities = {}
    for index in range(1, count + 1):
        token = f"T{index:03d}"
        failed = index in abstain
        semantic = (
            {
                "kind": "sherman_qc_v51_fail_closed_case_abstention",
                "pipeline_status": "FAILED_ABSTAIN",
                "predictions": [],
            }
            if failed
            else {"kind": "sherman_qc_v51_semantic_result", "predictions": [{"bend_id": token}]}
        )
        semantic_sha = None
        for branch in ("primary", "replay"):
            case_root = root / branch / token
            current_sha = _write(case_root / "semantic_result.json", semantic)
            semantic_sha = semantic_sha or current_sha
            record = {
                "case_id": token,
                "pass": branch,
                "scan_sha256": hashlib.sha256(f"scan-{token}".encode()).hexdigest(),
                "semantic_result_sha256": current_sha,
                "pipeline_success": not failed,
            }
            if failed:
                record["failure_type"] = "SOURCE_READ_FAILURE"
                record["case_failure_sha256"] = _write(
                    case_root / "case_failure.json",
                    {"case_id": token, "pass": branch, "failure_type": "SOURCE_READ_FAILURE"},
                )
            records.append(record)
        identities[token] = True
    case_ids = [f"T{index:03d}" for index in range(1, count + 1)]
    manifest = {
        "case_count": count,
        "case_ids": case_ids,
        "records": records,
        "semantic_replay_byte_identity": identities,
        "primary_success_count": count - len(abstain),
        "replay_success_count": count - len(abstain),
        "primary_failure_abstention_count": len(abstain),
        "replay_failure_abstention_count": len(abstain),
    }
    manifest_path = tmp_path / f"{name}-development_execution_manifest.json"
    _write(manifest_path, manifest)
    return manifest_path, root


def _fixture(tmp_path: Path, *, abstain_today: set[int] | None = None):
    legacy_catalog = {
        "schema_version": 1,
        "kind": "sherman_qc_bend_line_gold_catalog",
        "cases": [{"case_id": f"legacy/case-{index:03d}"} for index in range(1, 38)],
    }
    catalog_path = tmp_path / "bend_line_gold_catalog_v1.json"
    catalog_sha = _write(catalog_path, legacy_catalog)
    legacy_manifest, legacy_root = _execution(tmp_path, "legacy", 37)
    today_manifest, today_root = _execution(tmp_path, "today", 28, abstain=abstain_today)
    return catalog_path, catalog_sha, legacy_manifest, legacy_root, today_manifest, today_root


def _assemble(fixture):
    catalog, sha, legacy_manifest, legacy_root, today_manifest, today_root = fixture
    return assemble_prediction_map(
        catalog,
        legacy_manifest,
        legacy_root,
        today_manifest,
        today_root,
        expected_legacy_sha256=sha,
    )


def test_assembles_exact_65_key_canonical_mapping_without_mutating_catalog(tmp_path):
    fixture = _fixture(tmp_path)
    before = fixture[0].read_bytes()
    mapping = _assemble(fixture)
    assert len(mapping) == 65
    assert list(mapping) == sorted(mapping)
    assert mapping["legacy/case-001"].endswith("legacy/primary/T001/semantic_result.json")
    assert mapping["legacy/case-037"].endswith("legacy/primary/T037/semantic_result.json")
    assert mapping["T001"].endswith("today/primary/T001/semantic_result.json")
    assert mapping["T028"].endswith("today/primary/T028/semantic_result.json")
    assert fixture[0].read_bytes() == before


def test_permits_hashed_byte_identical_fail_closed_abstention(tmp_path):
    fixture = _fixture(tmp_path, abstain_today={4, 19})
    mapping = _assemble(fixture)
    payload = json.loads(Path(mapping["T004"]).read_text())
    assert payload == {
        "kind": "sherman_qc_v51_fail_closed_case_abstention",
        "pipeline_status": "FAILED_ABSTAIN",
        "predictions": [],
    }


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("identity_false", "semantic_replay_byte_identity"),
        ("missing_record", "record set"),
        ("replay_mismatch", "identity mismatch"),
        ("hash_mismatch", "artifact hash mismatch"),
        ("count_mismatch", "success_count mismatch"),
    ],
)
def test_rejects_incomplete_or_nonidentical_execution(tmp_path, mutation, message):
    fixture = list(_fixture(tmp_path))
    manifest_path, root = fixture[4], fixture[5]
    manifest = json.loads(manifest_path.read_text())
    if mutation == "identity_false":
        manifest["semantic_replay_byte_identity"]["T001"] = False
    elif mutation == "missing_record":
        manifest["records"].pop()
    elif mutation == "replay_mismatch":
        path = root / "replay/T001/semantic_result.json"
        sha = _write(path, {"kind": "sherman_qc_v51_semantic_result", "predictions": []})
        next(row for row in manifest["records"] if row["case_id"] == "T001" and row["pass"] == "replay")["semantic_result_sha256"] = sha
    elif mutation == "hash_mismatch":
        next(row for row in manifest["records"] if row["case_id"] == "T001")["semantic_result_sha256"] = "f" * 64
    else:
        manifest["primary_success_count"] -= 1
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match=message):
        _assemble(tuple(fixture))


def test_rejects_malformed_failure_abstention_sidecar(tmp_path):
    fixture = list(_fixture(tmp_path, abstain_today={2}))
    failure_path = fixture[5] / "primary/T002/case_failure.json"
    failure = json.loads(failure_path.read_text())
    failure["pass"] = "replay"
    _write(failure_path, failure)
    with pytest.raises(ValueError, match="case_failure.json"):
        _assemble(tuple(fixture))


def test_rejects_legacy_catalog_hash_or_key_collision(tmp_path):
    fixture = list(_fixture(tmp_path))
    with pytest.raises(ValueError, match="legacy catalog SHA"):
        assemble_prediction_map(
            fixture[0], fixture[2], fixture[3], fixture[4], fixture[5], expected_legacy_sha256="0" * 64
        )
    catalog = json.loads(fixture[0].read_text())
    catalog["cases"][0]["case_id"] = "T001"
    fixture[1] = _write(fixture[0], catalog)
    with pytest.raises(ValueError, match="collision-free"):
        _assemble(tuple(fixture))
