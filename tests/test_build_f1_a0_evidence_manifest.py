from __future__ import annotations

import json

from scripts.build_f1_a0_evidence_manifest import build_manifest, verify_manifest


def test_a0_manifest_is_deterministic_and_hash_sensitive(tmp_path) -> None:
    score = tmp_path / "score.json"
    readiness = tmp_path / "readiness.json"
    other = tmp_path / "other.json"
    score.write_text(json.dumps({"corpus_freeze_id": "freeze"}) + "\n")
    readiness.write_text(json.dumps({"goal_achieved": False}) + "\n")
    other.write_text("{}\n")
    files = {"gold37_score": score, "goal_readiness": readiness, "other": other}

    first = build_manifest(files)
    second = build_manifest(files)
    assert first == second
    assert first["disposition"] == "FALSIFIED_REPRESENTATION_NOT_PROMOTABLE"
    assert first["goal_achieved"] is False
    assert first["artifact_count"] == 3

    old_hash = next(
        row["sha256"] for row in first["artifacts"] if row["artifact_id"] == "other"
    )
    other.write_text('{"changed":true}\n')
    changed = build_manifest(files)
    new_hash = next(
        row["sha256"] for row in changed["artifacts"] if row["artifact_id"] == "other"
    )
    assert new_hash != old_hash
    assert verify_manifest(first) == [
        {
            "artifact_id": "other",
            "path": str(other.resolve()),
            "expected_size_bytes": 3,
            "observed_size_bytes": 17,
            "expected_sha256": old_hash,
            "observed_sha256": new_hash,
        }
    ]
