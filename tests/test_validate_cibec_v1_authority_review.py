from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_cibec_v1_authority_review import validate


def test_unsigned_candidate_remains_blocked() -> None:
    code, report = validate(Path("output/research/cibec-v1-authority-candidate-v2"))
    assert code == 2
    assert report["status"] == "AWAITING_AUTHORITY_REVIEW"
    assert report["fixture_count"] == 14
    assert report["physical_assets_opened"] == 0
    assert report["prediction_assets_opened"] == 0
    assert "MANUFACTURING_AUTHORITY_NAME_REQUIRED" in report["blockers"]
    assert "COUNTERSIGNING_AUTHORITY_NAME_REQUIRED" in report["blockers"]


def test_two_exact_hash_acceptances_clear_review_gate(tmp_path: Path) -> None:
    source = Path("output/research/cibec-v1-authority-candidate-v2")
    pack = tmp_path / "pack"
    pack.mkdir()
    for name in ("CIBEC_V1_CHARTER_CANDIDATE.json", "S01_S14_EXPECTATIONS_CANDIDATE.json", "AUTHORITY_ATTESTATION_TEMPLATE.json"):
        (pack / name).write_bytes((source / name).read_bytes())
    attestation = json.loads((pack / "AUTHORITY_ATTESTATION_TEMPLATE.json").read_text())
    payload_sha = attestation["review_payload_sha256"]
    attestation["manufacturing_authority"] = {"name": "Authority One", "decision": "ACCEPT", "signed_payload_sha256": payload_sha}
    attestation["countersigning_authority"] = {"name": "Authority Two", "decision": "ACCEPT", "signed_payload_sha256": payload_sha}
    (pack / "AUTHORITY_ATTESTATION_TEMPLATE.json").write_text(json.dumps(attestation))
    code, report = validate(pack)
    assert code == 0
    assert report["status"] == "READY_TO_MATERIALIZE_SIGNED_CIBEC"
    assert report["blockers"] == []


def test_edit_without_new_hash_is_rejected(tmp_path: Path) -> None:
    source = Path("output/research/cibec-v1-authority-candidate-v2")
    pack = tmp_path / "pack"
    pack.mkdir()
    for name in ("CIBEC_V1_CHARTER_CANDIDATE.json", "S01_S14_EXPECTATIONS_CANDIDATE.json", "AUTHORITY_ATTESTATION_TEMPLATE.json"):
        (pack / name).write_bytes((source / name).read_bytes())
    charter = json.loads((pack / "CIBEC_V1_CHARTER_CANDIDATE.json").read_text())
    charter = copy.deepcopy(charter)
    charter["decisions"]["COUNTING_UNIT"] += " EDITED"
    (pack / "CIBEC_V1_CHARTER_CANDIDATE.json").write_text(json.dumps(charter))
    code, report = validate(pack)
    assert code == 2
    assert "ATTESTATION_PAYLOAD_HASH_STALE" in report["blockers"]


def test_same_person_cannot_hold_both_authority_roles(tmp_path: Path) -> None:
    source = Path("output/research/cibec-v1-authority-candidate-v2")
    pack = tmp_path / "pack"
    pack.mkdir()
    for name in ("CIBEC_V1_CHARTER_CANDIDATE.json", "S01_S14_EXPECTATIONS_CANDIDATE.json", "AUTHORITY_ATTESTATION_TEMPLATE.json"):
        (pack / name).write_bytes((source / name).read_bytes())
    attestation = json.loads((pack / "AUTHORITY_ATTESTATION_TEMPLATE.json").read_text())
    payload_sha = attestation["review_payload_sha256"]
    signature = {"name": "Same Person", "decision": "ACCEPT", "signed_payload_sha256": payload_sha}
    attestation["manufacturing_authority"] = dict(signature)
    attestation["countersigning_authority"] = dict(signature)
    (pack / "AUTHORITY_ATTESTATION_TEMPLATE.json").write_text(json.dumps(attestation))
    code, report = validate(pack)
    assert code == 2
    assert "AUTHORITY_IDENTITIES_MUST_BE_DISTINCT" in report["blockers"]
