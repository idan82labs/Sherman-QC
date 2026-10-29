from __future__ import annotations

import hashlib

import pytest

from research.f1_physical_intake_v2 import IntakeError, prepare
from scripts.prepare_f1_physical_intake_v2 import _consumed_hashes


def _ply(seed: int) -> bytes:
    return (
        "ply\nformat ascii 1.0\n"
        "element vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
        f"comment seed {seed}\nend_header\n{seed} 0 0\n"
    ).encode()


def _payload(tmp_path, case_count: int = 2, family_count: int = 1):
    rows = []
    for index in range(case_count):
        path = tmp_path / f"source_{index}.ply"
        path.write_bytes(_ply(index))
        rows.append({
            "relative_path": path.name,
            "morphology_group_sha256": f"{1000 + index % family_count:064x}",
            "manufacturing_instance_sha256": f"{2000 + index:064x}",
            "scan_session_sha256": f"{3000 + index:064x}",
            "calibration_sha256": f"{4000 + index:064x}",
            "units": "mm",
            "coverage_reviewed": True,
            "quality_reviewed": True,
            "relationship_to_prior_design": "NOVEL_MORPHOLOGY",
        })
    return {
        "schema": "F1_PHYSICAL_INTAKE_CUSTODIAN_V2",
        "cohort": "UNTOUCHED_QUALIFICATION",
        "custodian_role": "custodian",
        "model_implementer_roles": ["model_team"],
        "blinding_salt_sha256": "a" * 64,
        "acquisition_protocol_sha256": "b" * 64,
        "cases": rows,
    }


def test_small_valid_intake_is_publicly_opaque_and_capacity_pending(tmp_path):
    result = prepare(_payload(tmp_path), intake_root=tmp_path, consumed_scan_hashes=set())
    public = result["public_manifest"]
    private = result["custodian_receipt"]
    assert public["status"] == "INTAKE_ACCEPTED_CAPACITY_PENDING"
    assert private["qualified_two_instance_family_count"] == 1
    assert private["qualifying_independent_instance_count"] == 2
    public_text = str(public)
    assert "source_" not in public_text
    assert "morphology_group_sha256" not in public_text.replace(str(public["identity_fields_absent"]), "")
    assert all(row["vertex_count"] == 1 for row in public["cases"])


def test_rejects_consumed_duplicate_and_forbidden_fields(tmp_path):
    payload = _payload(tmp_path)
    digest = hashlib.sha256((tmp_path / "source_0.ply").read_bytes()).hexdigest()
    with pytest.raises(IntakeError, match="development-consumed"):
        prepare(payload, intake_root=tmp_path, consumed_scan_hashes={digest})
    payload = _payload(tmp_path)
    payload["cases"][0]["expected_count"] = 3
    with pytest.raises(IntakeError, match="forbidden"):
        prepare(payload, intake_root=tmp_path, consumed_scan_hashes=set())


def test_requires_two_independent_instances_per_family_for_ready_gate(tmp_path):
    payload = _payload(tmp_path, case_count=30, family_count=15)
    ready = prepare(payload, intake_root=tmp_path, consumed_scan_hashes=set())
    assert ready["public_manifest"]["status"] == "READY_FOR_CUSTODIAN_SEAL"
    payload = _payload(tmp_path, case_count=30, family_count=15)
    for index in range(15):
        payload["cases"][index + 15]["manufacturing_instance_sha256"] = payload["cases"][index]["manufacturing_instance_sha256"]
    pending = prepare(payload, intake_root=tmp_path, consumed_scan_hashes=set())
    assert pending["public_manifest"]["status"] == "INTAKE_ACCEPTED_CAPACITY_PENDING"
    assert pending["custodian_receipt"]["qualifying_independent_instance_count"] == 0


def test_rejects_bad_ply_and_role_overlap(tmp_path):
    payload = _payload(tmp_path)
    (tmp_path / "source_0.ply").write_bytes(b"not a ply")
    with pytest.raises(IntakeError, match="PLY"):
        prepare(payload, intake_root=tmp_path, consumed_scan_hashes=set())
    payload = _payload(tmp_path)
    payload["model_implementer_roles"] = ["custodian"]
    with pytest.raises(IntakeError, match="may not"):
        prepare(payload, intake_root=tmp_path, consumed_scan_hashes=set())


def test_consumed_hash_registry_supports_external_hash_only_lineage(tmp_path):
    source = tmp_path / "consumed" / "source.ply"
    source.parent.mkdir()
    source.write_bytes(_ply(9))
    on_disk = hashlib.sha256(source.read_bytes()).hexdigest()
    external = "a" * 64

    observed = _consumed_hashes(
        {"intakes": [{
            "root": "consumed",
            "status": "DEVELOPMENT_CONSUMED",
            "scan_sha256s": [external],
        }]},
        tmp_path,
    )

    assert observed == {on_disk, external}
