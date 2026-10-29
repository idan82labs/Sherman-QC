"""Invented review records; no reference artifacts or predictions."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("consensus", Path(__file__).parents[1] / "scripts/freeze_cad_reference_consensus_v1.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, disagree=False, resolve=False):
    parts = [dict(part_id=f"case-{i:02d}", source_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                  inputevidence_sha256=hashlib.sha256(f"evidence-{i}".encode()).hexdigest(),
                  face_ids=[1, 2, 3, 4], unknown_face_ids=[3],
                  reference_status="partial", verified_nonbend_face_ids=[4],
                  reference_bends=[dict(bend_id="b", inner_face_ids=[1], outer_face_ids=[2])]) for i in range(1, 21)]
    second = json.loads(json.dumps(parts))
    for p in second:
        p["evidence_sha256"] = p.pop("inputevidence_sha256")
    second[0]["unknown_face_ids"] = [4]
    second[0]["verified_nonbend_face_ids"] = [3]
    if disagree:
        second[0]["reference_bends"] = []
        second[0]["verified_nonbend_face_ids"] = [1, 2, 3]
    names = ["reviewer-1-initial-valid.json", "reviewer-2.json", "reviewer-1-adjudication.json", "reviewer-2-adjudication.json", "reviewer-1-completeness-final.json"]
    values = [dict(parts=parts), dict(parts=second), dict(decisions=[]),
              dict(nonbend_resolutions=[dict(part_id="case-01", face_ids=[3])] if resolve else [],
                   proposed_complete_parts=[], remaining_partial_parts=[p["part_id"] for p in parts]),
              dict(decisions=[dict(part_id="case-01", accepted_nonbend_face_ids=[3])] if resolve else [],
                   complete_parts=[], remaining_partial_parts=[p["part_id"] for p in parts])]
    return [write(tmp_path / name, value) for name, value in zip(names, values)]


def test_unknown_union_and_exclusive_freeze(tmp_path):
    pins = fixture(tmp_path)
    module.freeze(tmp_path, *pins)
    result = json.loads((tmp_path / "REFERENCE_FROZEN.json").read_text())
    assert len(result["parts"]) == 20
    assert result["parts"][0]["unknown_face_ids"] == [3, 4]
    assert result["parts"][0]["reference_status"] == "partial"
    with pytest.raises(FileExistsError):
        module.freeze(tmp_path, *pins)


def test_agreed_nonbend_only_removes_explicit_resolution(tmp_path):
    module.freeze(tmp_path, *fixture(tmp_path, resolve=True))
    result = json.loads((tmp_path / "REFERENCE_FROZEN.json").read_text())
    assert result["parts"][0]["unknown_face_ids"] == [4]
    assert result["parts"][0]["verified_nonbend_face_ids"] == [3]


def test_positive_disagreement_refuses(tmp_path):
    with pytest.raises(ValueError, match="Positive groups not fully adjudicated"):
        module.freeze(tmp_path, *fixture(tmp_path, disagree=True))


def test_bad_review_hash_refuses(tmp_path):
    pins = fixture(tmp_path)
    pins[0] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        module.freeze(tmp_path, *pins)


def change(tmp_path, pins, filename, index, mutation):
    path = tmp_path / filename
    value = json.loads(path.read_text())
    mutation(value)
    pins[index] = write(path, value)


def test_duplicate_case_refuses(tmp_path):
    pins = fixture(tmp_path)
    change(tmp_path, pins, "reviewer-1-initial-valid.json", 0,
           lambda value: value["parts"][1].update(part_id="case-01"))
    with pytest.raises(ValueError, match="exactly all 20"):
        module.freeze(tmp_path, *pins)


def test_neutral_evidence_hash_disagreement_refuses(tmp_path):
    pins = fixture(tmp_path)
    change(tmp_path, pins, "reviewer-2.json", 1,
           lambda value: value["parts"][0].update(evidence_sha256="0" * 64))
    with pytest.raises(ValueError, match="Neutral evidence hash disagreement"):
        module.freeze(tmp_path, *pins)


def test_missing_explicit_completeness_census_refuses(tmp_path):
    pins = fixture(tmp_path)
    change(tmp_path, pins, "reviewer-1-completeness-final.json", 4,
           lambda value: value.update(remaining_partial_parts=[]))
    with pytest.raises(ValueError, match="Explicit completeness census"):
        module.freeze(tmp_path, *pins)


def test_complete_assertion_with_unknown_witness_refuses(tmp_path):
    pins = fixture(tmp_path)
    def make_complete(value, key):
        value[key] = ["case-01"]
        value["remaining_partial_parts"].remove("case-01")
    change(tmp_path, pins, "reviewer-1-completeness-final.json", 4,
           lambda value: make_complete(value, "complete_parts"))
    change(tmp_path, pins, "reviewer-2-adjudication.json", 3,
           lambda value: make_complete(value, "proposed_complete_parts"))
    with pytest.raises(ValueError, match="Completeness assertion and unknown witness"):
        module.freeze(tmp_path, *pins)
