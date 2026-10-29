"""Temporary invented derived records only; no corpus or native imports."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("comparison", Path(__file__).parents[1] / "scripts/score_cad_bend_references_v1.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, partial=False):
    directory = tmp_path / "invented_predictions"
    parts, cases, evidence = [], [], {}
    for i in range(1, 21):
        name = f"case-{i:02d}"
        source = hashlib.sha256(f"invented-{i}".encode()).hexdigest()
        parts.append(dict(part_id=name, source_sha256=source, face_ids=[1, 2, 3],
                          reference_status="partial" if partial else "complete",
                          unknown_face_ids=[3] if partial else [], verified_nonbend_face_ids=[] if partial else [3],
                          reference_bends=[dict(bend_id="b", inner_face_ids=[1], outer_face_ids=[2])]))
        cases.append(dict(sha256=source))
        result = dict(source_sha256=source, extraction_status="REFUSED" if i == 20 else "COMPLETE", extraction_reason="invented refusal")
        evidence[name + "/RESULT.json"] = save(directory / name / "RESULT.json", result)
        if i != 20:
            evidence[name + "/EXTRACTION.json"] = save(directory / name / "EXTRACTION.json", dict(bend_axes=[dict(feature_id="p", face_indices=[1, 2])]))
    candidate_pin = save(directory / "FROZEN.json", dict(cases=cases))
    post_pin = save(directory / "POST_AUDIT.json", dict(decision="ACCEPT_DIAGNOSTIC_WITH_LIMITATIONS", independent_non_author=True,
                    blocking_findings=[], frozen_sha256=candidate_pin, evidence=dict(derived_artifact_sha256=evidence)))
    reference = tmp_path / "invented_reference.json"
    reference_pin = save(reference, dict(parts=parts, reference_frozen=True, independent_reviews_complete=True))
    return reference, reference_pin, directory, post_pin


def run(tmp_path, partial=False):
    reference, reference_pin, directory, post_pin = fixture(tmp_path, partial)
    out = tmp_path / "comparison"
    module.freeze(out, reference, reference_pin, directory, post_pin)
    frozen_pin = hashlib.sha256((out / "FROZEN.json").read_bytes()).hexdigest()
    module.execute(out, frozen_pin)
    return out, frozen_pin, json.loads((out / "SCORE.json").read_text())


def test_complete_twenty_case_failure_retained_and_execution_exclusive(tmp_path):
    out, pin, score = run(tmp_path)
    assert score["coverage"]["total_parts"] == 20
    assert score["full_reference_micro"]["tp"] == 19
    assert score["full_reference_micro"]["fn"] == 1
    assert score["parts"][-1]["prediction_status"] == "failed"
    assert score["whole_corpus_fully_referenced"] is True
    with pytest.raises(FileExistsError):
        module.execute(out, pin)


def test_partial_never_claims_whole_corpus_precision_f1(tmp_path):
    _, _, score = run(tmp_path, partial=True)
    assert score["whole_corpus_fully_referenced"] is False
    assert score["full_reference_micro"]["f1"] is None
    assert score["all_verified_positive_recall"]["precision"] is None
    assert score["all_verified_positive_recall"]["f1"] is None
    assert score["all_verified_positive_recall"]["recall"] == .95


def test_wrong_reference_hash_refuses_before_freeze(tmp_path):
    reference, _, directory, post_pin = fixture(tmp_path)
    out = tmp_path / "comparison"
    with pytest.raises(ValueError, match="hash mismatch"):
        module.freeze(out, reference, "0" * 64, directory, post_pin)
    assert not out.exists()


def test_reference_source_binding_mismatch_refuses(tmp_path):
    reference, _, directory, post_pin = fixture(tmp_path)
    refs = json.loads(reference.read_text())
    refs["parts"][4]["source_sha256"] = "0" * 64
    pin = save(reference, refs)
    with pytest.raises(ValueError, match="input mismatch"):
        module.freeze(tmp_path / "comparison", reference, pin, directory, post_pin)


def test_mutated_prediction_is_terminal_failure_not_score(tmp_path):
    reference, reference_pin, directory, post_pin = fixture(tmp_path)
    out = tmp_path / "comparison"
    module.freeze(out, reference, reference_pin, directory, post_pin)
    pin = hashlib.sha256((out / "FROZEN.json").read_bytes()).hexdigest()
    save(directory / "case-01" / "EXTRACTION.json", dict(bend_axes=[]))
    with pytest.raises(ValueError, match="hash mismatch"):
        module.execute(out, pin)
    assert json.loads((out / "TERMINAL.json").read_text())["status"] == "FAILED"
    assert not (out / "SCORE.json").exists()


def test_freeze_exclusive(tmp_path):
    args = fixture(tmp_path)
    out = tmp_path / "comparison"
    module.freeze(out, *args)
    with pytest.raises(FileExistsError):
        module.freeze(out, *args)


@pytest.mark.parametrize("pin", [None, "", "0" * 63, "G" * 64])
def test_missing_or_malformed_hash_refuses_before_io(tmp_path, pin):
    with pytest.raises(ValueError, match="Exact SHA-256 pin required"):
        module.read(tmp_path / "does-not-exist", pin)


def test_duplicate_corpus_source_hash_refuses(tmp_path):
    reference, _, directory, _ = fixture(tmp_path)
    candidate = json.loads((directory / "FROZEN.json").read_text())
    candidate["cases"][1]["sha256"] = candidate["cases"][0]["sha256"]
    candidate_pin = save(directory / "FROZEN.json", candidate)
    post = json.loads((directory / "POST_AUDIT.json").read_text())
    post["frozen_sha256"] = candidate_pin
    post_pin = save(directory / "POST_AUDIT.json", post)
    reference_pin = hashlib.sha256(reference.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Complete 20-case census required"):
        module.freeze(tmp_path / "comparison", reference, reference_pin, directory, post_pin)
