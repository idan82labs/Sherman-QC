import copy
import json

import pytest

import bend_instance_labels as labels


def _payload(job_id="bend_test"):
    return {
        "schema_version": 1,
        "kind": "sherman_qc_bend_instance_labels",
        "job_id": job_id,
        "coordinate_frame": "reference_mesh_raw",
        "review_status": "in_progress",
        "manual_bends": [
            {
                "bend_id": "manual-1",
                "semantic": "conventional_bend",
                "source": "human_two_point",
                "line_start": [0.0, 0.0, 0.0],
                "line_end": [10.0, 2.0, -1.0],
            }
        ],
        "marker_reviews": [
            {"bend_id": "OB1", "decision": "real_bend", "reason": "Visible crease"},
            {"bend_id": "OB2", "decision": "not_bend_outline"},
            {"bend_id": "OB3", "decision": "uncertain", "reason": None},
        ],
    }


def test_empty_payload_is_deterministic():
    assert labels.empty_bend_instance_labels("bend_123") == {
        "schema_version": 1,
        "kind": "sherman_qc_bend_instance_labels",
        "job_id": "bend_123",
        "coordinate_frame": "reference_mesh_raw",
        "review_status": "in_progress",
        "manual_bends": [],
        "marker_reviews": [],
    }


def test_save_and_load_round_trip(tmp_path):
    payload = _payload()

    saved = labels.save_bend_instance_labels(tmp_path, payload, "bend_test")

    assert saved == payload
    assert labels.load_bend_instance_labels(tmp_path, "bend_test") == payload
    assert json.loads((tmp_path / labels.LABELS_FILENAME).read_text()) == payload
    assert not list(tmp_path.glob(f".{labels.LABELS_FILENAME}.*.tmp"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("kind", "other", "kind"),
        ("job_id", "different", "does not match"),
        ("coordinate_frame", "scan_raw", "coordinate_frame"),
        ("review_status", "pending", "review_status"),
        ("manual_bends", None, "manual_bends must be an array"),
        ("marker_reviews", None, "marker_reviews must be an array"),
    ],
)
def test_rejects_invalid_document_contract(field, value, message):
    payload = _payload()
    payload[field] = value

    with pytest.raises(labels.BendInstanceLabelsValidationError, match=message):
        labels.validate_bend_instance_labels(payload, "bend_test")


def test_rejects_duplicate_ids_in_each_collection():
    payload = _payload()
    payload["manual_bends"].append(copy.deepcopy(payload["manual_bends"][0]))
    with pytest.raises(labels.BendInstanceLabelsValidationError, match="duplicate manual bend_id"):
        labels.validate_bend_instance_labels(payload, "bend_test")

    payload = _payload()
    payload["marker_reviews"].append(copy.deepcopy(payload["marker_reviews"][0]))
    with pytest.raises(labels.BendInstanceLabelsValidationError, match="duplicate marker review bend_id"):
        labels.validate_bend_instance_labels(payload, "bend_test")


def test_rejects_noncanonical_whitespace_in_ids():
    payload = _payload()
    payload["manual_bends"][0]["bend_id"] = " manual-1 "
    with pytest.raises(labels.BendInstanceLabelsValidationError, match="whitespace"):
        labels.validate_bend_instance_labels(payload, "bend_test")

    payload = _payload(" bend_test ")
    with pytest.raises(labels.BendInstanceLabelsValidationError, match="whitespace"):
        labels.validate_bend_instance_labels(payload, "bend_test")


@pytest.mark.parametrize(("field", "value"), [("semantic", "outline"), ("source", "model")])
def test_rejects_wrong_or_missing_manual_bend_provenance(field, value):
    payload = _payload()
    payload["manual_bends"][0][field] = value
    with pytest.raises(labels.BendInstanceLabelsValidationError, match=field):
        labels.validate_bend_instance_labels(payload, "bend_test")

    payload = _payload()
    del payload["manual_bends"][0][field]
    with pytest.raises(labels.BendInstanceLabelsValidationError, match=field):
        labels.validate_bend_instance_labels(payload, "bend_test")


@pytest.mark.parametrize(
    ("container", "extra_key"),
    [("top", "notes"), ("manual", "confidence"), ("review", "reviewer")],
)
def test_rejects_unexpected_keys(container, extra_key):
    payload = _payload()
    if container == "top":
        target = payload
    elif container == "manual":
        target = payload["manual_bends"][0]
    else:
        target = payload["marker_reviews"][0]
    target[extra_key] = "unexpected"

    with pytest.raises(labels.BendInstanceLabelsValidationError, match="unexpected keys"):
        labels.validate_bend_instance_labels(payload, "bend_test")


def test_allows_only_string_optional_timestamps():
    payload = _payload()
    payload["created_at"] = "2026-07-15T18:00:00Z"
    payload["updated_at"] = "2026-07-15T18:01:00Z"
    assert labels.validate_bend_instance_labels(payload, "bend_test") == payload

    payload["updated_at"] = None
    with pytest.raises(labels.BendInstanceLabelsValidationError, match="updated_at"):
        labels.validate_bend_instance_labels(payload, "bend_test")


def test_accepts_complete_review_status():
    payload = _payload()
    payload["review_status"] = "complete"
    assert labels.validate_bend_instance_labels(payload, "bend_test") == payload


def test_invalid_collection_enum_types_raise_validation_errors():
    payload = _payload()
    payload["review_status"] = ["complete"]
    with pytest.raises(labels.BendInstanceLabelsValidationError, match="review_status"):
        labels.validate_bend_instance_labels(payload, "bend_test")

    payload = _payload()
    payload["marker_reviews"][0]["decision"] = ["real_bend"]
    with pytest.raises(labels.BendInstanceLabelsValidationError, match="decision"):
        labels.validate_bend_instance_labels(payload, "bend_test")


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ([0.0, 1.0], [1.0, 1.0, 1.0]),
        ([0.0, float("nan"), 0.0], [1.0, 1.0, 1.0]),
        ([0.0, 0.0, 0.0], [1.0, float("inf"), 1.0]),
        ([False, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ([2.0, 3.0, 4.0], [2.0, 3.0, 4.0]),
    ],
)
def test_rejects_invalid_or_zero_length_manual_lines(start, end):
    payload = _payload()
    payload["manual_bends"][0]["line_start"] = start
    payload["manual_bends"][0]["line_end"] = end

    with pytest.raises(labels.BendInstanceLabelsValidationError):
        labels.validate_bend_instance_labels(payload, "bend_test")


def test_rejects_invalid_marker_review_decision_and_reason():
    payload = _payload()
    payload["marker_reviews"][0]["decision"] = "outline"
    with pytest.raises(labels.BendInstanceLabelsValidationError, match="decision"):
        labels.validate_bend_instance_labels(payload, "bend_test")

    payload = _payload()
    payload["marker_reviews"][0]["reason"] = 42
    with pytest.raises(labels.BendInstanceLabelsValidationError, match="reason"):
        labels.validate_bend_instance_labels(payload, "bend_test")


def test_failed_atomic_replace_preserves_previous_file(tmp_path, monkeypatch):
    previous = _payload()
    labels.save_bend_instance_labels(tmp_path, previous, "bend_test")
    original_bytes = (tmp_path / labels.LABELS_FILENAME).read_bytes()
    replacement = _payload()
    replacement["manual_bends"][0]["line_end"] = [20.0, 3.0, 1.0]

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(labels.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        labels.save_bend_instance_labels(tmp_path, replacement, "bend_test")

    assert (tmp_path / labels.LABELS_FILENAME).read_bytes() == original_bytes
    assert not list(tmp_path.glob(f".{labels.LABELS_FILENAME}.*.tmp"))


def test_save_requires_existing_output_directory(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        labels.save_bend_instance_labels(missing, _payload(), "bend_test")


def test_production_app_registers_label_get_and_put_routes():
    from apps.api.main import create_app

    app = create_app()
    operations = app.openapi()["paths"]["/api/bend-inspection/{job_id}/labels"]
    assert set(operations) == {"get", "put"}
