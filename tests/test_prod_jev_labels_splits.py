"""C10 Jev harness: label sidecar semantics and grouped family splits (synthetic only)."""
from __future__ import annotations

import json

import httpx
import pytest

from research.jev import labels_v1 as L
from research.jev import splits_v1 as S
from research.jev.cli import write_synthetic_workspace


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _refuse(*_a, **_k):
        raise AssertionError("network access is forbidden in C10 tests")

    monkeypatch.setattr(httpx.Client, "send", _refuse)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def test_uncertain_marker_review_is_null_and_never_truth():
    row = L.make_label("r1", label_source=L.marker_label_source("uncertain"), group_sku="A")
    assert row.label is None
    assert L.evaluation_truth(row) is None


def test_wrong_location_is_excluded_from_truth():
    row = L.make_label("r1", label_source="operator_wrong_location", group_sku="A")
    assert row.label is None and L.evaluation_truth(row) is None
    with pytest.raises(L.LabelError):
        L.LabelRow("r1", True, "A", 1, "operator_wrong_location", 1.0)


@pytest.mark.parametrize("weak", sorted(L.WEAK_SOURCES))
def test_weak_labels_never_reach_the_truth_column(weak):
    row = L.make_label("r1", label_source="unreviewed", group_sku="A", weak_label_source=weak)
    assert row.weak_label == L.WEAK_SOURCES[weak]
    assert row.label is None
    assert L.evaluation_truth(row) is None
    # A weak source can never be used as the truth source.
    with pytest.raises(L.LabelError):
        L.make_label("r2", label_source=weak, group_sku="A")


def test_truth_sources_have_fixed_labels_and_roundtrip():
    for source, value in L.TRUTH_SOURCES.items():
        row = L.make_label("r", label_source=source, group_sku="A")
        assert L.evaluation_truth(row) == value
        assert L.LabelRow.from_json(json.loads(json.dumps(row.to_json()))) == row
        bad = row.to_json()
        bad["label"] = 1 - value
        with pytest.raises(L.LabelError):
            L.LabelRow.from_json(bad)


def test_recall_rows_have_no_state_and_are_not_candidate_truth():
    row = L.make_label("m1", label_source="operator_manual_missing", group_sku="A", has_state=False)
    assert row.label == 1 and not row.has_state
    assert L.evaluation_truth(row) is None
    with pytest.raises(L.LabelError):
        L.make_label("m2", label_source="operator_manual_missing", group_sku="A", has_state=True)
    with pytest.raises(L.LabelError):
        L.make_label("m3", label_source="operator_confirmed", group_sku="A", has_state=False)


def test_unknown_sources_and_mismatched_weak_labels_refused():
    with pytest.raises(L.LabelError):
        L.make_label("r", label_source="observability_state::FORMED", group_sku="A")
    with pytest.raises(L.LabelError):
        L.LabelRow("r", True, "A", None, "unreviewed", 0.0, weak_label=0,
                   weak_label_source="recipe_binding_bound_formed")


def _marker_doc(**overrides):
    doc = {"schema_version": 1, "kind": "sherman_qc_bend_instance_labels", "job_id": "j",
           "coordinate_frame": "reference_mesh_raw", "review_status": "complete", "manual_bends": [],
           "marker_reviews": [{"bend_id": "CAD_B1", "decision": "real_bend"},
                              {"bend_id": "DET_1", "decision": "not_bend_outline"},
                              {"bend_id": "DET_2", "decision": "uncertain"}]}
    doc.update(overrides)
    return doc


def test_marker_reviews_map_decisions_and_require_the_raw_frame():
    out = L.parse_marker_reviews(_marker_doc())
    assert out == {"CAD_B1": "human_marker_real_bend", "DET_1": "human_marker_not_bend_outline",
                   "DET_2": "human_marker_uncertain"}
    with pytest.raises(L.LabelError, match="frame"):
        L.parse_marker_reviews(_marker_doc(coordinate_frame="cad_model_mm"))
    with pytest.raises(L.LabelError):
        L.parse_marker_reviews(_marker_doc(kind="something"))
    dup = _marker_doc(marker_reviews=[{"bend_id": "X", "decision": "real_bend"}] * 2)
    with pytest.raises(L.LabelError, match="duplicate"):
        L.parse_marker_reviews(dup)
    with pytest.raises(L.LabelError):
        L.parse_marker_reviews(_marker_doc(marker_reviews=[{"bend_id": "X", "decision": "maybe"}]))


def test_candidate_label_doc_frame_and_sources():
    doc = {"schema": L.CANDIDATE_LABEL_DOC, "source": "synthetic_truth", "coordinate_frame": "cad_model_mm",
           "labels": {"e3|bend|x": "real_bend", "e3|crease|y": "not_bend_outline", "e3|crease|z": "uncertain"},
           "confuser_kinds": {"e3|crease|y": "duplicate_skin"}}
    labels, kinds = L.parse_candidate_label_doc(doc, required_frame="cad_model_mm")
    assert labels == {"e3|bend|x": "synthetic_truth_real", "e3|crease|y": "synthetic_truth_not_bend",
                      "e3|crease|z": "unreviewed"}
    assert kinds == {"e3|crease|y": "duplicate_skin"}
    with pytest.raises(L.LabelError, match="frame"):
        L.parse_candidate_label_doc(doc, required_frame="reference_mesh_raw")
    bad = dict(doc, confuser_kinds={"e3|crease|y": "laser_mark"})
    with pytest.raises(L.LabelError):
        L.parse_candidate_label_doc(bad, required_frame="cad_model_mm")


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def _manifest(**families):
    return {"schema": S.SPLIT_MANIFEST_SCHEMA,
            "families": families or {"F1": {"split": "dev"}, "F2": {"split": "eval"},
                                     "H": {"split": "eval", "flags": ["holdout"]},
                                     "R": {"split": "calib", "flags": ["retrospective"]}},
            "sku_to_family": {"A-1": "F1", "a 2": "F1", "B": "F2", "HX": "H", "RX": "R"}}


def test_flagged_families_are_excluded_whatever_the_manifest_says():
    m = S.load_split_manifest(_manifest())
    assert m.split_of("HX") == "excluded"
    assert any("H: eval -> excluded" in o for o in m.overrides)
    for flag in ("sealed", "one_shot"):
        doc = _manifest(F={"split": "dev", "flags": [flag]})
        doc["sku_to_family"] = {"X": "F"}
        assert S.load_split_manifest(doc).split_of("X") == "excluded"


def test_retrospective_family_stays_but_is_exploratory():
    m = S.load_split_manifest(_manifest())
    assert m.split_of("RX") == "calib" and m.is_exploratory("RX")
    assert not m.is_exploratory("B")


def test_unassigned_sku_is_excluded_and_canonicalised_lookup():
    m = S.load_split_manifest(_manifest())
    assert m.split_of("UNKNOWN") == "excluded" and m.split_of(None) == "excluded"
    assert m.split_of("A-2") == "dev"  # "a 2" canonicalises to "A-2"
    assert m.split_of(" a-1 ") == "dev"


def test_manifest_errors():
    doc = _manifest()
    doc["sku_to_family"]["A-1 "] = "F2"  # same canonical SKU, different family
    with pytest.raises(S.SplitError, match="two families"):
        S.load_split_manifest(doc)
    doc = _manifest()
    doc["sku_to_family"]["Z"] = "NOPE"
    with pytest.raises(S.SplitError, match="unknown family"):
        S.load_split_manifest(doc)
    with pytest.raises(S.SplitError):
        S.load_split_manifest(_manifest(F={"split": "train"}))
    with pytest.raises(S.SplitError):
        S.load_split_manifest(_manifest(F={"split": "dev", "flags": ["secret"]}))


def test_no_family_spans_two_splits_on_an_export(tmp_path):
    paths = write_synthetic_workspace(tmp_path / "ws")
    m = S.load_split_manifest(json.loads(open(paths["splits"]).read()))
    rows = [L.LabelRow.from_json(r) for r in L.read_jsonl(paths["onboarding_labels"])]
    rows += [L.LabelRow.from_json(r) for r in L.read_jsonl(tmp_path / "ws" / "scan" / "labels.jsonl")]
    assignment = m.assign(rows)
    by_family: dict[str, set[str]] = {}
    for row in rows:
        by_family.setdefault(m.family_of(row.group_sku) or "UNASSIGNED", set()).add(assignment[row.record_id])
    assert all(len(splits) == 1 for splits in by_family.values()), by_family
    assert {"dev", "calib", "eval", "excluded"} <= set(assignment.values())
    assert by_family["FAM-F1"] == {"excluded"}  # holdout flag
    # Split ids, family names and SKUs never appear in a state.
    texts = [open(paths["onboarding_states"]).read(), open(paths["scan_states"]).read()]
    for text in texts:
        for needle in ("FAM-", "SYN-", '"dev"', '"calib"', '"eval"', "holdout", "split"):
            assert needle not in text, needle
