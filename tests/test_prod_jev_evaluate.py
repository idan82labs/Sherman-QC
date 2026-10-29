"""C10 Jev harness: metrics, frozen operating points, baselines and honesty flags (synthetic)."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import httpx
import pytest

from research.jev import cli
from research.jev import client_v1 as C
from research.jev import evaluate_v1 as E
from research.jev.labels_v1 import LabelRow, make_label, read_jsonl, write_jsonl


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _refuse(*_a, **_k):
        raise AssertionError("network access is forbidden in C10 tests")

    monkeypatch.setattr(httpx.Client, "send", _refuse)


# --------------------------------------------------------------------------
# Pure metrics
# --------------------------------------------------------------------------


def test_metric_values():
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.35, 0.8]
    assert E.auc(y, p) == pytest.approx(0.75)
    assert E.brier(y, p) == pytest.approx((0.01 + 0.16 + 0.4225 + 0.04) / 4)
    expected_ll = -(math.log(0.9) + math.log(0.6) + math.log(0.35) + math.log(0.8)) / 4
    assert E.log_loss(y, p) == pytest.approx(expected_ll)
    assert E.auc([1, 1], [0.2, 0.3]) is None
    # Perfectly calibrated bins give ECE 0; an overconfident model does not.
    assert E.ece([0, 1] * 50, [0.5] * 100) == pytest.approx(0.0)
    assert E.ece([0] * 10, [0.95] * 10) == pytest.approx(0.95)
    assert E.log_loss([1], [0.0]) == pytest.approx(-math.log(E.LOG_EPS))


def test_recall_preserving_threshold_allows_exactly_one_percent_misses():
    p = [i / 100 for i in range(1, 101)]
    t = E.recall_preserving_threshold([1] * 100, p, 0.99)
    assert t == pytest.approx(0.02)  # one positive (0.01) may fall below
    assert sum(1 for x in p if x >= t) / 100 >= 0.99
    assert E.recall_preserving_threshold([1] * 50, p[:50], 0.99) == pytest.approx(0.01)  # 0 misses allowed
    assert E.recall_preserving_threshold([0, 0], [0.1, 0.2]) is None


def test_precision_threshold_and_decision_bands():
    y = [0, 0, 1, 1, 1]
    p = [0.2, 0.6, 0.5, 0.7, 0.9]
    assert E.precision_threshold(y, p, floor=0.5, target=0.99) == pytest.approx(0.7)
    assert E.precision_threshold([0, 0], [0.5, 0.9], floor=0.0) is None
    op = {"prune_below": 0.5, "accept_at": 0.7}
    assert [E.decide(x, op) for x in (0.49, 0.5, 0.69, 0.7)] == ["prune", "review", "review", "accept"]
    assert E.decide(0.99, {"prune_below": 0.5, "accept_at": None}) == "review"


# --------------------------------------------------------------------------
# End to end on the synthetic workspace
# --------------------------------------------------------------------------


@pytest.fixture()
def ws(tmp_path):
    paths = {k: Path(v) for k, v in cli.write_synthetic_workspace(tmp_path / "ws").items()}
    paths["tmp"] = tmp_path
    return paths


def _answers(ws, path: Path, score, **identity) -> Path:
    """Answers rows as live_run writes them (request identity included)."""
    labels = {r["record_id"]: LabelRow.from_json(r) for r in read_jsonl(ws["onboarding_labels"])}
    rows = []
    for r in read_jsonl(ws["onboarding_states"]):
        lab = labels[r["record_id"]]
        ident = C.request_identity(r["state"], model=identity.get("model", "jev-1.13.0"),
                                   variant=identity.get("variant", "v1"),
                                   cad_free=identity.get("cad_free", False))
        rows.append({"record_id": r["record_id"], **ident, "response_model": "jev-1.13.0",
                     "answers": {"is_real_bend": {"type": "noul", "noul": score(lab)}}, "usage": {}})
    write_jsonl(path, rows)
    return path


def _perfect(lab: LabelRow) -> float:
    return 0.95 if lab.label == 1 else 0.05


def _run(ws, name, **kw):
    return E.evaluate(labels=kw.pop("labels", ws["onboarding_labels"]), splits=kw.pop("splits", ws["splits"]),
                      states=ws["onboarding_states"], out_dir=ws["tmp"] / name, bootstrap_rounds=50, **kw)


def test_perfect_answers_score_well_and_report_is_complete(ws):
    report = _run(ws, "e1", answers=_answers(ws, ws["tmp"] / "a.jsonl", _perfect))
    jev = report["scorers"]["jev"]
    assert jev["eval"]["auc"] == pytest.approx(1.0) and not jev["eval"]["in_sample"]
    assert jev["dev"]["in_sample"]
    assert jev["eval"]["n"] == 5 and jev["eval"]["n_positive"] == 4
    assert jev["eval"]["recall_rows"] == 1  # the manual missed bend
    assert jev["eval"]["at_prune_below"]["recall"] == pytest.approx(1.0)
    assert jev["eval"]["at_prune_below"]["recall_end_to_end"] == pytest.approx(0.8)
    assert jev["eval"]["negatives_pruned_fraction"] == pytest.approx(1.0)
    assert jev["eval"]["worst_family_recall"]["family"] == "FAM-C"
    assert report["counts"]["truth_rows"] == 15 and report["counts"]["missing_answers"] == 0
    assert report["response_models"] == ["jev-1.13.0"]
    assert set(report["scorers"]) == {"jev", "engine_confidence", "logistic_regression"}
    assert report["scorer_notes"]["logistic_regression"]["n_fit"] == 5  # dev only
    op = json.loads((ws["tmp"] / "e1" / "operating_point.json").read_text())
    assert op["points"]["jev"]["fit_on"] == ["dev", "calib"] and op["points"]["jev"]["n_fit"] == 10
    assert report["operating_point_sha256"] == op["sha256"]


def test_operating_point_ignores_eval_answers(ws):
    _run(ws, "e1", answers=_answers(ws, ws["tmp"] / "a.jsonl", _perfect))
    labels = {r["record_id"]: LabelRow.from_json(r) for r in read_jsonl(ws["onboarding_labels"])}
    flipped = _answers(ws, ws["tmp"] / "b.jsonl",
                       lambda lab: (0.01 if lab.label == 1 else 0.99) if lab.group_sku == "SYN-ONB-C" else _perfect(lab))
    report = _run(ws, "e2", answers=flipped)
    op1 = json.loads((ws["tmp"] / "e1" / "operating_point.json").read_text())
    op2 = json.loads((ws["tmp"] / "e2" / "operating_point.json").read_text())
    assert op1["points"]["jev"] == op2["points"]["jev"]
    assert report["scorers"]["jev"]["eval"]["auc"] == pytest.approx(0.0)
    assert report["scorers"]["jev"]["eval"]["positives_pruned"] == 4
    assert labels  # used


def test_given_operating_point_is_reused_and_tamper_checked(ws):
    answers = _answers(ws, ws["tmp"] / "a.jsonl", _perfect)
    _run(ws, "e1", answers=answers)
    op_path = ws["tmp"] / "e1" / "operating_point.json"
    report = _run(ws, "e2", answers=answers, operating_point=op_path)
    assert report["operating_point_source"] == "given"
    doc = json.loads(op_path.read_text())
    doc["points"]["jev"]["prune_below"] = 0.0
    tampered = ws["tmp"] / "tampered.json"
    tampered.write_text(json.dumps(doc))
    with pytest.raises(E.EvaluationError, match="sha256"):
        _run(ws, "e3", answers=answers, operating_point=tampered)


def _rewrite(path: Path, out: Path, mutate) -> Path:
    rows = list(read_jsonl(path))
    mutate(rows)
    write_jsonl(out, rows)
    return out


def test_stale_or_foreign_answers_are_refused(ws):
    good = _answers(ws, ws["tmp"] / "a.jsonl", _perfect)
    report = _run(ws, "ok", answers=good)
    assert report["answers_identity"]["question_variant"] == "v1"
    assert report["answers_identity"]["ablation"] == {"cad_free": False, "context_free": False}

    def _body(rows):
        rows[0]["body_sha256"] = "0" * 64

    def _qset(rows):
        rows[0]["question_set_sha256"] = "1" * 64

    def _unknown(rows):
        rows[0]["record_id"] = "f" * 24

    def _dup(rows):
        rows.append(dict(rows[0]))

    def _no_identity(rows):
        for row in rows:
            del row["body_sha256"]

    def _ablation_claim(rows):
        rows[0]["ablation"] = {"cad_free": True, "context_free": False}  # body was not CAD-free

    for name, mutate, match in (("body", _body, "body_sha256"), ("qset", _qset, "question set"),
                                ("unknown", _unknown, "not in the current states"), ("dup", _dup, "duplicate"),
                                ("noid", _no_identity, "request identity"), ("abl", _ablation_claim, "body_sha256")):
        bad = _rewrite(good, ws["tmp"] / f"{name}.jsonl", mutate)
        with pytest.raises(E.EvaluationError, match=match):
            _run(ws, f"x-{name}", answers=bad)
    # A changed state makes every old answer stale.
    states = list(read_jsonl(ws["onboarding_states"]))
    states[0]["state"]["candidate"]["bend_line_length_mm"] = 1.0
    write_jsonl(ws["onboarding_states"], states)
    with pytest.raises(E.EvaluationError, match="body_sha256"):
        _run(ws, "x-state", answers=good)


def test_mixed_answer_identities_are_refused(ws):
    v1 = list(read_jsonl(_answers(ws, ws["tmp"] / "a.jsonl", _perfect)))
    cad_free = list(read_jsonl(_answers(ws, ws["tmp"] / "b.jsonl", _perfect, cad_free=True)))
    mixed = ws["tmp"] / "mixed.jsonl"
    write_jsonl(mixed, v1[:3] + cad_free[3:])
    with pytest.raises(E.EvaluationError, match="mix"):
        _run(ws, "mixed", answers=mixed)
    # Each identity alone is fine and recorded.
    report = _run(ws, "cf", answers=ws["tmp"] / "b.jsonl")
    assert report["answers_identity"]["ablation"]["cad_free"] is True


def test_given_operating_point_must_match_split_manifest_and_identity(ws):
    answers = _answers(ws, ws["tmp"] / "a.jsonl", _perfect)
    _run(ws, "e1", answers=answers)
    op_path = ws["tmp"] / "e1" / "operating_point.json"
    # A different split manifest.
    splits = json.loads(ws["splits"].read_text())
    splits["families"]["FAM-B"]["split"] = "eval"
    other = ws["tmp"] / "splits_other.json"
    other.write_text(json.dumps(splits))
    with pytest.raises(E.EvaluationError, match="split manifest"):
        _run(ws, "e2", answers=answers, splits=other, operating_point=op_path)
    # Answers of another identity (CAD-free bodies).
    cf = _answers(ws, ws["tmp"] / "cf.jsonl", _perfect, cad_free=True)
    with pytest.raises(E.EvaluationError, match="different model, question set"):
        _run(ws, "e3", answers=cf, operating_point=op_path)
    # Another baseline setting.
    with pytest.raises(E.EvaluationError, match="cad-free-baseline"):
        _run(ws, "e4", answers=answers, operating_point=op_path, cad_free_baseline=True)
    # Editing a bound field without re-hashing is tamper.
    doc = json.loads(op_path.read_text())
    doc["split_manifest_sha256"] = "2" * 64
    tampered = ws["tmp"] / "tampered.json"
    tampered.write_text(json.dumps(doc))
    with pytest.raises(E.EvaluationError, match="sha256"):
        _run(ws, "e5", answers=answers, operating_point=tampered)
    # Frozen without Jev answers, then used with answers: refused.
    _run(ws, "e6")
    with pytest.raises(E.EvaluationError, match="without Jev answers"):
        _run(ws, "e7", answers=answers, operating_point=ws["tmp"] / "e6" / "operating_point.json")
    report = _run(ws, "e8", answers=answers, operating_point=op_path)
    assert report["operating_point_source"] == "given"


def test_weak_labels_never_enter_truth(ws):
    rows = [LabelRow.from_json(r) for r in read_jsonl(ws["onboarding_labels"])]
    # Replace every truth label with "unreviewed + weak 1": nothing is left to evaluate.
    weak_only = []
    for row in rows:
        if row.has_state:
            weak_only.append(make_label(row.record_id, label_source="unreviewed", group_sku=row.group_sku,
                                        weak_label_source="recipe_binding_bound_formed"))
        else:
            weak_only.append(row)
    path = ws["tmp"] / "weak.jsonl"
    write_jsonl(path, [r.to_json() for r in weak_only])
    report = _run(ws, "ew", labels=path, answers=_answers(ws, ws["tmp"] / "a.jsonl", _perfect))
    assert report["counts"]["truth_rows"] == 0
    assert report["scorers"]["jev"]["eval"]["n"] == 0


def test_exploratory_and_in_sample_flags(ws):
    splits = json.loads(ws["splits"].read_text())
    splits["families"]["FAM-B"]["flags"] = ["consumed"]
    splits["families"]["FAM-C"]["split"] = "calib"
    path = ws["tmp"] / "splits2.json"
    path.write_text(json.dumps(splits))
    report = _run(ws, "ef", splits=path)
    assert "exploratory_not_unseen" in report["flags"] and "in_sample_only" in report["flags"]
    assert "jev" not in report["scorers"]  # no answers given: baselines only
    clean = _run(ws, "eg")
    assert clean["flags"] == []


def test_synthetic_only_flag(ws):
    rows = [LabelRow.from_json(r) for r in read_jsonl(ws["onboarding_labels"])]
    synthetic = []
    for row in rows:
        if row.label_source == "operator_confirmed":
            synthetic.append(make_label(row.record_id, label_source="synthetic_truth_real", group_sku=row.group_sku))
        elif row.label_source == "operator_removed_not_a_bend":
            synthetic.append(make_label(row.record_id, label_source="synthetic_truth_not_bend",
                                        group_sku=row.group_sku))
        else:
            synthetic.append(row)
    path = ws["tmp"] / "syn.jsonl"
    write_jsonl(path, [r.to_json() for r in synthetic])
    assert "synthetic_only" in _run(ws, "es", labels=path)["flags"]


def test_flagged_family_is_never_scored(ws):
    splits = json.loads(ws["splits"].read_text())
    splits["families"]["FAM-C"]["flags"] = ["holdout"]
    path = ws["tmp"] / "splits3.json"
    path.write_text(json.dumps(splits))
    report = _run(ws, "eh", splits=path, answers=_answers(ws, ws["tmp"] / "a.jsonl", _perfect))
    assert report["counts"]["by_split"]["eval"] == 0
    assert any("FAM-C" in o for o in report["split_overrides"])


def test_group_bootstrap_has_intervals_with_two_families(ws):
    splits = json.loads(ws["splits"].read_text())
    splits["families"]["FAM-B"]["split"] = "eval"  # FAM-B and FAM-C both in eval
    path = ws["tmp"] / "splits4.json"
    path.write_text(json.dumps(splits))
    report = _run(ws, "eb", splits=path, answers=_answers(ws, ws["tmp"] / "a.jsonl", _perfect))
    boot = report["scorers"]["jev"]["eval"]["bootstrap"]
    assert boot["rounds"] == 50 and boot["group"] == "sku_family"
    lo, hi = boot["pruning_rate_ci95"]
    assert 0.0 <= lo <= hi <= 1.0


def test_logistic_baseline_cad_free_uses_fewer_features(ws):
    full = _run(ws, "l1")["scorer_notes"]["logistic_regression"]
    cad_free = _run(ws, "l2", cad_free_baseline=True)["scorer_notes"]["logistic_regression"]
    assert full["status"] == cad_free["status"] == "FITTED"
    assert cad_free["features"] < full["features"]


def test_evaluate_refuses_a_non_empty_out_dir(ws):
    (ws["tmp"] / "busy").mkdir()
    (ws["tmp"] / "busy" / "x").write_text("x")
    with pytest.raises(E.EvaluationError):
        _run(ws, "busy")


def test_engine_confidence_baseline_on_scan_states(tmp_path):
    paths = {k: Path(v) for k, v in cli.write_synthetic_workspace(tmp_path / "ws").items()}
    scan_dir = tmp_path / "ws" / "scan"
    rows = [LabelRow.from_json(r) for r in read_jsonl(scan_dir / "labels.jsonl")]
    states = {r["record_id"]: r["state"] for r in read_jsonl(scan_dir / "states.jsonl")}
    relabelled = []
    for row in rows:
        st = states[row.record_id]
        if st["candidate_source"] == "SCAN_LEGACY_REPORT":
            real = st["cad_relation"].get("is_bound_to_planned_bend") is True
            relabelled.append(make_label(row.record_id, label_source="human_marker_real_bend" if real
                                         else "human_marker_not_bend_outline", group_sku=row.group_sku))
        else:
            relabelled.append(row)
    labels = tmp_path / "labels.jsonl"
    write_jsonl(labels, [r.to_json() for r in relabelled])
    report = E.evaluate(labels=labels, splits=paths["splits"], states=scan_dir / "states.jsonl",
                        out_dir=tmp_path / "ev", bootstrap_rounds=10)
    eng = report["scorers"]["engine_confidence"]["dev"]
    assert eng["n"] == 8 and report["scorer_notes"]["engine_confidence"]["coverage"] == pytest.approx(1.0)
    assert copy.deepcopy(report) == report
