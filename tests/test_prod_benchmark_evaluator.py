"""C12 benchmark: append-only case registry and the per-bend evaluator (tmp_path only)."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from contracts.production.inspection_result import ShopfloorInspectionResultV1
from domains.production.benchmark import evaluator as ev
from domains.production.benchmark.case_registry import (
    GENESIS_SHA256,
    BenchmarkCaseV1,
    CaseRegistry,
    RegistryError,
    main as registry_main,
)
from domains.production.engine.eval.scorer import CaseRow, CaseScore, CharRow

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "contracts" / "production" / "examples"
B1, B2, B3, B4 = "bu_12380e892df5584bcf79", "bu_9ff0ed344a32452b027e", "bu_6f6f1c8da12918ba6911", "bu_ef72a43a0a975c0e454c"


def _result(name: str = "shopfloor_result_v1.example.json") -> ShopfloorInspectionResultV1:
    return ShopfloorInspectionResultV1.model_validate(json.loads((EXAMPLES / name).read_text()))


def _q(value=None, **kw):
    """A truth quantity; ``snap=(nominal, tol)`` adds the recipe snapshot (tol None = explicit no tolerance)."""
    snap = kw.pop("snap", None)
    doc = {"value": value, "instrument": kw.pop("instrument", "CMM Zeiss Contura (calibrated)"), **kw}
    if snap is not None:
        nominal, tol = snap
        doc.update(nominal=nominal, tol_plus=tol, tol_minus=tol)
    return doc


MM = (30.0, 0.5)     # the example recipe's flange nominal and band
DEG = (90.0, 1.0)    # the example recipe's angle nominal and band


def _row(result: ShopfloorInspectionResultV1, truth: dict, **kw) -> dict:
    doc = {
        "case_id": kw.pop("case_id", "PHYS-0001"),
        "job_id": "line_" + "0" * 32,
        "recipe_sha256": result.recipe.recipe_sha256,
        "scan_sha256": result.scan.sha256,
        "sku": result.sku,
        "sku_family": kw.pop("sku_family", "HAT"),
        "split_group": kw.pop("split_group", "part:SYN-HAT-1"),
        "truth_source": kw.pop("truth_source", "metrology_measured"),
        "truth": truth,
        "registered_by": "tester",
    }
    doc.update(kw)
    return doc


TRUTH = {
    B1: {"flange_a_mm": _q(30.05, uncertainty=0.02, snap=MM), "included_deg": _q(90.25, uncertainty=0.05, snap=DEG)},
    B2: {"included_deg": _q(88.3, uncertainty=0.05, snap=DEG)},  # truth OUT, reported FAIL
    B3: {"flange_a_mm": _q(30.6, uncertainty=0.02, snap=MM)},    # truth OUT, reported REVIEW
    B4: {"flange_a_mm": _q(30.1, uncertainty=0.02, snap=MM),     # truth IN, reported NOT_MEASURED
         "flange_b_mm": _q(30.49, uncertainty=0.02, snap=MM)},   # within U of the limit: INDETERMINATE
}


# ---------------------------------------------------------------------------- registry


def test_registry_appends_chains_and_verifies(tmp_path):
    res = _result()
    reg = CaseRegistry(tmp_path / "bench" / "registry.jsonl")
    a = reg.append(_row(res, TRUTH))
    b = reg.append(_row(res, TRUTH, case_id="PHYS-0002"))
    assert a.prev_row_sha256 == GENESIS_SHA256
    assert b.prev_row_sha256 == a.row_sha256
    assert [r.case_id for r in reg.rows()] == ["PHYS-0001", "PHYS-0002"]
    assert reg.head_sha256() == b.row_sha256
    assert oct((tmp_path / "bench" / "registry.jsonl").stat().st_mode & 0o777) == "0o600"
    assert registry_main(["verify", "--registry", str(reg.path)]) == 0
    with pytest.raises(RegistryError) as e:
        reg.append(_row(res, TRUTH))
    assert e.value.code == "DUPLICATE_CASE"


def test_registry_detects_edits_and_deletions(tmp_path):
    res = _result()
    path = tmp_path / "registry.jsonl"
    reg = CaseRegistry(path)
    for i in range(3):
        reg.append(_row(res, TRUTH, case_id=f"PHYS-{i}"))
    lines = path.read_text().splitlines(keepends=True)
    # an edited truth value
    path.write_text(lines[0] + lines[1].replace("30.05", "30.01") + lines[2])
    with pytest.raises(RegistryError) as e:
        reg.rows()
    assert e.value.code == "CHAIN_BROKEN"
    # a deleted middle row
    path.write_text(lines[0] + lines[2])
    with pytest.raises(RegistryError) as e:
        reg.rows()
    assert e.value.code == "CHAIN_BROKEN"
    # a torn final write
    path.write_text("".join(lines)[:-5])
    with pytest.raises(RegistryError):
        reg.rows()
    assert registry_main(["verify", "--registry", str(path)]) == 2


def test_registry_refuses_custody_paths(tmp_path):
    for bad in (REPO / "data" / "registry.jsonl", REPO / "docs" / "registry.jsonl",
                tmp_path / "HOLDOUT10" / "registry.jsonl", tmp_path / "sealed_truth" / "r.jsonl"):
        with pytest.raises(RegistryError) as e:
            CaseRegistry(bad)
        assert e.value.code == "PATH_FORBIDDEN", bad
    assert not (REPO / "docs" / "registry.jsonl").exists()


def test_registry_truth_source_rules(tmp_path):
    res = _result()
    reg = CaseRegistry(tmp_path / "r.jsonl")
    with pytest.raises(RegistryError) as e:  # metrology needs an uncertainty
        reg.append(_row(res, {B1: {"flange_a_mm": _q(30.0, snap=MM)}}))
    assert e.value.code == "ROW_INVALID"
    # a measured truth needs the recipe snapshot: absent keys are refused, explicit nulls are accepted
    for partial in ({}, {"nominal": 30.0}, {"nominal": 30.0, "tol_plus": 0.5, "tol_minus": 0.5, "x": 0}):
        partial.pop("x", None)
        if len(partial) == 3:
            continue
        with pytest.raises(RegistryError) as e:
            reg.append(_row(res, {B1: {"flange_a_mm": _q(30.0, uncertainty=0.02, **partial)}}))
        assert e.value.code == "ROW_INVALID" and "recipe snapshot" in e.value.message
    with pytest.raises(RegistryError):  # an explicit null tolerance still needs its nominal
        reg.append(_row(res, {B1: {"flange_a_mm": _q(30.0, uncertainty=0.02, nominal=None, tol_plus=None,
                                                    tol_minus=None)}}))
    with pytest.raises(RegistryError):  # synthetic rows obey the same rule
        reg.append(_row(res, {B1: {"flange_a_mm": _q(30.0)}}, truth_source="synthetic"))
    with pytest.raises(RegistryError):  # an operator verdict is not a measurement
        reg.append(_row(res, {B1: {"flange_a_mm": _q(30.0, verdict="IN")}}, truth_source="operator_verdict"))
    with pytest.raises(RegistryError):  # not a bend_uid
        reg.append(_row(res, {"B1": {"flange_a_mm": _q(30.0, uncertainty=0.02, snap=MM)}}))
    with pytest.raises(RegistryError):  # unknown quantity
        reg.append(_row(res, {B1: {"radius_mm": _q(3.0, uncertainty=0.02, snap=MM)}}))
    ok = reg.append(_row(res, {B1: {"flange_a_mm": {"verdict": "OUT", "instrument": "operator gauge"}}},
                         truth_source="operator_verdict"))
    assert ok.truth[B1]["flange_a_mm"].verdict == "OUT"
    no_tol = reg.append(_row(res, {B1: {"flange_a_mm": _q(30.0, uncertainty=0.02, snap=(30.0, None))}},
                             case_id="PHYS-NO-TOL"))
    assert ev.truth_class(no_tol.truth[B1]["flange_a_mm"]) == "UNKNOWN"
    assert len(reg.rows()) == 2  # refused rows were never written
    # the explicit-null snapshot survives the JSON round trip (re-read rows still validate)
    assert reg.rows()[1].truth[B1]["flange_a_mm"].tol_plus is None


def test_registry_tail_truncation_needs_the_recorded_head(tmp_path):
    res = _result()
    path = tmp_path / "registry.jsonl"
    reg = CaseRegistry(path)
    for i in range(3):
        reg.append(_row(res, TRUTH, case_id=f"PHYS-{i}"))
    head = reg.head_sha256()
    assert len(reg.rows(expect_head=head)) == 3
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:2]))  # drop the last row: still a valid chain on its own
    assert len(reg.rows()) == 2
    with pytest.raises(RegistryError) as e:
        reg.rows(expect_head=head)
    assert e.value.code == "HEAD_MISMATCH"
    assert registry_main(["verify", "--registry", str(path)]) == 0
    assert registry_main(["verify", "--registry", str(path), "--expect-head", head]) == 2
    # growth after the recorded head is fine
    path.write_text("".join(lines))
    reg.append(_row(res, TRUTH, case_id="PHYS-9"))
    assert registry_main(["verify", "--registry", str(path), "--expect-head", head]) == 0


def test_registry_accepts_the_flat_survey_shape():
    res = _result()
    flat = {"flange_a_mm": 30.02, "included_deg": 90.1, "instrument": "CMM", "uncertainty": {"mm": 0.02, "deg": 0.05},
            "nominal": {"flange_a_mm": 30.0, "included_deg": 90.0},
            "tolerance": {"flange_a_mm": [0.5, 0.5], "included_deg": None}}
    row = BenchmarkCaseV1.model_validate({**_row(res, {B1: flat}), "registered_at": "2026-10-17T00:00:00.000Z"})
    assert row.truth[B1]["flange_a_mm"].uncertainty == 0.02
    assert row.truth[B1]["included_deg"].uncertainty == 0.05
    assert (row.truth[B1]["flange_a_mm"].nominal, row.truth[B1]["flange_a_mm"].tol_plus) == (30.0, 0.5)
    assert ev.truth_class(row.truth[B1]["flange_a_mm"]) == "IN"
    assert ev.truth_class(row.truth[B1]["included_deg"]) == "UNKNOWN"  # recipe has no angle tolerance
    with pytest.raises(ValueError):  # the flat shape without a snapshot is refused for a measured row
        BenchmarkCaseV1.model_validate({**_row(res, {B1: {"flange_a_mm": 30.0, "instrument": "CMM",
                                                          "uncertainty": 0.02}}),
                                        "registered_at": "2026-10-17T00:00:00.000Z"})
    with pytest.raises(ValueError):  # one scalar uncertainty cannot cover mm and deg
        BenchmarkCaseV1.model_validate({**_row(res, {B1: {"flange_a_mm": 30.0, "included_deg": 90.0,
                                                          "instrument": "CMM", "uncertainty": 0.02}}),
                                        "registered_at": "2026-10-17T00:00:00.000Z"})


# ---------------------------------------------------------------------------- evaluate


def _obs(case: ev.CaseEvaluation) -> dict[str, ev.Observation]:
    return {o.characteristic_id: o for o in case.observations}


def test_evaluate_errors_classes_and_denominators():
    res = _result()
    case = ev.evaluate(res, _row(res, TRUTH, registered_at="2026-10-17T00:00:00.000Z"))
    o = _obs(case)
    a1 = o[f"{B1}/FLANGE_LENGTH/A"]
    assert (a1.truth_class, a1.reported_class) == ("IN", "IN")
    assert a1.error == pytest.approx(30.08 - 30.05)  # reported actual - truth
    assert o[f"{B2}/INCLUDED_ANGLE"].truth_class == "OUT"
    assert o[f"{B2}/INCLUDED_ANGLE"].reported_class == "OUT"
    review = o[f"{B3}/FLANGE_LENGTH/A"]
    assert (review.truth_class, review.reported_class) == ("OUT", "REVIEW")
    assert review.error is None and review.error_measured == pytest.approx(30.45 - 30.6)
    nm = o[f"{B4}/FLANGE_LENGTH/A"]
    assert (nm.reported_class, nm.reason, nm.actual, nm.error) == ("NOT_MEASURED", "EDGE_NOT_SEEN", None, None)
    assert o[f"{B4}/FLANGE_LENGTH/B"].truth_class == "INDETERMINATE"
    assert case.part_truth == "NONCONFORMING"

    agg = ev.aggregate([case])["overall"]
    fl = agg["per_kind"]["FLANGE_LENGTH"]
    # decided: B1/A (+0.03) and B4/B (30.07 - 30.49); the REVIEW and the NOT_MEASURED value are out
    assert fl["errors_decided"]["n"] == 2
    assert fl["errors_decided"]["max_abs"] == pytest.approx(0.42)
    assert fl["errors_all_measured"]["n"] == 3
    assert agg["not_measured"] == {"numerator": 1, "denominator": 6, "value": pytest.approx(1 / 6)}
    cr = agg["characteristic_rates"]
    assert cr["missed_out_on_truth_out"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert cr["false_in_on_truth_out"]["numerator"] == 0
    assert cr["false_out_on_truth_in"] == {"numerator": 0, "denominator": 3, "value": 0.0}
    assert agg["characteristic_confusion"]["INDETERMINATE"]["IN"] == 1
    assert agg["part_rates"]["missed_out_on_nonconforming"] == {"numerator": 0, "denominator": 1, "value": 0.0}
    # U calibration over the 5 MEASURED values with truth: B1/A, B1 angle, B2 angle, B3/A (|-0.15| > 0.1), B4/B (0.42)
    assert agg["u_calibration"] == {"numerator": 3, "denominator": 5, "value": 0.6}


def test_evaluate_failed_job_keeps_every_quantity_in_the_denominator():
    res = _result()
    truth = {B1: {"flange_a_mm": _q(30.9, uncertainty=0.02, snap=MM)},
             B2: {"included_deg": _q(90.1, uncertainty=0.05, snap=(90.0, None))}}
    case = ev.evaluate(None, _row(res, truth, registered_at="2026-10-17T00:00:00.000Z"), run_status="WORKER_TIMEOUT")
    o = _obs(case)
    assert {x.reported_class for x in o.values()} == {"NOT_MEASURED"}
    assert {x.reported_state for x in o.values()} == {"RUN_FAILED"}
    assert o[f"{B1}/FLANGE_LENGTH/A"].truth_class == "OUT"  # from the recipe snapshot
    assert o[f"{B2}/INCLUDED_ANGLE"].truth_class == "UNKNOWN"  # no tolerance in the snapshot: never guessed
    agg = ev.aggregate([case])["overall"]
    assert agg["not_measured"] == {"numerator": 2, "denominator": 2, "value": 1.0}
    assert agg["run_status"] == {"WORKER_TIMEOUT": 1}
    assert agg["part_confusion"]["NONCONFORMING"] == {"RUN_FAILED": 1}


def test_truth_class_does_not_depend_on_the_run_outcome(monkeypatch):
    """Mutate-and-check: a failed run, a dropped characteristic or an engine-side tolerance
    edit must not move a truly-OUT quantity or part out of the denominators."""
    res = _result()
    row = _row(res, TRUTH, registered_at="2026-10-17T00:00:00.000Z")
    ok = ev.evaluate(res, row)
    failed = ev.evaluate(None, row, run_status="WORKER_TIMEOUT")
    by_ok = {o.characteristic_id: o.truth_class for o in ok.observations}
    by_failed = {o.characteristic_id: o.truth_class for o in failed.observations}
    assert by_ok == by_failed
    assert ok.part_truth == failed.part_truth == "NONCONFORMING"
    # the engine drops B2's angle (absent) and widens B3's band to 1.0 mm: truth must not move
    real_iter = ev.iter_characteristics

    def mutated_iter(bends, general):
        out = []
        for c in real_iter(bends, general):
            if c.characteristic_id == f"{B2}/INCLUDED_ANGLE":
                continue
            if c.characteristic_id == f"{B3}/FLANGE_LENGTH/A":
                c = c.model_copy(update={"tolerance": c.tolerance.model_copy(update={"plus": 1.0, "minus": 1.0}),
                                         "decision": "PASS"})
            out.append(c)
        return out

    monkeypatch.setattr(ev, "iter_characteristics", mutated_iter)
    mutated = ev.evaluate(res, row)
    mo = _obs(mutated)
    assert mo[f"{B2}/INCLUDED_ANGLE"].reported_state == "ABSENT"
    assert mo[f"{B3}/FLANGE_LENGTH/A"].reported_class == "IN"  # the engine now calls it IN ...
    assert {o.characteristic_id: o.truth_class for o in mutated.observations} == by_ok  # ... truth stays OUT
    assert mutated.part_truth == "NONCONFORMING"
    mrates = ev.aggregate([mutated])["overall"]["characteristic_rates"]
    assert mrates["missed_out_on_truth_out"] == {"numerator": 2, "denominator": 2, "value": 1.0}
    assert mrates["false_in_on_truth_out"]["numerator"] == 1
    # the denominators keep the truly-OUT rows in both runs
    for case in (ok, failed):
        rates = ev.aggregate([case])["overall"]
        assert rates["characteristic_rates"]["missed_out_on_truth_out"]["denominator"] == 2
        assert rates["part_rates"]["missed_out_on_nonconforming"]["denominator"] == 1
    assert ev.aggregate([failed])["overall"]["characteristic_rates"]["missed_out_on_truth_out"]["value"] == 1.0


def test_evaluate_invalid_inspection_and_identity_checks():
    res = _result("shopfloor_result_v1.invalid.example.json")
    case = ev.evaluate(res, _row(res, TRUTH, registered_at="2026-10-17T00:00:00.000Z"))
    assert {o.reported_class for o in case.observations} == {"NOT_MEASURED"}
    assert all(o.error is None and o.error_measured is None for o in case.observations)
    good = _result()
    other = _row(good, TRUTH, registered_at="2026-10-17T00:00:00.000Z", scan_sha256="f" * 64)
    with pytest.raises(ValueError, match="scan sha256"):
        ev.evaluate(good, other)


def test_aggregate_empty_denominators_are_none_and_worst_family():
    res = _result()
    conforming = {B1: {"flange_a_mm": _q(30.06, uncertainty=0.02, snap=MM)}}
    a = ev.evaluate(res, _row(res, conforming, case_id="A", sku_family="L", registered_at="2026-10-17T00:00:00.000Z"))
    b = ev.evaluate(res, _row(res, {B1: {"flange_a_mm": _q(29.70, uncertainty=0.02, snap=MM)}}, case_id="B",
                              sku_family="Z", registered_at="2026-10-17T00:00:00.000Z"))
    rep = ev.aggregate([a, b])
    only_a = rep["by_family"]["L"]
    # no truth OUT in family L: the miss rate is undefined, never a perfect 0
    assert only_a["characteristic_rates"]["missed_out_on_truth_out"] == {"numerator": 0, "denominator": 0, "value": None}
    assert only_a["part_rates"]["false_accept_on_nonconforming"]["value"] is None
    worst = rep["worst_family"]["per_kind.FLANGE_LENGTH.errors_decided.mae"]
    assert worst["family"] == "Z" and worst["value"] == pytest.approx(0.38)
    tie = rep["worst_family"]["not_measured.value"]
    assert tie["family"] is None and tie["note"] == "all evaluable families are equal"
    assert rep["overall"]["cases"] == 2 and set(rep["by_family"]) == {"L", "Z"}


# ---------------------------------------------------------------------------- E4 adapter


def _default(f) -> object:
    t = str(f.type)
    if t.startswith("str") and "None" not in t:
        return "x"
    if t.startswith("bool") and "None" not in t:
        return False
    if t.startswith("int") and "None" not in t:
        return 0
    if t.startswith("tuple"):
        return ()
    return None


def _char(**kw) -> CharRow:
    base = {f.name: _default(f) for f in fields(CharRow)}
    base.update(case_id="E2-X", family="HAT", kind="FLANGE_LENGTH", unit="mm", scorable=True, applicable=True,
                engine_state="MEASURED")
    base.update(kw)
    return CharRow(**base)


def _case(**kw) -> CaseRow:
    base = {f.name: _default(f) for f in fields(CaseRow)}
    base.update(case_id="E2-X", family="HAT", run_status="OK", complete=True, defined_part=True,
                summary_state="ALL_MEASURED_IN_TOLERANCE")
    base.update(kw)
    return CaseRow(**base)


def test_from_e4_score_relabels_only_scorable_rows():
    chars = [
        _char(characteristic_id="c1", decision="PASS", truth=10.0, actual=10.02, error=0.02, truth_in_tolerance=True),
        _char(characteristic_id="c2", decision="REVIEW", truth=10.0, actual=10.4, error=0.4, truth_in_tolerance=False),
        _char(characteristic_id="c3", engine_state="NOT_MEASURED", reason="OCCLUDED", truth=5.0, truth_in_tolerance=True),
        _char(characteristic_id="c4", scorable=False, decision="PASS", error=9.9),  # ill-defined truth: skipped
        _char(characteristic_id="c5", engine_state="ABSENT", truth=1.0, truth_in_tolerance=None),
    ]
    case = ev.from_e4_score(CaseScore(_case(), chars, []))
    o = _obs(case)
    assert set(o) == {"c1", "c2", "c3", "c5"}
    assert o["c1"].error == pytest.approx(0.02)
    assert o["c2"].reported_class == "REVIEW" and o["c2"].error is None and o["c2"].error_measured == 0.4
    assert o["c3"].reported_class == "NOT_MEASURED"
    assert (o["c5"].reported_class, o["c5"].truth_class) == ("NOT_MEASURED", "UNKNOWN")
    assert case.part_truth == "NONCONFORMING"  # c2 is truly out of tolerance

    failed = ev.from_e4_score(CaseScore(_case(run_status="ENGINE_TIMEOUT", summary_state=None), chars, []))
    assert {x.reported_state for x in failed.observations} == {"RUN_FAILED"}
    assert all(x.error is None for x in failed.observations)


def test_from_e4_score_part_truth_rules():
    ok = [_char(characteristic_id="c1", decision="PASS", truth=1.0, actual=1.0, error=0.0, truth_in_tolerance=True)]
    assert ev.from_e4_score(CaseScore(_case(), ok, [])).part_truth == "CONFORMING"
    # a mirrored or wrong part is nonconforming whatever its characteristics say
    assert ev.from_e4_score(CaseScore(_case(mirrored=True), ok, [])).part_truth == "NONCONFORMING"
    assert ev.from_e4_score(CaseScore(_case(wrong_sku=True), ok, [])).part_truth == "NONCONFORMING"
    # an unfinished part is never conforming as a whole
    assert ev.from_e4_score(CaseScore(_case(complete=False), ok, [])).part_truth == "UNDETERMINED"
    green_on_bad = ev.aggregate([ev.from_e4_score(CaseScore(_case(mirrored=True), ok, []))])["overall"]
    assert green_on_bad["part_rates"]["false_accept_on_nonconforming"] == {"numerator": 1, "denominator": 1, "value": 1.0}
