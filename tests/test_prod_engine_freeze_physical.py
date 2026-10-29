"""E4b physical evaluator: the frozen G2 / G3 rules (amendments 4d-f, 6, 7) and the E3-arm runner.

The gates are exercised on hand-built arm records (no real scan): each one is
shown to PASS on good records and to FAIL (or stay NOT_EVALUABLE) on the defect
it guards against. One integration test runs the real E3 command line on a
synthetic E2 scan through :func:`run_e3_case` (plumbing only, never accuracy).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from domains.production.engine.eval.freeze import physical as ph


def _pop(case_ref: str, n: int, excluded: tuple[str, ...] = (), family: str = "0000") -> dict:
    """Planned bends of one case; repeats of one family share the recipe, hence the bend uids."""
    bends = [{"bend_uid": f"bu_{family[:4]}{i:016d}", "sequence_index": i + 1, "angle_id": f"{case_ref}:A{i}",
              "side_ids": {"A": f"{case_ref}:F{i}A", "B": f"{case_ref}:F{i}B"}, "nominal_included_deg": 90.0}
             for i in range(n)]
    return {"source": "COMPILED", "bends": bends, "rule_excluded_ids": list(excluded)}


def _char(actual: float | None, dev: float | None, state: str = "MEASURED", decision: str = "PASS") -> dict:
    return {"state": state if actual is not None else "NOT_MEASURED", "actual": actual, "deviation": dev,
            "decision": decision if actual is not None else "UNKNOWN", "expected": None if actual is None else
            (actual - dev if dev is not None else None)}


def _e3(pop: dict, *, angle_dev=0.2, flange_dev=0.1, state="FORMED", missing=(), wall=10.0, status="OK",
        fail=(), stage=None) -> dict:
    rows = []
    for i, b in enumerate(pop["bends"]):
        gone = i in missing
        rows.append({"bend_uid": b["bend_uid"], "state": state if not isinstance(state, dict) else state[i],
                     "angle": _char(None if gone else 90.0 + angle_dev, None if gone else angle_dev),
                     "flange": {s: _char(None if gone else 30.0 + flange_dev, None if gone else flange_dev)
                                for s in ("A", "B")},
                     "fail_any": i in fail})
    return {"status": status, "wall_s": wall, "extraction": {"bends": rows, "stage": {"inferred_stage": stage}}}


def _legacy(pop: dict, *, delta=1.0, measured=True) -> dict:
    return {"bends": [{"bend_uid": b["bend_uid"],
                       "angle": {"measured": measured, "legacy_measured_deg": 90.0 + delta if measured else None,
                                 "legacy_delta_vs_e1_deg": delta if measured else None},
                       "edge_distance_proxy": {"measured_mm": 50.0 + delta if measured else None}}
                      for b in pop["bends"]]}


def _case(ref: str, fam: str, lists: list[str], **kw) -> dict:
    return {"case_ref": ref, "family": fam, "lists": lists, **kw}


@pytest.fixture()
def world():
    cases = [_case("pc_aaaa00000001", "11111111", ["G2_COMPLETE", "KNOWN_GOOD:stable", "REPEATABILITY:11111111"]),
             _case("pc_aaaa00000002", "11111111", ["G2_COMPLETE", "KNOWN_GOOD:stable", "REPEATABILITY:11111111"]),
             _case("pc_bbbb00000003", "22222222", ["G2_COMPLETE"]),
             _case("pc_cccc00000004", "33333333", ["G3_STAGED"], stage_class="STAGED_PARTIAL",
                   stage_keys={"expected_completed": 2, "expected_total_bends": 4}),
             _case("pc_cccc00000005", "33333333", ["G3_STAGED"], stage_class="BLANK_STAGE0",
                   stage_keys={"expected_completed": 0})]
    pops = {c["case_ref"]: _pop(c["case_ref"], 4 if c["family"] != "22222222" else 5, family=c["family"])
            for c in cases}
    doc = {"cases": cases, "known_good_families": {"stable": ["11111111"]},
           "repeatability_sets": [{"family": "11111111", "status": "EVALUABLE", "kind": "PHYSICAL_REPEATS",
                                   "case_refs": ["pc_aaaa00000001", "pc_aaaa00000002"]}], "gold_cases": []}
    return doc, pops


def test_g2_passes_on_good_records(world):
    doc, pops = world
    e3 = {r: _e3(p) for r, p in pops.items()}
    e3["pc_aaaa00000002"] = _e3(pops["pc_aaaa00000002"], angle_dev=0.3, flange_dev=0.2)
    legacy = {r: _legacy(p) for r, p in pops.items()}
    legacy["pc_aaaa00000002"] = _legacy(pops["pc_aaaa00000002"], delta=1.8)  # legacy repeats differ by 0.8
    rep = {"rates": {"v10_strict": {"rate": 0.8}, "dev": {"rate": 0.95}}}
    g2 = ph.score_g2(doc, e3=e3, legacy=legacy, pops=pops, compile_report=rep, truth=None, compiled_dirs={})
    gates = g2["gates"]
    assert gates["compile_rate"] == "PASS" and gates["b_angle_coverage"] == "PASS"
    assert gates["c_angle_known_good"] == "PASS" and gates["d_flange_known_good"] == "PASS"
    assert gates["e_runtime_and_failures"] == "PASS" and gates["f_repeatability"] == "PASS"
    assert g2["repeatability"]["e3_p95_angle_deg"] == pytest.approx(0.1)
    assert gates["a_line_metric"] == "NOT_EVALUABLE" and g2["verdict"] == "NOT_EVALUABLE"  # no Gold here


def test_g2_gates_fail_on_their_defects(world):
    doc, pops = world
    legacy = {r: _legacy(p, delta=0.1) for r, p in pops.items()}
    # (b) one of 13 G2 bends unmeasured is 92 % -> PASS; three unmeasured -> FAIL
    e3 = {r: _e3(p) for r, p in pops.items()}
    e3["pc_bbbb00000003"] = _e3(pops["pc_bbbb00000003"], missing=(0, 1, 2))
    g2 = ph.score_g2(doc, e3=e3, legacy=legacy, pops=pops, compile_report=None, truth=None, compiled_dirs={})
    assert g2["coverage"]["angle"]["rate"] == pytest.approx(10 / 13) and g2["gates"]["b_angle_coverage"] == "FAIL"
    # (c) the engine is worse than legacy on the bends both measure
    assert g2["angle_known_good"]["gates"]["no_worse_than_legacy"] == "FAIL"
    # (d) of 16 known-good sides, one 2.5 mm outlier leaves the P95 at 0.7 mm (PASS, still listed for a
    # crop); a second one moves the P95 to 2.5 mm (FAIL). Nothing is excluded.
    e3 = {r: _e3(p) for r, p in pops.items()}
    e3["pc_aaaa00000001"]["extraction"]["bends"][0]["flange"]["A"] = _char(32.5, 2.5)
    g2 = ph.score_g2(doc, e3=e3, legacy=legacy, pops=pops, compile_report=None, truth=None, compiled_dirs={})
    assert g2["flange_known_good"]["p95_abs_mm"] == pytest.approx(0.7) and g2["flange_known_good"]["gate"] == "PASS"
    assert len(g2["flange_known_good"]["large_deviations"]) == 1
    e3["pc_aaaa00000002"]["extraction"]["bends"][1]["flange"]["B"] = _char(32.5, 2.5)
    g2 = ph.score_g2(doc, e3=e3, legacy=legacy, pops=pops, compile_report=None, truth=None, compiled_dirs={})
    assert g2["flange_known_good"]["gate"] == "FAIL" and len(g2["flange_known_good"]["large_deviations"]) == 2
    # (e) a crash fails; an explicit refusal does not; a slow P95 fails
    e3 = {r: _e3(p) for r, p in pops.items()}
    e3["pc_bbbb00000003"]["status"] = "ENGINE_EXIT_2"
    g2 = ph.score_g2(doc, e3=e3, legacy=legacy, pops=pops, compile_report=None, truth=None, compiled_dirs={})
    assert g2["runtime"]["gates"]["crashes"] == "PASS" and len(g2["runtime"]["explicit_refusals"]) == 1
    e3["pc_bbbb00000003"]["status"] = "ENGINE_TIMEOUT"
    g2 = ph.score_g2(doc, e3=e3, legacy=legacy, pops=pops, compile_report=None, truth=None, compiled_dirs={})
    assert g2["runtime"]["gates"]["crashes"] == "FAIL"
    slow = {r: _e3(p, wall=60.0) for r, p in pops.items()}
    assert ph.score_g2(doc, e3=slow, legacy=legacy, pops=pops, compile_report=None, truth=None,
                       compiled_dirs={})["runtime"]["gates"]["p95_wall"] == "FAIL"
    # (f) repeats 0.5 deg apart fail, and are not better than a legacy at 0.0
    e3 = {r: _e3(p) for r, p in pops.items()}
    e3["pc_aaaa00000002"] = _e3(pops["pc_aaaa00000002"], angle_dev=0.7)
    g2 = ph.score_g2(doc, e3=e3, legacy=legacy, pops=pops, compile_report=None, truth=None, compiled_dirs={})
    assert g2["repeatability"]["gates"]["p95_angle"] == "FAIL"
    assert g2["repeatability"]["gates"]["angle_better_than_legacy"] == "FAIL"
    # 4d (rule E4B-4D-GATE-2): the frozen DEV recipes are gated on both denominators; a strict
    # single-source figure below 80 % is a diagnostic, never a gate
    rep = {"rates": {"v10_strict": {"rate": 0.75}, "dev": {"rate": 0.95}},
           "rates_source_assets": {"v10_strict": {"rate": 0.80}, "dev": {"rate": 0.95}}}
    g2 = ph.score_g2(doc, e3=e3, legacy=legacy, pops=pops, compile_report=rep, truth=None, compiled_dirs={})
    assert g2["gates"]["compile_rate"] == "PASS"
    assert g2["compile_rate"]["diagnostics"]["v10_strict"]["meets_floor"] is False
    assert g2["compile_rate"]["rule"]["rule_version"] == "E4B-4D-GATE-2"
    for low in ({**rep, "rates": {**rep["rates"], "dev": {"rate": 0.79}}},
                {**rep, "rates_source_assets": {**rep["rates_source_assets"], "dev": {"rate": 0.75}}}):
        assert ph.score_g2(doc, e3=e3, legacy=legacy, pops=pops, compile_report=low, truth=None,
                           compiled_dirs={})["gates"]["compile_rate"] == "FAIL"


def test_compile_gate_fields_of_a_compile_report():
    rates = {"dev": {"rate": 24 / 26}, "v10_strict": {"rate": 20 / 26}, "native_strict": {"rate": 18 / 26}}
    src = {"dev": {"rate": 19 / 20}, "v10_strict": {"rate": 16 / 20}, "native_strict": {"rate": 14 / 20}}
    f = ph.compile_gate_fields(rates, src)
    assert f["gate"]["verdict"] == "PASS" and f["gate"]["variant"] == "dev"
    assert f["meets_floor"] == {"dev": True, "native_strict": False, "v10_strict": False}
    assert f["meets_floor_source_assets"]["v10_strict"] is True  # 16 / 20 = 0.80 meets the floor
    assert [d["variant"] for d in f["figures"]["diagnostics"]] == ["v10_strict", "native_strict"]
    assert ph.compile_gate_fields({**rates, "dev": {"rate": 0.5}}, src)["gate"]["verdict"] == "FAIL"


def test_rule_declared_unsupported_is_the_only_exclusion(world):
    doc, pops = world
    pops["pc_bbbb00000003"] = _pop("pc_bbbb00000003", 5, excluded=("pc_bbbb00000003:A0", "pc_bbbb00000003:F0A"),
                                   family="22222222")
    e3 = {r: _e3(p) for r, p in pops.items()}
    e3["pc_bbbb00000003"] = _e3(pops["pc_bbbb00000003"], missing=(0,))
    cov = ph._coverage([c for c in doc["cases"] if "G2_COMPLETE" in c["lists"]], e3, pops)
    assert cov["angle"]["rule_excluded"] == 1 and cov["angle"]["planned"] == 12 and cov["angle"]["measured"] == 12
    assert cov["flange_sides"]["planned"] == 25 and cov["flange_sides"]["measured"] == 24  # side B still counts


def _truth(rows):
    return {(r, u): lab for r, u, lab in rows}


def test_g3_rules(world):
    doc, pops = world
    staged, blank = "pc_cccc00000004", "pc_cccc00000005"
    sb, bb = [b["bend_uid"] for b in pops[staged]["bends"]], [b["bend_uid"] for b in pops[blank]["bends"]]
    truth = _truth([(staged, sb[0], "FORMED"), (staged, sb[1], "FORMED"), (staged, sb[2], "NOT_FORMED"),
                    (staged, sb[3], "NOT_DETERMINABLE")] + [(blank, u, "NOT_FORMED") for u in bb])
    e3 = {staged: _e3(pops[staged], state={0: "FORMED", 1: "UNEXPECTED_FORMED", 2: "NOT_YET_FORMED", 3: "UNKNOWN"}),
          blank: _e3(pops[blank], state="NOT_YET_FORMED", stage=0)}
    g3 = ph.score_g3(doc, e3=e3, legacy={}, pops=pops, truth=truth)
    assert g3["state_match_rate"] == 1.0 and g3["gates"]["state_match_rate"] == "PASS"
    assert g3["gates"]["blank_stage0"] == "PASS" and g3["gates"]["unformed_reported_as_defect"] == "PASS"
    # expected_completed 2 of the owner's 4 = recipe 4: comparable, engine formed 2
    assert g3["formed_counts"][0]["expected_formed"] == 2 and g3["gates"]["formed_count_equals_expected_completed"] == "PASS"
    # mutations: an UNKNOWN on a FORMED bend, a FAIL on an unformed bend, a blank decoded as stage 1
    e3b = {staged: _e3(pops[staged], state={0: "UNKNOWN", 1: "FORMED", 2: "NOT_YET_FORMED", 3: "FORMED"}, fail=(2,)),
           blank: _e3(pops[blank], state={0: "FORMED", 1: "NOT_YET_FORMED", 2: "NOT_YET_FORMED",
                                          3: "NOT_YET_FORMED"}, stage=1)}
    g3 = ph.score_g3(doc, e3=e3b, legacy={}, pops=pops, truth=truth)
    assert g3["gates"]["state_match_rate"] == "FAIL" and g3["gates"]["unformed_reported_as_defect"] == "FAIL"
    assert g3["gates"]["blank_stage0"] == "FAIL" and len(g3["unverified_claims"]) == 1  # FORMED on NOT_DETERMINABLE
    assert ph.score_g3(doc, e3=e3, legacy={}, pops=pops, truth=None)["verdict"] == "NOT_EVALUABLE"


def test_expected_completed_is_mapped_onto_the_recipe_inventory_only_when_unambiguous():
    f = ph._expected_formed
    assert f({"stage_keys": {"expected_completed": 4, "expected_total_bends": 16}}, 16) == (4, "owner and recipe inventories agree")
    assert f({"stage_keys": {"expected_completed": 2, "expected_total_bends": 2}}, 1)[0] == 1  # all formed
    assert f({"stage_keys": {"expected_completed": 0, "expected_total_bends": 2}}, 1)[0] == 0
    target, why = f({"stage_keys": {"expected_completed": 1, "expected_total_bends": 2}}, 1)
    assert target is None and why.startswith("NOT_COMPARABLE")
    assert f({"stage_keys": {}}, 3) == (None, None)


def test_truth_table_loader_enforces_two_reviewers_and_tiebreak(tmp_path, world):
    doc, pops = world
    row = {"case_ref": "pc_x", "bend_uid": "bu_" + "1" * 20, "reviewer_1": "FORMED", "reviewer_2": "NOT_FORMED",
           "tiebreak": "FORMED", "final": "FORMED"}
    good = {"schema": ph.TRUTH_TABLE_SCHEMA, "frozen_at": "2026-10-17T12:00:00Z", "rows": [row]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(good))
    assert ph.load_truth_table(p) == {("pc_x", "bu_" + "1" * 20): "FORMED"}
    for bad in ({**good, "frozen_at": None}, {**good, "rows": [{**row, "final": "NOT_FORMED"}]},
                {**good, "rows": [{**row, "tiebreak": None}]}, {**good, "rows": [row, row]},
                {**good, "rows": [{**row, "reviewer_1": "MAYBE"}]}):
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            ph.load_truth_table(p)


def test_provisional_complete_case_moves_to_staged_when_truth_says_unformed(world):
    doc, pops = world
    doc["cases"][2]["provisional_complete"] = True
    uid = pops["pc_bbbb00000003"]["bends"][0]["bend_uid"]
    truth = {("pc_bbbb00000003", b["bend_uid"]): "FORMED" for b in pops["pc_bbbb00000003"]["bends"]}
    e3 = {r: _e3(p) for r, p in pops.items()}
    g2 = ph.score_g2(doc, e3=e3, legacy={}, pops=pops, compile_report=None, truth=truth, compiled_dirs={})
    assert g2["population"]["moved_to_staged_by_truth_table"] == []
    truth[("pc_bbbb00000003", uid)] = "NOT_FORMED"
    g2 = ph.score_g2(doc, e3=e3, legacy={}, pops=pops, compile_report=None, truth=truth, compiled_dirs={})
    assert g2["population"]["moved_to_staged_by_truth_table"] == ["pc_bbbb00000003"]


def test_run_e3_case_plumbing_on_a_synthetic_scan(tmp_path, monkeypatch):
    from domains.production.engine.compile import compile_part
    from domains.production.engine.eval.engine_output import load_recipe
    from domains.production.engine.synthetic.cases import generate_tier

    case_id = "E2-DEV-SMLB-000"
    generate_tier("smoke", "DEV", tmp_path / "tier", case_ids=[case_id])
    cdir = tmp_path / "compiled"
    src = tmp_path / "tier" / case_id
    assert compile_part(load_recipe(src / "recipe.json"), src / "part.step", cdir,
                        flat_path=src / "flat.dxf").status == "COMPILED"
    monkeypatch.setattr(ph, "custody_check", lambda p, purpose: Path(p).resolve())
    from domains.production.engine.eval.plyio import sha256_file

    case = {"case_ref": "pc_e2smoke00001", "scan_sha256": sha256_file(src / "scan.ply")}
    rec = ph.run_e3_case(case, src / "scan.ply", src / "recipe.json", cdir, tmp_path / "work")
    assert rec["schema"] == ph.PHYSICAL_RECORD_SCHEMA and rec["arm"] == "E3"
    assert rec["conditions"]["engine_argv_tail"] == ["-m", "domains.production.engine.inspect"]
    assert rec["conditions"]["engine_timeout_s"] == 180.0
    assert (tmp_path / "work/physical/E3/pc_e2smoke00001/record.json").is_file()
    assert rec["status"] in ("OK", "ENGINE_EXIT_2", "ENGINE_EXIT_3", "RESULT_INVALID", "RESULT_INCONSISTENT")
    if rec["status"] == "OK":
        assert len(rec["extraction"]["bends"]) == 1 and rec["extraction"]["validity"]
    bad = ph.run_e3_case({**case, "scan_sha256": "0" * 64}, src / "scan.ply", src / "recipe.json", cdir,
                         tmp_path / "work")
    assert bad["status"] != "OK"  # a wrong scan hash is refused, never scored


def test_g3_known_good_formed_bends_meet_g2_c_d_and_unformed_bends_are_ignored(world):
    doc, pops = world
    doc["known_good_families"] = {"stable": ["11111111", "33333333"]}
    staged = "pc_cccc00000004"
    uids = [b["bend_uid"] for b in pops[staged]["bends"]]
    truth = {(staged, uids[0]): "FORMED", (staged, uids[1]): "FORMED", (staged, uids[2]): "NOT_FORMED",
             (staged, uids[3]): "NOT_FORMED"}
    truth.update({("pc_cccc00000005", b["bend_uid"]): "NOT_FORMED" for b in pops["pc_cccc00000005"]["bends"]})
    states = {0: "FORMED", 1: "FORMED", 2: "NOT_YET_FORMED", 3: "NOT_YET_FORMED"}
    rec = _e3(pops[staged], state=states)
    # a huge "deviation" on a truly unformed bend must not count (its angle is the full sweep, not an error)
    rec["extraction"]["bends"][2]["angle"] = _char(180.0, 90.0)
    e3 = {staged: rec, "pc_cccc00000005": _e3(pops["pc_cccc00000005"], state="NOT_YET_FORMED", stage=0)}
    g3 = ph.score_g3(doc, e3=e3, legacy={}, pops=pops, truth=truth)
    assert g3["gates"]["formed_bends_known_good_angle_mae"] == "PASS"
    assert g3["formed_bends_known_good_angle"]["planned_bends"] == 2
    rec["extraction"]["bends"][0]["angle"] = _char(93.0, 3.0)
    g3 = ph.score_g3(doc, e3=e3, legacy={}, pops=pops, truth=truth)
    assert g3["gates"]["formed_bends_known_good_angle_mae"] == "FAIL"


def test_a_legacy_arm_that_measured_nothing_has_nothing_to_beat(world):
    doc, pops = world
    e3 = {r: _e3(p) for r, p in pops.items()}
    e3["pc_aaaa00000002"] = _e3(pops["pc_aaaa00000002"], angle_dev=0.3, flange_dev=0.2)
    silent = {r: _legacy(p, measured=False) for r, p in pops.items()}
    g2 = ph.score_g2(doc, e3=e3, legacy=silent, pops=pops, compile_report=None, truth=None, compiled_dirs={})
    assert g2["repeatability"]["gates"]["angle_better_than_legacy"] == "PASS"
    assert g2["repeatability"]["legacy_p95_angle_deg"] is None
    assert g2["angle_known_good"]["gates"]["no_worse_than_legacy"] == "PASS"
    assert g2["angle_known_good"]["gates"]["coverage_at_least_legacy"] == "PASS"
    # but an engine that measures nothing is never "better"
    blind = {r: _e3(p, missing=(0, 1, 2, 3, 4)) for r, p in pops.items()}
    g2 = ph.score_g2(doc, e3=blind, legacy=silent, pops=pops, compile_report=None, truth=None, compiled_dirs={})
    assert g2["repeatability"]["gates"]["angle_better_than_legacy"] == "NOT_EVALUABLE"
    assert g2["angle_known_good"]["gates"]["no_worse_than_legacy"] == "NOT_EVALUABLE"


def test_legacy_baseline_is_computed_without_any_engine(world):
    doc, pops = world
    legacy = {r: _legacy(p, delta=0.5) for r, p in pops.items()}
    legacy["pc_bbbb00000003"] = _legacy(pops["pc_bbbb00000003"], measured=False)
    base = ph.legacy_baseline(doc, legacy=legacy, pops=pops)
    assert base["g2_planned_bends"] == 13 and base["g2_angle_measured"] == 8
    assert base["known_good_angle_measured"] == 8 and base["known_good_mae_vs_e1_deg"] == pytest.approx(0.5)
    assert "line_metric" not in base  # no Gold case in this document


def _gold(cid: str, fam: str, lines: list[tuple[float, float]], **kw) -> dict:
    """Raw-frame BROAD gold: lines along x at heights y (80 mm long)."""
    return {"case_id": cid, "family": fam, "frame": "raw_scan", "raw_to_reference": None, "residual_mm": 0.0,
            "gold_lines": [{"bend_id": f"G{i}", "line_start": [0.0, y, z], "line_end": [80.0, y, z]}
                           for i, (y, z) in enumerate(lines)], **kw}


def _pred(lines: list[tuple[float, float]]) -> list[dict]:
    return [{"bend_id": f"P{i}", "line_start": [1.0, y + 0.5, z], "line_end": [79.0, y + 0.5, z]}
            for i, (y, z) in enumerate(lines)]


def test_broad_family_macro_definition_controls_and_recall_only():
    gold = [_gold("pc_1", "AAAAAAAA", [(0, 0), (40, 0)]), _gold("pc_2", "AAAAAAAA", [(0, 0)]),
            _gold("pc_3", "BBBBBBBB", [(0, 0), (40, 0)]),
            _gold("pc_4", "BBBBBBBB", [], zero_bend_control=True),
            _gold("pc_5", "CCCCCCCC", [(0, 0), (40, 0)], label_complete=False)]
    preds = {"pc_1": _pred([(0, 0), (40, 0)]), "pc_2": _pred([(0, 0), (200, 0)]),  # one FP in family A
             "pc_3": _pred([(0, 0)]),  # one FN in family B; pc_4 control gets one spurious line
             "pc_4": _pred([(10, 10)]), "pc_5": _pred([(0, 0), (300, 0)])}
    out = ph.broad_line_scores(gold, preds)
    fa, fb = out["per_family"]["AAAAAAAA"], out["per_family"]["BBBBBBBB"]
    assert (fa["tp"], fa["fp"], fa["fn"]) == (3, 1, 0) and (fb["tp"], fb["fp"], fb["fn"]) == (1, 0, 1)
    assert out["families"] == 2  # the recall-only family is not in the macro (no precision)
    assert out["family_macro_f1"] == pytest.approx((fa["f1"] + fb["f1"]) / 2)
    assert out["zero_bend_controls"] == {"cases": 1, "false_positive_lines": 1}
    with_ctrl_b = 2 * (1 / 2) * (1 / 2) / (1 / 2 + 1 / 2)
    assert out["family_macro_f1_with_controls"] == pytest.approx((fa["f1"] + with_ctrl_b) / 2)
    rec_only = next(r for r in out["cases"] if r["case_id"] == "pc_5")
    assert rec_only["disposition"] == "SCORED_RECALL_ONLY_PARTIAL_LABEL" and rec_only["false_positive"] is None
    assert out["micro"]["precision"] == pytest.approx(4 / 5) and out["true_positive"] == 4
    missing = ph.broad_line_scores(gold, {})  # no prediction at all: every gold line is a miss
    assert missing["family_macro_f1"] == 0.0 and missing["false_negative"] == 5


def test_broad_report_and_g2a_dual_report(world, monkeypatch):
    doc, pops = world
    doc = {**doc, "cases": doc["cases"] + [_case("pc_dddd00000006", "44444444", ["BROAD"], stage_class="COMPLETE")]}
    pops = {**pops, "pc_dddd00000006": _pop("pc_dddd00000006", 2, family="44444444")}
    doc["broad_gold_cases"] = [_gold("pc_dddd00000006", "44444444", [(0, 0), (40, 0)])]
    e3 = {c["case_ref"]: _e3(pops[c["case_ref"]]) for c in doc["cases"]}
    monkeypatch.setattr(ph, "_engine_line_predictions",
                        lambda ids, e3r, cd: {i: _pred([(0, 0), (40, 0)]) for i in ids})
    lanes = {"lanes": {"runtime": {"pc_dddd00000006": {"lines": _pred([(0, 0)])}}},
             "meta": {"runtime": {"conditions": "broad baseline runtime"}}}
    rep = ph.score_broad(doc, e3=e3, legacy={}, pops=pops, truth=None, compiled_dirs={}, reference_lanes=lanes)
    assert rep["verdict"] == "REPORT_ONLY" and rep["line_metric"]["engine"]["family_macro_f1"] == 1.0
    assert rep["line_metric"]["target_status"] == "MET" and rep["owner_target"]["family_macro_line_f1_min"] == 0.80
    assert rep["line_metric"]["reference_lanes"]["runtime"]["false_negative"] == 1
    assert rep["coverage"]["angle"]["planned"] == 2 and rep["population"]["broad_cases"] == 1
    # G2(a) dual report: the relabel is scored next to the human Gold11, the human set is the gate of record
    doc["gold_cases"] = [_gold("pc_aaaa00000001", "11111111", [(0, 0), (40, 0)])]
    doc["relabel_gold_cases"] = [_gold("pc_aaaa00000001", "11111111", [(0, 0), (40, 0), (80, 0)])]
    doc["relabel_snapshot_sha256"] = "f" * 64
    monkeypatch.setattr(ph, "_line_metric", lambda gold, e3r, cd, lg: {
        "engine": {"f1": 1.0 if len(gold[0]["gold_lines"]) == 2 else 0.8, "precision": 1.0, "case_count": 1},
        "verdict": "PASS" if len(gold[0]["gold_lines"]) == 2 else "FAIL"})
    g2 = ph.score_g2(doc, e3=e3, legacy={}, pops=pops, compile_report=None, truth=None, compiled_dirs={})
    dual = g2["line_metric_dual_report"]
    assert dual["gate_of_record"] == "HUMAN_GOLD11" and g2["gates"]["a_line_metric"] == "PASS"
    assert dual["relabel"]["f1"] == 0.8 and dual["human_on_the_same_cases"]["f1"] == 1.0
    assert dual["relabel_changed_since_freeze"] is None  # nothing re-read in this unit test


def _frozen_table(path: Path, rows: list[tuple[str, str, str]], frozen_at: str = "2026-10-17T12:00:00Z") -> Path:
    path.write_text(json.dumps({"schema": ph.TRUTH_TABLE_SCHEMA, "frozen_at": frozen_at, "rows": [
        {"case_ref": c, "bend_uid": u, "reviewer_1": lab, "reviewer_2": lab, "tiebreak": None, "final": lab}
        for c, u, lab in rows]}), encoding="utf-8")
    return path


def test_truth_relevant_cases_are_the_staged_and_provisional_ones():
    assert ph.truth_relevant({"lists": ["G3_STAGED", "G2_COMPLETE"]})
    assert ph.truth_relevant({"lists": ["BROAD", "BROAD_STAGED"]})
    assert ph.truth_relevant({"lists": ["G2_COMPLETE", "REPEATABILITY:15651004"], "provisional_complete": True})
    assert not ph.truth_relevant({"lists": ["G2_COMPLETE", "KNOWN_GOOD:stable", "BROAD"]})


def test_amendment7_gate_needs_a_frozen_complete_table_recorded_in_the_ledger(tmp_path):
    from domains.production.engine.eval import freeze as e4a

    need = {"pc_s": ["bu_1", "bu_2"], "pc_empty": []}
    gate = ph.truth_table_gate(None, need)
    assert not gate["ok"] and gate["reason"].startswith("NO_FROZEN_TRUTH_TABLE")
    assert ph.truth_table_gate(None, {"pc_empty": []})["ok"]  # nothing to protect: no rows
    part = _frozen_table(tmp_path / "part.json", [("pc_s", "bu_1", "FORMED")])
    assert ph.truth_table_gate(part, need)["reason"].startswith("TRUTH_TABLE_INCOMPLETE")
    full = _frozen_table(tmp_path / "full.json", [("pc_s", "bu_1", "FORMED"), ("pc_s", "bu_2", "NOT_FORMED")])
    assert ph.truth_table_gate(full, need)["ok"]  # no ledger given: frozen and complete is enough
    future = _frozen_table(tmp_path / "future.json", [("pc_s", "bu_1", "FORMED"), ("pc_s", "bu_2", "FORMED")],
                           frozen_at="2999-01-01T00:00:00Z")
    assert ph.truth_table_gate(future, need)["reason"].startswith("TRUTH_TABLE_FROZEN_AT_INVALID")
    ledger = tmp_path / "LEDGER.json"
    e4a.append_entry(ledger, label="pre-A1", items=e4a.freeze_items())
    gate = ph.truth_table_gate(full, need, ledger=ledger)
    assert not gate["ok"] and gate["reason"].startswith("TRUTH_TABLE_NOT_RECORDED_IN_LEDGER")
    e4a.append_entry(ledger, label="truth", items={"files": {"g3_truth_table_A1": {
        "path": str(full), "sha256": ph.sha256_file(full), "bytes": full.stat().st_size}}})
    gate = ph.truth_table_gate(full, need, ledger=ledger)
    assert gate["ok"] and gate["ledger_record"]["item"] == "files:g3_truth_table_A1"
    sealed = _frozen_table(tmp_path / "sealed.json", [("pc_s", "bu_1", "FORMED"), ("pc_s", "bu_2", "FORMED")])
    e4a.append_entry(ledger, label="truth sealed", items={"devcheck_sealed": {
        "g3_truth_table_SEALED": {"sha256": ph.sha256_file(sealed)}}})
    assert ph.ledger_record_of(ledger, ph.sha256_file(sealed))["item"] == "devcheck_sealed:g3_truth_table_SEALED"
    with pytest.raises(ValueError):  # an unfrozen table is an input error
        ph.truth_table_gate(_frozen_table(tmp_path / "draft.json", [("pc_s", "bu_1", "FORMED")], frozen_at=""), need)


def test_scorer_rejects_a_table_frozen_or_recorded_after_engine_output(tmp_path):
    from datetime import datetime, timedelta, timezone

    from domains.production.engine.eval import freeze as e4a

    def at(hours: float) -> str:  # relative to now: the ledger records entries at the real time
        return (datetime.now(timezone.utc) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

    table = _frozen_table(tmp_path / "t.json", [("pc_s", "bu_1", "FORMED")], frozen_at=at(-3))
    early = {"pc_s": {"recorded_at": at(-4)}, "pc_c": {"recorded_at": at(-5)}}
    assert ph.truth_table_order(table, early)["status"] == "REJECTED_FROZEN_AFTER_E3_OUTPUT"
    # an E3 record of a complete (STATE_CHECK) case before the freeze is not engine output on a
    # truth-relevant case: only the relevant records are compared
    mixed = {"pc_s": {"recorded_at": at(-2)}, "pc_c": {"recorded_at": at(-5)}}
    assert ph.truth_table_order(table, mixed)["status"] == "REJECTED_FROZEN_AFTER_E3_OUTPUT"
    assert ph.truth_table_order(table, mixed, relevant={"pc_s"})["status"] == "ACCEPTED"
    ledger = tmp_path / "LEDGER.json"
    e4a.append_entry(ledger, label="pre-A1", items=e4a.freeze_items())
    assert ph.truth_table_order(table, mixed, relevant={"pc_s"}, ledger=ledger)["status"] == \
        "REJECTED_NOT_RECORDED_IN_LEDGER"
    e4a.append_entry(ledger, label="truth", items={"files": {"g3_truth_table_A1": {
        "path": str(table), "sha256": ph.sha256_file(table), "bytes": 1}}})
    # frozen before the E3 record, but recorded in the ledger (now) after it
    assert ph.truth_table_order(table, mixed, relevant={"pc_s"}, ledger=ledger)["status"] == \
        "REJECTED_RECORDED_AFTER_E3_OUTPUT"
    later = {"pc_s": {"recorded_at": at(1)}, "pc_c": {"recorded_at": at(-5)}}
    assert ph.truth_table_order(table, later, relevant={"pc_s"}, ledger=ledger)["status"] == "ACCEPTED"
    assert ph.truth_table_order(table, {}, relevant={"pc_s"}, ledger=ledger)["status"] == "ACCEPTED"
    assert ph.truth_table_order(None, mixed)["status"] == "MISSING"


def test_recipe_line_ceiling_shows_what_the_frozen_gold_allows():
    gold = [_gold("pc_1", "AAAAAAAA", [(0, 0), (40, 0), (80, 0)]), _gold("pc_2", "AAAAAAAA", [(0, 0)]),
            _gold("pc_3", "BBBBBBBB", [(0, 0), (40, 0)]), _gold("pc_4", "BBBBBBBB", [], zero_bend_control=True),
            _gold("pc_5", "CCCCCCCC", [(0, 0)], label_complete=False)]
    pops = {"pc_1": _pop("pc_1", 2), "pc_2": _pop("pc_2", 4), "pc_3": _pop("pc_3", 2), "pc_4": _pop("pc_4", 1),
            "pc_5": _pop("pc_5", 0)}
    out = ph.recipe_line_ceiling(gold, pops, target=0.80)
    fa = out["per_family"]["AAAAAAAA"]
    assert (fa["true_positive_max"], fa["gold_lines"]) == (3, 4) and fa["f1_ceiling"] == pytest.approx(6 / 7)
    assert out["per_family"]["BBBBBBBB"]["f1_ceiling"] == 1.0 and "CCCCCCCC" not in out["per_family"]
    assert [c["case_id"] for c in out["capped_cases"]] == ["pc_1"]
    assert out["family_macro_f1_ceiling"] == pytest.approx((6 / 7 + 1.0) / 2) and out["target_reachable"]
    assert ph.recipe_line_ceiling(gold, {**pops, "pc_3": {"source": "NO_RECIPE", "bends": []}},
                                  target=0.80)["families_capped_below_target"] == ["BBBBBBBB"]
