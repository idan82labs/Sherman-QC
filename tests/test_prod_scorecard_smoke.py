"""C12 synthetic line scorecard: 2-case --fake-engine smoke run in tmp_path, plus refusals.

The fake engine is E4's truth oracle run as the line's engine step, so this test
exercises E2 generation, the E1 compile, the private LineStore, the JobRunner
(supervised subprocesses), the real C06 render step and the E4 scorer, without
the real E3. It takes about 15 s.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.production_line.ops import synthetic_scorecard as sc

REPO = Path(__file__).resolve().parents[1]
POINTS = 20_000


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    out = tmp_path_factory.mktemp("scorecard") / "sc"
    doc = sc.run_scorecard(sc.ScorecardConfig(out=out, cases=2, fake_engine=True, point_target=POINTS))
    return out, doc


def test_fake_engine_smoke_writes_an_honest_scorecard(card):
    out, doc = card
    md = (out / "scorecard.md").read_text(encoding="utf-8")
    assert md.splitlines()[0] == "# " + sc.HEADER
    assert "SYNTHETIC ENGINEERING CAPABILITY — NOT PHYSICAL ACCURACY — NOT PROMOTION EVIDENCE" in md
    assert "FAKE ENGINE" in md and "NOT the engine" in md
    assert "## Per-quantity error" in md and "| FLANGE_LENGTH | mm |" in md
    assert "## NOT_MEASURED rate" in md
    assert "duration_s | peak_rss_bytes |" in md
    on_disk = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
    assert on_disk["header"] == sc.HEADER and on_disk["engine_mode"] == "FAKE_ENGINE_E4_TRUTH_ORACLE"
    assert len(doc["cases"]) == 2
    for row in doc["cases"]:
        assert row["line_status"] == "OK" and row["run_status"] == "OK"
        assert row["duration_s"] > 0 and row["peak_rss_bytes"] > 0
        assert row["points"] > 0.9 * POINTS
        assert (out / "records" / f"{row['case_id']}.json").is_file()
        ws = out / "line" / "jobs" / row["job_id"]
        assert (ws / "result" / "shopfloor_result.json").is_file()  # the real C06 render step ran
    ov = doc["evaluation"]["overall"]
    assert ov["cases"] == 2 and ov["run_status"] == {"OK": 2}
    nm = ov["not_measured"]
    assert nm["denominator"] > 0 and nm["value"] == nm["numerator"] / nm["denominator"]
    # grid cell 0 is an exact nominal part, cell 1 carries injected errors
    assert [r["part_truth"] for r in doc["cases"]] == ["CONFORMING", "NONCONFORMING"]
    pr = ov["part_rates"]
    assert pr["false_accept_on_nonconforming"]["numerator"] == 0
    assert pr["false_out_on_conforming"] == {"numerator": 0, "denominator": 1, "value": 0.0}
    fl = ov["per_kind"]["FLANGE_LENGTH"]["errors_decided"]
    assert fl["n"] > 0 and fl["max_abs"] < 0.2  # oracle noise is 0.02 mm
    # the two FULL cells hide nothing: the occlusion block must say so instead of claiming coverage
    assert doc["occlusion"]["occluded_cases"] == 0 and doc["occlusion"]["exercised"] is False
    assert "## Occlusion" in md and "No occluded case in this run." in md


def test_scorecard_detects_a_biased_engine(tmp_path, monkeypatch):
    """Mutate-and-check: a fake engine that reads every flange 2.5 mm long must show up."""
    monkeypatch.setattr(sc, "FAKE_ENGINE_MODE", "noise=0.0,bias_flange=2.5")
    doc = sc.run_scorecard(sc.ScorecardConfig(out=tmp_path / "biased", cases=1, fake_engine=True, point_target=POINTS))
    ov = doc["evaluation"]["overall"]
    assert ov["per_kind"]["FLANGE_LENGTH"]["errors_all_measured"]["mae"] == pytest.approx(2.5, abs=1e-3)
    assert doc["cases"][0]["part_truth"] == "CONFORMING"
    assert ov["part_rates"]["false_out_on_conforming"] == {"numerator": 1, "denominator": 1, "value": 1.0}
    assert ov["characteristic_rates"]["false_out_on_truth_in"]["numerator"] > 0


def test_scorecard_refusals(tmp_path):
    inside = REPO / "output" / "c12-scorecard-never-created"
    with pytest.raises(sc.ScorecardError) as e:
        sc.run_scorecard(sc.ScorecardConfig(out=inside, cases=1, fake_engine=True))
    assert e.value.code == "OUT_FORBIDDEN"
    assert not inside.exists()
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "unrelated.txt").write_text("x")
    with pytest.raises(sc.ScorecardError) as e:
        sc.run_scorecard(sc.ScorecardConfig(out=busy, cases=1, fake_engine=True))
    assert e.value.code == "OUT_NOT_A_SCORECARD"
    assert (busy / "unrelated.txt").read_text() == "x"
    with pytest.raises(sc.ScorecardError) as e:
        sc.run_scorecard(sc.ScorecardConfig(out=tmp_path / "a", cases=1))  # neither engine option
    assert e.value.code == "VALUE_INVALID"
    with pytest.raises(sc.ScorecardError) as e:
        sc.run_scorecard(sc.ScorecardConfig(out=tmp_path / "b", cases=1, engine_root=tmp_path))  # not a bundle
    assert e.value.code == "BUNDLE_INVALID"
    with pytest.raises(sc.ScorecardError) as e:
        sc.run_scorecard(sc.ScorecardConfig(out=tmp_path / "c", cases=0, fake_engine=True))
    assert e.value.code == "VALUE_INVALID"
    assert sc.main(["--out", str(inside), "--cases", "1", "--fake-engine"]) == 2


def test_grid_is_frozen_asymmetric_and_dev_only():
    specs = sc.grid_specs(len(sc.GRID) + 2, seed=0, point_target=None)
    assert {c.family for _, c in specs} == {"L_BRACKET", "Z_STEP", "HAT"}
    assert all("symmetric" not in c.features for _, c in specs)
    assert {c.noise_mm for _, c in specs} == {0.02, 0.05, 0.10}
    assert any(c.plan in sc.OCCLUDED_PLANS for _, c in specs)  # the occluded-edge case
    assert all(s.seedset == "DEV" for s, _ in specs)
    assert specs[len(sc.GRID)][1] == sc.GRID[0] and specs[len(sc.GRID)][0].seed != specs[0][0].seed
    again = sc.grid_specs(3, seed=0, point_target=None)
    assert [s.seed for s, _ in again] == [s.seed for s, _ in specs[:3]]
    assert [s.seed for s, _ in sc.grid_specs(3, seed=1, point_target=None)] != [s.seed for s, _ in again]


def test_occluded_cell_really_hides_a_scored_characteristic(tmp_path):
    """The grid's occluded cell must hide at least one applicable characteristic in E2 truth
    (native density, as the scorecard runs it); otherwise it would not test occlusion at all."""
    from domains.production.engine.eval.truth import load_truth_case
    from domains.production.engine.synthetic.cases import generate_case

    idx = [i for i, c in enumerate(sc.GRID) if c.plan in sc.OCCLUDED_PLANS]
    assert idx, "the grid has no occluded cell"
    spec, cell = sc.grid_specs(max(idx) + 1, seed=0, point_target=None)[idx[0]]
    generate_case(spec, tmp_path, scanner_model=sc._scanner_for(cell), ray_threads=1)
    tc = load_truth_case(tmp_path / spec.case_id)
    hidden = sc.truth_unobservable(tc)
    assert hidden, f"{cell.label()} hides nothing in E2 truth"
    # and the summary reports it as exercised, with the reported classes counted
    row = {"plan": cell.plan, "run_status": "OK", "truth_unobservable": hidden,
           "unobservable_reported": {cid: "NOT_MEASURED:EDGE_NOT_SEEN" for cid in hidden}}
    occ = sc.occlusion_summary([row])
    assert occ["exercised"] is True and occ["reported_classes"] == {"NOT_MEASURED": len(hidden)}
    assert occ["decided_on_unobservable"] == 0
    decided = sc.occlusion_summary([{**row, "unobservable_reported": {hidden[0]: "MEASURED:PASS"},
                                     "unobservable_within_u": {hidden[0]: False}}])
    assert decided["decided_on_unobservable"] == 1
    assert decided["measured_on_unobservable_within_u"] == {"numerator": 0, "denominator": 1, "value": 0.0}
    assert occ["measured_on_unobservable_within_u"]["value"] is None  # nothing measured: undefined, not 1.0
    assert "within U: 0/1" in sc._occlusion_line(decided)
    # a grid-1 style occluded cell that hides nothing is reported as NOT exercised
    assert sc.occlusion_summary([{**row, "truth_unobservable": [], "unobservable_reported": {}}])["exercised"] is False
