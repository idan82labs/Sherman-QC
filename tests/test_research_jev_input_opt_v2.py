"""Focused synthetic tests for the Jev v2 input-optimisation harness (research/jev/*_v2.py)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from research.jev import arms_v2 as A
from research.jev.pool_v2 import PoolLine, cluster_lines, nms_select, pair_metrics_matrix
from research.jev.questions_v2 import QUESTION_SETS_V2, question_set_sha256, select_questions_v2
from research.jev.runner_v2 import dry_run, live_run, prepare
from research.jev.states_v2 import AUTHORED, StateV2Error, build_state_v2, check_state_v2
from research.jev.client_v1 import LiveNotAuthorized
from scripts.evaluate_f1_against_manual_gold import pair_metrics


def _row(case="c1", cand="P0", start=(0, 0, 0), end=(100, 0, 0), **feat):
    f = {"line_length_mm": float(np.linalg.norm(np.subtract(end, start))), "two_flange_slice_fraction": 0.8,
         "median_bend_angle_deg": 88.0, "median_corner_gap_mm": 1.2, "median_apex_side_mm": 0.6}
    f.update(feat)
    return {"case_id": case, "set": "pool", "cand": cand, "scan_sha256": "ab" * 32, "line_start": list(start),
            "line_end": list(end), "feat": f, "ctx": {"parallel_within_6mm_count": 1,
                                                      "nearest_parallel_offset_mm": 1.4}, "pool": {}}


def test_pair_metrics_matrix_matches_frozen_matcher():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 2, 3)) * 50
    b = rng.normal(size=(4, 2, 3)) * 50
    m = pair_metrics_matrix(a, b)
    for i in range(5):
        for j in range(4):
            ref = pair_metrics(a[i], b[j])
            for k in m:
                assert m[k][i, j] == pytest.approx(ref[k], abs=1e-9)


def test_cluster_and_nms_dedupe_near_duplicates():
    lines = [PoolLine(np.array([0., 0, 0]), np.array([100., 0, 0]), "g", {"corridor_point_count": 10,
                                                                         "finite_support_fraction": 1}),
             PoolLine(np.array([0., 0.5, 0]), np.array([99., 0.5, 0]), "g", {}),
             PoolLine(np.array([0., 30, 0]), np.array([100., 30, 0]), "g", {})]
    cl = cluster_lines(lines)
    assert sorted(len(c["members"]) for c in cl) == [1, 2]
    ends = [[l.start, l.end] for l in lines]
    assert nms_select(ends, [0.9, 0.8, 0.7], 0.5) == [0, 2]
    assert nms_select(ends, [0.9, 0.8, 0.4], 0.5) == [0]


def test_question_v2_encodes_owner_conventions():
    q = QUESTION_SETS_V2["v2"]["is_real_bend"]
    text = json.dumps(q).lower()
    assert "joggle (offset step) is two bends" in text
    assert "15 to 30 deg" in text
    not_bends = " ".join(q["criteria"]["false"]["not_bends"]).lower()
    assert "joggle" not in not_bends and "step" not in not_bends
    for word in ("hole", "thickness edge", "emboss", "rolled", "artifact"):
        assert word in not_bends
    assert question_set_sha256(select_questions_v2("v2"), "v2") != question_set_sha256(
        select_questions_v2("composite_v2"), "composite_v2")


@pytest.mark.parametrize("design", ["typed", "typed_reading", "composite"])
def test_state_has_no_coordinates_ids_or_data_text(design):
    st = build_state_v2(_row(start=(123.456, -7.0, 9.0), end=(223.456, -7.0, 9.0)), design)
    raw = json.dumps(st)
    assert "123.4" not in raw and "c1" not in raw and "P0" not in raw and "abab" not in raw

    def strings(n):
        if isinstance(n, dict):
            for k, v in n.items():
                yield k
                yield from strings(v)
        elif isinstance(n, str):
            yield n
    vals = [s for s in strings(st) if s not in AUTHORED]
    from research.jev.states_v2 import _KEYS
    assert set(vals) <= _KEYS  # everything that is not an authored string is a vocabulary key
    assert ("reading" in raw) == (design != "typed")
    bad = json.loads(raw)
    bad["candidate"]["line"]["length"]["meaning"] = "part 12345 file x.ply"
    with pytest.raises(StateV2Error):
        check_state_v2(bad)
    bad = json.loads(raw)
    bad["candidate"]["line"]["line_start"] = 1.0
    with pytest.raises(StateV2Error):
        check_state_v2(bad)


def test_runner_dry_run_and_live_guards(tmp_path):
    states = [("r1", build_state_v2(_row(), "typed_reading"))]
    s = dry_run(states, tmp_path / "dry", model="jev-1.13.0", variant="v2")
    assert s["records"] == 1 and s["est_cost_usd_uncached"] > 0
    with pytest.raises(LiveNotAuthorized):
        live_run(states, tmp_path / "cache", live=False, model="jev-1.13.0", variant="v2", budget_usd=1, env={})
    with pytest.raises(LiveNotAuthorized):
        live_run(states, tmp_path / "cache", live=True, model="jev-1.13.0", variant="v2", budget_usd=1, env={})
    with pytest.raises(LiveNotAuthorized):
        live_run(states, tmp_path / "cache", live=True, model="jev-1.13.0", variant="v2", budget_usd=0.0,
                 env={"TYPESAFE_API_KEY": "x"})

    class Fake:
        calls = 0

        def post(self, body):
            Fake.calls += 1
            return 200, {}, json.dumps({"model": "jev-1.13.0", "answers": {"is_real_bend": {"type": "noul", "noul": 0.7}},
                                        "usage": {"input_tokens": 100}}).encode()
    for _ in range(2):
        r = live_run(states, tmp_path / "cache", live=True, model="jev-1.13.0", variant="v2", budget_usd=1,
                     env={"TYPESAFE_API_KEY": "x"}, transport=Fake())
    assert Fake.calls == 1 and r["cached_before"] == 1  # second run served from cache
    assert prepare(states, model="jev-1.13.0", variant="v2")[0]["sha"]


def test_select_and_score_with_frozen_matcher():
    rows = [_row(cand="a"), _row(cand="b", start=(0, 0.3, 0), end=(100, 0.3, 0)),
            _row(cand="c", start=(0, 50, 0), end=(100, 50, 0))]
    truth = {"c1": {"zero_bend_control": False, "lines": [{"bend_id": "G1", "line_start": [0, 0, 0],
                                                          "line_end": [100, 0, 0]}]}}
    y = A.any_match_labels(rows, {"c1": truth["c1"]["lines"]})
    assert y.tolist() == [1, 1, 0]
    sel = A.select(rows, [0.9, 0.8, 0.6], 0.5)
    assert sel["c1"] == [0, 2]  # b is a near-duplicate of a
    agg = A.aggregate(A.score_cases(sel, rows, truth, ["c1"]), {"c1": "F"})
    assert (agg["tp"], agg["fp"], agg["fn"]) == (1, 1, 0)
    t, _ = A.fit_threshold(rows, [0.9, 0.8, 0.6], truth, ["c1"], {"c1": "F"})
    assert 0.6 < t <= 0.9
