"""zs_lane_v1: synthetic contract checks (no scan data; synthetic gold files in tmp_path only).

Covers: pooling (Z2 agreement flag vs Z2-only extra candidate), variant selection (z3 drops Z2-only rows and uses 21
features; z2 passes Z2 lines through), the fixed threshold, failed / missing outputs scored as all false negatives,
family bootstrap determinism, fingerprint sensitivity, pinned-model refusal, and the TEST-B seal of the guard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.zeroshot.zs_lane_v1 import core, fingerprint, lane, scoring  # noqa: E402


def _cand(s, e, incl=90.0, **kw):
    s, e = np.asarray(s, float), np.asarray(e, float)
    c = {"start": s.tolist(), "end": e.tolist(), "incl": incl, "length": float(np.linalg.norm(e - s)),
         "n_support": 20, "support_frac": 1.0, "incl_std": 0.5, "flank_min": 10.0, "flank_max": 20.0,
         "res_rms": 0.05, "axis_dev_deg": 0.2, "frac_outer": 1.0, "frac_inner": 0.0, "frac_both": 0.0,
         "flat_rms": 0.05, "gap_med": 0.0, "strip_frac": 1.0, "flank_frac": 1.0, "n_parts": 1, "n_dups": 0,
         "quality": 10.0, "family": 0}
    c.update(kw)
    return c


def test_pool_flags_agreement_and_adds_z2_only_line():
    z3 = {"candidates": [_cand([0, 0, 0], [100, 0, 0]), _cand([0, 50, 0], [100, 50, 0])], "t_mm": 1.5}
    z2 = [{"start": [1, 0.5, 0], "end": [99, 0.5, 0], "incl": 91.0},          # duplicates candidate 0
          {"start": [0, 0, 80], "end": [0, 100, 80], "incl": 135.0}]          # no Z3 counterpart
    C, X = core.pool(z3, z2, use_z2=True)
    assert len(C) == 3 and X.shape == (3, lane.Z3_FEATS + 5)
    assert C[0]["src"] == ["z2", "z3"] and C[1]["src"] == ["z3"]
    assert C[2]["src"] == ["z2"] and not C[2]["has_z3"] and C[2]["incl"] == pytest.approx(135.0)
    assert np.isnan(X[2, :lane.Z3_FEATS]).all() and X[0, lane.Z3_FEATS + 1] == 1 and X[1, lane.Z3_FEATS + 1] == 0
    C0, X0 = core.pool(z3, None, use_z2=False)
    assert len(C0) == 2 and np.allclose(X0[:, :lane.Z3_FEATS], X[:2, :lane.Z3_FEATS])


class _Model:
    def __init__(self, ncol):
        self.ncol = ncol

    def predict(self, X):
        assert X.shape[1] == self.ncol
        return np.linspace(0.2, 0.9, len(X))


def test_select_lines_variants_and_fixed_threshold():
    C = [{"start": [0, 0, 0], "end": [1, 0, 0], "has_z3": True},
         {"start": [0, 1, 0], "end": [1, 1, 0], "has_z3": True},
         {"start": [0, 2, 0], "end": [1, 2, 0], "has_z3": False}]
    X = np.zeros((3, lane.Z3_FEATS + 5))
    rec = {"candidates": C, "X": X, "z2_lines": [{"start": [0, 0, 5], "end": [1, 0, 5], "incl": 90.0}]}
    z3 = lane.select_lines(rec, "z3", {"threshold": 0.5}, {"model": _Model(lane.Z3_FEATS)})
    assert [c["start"] for c in z3] == [[0, 1, 0]]                      # probs 0.2, 0.9 over the 2 Z3 rows
    zs = lane.select_lines(rec, "zs", {"threshold": 0.5}, {"model": _Model(lane.Z3_FEATS + 5)})
    assert [c["start"] for c in zs] == [[0, 1, 0], [0, 2, 0]]           # probs 0.2, 0.55, 0.9
    z2 = lane.select_lines(rec, "z2", {"threshold": 0.5}, None)
    assert len(z2) == 1 and z2[0]["src"] == ["z2"]


def _gold(tmp_path, name, bends, no_bends=False):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"coordinate_frame": "source_ply_raw", "no_bends_present": no_bends,
                             "bends": [{"bend_id": f"B{i + 1}", "line_start": s, "line_end": e}
                                       for i, (s, e) in enumerate(bends)]}))
    return {"case_id": name, "family_id": name[:3], "tier": "T", "gold": {"kind": "ai_studio", "path": str(p)},
            "label_complete": True}


def test_failed_output_is_all_false_negatives_and_controls_separate(tmp_path):
    a = _gold(tmp_path, "fa1", [([0, 0, 0], [100, 0, 0]), ([0, 50, 0], [100, 50, 0])])
    b = _gold(tmp_path, "fb1", [([0, 0, 0], [80, 0, 0])])
    c = _gold(tmp_path, "fc1", [], no_bends=True)
    ra = scoring.score_scan(a, [{"start": [0, 0, 0.5], "end": [100, 0, 0.5]}])
    rb = scoring.score_scan(b, None)                                    # failed / missing output
    rc = scoring.score_scan(c, [{"start": [0, 0, 0], "end": [10, 0, 0]}])
    assert (ra["true_positive"], ra["false_positive"], ra["false_negative"]) == (1, 0, 1)
    assert rb["false_negative"] == 1 and rb["output_missing_or_failed"]
    agg = scoring.aggregate([ra, rb, rc], n_boot=200, seed=1)
    assert (agg["tp"], agg["fp"], agg["fn"]) == (1, 0, 2) and agg["scans"] == 2
    assert agg["controls"] == {"scans": 1, "fp_lines": 1} and agg["failed_or_missing_scans"] == 1
    assert scoring.bootstrap([ra, rb], 300, 7) == scoring.bootstrap([ra, rb], 300, 7)


def test_failed_payload_reads_as_missing(tmp_path):
    pay = core.payload(str(tmp_path / "x.ply"), None, [], status="FAILED", conversion={"errors": {"z3": "boom"}})
    p = tmp_path / "eval_lines.json"
    p.write_text(json.dumps(pay))
    assert scoring.eval_lines_of(p) is None and scoring.eval_lines_of(tmp_path / "none.json") is None


def test_fingerprint_is_order_free_and_sensitive():
    h = {"a.py": "1" * 64, "b.py": "2" * 64}
    assert fingerprint.combined(h) == fingerprint.combined(dict(reversed(list(h.items()))))
    assert fingerprint.combined(h) != fingerprint.combined({**h, "b.py": "3" * 64})
    fp = fingerprint.lane_fingerprint()
    assert set(fp["files"]) == set(fingerprint.LANE_FILES) and len(fp["sha256"]) == 64


def test_verify_pins_flags_model_change(tmp_path):
    m = tmp_path / "m.pkl"
    m.write_bytes(b"model-v1")
    fp = fingerprint.lane_fingerprint()
    cfg = {"code_fingerprint": fp, "variants": {"zs": {"model": str(m), "model_sha256": fingerprint.sha256_file(m)}}}
    assert lane.verify_pins(cfg, "zs") == []
    m.write_bytes(b"model-v2")
    assert lane.verify_pins(cfg, "zs") == [f"MODEL:{m}"]


def test_guard_keeps_testb_sealed_without_final_flag():
    from research.zeroshot.common.gold_guard import GoldGuardRefusal
    sid = scoring.split()["test_b"]["scans"][0]                          # an id only; no file is opened
    with pytest.raises(GoldGuardRefusal) as ex:
        lane._guard(sid, final_testb=False)
    assert ex.value.reason == "TEST_B_SEALED"
