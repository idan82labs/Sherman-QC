"""Synthetic tests for the E3 x Jev v2 A/B harness (research/jev/e3_ab_v1.py). No scans, no network."""
from __future__ import annotations

import math

import numpy as np

from research.jev import e3_ab_v1 as E

THR = {"lane+jev_composite_combined_filter": 0.3, "lane+best_feature_filter": 0.25, "pool+clf_logreg": 0.9,
       "pool+jev_plus_clf": 0.7}

# rows 0-1: engine lines; 2: pool duplicate of engine line 0; 3: new pool line; 4: weak pool line
ENDS = np.array([
    [[0, 0, 0], [100, 0, 0]],
    [[0, 50, 0], [100, 50, 0]],
    [[0, 0.5, 0], [99, 0.5, 0]],
    [[0, 100, 0], [100, 100, 0]],
    [[0, 150, 0], [100, 150, 0]],
], float)
SCORES = {
    "clf_logreg": {0: 0.1, 1: 0.95, 2: 0.99, 3: 0.97, 4: 0.5},
    "jev_composite_combined": {0: 0.2, 1: 0.9, 2: 0.9, 3: 0.1, 4: 0.9},
    "best_feature": {0: -math.inf, 1: 0.5, 2: 0.9, 3: 0.9, 4: 0.1},
    "jev_plus_clf": {2: 0.9, 3: 0.4, 4: 0.8},
}


def sel(arm):
    return sorted(E.select_arm(E.ARMS[arm], [0, 1], [2, 3, 4], ENDS, SCORES, THR))


def test_engine_alone_keeps_every_engine_line():
    assert sel("a_engine_alone") == [0, 1]


def test_filters_use_frozen_thresholds_and_missing_feature_rule():
    assert sel("b_engine+jev_composite_filter") == [1]
    assert sel("c_engine+single_feature_filter") == [1]  # no pool cluster -> dropped (frozen rule)
    assert sel("c_sens_missing_kept") == [0, 1]


def test_jev_filter_keeps_lines_without_an_answer():
    s = dict(SCORES, jev_composite_combined={1: 0.1})
    assert sorted(E.select_arm(E.ARMS["b_engine+jev_composite_filter"], [0, 1], [], ENDS, s, THR)) == [0]


def test_rescue_keeps_engine_lines_and_adds_only_new_pool_lines():
    assert sel("d_lr") == [0, 1, 3]  # 2 duplicates engine line 0 and is suppressed; 4 below threshold
    assert sel("d_lr_jev") == [0, 1]  # line 3 fails the Jev gate
    assert sel("d_jevclf") == [0, 1, 4]


def test_apply_v2_literal_can_drop_engine_lines():
    assert sel("d_apply_v2") == [1, 2, 3]  # engine line 0 ranks below its pool twin 2
    assert sorted(E.select_arm(E.ARMS["ref_pool_lr_alone"], [0, 1], [2, 3, 4], ENDS, SCORES, THR)) == [2, 3]


def test_frozen_thresholds_come_from_design_choice():
    t = E.frozen_thresholds()
    assert abs(t["pool+clf_logreg"] - 0.9785895158986075) < 1e-12
    assert abs(t["lane+best_feature_filter"] - 0.2576574115346552) < 1e-12
    assert abs(t["lane+jev_composite_combined_filter"] - 0.3066442115835676) < 1e-12
    assert abs(t["pool+jev_plus_clf"] - 0.7150033360494769) < 1e-12


def test_summarise_paired_delta_is_zero_against_itself():
    pc = {"s1": {"family": "f1", "control": False, "recall_only": False, "tp": 3, "fp": 1, "fn": 1},
          "s2": {"family": "f2", "control": False, "recall_only": False, "tp": 1, "fp": 0, "fn": 2},
          "s3": {"family": "f2", "control": False, "recall_only": True, "tp": 1, "fp": 0, "fn": 1},
          "c": {"family": "f3", "control": True, "recall_only": False, "tp": 0, "fp": 0, "fn": 0, "fp_lines": 2}}
    s = E.summarise(pc, pc, rounds=50)
    assert (s["tp"], s["fp"], s["fn"]) == (4, 1, 3)
    assert s["control_fp_lines"] == 2 and s["recall_only_cases"] == {"cases": 1, "tp": 1, "fn": 1}
    assert s["ci95"]["delta_f1"] == [0.0, 0.0]
