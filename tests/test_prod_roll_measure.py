"""Roll measurement prototype (research/zeroshot/roll_measure): synthetic-only unit tests (no scans, no gold)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from research.zeroshot.roll_measure import compare as C
from research.zeroshot.roll_measure import geom as G
from research.zeroshot.roll_measure import measure as M
from research.zeroshot.roll_measure import run as RUN
from research.zeroshot.roll_measure import synth


def _analyse(spec):
    mesh, truth = synth.make(spec)
    res, _ = RUN.analyse(mesh, M.RollParams(), log=lambda *a: None, overlay=False,
                         t_info={"t_mm": spec.t, "confidence": 1.0, "two_sided_fraction": 1.0})
    return res, truth


def test_fit_circle_recovers_radius_and_centre():
    rng = np.random.default_rng(0)
    phi = np.linspace(0.2, 1.9, 200)
    P = np.c_[3 + 50 * np.cos(phi), -2 + 50 * np.sin(phi)] + rng.normal(0, 0.02, (200, 2))
    c, R, rms = G.fit_circle(P)
    assert abs(R - 50) < 0.05 and np.linalg.norm(c - [3, -2]) < 0.1 and rms < 0.03


def test_angular_extent_handles_wrap():
    st, sw = G.angular_extent(np.radians([350, 355, 5, 10]))
    assert abs(math.degrees(sw) - 20) < 1e-6 and abs(math.degrees(st) - 350) < 1e-6


def test_cylinder_measured_against_truth():
    res, tr = _analyse(synth.SynthSpec(R=60, sweep_deg=90, t=2.0, L=60, f1=20, f2=20, noise_mm=0.05, spacing_mm=0.6, seed=3))
    assert res["n_regions"] == 1
    m = res["regions"][0]["measure"]
    assert abs(m["R_inner"]["mean"] - 60) < 0.1
    assert abs(m["R_outer"]["mean"] - 62) < 0.1
    assert abs(m["sweep_deg"]["mean"] - 90) < 1.0
    assert m["skins"] == "both"
    ad = np.array(m["axis"]["direction"])
    assert math.degrees(math.acos(min(1.0, abs(ad @ np.array(tr["axis_dir"]))))) < 0.2
    assert {m["runout_lines"]["start"]["type_majority"], m["runout_lines"]["end"]["type_majority"]} == {"tangent"}
    assert res["regions"][0]["measure"]["feature_class"]["class"] == "ROLL"


def test_cone_taper_magnitude():
    res, tr = _analyse(synth.SynthSpec(R=80, sweep_deg=120, t=2.0, L=100, taper=0.02, noise_mm=0.05, seed=4))
    m = res["regions"][0]["measure"]
    assert abs(abs(m["taper_mm_per_m"]["value"]) - 20.0) < 1.0


def test_sharp_bend_is_not_a_roll():
    # a press-brake bend: inner radius 2 mm (= t) between two long flats
    res, _ = _analyse(synth.SynthSpec(R=2.0, sweep_deg=90, t=2.0, L=60, f1=40, f2=40, noise_mm=0.05, seed=5))
    assert res["n_regions"] == 0


def test_large_radius_bend_is_classified_for_the_bend_lane():
    res, _ = _analyse(synth.SynthSpec(R=14.0, sweep_deg=90, t=2.0, L=60, f1=30, f2=30, noise_mm=0.03, seed=6))
    assert res["n_regions"] == 1
    assert res["regions"][0]["measure"]["feature_class"]["class"] == "LARGE_RADIUS_BEND"


def test_compare_measured_minus_nominal():
    res, _ = _analyse(synth.SynthSpec(R=60, sweep_deg=90, t=2.0, L=60, noise_mm=0.05, seed=7))
    m = res["regions"][0]["measure"]
    cad = {"rolls": [{"id": "R1", "r_inner_mm": 61.0, "r_outer_mm": 63.0, "thickness_mm": 2.0, "sweep_deg": 92.0,
                      "chord_inner_mm": 2 * 61 * math.sin(math.radians(46)), "chord_outer_mm": 2 * 63 * math.sin(math.radians(46)),
                      "axial_length_mm": 60.0, "is_cone": False, "taper": {"mm_per_m": 0.0, "dr_dz": 0.0},
                      "runouts": [{"which": "start", "end_type": "TANGENT_RUNOUT"}, {"which": "end", "end_type": "TANGENT_RUNOUT"}]}]}
    out = C.compare(m, cad)
    assert out["status"] == "COMPARED"
    assert abs(out["rows"]["r_inner_mm"]["delta"] + 1.0) < 0.1
    assert abs(out["rows"]["sweep_deg"]["delta"] + 2.0) < 1.0
    ov, note = C.cad_overlay(cad, m)
    assert "NOT registered" in note and len(ov["_circles_axis"]) == 2
