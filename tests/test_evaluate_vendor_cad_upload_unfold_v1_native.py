"""Explicit synthetic real-native upload -> catalog -> vendor GET -> V4 check.

This test must be selected by name. It constructs a new analytic quarter-bend
STEP in tmp_path; it never opens the five physical corpus sources.
"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
import time

import pytest

from scripts import evaluate_vendor_cad_upload_unfold_v1 as campaign


def test_independent_step_uses_actual_upload_route_and_vendor_get(tmp_path, monkeypatch):
    pytest.importorskip("OCP")
    from backend import vendor_unfold_worker_v3 as supervision
    try:
        if not supervision._group_rows(os.getpgrp()):
            pytest.skip("process-group monitor returned no rows")
    except (OSError, subprocess.SubprocessError):
        pytest.skip("process-group monitor unavailable in this sandbox")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Vec

    # Independent asymmetric extrusion with actual OCC cylindrical faces.
    profile = [(4., -13.), (4., 0.), (0., 4.), (-9., 4.),
               (-9., 5.), (0., 5.), (5., 0.), (5., -13.)]
    wire = BRepBuilderAPI_MakeWire()
    for i, point in enumerate(profile):
        following = profile[(i + 1) % len(profile)]
        if i in (1, 5):
            radius = 4. if i == 1 else 5.
            circle = gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), radius)
            edge = BRepBuilderAPI_MakeEdge(circle, 0., math.pi / 2.).Edge()
            if i == 5:
                edge.Reverse()
        else:
            edge = BRepBuilderAPI_MakeEdge(gp_Pnt(*point, 0), gp_Pnt(*following, 0)).Edge()
        wire.Add(edge)
    solid = BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(wire.Wire()).Face(),
                                  gp_Vec(0, 0, 17)).Shape()
    source = tmp_path / "independent-vendor-quarter-bend.step"
    writer = STEPControl_Writer()
    writer.Transfer(solid, STEPControl_AsIs)
    writer.Write(str(source))
    raw = source.read_bytes()
    expected = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(campaign, "CASE_SOURCES", (("SYNTHETIC", source.name, expected),))
    case = tmp_path / "case-01-SYNTHETIC"
    case.mkdir()
    started = time.monotonic()
    outcome = campaign._actual_upload_result(case, raw, "SYNTHETIC", expected)
    elapsed = time.monotonic() - started
    assert elapsed < campaign.WALL_SECONDS, (elapsed, outcome)
    assert outcome["status"] == "AVAILABLE", outcome
    assert outcome["candidate_count"] == outcome["flat_bend_count"] == 1
    assert outcome["units_verified"] is False
    assert outcome["manufacturing_ready"] is False
    assert outcome["visual_review_status"] == "PENDING_INDEPENDENT_POST"
    import json
    candidate = json.loads((case / "CANDIDATES.json").read_text())["candidates"][0]
    preview = json.loads((case / "FLAT_PREVIEW.json").read_text())
    assert preview["bend_lines"][0]["feature_id"] == candidate["feature_id"]
    assert candidate["feature_id"] != candidate["label"]
    setup = json.loads((case / "VENDOR_SETUP_RESPONSE.json").read_text())
    assert setup["candidates"] == [candidate]
    assert setup["unfold_preview"] == preview
    folded = json.loads((case / "FOLDED_ASSET.json").read_text())
    assert (case / folded["relative_path"]).exists()
    assert folded["source_sha256"] == expected
