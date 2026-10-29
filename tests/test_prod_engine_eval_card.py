"""E4a amendment-4b scanner model card: the estimators recover known parameters on
synthetic two-skin plates, the card loads with E2's own ScannerModel, and the
committed card states provenance MODEL_CARD with every field."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from domains.production.engine.eval.scanner_card import (
    ASSUMED_FIELDS,
    MEASURED_FIELDS,
    build_card,
    field_status,
    load_evidence,
    main,
    measure_points,
    validate_card,
)

REPO = Path(__file__).resolve().parents[1]
CARD = REPO / "docs" / "production" / "runlog" / "scanner_model_card.json"
DOC = REPO / "docs" / "production" / "ENGINE_EVAL.md"


def _plate(*, sigma=0.01, h=0.3, t=2.0, dup_fraction=0.15, dup_gap=0.08, flare_p=0.2, flare_mm=0.4,
           outliers=0.001, seed=0):
    """Two skins (z=0 up, z=-t down) of a 60 x 40 mm plate, edge walls on the four sides,
    a doubled top-skin band, flared wall columns and a few far outliers."""
    rng = np.random.default_rng(seed)
    xs, ys = np.arange(0, 60, h), np.arange(0, 40, h)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    top = np.column_stack([gx, gy, rng.normal(0, sigma, gx.size)])
    bot = np.column_stack([gx, gy, -t + rng.normal(0, sigma, gx.size)])
    pts = [top, bot]
    nrm = [np.tile([0, 0, 1.0], (gx.size, 1)), np.tile([0, 0, -1.0], (gx.size, 1))]
    band = top[:, 0] < 60 * dup_fraction
    dup = top[band].copy()
    dup[:, 2] = dup_gap + rng.normal(0, sigma, len(dup))
    pts.append(dup)
    nrm.append(np.tile([0, 0, 1.0], (len(dup), 1)))
    zs = np.arange(-t, 0.001, h)
    walls = []
    for axis, value, normal in ((0, 0.0, (-1, 0, 0)), (0, xs[-1], (1, 0, 0)), (1, 0.0, (0, -1, 0)), (1, ys[-1], (0, 1, 0))):
        along = ys if axis == 0 else xs
        for a in along:
            col = []
            for z in zs:
                p = [0.0, 0.0, z]
                p[axis] = value
                p[1 - axis] = a
                col.append(p)
            col = np.asarray(col)
            if rng.random() < flare_p:
                extra = col[-1].copy()
                extra[2] = rng.uniform(0.0, flare_mm) + 0.0  # sticks out above the top skin
                col = np.vstack([col, extra])
            walls.append(col)
            nrm.append(np.tile(normal, (len(col), 1)))
    pts += walls
    allp = np.vstack(pts)
    alln = np.vstack(nrm).astype(np.float64)
    n_out = int(outliers * len(allp))
    far = allp[rng.choice(len(allp), n_out)] + rng.normal(0, 1, (n_out, 3)) * 20.0
    allp = np.vstack([allp, far])
    alln = np.vstack([alln, np.tile([0, 0, 1.0], (n_out, 1))])
    return allp, alln


@pytest.fixture(scope="module")
def plate_stats():
    pts, nrm = _plate()
    return measure_points(pts, nrm, name="plate", n_patches=1500, n_edge_seeds=20000)


def test_spacing_noise_and_skin_separation_are_recovered(plate_stats):
    s = plate_stats
    assert s.nn_spacing_mm == pytest.approx(0.3, rel=0.05)
    assert s.noise_sigma_mm == pytest.approx(0.01, rel=0.35)
    assert s.skin_separation_median_mm == pytest.approx(2.0, abs=0.02)
    assert s.skin_separation_sigma_mm < 0.03


def test_duplicate_layers_flare_and_outliers_are_detected(plate_stats):
    s = plate_stats
    # the doubled band is 15 % of the top skin = 7.5 % of all skin area (seeds are area-uniform)
    assert 0.03 <= s.duplicate_layer_fraction <= 0.12
    assert s.duplicate_layer_gap_mm == pytest.approx(0.08, abs=0.03)
    assert s.edge_samples > 50
    # 20 % of edge columns flare by U(0, 0.4) mm; those below the 0.05 mm threshold are invisible
    assert 0.13 <= s.edge_flare_fraction <= 0.24
    assert 0.3 <= s.edge_flare_p95_mm <= 0.41
    assert s.outlier_fraction == pytest.approx(0.001, abs=0.0015)


def test_a_clean_plate_has_no_duplicates_or_flare():
    pts, nrm = _plate(dup_fraction=0.0, flare_p=0.0, outliers=0.0, seed=1)
    s = measure_points(pts, nrm, name="clean", n_patches=800, n_edge_seeds=10000)
    assert s.duplicate_layer_fraction == 0.0 and s.duplicate_layer_gap_mm is None
    assert s.edge_flare_fraction == 0.0 and s.outlier_fraction == 0.0


def test_a_surface_continuing_the_wall_is_not_flare():
    """A wall that continues into another surface (here: an upstand along the x = 0 edge,
    outer skin coplanar with the wall) is excluded, never counted as a 10 mm flare."""
    pts, nrm = _plate(dup_fraction=0.0, flare_p=0.0, outliers=0.0, seed=2)
    ys = np.arange(0, 40, 0.3)
    zs = np.arange(0.3, 10.0, 0.3)
    gy, gz = np.meshgrid(ys, zs, indexing="ij")
    up = np.column_stack([np.zeros(gy.size), gy.ravel(), gz.ravel()])
    pts = np.vstack([pts, up])
    nrm = np.vstack([nrm, np.tile([-1.0, 0.0, 0.0], (len(up), 1))])
    s = measure_points(pts, nrm, name="upstand", n_patches=500, n_edge_seeds=15000)
    assert s.edge_samples_complex > 0
    assert s.edge_flare_fraction is not None and s.edge_flare_fraction < 0.02


def test_card_maps_medians_onto_the_e2_model_and_loads_with_e2(plate_stats):
    stats = [plate_stats, replace(plate_stats, name="b", nn_spacing_mm=0.5), replace(plate_stats, name="c",
                                                                                    nn_spacing_mm=0.4)]
    card, evidence = build_card(stats, name="unit_card")
    assert card["provenance"] == "MODEL_CARD" and card["point_spacing_mm"] == pytest.approx(0.4)
    assert card["edge_flare_mm"] == pytest.approx(round(plate_stats.edge_flare_p95_mm / 0.95, 6))
    sha = validate_card(card)
    assert len(sha) == 64
    assert set(evidence["measured"]) == set(MEASURED_FIELDS)
    with pytest.raises(ValueError):
        build_card(stats[:2])
    with pytest.raises(ValueError):
        build_card([replace(s, has_normals=False) for s in stats])
    broken = dict(card)
    broken.pop("view_jitter_mm")
    with pytest.raises(ValueError):
        validate_card(broken)  # a MODEL_CARD must state every field (E2 rule)


def test_committed_card_states_every_field_and_its_status():
    card = json.loads(CARD.read_text(encoding="utf-8"))
    assert card["provenance"] == "MODEL_CARD"
    assert validate_card(card)
    status = field_status(card)
    assert set(status) == set(MEASURED_FIELDS + ASSUMED_FIELDS)
    for name in MEASURED_FIELDS + ASSUMED_FIELDS:
        assert name in card, name
        assert name in card["notes"]
    assert "Retained E2 ASSUMED" in card["notes"]
    # flare is a lower bound, the duplicate-skin offset an upper bound, dropout stays assumed
    assert status["edge_flare_mm"] == status["edge_flare_probability"] == "LOWER_BOUND"
    assert status["duplicate_skin_offset_mm"] == "UPPER_BOUND"
    assert all(status[f] == "ASSUMED" for f in ASSUMED_FIELDS)
    assert "LOWER BOUNDS" in card["notes"] and "PARTIAL" in card["notes"]


def test_field_status_never_invents_a_measurement():
    card = json.loads(CARD.read_text(encoding="utf-8"))
    stripped = dict(card, notes=card["notes"].split("FIELD_STATUS=")[0])
    assert field_status(stripped) is None  # a MODEL_CARD without its status line is not trusted
    forged = dict(card, notes=card["notes"].replace('"grazing_start_deg":"ASSUMED"', '"grazing_start_deg":"GUESSED"'))
    assert field_status(forged) is None
    assumed = {"provenance": "ASSUMED", "noise_sigma_mm": 0.025, "notes": "defaults"}
    assert field_status(assumed) == {"noise_sigma_mm": "ASSUMED"}


def test_committed_card_is_rebuilt_byte_for_byte_from_the_documented_evidence(tmp_path):
    """The per-scan evidence lives verbatim in ENGINE_EVAL.md; the documented rebuild
    (no scan is read) reproduces the committed card exactly, through both entry points."""
    evidence = load_evidence(DOC)
    assert len(evidence["scans"]) >= 3 and evidence["card_sha256"] == validate_card(json.loads(CARD.read_text()))
    out = tmp_path / "card.json"
    assert main(["--from-evidence", str(DOC), "--measured-on", "2026-10-17", "--out", str(out)]) == 0
    assert out.read_bytes() == CARD.read_bytes()
    cli = tmp_path / "card_cli.json"
    done = subprocess.run([sys.executable, "-m", "domains.production.engine.eval", "model-card", "--from-evidence",
                           str(DOC), "--measured-on", "2026-10-17", "--out", str(cli)],
                          capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stderr
    assert cli.read_bytes() == CARD.read_bytes()
