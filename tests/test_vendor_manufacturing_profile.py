"""Pure independently specified math; no server, CAD, scan or catalog imports."""
import math
import random

import pytest

from backend.vendor_manufacturing_profile import (
    CompensationProfile, CompensationRefused, SCHEMA, from_upstream_bend,
    manufacturing_release_status,
)


def profile(**changes):
    data = dict(schema=SCHEMA, profile_id="synthetic-oracle", revision="1",
                material="synthetic", process="synthetic", tooling="synthetic",
                thickness_mm=1.5, k_factor_ansi=0.4, inner_radius_min_mm=0.1,
                inner_radius_max_mm=100, sweep_min_deg=1, sweep_max_deg=175,
                evidence_sha256=None, purpose="synthetic_development")
    data.update(changes)
    return data


def test_independent_neutral_arc_oracle():
    rng = random.Random(20260929)
    for _ in range(2000):
        t, k, r, sweep = rng.uniform(.1, 10), rng.random(), rng.uniform(.1, 100), rng.uniform(1, 175)
        p = CompensationProfile.from_dict(profile(thickness_mm=t, k_factor_ansi=k))
        # Derive the arc as its circumference fraction, independently of API radians.
        expected = (2 * math.pi * (r + k * t)) * (sweep / 360)
        actual = p.allowance_mm(r, sweep, t, material="synthetic", process="synthetic", tooling="synthetic")
        assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("included", [5, 30, 90, 120, 179])
def test_skin_and_included_angle_conventions(included):
    up = from_upstream_bend(3, included, 1.5)
    down = from_upstream_bend(4.5, -included, 1.5)
    assert up["inner_radius_mm"] == down["inner_radius_mm"] == 3
    assert up["sweep_deg"] == down["sweep_deg"] == 180 - included


@pytest.mark.parametrize("field,value", [("k_factor_ansi", True), ("k_factor_ansi", float("nan")),
    ("k_factor_ansi", -0.1), ("thickness_mm", 0), ("sweep_max_deg", 180),
    ("purpose", "approved"), ("purpose", []), ("thickness_mm", 10 ** 1000),
    ("evidence_sha256", "fake"), ("tooling", "")])
def test_invalid_profiles_refused(field, value):
    with pytest.raises(CompensationRefused):
        CompensationProfile.from_dict(profile(**{field: value}))


@pytest.mark.parametrize("changes", [dict(material="other"), dict(tooling="other"),
    dict(process="other"), dict(thickness_mm=1.6), dict(inner_radius_mm=101), dict(sweep_deg=176)])
def test_applicability_refused(changes):
    values = dict(inner_radius_mm=3, sweep_deg=90, thickness_mm=1.5,
                  material="synthetic", process="synthetic", tooling="synthetic")
    values.update(changes)
    with pytest.raises(CompensationRefused):
        CompensationProfile.from_dict(profile()).allowance_mm(**values)


def test_no_default_or_fabricated_approval():
    with pytest.raises(CompensationRefused):
        CompensationProfile.from_dict({"schema": SCHEMA})
    p = CompensationProfile.from_dict(profile(evidence_sha256="a" * 64, purpose="shop_candidate"))
    assert not manufacturing_release_status(p)["cut_release_allowed"]
    assert not manufacturing_release_status(None)["manufacturing_ready"]
    assert p.sha256 == CompensationProfile.from_dict(dict(reversed(list(profile(evidence_sha256="a" * 64,
        purpose="shop_candidate").items())))).sha256


def test_direct_constructor_cannot_bypass_validation():
    data = profile(k_factor_ansi=-1)
    del data["schema"]
    with pytest.raises(CompensationRefused):
        CompensationProfile(**data)


def test_subnormal_included_angle_cannot_become_flat_fold():
    with pytest.raises(CompensationRefused, match="UNREPRESENTABLE"):
        from_upstream_bend(3, 5e-324, 1.5)
