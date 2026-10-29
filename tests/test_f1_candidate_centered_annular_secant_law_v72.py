from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pytest

import backend.f1_candidate_centered_annular_secant_law_v72 as v72


RADIAL = np.arange(0.0, 75.0 + 0.25, 0.5, dtype=np.float64)
AXIAL = np.arange(0.0, 170.0 + 0.25, 0.5, dtype=np.float64)
CAP_SAFE_RADIAL_HALF = np.asarray([index / 2.0 for index in range(147)], dtype=np.float64)
CAP_SAFE_SIGNED_HALF = np.asarray([index / 2.0 for index in range(-146, 147)], dtype=np.float64)
CAP_SAFE_RADIAL_QUARTER = np.asarray([index / 4.0 for index in range(293)], dtype=np.float64)
REMOVED_RADIAL_HALF = np.asarray([index / 2.0 for index in range(147, 151)], dtype=np.float64)
REMOVED_SIGNED_HALF = np.asarray(
    [index / 2.0 for index in (*range(-150, -146), *range(147, 151))],
    dtype=np.float64,
)
REMOVED_RADIAL_QUARTER = np.asarray([index / 4.0 for index in range(293, 301)], dtype=np.float64)


def _union(points: np.ndarray) -> np.ndarray:
    clean = np.asarray(points, dtype=np.float64)
    clean[clean == 0.0] = 0.0
    return np.unique(clean, axis=0)


def _wedge(theta_degrees: float, radial: np.ndarray = RADIAL) -> np.ndarray:
    angle = math.radians(theta_degrees) / 2.0
    plus = np.column_stack((radial * math.cos(angle), radial * math.sin(angle)))
    minus = np.column_stack((radial * math.cos(angle), -radial * math.sin(angle)))
    return _union(np.vstack((plus, minus)))


def _rounded(theta_degrees: float, rho: float, radial: np.ndarray = RADIAL) -> np.ndarray:
    theta = math.radians(theta_degrees)
    phi = theta / 2.0 + (math.pi - theta) / 2.0 * np.exp(-((radial / rho) ** 2))
    plus = np.column_stack((radial * np.cos(phi), radial * np.sin(phi)))
    minus = np.column_stack((radial * np.cos(phi), -radial * np.sin(phi)))
    return _union(np.vstack((plus, minus)))


def _roll(radius: float) -> np.ndarray:
    radial = RADIAL[RADIAL <= 2.0 * radius]
    x = radial * np.sqrt(np.maximum(0.0, 1.0 - radial * radial / (4.0 * radius * radius)))
    y = radial * radial / (2.0 * radius)
    return _union(np.vstack((np.column_stack((x, y)), np.column_stack((-x, y)))))


def _return(gap: float) -> np.ndarray:
    tail_x = np.arange(gap / 2.0, 75.0 + 0.25, 0.5, dtype=np.float64)
    tails = np.vstack(
        (
            np.column_stack((tail_x, np.full_like(tail_x, gap / 2.0))),
            np.column_stack((tail_x, np.full_like(tail_x, -gap / 2.0))),
        )
    )
    step = 0.5 / (gap / 2.0)
    alpha = np.arange(math.pi / 2.0, 3.0 * math.pi / 2.0 + step / 2.0, step)
    arc = np.column_stack((gap / 2.0 + gap / 2.0 * np.cos(alpha), gap / 2.0 * np.sin(alpha)))
    return _union(np.vstack((tails, arc)))


def _edge() -> np.ndarray:
    return np.column_stack((RADIAL, np.zeros_like(RADIAL)))


def _crossing(radial: np.ndarray = RADIAL) -> np.ndarray:
    return _union(
        np.vstack(
            (
                np.column_stack((radial, np.zeros_like(radial))),
                np.column_stack((-radial, np.zeros_like(radial))),
                np.column_stack((np.zeros_like(radial), radial)),
                np.column_stack((np.zeros_like(radial), -radial)),
            )
        )
    )


def _parallel(gap: float, x: np.ndarray | None = None) -> np.ndarray:
    if x is None:
        x = np.arange(-75.0, 75.0 + 0.25, 0.5, dtype=np.float64)
    return _union(
        np.vstack(
            (
                np.column_stack((x, np.full_like(x, gap / 2.0))),
                np.column_stack((x, np.full_like(x, -gap / 2.0))),
            )
        )
    )


def _extrude(profile: np.ndarray, axial: np.ndarray = AXIAL) -> np.ndarray:
    repeated = np.repeat(profile, len(axial), axis=0)
    z = np.tile(axial, len(profile))
    return _union(np.column_stack((repeated, z)))


def _piecewise_profile(inner: np.ndarray, low: float, high: float) -> np.ndarray:
    flat = _wedge(180.0)
    parts = []
    for z in AXIAL:
        profile = inner if low <= z <= high else flat
        parts.append(np.column_stack((profile, np.full(len(profile), z))))
    return _union(np.vstack(parts))


def _row(name: str, *, start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 170.0)) -> dict:
    return {
        "candidate_opaque_hash": name,
        "line_start": list(start),
        "line_end": list(end),
        "score": 999999.0,
        "selection_route": "forbidden-and-ignored",
    }


def _run(points: np.ndarray, rows: list[dict], spacing: np.ndarray | float = 1.0, faces=None):
    h = np.full(len(points), float(spacing), dtype=np.float64) if np.isscalar(spacing) else np.asarray(spacing, dtype=np.float64)
    return v72.apply_casl_v1({"predictions": rows}, points, faces, local_spacings=h)


def _artifact(result: dict) -> bytes:
    return v72.canonical_bytes(result["casl_v1_audit"])


def _classes(evaluations) -> dict[str, tuple[str, ...]]:
    return {value.candidate.opaque_hash: value.station_classes for value in evaluations}


def _assert_removed_tail_has_zero_query_membership(tail_points: np.ndarray, rows: list[dict]) -> None:
    """Audit removed records against the unchanged CASL station-annulus query."""
    outer_radius = 4.0 * (5.0 / 4.0) ** 13
    for ordinal, row in enumerate(rows):
        candidate = v72._candidate(row, ordinal)
        relative, along, valid_ids, station_ids = v72._station_assignment(candidate, tail_points)
        tau = along[valid_ids] - ((station_ids.astype(np.float64) + 0.5) / v72.STATION_COUNT) * candidate.length
        q = relative[valid_ids] - along[valid_ids, None] * candidate.direction
        radius = np.linalg.norm(q, axis=1)
        axial_half_width = min(3.0, candidate.length / 34.0)
        membership = (np.abs(tau) < axial_half_width) & (radius <= outer_radius)
        assert not np.any(membership), (
            f"removed analytic tail entered a frozen CASL query for {candidate.opaque_hash}: "
            f"{int(np.count_nonzero(membership))} records"
        )


GEOMETRIC_FIXTURES = [f"G{index:02d}" for index in range(1, 18)]


@pytest.mark.parametrize("name", GEOMETRIC_FIXTURES)
def test_v72_geometric_fixture(name: str) -> None:
    rows = [_row(name)]
    if name == "G01":
        points = _extrude(_wedge(90.0)); expected = v72.DECISION_CERTIFIED_KEEP
    elif name == "G02":
        points = _extrude(_wedge(160.0)); expected = v72.DECISION_CERTIFIED_KEEP
    elif name == "G03":
        points = _extrude(_rounded(90.0, 8.0)); expected = v72.DECISION_CERTIFIED_KEEP
    elif name == "G04":
        points = _extrude(_rounded(135.0, 16.0)); expected = v72.DECISION_CERTIFIED_KEEP
    elif name == "G05":
        points = _extrude(_return(6.0)); expected = v72.DECISION_CERTIFIED_KEEP
    elif name == "G06":
        points = _extrude(_rounded(90.0, 8.0)); points[:, 1] *= -1.0; expected = v72.DECISION_CERTIFIED_KEEP
    elif name == "G07":
        points = _piecewise_profile(_rounded(90.0, 8.0), 55.0, 115.0); expected = v72.DECISION_CERTIFIED_KEEP
    elif name == "G08":
        points = _piecewise_profile(_rounded(90.0, 8.0), 10.0, 145.0); expected = "NOT_SUPPRESS"
    elif name == "G09":
        profile = _rounded(90.0, 8.0)
        parts = []
        for ordinal, z in enumerate(AXIAL):
            current = profile
            if ordinal % 4 == 0:
                current = current[current[:, 1] <= 0.0] if (ordinal // 4) % 2 == 0 else current[current[:, 1] >= 0.0]
            parts.append(np.column_stack((current, np.full(len(current), z))))
        points = _union(np.vstack(parts)); expected = "NOT_SUPPRESS"
    elif name == "G10":
        first = _rounded(90.0, 8.0, CAP_SAFE_RADIAL_HALF) + np.asarray((-12.0, 0.0))
        second = _rounded(90.0, 8.0, CAP_SAFE_RADIAL_HALF) + np.asarray((12.0, 0.0))
        points = _extrude(_union(np.vstack((first, second))))
        rows = [
            _row("G10-A", start=(-12.0, 0.0, 0.0), end=(-12.0, 0.0, 170.0)),
            _row("G10-B", start=(12.0, 0.0, 0.0), end=(12.0, 0.0, 170.0)),
        ]
        removed_first = _rounded(90.0, 8.0, REMOVED_RADIAL_HALF) + np.asarray((-12.0, 0.0))
        removed_second = _rounded(90.0, 8.0, REMOVED_RADIAL_HALF) + np.asarray((12.0, 0.0))
        removed = _extrude(_union(np.vstack((removed_first, removed_second))))
        assert len(points) == 199_826
        _assert_removed_tail_has_zero_query_membership(removed, rows)
        expected = "NEITHER_SUPPRESS"
    elif name == "G11":
        points = _extrude(_wedge(180.0)); expected = v72.DECISION_SUPPRESS
    elif name == "G12":
        points = _extrude(_edge()); expected = v72.DECISION_SUPPRESS
    elif name == "G13":
        points = _extrude(_roll(20.0)); expected = v72.DECISION_SUPPRESS
    elif name == "G14":
        points = _extrude(_roll(80.0)); expected = v72.DECISION_SUPPRESS
    elif name == "G15":
        points = _extrude(_crossing(CAP_SAFE_RADIAL_HALF))
        assert len(points) == 199_485
        _assert_removed_tail_has_zero_query_membership(_extrude(_crossing(REMOVED_RADIAL_HALF)), rows)
        expected = v72.DECISION_ABSTAIN_KEEP
    elif name == "G16":
        points = _extrude(_parallel(6.0, CAP_SAFE_SIGNED_HALF))
        assert len(points) == 199_826
        _assert_removed_tail_has_zero_query_membership(_extrude(_parallel(6.0, REMOVED_SIGNED_HALF)), rows)
        expected = v72.DECISION_ABSTAIN_KEEP
    elif name == "G17":
        points = _extrude(_rounded(90.0, 8.0))
        rows = [_row(name, start=(4.0, 0.0, 0.0), end=(4.0, 0.0, 170.0))]
        expected = v72.DECISION_ABSTAIN_KEEP
    else:
        raise AssertionError(name)

    result, evaluations = _run(points, rows)
    replay, replay_evaluations = _run(points, rows)
    assert result["casl_v1_audit"]["status"] == "COMPLETE"
    assert len(evaluations) == len(rows)
    assert _artifact(result) == _artifact(replay)
    assert _classes(evaluations) == _classes(replay_evaluations)
    decisions = [value.decision for value in evaluations]
    if expected == "NOT_SUPPRESS":
        assert decisions[0] != v72.DECISION_SUPPRESS
    elif expected == "NEITHER_SUPPRESS":
        assert all(value != v72.DECISION_SUPPRESS for value in decisions)
    else:
        assert decisions == [expected] * len(rows)
    if name == "G06":
        base, base_evaluations = _run(_extrude(_rounded(90.0, 8.0)), [_row(name)])
        assert evaluations[0].station_classes == base_evaluations[0].station_classes
    if name == "G11":
        assert evaluations[0].summary["dominant_anti_class"] == "FLAT"
    if name == "G12":
        assert evaluations[0].summary["dominant_anti_class"] == "EDGE"
    if name in {"G13", "G14"}:
        assert evaluations[0].summary["dominant_anti_class"] == "ROLL"
    if name == "G17":
        assert evaluations[0].summary["positive_count"] == 0


def _ideal_shell(theta: float, radius: float, *, eta: float = 0.5) -> v72.ShellDescriptor:
    bisector = np.asarray((1.0, 0.0, 0.0))
    principal = np.asarray((0.0, 1.0, 0.0))
    b = math.cos(theta / 2.0)
    a = math.sin(theta / 2.0)
    plus = b * bisector + a * principal
    minus = b * bisector - a * principal
    return v72.ShellDescriptor(
        True, True, eta >= 0.08, False,
        theta >= 176.0 * v72.DEG and eta >= 0.10,
        radius, theta, eta, 0.0,
        2.0 * radius * math.sin(theta / 2.0),
        2.0 * math.cos(theta / 2.0) / radius,
        principal, bisector, (plus, minus), None,
    )


COMPARATOR_FIXTURES = [
    "wedge_lower", "wedge_upper", "flat", "ratio", "drift", "eta", "effective_count", "candidate",
]


@pytest.mark.parametrize("name", COMPARATOR_FIXTURES)
def test_v72_comparator_fixture(name: str) -> None:
    radii = [4.5, 5.7, 7.2, 9.1, 11.5]
    if name == "wedge_lower":
        exact = [_ideal_shell(8.0 * v72.DEG, radius) for radius in radii]
        below = [_ideal_shell(np.nextafter(8.0 * v72.DEG, -math.inf), radius) for radius in radii]
        assert v72._model_window(exact, "WEDGE")
        assert not v72._model_window(below, "WEDGE")
    elif name == "wedge_upper":
        exact = [_ideal_shell(174.0 * v72.DEG, radius) for radius in radii]
        above = [_ideal_shell(np.nextafter(174.0 * v72.DEG, math.inf), radius) for radius in radii]
        assert v72._model_window(exact, "WEDGE")
        assert not v72._model_window(above, "WEDGE")
    elif name == "flat":
        exact = [_ideal_shell(176.0 * v72.DEG, radius) for radius in radii]
        assert v72._model_window(exact, "FLAT")
    elif name == "ratio":
        assert v72._ratio_residual([1.0, 1.20]) <= 1.0
        assert v72._ratio_residual([1.0, np.nextafter(1.20, math.inf)]) > 1.0
    elif name == "drift":
        first = np.asarray((1.0, 0.0, 0.0))
        outside = np.asarray(
            (float.fromhex("0x1.fd31f94f867c6p-1"), float.fromhex("0x1.ac2609b3c578ap-4"), 0.0)
        )
        inside = np.asarray(
            (float.fromhex("0x1.fd31f94f867c7p-1"), float.fromhex("0x1.ac2609b3c573dp-4"), 0.0)
        )
        boundary = float.fromhex("0x1.acee9f37bebd5p-4")
        assert boundary == 6.0 * v72.DEG
        assert float(np.dot(outside, outside)) == 1.0
        assert float(np.dot(inside, inside)) == 1.0
        assert np.nextafter(float(np.dot(first, outside)), math.inf) == float(np.dot(first, inside))
        assert v72._safe_angle(first, outside) > boundary
        assert v72._safe_angle(first, inside) <= boundary
    elif name == "eta":
        exact = [_ideal_shell(90.0 * v72.DEG, radius, eta=0.08) for radius in radii]
        below = [_ideal_shell(90.0 * v72.DEG, radius, eta=np.nextafter(0.08, -math.inf)) for radius in radii]
        assert all(value.model_usable for value in exact)
        assert not any(value.model_usable for value in below)
    elif name == "effective_count":
        assert v72._effective_count(12.0, 12.0) == 12.0
        weight = float.fromhex("0x1.8000000000000p+3")
        squared_weight = float.fromhex("0x1.8000000000001p+3")
        below = v72._effective_count(weight, squared_weight)
        assert below == float.fromhex("0x1.7ffffffffffffp+3")
        assert below == np.nextafter(12.0, -math.inf)
        assert below < 12.0
    elif name == "candidate":
        base = [v72.CLASS_ANTI_ROLL] * 13 + [v72.CLASS_UNKNOWN] * 4
        decision, summary = v72.candidate_decision_from_classes(base)
        assert decision == v72.DECISION_SUPPRESS
        assert summary["anti_total"] == 13 and summary["dominant_anti_count"] == 13 and summary["dominant_anti_run"] == 13
        assert v72.candidate_decision_from_classes([v72.CLASS_ANTI_ROLL] * 12 + [v72.CLASS_UNKNOWN] * 5)[0] != v72.DECISION_SUPPRESS
        eleven = [v72.CLASS_ANTI_ROLL] * 11 + [v72.CLASS_ANTI_FLAT] * 2 + [v72.CLASS_UNKNOWN] * 4
        assert v72.candidate_decision_from_classes(eleven)[0] == v72.DECISION_SUPPRESS
        ten = [v72.CLASS_ANTI_ROLL] * 10 + [v72.CLASS_ANTI_FLAT] * 3 + [v72.CLASS_UNKNOWN] * 4
        assert v72.candidate_decision_from_classes(ten)[0] != v72.DECISION_SUPPRESS
        run8 = [v72.CLASS_ANTI_ROLL] * 8 + [v72.CLASS_UNKNOWN] + [v72.CLASS_ANTI_ROLL] * 3 + [v72.CLASS_ANTI_FLAT] * 2 + [v72.CLASS_UNKNOWN] * 3
        assert v72.candidate_decision_from_classes(run8)[0] == v72.DECISION_SUPPRESS
        run7 = [v72.CLASS_ANTI_ROLL] * 7 + [v72.CLASS_UNKNOWN] + [v72.CLASS_ANTI_ROLL] * 4 + [v72.CLASS_ANTI_FLAT] * 2 + [v72.CLASS_UNKNOWN] * 3
        assert v72.candidate_decision_from_classes(run7)[0] != v72.DECISION_SUPPRESS
        positive = [v72.CLASS_POSITIVE_WEDGE] + base[:-1]
        assert v72.candidate_decision_from_classes(positive)[0] != v72.DECISION_SUPPRESS
    else:
        raise AssertionError(name)


METAMORPHIC_FIXTURES = [
    "source_permutation", "candidate_permutation", "endpoint_reversal", "reflection", "axis_permutation",
    "translation", "scale_0_8", "scale_1_25", "refinement", "ignored_mesh",
]


@pytest.mark.parametrize("name", METAMORPHIC_FIXTURES)
def test_v72_metamorphic_fixture(name: str) -> None:
    base_radial = CAP_SAFE_RADIAL_HALF if name == "refinement" else RADIAL
    points = _extrude(_wedge(90.0, base_radial))
    rows = [_row("META")]
    baseline, baseline_evaluations = _run(points, rows)
    baseline_classes = _classes(baseline_evaluations)
    if name == "source_permutation":
        permutation = np.random.default_rng(7201).permutation(len(points))
        changed, evaluations = _run(points[permutation], rows)
        assert _artifact(changed) == _artifact(baseline)
    elif name == "candidate_permutation":
        two = [_row("META-A"), _row("META-B")]
        first, first_evaluations = _run(points, two)
        changed, evaluations = _run(points, list(reversed(two)))
        assert _classes(first_evaluations) == _classes(evaluations)
        assert _artifact(first) == _artifact(changed)
        return
    elif name == "endpoint_reversal":
        reversed_row = _row("META", start=(0.0, 0.0, 170.0), end=(0.0, 0.0, 0.0))
        changed, evaluations = _run(points, [reversed_row])
        assert _artifact(changed) == _artifact(baseline)
    elif name == "reflection":
        matrix = np.diag((-1.0, 1.0, 1.0))
        changed, evaluations = _run(points @ matrix.T, rows)
    elif name == "axis_permutation":
        matrix = np.asarray(((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)))
        transformed_rows = [_row("META", start=matrix @ np.asarray((0.0, 0.0, 0.0)), end=matrix @ np.asarray((0.0, 0.0, 170.0)))]
        changed, evaluations = _run(points @ matrix.T, transformed_rows)
    elif name == "translation":
        translation = np.asarray((1024.0, -512.0, 256.0))
        changed_rows = [_row("META", start=translation, end=translation + np.asarray((0.0, 0.0, 170.0)))]
        changed, evaluations = _run(points + translation, changed_rows)
    elif name in {"scale_0_8", "scale_1_25"}:
        scale = 0.8 if name == "scale_0_8" else 1.25
        changed_rows = [_row("META", end=(0.0, 0.0, 170.0 * scale))]
        changed, evaluations = _run(points * scale, changed_rows, spacing=scale)
    elif name == "refinement":
        refined = _extrude(_wedge(90.0, CAP_SAFE_RADIAL_QUARTER))
        assert len(points) == 99_913
        assert len(refined) == 199_485
        assert np.array_equal(CAP_SAFE_RADIAL_QUARTER[::2], CAP_SAFE_RADIAL_HALF)
        removed = _extrude(_wedge(90.0, REMOVED_RADIAL_QUARTER))
        _assert_removed_tail_has_zero_query_membership(removed, rows)
        changed, evaluations = _run(refined, rows, spacing=1.0 / math.sqrt(2.0))
    elif name == "ignored_mesh":
        changed, evaluations = _run(points, rows, faces={"adjacency": "arbitrary", "winding": [3, 2, 1]})
        assert _artifact(changed) == _artifact(baseline)
    else:
        raise AssertionError(name)
    assert _classes(evaluations) == baseline_classes


CAP_FIXTURES = [
    "source_cap", "candidate_cap", "nonfinite", "spacing_contract", "zero_length", "stress", "result_cap", "atomicity",
]


@pytest.mark.parametrize("name", CAP_FIXTURES)
def test_v72_cap_and_atomicity_fixture(name: str, tmp_path: Path) -> None:
    base_points = _extrude(_edge(), axial=np.arange(0.0, 170.0 + 0.25, 2.0))
    if name == "source_cap":
        points = np.zeros((200_001, 3), dtype=np.float64)
        result, evaluations = _run(points, [_row(name)])
        assert evaluations == () and result["casl_v1_audit"]["abort_reason"] == "SOURCE_POINT_CAP"
    elif name == "candidate_cap":
        result, evaluations = _run(base_points, [_row(f"CAP-{index}") for index in range(65)])
        assert evaluations == () and result["casl_v1_audit"]["abort_reason"] == "CANDIDATE_CAP"
    elif name == "nonfinite":
        for value in (math.nan, math.inf):
            points = base_points.copy(); points[0, 0] = value
            result, evaluations = _run(points, [_row(name)])
            assert evaluations == () and result["casl_v1_audit"]["abort_reason"] == "NONFINITE_INPUT"
    elif name == "spacing_contract":
        spacing = np.ones(len(base_points)); spacing[0] = 0.0
        result, evaluations = _run(base_points, [_row(name)], spacing=spacing)
        assert evaluations == () and result["casl_v1_audit"]["abort_reason"] == "SOURCE_SCALE_CONTRACT"
    elif name == "zero_length":
        result, evaluations = _run(base_points, [_row(name, end=(0.0, 0.0, 0.0))])
        assert evaluations == () and result["casl_v1_audit"]["abort_reason"] == "ZERO_LENGTH_CANDIDATE"
    elif name == "stress":
        radial = np.arange(1000, dtype=np.float64)
        axial = np.linspace(0.0, 170.0, 200, dtype=np.float64)
        points = np.column_stack((np.repeat(radial, len(axial)), np.zeros(200_000), np.tile(axial, len(radial))))
        result, evaluations = _run(points, [_row(f"STRESS-{index:02d}") for index in range(64)])
        replay, replay_evaluations = _run(points, [_row(f"STRESS-{index:02d}") for index in range(64)])
        assert result["casl_v1_audit"]["status"] == "COMPLETE" and len(evaluations) == 64
        assert _artifact(result) == _artifact(replay) and len(replay_evaluations) == 64
        counters = result["casl_v1_audit"]["work_counters"]
        assert counters["total_point_candidate_tests"] == 25_600_000
        assert counters["graph_vertices_visited"] == counters["graph_edges_visited"] == counters["mesh_adjacency_reads"] == 0
        assert result["casl_v1_operational_envelope"]["result_bytes"] <= 262_144
    elif name == "result_cap":
        with pytest.raises(v72.CASLResourceAbort, match="RESULT_BYTES_CAP"):
            v72.validate_result_bytes(b"x" * 262_145)
    elif name == "atomicity":
        target = tmp_path / "result.json"
        with pytest.raises(v72.CASLAtomicAbort, match="ATOMIC_COMMIT_FAILURE"):
            v72.commit_canonical_artifact(target, b"{}", inject_pre_rename_failure=True)
        assert not target.exists() and not target.with_suffix(".json.tmp").exists()
    else:
        raise AssertionError(name)
