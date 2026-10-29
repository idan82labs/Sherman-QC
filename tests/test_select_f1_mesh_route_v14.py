import numpy as np

from scripts.select_f1_mesh_route_v14 import recover_continuous_long_seed_families


def _row(candidate_id: str, seed: str, span: float, support: float = 1.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "seed_ids": [seed],
        "span_mm": span,
        "cluster_source_count": 2,
        "finite_support_fraction": support,
        "finite_support_longest_run_fraction": support,
        "endpoint_support_fraction": support,
        "two_sided_normal_angle_deg": 60.0,
        "transverse_angular_bin_fraction": 0.875,
    }


def test_recovers_one_best_continuous_axis_per_unrepresented_family() -> None:
    rows = [
        _row("selected", "A", 90.0),
        _row("weak_b", "B", 80.0),
        _row("best_b", "B", 95.0),
    ]
    selected, changes = recover_continuous_long_seed_families(
        rows, np.asarray([0.40, 0.20, 0.30]), [0], 100.0
    )
    assert selected == [0, 2]
    assert changes[0]["candidate_id"] == "best_b"


def test_rejects_fragmented_long_alias() -> None:
    rows = [_row("selected", "A", 90.0), _row("fragment", "B", 95.0, support=0.89)]
    assert recover_continuous_long_seed_families(
        rows, np.asarray([0.40, 0.30]), [0], 100.0
    ) == ([0], [])


def test_rejects_short_or_low_score_candidate() -> None:
    rows = [_row("selected", "A", 90.0), _row("short", "B", 69.0), _row("low", "C", 90.0)]
    assert recover_continuous_long_seed_families(
        rows, np.asarray([0.40, 0.30, 0.14]), [0], 100.0
    ) == ([0], [])
