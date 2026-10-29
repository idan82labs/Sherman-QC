import numpy as np

from scripts.select_f1_mesh_route_v13 import select_with_no_output_retry


def _row(candidate_id: str, y: float) -> dict:
    return {"candidate_id": candidate_id, "span_endpoints": [[0.0, y, 0.0], [100.0, y, 0.0]]}


def test_preserves_normal_precision_selection() -> None:
    selected, retried = select_with_no_output_retry([_row("a", 0.0)], np.asarray([0.50]))
    assert selected == [0]
    assert retried is False


def test_retries_only_when_initial_selection_is_empty() -> None:
    rows = [_row("a", 0.0), _row("b", 10.0)]
    selected, retried = select_with_no_output_retry(rows, np.asarray([0.42, 0.34]))
    assert selected == [0]
    assert retried is True


def test_does_not_force_candidate_below_retry_threshold() -> None:
    selected, retried = select_with_no_output_retry([_row("a", 0.0)], np.asarray([0.34]))
    assert selected == []
    assert retried is True
