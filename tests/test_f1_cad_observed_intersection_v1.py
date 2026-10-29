from __future__ import annotations

import numpy as np

from research.f1_cad_observed_intersection_v1 import observed_intersection, relocate_predictions


def _fold(offset: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(25.0)
    rows = []
    for x in np.linspace(0, 100, 61):
        for distance in np.linspace(2, 10, 12):
            rows.append([x, offset - distance, np.tan(angle) * distance])
            rows.append([x, offset + distance, np.tan(angle) * distance])
    return np.asarray(rows, dtype=np.float64)


def test_relocates_nominal_line_to_observed_fold_intersection():
    result = observed_intersection(_fold(3.0), [0, 0, 0], [100, 0, 0])
    assert result["relocated"] is True
    midpoint = np.mean(np.asarray([result["line_start"], result["line_end"]]), axis=0)
    assert abs(midpoint[1] - 3.0) < 0.5


def test_flat_or_sparse_evidence_abstains_without_deleting_event():
    flat = np.asarray([[x, y, 0] for x in np.linspace(0, 100, 51) for y in np.linspace(-10, 10, 21)], dtype=float)
    result = observed_intersection(flat, [0, 0, 0], [100, 0, 0])
    assert result["relocated"] is False
    predictions = [{"bend_id":"E1","line_start":[0,0,0],"line_end":[100,0,0]}]
    kept, records = relocate_predictions(flat, predictions)
    assert len(kept) == 1 and kept[0]["line_start"] == [0,0,0]
    assert records[0]["state"] == "ABSTAIN_RELOCATION"


def test_out_of_bound_intersection_abstains():
    result = observed_intersection(_fold(10.0), [0, 0, 0], [100, 0, 0])
    assert result["relocated"] is False
    assert result["reason"] in {"INTERSECTION_OUTSIDE_TRUST_BOUND", "LOCAL_SURFACE_FIT_UNTRUSTED"}
