from __future__ import annotations

import itertools

from backend.f1_strip_detector_completion import select_strip_detector_completion


def _bend(
    bend_id: str,
    source: str,
    *,
    anchor: tuple[float, float, float],
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    span: float = 100.0,
    width: float = 10.0,
    confidence: float = 0.9,
    angle: float = 90.0,
) -> dict:
    return {
        "bend_id": bend_id,
        "source_kind": source,
        "anchor": anchor,
        "axis_direction": axis,
        "span_mm": span,
        "cross_width_mm": width,
        "confidence": confidence,
        "angle_deg": angle,
        "visibility_state": "internal",
        "atom_ids": tuple(f"{bend_id}_{index}" for index in range(8)),
    }


def test_completion_promotes_long_fragment_and_adds_only_novel_detector_corridor() -> None:
    hypotheses = [
        _bend("S1", "zero_shot_strip_seed", anchor=(0.0, 0.0, 0.0)),
        _bend("S2", "zero_shot_strip_seed", anchor=(0.0, 40.0, 0.0)),
        _bend("S3_fragment", "zero_shot_strip_seed", anchor=(0.0, 80.0, 0.0), span=25.0),
        _bend("D2_duplicate", "detector_bend_seed", anchor=(2.0, 42.0, 0.0), span=110.0),
        _bend("D3_complete", "detector_bend_seed", anchor=(4.0, 88.0, 0.0), span=90.0),
        _bend(
            "D4_novel",
            "detector_bend_seed",
            anchor=(0.0, 120.0, 0.0),
            axis=(0.0, 0.0, 1.0),
            span=50.0,
        ),
    ]

    result = select_strip_detector_completion(hypotheses, local_spacing_mm=2.0)

    assert result.selected_bend_ids == ("S1", "S2", "D3_complete", "D4_novel")
    assert result.fragment_promotions == (("S3_fragment", "D3_complete"),)
    assert result.rejected_duplicate_ids == ("D2_duplicate", "S3_fragment")


def test_completion_requires_multiple_independent_strip_anchors() -> None:
    hypotheses = [
        _bend("S1", "zero_shot_strip_seed", anchor=(0.0, 0.0, 0.0)),
        _bend("D1", "detector_bend_seed", anchor=(0.0, 50.0, 0.0)),
    ]
    assert select_strip_detector_completion(hypotheses, local_spacing_mm=2.0).selected_bend_ids == ()


def test_completion_is_permutation_deterministic() -> None:
    hypotheses = [
        _bend("S1", "zero_shot_strip_seed", anchor=(0.0, 0.0, 0.0)),
        _bend("S2", "zero_shot_strip_seed", anchor=(0.0, 40.0, 0.0)),
        _bend("S3", "zero_shot_strip_seed", anchor=(0.0, 80.0, 0.0)),
        _bend("D4", "detector_bend_seed", anchor=(0.0, 120.0, 0.0)),
    ]
    outcomes = {
        select_strip_detector_completion(ordering, local_spacing_mm=2.0).selected_bend_ids
        for ordering in itertools.permutations(hypotheses)
    }
    assert outcomes == {("S1", "S2", "S3", "D4")}


def test_partial_channel_geometry_replaces_fragment_and_recovers_orthogonal_transition() -> None:
    """Regression for the anonymous geometry pattern exhibited by corpus case09."""
    hypotheses = [
        _bend(
            "B1", "zero_shot_strip_seed",
            anchor=(-230.3943, 40.0061, 313.2630),
            axis=(0.830408, -0.169490, -0.530750), span=137.9195, width=12.0071,
            confidence=1.0, angle=89.9388,
        ),
        _bend(
            "B2", "detector_bend_seed",
            anchor=(-195.9310, 51.9352, 247.0233),
            axis=(0.384117, -0.515143, 0.766213), span=98.3219, width=12.8521,
            confidence=0.8725, angle=90.9554,
        ),
        _bend(
            "B3", "zero_shot_strip_seed",
            anchor=(-316.1928, 95.9429, 278.1147),
            axis=(0.389147, -0.512490, 0.765453), span=199.3436, width=12.0071,
            confidence=1.0, angle=89.3761,
        ),
        _bend(
            "B4", "detector_bend_seed",
            anchor=(-324.0323, 56.7993, 255.4711),
            axis=(0.389188, -0.513957, 0.764448), span=169.5411, width=7.0627,
            confidence=0.7276, angle=159.77,
        ),
        _bend(
            "B5", "zero_shot_strip_seed",
            anchor=(-324.0323, 56.7993, 255.4711),
            axis=(0.389188, -0.513957, 0.764448), span=169.5411, width=12.0071,
            confidence=0.6, angle=20.23,
        ),
        _bend(
            "B6", "detector_bend_seed",
            anchor=(-290.8893, 30.5913, 337.4521),
            axis=(0.391979, 0.844695, 0.364476), span=49.1181, width=13.1919,
            confidence=0.8588, angle=90.2156,
        ),
        _bend(
            "B7", "detector_bend_seed",
            anchor=(-333.4759, 59.2760, 261.8863),
            axis=(0.388582, -0.513897, 0.764796), span=198.0452, width=7.6328,
            confidence=0.8146, angle=91.3848,
        ),
        _bend(
            "B8", "detector_bend_seed",
            anchor=(-334.9937, 55.9990, 260.4768),
            axis=(0.388552, -0.513962, 0.764768), span=198.6689, width=6.6973,
            confidence=0.8146, angle=105.0633,
        ),
        _bend(
            "B9", "zero_shot_strip_seed",
            anchor=(-211.3396, 55.8597, 221.7499),
            axis=(0.386350, -0.514186, 0.765732), span=26.6456, width=12.0071,
            confidence=1.0, angle=105.8188,
        ),
    ]

    result = select_strip_detector_completion(hypotheses, local_spacing_mm=2.0012)

    assert result.selected_bend_ids == ("B1", "B3", "B5", "B2", "B6")
    assert result.fragment_promotions == (("B9", "B2"),)
    assert result.rejected_duplicate_ids == ("B7", "B8", "B9")
