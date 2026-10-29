from backend.f1_calibrated_multiview_birth_only_v59 import compose_v59


def test_v59_preserves_v52_and_appends_only_raw_births() -> None:
    incumbent = {"bend_id": "A", "line_start": [0, 0, 0], "line_end": [1, 0, 0]}
    birth = {"bend_id": "B", "line_start": [0, 1, 0], "line_end": [1, 1, 0], "selection_route": "calibrated_multiview_turn_consensus_v58"}
    v52 = {"source_only": True, "research_only": True, "v52_result_sha256": "v52", "predictions": [incumbent]}
    v58 = {"source_only": True, "research_only": True, "v58_result_sha256": "v58", "predictions": [birth], "pruned_v52": [{"v52_ordinal": 0}]}
    result = compose_v59(v52, v58)
    assert result["predictions"] == [incumbent, birth]
    assert result["audit"] == {"retained_v52_count": 1, "raw_new_count": 1, "v58_prunes_ignored_for_visual_safety": 1, "retained_v52_byte_preservation": True}
