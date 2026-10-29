import numpy as np

from backend.f1_parallel_skin_plane_snap import snap_stitches_to_supported_skin_plane


def _flange(identifier, normal, d, count):
    return {"flange_id": identifier, "normal": normal, "d": d, "atom_ids": [f"{identifier}-{i}" for i in range(count)]}


def test_snaps_stitch_to_higher_support_parallel_skin():
    selected = [{
        "bend_id": "IBS1",
        "candidate_source": "interface_birth_stitch",
        "canonical_flange_pair": ["BASE", "BACK"],
        "span_endpoints": [[-20, 0, 4], [20, 0, 4]],
    }]
    flanges = [
        _flange("BASE", [0, 0, 1], 0, 100),
        _flange("BACK", [0, 1, 0], -4, 10),
        _flange("SKIN", [0, 1, 0], 0, 30),
    ]
    result = snap_stitches_to_supported_skin_plane(selected, flanges)
    assert len(result["replacements"]) == 1
    endpoints = np.asarray(result["selected"][0]["span_endpoints"])
    assert np.allclose(endpoints[:, 1], 0.0)
    assert np.allclose(endpoints[:, 2], 0.0)


def test_fails_closed_for_large_separation_or_non_stitch():
    selected = [{
        "bend_id": "IBS1",
        "candidate_source": "interface_birth_stitch",
        "canonical_flange_pair": ["BASE", "BACK"],
        "span_endpoints": [[-20, 0, 8], [20, 0, 8]],
    }]
    flanges = [
        _flange("BASE", [0, 0, 1], 0, 100),
        _flange("BACK", [0, 1, 0], -8, 10),
        _flange("SKIN", [0, 1, 0], 0, 30),
    ]
    result = snap_stitches_to_supported_skin_plane(selected, flanges)
    assert result["replacements"] == []
    ordinary = snap_stitches_to_supported_skin_plane(
        [{**selected[0], "candidate_source": "detector"}], flanges
    )
    assert ordinary["replacements"] == []


def test_prunes_stitch_without_minimum_interface_support():
    selected = [{
        "bend_id": "IBS1", "candidate_source": "interface_birth_stitch",
        "canonical_flange_pair": ["BASE", "TINY"], "span_endpoints": [[-10, 0, 0], [10, 0, 0]],
    }]
    flanges = [_flange("BASE", [0, 0, 1], 0, 100), _flange("TINY", [0, 1, 0], 0, 3)]
    result = snap_stitches_to_supported_skin_plane(selected, flanges)
    assert result["selected"] == []
    assert result["pruned"][0]["bend_id"] == "IBS1"
