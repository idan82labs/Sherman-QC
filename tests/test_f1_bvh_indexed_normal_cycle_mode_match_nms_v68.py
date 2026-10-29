import numpy as np

import backend.f1_bvh_indexed_normal_cycle_mode_match_nms_v68 as v68
import backend.f1_similarity_invariant_normal_cycle_mode_match_nms_v67 as v67
from tests.test_f1_similarity_invariant_normal_cycle_mode_match_nms_v67 import folded_strip, refine_triangles, strip_rows


def semantic(result):
    audit = result["ncmm_nms_v1_audit"]
    return {
        "predictions": result["predictions"],
        "pair_records": audit.get("pair_records"),
        "selected_matching": audit.get("selected_matching"),
        "suppressed_hashes": audit.get("suppressed_hashes"),
        "survivor_hashes": audit.get("survivor_hashes"),
    }


def test_bvh_query_is_conservative_for_crossing_segments():
    atoms = [
        v67.Atom(np.asarray((-1, 0, 0.0)), np.asarray((1, 0, 1.0)), 1.0, 1.0, "C0"),
        v67.Atom(np.asarray((5, 5, 5.0)), np.asarray((6, 6, 6.0)), 1.0, 1.0, "C0"),
    ]
    bvh = v68.AtomBVH(atoms)
    found = bvh.query(np.asarray((-0.1, -0.1, -0.1)), np.asarray((0.1, 0.1, 1.1)))
    assert found == [atoms[0]]


def test_indexed_and_bruteforce_f01_are_semantically_identical():
    vertices, faces = folded_strip()
    brute, _ = v67.apply_ncmm_nms_v1(strip_rows(), vertices, faces)
    indexed, _ = v68.apply_indexed_ncmm_nms_v1(strip_rows(), vertices, faces)
    assert semantic(indexed) == semantic(brute)


def test_indexed_replay_similarity_winding_and_refinement_are_invariant():
    vertices, faces = folded_strip()
    primary, _ = v68.apply_indexed_ncmm_nms_v1(strip_rows(), vertices, faces)
    replay, _ = v68.apply_indexed_ncmm_nms_v1(strip_rows(), vertices, faces)
    assert semantic(primary) == semantic(replay)

    for scale in (0.8, 1.25):
        current_vertices, current_faces = folded_strip(scale=scale)
        current, _ = v68.apply_indexed_ncmm_nms_v1(strip_rows(scale), current_vertices, current_faces, variant=True)
        expected = semantic(primary)
        observed = semantic(current)
        assert [row["route_hash"] for row in observed["predictions"]] == [row["route_hash"] for row in expected["predictions"]]
        for key in ("pair_records", "selected_matching", "suppressed_hashes", "survivor_hashes"):
            assert observed[key] == expected[key]

    winding, _ = v68.apply_indexed_ncmm_nms_v1(strip_rows(), vertices, faces[::-1, ::-1], variant=True)
    assert semantic(winding) == semantic(primary)
    rv, rf = refine_triangles(vertices, faces)
    refined, _ = v68.apply_indexed_ncmm_nms_v1(strip_rows(), rv, rf, variant=True)
    assert semantic(refined) == semantic(primary)
