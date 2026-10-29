import inspect

from backend.f1_surface_wedge_completion import complete_surface_wedges


def _edge(candidate_id, first, second, start, end, **extra):
    return {
        "candidate_id": candidate_id,
        "surface1_id": first,
        "surface2_id": second,
        "span_endpoints": [start, end],
        **extra,
    }


def test_completes_selected_surface_wedge_with_consensus_mesh_edge():
    selected = [
        _edge("base-left", 10, 20, [0, 0, 0], [0, 10, 0]),
        _edge("base-right", 10, 30, [1, 0, 0], [1, 10, 0]),
    ]
    completion = _edge(
        "scan_mesh_adjacency::SMA3",
        20,
        30,
        [20, 0, 0],
        [20, 20, 0],
        representative_source="scan_mesh_adjacency",
        cluster_source_count=2,
    )

    result = complete_surface_wedges(selected, [completion])

    assert [record["candidate_id"] for record in result["selected"]] == [
        "base-left",
        "base-right",
        "scan_mesh_adjacency::SMA3",
    ]
    assert result["added"][0]["cluster_source_count"] == 2
    assert result["selected"][-1]["selection_reason"] == "surface_wedge_completion"


def test_allows_short_single_source_but_rejects_long_one():
    selected = [
        _edge("base-left", 10, 20, [0, 0, 0], [0, 10, 0]),
        _edge("base-right", 10, 30, [1, 0, 0], [1, 10, 0]),
    ]
    short = _edge(
        "scan_mesh_adjacency::short", 20, 30, [20, 0, 0], [20, 8, 0]
    )
    long = _edge(
        "scan_mesh_adjacency::long", 20, 30, [30, 0, 0], [30, 20, 0]
    )

    result = complete_surface_wedges(selected, [long, short])

    assert [item["candidate_id"] for item in result["added"]] == [
        "scan_mesh_adjacency::short"
    ]


def test_rejects_nonmesh_nonwedge_duplicate_and_ambiguous_selected_edges():
    selected = [
        _edge("base-left", 10, 20, [0, 0, 0], [0, 10, 0]),
        _edge("base-right", 10, 30, [1, 0, 0], [1, 10, 0]),
    ]
    nonmesh = _edge("plane-lattice", 20, 30, [20, 0, 0], [20, 8, 0])
    nonwedge = _edge(
        "scan_mesh_adjacency::outside", 20, 40, [30, 0, 0], [30, 8, 0]
    )
    duplicate = _edge(
        "scan_mesh_adjacency::duplicate", 20, 30, [0, 0, 0], [0, 10, 0]
    )

    rejected = complete_surface_wedges(
        selected, [nonmesh, nonwedge, duplicate]
    )
    ambiguous = complete_surface_wedges(
        [selected[0], dict(selected[0]), selected[1]],
        [
            _edge(
                "scan_mesh_adjacency::candidate",
                20,
                30,
                [20, 0, 0],
                [20, 8, 0],
            )
        ],
    )

    assert rejected["added"] == []
    assert ambiguous["added"] == []


def test_is_count_free_nonrecursive_and_deterministic():
    signature = inspect.signature(complete_surface_wedges)
    assert all(
        forbidden not in name
        for name in signature.parameters
        for forbidden in ("expected", "target", "bend_count")
    )
    selected = [
        _edge("a-b", "a", "b", [0, 0, 0], [0, 10, 0]),
        _edge("a-c", "a", "c", [1, 0, 0], [1, 10, 0]),
    ]
    first = _edge(
        "scan_mesh_adjacency::b-c",
        "b",
        "c",
        [20, 0, 0],
        [20, 20, 0],
        cluster_source_count=2,
    )
    second = _edge(
        "scan_mesh_adjacency::c-d",
        "c",
        "d",
        [30, 0, 0],
        [30, 20, 0],
        cluster_source_count=3,
    )

    forward = complete_surface_wedges(selected, [first, second])
    reverse = complete_surface_wedges(selected, [second, first])

    assert forward["added"] == reverse["added"]
    assert [item["candidate_id"] for item in forward["added"]] == [
        "scan_mesh_adjacency::b-c"
    ]
