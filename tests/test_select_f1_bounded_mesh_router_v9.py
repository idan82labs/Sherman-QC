from scripts.select_f1_bounded_mesh_router_v9 import route


def test_route_preserves_non_mesh_selector() -> None:
    assert route(bounded_mesh=False, line_object_count=10) == "non_mesh_v6"
    assert route(bounded_mesh=False, line_object_count=1000) == "non_mesh_v6"


def test_route_splits_bounded_mesh_by_source_only_complexity() -> None:
    assert route(bounded_mesh=True, line_object_count=400) == "bounded_mesh_small_pool_recall"
    assert route(bounded_mesh=True, line_object_count=401) == "bounded_mesh_complex_pool_precision"
