from scripts.select_f1_mesh_route_v16 import (
    ROUTE_COMPLEX,
    ROUTE_FULL,
    ROUTE_NON_MESH,
    ROUTE_SMALL,
    classify_route,
)
from scripts.train_f1_bounded_mesh_router_v9 import SMALL_POOL_MAXIMUM


def _pool(*, available: bool, bounded: bool) -> dict:
    return {
        "generator_diagnostics": {
            "scan_mesh_adjacency": {
                "available": available,
                "bounded_decimation_used": bounded,
            }
        }
    }


def test_native_full_resolution_mesh_never_enters_bounded_v15_lane() -> None:
    assert classify_route(_pool(available=True, bounded=False), 790) == ROUTE_FULL


def test_bounded_small_pool_keeps_recall_lane() -> None:
    assert classify_route(_pool(available=True, bounded=True), SMALL_POOL_MAXIMUM) == ROUTE_SMALL


def test_bounded_complex_pool_uses_v15_lane() -> None:
    assert classify_route(_pool(available=True, bounded=True), SMALL_POOL_MAXIMUM + 1) == ROUTE_COMPLEX


def test_unavailable_mesh_keeps_non_mesh_lane_even_with_large_pool() -> None:
    assert classify_route(_pool(available=False, bounded=False), 1000) == ROUTE_NON_MESH
