from backend.cad_finite_trimmed_cylinder_domain_v1 import certify_graph_strip, certify_single_edge_end_join_core


def fixture():
    points = [[[0, 0], [0, 10]], [[1e-12, 10], [1, 12], [2-1e-12, 10]],
              [[2, 10], [2, 0]], [[2-1e-12, 0], [1, -2], [1e-12, 0]]]
    spans = [dict(poles=p, weights=[1.] * len(p), native_edge_id=i+1,
                  actual_interval_proven=True, native_start_vertex_id=i+1,
                  native_end_vertex_id=(i+1)%4+1) for i,p in enumerate(points)]
    joins = [dict(shared_vertex_id=(i+1)%4+1, source_tolerance_mm=1e-7, chart_gap_budget_mm=2e-7) for i in range(4)]
    return [spans], joins


def run(wires, joins):
    return certify_single_edge_end_join_core(wires, native_brep_valid=True, exact_active_domains=True, join_receipts=joins)


def test_tiny_native_endpoint_discrepancy_keeps_actual_geometry_and_excludes_sides():
    wires, joins = fixture()
    assert certify_graph_strip(wires, native_brep_valid=True, exact_active_domains=True)["status"] == "REFUSED"
    result = run(wires, joins)
    assert result["status"] == "PROVED"
    assert result["proof_kind"] == "NATIVE_TOLERANCE_QUALIFIED_INTERIOR_CORE"
    assert result["full_boundary_geometry_proven"] is False
    assert result["ordered_original_boundary"][1]["poles"] == wires[0][1]["poles"]
    assert result["angular_support_metric_mm"][0] > 2e-7
    assert result["angular_support_metric_mm"][1] < 2-2e-7


def test_shared_vertex_identity_and_budget_are_mandatory():
    wires, joins = fixture()
    wires[0][1]["native_start_vertex_id"] = 99
    assert run(wires, joins)["reason"] == "NATIVE_JOIN_VERTEX_IDENTITY_REQUIRED"
    wires, joins = fixture(); joins[0]["chart_gap_budget_mm"] = 1e-3
    assert run(wires, joins)["reason"] == "NATIVE_SOURCE_JOIN_BUDGET_INVALID"
    wires, joins = fixture(); wires[0][1]["poles"][0][0] = 1e-4
    assert run(wires, joins)["reason"] == "ACTUAL_JOIN_GAP_EXCEEDS_NATIVE_BUDGET"


def test_internal_native_edge_gap_never_qualified():
    wires, joins = fixture()
    wires[0][1:2] = [dict(poles=[[1e-12,10],[1,11]], weights=[1,1], native_edge_id=2, actual_interval_proven=True, native_start_vertex_id=2),
                     dict(poles=[[1+1e-12,11],[2-1e-12,10]], weights=[1,1], native_edge_id=2, actual_interval_proven=True, native_end_vertex_id=3)]
    joins.insert(1, None)
    assert run(wires, joins)["reason"] == "ORDERED_WIRE_NOT_EXACTLY_CLOSED"


def test_exact_internal_bezier_junction_supported():
    wires, joins = fixture()
    wires[0][1:2] = [dict(poles=[[1e-12,10],[1,11]], weights=[1,1], native_edge_id=2, actual_interval_proven=True, native_start_vertex_id=2),
                     dict(poles=[[1,11],[2-1e-12,10]], weights=[1,1], native_edge_id=2, actual_interval_proven=True, native_end_vertex_id=3)]
    joins.insert(1, None)
    assert run(wires, joins)["status"] == "PROVED"


def test_multinative_edge_end_chain_refuses_even_with_valid_join_receipt():
    wires, joins = fixture()
    wires[0][1:2] = [dict(poles=[[1e-12,10],[1,11]], weights=[1,1], native_edge_id=2, actual_interval_proven=True, native_start_vertex_id=2, native_end_vertex_id=5),
                     dict(poles=[[1,11],[2-1e-12,10]], weights=[1,1], native_edge_id=5, actual_interval_proven=True, native_start_vertex_id=5, native_end_vertex_id=3)]
    joins.insert(1, dict(shared_vertex_id=5, source_tolerance_mm=1e-7, chart_gap_budget_mm=2e-7))
    assert run(wires, joins)["reason"] == "QUALIFIED_END_CHAIN_MUST_BE_ONE_NATIVE_EDGE"
