"""Authored metric-boundary truth; no native objects or data."""
import copy
import math
import pytest
from scripts.extract_step_topology_bend_axes_v5 import certify_rectangle_segments


def segments(points):
    return [{'start':list(p),'end':list(q),'edge_id':i+1}
        for i,(p,q) in enumerate(zip(points,points[1:]+points[:1]))]


def rectangle():
    return segments([(0.,0.),(3.,0.),(3.,40.),(0.,40.)])


def test_actual_split_collinear_side_union_has_full_native_boundary():
    rows=segments([(0.,0.),(1.,0.),(3.,0.),(3.,12.),(3.,40.),(0.,40.),(0.,21.)])
    old=copy.deepcopy(rows); proof=certify_rectangle_segments(rows,1e-7)
    assert proof['status']=='PROVED' and rows==old
    assert proof['bounds_metric_mm']==[0.,3.,0.,40.]
    assert proof['edge_ids']==list(range(1,8))
    assert proof['simple_connected_outer_loop'] and proof['complete_side_coverage']
    assert proof['geometry_sampled'] is False


def test_input_order_and_segment_orientation_do_not_change_material_domain():
    rows=rectangle(); expected=certify_rectangle_segments(rows,1e-7)['bounds_metric_mm']
    rows=[{**r,'start':r['end'],'end':r['start']} for r in reversed(rows)]
    assert certify_rectangle_segments(rows,1e-7)['bounds_metric_mm']==expected


@pytest.mark.parametrize('points',[
    [(0.,0.),(3.,0.),(3.,40.),(1.,40.),(1.,20.),(0.,20.)],
    [(0.,0.),(3.,0.),(2.,40.),(0.,40.)],
    [(0.,0.),(3.,0.),(3.,40.),(0.,40.),(0.,20.),(1.,20.),(0.,20.)]])
def test_nonrectangle_oblique_scallop_and_retrace_are_not_bbox_material(points):
    assert certify_rectangle_segments(segments(points),1e-7)['status']=='REFUSED'


def test_gap_above_source_tolerance_is_not_closed_by_global_rounding():
    rows=rectangle(); rows[1]['start'][0]+=1e-5
    assert certify_rectangle_segments(rows,1e-7)['status']=='REFUSED'


def test_proven_endpoint_join_residual_is_not_reported_as_zero():
    rows=rectangle(); rows[1]['start'][1]=5e-8
    proof=certify_rectangle_segments(rows,1e-7)
    assert proof['status']=='PROVED'
    assert proof['maximum_residual_mm']==pytest.approx(5e-8)


def test_multiple_loops_hole_and_duplicate_native_edge_refuse():
    rows=rectangle()+segments([(1.,1.),(2.,1.),(2.,2.),(1.,2.)])
    for i,row in enumerate(rows): row['edge_id']=i+1
    assert certify_rectangle_segments(rows,1e-7)['status']=='REFUSED'
    rows=rectangle(); rows[1]['edge_id']=rows[0]['edge_id']
    assert certify_rectangle_segments(rows,1e-7)['status']=='REFUSED'


@pytest.mark.parametrize('tolerance',[0.,-1.,math.nan,math.inf,1e-4])
def test_invalid_or_overbroad_source_tolerance_refused(tolerance):
    assert certify_rectangle_segments(rectangle(),tolerance)['status']=='REFUSED'


def test_nonfinite_budget_and_ambiguous_endpoint_branch_refuse():
    rows=rectangle(); rows[0]['start'][0]=math.nan
    assert certify_rectangle_segments(rows,1e-7)['status']=='REFUSED'
    assert certify_rectangle_segments(rectangle()*65,1e-7)['status']=='REFUSED'
    rows=rectangle(); rows.append({'start':[0.,0.],'end':[1.,1.],'edge_id':5})
    assert certify_rectangle_segments(rows,1e-7)['status']=='REFUSED'
