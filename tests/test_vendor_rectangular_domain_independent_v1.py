"""Independent metric-domain fixtures; no native imports or physical inputs."""
import copy
import math
import pytest
from scripts.extract_step_topology_bend_axes_v5 import certify_rectangle_segments


def boundary(points):
    return [{"start": list(p), "end": list(points[(i+1)%len(points)]), "edge_id": i+1}
            for i,p in enumerate(points)]


@pytest.mark.parametrize("scale", [0.01, 1., 1000.])
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_split_rectangle_metric_translation_scale_orientation(scale, reverse):
    points=[(0,0),(1,0),(3,0),(3,2),(3,4),(2,4),(0,4),(0,1)]
    points=[(2000+scale*x,-5000+scale*y) for x,y in points]
    if reverse: points.reverse()
    segments=boundary(points); original=copy.deepcopy(segments)
    result=certify_rectangle_segments(segments,1e-7)
    assert result['status']=='PROVED'
    assert result['maximum_residual_mm']==0
    assert result['simple_connected_outer_loop'] and result['complete_side_coverage']
    assert segments==original


@pytest.mark.parametrize("points", [
    [(0,0),(4,0),(4,4),(2,4),(2,2),(0,2)],
    [(0,0),(4,4),(4,0),(0,4)],
    [(0,0),(4,0),(2,0),(4,0),(4,4),(0,4)],
    [(0,0),(4,0),(4,4),(0,4),(0,0)],
])
def test_notches_crossings_backtracking_and_zero_edges_refuse(points):
    assert certify_rectangle_segments(boundary(points),1e-7)['status']=='REFUSED'


def test_hole_is_not_discarded_or_bbox_promoted():
    outer=boundary([(0,0),(10,0),(10,10),(0,10)])
    inner=boundary([(3,3),(7,3),(7,7),(3,7)])
    for row in inner: row['edge_id']+=4
    assert certify_rectangle_segments(outer+inner,1e-7)['status']=='REFUSED'


def test_metric_gap_above_tolerance_refuses():
    rows=boundary([(0,0),(3,0),(3,4),(0,4)])
    rows[1]['start'][0]+=2e-6
    assert certify_rectangle_segments(rows,1e-7)['status']=='REFUSED'


def test_repeat_identity_refuses_even_identical_locus():
    rows=boundary([(0,0),(3,0),(3,4),(0,4)])
    rows[1]['edge_id']=rows[0]['edge_id']
    assert certify_rectangle_segments(rows,1e-7)['status']=='REFUSED'


@pytest.mark.parametrize('tol',[0.,-1.,1.0001e-5,float('nan'),float('inf'),True])
def test_invalid_declared_physical_tolerance_refuses(tol):
    assert certify_rectangle_segments(boundary([(0,0),(3,0),(3,4),(0,4)]),tol)['status']=='REFUSED'


def test_cylinder_metric_radius_does_not_disappear():
    # Same angular endpoint error has distinct physical lengths R*du.
    for radius,expected in [(1.,'PROVED'),(100.,'REFUSED')]:
        rows=boundary([(0,0),(3*radius,0),(3*radius,4),(0,4)])
        rows[1]['start'][0]+=radius*5e-8
        assert certify_rectangle_segments(rows,1e-7)['status']==expected


def test_segment_budget_refuses_without_native_work():
    rows=boundary([(0,0),(3,0),(3,4),(0,4)])*65
    assert certify_rectangle_segments(rows,1e-7)['status']=='REFUSED'
