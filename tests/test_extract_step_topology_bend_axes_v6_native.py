"""Authored pcurve representation qualification; root-supervised native only."""
import math
import os
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('SHERMAN_AUTHORED_NATIVE_PHASE')!='1',
    reason='root-coordinated bounded authored-native phase required')


def cylindrical_face(points,spline_edge=None,curved=False):
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.Geom2d import Geom2d_Line,Geom2d_BSplineCurve
    from OCP.gp import gp_Ax3,gp_Pnt,gp_Dir,gp_Pnt2d,gp_Dir2d,gp_Ax2d
    from OCP.TColgp import TColgp_Array1OfPnt2d
    from OCP.TColStd import TColStd_Array1OfReal,TColStd_Array1OfInteger
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeWire,BRepBuilderAPI_MakeFace
    surface=Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0.,0.,0.),gp_Dir(0.,0.,1.)),3.)
    wire=BRepBuilderAPI_MakeWire()
    for index,(p,q) in enumerate(zip(points,points[1:]+points[:1])):
        if index==spline_edge:
            poles=TColgp_Array1OfPnt2d(1,3); weights=TColStd_Array1OfReal(1,3)
            midpoint=[(p[0]+q[0])/2,(p[1]+q[1])/2]
            if curved: midpoint[1]+=float(curved)
            for i,value in enumerate((p,midpoint,q),1):
                poles.SetValue(i,gp_Pnt2d(*value)); weights.SetValue(i,2. if i==2 else 1.)
            knots=TColStd_Array1OfReal(1,2); knots.SetValue(1,0.); knots.SetValue(2,1.)
            mult=TColStd_Array1OfInteger(1,2); mult.SetValue(1,3); mult.SetValue(2,3)
            curve=Geom2d_BSplineCurve(poles,weights,knots,mult,2,False)
            edge=BRepBuilderAPI_MakeEdge(curve,surface,0.,1.)
        else:
            distance=math.hypot(q[0]-p[0],q[1]-p[1])
            curve=Geom2d_Line(gp_Ax2d(gp_Pnt2d(*p),gp_Dir2d(q[0]-p[0],q[1]-p[1])))
            edge=BRepBuilderAPI_MakeEdge(curve,surface,0.,distance)
        assert edge.IsDone(); wire.Add(edge.Edge())
    assert wire.IsDone()
    face=BRepBuilderAPI_MakeFace(surface,wire.Wire(),True)
    assert face.IsDone()
    return face.Face()


def domain_certificate(face):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from scripts.extract_step_topology_bend_axes_v6 import _domain_certificate
    indexed=TopTools_IndexedMapOfShape(); uses=[]; explorer=TopExp_Explorer(face,TopAbs_EDGE)
    while explorer.More(): uses.append(indexed.Add(explorer.Current())); explorer.Next()
    return _domain_certificate(face,BRepAdaptor_Surface(face),1,uses)


def test_authored_native_split_collinear_cylinder_boundary():
    face=cylindrical_face([(0.,0.),(math.pi/4,0.),(math.pi/2,0.),
        (math.pi/2,40.),(0.,40.)])
    proof=domain_certificate(face)
    assert proof['status']=='PROVED',proof
    assert proof['segment_count']==5 and len(proof['native_edge_ids'])==5
    assert proof['bounds_metric_mm']==pytest.approx([0.,3*math.pi/2,0.,40.])


def test_authored_exact_straight_positive_rational_pcurve():
    face=cylindrical_face([(0.,0.),(math.pi/2,0.),(math.pi/2,40.),(0.,40.)],spline_edge=0)
    proof=domain_certificate(face)
    assert proof['status']=='PROVED',proof
    assert any('BSplineCurve' in name for name in proof['pcurve_types'])
    assert proof['type_proven'] and proof['geometry_sampled'] is False


def test_curved_positive_rational_boundary_is_not_straight_by_samples():
    face=cylindrical_face([(0.,0.),(math.pi/2,0.),(math.pi/2,40.),(0.,40.)],spline_edge=0,curved=True)
    proof=domain_certificate(face)
    assert proof['status']=='REFUSED'
    assert proof['reason']=='PCURVE_OUTSIDE_CORRIDOR_OR_BACKTRACK'


def test_nonrectangular_authored_native_material_not_bounding_box():
    face=cylindrical_face([(0.,0.),(math.pi/2,0.),(math.pi/2,40.),
        (math.pi/4,40.),(math.pi/4,20.),(0.,20.)])
    proof=domain_certificate(face)
    assert proof['status']=='REFUSED'


def test_authored_inner_wire_hole_is_explicitly_unsupported():
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.TopAbs import TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    outer=cylindrical_face([(0.,0.),(math.pi/2,0.),(math.pi/2,40.),(0.,40.)])
    inner=cylindrical_face([(.3,10.),(.6,10.),(.6,20.),(.3,20.)])
    wires=TopExp_Explorer(inner,TopAbs_WIRE); assert wires.More()
    maker=BRepBuilderAPI_MakeFace(outer)
    maker.Add(TopoDS.Wire_s(wires.Current().Reversed()))
    assert maker.IsDone()
    proof=domain_certificate(maker.Face())
    assert proof['status']=='REFUSED' and proof['wire_count']==2
    assert proof['reason']=='MULTIPLE_WIRES_OR_HOLES_UNSUPPORTED'



def test_native_positive_rational_axis_corridor_within_same_source_tolerance():
    face=cylindrical_face([(0.,0.),(math.pi/2,0.),(math.pi/2,40.),(0.,40.)],
        spline_edge=0,curved=2e-8)
    proof=domain_certificate(face)
    assert proof['status']=='PROVED',proof
    assert proof['certificate_schema_version']==6
    assert proof['error_budget']['status']=='PROVED'
    assert 0<proof['error_budget']['maximum_coefficient_corridor_mm']<=proof['declared_tolerance_mm']/4
    assert proof['maximum_residual_mm']==proof['error_budget']['total_bound_mm']
    assert type(proof['maximum_residual_mm']) is float
    assert proof['maximum_residual_mm']<=proof['declared_tolerance_mm']


def test_native_positive_rational_axis_corridor_beyond_source_budget_refuses():
    face=cylindrical_face([(0.,0.),(math.pi/2,0.),(math.pi/2,40.),(0.,40.)],
        spline_edge=0,curved=1e-6)
    proof=domain_certificate(face)
    assert proof['status']=='REFUSED'
    assert proof['reason']=='PCURVE_OUTSIDE_CORRIDOR_OR_BACKTRACK'
