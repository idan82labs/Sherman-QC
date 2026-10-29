"""Authored active-spline interval fixtures; root-supervised execution only."""
import math
import os
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('SHERMAN_AUTHORED_NATIVE_PHASE')!='1',
    reason='root-coordinated bounded authored-native phase required')


def piecewise_curve():
    from OCP.Geom2d import Geom2d_BSplineCurve
    from OCP.gp import gp_Pnt2d
    from OCP.TColgp import TColgp_Array1OfPnt2d
    from OCP.TColStd import TColStd_Array1OfReal,TColStd_Array1OfInteger
    # Two quadratic Bezier spans: macroscopically curved first, straight second.
    poles=TColgp_Array1OfPnt2d(1,5)
    for i,p in enumerate([(-2.,5.),(-1.,10.),(0.,0.),(math.pi/4,0.),(math.pi/2,0.)],1):
        poles.SetValue(i,gp_Pnt2d(*p))
    weights=TColStd_Array1OfReal(1,5)
    for i in range(1,6): weights.SetValue(i,1.)
    knots=TColStd_Array1OfReal(1,3); mult=TColStd_Array1OfInteger(1,3)
    for i,(k,m) in enumerate([(0.,3),(1.,2),(2.,3)],1):
        knots.SetValue(i,k); mult.SetValue(i,m)
    return Geom2d_BSplineCurve(poles,weights,knots,mult,2,False)


def coefficient_proof(curve):
    from scripts.extract_step_topology_bend_axes_v7 import certify_axis_corridor
    return certify_axis_corridor([[3*curve.Pole(i).X(),curve.Pole(i).Y()]
        for i in range(1,curve.NbPoles()+1)],
        [curve.Weight(i) for i in range(1,curve.NbPoles()+1)],1e-7)


def test_actual_straight_span_restricted_coefficients_prove_without_carrier_mutation():
    from scripts.extract_step_topology_bend_axes_v7 import restrict_bspline_actual_interval
    original=piecewise_curve()
    assert coefficient_proof(original)['status']=='REFUSED'
    restricted,receipt=restrict_bspline_actual_interval(original,1.,2.)
    assert receipt['status']=='PROVED',receipt
    assert coefficient_proof(restricted)['status']=='PROVED'
    assert original.NbPoles()==5 and original.FirstParameter()==0.
    assert restricted.NbPoles()==3


def test_actual_curved_span_does_not_become_rectangle():
    from scripts.extract_step_topology_bend_axes_v7 import restrict_bspline_actual_interval
    restricted,receipt=restrict_bspline_actual_interval(piecewise_curve(),0.,1.)
    assert receipt['status']=='PROVED'
    assert coefficient_proof(restricted)['status']=='REFUSED'


def test_outside_underlying_knots_is_not_extrapolated():
    from scripts.extract_step_topology_bend_axes_v7 import restrict_bspline_actual_interval
    restricted,receipt=restrict_bspline_actual_interval(piecewise_curve(),-1.,1.)
    assert restricted is None
    assert receipt['reason']=='PCURVE_TRIM_OUTSIDE_UNDERLYING_PARAMETER_DOMAIN'


def test_native_face_uses_only_actual_straight_edge_interval():
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.Geom2d import Geom2d_Line
    from OCP.gp import gp_Ax3,gp_Pnt,gp_Dir,gp_Pnt2d,gp_Dir2d,gp_Ax2d
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeWire,BRepBuilderAPI_MakeFace
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from scripts.extract_step_topology_bend_axes_v7 import _domain_certificate
    surface=Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0.,0.,0.),gp_Dir(0.,0.,1.)),3.)
    wire=BRepBuilderAPI_MakeWire()
    edge=BRepBuilderAPI_MakeEdge(piecewise_curve(),surface,1.,2.)
    assert edge.IsDone(); wire.Add(edge.Edge())
    for p,q in [((math.pi/2,0.),(math.pi/2,40.)),
                ((math.pi/2,40.),(0.,40.)),((0.,40.),(0.,0.))]:
        curve=Geom2d_Line(gp_Ax2d(gp_Pnt2d(*p),gp_Dir2d(q[0]-p[0],q[1]-p[1])))
        maker=BRepBuilderAPI_MakeEdge(curve,surface,0.,math.hypot(q[0]-p[0],q[1]-p[1]))
        assert maker.IsDone(); wire.Add(maker.Edge())
    assert wire.IsDone()
    maker=BRepBuilderAPI_MakeFace(surface,wire.Wire(),True); assert maker.IsDone()
    face=maker.Face(); indexed=TopTools_IndexedMapOfShape(); uses=[]
    explorer=TopExp_Explorer(face,TopAbs_EDGE)
    while explorer.More(): uses.append(indexed.Add(explorer.Current())); explorer.Next()
    from scripts.extract_step_topology_bend_axes_v6 import _domain_certificate as old_certificate
    old=old_certificate(face,BRepAdaptor_Surface(face),1,uses)
    assert old['status']=='REFUSED' and old['reason']=='PCURVE_OUTSIDE_CORRIDOR_OR_BACKTRACK',old
    receipt=_domain_certificate(face,BRepAdaptor_Surface(face),1,uses)
    assert receipt['status']=='PROVED',receipt
    assert receipt['certificate_schema_version']==7
    proof=next(p for p in receipt['coefficient_proofs'] if 'active_interval_proof' in p)
    assert proof['active_interval_proof']['actual_trim_interval']==[1.,2.]
    assert receipt['bounds_metric_mm']==pytest.approx([0.,3*math.pi/2,0.,40.])
