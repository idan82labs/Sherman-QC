"""Formula-authored native bent-sheet positives and tube negatives."""
import math
import os
import pytest

pytestmark=pytest.mark.skipif(os.environ.get("SHERMAN_AUTHORED_NATIVE_PHASE")!="1",
    reason="root supervised authored native gate")


def authored_shape():
    from tests.test_extract_step_topology_bend_axes_v3_native import authored_l_bend
    return authored_l_bend()


def save(shape,path):
    from tests.test_extract_step_topology_bend_axes_v3_native import write_step
    write_step(shape,path)


def test_rectangular_incumbent_is_preserved_verbatim(tmp_path):
    from scripts.extract_step_topology_bend_axes_v7 import extract_step_bend_axes as old
    from scripts.extract_step_topology_bend_axes_v8 import extract_step_bend_axes as new
    path=tmp_path/"authored-rectangle.step";save(authored_shape(),path)
    before,after=old(path),new(path)
    assert len(before["bend_axes"])==1
    assert after["bend_axes"]==before["bend_axes"]


def test_sloping_end_cap_outside_parameter_domain_remains_refused(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Pnt,gp_Trsf,gp_Ax1,gp_Dir
    from OCP.BRepCheck import BRepCheck_Analyzer
    from scripts.extract_step_topology_bend_axes_v7 import extract_step_bend_axes as old
    from scripts.extract_step_topology_bend_axes_v8 import extract_step_bend_axes as new
    tool=BRepPrimAPI_MakeBox(gp_Pnt(-100.,-100.,30.),200.,200.,100.).Shape()
    rotation=gp_Trsf();rotation.SetRotation(gp_Ax1(gp_Pnt(0.,0.,30.),gp_Dir(0.,1.,0.)),.12)
    tool=BRepBuilderAPI_Transform(tool,rotation,True).Shape()
    cutter=BRepAlgoAPI_Cut(authored_shape(),tool);cutter.Build();assert cutter.IsDone()
    assert BRepCheck_Analyzer(cutter.Shape()).IsValid()
    path=tmp_path/"authored-curved-trim.step";save(cutter.Shape(),path)
    before,after=old(path),new(path)
    assert before["bend_axes"]==[]
    diagnostic={"rejections":after["rejected_groups"],"graph": [{"face":r["face_index"],"reason":r.get("graph_strip_certificate",{}).get("reason"),"strict":r.get("graph_strip_certificate",{}).get("strict_refusal",{}).get("reason")} for r in after["partial_cylindrical_faces"]]}
    assert after["bend_axes"]==[],diagnostic
    assert all(r["graph_strip_certificate"]["reason"]=="PCURVE_TRIM_OUTSIDE_UNDERLYING_PARAMETER_DOMAIN" for r in after["partial_cylindrical_faces"]),diagnostic


def test_authored_exact_native_curved_pcurve_domain_is_certified():
    from OCP.Geom import Geom_CylindricalSurface
    from OCP.Geom2d import Geom2d_BezierCurve,Geom2d_Line
    from OCP.gp import gp_Ax3,gp_Pnt,gp_Dir,gp_Pnt2d,gp_Dir2d,gp_Ax2d
    from OCP.TColgp import TColgp_Array1OfPnt2d
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeWire,BRepBuilderAPI_MakeFace
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepLib import BRepLib
    from OCP.TopExp import TopExp
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from scripts.extract_step_topology_bend_axes_v8 import native_graph_certificate
    surface=Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0.,0.,0.),gp_Dir(0.,0.,1.)),3.)
    poles=TColgp_Array1OfPnt2d(1,3)
    for i,p in enumerate([(0.,0.),(math.pi/4,-2.),(math.pi/2,0.)],1):
        poles.SetValue(i,gp_Pnt2d(*p))
    wire=BRepBuilderAPI_MakeWire()
    curved=BRepBuilderAPI_MakeEdge(Geom2d_BezierCurve(poles),surface,0.,1.)
    assert curved.IsDone();wire.Add(curved.Edge())
    for p,q in [((math.pi/2,0.),(math.pi/2,30.)),((math.pi/2,30.),(0.,30.)),((0.,30.),(0.,0.))]:
        line=Geom2d_Line(gp_Ax2d(gp_Pnt2d(*p),gp_Dir2d(q[0]-p[0],q[1]-p[1])))
        edge=BRepBuilderAPI_MakeEdge(line,surface,0.,math.hypot(q[0]-p[0],q[1]-p[1]));assert edge.IsDone();wire.Add(edge.Edge())
    assert wire.IsDone()
    face=BRepBuilderAPI_MakeFace(surface,wire.Wire(),True).Face()
    assert BRepLib.BuildCurves3d_s(face)
    emap=TopTools_IndexedMapOfShape();TopExp.MapShapes_s(face,TopAbs_EDGE,emap)
    proof=native_graph_certificate(face,BRepAdaptor_Surface(face),emap)
    assert proof["status"]=="PROVED",proof
    assert proof["detection_only"] and not proof["unfold_certificate"]
    assert proof["safe_rectangular_interior_core_mm"][2:]==pytest.approx([0.,30.])
    assert any(len(s["poles"])==3 for s in proof["ordered_original_boundary"])


def test_closed_tube_does_not_become_a_bend(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from scripts.extract_step_topology_bend_axes_v8 import extract_step_bend_axes
    cut=BRepAlgoAPI_Cut(BRepPrimAPI_MakeCylinder(3.,40.).Shape(),BRepPrimAPI_MakeCylinder(2.,40.).Shape())
    cut.Build();assert cut.IsDone()
    path=tmp_path/"authored-tube.step";save(cut.Shape(),path)
    assert extract_step_bend_axes(path)["bend_axes"]==[]
