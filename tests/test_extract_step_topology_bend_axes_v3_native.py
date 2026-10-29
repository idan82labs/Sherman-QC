"""Authored analytic STEP fixtures; execution requires explicit native-phase gate.

Truth is the construction formula, never an incumbent extractor result. Authoring
this file does not authorize native execution or corpus access.
"""
import math
import os
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('SHERMAN_AUTHORED_NATIVE_PHASE')!='1',
    reason='root-coordinated bounded authored-native phase required')


def authored_l_bend():
    from OCP.gp import gp_Pnt,gp_Vec
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeWire,BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    point=lambda xy:gp_Pnt(xy[0],xy[1],0.)
    edges=[]
    path=[(2.,0.),(2.,-10.),(3.,-10.),(3.,0.)]
    for p,q in zip(path,path[1:]):
        edges.append(BRepBuilderAPI_MakeEdge(point(p),point(q)).Edge())
    outer=GC_MakeArcOfCircle(point((3.,0.)),point((3/math.sqrt(2),3/math.sqrt(2))),point((0.,3.))).Value()
    edges.append(BRepBuilderAPI_MakeEdge(outer).Edge())
    path=[(0.,3.),(-10.,3.),(-10.,2.),(0.,2.)]
    for p,q in zip(path,path[1:]):
        edges.append(BRepBuilderAPI_MakeEdge(point(p),point(q)).Edge())
    # Reversed inner arc is authored independently: native circle axis is -Z.
    inner=GC_MakeArcOfCircle(point((0.,2.)),point((math.sqrt(2),math.sqrt(2))),point((2.,0.))).Value()
    edges.append(BRepBuilderAPI_MakeEdge(inner).Edge())
    wire=BRepBuilderAPI_MakeWire()
    for edge in edges: wire.Add(edge)
    assert wire.IsDone()
    face=BRepBuilderAPI_MakeFace(wire.Wire())
    assert face.IsDone()
    return BRepPrimAPI_MakePrism(face.Face(),gp_Vec(0.,0.,40.)).Shape()


def write_step(shape,path):
    from OCP.STEPControl import STEPControl_Writer,STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
    writer=STEPControl_Writer()
    assert writer.Transfer(shape,STEPControl_AsIs)==IFSelect_RetDone
    assert writer.Write(str(path))==IFSelect_RetDone


def assert_one_bend(result,expected):
    assert len(result['bend_axes'])==1, result['rejected_groups']
    bend=result['bend_axes'][0]
    assert bend['radii_mm']==pytest.approx([2.,3.])
    assert bend['axial_span_mm']==pytest.approx(40.)
    assert bend['angular_sweeps_deg']==pytest.approx([90.])
    got=np.array([bend['canonical_line_start'],bend['canonical_line_end']])
    # Endpoint ordering is a convention; compare unordered finite geometry.
    assert min(np.max(np.abs(got-expected)),np.max(np.abs(got-expected[::-1])))<1e-6
    assert bend['domain_receipt']['tangent_flange_rails_proven']
    assert bend['canonical_line_rule']=='GEOMETRIC_MID_THICKNESS_HALF_ANGULAR_TURN'


def test_authored_l_negative_inner_axis_actual_step_parse(tmp_path):
    from scripts.extract_step_topology_bend_axes_v3 import extract_step_bend_axes
    path=tmp_path/'authored-l.step'; write_step(authored_l_bend(),path)
    mid=2.5/math.sqrt(2)
    assert_one_bend(extract_step_bend_axes(path),np.array([[mid,mid,0.],[mid,mid,40.]]))


def test_transformed_l_actual_step_parse_and_folded_complete_faces(tmp_path):
    from OCP.gp import gp_Trsf,gp_Ax1,gp_Pnt,gp_Dir,gp_Vec
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from scripts.extract_step_topology_bend_axes_v3 import extract_step_bend_axes
    from backend.vendor_native_folded_mesh_v1 import read_folded_native
    # Proper 180-degree rotation reverses world forming axis, without changing
    # surface winding or manufacturing geometry.
    trsf=gp_Trsf(); trsf.SetRotation(gp_Ax1(gp_Pnt(0.,0.,0.),gp_Dir(1.,0.,0.)),math.pi)
    trsf.SetTranslationPart(gp_Vec(13.,-7.,23.))
    transformed=BRepBuilderAPI_Transform(authored_l_bend(),trsf,True).Shape()
    path=tmp_path/'authored-l-transformed.step'; write_step(transformed,path)
    mid=2.5/math.sqrt(2)
    expected=np.array([[13.+mid,-7.-mid,23.],[13.+mid,-7.-mid,-17.]])
    result=extract_step_bend_axes(path); assert_one_bend(result,expected)
    faces=read_folded_native(path)
    assert len(faces)==10  # eight extrusion sides and two end caps
    assert all(len(face.triangles)>0 and np.isfinite(face.vertices).all() for face in faces)


def test_authored_cylindrical_tube_is_not_forming_bend(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from scripts.extract_step_topology_bend_axes_v3 import extract_step_bend_axes
    outer=BRepPrimAPI_MakeCylinder(3.,40.).Shape()
    inner=BRepPrimAPI_MakeCylinder(2.,40.).Shape()
    cut=BRepAlgoAPI_Cut(outer,inner); cut.Build(); assert cut.IsDone()
    path=tmp_path/'authored-tube.step'; write_step(cut.Shape(),path)
    result=extract_step_bend_axes(path)
    assert result['bend_axes']==[]
    assert result['rejected_groups']
