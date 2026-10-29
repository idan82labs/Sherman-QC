"""Independent world-radial-vector fixtures; no native geometry or CAD access."""
import copy
import math
import numpy as np
import pytest
from scripts import extract_step_topology_bend_axes_v4 as candidate


def rows_for_pose(axis,origin,rotate_outer=math.pi/2,reverse_outer=False):
    a=np.asarray(axis,dtype=float); a/=np.linalg.norm(a); origin=np.asarray(origin,dtype=float)
    x,y=candidate._basis(a)
    rows=[]
    for index,radius in enumerate((2.,3.),1):
        angle=0. if index==1 else rotate_outer
        xx=math.cos(angle)*x+math.sin(angle)*y
        yy=-math.sin(angle)*x+math.cos(angle)*y
        aa=a.copy()
        if index==2 and reverse_outer: aa=-aa; yy=-yy
        start=math.atan2(np.dot(x,yy),np.dot(x,xx))
        end=math.atan2(np.dot(y,yy),np.dot(y,xx))
        if np.dot(aa,a)>0:
            intervals=candidate._intervals(start,math.pi/2)
        else:
            intervals=candidate._intervals(end,math.pi/2)
        axial=sorted([float(np.dot(origin,aa)),float(np.dot(origin+40*a,aa))])
        rows.append({'face_index':index,'solid_id':1,'radius_mm':radius,
            'line_start':origin.tolist(),'line_end':(origin+40*a).tolist(),
            'axial_interval':axial,'edge_ids':[],'domain_certified':True,
            'angular_intervals':intervals,'angular_sweep_deg':90.,'axial_span_mm':40.,
            'angular_frame':{'axis':aa.tolist(),'x_axis':xx.tolist(),'y_axis':yy.tolist()},
            'tangent_rails':[{'angle_rad':start%(2*math.pi),'axial_interval':axial},
                {'angle_rad':end%(2*math.pi),'axial_interval':axial}]})
    radial=(x+y)/math.sqrt(2)
    return rows,np.array([origin+2.5*radial,origin+40*a+2.5*radial])


@pytest.mark.parametrize('axis',[[1e-17,0.,1.],[0.,1e-17,1.],[1.,1e-17,0.],
    [1e-17,1.,0.],[1.,1.,1.],[.3,-.4,.8660254037844386]])
@pytest.mark.parametrize('origin',[[0.,0.,0.],[111.546,-43.1,27.],[100000.,-100000.,999.]])
@pytest.mark.parametrize('reverse_outer',[False,True])
def test_explicit_frames_survive_near_axis_seed_ties_translation_and_reversal(axis,origin,reverse_outer):
    rows,expected=rows_for_pose(axis,origin,reverse_outer=reverse_outer)
    old=copy.deepcopy(rows)
    result=candidate.build_features(rows)
    assert len(result['bend_axes'])==1,result['rejected_groups']
    bend=result['bend_axes'][0]
    actual=np.array([bend['canonical_line_start'],bend['canonical_line_end']])
    assert min(np.max(np.abs(actual-expected)),np.max(np.abs(actual-expected[::-1])))<1e-7
    assert rows==old
    assert bend['carrier_receipt']['source_angular_frames'][1]['source_frame']==rows[1]['angular_frame']


def test_unrecorded_frame_is_not_reinterpreted_using_endpoint_basis():
    rows,_=rows_for_pose([1e-17,0.,1.],[111.,-43.,2.])
    del rows[0]['angular_frame']
    result=candidate.build_features(rows)
    assert not result['bend_axes']
    assert result['partial_cylindrical_faces'][0]['frame_diagnostic']=='EXPLICIT_VALID_ANGULAR_FRAME_REQUIRED'


def test_opposite_sector_is_not_created_or_merged_by_basis_transport():
    rows,_=rows_for_pose([0.,0.,1.],[0.,0.,0.])
    for rail in rows[1]['tangent_rails']: rail['angle_rad']=(rail['angle_rad']+math.pi)%(2*math.pi)
    rows[1]['angular_intervals']=candidate._intervals(math.pi/2,math.pi/2)
    assert not candidate.build_features(rows)['bend_axes']


def test_frame_axis_incompatible_with_carrier_is_refused():
    rows,_=rows_for_pose([0.,0.,1.],[0.,0.,0.])
    rows[0]['angular_frame']={'axis':[1.,0.,0.],'x_axis':[0.,1.,0.],'y_axis':[0.,0.,1.]}
    assert not candidate.build_features(rows)['bend_axes']


def test_near_zero_wrapped_intersection_recovers_true_arc_not_seam_rails():
    start=5.731117953590423; sweep=1.4110197787874115; end=.8589524251982485
    intervals=[[1.1102230246251563e-16,end],[start,2*math.pi]]
    arc=candidate._connected_circular_arc(intervals)
    assert arc==pytest.approx((start,sweep),abs=1e-14)


def test_disjoint_sectors_and_full_circle_are_not_one_connected_partial_arc():
    assert candidate._connected_circular_arc([[0.,.5],[1.,1.5]]) is None
    assert candidate._connected_circular_arc([[0.,2*math.pi]]) is None
    assert candidate._connected_circular_arc([[0.,.5],[.5+1e-6,1.]]) is None


def test_independent_arbitrary_sector_so3_frames_wrapping_and_axis_reversal():
    rng=np.random.default_rng(731)
    for k in range(50):
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(q)<0: q[:,0]*=-1
        x,y,a=q[:,0],q[:,1],q[:,2]
        origin=rng.uniform(-1e5,1e5,3)
        theta=rng.uniform(0,2*math.pi); sweep=rng.uniform(.3,2.8)
        phi=rng.uniform(-math.pi,math.pi)
        first=math.cos(theta)*x+math.sin(theta)*y
        last=math.cos(theta+sweep)*x+math.sin(theta+sweep)*y
        rows=[]
        for index,radius in enumerate((2.,3.),1):
            aa=a.copy(); xx=x.copy(); yy=y.copy()
            if index==2:
                xx=math.cos(phi)*x+math.sin(phi)*y
                yy=-math.sin(phi)*x+math.cos(phi)*y
                if k%2: aa=-aa; yy=-yy
            lo=math.atan2(np.dot(first,yy),np.dot(first,xx))
            hi=math.atan2(np.dot(last,yy),np.dot(last,xx))
            axial=sorted([float(np.dot(origin,aa)),float(np.dot(origin+40*a,aa))])
            rows.append({'face_index':index,'solid_id':1,'radius_mm':radius,
                'line_start':origin.tolist(),'line_end':(origin+40*a).tolist(),
                'axial_interval':axial,'edge_ids':[],'domain_certified':True,
                'angular_intervals':candidate._intervals(lo if np.dot(aa,a)>0 else hi,sweep),
                'angular_sweep_deg':math.degrees(sweep),'axial_span_mm':40.,
                'angular_frame':{'axis':aa.tolist(),'x_axis':xx.tolist(),'y_axis':yy.tolist()},
                'tangent_rails':[{'angle_rad':lo%(2*math.pi),'axial_interval':axial},
                    {'angle_rad':hi%(2*math.pi),'axial_interval':axial}]})
        result=candidate.build_features(rows)
        assert len(result['bend_axes'])==1,(k,result['rejected_groups'])
        radial=math.cos(theta+sweep/2)*x+math.sin(theta+sweep/2)*y
        expected=np.array([origin+2.5*radial,origin+40*a+2.5*radial])
        bend=result['bend_axes'][0]
        actual=np.array([bend['canonical_line_start'],bend['canonical_line_end']])
        assert min(np.max(np.abs(actual-expected)),np.max(np.abs(actual-expected[::-1])))<1e-7
