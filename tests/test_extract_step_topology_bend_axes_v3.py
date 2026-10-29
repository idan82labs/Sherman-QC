"""Pure authored row fixtures; no native files or geometry jobs."""
import copy
import itertools
import math
import numpy as np
import pytest
from scripts import extract_step_topology_bend_axes_v3 as candidate


def row(index,radius,start=0.,sweep=math.pi/2,solid=1,edges=()):
    return {'face_index':index,'solid_id':solid,'radius_mm':radius,
        'line_start':[0.,0.,0.],'line_end':[0.,0.,40.],
        'axial_span_mm':40.,'angular_sweep_deg':math.degrees(sweep),
        'angular_intervals':candidate._intervals(start,sweep),'axial_interval':[0.,40.],
        'edge_ids':list(edges),'domain_certified':True,'tangent_rails':[
            {'angle_rad':start%candidate.TAU,'axial_interval':[0.,40.]},
            {'angle_rad':(start+sweep)%candidate.TAU,'axial_interval':[0.,40.]}]}


def test_pair_and_input_preservation():
    rows=[row(2,2.),row(5,3.)]; old=copy.deepcopy(rows)
    result=candidate.build_features(rows); bend=result['bend_axes'][0]
    assert len(result['bend_axes'])==1 and rows==old
    assert bend['face_indices']==[2,5]
    assert bend['canonical_line_rule']=='GEOMETRIC_MID_THICKNESS_HALF_ANGULAR_TURN'
    assert bend['operation_group_id'] is None


def test_opposite_sectors_separate_and_order_invariant():
    rows=[row(1,2.),row(2,3.),row(3,2.,math.pi),row(4,3.,math.pi)]
    expected=candidate.build_features(rows)
    assert len(expected['bend_axes'])==2
    for perm in itertools.permutations(rows):
        assert candidate.build_features(list(perm))==expected


def test_solid_ownership_disallows_cross_solid_pair():
    assert candidate.build_features([row(1,2.),row(2,3.,solid=2)])['bend_axes']==[]


def test_full_hole_fragments_rejected_before_pairing():
    rows=[]
    for skin,radius in enumerate((2.,3.)):
        for i in range(4):
            rows.append(row(skin*4+i+1,radius,i*math.pi/2,edges=(skin*10+i,skin*10+(i+1)%4)))
    result=candidate.build_features(rows)
    assert result['bend_axes']==[]
    assert all(r['rejection_reason']=='full_cylinder_domain' for r in result['rejected_groups'])


def test_splits_join_only_native_connected_same_radius():
    rows=[row(1,2.,0.,math.pi/4,edges=(9,)),row(2,2.,math.pi/4,math.pi/4,edges=(9,)),row(3,3.)]
    result=candidate.build_features(rows)
    assert len(result['bend_axes'])==1
    assert result['bend_axes'][0]['face_indices']==[1,2,3]
    rows[1]['edge_ids']=[]
    assert candidate.build_features(rows)['bend_axes']==[]


def test_holes_or_uncertified_domains_are_not_box_guesses():
    rows=[row(1,2.),row(2,3.)]; rows[0]['domain_certified']=False
    result=candidate.build_features(rows)
    assert not result['bend_axes']
    assert result['rejected_groups'][0]['rejection_reason']=='uncertified_trim_domain_or_solid_owner'


def test_pair_without_full_tangent_flange_rails_is_ambiguous():
    rows=[row(1,2.),row(2,3.)]; rows[0]['tangent_rails']=[]
    result=candidate.build_features(rows)
    assert not result['bend_axes']
    assert result['rejected_groups'][0]['rejection_reason']=='ambiguous_pair_missing_two_full_tangent_flange_rails'


def test_three_radius_ambiguity_refuses_arbitrary_pair_selection():
    result=candidate.build_features([row(1,2.),row(2,3.),row(3,4.)])
    assert not result['bend_axes']
    assert all(r['rejection_reason']=='ambiguous_radius_pair' for r in result['rejected_groups'])


def test_periodic_seam_marker_is_sector_midpoint():
    result=candidate.build_features([row(1,2.,7*math.pi/4),row(2,3.,7*math.pi/4)])
    assert len(result['bend_axes'])==1
    bend=result['bend_axes'][0]
    assert abs(bend['canonical_line_start'][0])<1e-12
    assert abs(bend['canonical_line_start'][1]-2.5)<1e-12


def test_negative_native_axis_signed_parameters_not_canonicalized_before_evaluation():
    direction=np.array([-1.,0.,0.]); original=direction.copy()
    a,ends=candidate._native_axis_geometry([10.,20.,30.],direction,2.,5.)
    assert np.array_equal(direction,original)
    assert a==pytest.approx([1.,0.,0.])
    assert np.array(ends)==pytest.approx(np.array([[5.,20.,30.],[8.,20.,30.]]))


def test_loose_v2_coaxial_tolerances_do_not_establish_shared_world_domain():
    rows=[row(1,2.),row(2,3.)]
    rows[1]['line_start'][0]=0.01; rows[1]['line_end'][0]=0.01
    assert not candidate.build_features(rows)['bend_axes']
    rows=[row(1,2.),row(2,3.)]; rows[1]['line_end'][0]=0.001
    assert not candidate.build_features(rows)['bend_axes']


@pytest.mark.parametrize('field,value',[('radius_mm',math.nan),('radius_mm',-1.),
    ('axial_interval',[0.,math.inf]),('angular_intervals',[[0.,math.nan]]),
    ('angular_intervals',[[-1.,1.]]),('line_end',[0.,0.,0.])])
def test_invalid_rows_preserved_as_rejections(field,value):
    malformed=row(1,2.); malformed[field]=value
    result=candidate.build_features([malformed,row(2,3.)])
    assert not result['bend_axes']
    assert result['rejected_groups'][0]['rejection_reason']=='invalid_nonfinite_or_malformed_row'


def test_duplicate_native_face_identity_is_refused():
    result=candidate.build_features([row(1,2.),row(1,3.)])
    assert not result['bend_axes']
    assert all(r['rejection_reason']=='duplicate_native_face_identity' for r in result['rejected_groups'])


def test_carrier_residuals_and_raw_native_identity_receipted():
    bend=candidate.build_features([row(2,2.),row(5,3.)])['bend_axes'][0]
    assert bend['carrier_receipt']['raw_native_face_indices']==[2,5]
    assert bend['carrier_receipt']['maximum_axis_offset_mm']==0.
