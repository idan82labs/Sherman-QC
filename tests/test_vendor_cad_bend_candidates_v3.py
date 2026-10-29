"""Pure authored adapter contracts; no CAD readers/native imports."""
import copy
import math
import pytest
from backend.vendor_cad_bend_candidates_v3 import axis_rows_v3,extraction_to_candidates,MARKER_RULE
from backend.vendor_unfold_contract_v2 import PreviewContractError


def axis():
    mid=2.5/math.sqrt(2)
    return {'feature_id':'S1_F2_5','bend_id':'STEP_BEND_S1_F2_5','operation_group_id':None,
        'face_indices':[2,5],'surface_count':2,'radii_mm':[2.,3.],'angular_sweeps_deg':[90.],
        'axial_span_mm':40.,'line_start':[0.,0.,0.],'line_end':[0.,0.,40.],
        'canonical_line_start':[mid,mid,0.],'canonical_line_end':[mid,mid,40.],
        'canonical_line_rule':MARKER_RULE,'domain_receipt':{
            'kind':'CERTIFIED_RECTANGULAR_NATIVE_PCURVE_DOMAIN','solid_id':1,
            'angular_intervals_rad':[[0.,math.pi/2]],'axial_interval_mm':[0.,40.],
            'tangent_flange_rails_proven':True,'material_thickness_proven':False},
        'carrier_receipt':{'raw_native_face_indices':[2,5],
            'maximum_axis_offset_mm':0.,'maximum_axis_angle_deg':0.}}


def test_preserves_single_sweep_pair_native_identity_and_mid_thickness_marker():
    a=axis(); old=copy.deepcopy(a); rows=axis_rows_v3([a]); r=rows[0]
    assert a==old and r['feature_id']==a['feature_id']
    assert r['line_start']==a['canonical_line_start'] and r['target_radius']==2.
    assert r['target_angle']==90.
    assert r['v3_provenance']['pair_radii_mm']==[2.,3.]
    assert r['v3_provenance']['angular_sweeps_deg']==[90.]
    assert r['v3_provenance']['native_face_indices']==[2,5]
    assert r['v3_provenance']['scan_support'] is None
    rows[0]['v3_provenance']['native_face_indices'].append(9)
    assert a==old


def test_explicit_mapping_requires_same_v3_native_features():
    bundle=extraction_to_candidates({'schema_version':3,'kind':'sherman_qc_step_topology_bend_axes','bend_axes':[axis()]})
    assert bundle['mapping_receipt']['native_reader_must_use_same_v3_features']
    assert not bundle['mapping_receipt']['v2_native_feature_match_is_compatible']


@pytest.mark.parametrize('field,value',[('angular_sweeps_deg',[90.,90.]),
    ('radii_mm',[3.,2.]),('angular_sweeps_deg',[180.]),
    ('canonical_line_rule','NEUTRAL_RADIUS_HALF_ANGULAR_TURN'),
    ('face_indices',[2,2]),('canonical_line_end',[1.,1.,30.]),('radii_mm',[math.nan,3.])])
def test_refuses_schema_and_geometry_forgery(field,value):
    a=axis(); a[field]=value
    with pytest.raises(PreviewContractError): axis_rows_v3([a])


def test_split_native_faces_preserved_without_duplicating_sweep():
    a=axis(); a['face_indices']=[2,3,5]; a['surface_count']=3
    a['carrier_receipt']['raw_native_face_indices']=[2,3,5]
    assert axis_rows_v3([a])[0]['v3_provenance']['native_face_indices']==[2,3,5]


def test_duplicates_and_uncertified_rails_refused():
    with pytest.raises(PreviewContractError): axis_rows_v3([axis(),axis()])
    a=axis(); a['domain_receipt']['tangent_flange_rails_proven']=False
    with pytest.raises(PreviewContractError): axis_rows_v3([a])


def test_refuses_v2_payload_without_implicit_renumbering():
    with pytest.raises(PreviewContractError): extraction_to_candidates({'schema_version':2,'bend_axes':[axis()]})
