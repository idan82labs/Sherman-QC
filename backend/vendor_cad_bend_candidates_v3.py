"""Pure V3 geometric-feature adapter; no CAD reader or scoring imports.

Metadata is nominal CAD geometry, never observed scan support. V5's independent
V2 native feature extraction is not automatically compatible with these IDs;
the next native reader must explicitly consume/check this mapping receipt.
"""
from __future__ import annotations
import copy
import math
from backend.vendor_unfold_contract_v2 import canonical_candidates,PreviewContractError,MAX_COORDINATE

MARKER_RULE='GEOMETRIC_MID_THICKNESS_HALF_ANGULAR_TURN'


def _require(condition,reason):
    if not condition: raise PreviewContractError(reason)


def _numbers(values):
    return all(type(v) in (int,float) and math.isfinite(v) and abs(v)<=MAX_COORDINATE for v in values)


def axis_rows_v3(axes,tolerance_angle=5.0):
    _require(type(tolerance_angle) in (int,float) and math.isfinite(tolerance_angle)
        and tolerance_angle>0,'INVALID_ANGLE_TOLERANCE')
    _require(type(axes) is list,'V3_INVALID_AXES')
    rows=[]; receipts={}
    for axis in axes:
        _require(type(axis) is dict,'V3_INVALID_AXIS')
        radii=axis.get('radii_mm'); sweep=axis.get('angular_sweeps_deg'); ids=axis.get('face_indices')
        _require(type(radii) is list and len(radii)==2 and type(sweep) is list and len(sweep)==1
            and _numbers(radii+sweep),'V3_INVALID_PAIR_METADATA')
        _require(0<radii[0]<radii[1]<=MAX_COORDINATE and 15<=sweep[0]<=175,
            'V3_DOWNSTREAM_RADIUS_OR_SWEEP_UNSUPPORTED')
        _require(type(ids) is list and len(ids)>=2 and all(type(i) is int and i>0 for i in ids)
            and ids==sorted(set(ids)) and type(axis.get('surface_count')) is int
            and axis['surface_count']==len(ids),'V3_INVALID_NATIVE_FACE_PROVENANCE')
        _require(axis.get('canonical_line_rule')==MARKER_RULE,'V3_MARKER_SEMANTICS_MISMATCH')
        domain=axis.get('domain_receipt'); carrier=axis.get('carrier_receipt')
        _require(type(domain) is dict and domain.get('kind')=='CERTIFIED_RECTANGULAR_NATIVE_PCURVE_DOMAIN'
            and domain.get('tangent_flange_rails_proven') is True
            and type(domain.get('solid_id')) is int and domain['solid_id']>0,
            'V3_MISSING_FINITE_DOMAIN_RECEIPT')
        _require(type(carrier) is dict and carrier.get('raw_native_face_indices')==ids,
            'V3_CARRIER_PROVENANCE_MISMATCH')
        _require(_numbers([carrier.get('maximum_axis_offset_mm'),carrier.get('maximum_axis_angle_deg')])
            and 0<=carrier['maximum_axis_offset_mm']<=1e-7
            and 0<=carrier['maximum_axis_angle_deg']<=1e-5,'V3_CARRIER_BOUNDS_UNSUPPORTED')
        intervals=domain.get('angular_intervals_rad'); axial=domain.get('axial_interval_mm')
        _require(type(intervals) is list and 1<=len(intervals)<=2
            and all(type(i) is list and len(i)==2 and _numbers(i)
                and 0<=i[0]<i[1]<=2*math.pi+1e-8 for i in intervals)
            and type(axial) is list and len(axial)==2 and _numbers(axial) and axial[1]>axial[0],
            'V3_INVALID_FINITE_DOMAIN')
        _require(abs(math.degrees(sum(i[1]-i[0] for i in intervals))-sweep[0])<=1e-6
            and type(axis.get('axial_span_mm')) in (int,float)
            and math.isfinite(axis['axial_span_mm'])
            and abs(axial[1]-axial[0]-axis['axial_span_mm'])<=1e-7,
            'V3_DOMAIN_METADATA_MISMATCH')
        feature=axis.get('feature_id')
        _require(type(feature) is str and feature not in receipts,'V3_DUPLICATE_OR_INVALID_FEATURE_ID')
        lines=canonical_candidates([{'feature_id':feature,'line_start':axis.get('line_start'),
            'line_end':axis.get('line_end')}])
        rows.append({'feature_id':feature,'line_start':axis.get('canonical_line_start'),
            'line_end':axis.get('canonical_line_end'),'target_angle':float(sweep[0]),
            'target_radius':float(radii[0])})
        marker=canonical_candidates([rows[-1]])[0]
        _require(abs(math.dist(marker['line_start'],marker['line_end'])-axis['axial_span_mm'])<=1e-7
            and abs(math.dist(lines[0]['line_start'],lines[0]['line_end'])-axis['axial_span_mm'])<=1e-7,
            'V3_FINITE_MARKER_SPAN_MISMATCH')
        receipts[feature]={'extractor_schema_version':3,'native_feature_id':feature,
            'native_bend_id':axis.get('bend_id'),'operation_group_id':axis.get('operation_group_id'),
            'native_face_indices':copy.deepcopy(ids),'pair_radii_mm':copy.deepcopy(radii),
            'angular_sweeps_deg':copy.deepcopy(sweep),'curvature_axis':lines[0],
            'marker_rule':MARKER_RULE,'domain_receipt':copy.deepcopy(domain),
            'carrier_receipt':copy.deepcopy(carrier),'units':'mm_assumed_not_verified',
            'nominal_reference_only':True,'scan_support':None,'physical_accuracy':None,
            'tolerance_angle':float(tolerance_angle)}
    canonical=canonical_candidates(rows)
    lookup={r['feature_id']:r for r in rows}
    return [{**r,'target_angle':lookup[r['feature_id']]['target_angle'],
        'target_radius':lookup[r['feature_id']]['target_radius'],
        'v3_provenance':receipts[r['feature_id']]} for r in canonical]


def extraction_to_candidates(payload,tolerance_angle=5.0):
    _require(type(payload) is dict and payload.get('schema_version')==3
        and payload.get('kind')=='sherman_qc_step_topology_bend_axes','V3_EXTRACTION_REQUIRED')
    candidates=axis_rows_v3(payload.get('bend_axes'),tolerance_angle)
    return {'candidates':candidates,'mapping_receipt':{'adapter_version':3,
        'native_extractor_schema_version':3,'candidate_ids':[r['feature_id'] for r in candidates],
        'marker_rule':MARKER_RULE,'native_reader_must_use_same_v3_features':True,
        'v2_native_feature_match_is_compatible':False,'manufacturing_ready':False}}
