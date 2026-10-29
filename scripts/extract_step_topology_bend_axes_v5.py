#!/usr/bin/env python3
"""V5 finite native cylinders with coefficient-certified split rectangle domains.

Native acceptance currently requires a rectangular trim proven by four linear
axis-aligned pcurves. Other domains remain diagnostics, never guessed material.
"""
from __future__ import annotations

import math
import copy
from pathlib import Path
from typing import Any
import numpy as np
from scripts import extract_step_topology_bend_axes_v2 as v2

TAU = 2.0 * math.pi
DOMAIN_EPS = 1e-8
CARRIER_POSITION_TOL_MM = 1e-7
CARRIER_DIRECTION_TOL = 1e-9
MAX_NATIVE_FACES = 10000
MAX_NATIVE_EDGES = 50000


def _basis(a):
    seed = np.eye(3)[int(np.argmin(np.abs(a)))]
    e1 = np.cross(a, seed); e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


def _intervals(start, sweep):
    if sweep >= TAU-DOMAIN_EPS:
        return [(0., TAU)]
    start %= TAU
    # Floating remainder can round a tiny negative angle to exactly TAU.
    if start >= TAU:
        start = 0.
    end = start+sweep
    pieces = [(start, end)] if end <= TAU else [(0., end-TAU), (start, TAU)]
    return [(lo,hi) for lo,hi in pieces if hi>lo]


def _union(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]+DOMAIN_EPS:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def _measure(intervals):
    return sum(b-a for a, b in _union(intervals))


def _intersection(left, right):
    return _union([(max(a,c), min(b,d)) for a,b in left for c,d in right
                   if min(b,d)-max(a,c) > DOMAIN_EPS])


def _angular_touch(left, right):
    return any(max(a,c)<=min(b,d)+DOMAIN_EPS for a,b in left for c,d in right) or (
        any(abs(a)<DOMAIN_EPS for a,b in left) and any(abs(b-TAU)<DOMAIN_EPS for a,b in right)) or (
        any(abs(a)<DOMAIN_EPS for a,b in right) and any(abs(b-TAU)<DOMAIN_EPS for a,b in left))


def _connected_circular_arc(intervals):
    """Recover one connected circular arc without assuming a zero seam endpoint.

    Remove the largest complementary gap. Every other complementary gap must
    sum to at most DOMAIN_EPS: disjoint angular sectors cannot become one bend.
    """
    union=_union(intervals)
    if not union:
        return None
    gaps=[]
    for index,(_,end) in enumerate(union):
        following=union[(index+1)%len(union)][0]
        if index==len(union)-1:
            following+=TAU
        gaps.append((max(0.,following-end),index))
    gap,index=max(gaps,key=lambda row:(row[0],-row[1]))
    span=TAU-gap
    if abs(span-_measure(union))>DOMAIN_EPS or span<=DOMAIN_EPS or span>=TAU-DOMAIN_EPS:
        return None
    start=union[(index+1)%len(union)][0]%TAU
    return start,span


def _same_carrier(a, b, same_radius=False):
    da=v2._normalized_direction(np.asarray(a['line_end'])-np.asarray(a['line_start']))
    db=v2._normalized_direction(np.asarray(b['line_end'])-np.asarray(b['line_start']))
    delta=np.asarray(b['line_start'])-np.asarray(a['line_start'])
    offset=float(np.linalg.norm(delta-da*np.dot(delta,da)))
    return (a['solid_id'] == b['solid_id'] and np.linalg.norm(np.cross(da,db))<=CARRIER_DIRECTION_TOL
            and offset<=CARRIER_POSITION_TOL_MM
            and (not same_radius or abs(a['radius_mm']-b['radius_mm']) <= 1e-6))


def _valid_row(row):
    try:
        vectors=np.asarray([row['line_start'],row['line_end']],dtype=float)
        axial=np.asarray(row['axial_interval'],dtype=float)
        angles=np.asarray(row['angular_intervals'],dtype=float)
        return bool(vectors.shape==(2,3) and np.isfinite(vectors).all()
            and np.linalg.norm(vectors[1]-vectors[0])>DOMAIN_EPS
            and axial.shape==(2,) and np.isfinite(axial).all() and axial[1]>axial[0]
            and angles.ndim==2 and angles.shape[1]==2 and len(angles)>0
            and np.isfinite(angles).all() and np.all(angles[:,0]>=0)
            and np.all(angles[:,1]<=TAU+DOMAIN_EPS) and np.all(angles[:,1]>angles[:,0])
            and math.isfinite(float(row['radius_mm'])) and float(row['radius_mm'])>0
            and isinstance(row['face_index'],int) and row['face_index']>0)
    except (KeyError,ValueError,TypeError):
        return False


def _native_axis_geometry(location,native_direction,v0,v1):
    native=np.asarray(native_direction,dtype=float).copy()
    canonical=v2._normalized_direction(native.copy())
    return canonical,v2._cylinder_axis_endpoints(np.asarray(location,dtype=float),native,v0,v1)



def _frame(row):
    frame=row.get('angular_frame')
    if type(frame) is not dict:
        raise ValueError('EXPLICIT_ANGULAR_FRAME_REQUIRED')
    a=np.asarray(frame.get('axis'),dtype=float)
    x=np.asarray(frame.get('x_axis'),dtype=float)
    y=np.asarray(frame.get('y_axis'),dtype=float)
    matrix=np.asarray([a,x,y])
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or not np.allclose(
            matrix@matrix.T,np.eye(3),rtol=0,atol=1e-9) or np.linalg.norm(np.cross(x,y)-a)>1e-9:
        raise ValueError('INVALID_ANGULAR_FRAME')
    endpoint=v2._normalized_direction(np.asarray(row['line_end'])-np.asarray(row['line_start']))
    if np.linalg.norm(np.cross(a,endpoint))>CARRIER_DIRECTION_TOL:
        raise ValueError('ANGULAR_FRAME_AXIS_NOT_CARRIER')
    return a,x,y


def _transport_carrier_frames(rows):
    """Convert recorded physical radial vectors; never reinterpret naked angles."""
    result=[]; reference=[]
    for original in rows:
        row=copy.deepcopy(original)
        try:
            a,x,y=_frame(row)
        except (ValueError,TypeError):
            row['domain_certified']=False
            row['frame_diagnostic']='EXPLICIT_VALID_ANGULAR_FRAME_REQUIRED'
            result.append(row)
            continue
        match=next((r for r in reference if _same_carrier(row,r)),None)
        if match is None:
            # Preserve the authored source basis: basis selection is not repeated.
            match=row
            reference.append(row)
        ca,cx,cy=_frame(match)
        sign=1 if np.dot(a,ca)>0 else -1
        def angle(theta):
            radial=math.cos(theta)*x+math.sin(theta)*y
            return math.atan2(float(np.dot(radial,cy)),float(np.dot(radial,cx)))%TAU
        intervals=[]
        for lo,hi in row['angular_intervals']:
            intervals.extend(_intervals(angle(lo if sign>0 else hi),hi-lo))
        row['angular_frame_provenance']={'source_frame':copy.deepcopy(original['angular_frame']),
            'source_angular_intervals_rad':copy.deepcopy(original['angular_intervals']),
            'reference_face_index':match['face_index'],
            'axis_direction_cross_residual':float(np.linalg.norm(np.cross(a,ca))),
            'angular_transport_rule':'WORLD_RADIAL_VECTOR_TO_COMMON_CARRIER_FRAME'}
        row['angular_intervals']=_union(intervals)
        for rail in row.get('tangent_rails',[]):
            rail['angle_rad']=angle(rail['angle_rad'])
        # Signed axial domains also change sign when the source frame is reversed.
        if sign<0:
            row['axial_interval']=sorted(-float(z) for z in row['axial_interval'])
            for rail in row.get('tangent_rails',[]):
                rail['axial_interval']=sorted(-float(z) for z in rail['axial_interval'])
        row['angular_frame']={'axis':ca.tolist(),'x_axis':cx.tolist(),'y_axis':cy.tolist()}
        result.append(row)
    return result


def build_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure-row deterministic grouping; certified angular intervals are radians.

    Axial domains use world projection on the canonical axis. Each input row is
    one rectangular material patch. No caller input is mutated.
    """
    invalid=[r for r in rows if not _valid_row(r)]
    valid=[r for r in rows if _valid_row(r)]
    duplicates={r['face_index'] for r in valid if sum(s['face_index']==r['face_index'] for s in valid)>1}
    ordered = sorted(valid, key=lambda r: (str(r.get('solid_id')), r['face_index']))
    ordered = _transport_carrier_frames(ordered)
    rejected = []; good = []
    for row in invalid:
        rejected.append({'face_indices':[row.get('face_index')],'rejection_reason':'invalid_nonfinite_or_malformed_row'})
    for row in ordered:
        if row['face_index'] in duplicates:
            rejected.append({'face_indices':[row['face_index']],'rejection_reason':'duplicate_native_face_identity'})
        elif not row.get('domain_certified') or row.get('solid_id') is None:
            rejected.append({'face_indices':[row['face_index']], 'rejection_reason':
                             'uncertified_trim_domain_or_solid_owner'})
        else:
            good.append(row)
    components = []
    for row in good:
        eligible = []
        for component in components:
            # Complete-link carrier agreement prevents tolerance-chain drift.
            if not all(_same_carrier(row,r,True) for r in component):
                continue
            if any(row['angular_frame_provenance']['reference_face_index'] !=
                   r['angular_frame_provenance']['reference_face_index'] for r in component):
                continue
            connected = any(set(row['edge_ids']) & set(r['edge_ids']) and
                _angular_touch(row['angular_intervals'],r['angular_intervals']) and
                min(row['axial_interval'][1],r['axial_interval'][1]) >=
                max(row['axial_interval'][0],r['axial_interval'][0])-DOMAIN_EPS
                for r in component)
            if connected:
                eligible.append(component)
        if len(eligible) > 1:
            rejected.append({'face_indices':[row['face_index']],
                             'rejection_reason':'ambiguous_split_component_bridge'})
        elif eligible:
            eligible[0].append(row)
        else:
            components.append([row])
    carriers = []
    for component in components:
        angles = _union([i for r in component for i in r['angular_intervals']])
        ids = sorted(r['face_index'] for r in component)
        if _measure(angles) >= TAU-DOMAIN_EPS:
            rejected.append({'face_indices':ids,'rejection_reason':'full_cylinder_domain'})
            continue
        # Require identical axial support for joined angular fragments. More
        # complex unions need a polygon-domain successor, not a bounding box.
        axial = component[0]['axial_interval']
        if any(not np.allclose(r['axial_interval'],axial,rtol=0,atol=DOMAIN_EPS)
               for r in component):
            rejected.append({'face_indices':ids,'rejection_reason':'nonrectangular_split_domain'})
            continue
        carriers.append({'rows':component,'angles':angles,'axial':axial,'ids':ids})
    pairs = []
    for i,left in enumerate(carriers):
        for j in range(i+1,len(carriers)):
            right = carriers[j]; a=left['rows'][0]; b=right['rows'][0]
            if a['angular_frame_provenance']['reference_face_index'] != b['angular_frame_provenance']['reference_face_index']:
                continue
            separation=abs(a['radius_mm']-b['radius_mm'])
            if not _same_carrier(a,b) or not v2.MIN_RADIUS_SEPARATION_MM <= separation <= v2.MAX_RADIUS_SEPARATION_MM:
                continue
            common=_intersection(left['angles'],right['angles'])
            if (abs(_measure(common)-_measure(left['angles'])) > DOMAIN_EPS or
                abs(_measure(common)-_measure(right['angles'])) > DOMAIN_EPS):
                continue
            if _connected_circular_arc(common) is None:
                continue
            lo=max(left['axial'][0],right['axial'][0]); hi=min(left['axial'][1],right['axial'][1])
            if not v2._eligible_native_cylinder(math.degrees(_measure(common)),hi-lo):
                continue
            pairs.append((i,j,common,lo,hi))
    degree={i:sum(i in p[:2] for p in pairs) for i in range(len(carriers))}
    axes=[]
    for i,j,common,lo,hi in pairs:
        if degree[i]!=1 or degree[j]!=1:
            continue
        l,r=carriers[i],carriers[j]; row=l['rows'][0]
        # A radius pair is not sufficient: both skins require native tangent
        # plate rails at both angular boundaries over the full paired span.
        arc_start,arc_span=_connected_circular_arc(common)
        boundaries=[arc_start,(arc_start+arc_span)%TAU]
        def has_rails(carrier):
            rails=[rail for patch in carrier['rows'] for rail in patch.get('tangent_rails',[])]
            return all(any(abs(math.atan2(math.sin(rail['angle_rad']-theta),
                math.cos(rail['angle_rad']-theta)))<=DOMAIN_EPS and
                rail['axial_interval'][0]<=lo+DOMAIN_EPS and
                rail['axial_interval'][1]>=hi-DOMAIN_EPS for rail in rails) for theta in boundaries)
        if not has_rails(l) or not has_rails(r):
            rejected.append({'face_indices':sorted(l['ids']+r['ids']),
                'rejection_reason':'ambiguous_pair_missing_two_full_tangent_flange_rails'})
            continue
        a=np.asarray(row['angular_frame']['axis'],dtype=float)
        origin=np.asarray(row['line_start']); c=origin-a*np.dot(origin,a)
        e1=np.asarray(row['angular_frame']['x_axis'],dtype=float)
        e2=np.asarray(row['angular_frame']['y_axis'],dtype=float)
        # Circular midpoint handles domains crossing the periodic seam.
        theta=arc_start+arc_span/2
        radius=(row['radius_mm']+r['rows'][0]['radius_mm'])/2
        center=c+radius*(math.cos(theta)*e1+math.sin(theta)*e2)
        ids=sorted(l['ids']+r['ids']); identity='S'+str(row['solid_id'])+'_F'+'_'.join(map(str,ids))
        axes.append({'bend_id':'STEP_BEND_'+identity,'group_id':'STEP_CYL_'+identity,
            'feature_id':identity,'operation_group_id':None,'face_indices':ids,
            'radii_mm':sorted([row['radius_mm'],r['rows'][0]['radius_mm']]),
            'radius_separation_mm':abs(row['radius_mm']-r['rows'][0]['radius_mm']),
            'surface_count':len(ids),'line_start':(c+a*lo).tolist(),'line_end':(c+a*hi).tolist(),
            'canonical_line_start':(center+a*lo).tolist(),'canonical_line_end':(center+a*hi).tolist(),
            'canonical_line_rule':'GEOMETRIC_MID_THICKNESS_HALF_ANGULAR_TURN',
            'axial_span_mm':hi-lo,'angular_sweeps_deg':[round(math.degrees(_measure(common)),6)],
            'domain_receipt':{'kind':'CERTIFIED_RECTANGULAR_NATIVE_PCURVE_DOMAIN',
                'angular_intervals_rad':common,'axial_interval_mm':[lo,hi],
                'connected_arc_start_rad':arc_start,'connected_arc_span_rad':arc_span,
                'native_domain_certificates':[copy.deepcopy(patch.get('native_domain_certificate',{}))
                    for patch in l['rows']+r['rows']],
                'certificate_schema_version':5,
                'solid_id':row['solid_id'],'tangent_flange_rails_proven':True,'material_thickness_proven':False,
                'forming_operation_identity_proven':False}})
        allrows=l['rows']+r['rows']
        reference=allrows[0]
        metrics=[v2._line_metrics(reference,patch) for patch in allrows]
        axes[-1]['carrier_receipt']={'maximum_axis_offset_mm':max(m['axis_offset_mm'] for m in metrics),
            'maximum_axis_angle_deg':max(m['axis_angle_deg'] for m in metrics),
            'position_tolerance_mm':CARRIER_POSITION_TOL_MM,'direction_cross_tolerance':CARRIER_DIRECTION_TOL,
            'raw_native_face_indices':ids,
            'angular_frame':copy.deepcopy(row['angular_frame']),
            'source_angular_frames':[{'face_index':patch['face_index'],
                **copy.deepcopy(patch['angular_frame_provenance'])} for patch in allrows]}
    for i,carrier in enumerate(carriers):
        if degree[i]!=1:
            rejected.append({'face_indices':carrier['ids'],'rejection_reason':
                'ambiguous_radius_pair' if degree[i]>1 else 'missing_compatible_finite_pair'})
    return {'bend_axes':axes,'rejected_groups':rejected,'partial_cylindrical_faces':ordered}



MAX_DOMAIN_SEGMENTS=256
MAX_DOMAIN_TOLERANCE_MM=1e-5


def certify_rectangle_segments(segments,tolerance_mm):
    """Prove one simple rectangular boundary from actual metric pcurve segments."""
    receipt={'status':'REFUSED','kind':'NATIVE_PCURVE_RECTANGLE_SEGMENT_UNION',
        'tolerance_mm':tolerance_mm,'segment_count':len(segments),'maximum_residual_mm':None}
    def refuse(reason):
        return {**receipt,'reason':reason}
    if type(tolerance_mm) not in (int,float) or not math.isfinite(tolerance_mm) or not 0<tolerance_mm<=MAX_DOMAIN_TOLERANCE_MM:
        return refuse('INVALID_SOURCE_PHYSICAL_TOLERANCE')
    if not 4<=len(segments)<=MAX_DOMAIN_SEGMENTS:
        return refuse('BOUNDARY_SEGMENT_COUNT_UNSUPPORTED')
    try:
        points=np.asarray([[r['start'],r['end']] for r in segments],dtype=float)
        ids=[r['edge_id'] for r in segments]
    except (KeyError,ValueError,TypeError):
        return refuse('INVALID_BOUNDARY_SEGMENTS')
    if points.shape!=(len(segments),2,2) or not np.isfinite(points).all() or len(set(ids))!=len(ids):
        return refuse('INVALID_OR_REPEATED_NATIVE_BOUNDARY')
    tol=float(tolerance_mm); lengths=np.linalg.norm(points[:,1]-points[:,0],axis=1)
    if np.any(lengths<=tol):
        return refuse('DEGENERATE_BOUNDARY_SEGMENT')
    flat=points.reshape(-1,2)
    links={}
    for i,p in enumerate(flat):
        neighbors=[j for j,q in enumerate(flat) if j!=i and j//2!=i//2 and np.linalg.norm(p-q)<=tol]
        if len(neighbors)!=1:
            return refuse('BOUNDARY_JOIN_GAP_BRANCH_OR_AMBIGUITY')
        links[i]=neighbors[0]
    # Alternating native-segment and endpoint joins must traverse one loop.
    visited=set(); cursor=0
    for _ in range(len(segments)):
        if cursor//2 in visited:
            return refuse('MULTIPLE_OR_RETRACED_BOUNDARY_LOOPS')
        visited.add(cursor//2); cursor=links[cursor^1]
    if cursor!=0 or len(visited)!=len(segments):
        return refuse('MULTIPLE_OR_OPEN_BOUNDARY_LOOPS')
    low=np.min(flat,axis=0); high=np.max(flat,axis=0)
    if np.any(high-low<=2*tol):
        return refuse('DEGENERATE_RECTANGLE')
    sides={0:[],1:[],2:[],3:[]}
    residual=max(float(np.linalg.norm(flat[i]-flat[j])) for i,j in links.items())
    for p,q in points:
        matches=[]
        for dim in (0,1):
            other=1-dim
            for side,bound in enumerate((low[dim],high[dim])):
                deviation=max(abs(p[dim]-bound),abs(q[dim]-bound))
                if deviation<=tol and abs(q[other]-p[other])>tol:
                    matches.append((2*dim+side,other,deviation))
        if len(matches)!=1:
            return refuse('NONRECTANGULAR_OR_UNCERTAIN_SEGMENT')
        side,other,deviation=matches[0]; residual=max(residual,deviation)
        sides[side].append(sorted([float(p[other]),float(q[other])]))
    for side,intervals in sides.items():
        if not intervals: return refuse('MISSING_RECTANGLE_SIDE')
        other=1-side//2; ordered=sorted(intervals)
        if abs(ordered[0][0]-low[other])>tol or abs(ordered[-1][1]-high[other])>tol:
            return refuse('INCOMPLETE_SIDE_COVERAGE')
        for previous,next_interval in zip(ordered,ordered[1:]):
            difference=next_interval[0]-previous[1]
            if abs(difference)>tol:
                return refuse('SIDE_GAP_OR_OVERLAP')
            residual=max(residual,abs(difference))
    return {**receipt,'status':'PROVED','reason':None,'bounds_metric_mm':
        [float(low[0]),float(high[0]),float(low[1]),float(high[1])],
        'maximum_residual_mm':residual,'edge_ids':ids,
        'simple_connected_outer_loop':True,'complete_side_coverage':True,
        'holes_supported':False,'geometry_sampled':False}


def _domain_certificate(face,surface,face_index,edge_ids):
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve2d
    from OCP.GeomAbs import GeomAbs_Line,GeomAbs_BSplineCurve,GeomAbs_BezierCurve
    from OCP.TopAbs import TopAbs_EDGE,TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    tolerance=float(BRep_Tool.Tolerance_s(face))
    base={'face_index':face_index,'wire_count':0,'pcurve_types':[],
        'type_proven':False,'declared_tolerance_mm':tolerance,
        'metric':'CYLINDER_R_SQUARED_DU_SQUARED_PLUS_DV_SQUARED',
        'native_edge_ids':list(edge_ids),'status':'REFUSED'}
    wires=TopExp_Explorer(face,TopAbs_WIRE)
    while wires.More():
        base['wire_count']+=1; wires.Next()
        if base['wire_count']>1: return {**base,'reason':'MULTIPLE_WIRES_OR_HOLES_UNSUPPORTED'}
    if base['wire_count']!=1: return {**base,'reason':'SINGLE_NATIVE_OUTER_WIRE_REQUIRED'}
    segments=[]; explorer=TopExp_Explorer(face,TopAbs_EDGE)
    radius=float(surface.Cylinder().Radius())
    while explorer.More():
        if len(segments)>=MAX_DOMAIN_SEGMENTS: return {**base,'reason':'BOUNDARY_SEGMENT_BUDGET'}
        curve=BRepAdaptor_Curve2d(TopoDS.Edge_s(explorer.Current()),face)
        kind=curve.GetType(); base['pcurve_types'].append(str(kind))
        first=float(curve.FirstParameter()); last=float(curve.LastParameter())
        if not math.isfinite(first) or not math.isfinite(last) or last<=first:
            return {**base,'reason':'INVALID_PCURVE_INTERVAL'}
        if kind in (GeomAbs_BSplineCurve,GeomAbs_BezierCurve):
            spline=curve.BSpline() if kind==GeomAbs_BSplineCurve else curve.Bezier()
            count=spline.NbPoles()
            if not 2<=count<=MAX_DOMAIN_SEGMENTS: return {**base,'reason':'PCURVE_COEFFICIENT_BUDGET'}
            poles=np.asarray([[spline.Pole(i).X()*radius,spline.Pole(i).Y()] for i in range(1,count+1)])
            weights=np.asarray([spline.Weight(i) for i in range(1,count+1)])
            if not np.isfinite(poles).all() or not np.isfinite(weights).all() or np.any(weights<=0):
                return {**base,'reason':'INVALID_POSITIVE_RATIONAL_COEFFICIENTS'}
            constant=[dim for dim in (0,1) if np.all(poles[:,dim]==poles[0,dim])]
            if len(constant)!=1: return {**base,'reason':'PCURVE_NOT_COEFFICIENT_CERTIFIED_AXIS_LINE'}
            delta=np.diff(poles[:,1-constant[0]])
            if not (np.all(delta>=0) or np.all(delta<=0)) or not np.any(delta!=0):
                return {**base,'reason':'PCURVE_BACKTRACK_OR_DEGENERACY'}
            if kind==GeomAbs_BSplineCurve:
                degree=spline.Degree()
                if spline.IsPeriodic() or spline.Multiplicity(1)!=degree+1 or spline.Multiplicity(spline.NbKnots())!=degree+1 or first!=spline.Knot(1) or last!=spline.Knot(spline.NbKnots()):
                    return {**base,'reason':'PCURVE_UNPROVEN_TRIMMED_COEFFICIENT_DOMAIN'}
                if any(spline.Multiplicity(k)>degree for k in range(2,spline.NbKnots())):
                    return {**base,'reason':'PCURVE_DISCONTINUOUS_INTERIOR_KNOT'}
            elif first!=0. or last!=1.:
                return {**base,'reason':'PCURVE_UNPROVEN_TRIMMED_COEFFICIENT_DOMAIN'}
        elif kind!=GeomAbs_Line:
            return {**base,'reason':'PCURVE_REPRESENTATION_UNSUPPORTED'}
        p=curve.Value(first); q=curve.Value(last)
        # The actual trim endpoints define the segment, not a pole/bounds box.
        segments.append({'start':[p.X()*radius,p.Y()],'end':[q.X()*radius,q.Y()],
            'edge_id':edge_ids[len(segments)]})
        explorer.Next()
    proof=certify_rectangle_segments(segments,tolerance)
    base.update(proof); base['native_edge_ids']=list(edge_ids)
    base['type_proven']=proof['status']=='PROVED'
    if proof['status']=='PROVED':
        u0,u1,v0,v1=proof['bounds_metric_mm']
        # Existing axis/sweep computation uses adaptor bounds. Publish a feature
        # only when those bounds agree with the proved actual boundary locus.
        expected=np.asarray([surface.FirstUParameter()*radius,surface.LastUParameter()*radius,
            surface.FirstVParameter(),surface.LastVParameter()],dtype=float)
        bounds_residual=float(np.max(np.abs(expected-np.asarray([u0,u1,v0,v1]))))
        base['adaptor_bounds_residual_mm']=bounds_residual if math.isfinite(bounds_residual) else None
        if not np.isfinite(expected).all() or bounds_residual>tolerance:
            base.update(status='REFUSED',type_proven=False,reason='ADAPTOR_BOUNDS_NOT_PROVED_BOUNDARY_DOMAIN')
    return base

def extract_step_bend_axes(path: Path) -> dict[str, Any]:
    from OCP.BRepAdaptor import BRepAdaptor_Surface,BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Cylinder,GeomAbs_Plane,GeomAbs_Line
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE,TopAbs_EDGE,TopAbs_SOLID
    from OCP.TopExp import TopExp,TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS
    reader=STEPControl_Reader()
    if reader.ReadFile(str(path))!=IFSelect_RetDone or reader.TransferRoots()<=0:
        raise ValueError('STEP read/transfer failure')
    shape=reader.OneShape(); fmap=TopTools_IndexedMapOfShape(); emap=TopTools_IndexedMapOfShape()
    # Admission before building global inventories: bounded traversals and exactly
    # one valid solid. Multiple owners are not silently pooled into features.
    solids=TopExp_Explorer(shape,TopAbs_SOLID); solid_count=0; solid=None
    while solids.More():
        solid_count+=1; solid=solids.Current()
        if solid_count>1: break
        solids.Next()
    from OCP.BRepCheck import BRepCheck_Analyzer
    admission=None
    if solid_count!=1: admission='SINGLE_SOLID_REQUIRED'
    elif not BRepCheck_Analyzer(solid).IsValid(): admission='INVALID_NATIVE_SOLID'
    for kind,limit in ((TopAbs_FACE,MAX_NATIVE_FACES),(TopAbs_EDGE,MAX_NATIVE_EDGES)):
        if admission: break
        bounded=TopExp_Explorer(shape,kind); count=0
        while bounded.More():
            count+=1
            if count>limit: admission='NATIVE_TOPOLOGY_BUDGET_EXCEEDED'; break
            bounded.Next()
    if admission:
        return {'schema_version':5,'kind':'sherman_qc_step_topology_bend_axes','step_path':str(path.resolve()),
            'bend_axes':[],'partial_cylindrical_faces':[],'rejected_groups':[],
            'surface_diagnostics':[{'status':admission}],
            'contract':{'units':'mm_assumed_not_verified','admission_status':admission}}
    TopExp.MapShapes_s(shape,TopAbs_FACE,fmap); TopExp.MapShapes_s(shape,TopAbs_EDGE,emap)
    owners={}; solids=TopExp_Explorer(shape,TopAbs_SOLID); sid=0
    while solids.More():
        sid+=1; faces=TopExp_Explorer(solids.Current(),TopAbs_FACE)
        while faces.More():
            owners.setdefault(fmap.FindIndex(faces.Current()),set()).add(sid); faces.Next()
        solids.Next()
    edge_faces={}
    for index in range(1,fmap.Extent()+1):
        edges=TopExp_Explorer(fmap.FindKey(index),TopAbs_EDGE)
        while edges.More():
            edge_faces.setdefault(emap.FindIndex(edges.Current()),set()).add(index); edges.Next()
    rows=[]; diagnostics=[]
    for index in range(1,fmap.Extent()+1):
        face=TopoDS.Face_s(fmap.FindKey(index)); surface=BRepAdaptor_Surface(face)
        if surface.GetType()!=GeomAbs_Cylinder:
            diagnostics.append({'face_index':index,'surface_type':str(surface.GetType()),
                'status':'UNSUPPORTED_ANALYTIC_CYLINDER_CERTIFICATION'}); continue
        cyl=surface.Cylinder(); axis=cyl.Axis()
        xyz=lambda p: np.array([p.X(),p.Y(),p.Z()],dtype=float)
        location=xyz(axis.Location()); native=xyz(axis.Direction())
        a,ends=_native_axis_geometry(location,native,surface.FirstVParameter(),surface.LastVParameter())
        e1,e2=_basis(a); pos=cyl.Position(); x=xyz(pos.XDirection()); y=xyz(pos.YDirection())
        u0=surface.FirstUParameter(); u1=surface.LastUParameter()
        radial=math.cos(u0)*x+math.sin(u0)*y
        theta=math.atan2(np.dot(radial,e2),np.dot(radial,e1))
        orientation=1 if np.dot(np.cross(x,y),a)>0 else -1
        sweep=abs(u1-u0); start=theta if orientation>0 else theta-sweep
        edge_ids=[]; edges=TopExp_Explorer(face,TopAbs_EDGE)
        while edges.More(): edge_ids.append(emap.FindIndex(edges.Current())); edges.Next()
        ownership=owners.get(index,set())
        rails=[]
        for edge_id in sorted(set(edge_ids)):
            edge=TopoDS.Edge_s(emap.FindKey(edge_id)); curve=BRepAdaptor_Curve(edge)
            if curve.GetType()!=GeomAbs_Line: continue
            p=xyz(curve.Value(curve.FirstParameter())); q=xyz(curve.Value(curve.LastParameter()))
            if np.linalg.norm(q-p)<=DOMAIN_EPS or np.linalg.norm(np.cross(q-p,a))>DOMAIN_EPS: continue
            radial_point=p-location-a*np.dot(p-location,a)
            if np.linalg.norm(radial_point)<=DOMAIN_EPS: continue
            radial_point/=np.linalg.norm(radial_point)
            for neighbor in sorted(edge_faces.get(edge_id,set())-{index}):
                if owners.get(neighbor)!=ownership or len(ownership)!=1: continue
                ns=BRepAdaptor_Surface(TopoDS.Face_s(fmap.FindKey(neighbor)))
                if ns.GetType()!=GeomAbs_Plane: continue
                normal=xyz(ns.Plane().Axis().Direction())
                if abs(abs(float(np.dot(normal,radial_point)))-1.)>DOMAIN_EPS: continue
                rails.append({'angle_rad':math.atan2(np.dot(radial_point,e2),np.dot(radial_point,e1))%TAU,
                    'axial_interval':sorted([float(np.dot(p,a)),float(np.dot(q,a))]),
                    'edge_id':edge_id,'plane_face_index':neighbor})
        domain_certificate=_domain_certificate(face,surface,index,edge_ids)
        rows.append({'face_index':index,'solid_id':next(iter(ownership)) if len(ownership)==1 else None,
            'radius_mm':float(cyl.Radius()),'line_start':ends[0].tolist(),'line_end':ends[1].tolist(),
            'axial_span_mm':float(np.linalg.norm(ends[1]-ends[0])),
            'angular_sweep_deg':math.degrees(sweep),'angular_intervals':_intervals(start,sweep),
            'axial_interval':sorted(float(np.dot(p,a)) for p in ends),'edge_ids':sorted(set(edge_ids)),
            'domain_certified':domain_certificate['status']=='PROVED','native_type':'CYLINDER',
            'native_domain_certificate':domain_certificate,
            'tangent_rails':rails,'angular_frame':{'axis':a.tolist(),
                'x_axis':e1.tolist(),'y_axis':e2.tolist()}})
    result=build_features(rows)
    return {'schema_version':5,'kind':'sherman_qc_step_topology_bend_axes','step_path':str(path.resolve()),
        **result,'surface_diagnostics':diagnostics,'contract':{'units':'mm_assumed_not_verified',
        'marker_semantics':'GEOMETRIC_MID_THICKNESS_NOT_NEUTRAL_AXIS',
        'angular_frame_policy':'EXPLICIT_SOURCE_WORLD_RADIAL_TRANSPORT_TO_COMMON_STORED_FRAME',
        'minimum_angular_sweep_deg':v2.MIN_ANGULAR_SWEEP_DEG,'minimum_axial_span_mm':v2.MIN_AXIAL_SPAN_MM,
        'thresholds_are_legacy_eligibility_not_manufacturing_tolerances':True,
        'spline_cylinder_certification':'UNSUPPORTED_NO_SAMPLED_PROMOTION',
        'maximum_native_faces':MAX_NATIVE_FACES,'maximum_native_edges':MAX_NATIVE_EDGES,
        'carrier_position_tolerance_mm':CARRIER_POSITION_TOL_MM,
        'carrier_direction_cross_tolerance':CARRIER_DIRECTION_TOL,
        'forming_feature_semantics':'GEOMETRIC_PAIR_NOT_MANUFACTURING_OPERATION_PROOF'}}
