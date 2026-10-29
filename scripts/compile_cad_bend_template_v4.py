#!/usr/bin/env python3
"""Compile retained cylindrical events plus coherent major planar-region folds."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if str(ROOT/'backend') not in sys.path:sys.path.insert(0,str(ROOT/'backend'))
import open3d as o3d
from cad_import import import_cad_file
from cad_bend_extractor import CADBendExtractor
from scripts.compile_cad_bend_template_v1 import _atomic,_sha
from scripts.compile_cad_bend_template_v2 import compile_template as compile_v2

def compile_template(step:Path)->dict:
    base=compile_v2(step); imported=import_cad_file(str(step),deflection=.05)
    if not imported.success or imported.mesh is None:raise ValueError(f'CAD tessellation failed: {imported.error}')
    mesh=o3d.geometry.TriangleMesh();mesh.vertices=o3d.utility.Vector3dVector(np.asarray(imported.mesh.vertices));mesh.triangles=o3d.utility.Vector3iVector(np.asarray(imported.mesh.faces))
    extraction=CADBendExtractor(coplanar_threshold=15.0,min_area_pct=.2,max_bend_gap=20.0,min_bend_angle=8.0,dedup_distance=8.0).extract_from_mesh(mesh)
    events=list(base['events'])
    for bend in extraction.bends:
        events.append({'event_id':f'CAD_EVENT_{len(events)+1:04d}','source_bend_id':f'PLANAR_REGION_{bend.bend_id:04d}','step_face_indices':[],'cad_line_start_mm':bend.bend_line_start.tolist(),'cad_line_end_mm':bend.bend_line_end.tolist(),'radii_mm':[],'angular_sweeps_deg':[],'axial_span_mm':float(np.linalg.norm(bend.bend_line_end-bend.bend_line_start)),'relation':'MAJOR_PLANAR_REGION_PAIR_TO_ONE_FOLD_EVENT','canonical_line_rule':'PLANAR_REGION_INTERSECTION','surface_region_ids':[bend.surface1_id,bend.surface2_id],'normal_angle_deg':bend.angle_between_normals})
    return {**base,'schema':'SHERMAN_QC_CAD_BEND_TEMPLATE_V4','event_count':len(events),'events':events,'planar_region_event_count':extraction.total_bends,'cad_tessellator_sha256':_sha(ROOT/'backend/cad_import.py'),'planar_region_extractor_sha256':_sha(ROOT/'backend/cad_bend_extractor.py')}

def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--step',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=compile_template(a.step.resolve());_atomic(a.output,x);print(json.dumps({'status':'COMPLETE','event_count':x['event_count'],'planar_region_event_count':x['planar_region_event_count']},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
