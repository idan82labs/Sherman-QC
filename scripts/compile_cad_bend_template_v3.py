#!/usr/bin/env python3
"""Compile cylindrical and paired sharp-planar STEP folds into one template."""

from __future__ import annotations

import argparse, json, statistics, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from scripts.compile_cad_bend_template_v1 import _atomic, _sha
from scripts.compile_cad_bend_template_v2 import compile_template as compile_v2
from scripts.extract_step_planar_fold_events import extract_planar_fold_events


def compile_template(step: Path) -> dict:
    base=compile_v2(step)
    separations=[max(e["radii_mm"])-min(e["radii_mm"]) for e in base["events"] if len(e["radii_mm"])>=2]
    if not separations: raise ValueError("cannot infer material thickness from STEP cylindrical skin pairs")
    thickness=float(statistics.median(separations))
    sharp=extract_planar_fold_events(step,material_thickness_mm=thickness)
    events=list(base["events"])
    for row in sharp["events"]:
        events.append({**row,"event_id":f"CAD_EVENT_{len(events)+1:04d}","source_bend_id":row["event_id"],"step_face_indices":[],"radii_mm":[],"angular_sweeps_deg":[],"axial_span_mm":float(((sum((a-b)**2 for a,b in zip(row['cad_line_start_mm'],row['cad_line_end_mm'])))**0.5)),"canonical_line_rule":"PAIRED_PLANAR_SKIN_MIDPOINT"})
    return {**base,"schema":"SHERMAN_QC_CAD_BEND_TEMPLATE_V3","event_count":len(events),"events":events,"sharp_planar_event_count":sharp["event_count"],"raw_planar_dihedral_edge_count":sharp["raw_planar_dihedral_edges"],"material_thickness_mm":thickness,"extractor_v3_sha256":_sha(Path(__file__).with_name("extract_step_planar_fold_events.py"))}


def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--step',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=compile_template(a.step.resolve());_atomic(a.output,x);print(json.dumps({'status':'COMPLETE','event_count':x['event_count'],'sharp_planar_event_count':x['sharp_planar_event_count']},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
