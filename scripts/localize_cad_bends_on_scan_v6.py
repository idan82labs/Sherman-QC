#!/usr/bin/env python3
"""Count-invariant dual-canonical relocation over EXP_CAD_0002."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.localize_cad_bends_on_scan_v5 import _key
from scripts.localize_cad_bends_on_scan_v1 import localize as localize_v1
def localize(template:dict,scan:Path,registration:dict)->dict:
 if template.get('schema')!='SHERMAN_QC_CAD_BEND_TEMPLATE_V2':raise ValueError('invalid CAD bend template V2')
 neutral=localize_v1({**template,'schema':'SHERMAN_QC_CAD_BEND_TEMPLATE_V1'},scan,registration)
 if neutral['status']=='ABSTAIN':return {**neutral,'schema':'SHERMAN_QC_CAD_ASSISTED_LOCALIZATION_V6','route':'CAD_COUNT_INVARIANT_DUAL_CANONICAL_RELOCATION_V6','canonical_arbitration':[]}
 curvature=localize_v1({**template,'schema':'SHERMAN_QC_CAD_BEND_TEMPLATE_V1','events':[{**e,'cad_line_start_mm':e['curvature_center_line_start_mm'],'cad_line_end_mm':e['curvature_center_line_end_mm']} for e in template['events']]},scan,registration)
 npred={x['bend_id']:x for x in neutral['predictions']};cpred={x['bend_id']:x for x in curvature['predictions']};selected=[];audit=[]
 for event,nrec,crec in zip(template['events'],neutral['records'],curvature['records']):
  if event['event_id'] not in npred:continue
  variant='CURVATURE' if event['event_id'] in cpred and _key(crec,'CURVATURE')>_key(nrec,'NEUTRAL') else 'NEUTRAL';p=(cpred if variant=='CURVATURE' else npred)[event['event_id']];selected.append({**p,'assignment_source':'CAD_COUNT_INVARIANT_DUAL_CANONICAL_RELOCATION','canonical_variant':variant});audit.append({'event_id':event['event_id'],'selected_variant':variant})
 selected.sort(key=lambda x:x['bend_id'])
 return {**neutral,'schema':'SHERMAN_QC_CAD_ASSISTED_LOCALIZATION_V6','route':'CAD_COUNT_INVARIANT_DUAL_CANONICAL_RELOCATION_V6','predictions':selected,'localized_bend_count':len(selected),'canonical_arbitration':audit}
