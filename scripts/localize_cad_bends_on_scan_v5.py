#!/usr/bin/env python3
"""Select curvature-center or neutral-surface line per CAD event from scan evidence."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.apply_f1_step_topology_targeted_verifier import _duplicate
from scripts.localize_cad_bends_on_scan_v1 import localize as localize_v1

def _key(record:dict,variant:str)->tuple:
 s=record.get('scan_support') or {};t=record.get('normal_turn_verification') or {}
 return (int(record.get('accepted') is True),int(t.get('passed') is True),float(t.get('minimum_side_fraction') or 0),float(s.get('support_fraction') or 0),-float(s.get('median_distance_mm') or 1e30),int(variant=='NEUTRAL'))

def localize(template:dict,scan:Path,registration:dict)->dict:
 if template.get('schema')!='SHERMAN_QC_CAD_BEND_TEMPLATE_V2':raise ValueError('invalid CAD bend template V2')
 neutral={**template,'schema':'SHERMAN_QC_CAD_BEND_TEMPLATE_V1'}
 curvature={**neutral,'events':[{**e,'cad_line_start_mm':e['curvature_center_line_start_mm'],'cad_line_end_mm':e['curvature_center_line_end_mm']} for e in template['events']]}
 nr=localize_v1(neutral,scan,registration);cr=localize_v1(curvature,scan,registration)
 if nr['status']=='ABSTAIN':
  return {**nr,'schema':'SHERMAN_QC_CAD_ASSISTED_LOCALIZATION_V5','route':'CAD_DUAL_CANONICAL_SCAN_ARBITRATED_V5','canonical_arbitration':[]}
 npred={x['bend_id']:x for x in nr['predictions']};cpred={x['bend_id']:x for x in cr['predictions']};selected=[];audit=[]
 for event,nrec,crec in zip(template['events'],nr['records'],cr['records']):
  variant='NEUTRAL' if _key(nrec,'NEUTRAL')>=_key(crec,'CURVATURE') else 'CURVATURE';rec=nrec if variant=='NEUTRAL' else crec;pred=(npred if variant=='NEUTRAL' else cpred).get(event['event_id'])
  accepted=False
  if pred is not None and not any(_duplicate(pred,x) for x in selected):selected.append({**pred,'assignment_source':'CAD_DUAL_CANONICAL_SCAN_ARBITRATED','canonical_variant':variant});accepted=True
  audit.append({'event_id':event['event_id'],'selected_variant':variant,'selected_key':_key(rec,variant),'accepted':accepted})
 selected.sort(key=lambda x:x['bend_id'])
 return {**nr,'schema':'SHERMAN_QC_CAD_ASSISTED_LOCALIZATION_V5','route':'CAD_DUAL_CANONICAL_SCAN_ARBITRATED_V5','cad_template_sha256':hashlib.sha256((json.dumps(template,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest(),'localized_bend_count':len(selected),'predictions':selected,'canonical_arbitration':audit}
