#!/usr/bin/env python3
"""Localize V4 cylindrical plus planar-region CAD events."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.localize_cad_bends_on_scan_v1 import localize as localize_v1
def localize(template:dict,scan:Path,registration:dict)->dict:
 if template.get('schema')!='SHERMAN_QC_CAD_BEND_TEMPLATE_V4':raise ValueError('invalid CAD bend template V4')
 result=localize_v1({**template,'schema':'SHERMAN_QC_CAD_BEND_TEMPLATE_V1'},scan,registration)
 result.update({'schema':'SHERMAN_QC_CAD_ASSISTED_LOCALIZATION_V4','route':'CAD_CYLINDER_PLUS_PLANAR_REGION_SCAN_LOCALIZED_V4','cad_template_sha256':hashlib.sha256((json.dumps(template,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()})
 for row in result['predictions']:row['assignment_source']='CAD_CYLINDER_OR_PLANAR_REGION_SCAN_LOCALIZED'
 return result
