#!/usr/bin/env python3
"""Fail-closed metadata audit for physical-scan qualification capacity."""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile
from pathlib import Path

ALLOWED={"DEVELOPMENT_CONSUMED","DEVELOPMENT_CONSUMED_EVALUATOR_OPAQUE","UNTOUCHED_PENDING_CUSTODIAN","UNTOUCHED_CUSTODIAN_SEALED"}

def audit(registry:dict,root:Path)->dict:
 if registry.get('schema')!='SHERMAN_QC_PHYSICAL_LINEAGE_REGISTRY_V2':raise ValueError('invalid lineage registry schema')
 rows=[];seen_roots=set();sha_status={}
 for item in registry.get('intakes') or []:
  rel=str(item.get('root') or '');status=str(item.get('status') or '')
  if not rel or rel in seen_roots:raise ValueError('intake roots must be nonempty and unique')
  if status not in ALLOWED:raise ValueError(f'invalid custody status: {status}')
  seen_roots.add(rel);path=(root/rel).resolve();files=sorted(path.rglob('*.ply')) if path.is_dir() else []
  hashes=[]
  for p in files:
   digest=hashlib.sha256(p.read_bytes()).hexdigest();hashes.append(digest);sha_status.setdefault(digest,set()).add(status)
  rows.append({'root_token':hashlib.sha256(rel.encode()).hexdigest()[:16],'status':status,'ply_count':len(files),'unique_scan_hash_count':len(set(hashes))})
 conflicts=[h for h,s in sha_status.items() if 'UNTOUCHED_CUSTODIAN_SEALED' in s and len(s)>1]
 eligible={h for h,s in sha_status.items() if s=={'UNTOUCHED_CUSTODIAN_SEALED'}}
 policy=registry['qualification_policy'];minimum=int(policy['minimum_unique_scans'])
 return {'schema':'SHERMAN_QC_PHYSICAL_LINEAGE_CAPACITY_AUDIT_V2','status':'FAIL_CLOSED' if conflicts or len(eligible)<minimum else 'SCAN_QUANTITY_READY_FAMILY_CUSTODY_STILL_REQUIRED','intakes':rows,'unique_physical_scan_hashes':len(sha_status),'untouched_eligible_unique_scans':len(eligible),'minimum_unique_scans':minimum,'minimum_morphology_families':int(policy['minimum_morphology_families']),'sealed_vs_consumed_hash_conflicts':len(conflicts),'family_membership_opened':False,'truth_opened':False,'predictions_opened':False}

def atomic(path:Path,payload:dict)->None:
 path.parent.mkdir(parents=True,exist_ok=True);data=(json.dumps(payload,sort_keys=True,separators=(',',':'))+'\n').encode();fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=audit(json.loads(a.registry.read_text()),a.root.resolve());atomic(a.output,x);print(json.dumps({'status':x['status'],'unique_scans':x['unique_physical_scan_hashes'],'eligible':x['untouched_eligible_unique_scans'],'conflicts':x['sealed_vs_consumed_hash_conflicts']},sort_keys=True));return 0 if x['status']!='FAIL_CLOSED' else 2
if __name__=='__main__':raise SystemExit(main())
