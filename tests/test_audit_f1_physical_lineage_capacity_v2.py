from pathlib import Path
from scripts.audit_f1_physical_lineage_capacity_v2 import audit

def registry(rows):
 return {'schema':'SHERMAN_QC_PHYSICAL_LINEAGE_REGISTRY_V2','intakes':rows,'qualification_policy':{'eligible_status':'UNTOUCHED_CUSTODIAN_SEALED','minimum_unique_scans':2,'minimum_morphology_families':1,'model_implementer_may_materialize_seal':False}}

def test_consumed_files_never_count_as_eligible(tmp_path):
 d=tmp_path/'consumed';d.mkdir();(d/'a.ply').write_bytes(b'a')
 out=audit(registry([{'root':'consumed','status':'DEVELOPMENT_CONSUMED'}]),tmp_path)
 assert out['untouched_eligible_unique_scans']==0 and out['status']=='FAIL_CLOSED'

def test_duplicate_between_consumed_and_sealed_is_conflict(tmp_path):
 a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir();(a/'x.ply').write_bytes(b'x');(b/'x.ply').write_bytes(b'x')
 out=audit(registry([{'root':'a','status':'DEVELOPMENT_CONSUMED'},{'root':'b','status':'UNTOUCHED_CUSTODIAN_SEALED'}]),tmp_path)
 assert out['sealed_vs_consumed_hash_conflicts']==1 and out['status']=='FAIL_CLOSED'

def test_unique_sealed_quantity_can_pass_scan_gate(tmp_path):
 d=tmp_path/'sealed';d.mkdir();(d/'a.ply').write_bytes(b'a');(d/'b.ply').write_bytes(b'b')
 out=audit(registry([{'root':'sealed','status':'UNTOUCHED_CUSTODIAN_SEALED'}]),tmp_path)
 assert out['untouched_eligible_unique_scans']==2
 assert out['status']=='SCAN_QUANTITY_READY_FAMILY_CUSTODY_STILL_REQUIRED'
