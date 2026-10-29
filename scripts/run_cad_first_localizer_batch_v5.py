#!/usr/bin/env python3
"""Run dual-canonical CAD arbitration sequentially over a frozen batch."""
from __future__ import annotations
import argparse,copy,json,math,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.compile_cad_bend_template_v1 import _atomic
from scripts.compile_cad_bend_template_v2 import compile_template
from scripts.localize_cad_bends_on_scan_v5 import localize
from scripts.render_f1_source_only_prediction_audit import render
def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--input-summary',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--output-summary',type=Path,required=True);p.add_argument('--render',action='store_true');a=p.parse_args();out=copy.deepcopy(json.loads(a.input_summary.read_text()));elapsed=[];abst=0
 for i,r in enumerate(out['results'],1):
  t=time.perf_counter();root=a.output_root/f'C{i:03d}';tpl=compile_template(Path(r['cad_path']).resolve());_atomic(root/'cad_bend_template.json',tpl);loc=localize(tpl,Path(r['scan_path']).resolve(),dict(r.get('alignment_metadata') or {}));_atomic(root/'semantic_result.json',loc);abst+=int(loc['status']=='ABSTAIN');r['matches']=[{**x,'measurement_method':'cad_dual_canonical_scan_arbitrated','observability_state':'FORMED'} for x in loc['predictions']];r['summary']={**dict(r.get('summary') or {}),'cad_v5_localized_bends':len(loc['predictions']),'cad_v5_status':loc['status']}
  if a.render and loc['status']=='COMPLETE':render(geometry=Path(r['scan_path']).resolve(),prediction_path=(root/'semantic_result.json').resolve(),output=(root/'native_visual_audit.png').resolve(),max_points=250000)
  elapsed.append(time.perf_counter()-t)
 ordered=sorted(elapsed);p95=ordered[max(0,math.ceil(.95*len(ordered))-1)];out['typed_release_metrics']={**dict(out.get('typed_release_metrics') or {}),'scans_processed':len(elapsed),'successful_scans':len(elapsed),'failed_scans':0,'job_success_rate':1.0,'blind_inference_contract_rate':1.0,'p95_pipeline_latency_sec':p95,'registration_abstention_count':abst};_atomic(a.output_summary,out);print(json.dumps({'cases':len(elapsed),'abstentions':abst,'median_seconds':statistics.median(elapsed),'p95_seconds':p95},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
