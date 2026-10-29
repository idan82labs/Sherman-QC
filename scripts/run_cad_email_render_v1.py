"""Bounded derived-only rendering of a single pinned email CAD evidence set."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))


def run(evidence,evidence_sha,output,prediction=None,prediction_sha=None):
    from scripts.evaluate_vendor_cad_upload_unfold_v2 import _ps_rows,_tree,_kill_seen_groups
    supervision=output.with_suffix(".supervision")
    supervision.mkdir(exist_ok=False)
    command=[sys.executable,"scripts/render_cad_email_transfer_v1.py","--evidence",str(evidence),"--evidence-sha256",evidence_sha,"--output",str(output)]
    if prediction is not None:
        command += ["--prediction",str(prediction),"--prediction-sha256",prediction_sha]
    pins={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("scripts/run_cad_email_render_v1.py","scripts/render_cad_email_transfer_v1.py","scripts/evaluate_vendor_cad_upload_unfold_v2.py")}
    def accounting():
        logs=sum((supervision/name).stat().st_size for name in ("stdout.log","stderr.log") if (supervision/name).exists())
        artifacts=sum(p.stat().st_size for p in supervision.rglob("*") if p.is_file())+sum(p.stat().st_size for p in (output,output.with_suffix(".json")) if p.exists())
        return logs,artifacts
    def write(name,value):
        with (supervision/name).open("x") as stream: json.dump(value,stream,indent=2,sort_keys=True)
    write("FROZEN.json",{"command":command,"source_pins":pins,"seconds":120,"rss_kib":2*1024**2,"log_bytes":1_000_000,"artifact_bytes":40_000_000,"raw_cad_access":False,"retry":False})
    process=None; groups=set(); reason=None; peak=0; cleaned=False; start=time.monotonic()
    try:
        with (supervision/"stdout.log").open("xb") as out,(supervision/"stderr.log").open("xb") as err:
            process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,start_new_session=True,env={**os.environ,"MPLCONFIGDIR":str(supervision/"mpl-cache"),"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
            groups.add(process.pid)
            while process.poll() is None:
                rows=_ps_rows(); owned=_tree(rows,process.pid); groups.update(rows[p][1] for p in owned)
                rss=sum(row[2] for row in rows.values() if row[1] in groups); peak=max(peak,rss)
                logs,artifacts=accounting()
                if time.monotonic()-start>=120 or rss>2*1024**2 or logs>1_000_000 or artifacts+8192>40_000_000:
                    reason="RESOURCE_BUDGET"; break
                time.sleep(.05)
    except Exception as exc: reason="SUPERVISION_FAILURE:"+type(exc).__name__
    finally:
        try:
            cleaned=_kill_seen_groups(groups) if groups else process is None
            if process is not None: process.wait(timeout=3)
        except Exception: cleaned=False
        try:
            logs,artifacts=accounting()
            if logs>1_000_000 or artifacts+8192>40_000_000: reason=reason or "FINAL_STORAGE_BUDGET"
            if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=pin for name,pin in pins.items()): reason=reason or "SOURCE_CHANGED"
        except Exception:
            logs=artifacts=None; reason=reason or "FINAL_ACCOUNTING_FAILURE"
        terminal={"outer_exit":None if process is None else process.returncode,"cleanup_known":cleaned,"reason":reason,"elapsed_seconds":time.monotonic()-start,"peak_sampled_rss_kib":peak,"log_bytes":logs,"artifact_bytes_before_terminal":artifacts}
        write("TERMINAL.json",terminal)
    print(json.dumps(terminal)); return 0 if terminal["outer_exit"]==0 and cleaned and reason is None else 1


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence",required=True,type=Path); p.add_argument("--evidence-sha256",required=True)
    p.add_argument("--output",required=True,type=Path); p.add_argument("--prediction",type=Path); p.add_argument("--prediction-sha256")
    a=p.parse_args(); raise SystemExit(run(a.evidence,a.evidence_sha256,a.output,a.prediction,a.prediction_sha256))
