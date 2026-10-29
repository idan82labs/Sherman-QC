"""Turn raw `zs_rust lane` stage dumps (<run>/t<N>/<sid>/dump.json, from portable_run.py --dump) into the DEV dump
layout of zs_lane_v3_rs.dev_dump (<out>/<sid>.json: NaN restored as the adapter does, scan_id / scan_path set), for
oof.py. Point-cloud scans (exit 10: the binary does not process them; the adapter runs the frozen Python revision-2
lane) are copied unchanged from the reference dump directory given by --pc-from.
usage: .venv/bin/python -B docs/.../v3_rust/tools/portable_dump_prep.py RUN_T_DIR LIST.tsv OUT_DIR --pc-from REF_DUMP_DIR"""
import argparse, json, shutil, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v3_rs.lane import _nan_fields
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("list"); ap.add_argument("out"); ap.add_argument("--pc-from", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
n = pc = 0
for line in Path(a.list).read_text().splitlines():
    sid, path = line.split("\t")[:2]
    src = Path(a.run) / sid / "dump.json"
    if src.is_file():
        d = _nan_fields(json.loads(src.read_text()))
        d["scan_id"], d["scan_path"] = sid, path
        (out / f"{sid}.json").write_text(json.dumps(d)); n += 1
    else:
        shutil.copy2(Path(a.pc_from) / f"{sid}.json", out / f"{sid}.json"); pc += 1
print(json.dumps({"binary_dumps": n, "point_cloud_copied": pc}))
