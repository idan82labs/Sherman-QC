"""cProfile of zs_lane_v3 core.process on given scans (timing only; no scoring). Guard + custody per scan.
usage: sherman_slot.sh .venv/bin/python -B docs/.../v3_rust/tools/profile_v3.py --out DIR scan [scan ...]"""
import argparse, cProfile, json, pstats, sys, time, io
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from research.zeroshot.zs_lane_v3_rs import rev2ref; rev2ref.install()  # frozen revision 2, not the live package
from research.zeroshot.zs_lane_v3 import core
from research.zeroshot.common.gold_guard import require_runnable
from research.zeroshot.z3_slicing.custody import check_path

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("scans", nargs="*"); ap.add_argument("--list", default="")
a = ap.parse_args()
if a.list: a.scans += [l.rstrip("\n") for l in open(a.list) if l.strip()]
cfg = json.loads(rev2ref.REV2_CONFIG.read_text())
core.configure(cfg, "zs")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for s in a.scans:
    d = require_runnable(s, final_testb=False)
    path = str(check_path(s))
    pr = cProfile.Profile(); t0 = time.time()
    pr.enable(); rec = core.process(path); pr.disable()
    wall = time.time() - t0
    sid = d["scan_id"]
    pr.dump_stats(str(out / f"{sid}.prof"))
    sio = io.StringIO(); st = pstats.Stats(pr, stream=sio); st.sort_stats("cumulative").print_stats(60)
    st.sort_stats("tottime").print_stats(40)
    (out / f"{sid}.txt").write_text(sio.getvalue())
    (out / f"{sid}.json").write_text(json.dumps({"scan_id": sid, "wall_s": wall, "stages": rec["stages"],
                                                 "rss_gib": core.V2._rss_gib()}, indent=1))
    print(sid, round(wall, 1), json.dumps(rec["stages"]), flush=True)
