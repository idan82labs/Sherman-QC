"""Job list for the A0 (scan-only typed-union = best_f1 without CAD) baseline runs. No labels read."""
import json, os
from pathlib import Path
H = Path(__file__).resolve().parents[1]
inv = {r["scan_id"]: r for r in json.loads((H / "INVENTORY.json").read_text())["scans"]}
sp = json.loads((H / "SPLIT_FROZEN.json").read_text())
W = "/private/tmp/claude-501/sherman_zeroshot_work"
jobs = []
for sid in ("sc_03395097cd96", "sc_050cba324c35"):  # determinism replay of two existing A0 outputs
    jobs.append({"id": sid, "kind": "replay", "scan": os.path.expanduser(inv[sid]["path"]),
                 "out": str(H / "baseline/a0_replay" / sid), "work": f"{W}/replay/{sid}"})
for sid in sp["dev"]["scans"]:
    if "A0_ZERO_SHOT_OUTPUT_EXISTS" not in inv[sid]["engine_history"]:
        jobs.append({"id": sid, "kind": "dev", "scan": os.path.expanduser(inv[sid]["path"]),
                     "out": str(H / "baseline/a0_runs" / sid), "work": f"{W}/dev/{sid}"})
for sid in sp["test_b"]["scans"]:
    jobs.append({"id": sid, "kind": "testb", "scan": os.path.expanduser(inv[sid]["path"]),
                 "out": str(H / "baseline/a0_runs_TESTB_SEALED" / sid), "work": f"{W}/testb_SEALED/{sid}"})
(H / "baseline/a0_jobs.json").write_text(json.dumps(jobs, indent=1))
print(len(jobs), sum(not Path(j["scan"]).is_file() for j in jobs))
