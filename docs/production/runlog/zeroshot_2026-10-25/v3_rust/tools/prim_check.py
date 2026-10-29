"""Primitive equivalence: numpy default_rng (random, choice variants) and scipy cKDTree kNN order vs zs_rust selftest."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
REPO = Path(__file__).resolve().parents[6]
BIN = REPO / "research/zeroshot/zs_rust/target/release/zs_rust"
W = Path(sys.argv[1]); W.mkdir(parents=True, exist_ok=True)
rng0 = np.random.default_rng(123)
p = rng0.random(50000) ** 3; p /= p.sum(); np.save(W / "prob.npy", p)
ops = [{"seed": 0, "op": "random", "n": 1000}, {"seed": 1, "op": "random", "n": 10},
       {"seed": 0, "op": "choice_noreplace", "pop": 200000, "size": 150000},
       {"seed": 1, "op": "choice_noreplace", "pop": 2_000_000, "size": 20000},
       {"seed": 0, "op": "choice_noreplace", "pop": 9000, "size": 9000},
       {"seed": 0, "op": "choice_noreplace", "pop": 5000, "size": 300},
       {"seed": 0, "op": "choice_p", "p": "prob.npy", "size": 20000},
       {"seed": 0, "op": "choice_p_noreplace", "p": "prob.npy", "size": 20000}]
ref = []
for o in ops:
    r = np.random.default_rng(o["seed"])
    if o["op"] == "random": ref.append(r.random(o["n"]).tolist())
    elif o["op"] == "choice_noreplace": ref.append(r.choice(o["pop"], o["size"], replace=False).tolist())
    elif o["op"] == "choice_p": ref.append(r.choice(len(p), size=o["size"], p=p).tolist())
    else: ref.append(r.choice(len(p), size=o["size"], replace=False, p=p).tolist())
# knn on a jittered grid (many exact ties) and on random points
g = np.stack(np.meshgrid(np.arange(60), np.arange(60), np.arange(8), indexing="ij"), -1).reshape(-1, 3) * 0.5
pts = np.vstack([g, rng0.normal(size=(20000, 3)) * 30])
np.save(W / "pts.npy", pts)
job = {"rng": ops, "knn": {"P": "pts.npy", "k": 17}}
(W / "job.json").write_text(json.dumps(job))
subprocess.run([str(BIN), "selftest", "--job", str(W / "job.json"), "--out", str(W / "out.json"), "--threads", "4"], check=True)
out = json.loads((W / "out.json").read_text())
for o, a, b in zip(ops, ref, out["rng"]):
    print(o["op"], o.get("pop"), o.get("size"), "EQUAL" if a == b else f"DIFF first at {next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y) if len(a)==len(b) else 'len'}")
dd, ii = cKDTree(pts).query(pts, k=17)
ri = np.load(W / "knn_idx.npy"); rd = np.load(W / "knn_d.npy")
print("knn idx equal", np.array_equal(ii, ri), "rows differ", int((ii != ri).any(1).sum()), "dist equal", np.array_equal(dd, rd),
      "rows with ties", int((dd[:, 1:] == dd[:, :-1]).any(1).sum()))
