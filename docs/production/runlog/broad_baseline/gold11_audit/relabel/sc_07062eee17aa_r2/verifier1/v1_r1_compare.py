"""Compare r1 vs r2 B2 line against verifier1's measured outer-apex stations (no scan load)."""
import json, numpy as np, os
H = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(H, "apex3d_B2.npz")); A = D["apex"]; BL = D["bl3"]
for tag, p in (("r1", os.path.join(H, "../../sc_07062eee17aa/lines.json")), ("r2", os.path.join(H, "../lines.json"))):
    B = [b for b in json.load(open(p))["bends"] if b["bend_id"] == "B2"][0]
    s = np.array(B["label_line"]["start"]); e = np.array(B["label_line"]["end"]); d = (e - s) / np.linalg.norm(e - s)
    foot = s + np.outer((A - s) @ d, d); r = foot - A
    dep = np.einsum("ij,ij->i", r, BL)  # + = label deeper than outer apex (into material); - = toward sharp
    dist = np.linalg.norm(r, axis=1)
    print(tag, "n=%d depth median %+.3f p10 %+.3f p90 %+.3f | dist median %.3f p90 %.3f max %.3f" % (len(A), np.median(dep), np.percentile(dep, 10), np.percentile(dep, 90), np.median(dist), np.percentile(dist, 90), dist.max()))
