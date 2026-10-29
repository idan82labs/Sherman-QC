"""Auditor-1 (sc_cec08dfca9ab) helpers: part frame, gold lines. Private scratch dir only."""
import json
import numpy as np

W = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_cec08dfca9ab_k7v2/"
OUTD = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_cec08dfca9ab/"
GOLD = "/Users/idantal/ShermanQC/FullProject/output/bend_label_03_10839000/bend_instance_labels.json"


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def dominant_dirs(nrm, k=6, seed=0):
    rng = np.random.default_rng(seed)
    sub = nrm[rng.choice(len(nrm), 60000, replace=False)]
    rem = sub.copy(); out = []
    for _ in range(k):
        cand = rem[rng.choice(len(rem), min(400, len(rem)), replace=False)]
        cnt = (np.abs(cand @ rem.T) > np.cos(np.radians(5))).sum(1)
        cc = cand[np.argmax(cnt)]
        m = np.abs(rem @ cc) > np.cos(np.radians(5))
        pts = rem[m] * np.sign(rem[m] @ cc)[:, None]
        cc = unit(pts.mean(0))
        m = np.abs(rem @ cc) > np.cos(np.radians(8))
        out.append((cc, int(m.sum())))
        rem = rem[~m]
        if len(rem) < 500:
            break
    return out


def load():
    d = np.load(W + "cec08_scan.npz")
    xyz, nrm, faces = d["xyz"], d["nrm"], d["faces"]
    try:
        D = np.load(W + "cec08_domdirs.npy")
    except FileNotFoundError:
        D = np.array([c for c, n in dominant_dirs(nrm)])
        np.save(W + "cec08_domdirs.npy", D)
    ez = unit(D[0])
    ex = unit(D[1] - (D[1] @ ez) * ez)
    ey = np.cross(ez, ex)
    R = np.stack([ex, ey, ez])
    c = xyz.mean(0)
    gold = json.load(open(GOLD))["manual_bends"]
    G = {g["bend_id"]: (np.array(g["line_start"], float), np.array(g["line_end"], float)) for g in gold}
    return xyz, nrm, faces, R, c, G


def to_part(p, R, c):
    return (np.asarray(p) - c) @ R.T


def from_part(q, R, c):
    return np.asarray(q) @ R + c
