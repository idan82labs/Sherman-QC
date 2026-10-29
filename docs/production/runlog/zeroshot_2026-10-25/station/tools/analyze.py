"""Collect DENSE_READINESS records -> results.json (+ prints tables)."""
import glob
import json
import os
import sys

R = "/private/tmp/claude-501/dense_ready/runs"
H = os.path.dirname(os.path.abspath(__file__))
meta = json.load(open(f"{H}/meta.json"))
out_path = sys.argv[1] if len(sys.argv) > 1 else f"{H}/results.json"


def load(name):
    p = f"{R}/{name}/record.json"
    return json.load(open(p)) if os.path.exists(p) else None


def faces(tag, lab):
    if lab == "n1":
        src = meta[tag]["src"]
    else:
        src = f"/private/tmp/claude-501/dense_ready/meshes/{tag}_{lab}.ply"
    with open(src, "rb") as fh:
        v = f = None
        for _ in range(30):
            l = fh.readline().decode("latin1").strip()
            if l.startswith("element vertex"):
                v = int(l.split()[-1])
            if l.startswith("element face"):
                f = int(l.split()[-1])
            if l == "end_header":
                break
    return v, f, os.path.getsize(src)


def e3_stage_durations(rec):
    st = rec.get("stage_starts") or []
    end = rec.get("engine_log_last_s")
    out = {}
    for i, (name, t) in enumerate(st):
        nxt = st[i + 1][1] if i + 1 < len(st) else end
        if nxt is not None:
            out[name] = round(nxt - t, 2)
    return out


def cmp_e3(base, rec):
    if not base or not rec or "chars" not in base or "chars" not in rec:
        return None
    b = {(c[0], c[2], c[3]): c for c in base["chars"]}
    d = {(c[0], c[2], c[3]): c for c in rec["chars"]}
    keys = sorted(set(b) | set(d), key=str)
    dec_changes, state_changes, bend_state_changes, dv = [], [], [], []
    for k in keys:
        x, y = b.get(k), d.get(k)
        if x is None or y is None:
            dec_changes.append({"char": k, "base": x and x[5], "dense": y and y[5]})
            continue
        if x[1] != y[1]:
            bend_state_changes.append({"char": k, "base": x[1], "dense": y[1]})
        if x[4] != y[4]:
            state_changes.append({"char": k, "base": [x[4], x[8]], "dense": [y[4], y[8]]})
        if x[5] != y[5]:
            dec_changes.append({"char": k, "base": [x[5], x[6]], "dense": [y[5], y[6]]})
        if isinstance(x[6], (int, float)) and isinstance(y[6], (int, float)):
            dv.append({"char": "/".join(str(t) for t in k), "kind": k[1], "delta": round(y[6] - x[6], 4),
                       "U": x[7] if isinstance(x[7], (int, float)) else None})
    worst = sorted(dv, key=lambda r: -abs(r["delta"]))[:5]
    by_kind = {}
    for r in dv:
        by_kind.setdefault(r["kind"], []).append(abs(r["delta"]))
    return {"state_base": [base.get("state"), base.get("summary"), base.get("stage_status")],
            "state_dense": [rec.get("state"), rec.get("summary"), rec.get("stage_status")],
            "validity_summary_equal": [base.get("state"), base.get("summary"), base.get("stage_status")] ==
                                      [rec.get("state"), rec.get("summary"), rec.get("stage_status")],
            "n_chars": len(keys), "decision_changes": dec_changes, "char_state_changes": state_changes,
            "bend_state_changes": bend_state_changes,
            "max_abs_delta_by_kind": {k: round(max(v), 4) for k, v in by_kind.items()},
            "worst_value_deltas": worst}


def line_dist(a, b):
    import math
    def pd(p, q):
        return math.dist(p, q)
    s1 = max(pd(a["start_mm"], b["start_mm"]), pd(a["end_mm"], b["end_mm"]))
    s2 = max(pd(a["start_mm"], b["end_mm"]), pd(a["end_mm"], b["start_mm"]))
    return min(s1, s2)


def cmp_zs(base, rec):
    if not base or not rec or base.get("lines") is None or rec.get("lines") is None:
        return None
    A, B = base["lines"], rec["lines"]
    used, pairs = set(), []
    for a in A:
        best = None
        for j, b in enumerate(B):
            if j in used:
                continue
            dd = line_dist(a, b)
            if best is None or dd < best[0]:
                best = (dd, j)
        if best and best[0] < 6.0:
            used.add(best[1])
            pairs.append(round(best[0], 3))
    return {"n_base": len(A), "n_dense": len(B), "matched_within_6mm": len(pairs),
            "max_endpoint_shift_mm": max(pairs) if pairs else None, "identical": base.get("lines_sha256") == rec.get("lines_sha256")}


rows = []
for tag, m in meta.items():
    labs = ["n1"] + [f"n{n}" for n in m["ns"]]
    b3, bz = load(f"{tag}_n1_e3_t1"), load(f"{tag}_n1_zs_t1")
    for lab in labs:
        try:
            v, f, size = faces(tag, lab)
        except FileNotFoundError:
            continue
        e3, zs = load(f"{tag}_{lab}_e3_t1"), load(f"{tag}_{lab}_zs_t1")
        row = {"set": tag, "scan_id": m["scan_id"], "mesh": "original" if lab == "n1" else f"subdiv N={lab[1:]}",
               "vertices": v, "faces": f, "bytes": size}
        if e3:
            row["e3"] = {k: e3.get(k) for k in ("rc", "wall_s", "peak_rss_bytes", "timed_out", "rss_exceeded", "state",
                                                "summary", "stage_status")}
            row["e3"]["stages_s"] = e3_stage_durations(e3)
            row["e3"]["vs_original"] = cmp_e3(b3, e3) if lab != "n1" else None
        if zs:
            row["zs"] = {k: zs.get(k) for k in ("rc", "wall_s", "peak_rss_bytes", "timed_out", "rss_exceeded", "status",
                                                "n_lines", "code_fingerprint")}
            row["zs"]["stages"] = zs.get("stages")
            row["zs"]["vs_original"] = cmp_zs(bz, zs) if lab != "n1" else None
        rows.append(row)

threads = []
for p in sorted(glob.glob(f"{R}/*_t*")):
    name = os.path.basename(p)
    if name.endswith("_t1") and "thr" not in name:
        continue
    rec = load(name)
    if rec and "thr" in name:
        threads.append({"run": name, "threads": rec["threads"], "mode": rec["mode"], "wall_s": rec["wall_s"],
                        "peak_rss_bytes": rec["peak_rss_bytes"], "rc": rec["rc"],
                        "result_stripped_sha256": rec.get("result_stripped_sha256"),
                        "files_sha256": rec.get("files_sha256"), "lines_sha256": rec.get("lines_sha256"),
                        "stages": rec.get("stages") or e3_stage_durations(rec)})
json.dump({"rows": rows, "threads": threads}, open(out_path, "w"), indent=1, default=str)
for r in rows:
    e, z = r.get("e3") or {}, r.get("zs") or {}
    ve, vz = e.get("vs_original") or {}, z.get("vs_original") or {}
    print(f"{r['set']:3} {r['mesh']:13} F={r['faces']/1e6:6.2f}M | E3 {e.get('wall_s')}s {round((e.get('peak_rss_bytes') or 0)/2**30,2)}G "
          f"{e.get('state')}/{e.get('summary')} dec_chg={len(ve.get('decision_changes', [])) if ve else '-'} "
          f"maxd={ve.get('max_abs_delta_by_kind') if ve else '-'} | zs {z.get('wall_s')}s "
          f"{round((z.get('peak_rss_bytes') or 0)/2**30,2)}G {z.get('status')} n={z.get('n_lines')} {vz}")
for t in threads:
    print(t["run"], t["wall_s"], round((t["peak_rss_bytes"] or 0) / 2**30, 2), t.get("result_stripped_sha256") or t.get("lines_sha256"))
