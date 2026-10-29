"""Collect the zs_rust port's evidence into v3_rust/results.json and print the markdown tables used in PORT.md.
Reads only files written by the other tools (bench outputs, equivalence / speed json); nothing is run or scored.
usage: .venv/bin/python -B docs/.../v3_rust/tools/make_results.py BENCH_ROOT
  BENCH_ROOT holds L_dev2, L_dev3 (DEV, revision 2 / 3 configs) and L_n1_2, L_n1_3 (N1 timing only), t1/t4/t8/t16."""
import hashlib, json, os, statistics as st, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[6]
P = REPO / "docs/production/runlog/zeroshot_2026-10-25"
V = P / "v3_rust"
B = Path(sys.argv[1])
TH = (1, 4, 8, 16)


def q(a, p):
    a = sorted(a)
    k = (len(a) - 1) * p
    f = int(k)
    c = min(f + 1, len(a) - 1)
    return a[f] + (a[c] - a[f]) * (k - f)


def load(p):
    return json.loads(Path(p).read_text())


def strip(j):
    j = dict(j)
    j.pop("created_at", None)
    c = dict(j["conversion"])
    c.pop("stages", None)
    if "rust" in c:
        c["rust"] = {k: v for k, v in c["rust"].items() if k != "threads"}
    j["conversion"] = c
    return json.dumps(j)


def bench(root, ref):
    ids = sorted(os.listdir(root / "t1"))
    out = {"scans": len(ids)}
    walls = {t: [] for t in TH}
    rss = {t: [] for t in TH}
    det = byte = 0
    fallback = []
    eq = {t: 0 for t in TH}
    for s in ids:
        recs = {t: load(root / f"t{t}/{s}/run_record.json") for t in TH}
        evs = {t: load(root / f"t{t}/{s}/eval_lines.json") for t in TH}
        det += len({strip(evs[t]) for t in TH}) == 1
        byte += len({json.dumps(evs[t]["owned_bend_regions"]) for t in TH}) == 1
        if ref is not None:
            r = load(ref / s / "eval_lines.json")["owned_bend_regions"]
            for t in TH:
                eq[t] += json.dumps(r) == json.dumps(evs[t]["owned_bend_regions"])
        if "rust_fallback" in recs[1]:
            fallback.append(s)
            continue
        for t in TH:
            walls[t].append(recs[t]["wall_s"])
            rss[t].append(recs[t]["peak_rss_gib"])
    out["point_cloud_fallback"] = fallback
    out["deterministic_all_fields_1_4_8_16"] = det
    out["regions_byte_identical_1_4_8_16"] = byte
    if ref is not None:
        out["owned_bend_regions_identical_to_python"] = {f"t{t}": eq[t] for t in TH}
    out["timing_rust_scans"] = {f"t{t}": {"n": len(walls[t]), "p50": round(q(walls[t], .5), 3), "p95": round(q(walls[t], .95), 3),
                                          "max": round(max(walls[t]), 3), "rss_max_gib": round(max(rss[t]), 3)} for t in TH}
    return out, walls


res = {"schema": "sherman.zs_rust.port_results.v1"}
cfg2 = load(REPO / "research/zeroshot/zs_lane_v3_rs/config_v3_rs.json")
cfg3 = load(REPO / "research/zeroshot/zs_lane_v3_rs/config_v3r3_rs.json")
binp = REPO / "research/zeroshot/zs_rust/target/release/zs_rust"
res["binary"] = {"sha256": hashlib.sha256(binp.read_bytes()).hexdigest(), "bytes": binp.stat().st_size,
                 "cargo_lock_sha256": cfg2["rust"]["cargo_lock_sha256"], "build": cfg2["rust"]["build"]}
res["configs"] = {"rev2": {"path": "research/zeroshot/zs_lane_v3_rs/config_v3_rs.json", "code_fingerprint": cfg2["code_fingerprint"]["sha256"],
                           "base_v3_code_fingerprint": cfg2["base_v3_code_fingerprint"], "base_v3_config_sha256": cfg2["base_v3_config_sha256"]},
                  "rev3": {"path": "research/zeroshot/zs_lane_v3_rs/config_v3r3_rs.json", "code_fingerprint": cfg3["code_fingerprint"]["sha256"],
                           "base_v3_code_fingerprint": cfg3["base_v3_code_fingerprint"], "base_v3_config_sha256": cfg3["base_v3_config_sha256"]}}
d2, w2 = bench(B / "L_dev2", P / "v3/rev2/dev_cli/zs")
d3, w3 = bench(B / "L_dev3", P / "v3/rev3/dev_cli/zs")
res["dev_rev2"] = d2
res["dev_rev3"] = d3
# Python revision 2 / 3 DEV CLI on the same (mesh) scans; revision 2's four host-stalled scans excluded
for name, ref in (("python_rev2_dev_cli", P / "v3/rev2/dev_cli/zs"), ("python_rev3_dev_cli", P / "v3/rev3/dev_cli/zs")):
    ids = [s for s in sorted(os.listdir(B / "L_dev2/t1")) if "rust_fallback" not in load(B / f"L_dev2/t1/{s}/run_record.json")]
    pw = [load(ref / s / "run_record.json")["wall_s"] for s in ids]
    keep = [w for w in pw if w < 100]
    res[name] = {"n": len(keep), "excluded_host_stall": len(pw) - len(keep), "p50": round(q(keep, .5), 2), "p95": round(q(keep, .95), 2),
                 "max": round(max(keep), 2)}
    rw = d2 if name.endswith("rev2_dev_cli") else d3
    walls = w2 if name.endswith("rev2_dev_cli") else w3
    sp = {}
    for t in TH:
        r = [p / r for p, r in zip(pw, walls[t]) if p < 100]
        sp[f"t{t}"] = {"median_per_scan": round(st.median(r), 2), "sum_ratio": round(sum(p for p in pw if p < 100) / sum(r_ for p, r_ in zip(pw, walls[t]) if p < 100), 2)}
    res[name]["rust_speedup"] = sp
n1 = {}
for rev, root in (("rev2", B / "L_n1_2"), ("rev3", B / "L_n1_3")):
    for s in sorted(os.listdir(root / "t1")):
        row = n1.setdefault(s, {})
        evs = [json.dumps(load(root / f"t{t}/{s}/eval_lines.json")["owned_bend_regions"]) for t in TH]
        row[rev] = {f"t{t}": {"wall_s": load(root / f"t{t}/{s}/run_record.json")["wall_s"],
                              "rss_gib": round(load(root / f"t{t}/{s}/run_record.json")["peak_rss_gib"], 3)} for t in TH}
        row[rev]["identical_across_threads"] = len(set(evs)) == 1
res["n1_timing_only"] = n1
res["equivalence"] = {k: load(V / f"equivalence/{k}.json") if (V / f"equivalence/{k}.json").is_file() else None
                      for k in ("dump_stage_cmp",)}
for k in ("dense_rev2_lines", "dense_rev3_lines", "r3_budget_mesh"):
    d = load(V / f"equivalence/{k}.json")
    if k == "r3_budget_mesh":
        res[k] = {"meshes": len(d), "all_equal": sum(v["V_equal"] and v["F_equal"] and v["info_equal"] for v in d.values())}
    else:
        res[k] = {"meshes": len(d), "identical": sum(v["identical"] for v in d.values()),
                  "budget_info_identical": sum(v.get("budget_info_identical", False) for v in d.values())}
mr = load(V / "equivalence/mesh_read.json")
res["mesh_read"] = {"scans": len(mr), "ok": sum(v.get("ok", False) for v in mr.values())}
oof = load(V / "oof/REV2_OOF_SUMMARY.json")
res["oof_rev2_dump"] = {k: oof.get(k) for k in ("rev2_fixed_v3_fold_models", "rev2", "dump_identity")}
dense = load(V / "speed/dense_rust.json")
d2e = load(V / "equivalence/dense_rev2_lines.json")
d3e = load(V / "equivalence/dense_rev3_lines.json")
rows = {}
for k, v in dense.items():
    m, mode = k.split("|")[0], k.split("|")[1]
    if mode.startswith("python"):
        continue
    t = k.split("|")[2]
    rows.setdefault(m, {}).setdefault(mode, {})[t] = {"wall_s": v.get("wall_s"), "rss_gib": v.get("peak_rss_gib"), "n_lines": v.get("n_lines"),
                                                      "lines_sha": v.get("lines_sha")}
for m in rows:
    for mode in rows[m]:
        rows[m][mode]["identical_across_threads"] = len({x["lines_sha"] for x in rows[m][mode].values() if isinstance(x, dict)}) == 1
    if m in d2e:
        rows[m]["python_rev2"] = {"wall_s": d2e[m]["python_wall_s"], "rss_gib": d2e[m]["python_rss_gib"], "faces_in": d2e[m]["budget"].get("faces_in")}
    if m in d3e:
        rows[m]["python_rev3"] = {"wall_s": d3e[m]["python_wall_s"], "rss_gib": d3e[m]["python_rss_gib"]}
res["dense"] = rows
res["stress"] = {f: load(V / f"speed/{f}.json") for f in ("stress_sc_3b3e8c9f04e3_x4", "stress_sc_ead69b8181a8_x4")}
(V / "results.json").write_text(json.dumps(res, indent=1) + "\n")

# ---- markdown
print("DEV (rust scans)")
for nm, d in (("rev2", d2), ("rev3", d3)):
    for t in TH:
        x = d["timing_rust_scans"][f"t{t}"]
        print(f"| Rust {nm}, {t} thr | {x['n']} | {x['p50']:.2f} s | {x['p95']:.2f} s | {x['max']:.2f} s | {x['rss_max_gib']:.2f} GiB |")
for nm in ("python_rev2_dev_cli", "python_rev3_dev_cli"):
    print(nm, res[nm])
print("\nDENSE")
print("| Mesh | Faces in | Python rev2 | Rust rev2 1 / 16 thr | Python rev3 | Rust rev3 1 / 4 / 8 / 16 thr | Rust rev3 RSS | budget off 1 / 16 thr, RSS |")
order = sorted(rows, key=lambda m: rows[m].get("python_rev2", {}).get("faces_in") or 0)
for m in order:
    r = rows[m]
    g = lambda mode, t: r.get(mode, {}).get(f"t{t}", {}).get("wall_s")  # noqa: E731
    gr = lambda mode, t: r.get(mode, {}).get(f"t{t}", {}).get("rss_gib")  # noqa: E731
    fi = r.get("python_rev2", {}).get("faces_in")
    print(f"| {m} | {fi/1e6:.1f} M | {r['python_rev2']['wall_s']:.1f} s, {r['python_rev2']['rss_gib']:.2f} GiB | {g('rev2',1)} / {g('rev2',16)} s, {gr('rev2',16)} GiB | "
          f"{r['python_rev3']['wall_s']:.1f} s, {r['python_rev3']['rss_gib']:.2f} GiB | {g('rev3',1)} / {g('rev3',4)} / {g('rev3',8)} / {g('rev3',16)} s | {gr('rev3',16)} GiB | "
          f"{g('off',1)} / {g('off',16)} s, {gr('off',16)} GiB |")
print("\nN1")
for s, row in n1.items():
    print(s, {rev: [row[rev][f't{t}']['wall_s'] for t in TH] for rev in ('rev2', 'rev3')}, {rev: row[rev]['t16']['rss_gib'] for rev in ('rev2', 'rev3')})
print("\nother", json.dumps({k: res[k] for k in ("r3_budget_mesh", "dense_rev2_lines", "dense_rev3_lines", "mesh_read", "binary")}))
