"""One coverage job: one frozen zs lane, blind (no CAD priors), on one staged scan -> one JSON.

Blind configuration = the 30820000_A01 owner demo (freeze seq 34; scratch blind.py): n_bends uncapped (999),
t_mm = the Z2 self-recipe thickness estimate on the scan (z2_self_recipe.run.run_scan diag t_mm, the same value
zs_lane_v1.core.run_z2 returns), max_families 20 (lane_cli.MAX_FAMILIES). Lane functions are lane_cli.run_lane
(zs_lane_v1) / lane_cli.run_lane_v3 (zs_lane_v3 rev3), after PIN.verify == [] and an exact pin-file sha check;
PIN.unpinned_loaded is checked afterwards, as lane_cli does. The scan is staged as input/scan.<ext> (symlink).
Usage: cov_worker.py <lane> <scan_path> <scan_sha256> <out.json> [t_mm_from_previous_lane]
"""
import json, os, pickle, resource, sys, tempfile, time, traceback
from pathlib import Path

lane, scan_src, scan_sha, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
t_given = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] not in ("", "None") else None
PIN_SHA = {"zs_lane_v1": "0087fb2383a412c3cf20b509a5cbfc6628f0d2970b3ca502229067dd952722f1",
           "zs_lane_v3": "711bd079a04286fa89f711314cb4a60f5ae8a1fec85862275d7497263153d5be"}
FP = {"zs_lane_v1": "c712c15c0843d89db98eca5d8b757e75976f884520e5543933acd4fabb64555b",
      "zs_lane_v3": "871f23fa0f46d7f827a31e3c3bd075572dca8adb096569541942c973c1d487b7"}
t0 = time.time()
doc = {"schema": "sherman.zs_coverage.run.v1", "lane": lane, "scan_sha256": scan_sha, "status": "FAILED",
       "config": {"cad_priors": None, "n_bends": 999, "max_families": 20, "t_mm_source": "z2_self_recipe estimate"},
       "errors": {}, "lines": []}


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def write():
    doc["wall_s"] = round(time.time() - t0, 3)
    doc["peak_rss_gib"] = round(rss_gib(), 3)
    doc["n_lines"] = len(doc["lines"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(tmp, out)


try:
    from domains.production.zeroshot import pin as PIN
    from domains.production.zeroshot.lane_cli import run_lane, run_lane_v3
    root = PIN.root_of_package()
    pin_path = root / PIN.pin_rel(lane)
    doc["pin_sha256"] = PIN.sha256_file(pin_path)
    pin = PIN.load_pin(root, lane=lane)
    doc["code_fingerprint"] = pin["code_fingerprint"]["sha256"]
    bad = PIN.verify(root, pin)
    if bad or doc["pin_sha256"] != PIN_SHA[lane] or doc["code_fingerprint"] != FP[lane] or pin["lane"] != lane:
        doc.update(status="PIN_MISMATCH", pin_mismatch=bad[:50])
        write(); sys.exit(1)
    with tempfile.TemporaryDirectory(prefix="zscov_") as td:
        staged = Path(td) / "input" / ("scan." + scan_src.suffix.lower().lstrip("."))
        staged.parent.mkdir()
        os.symlink(scan_src, staged)
        sha = PIN.sha256_file(staged)
        if sha != scan_sha:
            doc.update(status="REFUSED_INPUT", errors={"input": f"sha256 mismatch {sha}"}); write(); sys.exit(1)
        models = {k: pickle.loads((root / m["rel_path"]).read_bytes()) for k, m in PIN.pinned_models(pin).items()}
        if t_given is None:
            from research.zeroshot.z2_self_recipe.run import run_scan
            tz = time.time()
            _p, diag, _s = run_scan(staged)
            del _p, _s
            t = float(diag["t_mm"]); doc["t_estimate_wall_s"] = round(time.time() - tz, 3); doc["t_mm_source"] = "computed"
        else:
            t = t_given; doc["t_mm_source"] = "reused from the zs_lane_v1 job on the same bytes"
        facts = {"n_bends": 999, "t_mm": t, "max_families": 20}
        doc["priors"] = facts; doc["t_mm_scan"] = t
        if lane == "zs_lane_v3":
            res = run_lane_v3(staged, facts, models, pin["lane_params"])
        else:
            res = run_lane(staged, facts, models["zs"])
        doc.update(res)
        me = Path(__file__).resolve().relative_to(root.resolve()).as_posix()
        unp = [u for u in PIN.unpinned_loaded(root, pin) if u != me]
        if unp:
            doc["status"] = "UNPINNED_CODE"
            doc["unpinned_loaded"] = unp[:50]
            doc["unpinned_note"] = "lane result kept for audit; lane_cli would blank it"
except SystemExit:
    raise
except BaseException as exc:  # noqa: BLE001
    doc.update(status="FAILED", lines=[])
    doc["errors"] = {**doc.get("errors", {}), "worker": f"{type(exc).__name__}: {exc}"[:500],
                     "trace": traceback.format_exc()[-1500:]}
write()
print(f"{lane} {doc['status']} lines={len(doc['lines'])} wall={doc['wall_s']:.1f}s rss={doc['peak_rss_gib']:.2f}G", flush=True)
sys.exit(0 if doc["status"] == "OK" else 1)
