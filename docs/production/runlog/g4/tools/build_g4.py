"""G4 render step (wave 4): E3 packet (final A1 engine e3-1.9.0) + LEGACY packet, per-scan review dirs.

Subcommands (each heavy one is run by the caller under the shared geometry lock):
  jobs           write the job list (no rendering) to W/jobs_e3.json
  test CASE      render one physical case's globals + close-ups to W/test (timing / visual QA)
  legacy-run     re-run the unchanged legacy worker on the frozen LEGACY_LISTS cases into W/legacy_work
  packet-e3      build the E3 packet (RL/g4/packet_e3) + sealed key -> RL/g4/canary_key_SEALED.json
  packet-legacy  build the LEGACY packet (RL/g4/packet_legacy) + key -> RL/g4/canary_key_legacy_SEALED.json
  split ARM      per-scan review dirs RL/g4/review_<arm>/<scan_ref>/ (hard links + scan_manifest.json)

Images are palette re-encoded inside render._save (before every sha256 is taken), so the packet's
manifest and key carry the final file hashes.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
os.chdir(REPO)

S = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
W = S / "g4w"
RL = REPO / "docs/production/runlog"
G4 = RL / "g4"
PHYS = S / "a1_iter7m/phys"          # iter-7 MEASURE (final engine e3-1.9.0, default config)
SYN_RUN = S / "a1_iter7m/g1_regression"  # iter-7 MEASURE G1 regression tier (DEV seeds)
SYN_TIER = S / "a1_iter0/e2_regression"
FINAL_ENGINE = "e3-1.9.0"
FINAL_CONFIG = "cb14d5ca9833617babdf18a2d39f354bccf2c693542e1742fa9ab6619279469f"

# Synthetic sample, chosen by SCENARIO only (no engine output looked at for the choice):
# per family the COMPLETE_NOMINAL -000, the STAGE_1 -003 and the STAGE_K -005 (L_BRACKET has no
# STAGE_K: its -005 is PARTIAL_BEND), plus stage-0 blanks, a missing bend, an order swap and the
# symmetric U channel.
SYN_CASES = [f"E2-DEV-{fam}-{n}" for fam in ("LB", "UC", "ZS", "HT", "CR", "TD", "PN") for n in ("000", "003", "005")]
SYN_CASES += ["E2-DEV-LB-002", "E2-DEV-HT-002", "E2-DEV-TD-006", "E2-DEV-UC-006", "E2-DEV-UCS-000"]


def _compact_save_patch() -> None:
    from domains.production.engine.eval import render
    from domains.production.engine.eval.freeze.common import compact_png

    orig = render._save

    def _save(fig, path, meta):
        orig(fig, path, meta)
        import matplotlib.pyplot as plt

        plt.close(fig)
        return compact_png(path)

    render._save = _save


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def physical_jobs():
    from domains.production.engine.eval.g4 import RenderJob

    doc = read_json(RL / "freeze/cases_A1.json")
    table = read_json(RL / "freeze/g3_truth_table.json")
    not_formed = {}
    for r in table["rows"]:
        st = r.get("final") or r.get("state") or (r.get("final_state"))
        if isinstance(st, dict):
            st = st.get("state")
        not_formed.setdefault(r["case_ref"], []).append(st == "NOT_FORMED")
    jobs, skipped = [], []
    for c in doc["cases"]:
        eng = PHYS / "physical/E3" / c["case_ref"] / "engine"
        comp = PHYS / "compiled/dev" / c["recipe_sku"] / "compiled_part.json"
        rec = RL / "freeze/recipes" / f"{c['recipe_sku']}.recipe.json"
        if not (eng / "result.json").is_file() or not comp.is_file():
            skipped.append({"case_ref": c["case_ref"], "why": "no E3 output (recipe compile BLOCKED)"})
            continue
        res = read_json(eng / "result.json")
        assert res["engine"]["engine_version"] == FINAL_ENGINE, c["case_ref"]
        assert res["engine"]["config_sha256"] == FINAL_CONFIG, c["case_ref"]
        nf = not_formed.get(c["case_ref"])
        incomplete = None if nf is None else any(nf)
        jobs.append(RenderJob(engine_dir=eng, compiled_json=comp, recipe_json=rec, case_ref=c["case_ref"],
                              source="E3", population="PHYSICAL_DEV", incomplete=incomplete))
    return jobs, skipped


def synthetic_jobs():
    from domains.production.engine.eval.g4 import RenderJob

    jobs = []
    for cid in SYN_CASES:
        r = read_json(SYN_RUN / "cases" / f"{cid}.json")
        eng = Path(r["engine_output"]["dir"])
        comp = Path(r["compile"]["compiled_dir"]) / "compiled_part.json"
        rec = eng.parent / "input" / "recipe.json"
        res = read_json(eng / "result.json")
        assert res["engine"]["engine_version"] == FINAL_ENGINE, cid
        assert res["engine"]["config_sha256"] == FINAL_CONFIG, cid
        truth = read_json(SYN_TIER / cid / "truth.json")
        planned = len(truth["expected"]["bend_states"])
        incomplete = truth["stage"]["formed_count"] < planned
        jobs.append(RenderJob(engine_dir=eng, compiled_json=comp, recipe_json=rec, case_ref=cid, source="E3",
                              population="SYNTHETIC", incomplete=incomplete))
    return jobs


def cmd_jobs():
    pj, skipped = physical_jobs()
    sj = synthetic_jobs()
    out = {"physical": [{"case_ref": j.case_ref, "incomplete": j.incomplete} for j in pj], "skipped": skipped,
           "synthetic": [{"case_ref": j.case_ref, "incomplete": j.incomplete} for j in sj]}
    (W / "jobs_e3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"physical": len(pj), "synthetic": len(sj),
                      "synthetic_incomplete": sum(1 for j in sj if j.incomplete), "skipped": skipped}))


def cmd_test(case_ref):
    from domains.production.engine.eval.render import plan_closeups, render_closeup, render_global_views, \
        scene_from_engine_output

    _compact_save_patch()
    pj, _ = physical_jobs()
    j = next(x for x in pj if x.case_ref == case_ref)
    out = W / "test" / case_ref
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    scene = scene_from_engine_output(j.engine_dir, j.compiled_json, j.recipe_json, title="test")
    t1 = time.time()
    recs = render_global_views(scene, out, show_values=False)
    t2 = time.time()
    for plan in plan_closeups(scene):
        recs.append(render_closeup(scene, plan, out / f"closeup_{plan.bend_uid}_{plan.tile_index}.png"))
    t3 = time.time()
    print(json.dumps({"scene_s": round(t1 - t0, 1), "globals_s": round(t2 - t1, 1), "closeups_s": round(t3 - t2, 1),
                      "n": len(recs), "points": None if scene.points is None else len(scene.points)}))


def legacy_cases():
    doc = read_json(RL / "freeze/cases_A1.json")
    from domains.production.engine.eval.freeze.prefreeze import LEGACY_LISTS

    return [c for c in doc["cases"] if any(l.split(":")[0] in LEGACY_LISTS for l in c["lists"])]


def cmd_legacy_run(only=None):
    from domains.production.engine.eval.freeze.legacy import run_legacy_case
    from domains.production.engine.eval.freeze.prefreeze import cad_path_of, scan_path_of

    work = W / "legacy_work"
    work.mkdir(parents=True, exist_ok=True)
    frozen = {c["case_ref"]: c for c in read_json(RL / "legacy_control/SUMMARY.json")["cases"]}
    rows = []
    for c in legacy_cases():
        if only and c["case_ref"] not in only:
            continue
        run = run_legacy_case(c, scan_path_of(c), cad_path_of(c), work)
        (work / "legacy" / c["case_ref"] / "run.json").write_text(json.dumps(run, indent=1, default=str))
        f = frozen.get(c["case_ref"], {})
        rows.append({"case_ref": c["case_ref"], "status": run["status"], "frozen_status": f.get("status"),
                     "wall_s": run["wall_s"], "peak_rss_gib": round((run["peak_rss_bytes"] or 0) / 1024**3, 2)})
        print(json.dumps(rows[-1]), flush=True)
    prev = read_json(W / "legacy_runs.json") if (W / "legacy_runs.json").is_file() and only else []
    merged = {r["case_ref"]: r for r in prev}
    merged.update({r["case_ref"]: r for r in rows})
    (W / "legacy_runs.json").write_text(json.dumps(sorted(merged.values(), key=lambda r: r["case_ref"]), indent=1))


def legacy_jobs():
    from domains.production.engine.eval.engine_output import load_compiled
    from domains.production.engine.eval.freeze.legacy import legacy_render_scene, parse_legacy_report
    from domains.production.engine.eval.g4 import RenderJob

    table = read_json(RL / "freeze/g3_truth_table.json")
    nf = {}
    for r in table["rows"]:
        st = r.get("final") or r.get("state")
        if isinstance(st, dict):
            st = st.get("state")
        nf.setdefault(r["case_ref"], []).append(st == "NOT_FORMED")
    jobs, rows = [], []
    for c in legacy_cases():
        cdir = PHYS / "compiled/dev" / c["recipe_sku"]
        runp = W / "legacy_work/legacy" / c["case_ref"] / "run.json"
        if not (cdir / "compiled_part.json").is_file():
            rows.append({"case_ref": c["case_ref"], "rendered": False, "why": "recipe compile BLOCKED"})
            continue
        run = read_json(runp)
        compiled = load_compiled(cdir / "compiled_part.json")
        if compiled.status != "COMPILED":
            rows.append({"case_ref": c["case_ref"], "rendered": False, "why": f"recipe compile {compiled.status}"})
            continue
        parsed = parse_legacy_report(run, compiled, cdir)
        frozen = read_json(RL / "legacy_control" / c["case_ref"] / "legacy_case.json")
        same = [(b["bend_uid"], b["line"]["measured"]) for b in parsed["bends"]] == \
               [(b["bend_uid"], b["line"]["measured"]) for b in frozen["bends"]]
        scene = legacy_render_scene(parsed, run, compiled, cdir)
        jobs.append(RenderJob(engine_dir=None, compiled_json=None, recipe_json=None, case_ref=c["case_ref"],
                              source="LEGACY", population="PHYSICAL_DEV",
                              incomplete=None if c["case_ref"] not in nf else any(nf[c["case_ref"]]), scene=scene))
        rows.append({"case_ref": c["case_ref"], "rendered": True, "status": run["status"],
                     "validity": scene.validity_state, "lines_measured": parsed["counts"].get("line_measured"),
                     "measured_flags_match_frozen_E4b": same})
    return jobs, rows


def _seal(sealed_dir: Path, dst: Path) -> str:
    """Move the builder's key to its single sealed location (mode 0400) and remove the staging dir."""
    import hashlib

    src = sealed_dir / "canary_key.json"
    data = src.read_bytes()
    if dst.exists():
        os.chmod(dst, 0o600)
        dst.unlink()
    dst.write_bytes(data)
    os.chmod(dst, 0o400)
    os.chmod(src, 0o600)
    shutil.rmtree(sealed_dir)
    return hashlib.sha256(data).hexdigest()


def _build(jobs, packet: Path, key_dst: Path, tag: str):
    from domains.production.engine.eval.g4 import build_review_packet

    _compact_save_patch()
    if packet.exists():
        shutil.rmtree(packet)
    sealed = W / f"sealed_tmp_{tag}"
    if sealed.exists():
        for p in sealed.rglob("*"):
            os.chmod(p, 0o700 if p.is_dir() else 0o600)
        shutil.rmtree(sealed)
    t0 = time.time()
    res = build_review_packet(jobs, packet, sealed)
    sha = _seal(sealed, key_dst)
    assert sha == res["canary_key_sha256"]
    res.update({"wall_s": round(time.time() - t0, 1), "key": str(key_dst)})
    res.pop("sealed", None)
    # the canary count is withheld from anything a reviewer can open
    (W / f"build_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "canaries"}))


def cmd_packet_e3():
    pj, skipped = physical_jobs()
    _build(pj + synthetic_jobs(), G4 / "packet_e3", G4 / "canary_key_SEALED.json", "e3")


def cmd_packet_legacy():
    jobs, rows = legacy_jobs()
    (W / "legacy_jobs.json").write_text(json.dumps(rows, indent=1))
    _build(jobs, G4 / "packet_legacy", G4 / "canary_key_legacy_SEALED.json", "legacy")


def cmd_split(arm):
    packet = G4 / f"packet_{arm}"
    man = read_json(packet / "review_manifest.json")
    out_root = G4 / f"review_{arm}"
    if out_root.exists():
        shutil.rmtree(out_root)
    by_scan = {}
    for e in man["images"]:
        by_scan.setdefault(e["scan_ref"], []).append(e)
    items = []
    for ref, entries in sorted(by_scan.items()):
        d = out_root / ref
        d.mkdir(parents=True)
        for e in entries:
            os.link(packet / e["file"], d / Path(e["file"]).name)
        sub = {"schema": "sherman_g4_scan_review_v1", "scan_ref": ref, "packet": str(packet),
               "packet_manifest_sha256_commitment": man["canary_key_sha256"], "protocol": man["protocol"],
               "instructions": man["instructions"], "lands_rule": man["lands_rule"],
               "images": [{**e, "file": Path(e["file"]).name} for e in entries]}
        (d / "scan_manifest.json").write_text(json.dumps(sub, indent=1, ensure_ascii=False))
        uids = list(dict.fromkeys(e["subject"]["bend_uid"] for e in entries if e.get("subject")))
        items.append({"scan_id": ref, "image_dir": str(d), "bend_uids": uids,
                      "closeups": sum(1 for e in entries if e["kind"] == "CLOSEUP"),
                      "globals": sum(1 for e in entries if e["kind"] != "CLOSEUP")})
    (W / f"items_{arm}.json").write_text(json.dumps(items, indent=1))
    print(json.dumps({"scans": len(items), "images": len(man["images"])}))


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"jobs": cmd_jobs, "test": lambda: cmd_test(rest[0]), "legacy-run": lambda: cmd_legacy_run(rest or None),
     "packet-e3": cmd_packet_e3, "packet-legacy": cmd_packet_legacy, "split": lambda: cmd_split(rest[0])}[cmd]()
