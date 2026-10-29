"""Compile-rate sweep over every authorized unique STEP (no labels). usage: compile_all.py OUT_DIR [sha_prefix...]
Per unique STEP sha: E1-DEV-POLICY-1 build (dev_candidates + dev_recipe) then the E1 compile CLI (300 s, 4 GiB).
Exclusions: holdout / EHO4 / N001-N016 / sealed-truth / rhino-preview paths."""
import hashlib, json, re, sys, time
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
from domains.production.engine.eval.freeze import recipes
from domains.production.engine.eval.freeze.common import dev_sku_for, EHO4_PARTS
from domains.production.engine.eval.runner import RunnerConfig, engine_env
from domains.production.engine.compile import dev_recipe as dr
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
only = sys.argv[2:]
roots = [REPO.parent / "source-assets/02_CAD_QC_3D", REPO / "data/bend_corpus"]
bad = re.compile(r"(?i)holdout|sealed_truth|eho4|one_shot|oneshot|rhino.preview|(^|[^a-z0-9])n0(0[1-9]|1[0-6])([^0-9]|$)")
eho = re.compile(r"(?<!\d)(" + "|".join(EHO4_PARTS) + r")(?!\d)")
seen, rows = {}, []
for r in roots:
    for p in sorted(r.rglob("*")):
        if p.suffix.lower() not in (".stp", ".step") or not p.is_file():
            continue
        if bad.search(str(p)) or eho.search(str(p)):
            rows.append({"path": str(p), "status": "EXCLUDED_CUSTODY"}); continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha in seen:
            continue
        seen[sha] = p
stats = []
env = engine_env(RunnerConfig(tier_dir=OUT, out_dir=OUT))
for sha, p in seen.items():
    if only and not any(sha.startswith(o) for o in only):
        continue
    sku = dev_sku_for(sha); wd = OUT / sku; wd.mkdir(exist_ok=True)
    rec = {"sha": sha[:12], "sku": sku, "path": str(p)}
    t0 = time.time()
    try:
        cands, rep = dr.dev_candidates(p)
    except Exception as exc:
        cands, rep = [], {"error": f"{type(exc).__name__}: {exc}"[:300]}
    rec["candidates"] = len(cands); rec["merge_error"] = rep.get("merge_error") or rep.get("error")
    rec["counts"] = rep.get("counts")
    rec["excluded_by_rule"] = [e.get("rule") for e in rep.get("excluded_by_unsupported_rule") or []]
    rec["added_from_e1_graph"] = rep.get("added_from_e1_graph")
    if not cands:
        rec.update(status="NO_RECIPE", reason="NO_CANDIDATE_BENDS" + (":ALL_UNSUPPORTED_" + "+".join(sorted(set(rec["excluded_by_rule"]))) if rec["excluded_by_rule"] else ""))
    else:
        try:
            recp = dr.dev_recipe(p, cands, sku=sku, inspection_stage="FINAL")
        except Exception as exc:
            rec.update(status="NO_RECIPE", reason=f"RECIPE_BUILD_REFUSED: {type(exc).__name__}: {exc}"[:300])
        else:
            rp = wd / "dev.recipe.json"; recipes._write_recipe(rp, recp)
            c = recipes.compile_recipe(rp, p, wd / "compiled", env=env)
            rec.update(status=c.get("status"), reason=c.get("reason"), detail=c.get("detail"), bends=c.get("bends"),
                       compile_wall_s=c.get("wall_s"), peak_rss=c.get("peak_rss_bytes"))
    rec["wall_s"] = round(time.time() - t0, 1)
    stats.append(rec); print(json.dumps({k: rec.get(k) for k in ("sha", "path", "status", "reason", "bends", "wall_s")}), flush=True)
n = len(stats); ok = sum(r["status"] == "COMPILED" for r in stats)
summary = {"unique_steps": n, "compiled": ok, "rate": ok / n if n else None, "excluded_paths": [r for r in rows], "rows": stats}
(OUT / "compile_all.json").write_text(json.dumps(summary, indent=1, default=str))
print(f"COMPILED {ok}/{n}")
