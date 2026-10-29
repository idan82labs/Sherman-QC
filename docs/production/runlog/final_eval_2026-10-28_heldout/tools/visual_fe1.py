"""FE1 visual check: one iso overview PNG per scored scan for the primary arm (zs_lane_v3 rev3), gold vs engine lines
(TP gold green with matched engine dashed dark green, FN orange, FP red), titled with the gid only (no part names).
Uses the N1 rendering helpers (zeroshot_2026-10-25/tools/newscans_n1_visuals.py: decimate/overview) and
research/bench_labeling render/scanio. Reads results.json (already written) for TP/FP/FN.
Writes visual/<gid>.png and visual/index.html. Heavy -> run through the slot:
  sherman_slot.sh .venv/bin/python -B <this> [arm]
"""
import html, importlib.util, json, sys
from pathlib import Path

REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path.insert(0, str(REPO))
D = REPO / "docs/production/runlog/final_eval_2026-10-28_heldout"
spec = importlib.util.spec_from_file_location("n1vis", REPO / "docs/production/runlog/zeroshot_2026-10-25/tools/newscans_n1_visuals.py")
NV = importlib.util.module_from_spec(spec); spec.loader.exec_module(NV)
from research.bench_labeling.scanio import load_scan  # noqa: E402
from research.bench_labeling import render as RD  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "zs_lane_v3"


def main():
    res = json.loads((D / "results.json").read_text())
    rows = {r["gid"]: r for r in res["arms"][ARM]["per_scan"]}
    man = json.loads((D / "adjudication/GOLD_ROOT_MANIFEST.json").read_text())["labels"]
    out = D / "visual"; out.mkdir(exist_ok=True)
    cards = []
    for gid, row in sorted(rows.items()):
        gold = json.loads((REPO / man[gid]["label_path"]).read_text())["bends"]
        run = json.loads((D / "runs" / ARM / f"{gid}.json").read_text())
        pred = [{"line_start": l["start_mm"], "line_end": l["end_mm"]} for l in run["lines"]] if run.get("status") == "OK" else []
        rec = {"matches": [{"gold_id": m["gold_id"], "prediction_id": m["pred_id"]} for m in row["matches"]]}
        scan = load_scan(str(REPO / f"docs/production/runlog/fe1_stage_2026-10-28/scans/{gid}.ply"))
        pts, nrm, vox = NV.decimate(scan.points, scan.normals)
        frames = RD.camera_frames(pts)
        f1 = "n/a (zero-bend)" if row["f1"] is None else f"{row['f1']:.3f}"
        head = (f"{gid} | {ARM} | gold {row['gold']} | engine {row['pred']} | TP {row['tp']} FP {row['fp']} FN {row['fn']} "
                f"| F1 {f1}{'' if row['status'] == 'OK' else ' | ENGINE FAILED: ' + str(row['status'])}")
        p = out / f"{gid}.png"
        NV.overview({"name": gid}, pts, nrm, vox, frames["views"]["iso"], "iso", gold, pred, rec, p, head)
        NV.shrink(p)
        cards.append((gid, row, f1))
        print(gid, "ok", flush=True)
    a = res["arms"][ARM]["primary"]
    body = "".join(
        f'<figure><img src="{g}.png" alt="{g}" loading="lazy"><figcaption><b>{g}</b> &middot; {html.escape(r["subset"])} &middot; '
        f'gold {r["gold"]}, engine {r["pred"]} &middot; TP {r["tp"]} / FP {r["fp"]} / FN {r["fn"]} &middot; F1 {f1}'
        f'{"" if r["status"] == "OK" else " &middot; <b>engine failed</b>"}</figcaption></figure>' for g, r, f1 in cards)
    (out / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FE1 visual check</title><style>
:root{{--bg:#fff;--fg:#111;--mut:#555}}@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mut:#aaa}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1500px;padding:16px}}
figure{{margin:0 0 28px}}img{{width:100%;height:auto;border:1px solid #8884;background:#fff}}figcaption{{color:var(--mut)}}
.k span{{display:inline-block;width:14px;height:6px;margin:0 6px 2px 14px}}</style></head><body>
<h1>FE1 held-out run: gold vs engine lines</h1>
<p>Engine {ARM}, blind no-CAD, one run per scan. Micro line-F1 {a['micro_f1']} (95 % CI {a['micro_ci95']}), family-macro {a['family_macro_f1']} (95 % CI {a['macro_ci95']}); TP {a['tp']} / FP {a['fp']} / FN {a['fn']}.</p>
<p class="k"><span style="background:#1a9850"></span>gold line found (engine line dashed dark green)<span style="background:#fd8d3c"></span>gold line missed (FN)<span style="background:#d73027"></span>engine line with no gold (FP)</p>
{body}</body></html>""")


if __name__ == "__main__":
    main()
