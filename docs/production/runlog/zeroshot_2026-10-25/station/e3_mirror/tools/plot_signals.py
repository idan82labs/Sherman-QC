"""plot_signals.py OUT.png: the mirror-identity signals on every pass (primary + 2 resamples) of the forced-check
diagnostic runs. Left: explained gain vs total gain of the mirrored fit over the proper one (genuine natives vs
distinguishable mirrored scans; live-accepted mirrored scans marked). Right: hole score gain for the near-achiral
parts (10839000, 50757000)."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/e3mirror"
C = json.load(open(f"{S}/collected.json"))
fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.6, 1]})
def pts(rows, sel=lambda x: True):
    out = []
    for x in rows:
        if not sel(x): continue
        for p in x.get("e3m") or []:
            if "mirror_total" not in p: continue
            out.append((p["mirror_explained"] - p["best_explained"], p["mirror_total"] - p["best_total"], x, p))
    return out
nat = pts(C["native"], lambda x: (x["diag"] or {}).get("state") == "VALID")
mir_ok = pts(C["mirror"], lambda x: x["distinguishable"] is not False and (x["live"] or {}).get("state") == "VALID")
mir_ref = pts(C["mirror"], lambda x: x["distinguishable"] is not False and (x["live"] or {}).get("state") not in (None, "VALID"))
mir_same = pts(C["mirror"], lambda x: x["distinguishable"] is False)
def sc(a, data, **kw):
    if data: a.scatter([d[0] for d in data], [d[1] for d in data], **kw)
sc(ax, mir_ref, s=14, c="#f4a261", alpha=0.6, label=f"mirrored, refused live ({len(mir_ref)} passes)")
sc(ax, mir_same, s=14, c="#8ab17d", marker="s", alpha=0.6, label=f"mirror of an achiral part = same part ({len(mir_same)})")
sc(ax, nat, s=14, c="#264653", alpha=0.6, label=f"genuine native scans ({len(nat)} passes)")
sc(ax, mir_ok, s=36, c="#d62828", marker="x", label=f"mirrored, ACCEPTED VALID live ({len(mir_ok)} passes)")
ax.axvline(0.02, c="#d62828", ls="--", lw=1); ax.axhline(0.03, c="#d62828", ls=":", lw=1); ax.axhline(0.08, c="grey", ls=":", lw=1)
ax.text(0.022, -0.03, "prototype:\nexplained\ngain 0.02", color="#d62828", fontsize=8)
ax.text(-0.6, 0.035, "prototype: total margin 0.03 (exact stage)", color="#d62828", fontsize=8)
ax.text(-0.6, 0.085, "live mirror_margin 0.08 (only when 'suspicious')", color="grey", fontsize=8)
ax.set_xscale("symlog", linthresh=0.01); ax.set_yscale("symlog", linthresh=0.01)
ax.set_xlabel("explained share: mirrored fit - proper fit"); ax.set_ylabel("total score: mirrored fit - proper fit")
ax.set_title("Forced mirror re-registration on every pass (G1 + A1 + E2E part, e3-1.14.0 diagnostic copy)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
# right: hole gain for near-achiral parts
near = ("pc_71cf0b216f43", "pc_08686dc1d5d4", "pc_cec08dfca9ab", "pc_752e2bd54aad", "pc_777c64c88967")
def hg(p):
    pp, mp = p.get("proper_pool") or [], p.get("mirror_pool") or []
    if not pp or not mp: return None
    return max(a[2] - a[3] for a in mp) - max(a[2] - a[3] for a in pp)
g = [hg(d[3]) for d in pts(C["native"], lambda x: x["case"] in near)]
m = [hg(d[3]) for d in pts(C["mirror"], lambda x: x["case"] in near)]
g = [v for v in g if v is not None]; m = [v for v in m if v is not None]
bins = np.arange(-16.5, 4.5, 1)
bx.hist(g, bins=bins, color="#264653", alpha=0.7, label=f"genuine ({len(g)} passes)")
bx.hist(m, bins=bins, color="#d62828", alpha=0.6, label=f"mirrored ({len(m)} passes)")
bx.axvline(1.5, c="#d62828", ls="--", lw=1); bx.text(1.6, bx.get_ylim()[1] * 0.9, "hole_evidence_min_diff 2", color="#d62828", fontsize=8)
bx.set_xlabel("hole score (confirmed - contradicted), mirrored pool - proper pool")
bx.set_title("Near-achiral parts (10839000, 50757000): holes are the evidence\n(passes where the mirror search missed its pose sit far left)")
bx.legend(fontsize=8); bx.grid(alpha=0.3)
fig.savefig(sys.argv[1], dpi=110, bbox_inches="tight")
print(len(nat), len(mir_ok), len(mir_ref), len(mir_same), len(g), len(m))
