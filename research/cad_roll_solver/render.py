"""Visual check of the roll solver: CAD mesh + roll axis, run-out lines, fitted arcs at 3 stations, E1 bend lines.

usage: .venv/bin/python -B -m research.cad_roll_solver.render --pkl DIR/NAME.render.pkl --out PNG_DIR

Per part: ``<name>_iso.png`` (all rolls) and, per roll, ``<name>_<roll>_end.png`` (looking along the roll axis).
Orthographic projection, painter's order for the shaded mesh, and a z-buffer for the overlays: an overlay
segment hidden behind material is drawn thin and dotted, a visible one solid. Matplotlib/Agg, CPU only.

Styles: roll axis magenta dash-dot; run-out lines orange (outer skin solid, inner skin dashed), in the
bend-line style of the E3 overlays (solid on material); fitted arcs at the 3 stations cyan; E1 bend lines
(OML virtual sharp) red; junction-bend tangent lines green.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import io
import math
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

C_AXIS, C_RUNOUT, C_ARC, C_E1, C_JB = "#C026D3", "#EA580C", "#0891B2", "#DC2626", "#16A34A"
W, H, DPI = 1200, 900, 100


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _camera(view_dir, up_hint):
    f = _unit(view_dir)  # from the camera into the scene
    up = np.asarray(up_hint, float) - float(np.dot(up_hint, f)) * f
    if np.linalg.norm(up) < 1e-6:
        up = np.array([0, 0, 1.0]) - f[2] * f
        if np.linalg.norm(up) < 1e-6:
            up = np.array([0, 1.0, 0]) - f[1] * f
    up = _unit(up)
    right = np.cross(f, up)
    return right, up, f


class _ZBuf:
    def __init__(self, xy, depth, tris, lo, hi, n=700):
        self.lo, self.hi, self.n = lo, hi, n
        self.scale = (n - 1) / max(hi - lo)
        self.z = np.full((n, n), np.inf)
        P = (xy - lo) * self.scale
        for t in tris:
            a, b, c = P[t]
            x0, x1 = int(max(0, math.floor(min(a[0], b[0], c[0])))), int(min(n - 1, math.ceil(max(a[0], b[0], c[0]))))
            y0, y1 = int(max(0, math.floor(min(a[1], b[1], c[1])))), int(min(n - 1, math.ceil(max(a[1], b[1], c[1]))))
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1), indexing="ij")
            d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(d) < 1e-12:
                continue
            w1 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / d
            w2 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / d
            w3 = 1 - w1 - w2
            m = (w1 >= -0.02) & (w2 >= -0.02) & (w3 >= -0.02)
            if not m.any():
                continue
            zz = w1 * depth[t[0]] + w2 * depth[t[1]] + w3 * depth[t[2]]
            sub = self.z[x0:x1 + 1, y0:y1 + 1]
            np.minimum(sub, np.where(m, zz, np.inf), out=sub)

    def visible(self, xy, depth, eps):
        """Visible when within eps + 2 pixels of depth of the front surface at the point's pixel (the slack
        keeps grazing skins visible) ; any of the 3x3 pixels passing counts (a line on a silhouette edge)."""
        P = np.clip(np.round((xy - self.lo) * self.scale).astype(int), 0, self.n - 1)
        tol = eps + 2.0 / self.scale
        ok = np.zeros(len(P), bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                q = np.clip(P + [dx, dy], 0, self.n - 1)
                ok |= depth <= self.z[q[:, 0], q[:, 1]] + tol
        return ok


def _polyline(ax, pts3, cam, zb, colour, lw, style="-", label=None, eps=0.6, hidden=True):
    right, up, f = cam
    pts3 = np.asarray(pts3, float)
    if len(pts3) == 2:
        pts3 = pts3[0] + np.linspace(0, 1, 60)[:, None] * (pts3[1] - pts3[0])
    xy = np.stack([pts3 @ right, pts3 @ up], 1)
    dep = pts3 @ f
    vis = zb.visible(xy, dep, eps) if hidden else np.ones(len(xy), bool)
    i = 0
    while i < len(xy) - 1:
        j = i
        while j < len(xy) - 1 and vis[j + 1] == vis[i]:
            j += 1
        seg = xy[i:j + 2] if j + 2 <= len(xy) else xy[i:j + 1]
        if vis[i]:
            ax.plot(seg[:, 0], seg[:, 1], linestyle=style, color=colour, lw=lw, zorder=6, solid_capstyle="round")
        else:
            ax.plot(seg[:, 0], seg[:, 1], linestyle=":", color=colour, lw=max(0.8, lw * 0.45), zorder=5, alpha=0.8)
        i = j + 1
    if label:
        k = int(np.argmax(vis)) if vis.any() else 0
        ax.text(xy[k, 0], xy[k, 1], " " + label, color=colour, fontsize=9, fontweight="bold", zorder=9,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 0.5})


def _arc_pts(F, z, r, n=90):
    s, w = F["cover"]
    a = s + np.linspace(0, w, n)
    c = F["origin"] + z * F["axis"]
    return c + r * (np.cos(a)[:, None] * F["e1"] + np.sin(a)[:, None] * F["e2"])


def render_view(R, cam, title, out_png, end_on=None):
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    V, T, doc = R["vertices"].astype(float), R["triangles"], R["doc"]
    right, up, f = cam
    xy = np.stack([V @ right, V @ up], 1)
    dep = V @ f
    lo, hi = xy.min(0), xy.max(0)
    if end_on is not None:  # keep the axis marker in the frame (a cone's axis can lie outside the mesh box)
        c = end_on["origin"] + 0.5 * (end_on["z"][0] + end_on["z"][1]) * end_on["axis"]
        cxy = np.array([c @ right, c @ up])
        lo, hi = np.minimum(lo, cxy), np.maximum(hi, cxy)
    pad = 0.06 * float(max(hi - lo))
    lo, hi = lo - pad, hi + pad
    zb = _ZBuf(xy, dep, T, lo, hi)
    tri = V[T]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nn = np.linalg.norm(nrm, axis=1)
    nrm = nrm / np.maximum(nn, 1e-12)[:, None]
    shade = 0.35 + 0.6 * np.abs(nrm @ f)
    order = np.argsort(-dep[T].mean(1))
    cols = np.clip(np.stack([0.80 * shade, 0.83 * shade, 0.88 * shade, np.ones_like(shade)], 1), 0, 1)
    fig = Figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.01, 0.06, 0.98, 0.86])
    ax.add_collection(PolyCollection(xy[T[order]], facecolors=cols[order], edgecolors=cols[order], linewidths=0.15,
                                     zorder=2, rasterized=True))
    hid = end_on is None
    if end_on is not None:
        F = end_on
        c = F["origin"] + 0.5 * (F["z"][0] + F["z"][1]) * F["axis"]
        cx, cy = float(c @ right), float(c @ up)
        ax.plot([cx], [cy], marker="+", ms=22, mew=2.2, color=C_AXIS, zorder=9)
        ax.text(cx, cy, f"  {F['id']} axis", color=C_AXIS, fontsize=10, fontweight="bold", zorder=9)
        s0, w0 = F["cover"]
        zm = 0.5 * (F["z"][0] + F["z"][1])
        for frac, txt in ((0.5, "Ro"), (0.35, "Ri")):
            a = s0 + frac * w0
            r = (F["r_out0"] if txt == "Ro" else F["r_in0"]) + F["k"] * zm
            p = c + r * (math.cos(a) * F["e1"] + math.sin(a) * F["e2"])
            ax.plot([cx, p @ right], [cy, p @ up], color=C_AXIS, lw=0.9, ls="--", zorder=8)
            ax.text(0.5 * (cx + p @ right), 0.5 * (cy + p @ up), f" {txt} {r:.3f}", color=C_AXIS, fontsize=9, zorder=9)
    for F in R["features"]:
        is_roll = F["kind"] == "ROLL"
        z0, z1 = F["z"]
        if is_roll:
            ext = 0.06 * (z1 - z0)
            if end_on is None:
                _polyline(ax, [F["origin"] + (z0 - ext) * F["axis"], F["origin"] + (z1 + ext) * F["axis"]], cam, zb,
                          C_AXIS, 1.8, "-.", f"{F['id']} axis", hidden=False)
            for fr in (0.25, 0.5, 0.75):
                z = z0 + fr * (z1 - z0)
                for r, lw in ((F["r_out0"] + F["k"] * z, 2.0 if hid else 1.1), (F["r_in0"] + F["k"] * z, 1.2 if hid else 1.1)):
                    _polyline(ax, _arc_pts(F, z, r), cam, zb, C_ARC, lw, hidden=hid)
        ro = F["runouts"]
        for end in ro.get("ends", []):
            for skin, style in (("outer", "-"), ("inner", "--")):
                e = end[skin]
                colour = C_RUNOUT if is_roll else C_JB
                lab = (f"{F['id']} {end['which']}" if skin == "outer" else None)
                if end_on is not None:
                    q = np.asarray(e["start_mm"])
                    ax.plot([q @ right], [q @ up], marker="o" if skin == "outer" else "s", ms=9 if skin == "outer" else 6,
                            color=colour, mfc="none" if skin == "inner" else colour, mew=2, zorder=9)
                    if lab:
                        ax.text(q @ right, q @ up, "  " + lab, color=colour, fontsize=9, fontweight="bold", zorder=9)
                elif e["length_mm"] > 0:
                    _polyline(ax, [e["start_mm"], e["end_mm"]], cam, zb, colour, 3.0 if skin == "outer" else 1.8,
                              style, lab)
    for b in doc.get("e1_bends", []):
        if b["class"] == "ROLLED":  # an E1 ROLLED (a) graph bend is drawn as the roll it became, not as a bend line
            continue
        s = b["oml_virtual_sharp"]
        if end_on is not None:
            q = np.asarray(s["start_mm"])
            ax.plot([q @ right], [q @ up], marker="X", ms=10, color=C_E1, zorder=9)
            ax.text(q @ right, q @ up, f"  E1 {b['graph_bend_id']}", color=C_E1, fontsize=9, fontweight="bold", zorder=9)
        else:
            _polyline(ax, [s["start_mm"], s["end_mm"]], cam, zb, C_E1, 2.6, "-", f"E1 {b['graph_bend_id']}", eps=2.5)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.01, 0.975, title, fontsize=11, fontweight="bold", va="top")
    legend = [Line2D([], [], color=C_AXIS, ls="-.", lw=1.8, label="roll axis"),
              Line2D([], [], color=C_RUNOUT, lw=3, label="roll run-out, outer skin"),
              Line2D([], [], color=C_RUNOUT, lw=1.8, ls="--", label="roll run-out, inner skin"),
              Line2D([], [], color=C_ARC, lw=2, label="fitted arcs at 25/50/75 % (outer thick, inner thin)"),
              Line2D([], [], color=C_E1, lw=2.6, label="E1 bend line (OML virtual sharp)"),
              Line2D([], [], color=C_JB, lw=3, label="junction bend tangent lines"),
              Line2D([], [], color="#555555", lw=0.9, ls=":", label="hidden behind material")]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8, frameon=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    data = buf.getvalue()
    if len(data) > 590_000:
        from PIL import Image

        im = Image.open(io.BytesIO(data)).convert("RGB").quantize(colors=128)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
    Path(out_png).write_bytes(data)
    return len(data)


def render_part(pkl: Path, out_dir: Path) -> list[tuple[str, int]]:
    R = pickle.loads(Path(pkl).read_bytes())
    doc = R["doc"]
    name = doc["source"].get("name") or Path(pkl).stem.replace(".render", "")
    out = []
    rolls = [F for F in R["features"] if F["kind"] == "ROLL"]
    if not rolls:
        return out
    main = max(rolls, key=lambda F: F["cover"][1] * F["r_in0"] * (F["z"][1] - F["z"][0]))
    s, w = main["cover"]
    bulge = math.cos(s + w / 2) * main["e1"] + math.sin(s + w / 2) * main["e2"]
    side = np.cross(main["axis"], bulge)
    view = _unit(bulge) * 0.6 - main["axis"] * 0.45 + side * 0.35  # from the concave side: both run-outs visible
    by_id = {r["id"]: r for r in doc["rolls"]}

    def brief(rid):
        r = by_id[rid]
        ro = ", ".join(f"{x['which']}={x['class']}" for x in r["runouts"]) or "closed"
        return (f"{rid}: Ri {r['r_inner_mm']:.3f} / Ro {r['r_outer_mm']:.3f} mm, t {r['thickness_mm']:.3f}, sweep "
                f"{r['sweep_deg']:.2f} deg, L {r['axial_length_mm']:.1f} mm{' CONE' if r['is_cone'] else ''}; run-outs {ro}")
    title = f"{name}  ISO  |  " + " ; ".join(brief(F["id"]) for F in rolls[:2])
    p = out_dir / f"{name}_iso.png"
    out.append((p.name, render_view(R, _camera(view, main["axis"]), title, p)))
    for F in rolls:
        s, w = F["cover"]
        bul = math.cos(s + w / 2) * F["e1"] + math.sin(s + w / 2) * F["e2"]
        p = out_dir / f"{name}_{F['id']}_end.png"
        out.append((p.name, render_view(R, _camera(-F["axis"], bul), f"{name}  END-ON along {F['id']} axis  |  " +
                                        brief(F["id"]), p, end_on=F)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in a.pkl:
        for name, size in render_part(Path(p), out):
            print(name, size)


if __name__ == "__main__":
    main()
