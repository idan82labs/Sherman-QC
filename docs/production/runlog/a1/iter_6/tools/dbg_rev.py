"""Debug the wrong-way (mirrored hinge) search on one A1 case: in-process E3 up to the articulated fits, with
_reversed_reading instrumented for the listed bends. usage: dbg_rev.py CASE B2,B14 [--png PREFIX]"""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
from domains.production.engine.inspect import articulated as AR
from domains.production.engine.inspect import engine as E
from domains.production.engine.inspect.cloud import adaptive_voxel, estimate_normals, median_spacing, neighbour_graph, voxelize
from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import PartView, load_inputs
from domains.production.engine.inspect.registration import DatumModel
from domains.production.engine.inspect.scan_io import read_scan
SP = Path("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad")
SRC = SP / "a1_iter5/phys"
REPO = Path("/Users/idantal/ShermanQC/FullProject")
cid, labels = sys.argv[1], set(sys.argv[2].split(","))
png = sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None
case = {c["case_ref"]: c for c in json.loads((REPO / "docs/production/runlog/freeze/cases_A1.json").read_text())["cases"]}[cid]
inp = SRC / "physical/E3" / cid / "input"
scan = inp / "scan.ply"
cfg = InspectConfig()
inputs = load_inputs(SRC / "compiled/dev" / case["recipe_sku"], inp / "recipe.json", scan, hashlib.sha256(scan.read_bytes()).hexdigest())
part = PartView(inputs.compiled, inputs.compiled_dir)
orig = AR._reversed_reading
calls = [0]

def patched(full_reg, vox, vox_reg, vox_nrm, owner, part_, bend, parent, frame, sweep_deg, cad_score, reach, cfg_):
    r = orig(full_reg, vox, vox_reg, vox_nrm, owner, part_, bend, parent, frame, sweep_deg, cad_score, reach, cfg_)
    if bend.label in labels:
        calls[0] += 1
        fm, mb = AR.mirrored_hinge_frame(part_, bend, parent, frame)
        lo = max(-cfg_.theta_min_deg, min(cfg_.flat_sweep_max_deg, cfg_.flat_fraction_max * sweep_deg)) + 1.0
        hi = min(sweep_deg + cfg_.theta_overbend_deg, 180.0 - cfg_.reversed_wrap_margin_deg - 1.0)
        widx, wz, wu, wv = AR._band_points(full_reg, fm, 2.0, radius=reach)
        free = owner[vox.inverse[widx]] < 0
        print(f"--- call {calls[0]} {bend.label} sweep {sweep_deg:.1f} cad_score {cad_score:.0f} band pts {len(widx)} free {int(free.sum())} range {lo:.1f}..{hi:.1f}")
        widx, wz, wu, wv = widx[free], wz[free], wu[free], wv[free]
        inb = np.hypot(wu, wv) < fm.band
        th = np.radians(np.arange(lo, hi + 1e-9, 1.0))
        sc, rh, ni, no = AR._hough(wz[inb], wu[inb], wv[inb], fm, th, cfg_, fm.band)
        top = np.argsort(-sc)[:5]
        print("   coarse top:", [(round(math.degrees(th[k]), 1), round(float(sc[k])), round(float(ni[k])), round(float(no[k]))) for k in top])
        thf = np.radians(np.arange(lo, hi + 1e-9, 1.0))
        sc2, rh2, ni2, no2 = AR._hough(wz, wu, wv, fm, thf, cfg_, reach)
        top = np.argsort(-sc2)[:5]
        print("   reach top:", [(round(math.degrees(thf[k]), 1), round(float(sc2[k])), round(float(ni2[k])), round(float(no2[k])), round(float(rh2[k]), 2), round(fm.rho_iml_at(float(thf[k])), 2)) for k in top])
        print("   adopted:", None if r is None else (round(math.degrees(r[1]), 2), round(r[3]), r[4], r[6], {k: round(v, 2) for k, v in r[7].items()}))
        if png:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            a_w, a_u, a_v = AR._band_points(full_reg, frame, 2.0, radius=reach)[1:]
            ax.scatter(a_u, a_v, s=0.2, c="0.6")
            o2 = fm.o - frame.o
            ax.plot([0, float(o2 @ frame.u)], [0, float(o2 @ frame.v)], "rx")
            ax.set_aspect("equal"); ax.set_xlim(-12, 18); ax.set_ylim(-18, 12)
            ax.set_title(f"{bend.label} CAD frame (u,v); x = CAD axis and mirrored axis")
            fig.savefig(f"{png}_{bend.label}_{calls[0]}.png", dpi=80); plt.close(fig)
            # mirrored frame: profile histogram of rho at the reach-top angles, per angle
            for kk in np.argsort(-sc2)[:3]:
                t_ = float(thf[kk]); c_, s_ = math.cos(t_), math.sin(t_)
                sv = wu * c_ + wv * s_; rr = -wu * s_ + wv * c_
                sel = (sv > 1.0) & (sv < reach)
                h_, e_ = np.histogram(rr[sel], bins=np.arange(-4.0, 3.0, 0.1))
                print(f"   rho hist at {math.degrees(t_):.1f} (s>1): " + " ".join(f"{e_[i]:.1f}:{h_[i]}" for i in range(len(h_)) if h_[i] > 20))
                for s_lo, s_hi in ((1, 3), (3, 6), (6, 10)):
                    sel2 = (sv > s_lo) & (sv < s_hi)
                    h2, _ = np.histogram(rr[sel2], bins=np.arange(-4.0, 3.0, 0.1))
                    print(f"      s {s_lo}-{s_hi}: " + " ".join(f"{e_[i]:.1f}:{h2[i]}" for i in range(len(h2)) if h2[i] > 10))
    return r
AR._reversed_reading = patched
pts = read_scan(scan).points
spacing = median_spacing(pts)
voxel = max(adaptive_voxel(pts, spacing, cfg.voxel_min_mm, cfg.voxel_max_mm, cfg.voxel_target_points), min(cfg.voxel_max_mm, spacing))
vox = voxelize(pts, voxel); vox.slab = cfg.slab_mode(part.t, voxel)
estimate_normals(vox, cfg.normal_neighbours, normal_voxel=max(voxel, cfg.thin_normal_voxel_factor * part.t) if vox.slab else None)
neighbour_graph(vox, cfg.graph_radius_voxels * voxel)
model = DatumModel.from_part(part)
reg_src = E._decimate(pts, cfg.registration_decimate_mm) if spacing < cfg.registration_decimate_mm else pts
budget = E._Budget(170)
proper = E._register(reg_src, vox, part, model, cfg, budget, lambda m: None, median_spacing(reg_src) if reg_src is not pts else spacing, mirrored=False)
for h in proper[:cfg.articulated_hypotheses]:
    E._articulate(h, pts, vox, part, model, cfg, budget)
    print(f"hyp total {h.total:.4f} expl {h.explained:.3f} states " + " ".join(f"{part.bends[j].label}:{d.state[:3]}" for j, d in enumerate(h.decodes)))
    for j, b in enumerate(part.bends):
        if b.label in labels:
            bf = h.art.bends[j]
            print(f"   {b.label} found={bf.found} reversed={bf.reversed} theta={bf.theta_deg} score={bf.score:.0f} pts={bf.child_line_points} cov={bf.axial_coverage:.2f} reason={bf.reason} detail={ {k: round(v, 2) for k, v in bf.detail.items()} }")
