exec(open('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/fixes2/seeddiag.py').read().split('print("coarse hyps"')[0])
def inplane(T):
    # in-plane offset (mm) and height offset of the datum origin mapped by T^-1 vs Ti^-1
    a = RG.apply(np.linalg.inv(T), model.p0[None])[0]; b = RG.apply(np.linalg.inv(Ti), model.p0[None])[0]
    dd = a - b; return round(float(np.linalg.norm(dd)), 1)
for h in hyps:
    if h.segment == 2:
        a, d = near(h.T)
        if a < 10:
            Tf, sc = RG.fine_search(reg, h.T, model, cfg, with_score=True, walks=0)
            fit = RG.refine_datum(reg, Tf, model, cfg, median_spacing(reg))
            print(f"seed seg2 fam {h.family} h {h.height} seed? {h.segment_seed} coarse {h.score:.3f} turn {a:.1f} off {inplane(h.T)} -> fine {sc:.3f} off {inplane(Tf)} turn {near(Tf)[0]:.1f} -> refined off {inplane(fit.T)} score {fit.score:.3f} undetermined {fit.undetermined.shape[0]}")
Tf, sc = RG.fine_search(reg, Ti, model, cfg, with_score=True, walks=0); print("ICP fine", round(sc,3), inplane(Tf))
for h in hyps:
    if h.segment == 2 and near(h.T)[0] < 10:
        Tf, sc = RG.fine_search(reg, h.T, model, cfg, with_score=True, walks=0)
        fit = RG.refine_datum(reg, Tf, model, cfg, median_spacing(reg))
        for k in range(3):
            T2, s2 = RG.fine_search(reg, fit.T, model, cfg, with_score=True, walks=0)
            fit2 = RG.refine_datum(reg, T2, model, cfg, median_spacing(reg))
            print("re-search", k, round(s2, 3), "off", inplane(T2), "refined off", inplane(fit2.T), "score", round(fit2.score, 3))
            fit = fit2
