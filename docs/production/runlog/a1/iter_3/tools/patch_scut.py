"""Prototype (scratch tree only): a short child's line support stops at the grandchild bend's tangent lines."""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "domains/production/engine/inspect/articulated.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)


rep("""           continuity: bool = False, grand_u: float | None = None):""",
    """           continuity: bool = False, grand_u: float | None = None, s_cut=None):""")
rep("""        sel = (sv > s_min) & (sv < s_max)
        if not sel.any():
            continue
        rho = -uc[sel] * s + vc[sel] * c""",
    """        sel = (sv > s_min) & (sv < s_max)
        if s_cut is not None:
            sel &= ~((zc > s_cut[1]) & (zc < s_cut[2]) & (sv >= s_cut[0][k]))
        if not sel.any():
            continue
        rho = -uc[sel] * s + vc[sel] * c""")
rep("""                      grand_u: float | None = None) -> tuple[float, float, float, float, float] | None:""",
    """                      grand_u: float | None = None, s_cut_fn=None) -> tuple[float, float, float, float, float] | None:""")
rep("""    sc, _, _, _ = _hough(zc, uc, vc, frame, coarse, wide, reach, grand_u=grand_u)""",
    """    sc, _, _, _ = _hough(zc, uc, vc, frame, coarse, wide, reach, grand_u=grand_u,
                         s_cut=None if s_cut_fn is None else s_cut_fn(coarse))""")
rep("""    sc2, rh2, ni2, no2 = _hough(zc, uc, vc, frame, fine, wide, reach, grand_u=grand_u)""",
    """    sc2, rh2, ni2, no2 = _hough(zc, uc, vc, frame, fine, wide, reach, grand_u=grand_u,
                                s_cut=None if s_cut_fn is None else s_cut_fn(fine))""")
rep("""        grand_u = _grand_offset(part, bend, grand, parent, frame) if short else None
        sc, rh, ni, no = _hough(zc, uc, vc, frame, coarse, cfg, frame.band, continuity=short, grand_u=grand_u)""",
    """        grand_u = _grand_offset(part, bend, grand, parent, frame) if short else None

        def _cut(ths, _g=grand, _b=bend, _p=parent, _f=frame):
            if _g is None:
                return None
            lims = [_short_child_limit(part, _b, _g, _p, _f, float(t), False, cfg) for t in ths]
            return np.asarray([x[0] for x in lims]), lims[0][1], lims[0][2]
        sc, rh, ni, no = _hough(zc, uc, vc, frame, coarse, cfg, frame.band, continuity=short, grand_u=grand_u,
                                s_cut=_cut(coarse))""")
rep("""        sc2, rh2, ni2, no2 = _hough(wz, wu, wv, frame, fine, cfg, reach, continuity=short, grand_u=grand_u)""",
    """        sc2, rh2, ni2, no2 = _hough(wz, wu, wv, frame, fine, cfg, reach, continuity=short, grand_u=grand_u,
                                    s_cut=_cut(fine))""")
rep("""            alt = _displaced_formed(wz, wu, wv, frame, sweep_deg, cfg, reach, grand_u=grand_u)""",
    """            alt = _displaced_formed(wz, wu, wv, frame, sweep_deg, cfg, reach, grand_u=grand_u, s_cut_fn=_cut)""")
p.write_text(s)
print("patched", p)
