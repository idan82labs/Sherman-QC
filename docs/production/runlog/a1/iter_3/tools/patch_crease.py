import re
p='/Users/idantal/ShermanQC/FullProject/domains/production/engine/inspect/engine.py'
s=open(p).read()
old='''        ctx.crease_sharps.setdefault(c.panel, []).append((point, c.direction, c.sharp_u, c.model.crease_id))'''
new='''        ctx.crease_sharps.setdefault(c.panel, []).append((point, c.direction, c.sharp_u, c.model.crease_id,
                                                          np.asarray(c.model.polyline_mm, dtype=np.float64)))'''
assert old in s, "engine tuple"
s=s.replace(old,new)
open(p,'w').write(s)

p='/Users/idantal/ShermanQC/FullProject/domains/production/engine/inspect/metrology.py'
s=open(p).read()
old='''        u_far = 0.0
        along = [c for c in creases if abs(float(unit(c[1]) @ bm.line_dir)) >= 0.8]'''
new='''        u_far = 0.0
        along = [c for c in creases if abs(float(unit(c[1]) @ bm.line_dir)) >= 0.8 and _crease_covers(c, bm) is not False]'''
assert old in s, "along"
s=s.replace(old,new)
old='''def _crease_crossings(bm: BendMeasure, w: np.ndarray, d_t: float, creases) -> list[tuple[float, float]]:
    """Per valid station: (distance to the nearest crease OML sharp along ``w``, its U) for creases along the bend."""
    out: list[tuple[float, float]] = []
    along = [c for c in creases if c[0] is not None and abs(float(unit(c[1]) @ bm.line_dir)) >= 0.8]
    if not along:
        return out
    for i in np.nonzero(bm.valid)[0]:
        best = None
        for cp, cd, cu, _cid in along:
            ls = _station_crossing(bm.sharp[i], bm.line_dir, w, cp, unit(cd))'''
new='''def _crease_covers(crease, bm: BendMeasure, z: float | None = None) -> bool | None:
    """Does the crease's own extent reach the bend's stations (or station position ``z``)? None: extent unknown.

    A crease is found where a segment hinges off the panel; it ends a FREE_EDGE side only
    over the length it actually runs, never along the whole flange as an unbounded line.
    """
    if len(crease) < 5 or crease[4] is None or len(crease[4]) < 2 or bm.z is None or len(bm.z) == 0:
        return None
    zc = (np.asarray(crease[4], dtype=np.float64) - bm.line_point) @ bm.line_dir
    half = 0.5 * float(bm.width)
    if z is None:
        return bool(zc.max() >= float(np.min(bm.z)) - half and zc.min() <= float(np.max(bm.z)) + half)
    return bool(zc.min() - half <= z <= zc.max() + half)


def _crease_crossings(bm: BendMeasure, w: np.ndarray, d_t: float, creases) -> list[tuple[float, float]]:
    """Per valid station: (distance to the nearest crease OML sharp along ``w``, its U) for creases along the bend
    whose extent covers that station (:func:`_crease_covers`)."""
    out: list[tuple[float, float]] = []
    along = [c for c in creases if c[0] is not None and abs(float(unit(c[1]) @ bm.line_dir)) >= 0.8]
    if not along:
        return out
    for i in np.nonzero(bm.valid)[0]:
        best = None
        for c in along:
            if _crease_covers(c, bm, float(bm.z[i])) is False:
                continue
            cp, cd, cu = c[0], c[1], c[2]
            ls = _station_crossing(bm.sharp[i], bm.line_dir, w, cp, unit(cd))'''
assert old in s, "crossings"
s=s.replace(old,new)
open(p,'w').write(s)
print("patched")
