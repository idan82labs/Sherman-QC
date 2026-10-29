"""Apply the far-end (adjacent bend) flat-material search to metrology.py (A1 iter 7, v2). Run once."""
from pathlib import Path
p = Path("/Users/idantal/ShermanQC/FullProject/domains/production/engine/inspect/metrology.py")
s = p.read_text()
anchor = "def _in_plane_dir(plane: Plane, a: np.ndarray, toward: np.ndarray) -> np.ndarray:"
assert s.count(anchor) == 1 and "_flat_end(" not in s
fn = '''def _flat_end(s: np.ndarray, d: np.ndarray, z: np.ndarray, start: float, end0: float, t: float,
              zlo: float, zhi: float, cfg: InspectConfig) -> float:
    """Where a side's flat material ends toward an adjacent bend (at most ``end0``): short of that bend's own arc.

    The mirror of :func:`_flat_start` at the far end of a side bounded by another
    bend, whose window stops 0.5 mm short of that bend's CAD tangent line: a real
    radius larger than the CAD's brings its arc's tail into the window from there.
    The strip ``flat_start_test_mm`` wide at the window's end is tested against the
    line of up to ``flat_start_ref_mm`` of skin ending ``flat_start_gap_mm`` before
    it; the adjacent bend may turn either way about this side, so a median residual
    beyond ``flat_start_dev_mm`` on either side (or a strip that lost its skin) moves
    the end in, by at most ``flat_start_max_shift_mm`` and never below half a
    reference past ``start``.
    """
    g, gap, step = cfg.flat_start_test_mm, cfg.flat_start_gap_mm, cfg.flat_start_step_mm
    ref_min = 0.5 * cfg.flat_start_ref_mm
    limit = max(end0 - cfg.flat_start_max_shift_mm, start + g + gap + ref_min)
    if cfg.flat_start_max_shift_mm <= 0.0 or limit > end0 - step:
        return end0
    band = 0.45 * t
    sel = (s >= start) & (s <= end0) & (z >= zlo) & (z <= zhi)
    ss, dd, zz = s[sel], d[sel], z[sel]
    need = cfg.min_station_points * cfg.stations_min
    skins = [(m, lv) for m, lv in ((np.abs(dd) < band, 0.0), (np.abs(dd + t) < band, -t)) if int(m.sum()) >= need]
    if not skins:
        return end0
    keep, level = max(skins, key=lambda k: int(k[0].sum()))
    ss, dd, zz = ss[keep], dd[keep] - level, zz[keep]
    n_chunks = max(1, int(round((zhi - zlo) / cfg.flat_start_chunk_mm)))
    chunk = np.clip(((zz - zlo) / max(zhi - zlo, 1e-9) * n_chunks).astype(np.int64), 0, n_chunks - 1)
    groups = [np.nonzero(chunk == k)[0] for k in range(n_chunks)]
    s1 = end0
    while s1 >= limit - 1e-9:
        r1 = s1 - g - gap
        ref_len = min(cfg.flat_start_ref_mm, r1 - start)
        res = []
        expected = 0.0
        for gi in groups:
            sg, dg = ss[gi], dd[gi]
            ref = (sg > r1 - ref_len) & (sg <= r1)
            if int(ref.sum()) < 20 or float(np.ptp(sg[ref])) < 0.6 * ref_len:
                continue
            line = _robust_line(sg[ref], dg[ref])
            if line is None:
                continue
            tst = (sg > s1 - g) & (sg <= s1)
            r = dg[tst] - (line[0] + line[1] * sg[tst])
            res.append(r[np.abs(r) < band])
            expected += float(ref.sum()) * g / ref_len
        r = np.concatenate(res) if res else np.zeros(0)
        if expected < 20.0:
            return float(s1)
        departed = len(r) < 0.5 * expected or abs(float(np.median(r))) > cfg.flat_start_dev_mm
        if not departed:
            return float(s1)
        s1 -= step
    return float(max(s1, limit))


def _adjacent_limited(part: PartView, bend: BendView, panel_idx: int, d_t: float, depth: float, cfg: InspectConfig
                      ) -> bool:
    """Does the side's window end at its adjacent bend (that bend's CAD tangent line), not at the fit depth?"""
    side = bend.side_of_panel(panel_idx)
    nom = part.nominals.get(f"{bend.uid}/FLANGE_LENGTH/{side}")
    if nom is None or nom.nominal is None or nom.boundary_kind != "ADJACENT_BEND":
        return False
    if nom.adjacent_bend_uid not in part.bend_index:
        return False
    other = part.bends[part.bend_index[nom.adjacent_bend_uid]]
    far = float(nom.nominal) - (other.ro * math.tan(min(other.sweep, math.radians(170.0)) / 2.0) + 0.5)
    return far < d_t + depth


'''
s = s.replace(anchor, fn + anchor)
old = '''    if not model_c:
        start_c = _flat_start(s_c, dc, z, start_c, depth_c, t, zlo_nom, zhi_nom, cfg)
'''
new = '''    if not model_c:
        start_c = _flat_start(s_c, dc, z, start_c, depth_c, t, zlo_nom, zhi_nom, cfg)
    # ... and ends short of an adjacent bend's own arc (the same tail, from the far side)
    if not model_p and _adjacent_limited(part, bend, bend.parent, d_t, depth, cfg):
        depth_p = _flat_end(s_p, dp, z, start_p, depth_p, t, zlo_nom, zhi_nom, cfg)
    if not model_c and _adjacent_limited(part, bend, bend.child, d_t, depth, cfg):
        depth_c = _flat_end(s_c, dc, z, start_c, depth_c, t, zlo_nom, zhi_nom, cfg)
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
print("patched")
