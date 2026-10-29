p='/Users/idantal/ShermanQC/FullProject/domains/production/engine/inspect/registration.py'
s=open(p).read()
old='''        rel = uv[sel] - c
        rr = np.linalg.norm(rel, axis=1)
        inner = rr < max(0.5, r_h - 1.0)
        if inner.sum() > max(8, 0.05 * np.pi * r_h * r_h / max(spacing * spacing, 1e-6)):
            hole_mismatch += 1  # material where the CAD has a hole'''
new='''        rel = uv[sel] - c
        rr = np.linalg.norm(rel, axis=1)
        r_in = max(0.5, r_h - 1.0)
        inner = rr < r_in
        open_share = 1.0
        blocked_of = getattr(model, "blocked_points", None)
        if blocked_of is not None and len(blocked_of(k_hole)) and inner.any():
            # compiled stage material inside the hole's loop (the smaller bore of a stepped or countersunk
            # hole, a tab lying flat in it): scan material there is the part's own, never a filled hole
            inner_idx = np.nonzero(inner)[0]
            inner = np.zeros(len(rr), dtype=bool)
            inner[inner_idx[model.open_cells(k_hole, uv[sel][inner_idx])]] = True
            open_share = _open_disc_share(model, k_hole, c, r_in)
        if open_share >= _HOLE_FILL_MIN_OPEN_SHARE and \\
                inner.sum() > max(8, 0.05 * open_share * np.pi * r_h * r_h / max(spacing * spacing, 1e-6)):
            hole_mismatch += 1  # material where the CAD has a hole'''
assert old in s, "block"
s=s.replace(old,new)
old='''def _hole_bins(r_h: float, spacing: float, cfg: InspectConfig) -> int:'''
new='''#: A round hole whose inner disc is open (clear of compiled stage material) over less than this share is
#: not judged for fill: what the scan shows there is the part's own material.
_HOLE_FILL_MIN_OPEN_SHARE = 0.25


def _open_disc_share(model: "DatumModel", k: int, c: np.ndarray, r: float, cell: float = 0.25) -> float:
    """Share of the disc of radius ``r`` about ``c`` that is clear of compiled stage material in hole ``k``."""
    g = np.arange(-r, r + 1e-9, cell)
    xx, yy = np.meshgrid(g, g)
    cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
    cells = cells[np.hypot(cells[:, 0], cells[:, 1]) <= r] + c
    if len(cells) == 0:
        return 1.0
    return float(np.mean(model.open_cells(k, cells)))


def _hole_bins(r_h: float, spacing: float, cfg: InspectConfig) -> int:'''
assert old in s, "bins"
s=s.replace(old,new,1)
open(p,'w').write(s)
print("patched")
