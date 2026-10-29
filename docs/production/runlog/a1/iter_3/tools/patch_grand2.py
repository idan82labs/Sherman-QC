p='/Users/idantal/ShermanQC/FullProject/domains/production/engine/inspect/articulated.py'
s=open(p).read()
def rep(old, new, count=1):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, count)

# 1. _hough: drop the hard clip
rep('''def _hough(zc, uc, vc, frame: HingeFrame, thetas: np.ndarray, cfg: InspectConfig, s_max: float,
           continuity: bool = False, grand_u: float | None = None):''','''def _hough(zc, uc, vc, frame: HingeFrame, thetas: np.ndarray, cfg: InspectConfig, s_max: float,
           continuity: bool = False):''')
rep('''    ``grand_u``: the child carries a parallel bend whose axis lies ``grand_u``
    along ``u`` from this one's (the child flat). A line at ``theta`` whose hinge
    sits ``d`` farther along ``u`` is offset by ``-d sin theta`` in ``rho``; a hinge
    nearer that bend than this one is that bend's line (this one flat, that one
    formed: a joggle or a hook half made), so on that side the window reaches at
    most ``max(hinge_rho_slack_flat_mm, grand_u / 2 |sin theta|)``.

''','')
rep('''    lows, highs = halves.copy(), halves.copy()
    if grand_u is not None and grand_u > 0:
        for k, th in enumerate(thetas):
            sth = math.sin(float(th))
            lim = max(float(cfg.hinge_rho_slack_flat_mm), 0.5 * float(grand_u) * abs(sth))
            if sth >= 0:
                lows[k] = min(lows[k], lim)
            else:
                highs[k] = min(highs[k], lim)
''','')
rep('''        ci_lo = int(math.floor((centres[k] - lows[k] - lo_r) / hb))
        ci_hi = int(math.ceil((centres[k] + highs[k] - lo_r) / hb))''','''        ci_lo = int(math.floor((centres[k] - halves[k] - lo_r) / hb))
        ci_hi = int(math.ceil((centres[k] + halves[k] - lo_r) / hb))''')
# 2. _displaced_formed: drop grand_u
rep('''def _displaced_formed(zc, uc, vc, frame: HingeFrame, sweep_deg: float, cfg: InspectConfig, reach: float,
                      grand_u: float | None = None) -> tuple[float, float, float, float, float] | None:''','''def _displaced_formed(zc, uc, vc, frame: HingeFrame, sweep_deg: float, cfg: InspectConfig, reach: float
                      ) -> tuple[float, float, float, float, float] | None:''')
rep('''    line inside the narrow window was already weighed by the narrow search.
    ``grand_u`` limits the side toward a parallel bend the child carries, as in
    :func:`_hough`.
    """''','''    line inside the narrow window was already weighed by the narrow search.
    """''')
rep('''    sc, _, _, _ = _hough(zc, uc, vc, frame, coarse, wide, reach, grand_u=grand_u)''','''    sc, _, _, _ = _hough(zc, uc, vc, frame, coarse, wide, reach)''')
rep('''    sc2, rh2, ni2, no2 = _hough(zc, uc, vc, frame, fine, wide, reach, grand_u=grand_u)''','''    sc2, rh2, ni2, no2 = _hough(zc, uc, vc, frame, fine, wide, reach)''')
# 3. helpers
rep('''def _grand_offset(part: PartView, bend: BendView, grand: int, parent: PanelFit, frame: HingeFrame) -> float | None:''','''def _hinge_shift(frame: HingeFrame, theta: float, rho_i: float) -> float | None:
    """How far along ``u`` (toward the child) the hinge of the line ``(theta, rho_i)`` sits from where the fold
    kinematics put this bend's hinge at ``theta`` (a line through a hinge moved by ``d`` is offset by
    ``-d sin theta``); None near flat, where a line has no hinge."""
    s = math.sin(theta)
    if abs(s) < 0.05:
        return None
    return (frame.rho_iml_at(theta) - rho_i) / s


def _flat_continuation(zc, uc, vc, frame: HingeFrame, sweep_deg: float, cfg: InspectConfig, s_max: float
                       ) -> tuple[float, float, float, float, float] | None:
    """``(theta, rho_iml, score, n_iml, n_oml)`` of the parent's sheet continued flat across this bend's hinge
    (within the decode's flat limit, material continuous from the hinge), else None."""
    lim = min(float(cfg.flat_sweep_max_deg), float(cfg.flat_fraction_max) * float(sweep_deg))
    th = np.radians(np.arange(-lim, lim + 1e-9, 0.25))
    sc, rh, ni, no = _hough(zc, uc, vc, frame, th, cfg, s_max, continuity=True)
    k = int(np.argmax(sc))
    if sc[k] <= 0.0:
        return None
    return float(th[k]), float(rh[k]), float(sc[k]), float(ni[k]), float(no[k])


def _grand_offset(part: PartView, bend: BendView, grand: int, parent: PanelFit, frame: HingeFrame) -> float | None:''')
# 4. fit_articulated
rep('''        sc, rh, ni, no = _hough(zc, uc, vc, frame, coarse, cfg, frame.band, continuity=short, grand_u=grand_u)''','''        sc, rh, ni, no = _hough(zc, uc, vc, frame, coarse, cfg, frame.band, continuity=short)''')
rep('''        sc2, rh2, ni2, no2 = _hough(wz, wu, wv, frame, fine, cfg, reach, continuity=short, grand_u=grand_u)
        k2 = int(np.argmax(sc2))
        theta, rho_i = float(fine[k2]), float(rh2[k2])
        score = float(sc2[k2])
        detail = {"n_iml": float(ni2[k2]), "n_oml": float(no2[k2]), "coarse_profile_max": float(sc.max())}
        if short:
            detail["short_child"] = 1.0''','''        sc2, rh2, ni2, no2 = _hough(wz, wu, wv, frame, fine, cfg, reach, continuity=short)
        k2 = int(np.argmax(sc2))
        theta, rho_i = float(fine[k2]), float(rh2[k2])
        score = float(sc2[k2])
        detail = {"n_iml": float(ni2[k2]), "n_oml": float(no2[k2]), "coarse_profile_max": float(sc.max())}
        if short:
            detail["short_child"] = 1.0

        def _grandchilds(th_: float, rho_: float) -> bool:
            """A formed-like line whose hinge lies nearer the child's parallel bend than this one, while the
            parent's sheet continues flat from this hinge: that bend is formed and this one is not yet."""
            if grand_u is None or abs(math.degrees(th_)) <= cfg.flat_sweep_max_deg:
                return False
            shift = _hinge_shift(frame, th_, rho_)
            return shift is not None and shift > 0.5 * grand_u
        if _grandchilds(theta, rho_i):
            flat_line = _flat_continuation(zc, uc, vc, frame, sweep_deg, cfg, frame.band)
            if flat_line is not None:
                detail.update({"grandchild_line_deg": math.degrees(theta), "n_iml": flat_line[3], "n_oml": flat_line[4]})
                theta, rho_i, score = flat_line[0], flat_line[1], flat_line[2]''')
rep('''            alt = _displaced_formed(wz, wu, wv, frame, sweep_deg, cfg, reach, grand_u=grand_u)
            if alt is not None and alt[2] >= cfg.displaced_support_ratio * max(score, 1.0):''','''            alt = _displaced_formed(wz, wu, wv, frame, sweep_deg, cfg, reach)
            if alt is not None and _grandchilds(alt[0], alt[1]) and \\
                    _flat_continuation(zc, uc, vc, frame, sweep_deg, cfg, frame.band) is not None:
                alt = None  # the next bend's formed line beyond a flat riser, not this one displaced
            if alt is not None and alt[2] >= cfg.displaced_support_ratio * max(score, 1.0):''')
# 5. module docstring
rep('''   defect is measured instead of read as a half-formed bend. On such a child the
   window stops, on the side toward the parallel bend, halfway to that bend's
   hinge (``grand_u`` in :func:`_hough`): a formed tab beyond a still flat riser
   is the parallel bend's line, not this one's displaced. A line seen on one skin''','''   defect is measured instead of read as a half-formed bend. On such a child, a
   formed-like line whose hinge lies nearer the parallel bend's axis than this
   one's, while the parent's sheet continues flat from this hinge
   (:func:`_flat_continuation`), is the parallel bend's line: that bend is formed
   and this one is not yet (a joggle or a hook half made). A line seen on one skin''')
open(p,'w').write(s)
print("patched")
