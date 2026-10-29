"""zs_lane_v4 selection and line post-processing (runtime side; deterministic; no labels).

Given the pooled candidates of a v4 record and their selector probabilities:
  1. selection at threshold ``thr`` with zs_lane_v2's sweep gate (``sweep_min``) and, optionally, zs_lane_v2's
     plate-continuity join (join_low 0 + join dedupe, as frozen in v3) -- zs_lane_v2.postproc.postprocess, unchanged;
  2. EXT (zs_lane_v3.fixes.extend) in one of three modes:
       ``off``   no extension;
       ``v3``    the frozen v3 rule (ratio 1.3, containment 0.6, then drop the output lines it duplicates);
       ``safe``  the v3 rule, except that a line is NOT extended when the extended segment would duplicate (and so
                 drop) another output line. An extension that swallows another detected segment is exactly the
                 lanced-tab failure of v3 (one 413 mm line through 14 separate tabs, gb_68cd06): separate
                 segments on one line are left to the plate-continuity join, which tests whether they share plates;
       ``safe_sup`` ``safe`` and the extension must be supported by a formed bend: apex-touch fraction over the
                 extension >= ``ext_touch`` and no unsupported run longer than ``ext_gap_t`` x t (+1 mm);
  3. optional near-duplicate suppression (``nms``): a line is dropped when a higher-scored output line is
     equivalent to it under the matcher's geometric gates (axis <= 5 deg, offset <= 4 mm, >= 0.6 of both spans);
  4. optional hem lines from the hem generator (``hem``; see hem.py), added unless they duplicate an output line.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from research.zeroshot.zs_lane_v4 import profiles as PR


@dataclass(frozen=True)
class SelParams:
    thr: float = 0.5
    sweep_min: float = 0.25
    join: bool = True
    ext: str = "safe"            # off | v3 | safe | safe_sup
    ext_touch: float = 0.8
    ext_gap_t: float = 2.0
    nms: bool = False
    hem: bool = False
    hem_thr: float = 0.5
    hyp: bool = False            # extent hypotheses (hypotheses.py) compete with the candidates (select_h)
    alt_guard: bool = True       # an extent hypothesis may not swallow another selected candidate

    def key(self) -> str:
        return ",".join(f"{k}={v}" for k, v in asdict(self).items())


def _post(cands: list[dict], prob: np.ndarray, ev: dict | None, prm: SelParams) -> list[dict]:
    from research.zeroshot.zs_lane_v2.postproc import PostParams, postprocess
    pp = PostParams(join=prm.join, join_low=0.0 if prm.join else 1.0, dedupe_join=prm.join, sweep_min=prm.sweep_min)
    out = postprocess(cands, prob, prm.thr, ev, pp)
    for o in out:
        o.pop("_i", None)
    return out


def _ext_supported(rec: dict, line: dict, ext_lo: float, ext_hi: float, prm: SelParams) -> bool:
    """Is the extension [ext_lo, 0) U (L, ext_hi] (mm along the line's main candidate) supported by a formed bend?"""
    prof = (rec.get("profiles") or {}).get(str(line["parts"][0])) if line.get("parts") else None
    if prof is None:
        return False
    t = float((rec.get("meta") or {}).get("t_mm") or 1.5)
    L = prof["L"]
    parts = [(ext_lo, 0.0), (L, ext_hi)]
    tot = sum(max(0.0, b - a) for a, b in parts)
    if tot <= 0.5:
        return True
    fr = sum(PR.frac(prof, "touch", a, b) * (b - a) for a, b in parts if b - a > 0.5) / tot
    gap = max(PR.longest_gap(prof, "touch", a, b) for a, b in parts if b - a > 0.5)
    return fr >= prm.ext_touch and gap <= prm.ext_gap_t * t + 1.0


def _extend(rec: dict, lines: list[dict], prm: SelParams) -> list[dict]:
    from research.zeroshot.zs_lane_v3.fixes import FixParams, extend, is_dup
    cands, z2 = rec["candidates"], rec.get("z2_lines") or []
    fx = FixParams(ext=True, ext_dedupe=True, z3r=False)
    if prm.ext == "v3":
        return extend(lines, cands, z2, fx)
    # per line: the v3 extension computed alone, kept only when it swallows no other output line
    ext_alone = extend(lines, cands, z2, FixParams(ext=True, ext_dedupe=False, z3r=False))
    out = []
    for k, (orig, ex) in enumerate(zip(lines, ext_alone)):
        if ex.get("fix") != "EXT":
            out.append(orig)
            continue
        swallows = any(is_dup(ex["start_s"], ex["end_s"], o["start_s"], o["end_s"])
                       for m, o in enumerate(lines) if m != k)
        ok = not swallows
        if ok and prm.ext == "safe_sup":
            a = np.asarray(orig["start_s"], float)
            b = np.asarray(orig["end_s"], float)
            u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            lo_mm, hi_mm = ex["ext_mm"]                # mm added before the start / after the end (snapped line)
            # map onto the main candidate's raw axis (same direction up to sign; lengths preserved)
            c = cands[orig["parts"][0]]
            ra, rb = np.asarray(c["start"], float), np.asarray(c["end"], float)
            ru = (rb - ra) / max(np.linalg.norm(rb - ra), 1e-9)
            Lc = float(np.linalg.norm(rb - ra))
            s_a = float((a - ra) @ ru)
            s_b = float((b - ra) @ ru)
            if float(u @ ru) >= 0:
                e0, e1 = min(s_a, s_b) - lo_mm, max(s_a, s_b) + hi_mm
            else:
                e0, e1 = min(s_a, s_b) - hi_mm, max(s_a, s_b) + lo_mm
            ok = _ext_supported(rec, orig, min(e0, 0.0), max(e1, Lc), prm)
        out.append(ex if ok else orig)
    return out


def _nms(lines: list[dict]) -> list[dict]:
    from research.zeroshot.zs_lane_v4.evalkit import FastScorer
    if len(lines) < 2:
        return lines
    order = sorted(range(len(lines)), key=lambda k: (-float(lines[k].get("score") or 0.0), k))
    keep: list[int] = []
    for k in order:
        P = np.array([[lines[k]["start_s"], lines[k]["end_s"]]], float)
        if keep:
            G = np.array([[lines[m]["start_s"], lines[m]["end_s"]] for m in keep], float)
            adm, _ = FastScorer.admissible(G, P)
            if adm.any():
                continue
        keep.append(k)
    return [lines[k] for k in sorted(keep)]


def select(rec: dict, prob: np.ndarray, prm: SelParams, hem_lines: list[dict] | None = None) -> list[dict]:
    """Output lines (start_s / end_s snapped placement, score, src, incl, parts, op, fix)."""
    cands = rec.get("candidates") or []
    lines = _post(cands, np.asarray(prob, float), rec.get("evidence"), prm) if cands else []
    if prm.ext != "off" and lines:
        lines = _extend(rec, lines, prm)
    if prm.nms:
        lines = _nms(lines)
    if prm.hem and hem_lines:
        from research.zeroshot.zs_lane_v3.fixes import is_dup
        for h in hem_lines:
            if float(h.get("score", 1.0)) < prm.hem_thr:
                continue
            if any(is_dup(h["start_s"], h["end_s"], o["start_s"], o["end_s"]) for o in lines):
                continue
            lines.append(dict(h, op="keep", fix="HEM"))
    return lines


def as_eval(lines: list[dict]) -> list[dict]:
    return [{"start": l["start_s"], "end": l["end_s"], "score": l.get("score"), "op": l.get("op"),
             "fix": l.get("fix"), "src": l.get("src")} for l in lines]


def select_h(rec: dict, alts: list[dict], prob: np.ndarray, prob_alt: np.ndarray, prm: SelParams,
             hem_lines: list[dict] | None = None) -> list[dict]:
    """Greedy competition of the pooled candidates and their extent hypotheses (hypotheses.py): in decreasing
    probability (ties: candidates first, then index), a line >= thr is taken unless it duplicates an already taken
    line (4 deg / 2.5 mm / 50 % of the shorter). With ``alt_guard`` a z2ext / union / join hypothesis is never taken
    when it would swallow a selected candidate that is not one of its parents: separate segments on one line (a row
    of tabs) stay separate. zs_lane_v2's sweep gate applies to the candidates; optional matcher-equivalence NMS and
    hem lines as in ``select``."""
    from research.zeroshot.zs_lane_v3.fixes import is_dup
    C = rec.get("candidates") or []
    ev = rec.get("evidence") or {}
    sw = ev.get("sweep")
    elig = []
    for i, c in enumerate(C):
        if prob[i] < prm.thr:
            continue
        if prm.sweep_min > 0 and sw is not None and i < len(sw) and sw[i] is not None and sw[i][0] < prm.sweep_min:
            continue
        elig.append(i)
    rows = [(float(prob[i]), 0, i, C[i]["start_s"], C[i]["end_s"], "base", [i]) for i in elig]
    for r, h in enumerate(alts):
        if prob_alt[r] >= prm.thr:
            rows.append((float(prob_alt[r]), 1, r, h["start_s"], h["end_s"], h["kind"], h["parents"]))
    rows.sort(key=lambda x: (-x[0], x[1], x[2]))
    taken: list[tuple] = []
    for x in rows:
        if any(is_dup(x[3], x[4], y[3], y[4]) for y in taken):
            continue
        if prm.alt_guard and x[1] == 1 and x[5] in ("z2ext", "union", "join"):
            if any(m not in x[6] and is_dup(x[3], x[4], C[m]["start_s"], C[m]["end_s"]) for m in elig):
                continue
        taken.append(x)
    lines = []
    for p_, isalt, k, s_, e_, kind, par in taken:
        src = sorted(set(z for m in par for z in (C[m].get("src") or [])))
        lines.append({"start_s": list(map(float, s_)), "end_s": list(map(float, e_)), "score": p_, "src": src,
                      "incl": C[par[0]].get("incl"), "parts": list(par), "op": "keep" if kind == "base" else kind,
                      "fix": None if kind == "base" else "HYP"})
    if prm.nms:
        lines = _nms(lines)
    if prm.hem and hem_lines:
        for h in hem_lines:
            if float(h.get("score", 1.0)) < prm.hem_thr:
                continue
            if any(is_dup(h["start_s"], h["end_s"], o["start_s"], o["end_s"]) for o in lines):
                continue
            lines.append(dict(h, op="keep", fix="HEM"))
    return lines
