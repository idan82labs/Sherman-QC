"""Flat (unfolded) part parsing: DXF via ezdxf, flat STEP via OCP (plan §3.1 "Flat binding").

The parser returns a :class:`ParsedFlat` in the flat's own 2D drawing frame (mm):

- ``outer``: the outer cut loop, ``inner``: hole / cut-out loops, all closed
  polylines (arcs, circles, ellipses and splines are discretised);
- ``bend_lines``: straight bend lines with deterministic ids ``L1..Ln``
  (sorted by midpoint), found by **layer** (a name containing BEND, FOLD,
  BIEGE, KANT, BENDLINE or the Hebrew כיפוף / קיפול), by **linetype** (dashed,
  center, phantom, hidden, dot families, on the entity or via its layer) or by
  **annotation** (a TEXT/MTEXT such as ``UP 90°``, ``DOWN 45``, ``90° DN``
  within reach of a straight line). Annotations also give direction and angle.
  A short line centred on a hole (within 1.5 hole radii, at most 4 radii
  long) is the hole's centre mark, not a bend line, and is dropped (noted).
- DXF units come from ``$INSUNITS`` (inch, cm, m are scaled to mm; unitless is
  taken as mm and noted).

A flat STEP is a planar solid without annotations: its largest planar face
gives the loops (in that face's plane frame) and the skin separation gives the
thickness; ``bend_lines`` is empty and lines are inferred by the binding
(:mod:`.flat_binding`), with ids ``I1..In``.

Failures raise :class:`FlatParseError` (compile issue FLAT_PARSE_FAILED).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from domains.production.engine.compile.geometry import polygon_area_2d

__all__ = ["FlatParseError", "FlatBendLine", "ParsedFlat", "parse_flat", "BEND_LAYER_RE", "BEND_LINETYPE_RE"]

BEND_LAYER_RE = re.compile(r"(BEND|FOLD|BIEG|KANT|ABKANT|BENDLINE|כיפוף|קיפול)", re.IGNORECASE)
BEND_LINETYPE_RE = re.compile(r"(DASH|CENTER|CENTRE|PHANTOM|HIDDEN|DOT|DIVIDE|BORDER)", re.IGNORECASE)
_ANNOTATION_RE = re.compile(
    r"(?:(?P<dir1>UP|DOWN|DN|UPWARD|DOWNWARD|למעלה|למטה)\s*(?P<ang1>\d+(?:[.,]\d+)?)?\s*(?:°|DEG|%%D)?)"
    r"|(?:(?P<ang2>\d+(?:[.,]\d+)?)\s*(?:°|DEG|%%D)\s*(?P<dir2>UP|DOWN|DN|UPWARD|DOWNWARD|למעלה|למטה)?)",
    re.IGNORECASE)
_INSUNITS = {0: (1.0, "unitless (taken as mm)"), 1: (25.4, "inch"), 2: (304.8, "foot"), 4: (1.0, "mm"),
             5: (10.0, "cm"), 6: (1000.0, "m"), 8: (25.4e-6, "microinch"), 9: (25.4e-3, "mil"),
             13: (1e-3, "micron")}


class FlatParseError(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass
class FlatBendLine:
    line_id: str
    start: np.ndarray
    end: np.ndarray
    source: str  # LAYER | LINETYPE | ANNOTATION | INFERRED
    layer: str = ""
    linetype: str = ""
    direction: str | None = None
    angle_deg: float | None = None
    annotation: str | None = None

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.start + self.end)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    def to_json(self) -> dict:
        return {"line_id": self.line_id, "start_mm": [round(float(c), 6) for c in self.start],
                "end_mm": [round(float(c), 6) for c in self.end], "source": self.source, "layer": self.layer,
                "linetype": self.linetype, "direction": self.direction, "angle_deg": self.angle_deg,
                "annotation": self.annotation}


@dataclass
class ParsedFlat:
    format: str  # DXF | STEP
    units: str
    scale_to_mm: float
    outer: np.ndarray
    inner: list[np.ndarray]
    bend_lines: list[FlatBendLine]
    thickness_mm: float | None = None
    notes: list[str] = field(default_factory=list)

    def loops(self) -> list[np.ndarray]:
        return [self.outer, *self.inner]

    def outline_samples(self, spacing: float = 0.5) -> np.ndarray:
        """Points every ~``spacing`` mm along all loops."""
        return self.outline_segments(spacing)[0]

    def outline_segments(self, spacing: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(points, seg_start, seg_end)``: samples every ~``spacing`` mm along all loops, each with
        the loop segment it lies on (for point-to-segment distances, free of the sampling phase)."""
        pts, seg_a, seg_b = [], [], []
        for loop in self.loops():
            closed = np.vstack([loop, loop[:1]])
            for a, b in zip(closed[:-1], closed[1:]):
                n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / spacing)))
                s = np.arange(n)[:, None] / n
                pts.append(a + s * (b - a))
                seg_a.append(np.broadcast_to(a, (n, 2)))
                seg_b.append(np.broadcast_to(b, (n, 2)))
        if not pts:
            empty = np.zeros((0, 2))
            return empty, empty, empty
        return np.concatenate(pts), np.concatenate(seg_a), np.concatenate(seg_b)

    def summary(self) -> dict:
        return {"format": self.format, "units": self.units, "scale_to_mm": self.scale_to_mm,
                "outer_points": int(len(self.outer)), "inner_loops": len(self.inner),
                "outer_area_mm2": abs(polygon_area_2d(self.outer)),
                "bend_lines": [b.to_json() for b in self.bend_lines], "thickness_mm": self.thickness_mm,
                "notes": self.notes[:50]}


def parse_flat(path: str | Path, fmt: str | None = None) -> ParsedFlat:
    """Parse a flat DXF or flat STEP (``fmt`` = DXF | STEP, else from the extension)."""
    p = Path(path)
    kind = (fmt or ("DXF" if p.suffix.lower() == ".dxf" else "STEP")).upper()
    if kind == "DXF":
        return _parse_dxf(p)
    if kind == "STEP":
        return _parse_flat_step(p)
    raise FlatParseError("FLAT_FORMAT_UNSUPPORTED", kind)


# --------------------------------------------------------------------------- DXF


def _flatten(entities, depth: int = 0):
    for e in entities:
        if e.dxftype() == "INSERT":
            if depth > 8:
                continue
            try:
                yield from _flatten(e.virtual_entities(), depth + 1)
            except Exception:
                continue
        else:
            yield e


def _discretize(e, scale: float) -> list[np.ndarray] | None:
    """Polyline(s) of a curve entity in mm (2D), or None when not a curve."""
    t = e.dxftype()
    try:
        if t == "LINE":
            a, b = e.dxf.start, e.dxf.end
            return [np.array([[a.x, a.y], [b.x, b.y]]) * scale]
        if t in ("ARC", "CIRCLE", "ELLIPSE", "SPLINE"):
            pts = list(e.flattening(0.01 / max(scale, 1e-9)))
            arr = np.array([[q.x, q.y] for q in pts]) * scale
            return [arr] if len(arr) >= 2 else None
        if t == "LWPOLYLINE":
            pts = []
            for seg in e.virtual_entities():
                sub = _discretize(seg, 1.0)
                if sub:
                    pts.append(sub[0])
            if not pts:
                return None
            arr = np.concatenate([pts[0]] + [s[1:] for s in pts[1:]]) * scale
            if e.closed and len(arr) >= 3 and np.linalg.norm(arr[0] - arr[-1]) > 1e-9:
                arr = np.vstack([arr, arr[:1]])
            return [arr]
        if t == "POLYLINE":
            pts = []
            for seg in e.virtual_entities():
                sub = _discretize(seg, 1.0)
                if sub:
                    pts.append(sub[0])
            if not pts:
                return None
            return [np.concatenate([pts[0]] + [s[1:] for s in pts[1:]]) * scale]
    except Exception:
        return None
    return None


def _linetype_of(doc, e) -> str:
    lt = (e.dxf.get("linetype", "BYLAYER") or "BYLAYER").upper()
    if lt in ("BYLAYER", "BYBLOCK"):
        try:
            layer = doc.layers.get(e.dxf.layer)
            lt = (layer.dxf.linetype or "CONTINUOUS").upper()
        except Exception:
            lt = "CONTINUOUS"
    return lt


def _parse_annotation(text: str) -> tuple[str | None, float | None] | None:
    m = _ANNOTATION_RE.search(text.replace(",", "."))
    if not m:
        return None
    word = m.group("dir1") or m.group("dir2")
    ang = m.group("ang1") or m.group("ang2")
    direction = None
    if word:
        direction = "UP" if word.upper().startswith("UP") or word == "למעלה" else "DOWN"
    angle = float(ang) if ang else None
    if direction is None and angle is None:
        return None
    return direction, angle


def _chain_loops(chains: list[np.ndarray], tol: float) -> tuple[list[np.ndarray], int]:
    """Join open polylines end-to-end into closed loops; returns (loops, open chains left)."""
    loops = []
    open_chains = []
    for c in chains:
        if len(c) >= 4 and np.linalg.norm(c[0] - c[-1]) <= tol:
            loops.append(c[:-1])
        else:
            open_chains.append(c)
    used = [False] * len(open_chains)
    for i in range(len(open_chains)):
        if used[i]:
            continue
        used[i] = True
        cur = open_chains[i]
        progress = True
        while progress and np.linalg.norm(cur[0] - cur[-1]) > tol:
            progress = False
            for j in range(len(open_chains)):
                if used[j]:
                    continue
                c = open_chains[j]
                if np.linalg.norm(cur[-1] - c[0]) <= tol:
                    cur = np.vstack([cur, c[1:]])
                elif np.linalg.norm(cur[-1] - c[-1]) <= tol:
                    cur = np.vstack([cur, c[::-1][1:]])
                elif np.linalg.norm(cur[0] - c[-1]) <= tol:
                    cur = np.vstack([c, cur[1:]])
                elif np.linalg.norm(cur[0] - c[0]) <= tol:
                    cur = np.vstack([c[::-1], cur[1:]])
                else:
                    continue
                used[j] = True
                progress = True
        if len(cur) >= 4 and np.linalg.norm(cur[0] - cur[-1]) <= tol:
            loops.append(cur[:-1])
    left = sum(1 for k, c in enumerate(open_chains) if not used[k])
    return loops, left


def _dedupe(poly: np.ndarray, tol: float) -> np.ndarray:
    keep = [poly[0]]
    for q in poly[1:]:
        if np.linalg.norm(q - keep[-1]) > tol:
            keep.append(q)
    if len(keep) > 1 and np.linalg.norm(keep[0] - keep[-1]) <= tol:
        keep.pop()
    return np.array(keep)


def _point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
    from matplotlib.path import Path as MplPath

    return bool(MplPath(poly).contains_point(tuple(pt)))


def _parse_dxf(path: Path) -> ParsedFlat:
    try:
        import ezdxf
    except ImportError as exc:  # pragma: no cover - ezdxf 1.4.5 is a production requirement
        raise FlatParseError("EZDXF_UNAVAILABLE", str(exc)) from exc
    try:
        doc = ezdxf.readfile(str(path))
    except Exception as exc:
        raise FlatParseError("DXF_UNREADABLE", f"{type(exc).__name__}: {exc}"[:300]) from exc
    notes: list[str] = []
    code = int(doc.header.get("$INSUNITS", 0) or 0)
    scale, units = _INSUNITS.get(code, (1.0, f"INSUNITS {code} (taken as mm)"))
    if code not in _INSUNITS:
        notes.append(f"unknown $INSUNITS {code}; taken as mm")
    if code == 0:
        notes.append("DXF has no units ($INSUNITS 0); taken as mm")
    curves, bend_candidates, annotations = [], [], []
    for e in _flatten(doc.modelspace()):
        t = e.dxftype()
        if t in ("TEXT", "MTEXT", "ATTRIB"):
            try:
                text = e.plain_text() if t == "MTEXT" else e.dxf.text
                ins = e.dxf.insert
                annotations.append((str(text), np.array([ins.x, ins.y]) * scale))
            except Exception:
                continue
            continue
        polys = _discretize(e, scale)
        if not polys:
            continue
        layer = str(e.dxf.get("layer", "0"))
        linetype = _linetype_of(doc, e)
        poly = polys[0]
        straight = t == "LINE" or (t in ("LWPOLYLINE", "POLYLINE") and len(poly) == 2)
        by_layer = bool(BEND_LAYER_RE.search(layer))
        by_lt = bool(BEND_LINETYPE_RE.search(linetype))
        if straight and (by_layer or by_lt):
            bend_candidates.append({"a": poly[0], "b": poly[-1], "layer": layer, "linetype": linetype,
                                    "source": "LAYER" if by_layer else "LINETYPE"})
        elif by_layer or by_lt:
            notes.append(f"non-straight entity on bend layer/linetype ignored ({t} {layer}/{linetype})")
        else:
            curves.append({"poly": poly, "layer": layer, "straight": straight})
    if not curves:
        raise FlatParseError("NO_OUTLINE", "the DXF has no outline geometry")
    extent = float(np.ptp(np.concatenate([c["poly"] for c in curves]), axis=0).max())
    tol = max(1e-3, 1e-5 * extent)
    # annotation-only bend lines: a straight solid line next to a bend annotation
    parsed_ann = [(txt, pos, _parse_annotation(txt)) for txt, pos in annotations]
    parsed_ann = [(txt, pos, pa) for txt, pos, pa in parsed_ann if pa is not None]
    reach = max(15.0, 0.05 * extent)
    for txt, pos, pa in parsed_ann:
        if any(_seg_dist(pos, c["a"], c["b"]) <= reach for c in bend_candidates):
            continue
        best = None
        for idx, c in enumerate(curves):
            if not c["straight"]:
                continue
            d = _seg_dist(pos, c["poly"][0], c["poly"][-1])
            if d <= reach and (best is None or d < best[0]):
                best = (d, idx)
        if best is not None:
            c = curves.pop(best[1])
            bend_candidates.append({"a": c["poly"][0], "b": c["poly"][-1], "layer": c["layer"],
                                    "linetype": "CONTINUOUS", "source": "ANNOTATION"})
    loops, left = _chain_loops([c["poly"] for c in curves], tol)
    if left:
        notes.append(f"{left} open outline chains ignored")
    loops = [_dedupe(lp, tol) for lp in loops]
    loops = [lp for lp in loops if len(lp) >= 3 and abs(polygon_area_2d(lp)) > tol * tol]
    if not loops:
        raise FlatParseError("NO_CLOSED_OUTLINE", "outline entities do not form a closed loop")
    areas = [abs(polygon_area_2d(lp)) for lp in loops]
    oi = int(np.argmax(areas))
    outer = loops[oi]
    inner = []
    for i, lp in enumerate(loops):
        if i == oi:
            continue
        if _point_in_poly(lp.mean(axis=0), outer) or _point_in_poly(lp[0], outer):
            inner.append(lp)
        else:
            notes.append("a closed loop outside the outer outline was ignored")
    holes = []
    for lp in inner:
        centre = lp.mean(axis=0)
        holes.append((centre, float(np.max(np.linalg.norm(lp - centre, axis=1)))))
    lines = []
    marks = 0
    for c in bend_candidates:
        a, b = np.asarray(c["a"], dtype=np.float64), np.asarray(c["b"], dtype=np.float64)
        if np.linalg.norm(b - a) <= 10 * tol:
            continue
        length, mid = float(np.linalg.norm(b - a)), 0.5 * (a + b)
        if any(float(np.linalg.norm(mid - centre)) <= 1.5 * r and length <= 4.0 * r for centre, r in holes):
            marks += 1  # a hole's centre mark on a centre/dashed linetype, not a bend line
            continue
        direction, angle, text = None, None, None
        best = None
        for txt, pos, pa in parsed_ann:
            d = _seg_dist(pos, a, b)
            if d <= reach and (best is None or d < best[0]):
                best = (d, txt, pa)
        if best is not None:
            text, (direction, angle) = best[1], best[2]
        if direction is None and re.search(r"UP", c["layer"], re.IGNORECASE):
            direction = "UP"
        elif direction is None and re.search(r"DOWN|DN", c["layer"], re.IGNORECASE):
            direction = "DOWN"
        lines.append(FlatBendLine("", a, b, c["source"], c["layer"], c["linetype"], direction, angle, text))
    if marks:
        notes.append(f"{marks} hole centre marks on bend layers/linetypes ignored")
    lines.sort(key=lambda ln: (round(float(ln.midpoint[0]), 3), round(float(ln.midpoint[1]), 3)))
    for i, ln in enumerate(lines, start=1):
        ln.line_id = f"L{i}"
    return ParsedFlat("DXF", units, scale, outer, inner, lines, None, notes)


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    s = 0.0 if denom <= 1e-18 else min(1.0, max(0.0, float((p - a) @ ab) / denom))
    return float(np.linalg.norm(p - (a + s * ab)))


# --------------------------------------------------------------------------- STEP


def _parse_flat_step(path: Path) -> ParsedFlat:
    from domains.production.engine.compile.brep import CadReadError, load_step

    try:
        model = load_step(path)
    except CadReadError as exc:
        raise FlatParseError("FLAT_STEP_UNREADABLE", f"{exc.reason}: {exc.detail}") from exc
    except Exception as exc:
        raise FlatParseError("FLAT_STEP_UNREADABLE", f"{type(exc).__name__}: {exc}"[:300]) from exc
    planes = [f for f in model.faces if f.kind == "plane"]
    if not planes:
        raise FlatParseError("FLAT_STEP_NOT_PLANAR", "no planar face")
    top = max(planes, key=lambda f: (f.area, -f.index))
    n = top.normal
    partners = [f for f in planes if f is not top and float(f.normal @ n) < -math.cos(math.radians(0.5))]
    if not partners:
        raise FlatParseError("FLAT_STEP_NOT_PLANAR", "no opposite skin parallel to the largest face")
    bottom = max(partners, key=lambda f: (f.area, -f.index))
    thickness = abs(float((bottom.centroid - top.centroid) @ n))
    if abs(bottom.area - top.area) > 0.05 * top.area:
        raise FlatParseError("FLAT_STEP_NOT_PLANAR", "the two largest planar faces are not the skins of one plate")
    for f in model.faces:
        if f.kind != "plane" or f is top or f is bottom:
            continue
        if abs(float(f.normal @ n)) > math.cos(math.radians(0.5)) and f.area > 0.01 * top.area:
            raise FlatParseError("FLAT_STEP_NOT_PLANAR", "a third large face parallel to the skins (not a flat plate)")
    x_dir = top.x_dir if top.x_dir is not None else np.array([1.0, 0.0, 0.0])
    x_dir = x_dir - float(x_dir @ n) * n
    x_dir = x_dir / np.linalg.norm(x_dir)
    y_dir = np.cross(n, x_dir)
    origin = top.origin
    loops2d = []
    outer = None
    for loop in top.loops:
        pts = np.concatenate([u.xyz[:-1] for u in loop.uses if len(u.xyz) >= 2])
        rel = pts - origin
        p2 = np.stack([rel @ x_dir, rel @ y_dir], axis=1)
        p2 = _dedupe(p2, 1e-6)
        if len(p2) >= 3:
            loops2d.append(p2)
            if loop.is_outer:
                outer = p2
    if outer is None:
        raise FlatParseError("NO_CLOSED_OUTLINE", "flat STEP top face has no outer loop")
    inner = [lp for lp in loops2d if lp is not outer]
    return ParsedFlat("STEP", "mm (STEP model units)", 1.0, outer, inner, [], thickness,
                      [f"flat STEP: top face {top.index}, thickness {thickness:.4f} mm"])
