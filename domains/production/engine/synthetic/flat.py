"""Flat pattern (DXF) and synthetic drawing (PDF) for E2 parts.

DXF conventions (ezdxf, R2010, ``$INSUNITS = 4`` mm, ``$MEASUREMENT = 1``),
viewed from the flat frame's +Z (skin 0) side:

- layer ``OUTLINE``: the blank's outer loop as one closed ``LWPOLYLINE`` and
  every hole as a ``CIRCLE`` (inner loops);
- layer ``BEND``: one ``LINE`` per bend (linetype ``DASHED``), the bend line
  = centre of the bend-allowance strip, spanning the bend; plus one ``TEXT``
  annotation per bend at its midpoint, ``"UP 90.00° R2.00"``. The annotated
  angle is the **bend (sweep) angle** as most CAD flat exports print it;
  included = 180 - sweep. ``UP`` means towards the viewer (flat +Z).

The outline is the union of the stage-0 panel polygons, produced by an Euler
tour of the panel tree (every bend strip is full-width between its two
attachment edges), with collinear vertices merged. Output bytes are
deterministic (ezdxf's fixed-metadata switch is set only while writing).

The drawing PDF (ReportLab, ``invariant=1``) shows the flat pattern with
numbered bend lines and a bend table; it exists so onboarding tests have a
drawing to upload. It is labelled SYNTHETIC.
"""
from __future__ import annotations

import io
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import FREE, UP, FoldedPart, FormState, GeometryError, PartDesign

__all__ = [
    "FlatPattern",
    "flat_pattern",
    "write_flat_dxf",
    "read_flat_dxf",
    "write_drawing_pdf",
    "DXF_OUTLINE_LAYER",
    "DXF_BEND_LAYER",
]

DXF_OUTLINE_LAYER = "OUTLINE"
DXF_BEND_LAYER = "BEND"


@dataclass(frozen=True)
class FlatPattern:
    """Flat-frame geometry of the blank (mm)."""

    outline: np.ndarray  # (N, 2) CCW outer loop
    holes: tuple[tuple[float, float, float], ...]  # (cx, cy, r)
    bend_lines: tuple[tuple[str, np.ndarray, np.ndarray, str, float, float], ...]  # (key, p0, p1, dir, sweep, Ri)

    @property
    def area(self) -> float:
        x, y = self.outline[:, 0], self.outline[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) - sum(math.pi * h[2] ** 2 for h in self.holes)


def _merge_collinear(pts: np.ndarray) -> np.ndarray:
    out = list(pts)
    changed = True
    while changed and len(out) > 3:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
            ab, bc = b - a, c - b
            if np.linalg.norm(ab) < 1e-9:
                del out[i]
                changed = True
                break
            cross = ab[0] * bc[1] - ab[1] * bc[0]
            if abs(cross) < 1e-9 * max(1.0, np.linalg.norm(ab) * np.linalg.norm(bc)) and float(ab @ bc) > 0:
                del out[i]
                changed = True
                break
    return np.asarray(out)


def flat_pattern(design: PartDesign) -> FlatPattern:
    """The blank (stage 0) of ``design`` in the flat frame."""
    fp0 = FoldedPart(design, FormState.nominal(design, formed=()), datum=design.default_datum())
    partner: dict[tuple[str, str], tuple[str, int]] = {}
    for b in design.bends:
        ia = fp0.tags[b.panel_a].index(b.key)
        ib = fp0.tags[b.panel_b].index(b.key)
        partner[(b.panel_a, b.key)] = (b.panel_b, ib)
        partner[(b.panel_b, b.key)] = (b.panel_a, ia)

    def walk(panel: str, skip_edge: int | None, depth: int) -> list[np.ndarray]:
        if depth > len(design.panels) + 1:
            raise GeometryError("flat tour did not terminate")
        pts, tags = fp0.outline[panel], fp0.tags[panel]
        n = len(pts)
        idxs = list(range(n)) if skip_edge is None else [(skip_edge + 1 + k) % n for k in range(n - 1)]
        out: list[np.ndarray] = []
        for c, i in enumerate(idxs):
            if not (skip_edge is not None and c == 0):
                out.append(pts[i])
            tag = tags[i]
            if tag != FREE:
                other, j = partner[(panel, tag)]
                out.extend(walk(other, j, depth + 1))
        return out

    loop = _merge_collinear(np.asarray(walk(fp0.datum, None, 0)))
    x, y = loop[:, 0], loop[:, 1]
    if 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) <= 0:
        raise GeometryError("flat outline is not counter-clockwise")
    holes = tuple(
        (float(h.center[0]), float(h.center[1]), float(h.radius)) for p in design.panels for h in p.holes
    )
    lines = []
    for b in design.bends:
        lp = np.asarray(b.line_point, dtype=float)
        d = np.asarray(b.line_dir, dtype=float)
        lines.append((b.key, lp + b.extent[0] * d, lp + b.extent[1] * d, b.direction, b.sweep_deg, b.inner_radius))
    return FlatPattern(loop, holes, tuple(lines))


class _FixedEzdxfMeta:
    """Context manager: ezdxf writes fixed GUIDs/timestamps (deterministic bytes)."""

    def __enter__(self):
        import ezdxf

        self._old = ezdxf.options.write_fixed_meta_data_for_testing
        ezdxf.options.write_fixed_meta_data_for_testing = True

    def __exit__(self, *exc):
        import ezdxf

        ezdxf.options.write_fixed_meta_data_for_testing = self._old
        return False


def write_flat_dxf(design: PartDesign, path: str | os.PathLike) -> str:
    """Write the flat DXF of ``design``; returns its sha256."""
    import hashlib

    import ezdxf

    pattern = flat_pattern(design)
    with _FixedEzdxfMeta():
        data = _build_dxf_bytes(ezdxf, pattern)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    return hashlib.sha256(data).hexdigest()


def _build_dxf_bytes(ezdxf, pattern: FlatPattern) -> bytes:
    doc = ezdxf.new("R2010", setup=True)
    # ezdxf adds CLASS records at write time while iterating a set of entity types,
    # whose order follows Python's per-process string hashing; a sorted list makes
    # the bytes independent of PYTHONHASHSEED (this document only).
    entitydb = doc.entitydb
    entitydb.dxf_types_in_use = lambda: sorted({e.dxftype() for e in entitydb.values()})  # type: ignore[method-assign]
    doc.units = 4
    doc.header["$MEASUREMENT"] = 1
    doc.header["$INSUNITS"] = 4
    doc.layers.add(DXF_OUTLINE_LAYER, color=7)
    doc.layers.add(DXF_BEND_LAYER, color=1, linetype="DASHED")
    msp = doc.modelspace()
    msp.add_lwpolyline([(float(px), float(py)) for px, py in pattern.outline], close=True, dxfattribs={"layer": DXF_OUTLINE_LAYER})
    for cx, cy, r in pattern.holes:
        msp.add_circle((cx, cy), r, dxfattribs={"layer": DXF_OUTLINE_LAYER})
    size = float(np.ptp(pattern.outline, axis=0).max())
    height = float(min(6.0, max(2.0, size / 80.0)))
    for key, p0, p1, direction, sweep, ri in pattern.bend_lines:
        msp.add_line((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])), dxfattribs={"layer": DXF_BEND_LAYER, "linetype": "DASHED"})
        mid = 0.5 * (p0 + p1)
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        normal = np.array([-d[1], d[0]])
        pos = mid + normal * (0.8 * height)
        text = f"{'UP' if direction == UP else 'DOWN'} {sweep:.2f}° R{ri:.2f}"
        rot = math.degrees(math.atan2(d[1], d[0]))
        msp.add_text(text, dxfattribs={"layer": DXF_BEND_LAYER, "height": height, "rotation": rot}).set_placement((float(pos[0]), float(pos[1])))
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode("utf-8")


def read_flat_dxf(path: str | os.PathLike) -> dict[str, object]:
    """Parse a DXF written by :func:`write_flat_dxf` (used by tests and tools).

    Returns ``{"outline": (N,2) array, "holes": [(cx,cy,r)], "bend_lines":
    [(p0, p1)], "annotations": [(text, (x, y))], "units": int}``."""
    import ezdxf

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    outline = None
    holes = []
    lines = []
    notes = []
    for e in msp:
        layer = e.dxf.layer
        if e.dxftype() == "LWPOLYLINE" and layer == DXF_OUTLINE_LAYER:
            outline = np.asarray([(p[0], p[1]) for p in e.get_points("xy")])
        elif e.dxftype() == "CIRCLE" and layer == DXF_OUTLINE_LAYER:
            holes.append((float(e.dxf.center.x), float(e.dxf.center.y), float(e.dxf.radius)))
        elif e.dxftype() == "LINE" and layer == DXF_BEND_LAYER:
            lines.append((np.array([e.dxf.start.x, e.dxf.start.y]), np.array([e.dxf.end.x, e.dxf.end.y])))
        elif e.dxftype() == "TEXT" and layer == DXF_BEND_LAYER:
            notes.append((e.dxf.text, (float(e.dxf.insert.x), float(e.dxf.insert.y))))
    if outline is None:
        raise GeometryError("DXF has no outline polyline")
    return {"outline": outline, "holes": holes, "bend_lines": lines, "annotations": notes, "units": int(doc.header.get("$INSUNITS", 0))}


def write_drawing_pdf(
    design: PartDesign,
    path: str | os.PathLike,
    *,
    sku: str,
    revision: str,
    part_name: str,
    bend_rows: Sequence[dict[str, object]],
) -> str:
    """A one-page synthetic drawing (flat pattern + bend table); returns its sha256.

    ``bend_rows``: per bend in sequence order, keys ``label``, ``key``,
    ``direction``, ``included_deg``, ``sweep_deg``, ``inner_radius_mm``,
    ``move`` (group number or ``""``), ``flange_a_mm``, ``flange_b_mm``."""
    import hashlib

    from reportlab.lib.pagesizes import A3, landscape
    from reportlab.pdfgen import canvas

    pattern = flat_pattern(design)
    buf = io.BytesIO()
    page_w, page_h = landscape(A3)
    c = canvas.Canvas(buf, pagesize=(page_w, page_h), invariant=1)
    c.setTitle(f"{sku} rev {revision} synthetic drawing")
    c.setAuthor("sherman_qc_e2_synthetic")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, page_h - 40, f"{sku}  REV {revision}  -  {part_name}")
    c.setFont("Helvetica", 10)
    c.drawString(40, page_h - 58, f"SYNTHETIC TEST PART (E2) - NOT A PRODUCTION DRAWING.  t = {design.thickness:g} mm, K = {design.k_factor:g}, units mm / deg.")
    c.drawString(40, page_h - 72, "Angles: included angle (180 = flat); flat bend notes print the bend (sweep) angle. Flange lengths are OML (outside mould line).")
    # flat pattern view
    box = (40, 200, page_w * 0.62, page_h - 100)
    lo = pattern.outline.min(axis=0)
    hi = pattern.outline.max(axis=0)
    scale = min((box[2] - box[0]) / max(hi[0] - lo[0], 1e-6), (box[3] - box[1]) / max(hi[1] - lo[1], 1e-6))

    def tx(p):
        return box[0] + (p[0] - lo[0]) * scale, box[1] + (p[1] - lo[1]) * scale

    path_obj = c.beginPath()
    first = tx(pattern.outline[0])
    path_obj.moveTo(*first)
    for p in pattern.outline[1:]:
        path_obj.lineTo(*tx(p))
    path_obj.close()
    c.setLineWidth(1.0)
    c.drawPath(path_obj, stroke=1, fill=0)
    for cx, cy, r in pattern.holes:
        x, y = tx((cx, cy))
        c.circle(x, y, r * scale, stroke=1, fill=0)
    label_of = {str(row["key"]): str(row["label"]) for row in bend_rows}
    c.setDash(6, 3)
    c.setFont("Helvetica-Bold", 9)
    for key, p0, p1, direction, sweep, ri in pattern.bend_lines:
        a, b = tx(p0), tx(p1)
        c.line(a[0], a[1], b[0], b[1])
        c.drawString(0.5 * (a[0] + b[0]) + 3, 0.5 * (a[1] + b[1]) + 3, label_of.get(key, key))
    c.setDash()
    # bend table
    c.setFont("Helvetica-Bold", 10)
    x0, y = page_w * 0.66, page_h - 110
    headers = ["Bend", "Dir", "Incl.", "Sweep", "Ri", "Move", "A (OML)", "B (OML)"]
    widths = [40, 40, 50, 50, 40, 40, 60, 60]
    xx = x0
    for h, w in zip(headers, widths):
        c.drawString(xx, y, h)
        xx += w
    c.setFont("Helvetica", 10)
    for row in bend_rows:
        y -= 16
        vals = [
            str(row["label"]),
            str(row["direction"]),
            f"{float(row['included_deg']):.2f}",
            f"{float(row['sweep_deg']):.2f}",
            f"{float(row['inner_radius_mm']):.2f}",
            str(row.get("move", "")),
            "" if row.get("flange_a_mm") is None else f"{float(row['flange_a_mm']):.3f}",
            "" if row.get("flange_b_mm") is None else f"{float(row['flange_b_mm']):.3f}",
        ]
        xx = x0
        for v, w in zip(vals, widths):
            c.drawString(xx, y, v)
            xx += w
    c.showPage()
    c.save()
    data = buf.getvalue()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    return hashlib.sha256(data).hexdigest()
