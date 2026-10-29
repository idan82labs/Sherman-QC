"""E2 truth against the files E2 writes: STEP BReps, the flat DXF and ZEISS-like names.

Plan §4 E2 asks for "truth exactness on built STEP (measure your own BRep
analytically to cross-check generator)" and "flat dev length consistency".
This module reads generated case directories back from disk and re-derives
every truth number it checks from the written files, with code that shares
nothing with the generator's geometry model (``geometry.py`` / ``analytic.py``):

- **STEP** (``part.step``, ``as_built.step``, ``stages/S<k>.step``). A bend is a
  coaxial pair of cylinder faces of radii Ri and Ri + t; its OML virtual sharp is
  the intersection of the two planes tangent to the outer cylinder; its span is
  the outer cylinder's extent along that line. A flange (or backgauge) length is
  measured per station (41 stations on the span trimmed by min(2 mm, 5 %), the
  contract rule) as the farthest crossing of the station line with the outer
  wire of the side's planar OML face. The exit edge gives the boundary kind:
  shared with a bend cylinder -> ADJACENT_BEND, measured to that bend's sharp
  (projected into the face plane); otherwise FREE_EDGE. The value is the median
  over the stations of the majority kind. On a stage model the side face spans
  the whole flat island (coplanar faces are unified), so the same measurement
  is the backgauge.
- **Bend-line position**: the as-built sharp (``as_built.step``) against the
  reference stage model's sharp (``stages/S<k>.step``), at the reference
  mid-span, as a 3D offset vector and along side A's measured outward direction.
- **DXF** (``flat.dxf``, read with ezdxf only): per station, walking from the
  bend line across the blank, the first formed bend line crossed (ADJACENT) or
  the farthest outline exit (FREE). Folded OML length = flat distance + BD/2 for
  each bend at either end, ``BD = 2 Ro tan(a/2) - a (Ri + K t)`` with the angle
  and radius parsed from the DXF bend notes.

Truth is used only to *identify* things before the comparison: bends (each truth
OML sharp line must coincide with a measured sharp within 1e-6 mm, which checks
those lines too) and which tangent plane is side A or B (truth's own panel
transforms map the side's outward direction into the measured pose). Mutation
tests show that 1e-3 errors in truth, a wrong STEP file, a 0.01 mm shift of a
DXF bend line and a wrong bend note are each reported.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Line, GeomAbs_Plane
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

from domains.production.engine.synthetic.brep import read_step
from domains.production.engine.synthetic.cases import (
    FAMILY_CODES,
    REVISIONS,
    TIERS,
    case_seed,
    generate_case,
    opaque_sku,
    plan_cases,
    zeiss_scan_filename,
)
from domains.production.engine.synthetic.families import FAMILIES
from domains.production.engine.synthetic.injection import Scenario
from domains.production.recipe.sku import parse_scan_filename

#: Length / angle agreement required between truth and the written files.
TOL_MM = 1e-6
TOL_DEG = 1e-6
_JITTER = 1e-6 * (math.sqrt(5.0) - 1.0) / 2.0  # the contract stations' sub-micron offset (never on a vertex)

#: (case id in the DEV ``full`` plan, what it exercises). Together: all seven
#: families; angle / flange / bend-line-shift errors, springback, prefix stages
#: with a backgauge (also on notched and partial-span islands), a partial bend
#: and an unplanned crease.
CASES = {
    "E2-DEV-LB-001": "complete part, injected errors, acute bend, mesh scan",
    "E2-DEV-UC-003": "stage 1: backgauge across the unformed bend, short flange",
    "E2-DEV-ZS-007": "complete part, springback on a machine-move group",
    "E2-DEV-HT-005": "stage k: backgauge, bend-line shift on the just-formed bend",
    "E2-DEV-CR-011": "partially formed bend (prefix stage)",
    "E2-DEV-TD-008": "two-direction part with errors, Ri >= 4t",
    "E2-DEV-TD-023": "two-direction part at stage k (backgauge across the tab)",
    "E2-DEV-PN-010": "panel with holes and notches at stage k, acute bend",
    "E2-DEV-PN-014": "unplanned crease splitting a flange",
}


# ---------------------------------------------------------------------------
# station rule (ENGINE_IO §1, restated here)
# ---------------------------------------------------------------------------


def _stations(s0: float, s1: float, n: int = 41, trim_mm: float = 2.0, trim_fraction: float = 0.05) -> np.ndarray:
    tau = min(trim_mm, trim_fraction * (s1 - s0))
    a, b = s0 + tau, s1 - tau
    return a + (np.arange(n) + 0.5) / n * (b - a) + _JITTER


def _majority(rows: list[tuple[str, str | None, float]]) -> dict[str, Any]:
    """Majority (kind, adjacent bend) group of per-station rows and its statistics."""
    if not rows:
        return {"kind": "UNDEFINED", "adjacent": None, "fraction": 0.0, "median": None, "min": None, "max": None, "stations": 0}
    counts = Counter((k, a) for k, a, _ in rows)
    (kind, adj), count = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1] or ""))[0]
    vals = np.asarray([v for k, a, v in rows if (k, a) == (kind, adj)])
    return {"kind": kind, "adjacent": adj, "fraction": count / len(rows), "median": float(np.median(vals)),
            "min": float(vals.min()), "max": float(vals.max()), "stations": len(rows)}


# ---------------------------------------------------------------------------
# STEP measurement
# ---------------------------------------------------------------------------


def _vec3(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()])


def _edges(shape) -> list:
    out = []
    ex = TopExp_Explorer(shape, TopAbs_EDGE)
    while ex.More():
        out.append(TopoDS.Edge_s(ex.Current()))
        ex.Next()
    return out


def _edge_points(edge) -> tuple[Any, np.ndarray, np.ndarray]:
    c = BRepAdaptor_Curve(edge)
    return c.GetType(), _vec3(c.Value(c.FirstParameter())), _vec3(c.Value(c.LastParameter()))


def _dist_point_line(p: np.ndarray, a: np.ndarray, d: np.ndarray) -> float:
    v = p - a
    return float(np.linalg.norm(v - (v @ d) * d))


class StepModel:
    """A STEP solid measured from its faces (see the module docstring)."""

    def __init__(self, path: Path, thickness: float) -> None:
        self.shape = read_step(path)
        self.faces: list[dict[str, Any]] = []
        ex = TopExp_Explorer(self.shape, TopAbs_FACE)
        while ex.More():
            f = TopoDS.Face_s(ex.Current())
            s = BRepAdaptor_Surface(f)
            rec: dict[str, Any] = {"face": f, "type": s.GetType()}
            if rec["type"] == GeomAbs_Plane:
                pl = s.Plane()
                n = _vec3(pl.Axis().Direction())
                rec.update(point=_vec3(pl.Location()), normal=-n if f.Orientation() == TopAbs_REVERSED else n)
            elif rec["type"] == GeomAbs_Cylinder:
                cy = s.Cylinder()
                rec.update(axis_point=_vec3(cy.Axis().Location()), axis_dir=_vec3(cy.Axis().Direction()), radius=cy.Radius())
            self.faces.append(rec)
            ex.Next()
        self.emap = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(self.shape, TopAbs_EDGE, TopAbs_FACE, self.emap)
        cyls = [f for f in self.faces if f["type"] == GeomAbs_Cylinder]
        self.bends: list[dict[str, Any]] = []
        for outer in cyls:
            inner = [c for c in cyls if c is not outer and abs(outer["radius"] - c["radius"] - thickness) < 1e-7
                     and abs(abs(c["axis_dir"] @ outer["axis_dir"]) - 1.0) < 1e-9
                     and _dist_point_line(c["axis_point"], outer["axis_point"], outer["axis_dir"]) < 1e-6]
            if len(inner) != 1:
                continue  # a hole wall, or the inner cylinder of a pair
            planes = self._tangent_planes(outer)
            assert len(planes) == 2, f"a bend cylinder has {len(planes)} tangent OML planes"
            n1, n2 = planes[0]["normal"], planes[1]["normal"]
            d = np.cross(n1, n2)
            d /= np.linalg.norm(d)
            p = np.linalg.solve(np.array([n1, n2, d]), np.array([n1 @ planes[0]["point"], n2 @ planes[1]["point"], d @ planes[0]["point"]]))
            verts = np.asarray([q for e in _edges(outer["face"]) for q in _edge_points(e)[1:]])
            self.bends.append({"outer": outer, "inner": inner[0], "planes": planes, "p": p, "d": d,
                               "included": 180.0 - math.degrees(math.acos(max(-1.0, min(1.0, float(n1 @ n2))))),
                               "verts": verts, "id": None})

    def _tangent_planes(self, cyl: dict[str, Any]) -> list[dict[str, Any]]:
        planes: list[dict[str, Any]] = []
        for e in _edges(cyl["face"]):
            for nb in self.emap.FindFromKey(e):
                f = TopoDS.Face_s(nb)
                if f.IsSame(cyl["face"]):
                    continue
                rec = next(x for x in self.faces if x["face"].IsSame(f))
                if rec["type"] == GeomAbs_Plane and abs(rec["normal"] @ cyl["axis_dir"]) < 1e-9 and not any(r["face"].IsSame(f) for r in planes):
                    planes.append(rec)
        return planes

    def identify(self, lines: dict[str, dict[str, list[float]]]) -> list[str]:
        """Name the measured bends by truth OML sharp lines (id -> {start_mm, end_mm});
        orient each sharp along its truth line and set its span. Returns problems."""
        problems = []
        for bid, line in lines.items():
            a, b = np.asarray(line["start_mm"]), np.asarray(line["end_mm"])
            hits = [m for m in self.bends if max(_dist_point_line(a, m["p"], m["d"]), _dist_point_line(b, m["p"], m["d"])) < TOL_MM]
            if len(hits) != 1:
                problems.append(f"{bid}: truth OML sharp matches {len(hits)} measured bends")
                continue
            m = hits[0]
            m["id"] = bid
            if m["d"] @ (b - a) < 0:
                m["d"] = -m["d"]
            s = (m["verts"] - m["p"]) @ m["d"]
            m["span"] = (float(s.min()), float(s.max()))
            for end, name in ((a, "start"), (b, "end")):  # the truth line spans exactly the bend
                if abs((end - m["p"]) @ m["d"] - (m["span"][0] if name == "start" else m["span"][1])) > TOL_MM:
                    problems.append(f"{bid}: truth sharp {name} is not at the measured bend end")
        unnamed = [m for m in self.bends if m["id"] is None]
        if unnamed:
            problems.append(f"{len(unnamed)} measured bend(s) have no truth bend")
        return problems

    def bend(self, bid: str) -> dict[str, Any]:
        return next(m for m in self.bends if m["id"] == bid)

    def _owner(self, edge) -> dict[str, Any] | None:
        for nb in self.emap.FindFromKey(edge):
            f = TopoDS.Face_s(nb)
            for m in self.bends:
                if f.IsSame(m["outer"]["face"]) or f.IsSame(m["inner"]["face"]):
                    return m
        return None

    def side(self, bid: str, outward: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        """The tangent OML plane containing ``outward`` and its measured outward unit vector."""
        m = self.bend(bid)
        cands = [pl for pl in m["planes"] if abs(pl["normal"] @ outward) < 1e-6]
        assert len(cands) == 1, f"{bid}: {len(cands)} tangent planes contain the side direction"
        plane = cands[0]
        u = np.cross(m["d"], plane["normal"])
        u /= np.linalg.norm(u)
        wire_pts = np.asarray([q for e in _edges(BRepTools.OuterWire_s(plane["face"])) for q in _edge_points(e)[1:]])
        if float(np.max((wire_pts - m["p"]) @ u)) < 0:  # the face lies on the side's outward half-plane
            u = -u
        return plane, u

    def far_boundary(self, bid: str, outward: np.ndarray) -> dict[str, Any]:
        """Per-station farthest crossing on the side face (module docstring)."""
        m = self.bend(bid)
        plane, u = self.side(bid, outward)
        p0, d = m["p"], m["d"]
        segs = []
        for e in _edges(BRepTools.OuterWire_s(plane["face"])):
            typ, q0, q1 = _edge_points(e)
            assert typ == GeomAbs_Line, "outer wires of sheet faces are polygons"
            segs.append(((q0 - p0) @ d, (q0 - p0) @ u, (q1 - p0) @ d, (q1 - p0) @ u, e))
        rows: list[tuple[str, str | None, float]] = []
        for v in _stations(*m["span"]):
            best = None
            for x0, y0, x1, y1, e in segs:
                if (x0 >= v) != (x1 >= v):  # half-open crossing rule
                    lam = y0 + (v - x0) * (y1 - y0) / (x1 - x0)
                    if best is None or lam > best[0]:
                        best = (lam, e)
            if best is None:
                continue
            lam, edge = best
            other = self._owner(edge)
            if other is None:
                rows.append(("FREE_EDGE", None, float(lam)))
                continue
            a0, a1 = float((other["p"] - p0) @ d), float(other["d"] @ d)
            b0, b1 = float((other["p"] - p0) @ u), float(other["d"] @ u)
            if abs(a1) < 1e-12:
                continue
            rows.append(("ADJACENT_BEND", other["id"], b0 + (v - a0) / a1 * b1))
        return _majority(rows)


# ---------------------------------------------------------------------------
# truth helpers
# ---------------------------------------------------------------------------


def _rotation(m: list[list[float]]) -> np.ndarray:
    return np.asarray(m, dtype=float)[:3, :3]


def _stage_index(truth: dict[str, Any]) -> int | None:
    sid = truth["stage"]["expected_stage_model_id"]
    return None if sid is None else int(sid[1:])


def _out_in_pose(truth: dict[str, Any], bend: dict[str, Any], letter: str, pose: str, k: int | None) -> np.ndarray:
    """Side ``letter``'s outward direction (final pose in truth) mapped into a pose
    through truth's own panel transforms: ``final``, ``stage`` (S_k) or ``actual``
    (the as-built part, whose reference is S_k: a prefix-stage case only)."""
    side = bend["nominal"]["sides"][letter]
    out = np.asarray(side["outward_dir"])
    if pose == "final":
        return out
    if k is None:
        raise ValueError("a stage or as-built pose needs a prefix-stage case (truth.stage.expected_stage_model_id)")
    to_stage = _rotation(truth["stage_models_nominal"][k]["panel_transforms_final_to_stage"][side["panel"]])
    out = to_stage @ out
    if pose == "stage":
        return out
    return _rotation(truth["actual_panel_transforms_reference_to_actual"][side["panel"]]) @ out


def _compare(problems: list[str], what: str, measured: dict[str, Any], value: float | None, stations: dict[str, Any] | None) -> None:
    if value is None:
        return
    if measured["median"] is None or abs(measured["median"] - value) > TOL_MM:
        problems.append(f"{what}: measured {measured['median']} vs truth {value}")
    if stations is not None:
        adj = stations["groups"][0]["adjacent_bend"] if stations["groups"] else None
        if (measured["kind"], measured["adjacent"]) != (stations["boundary_kind"], adj):
            problems.append(f"{what}: measured boundary {measured['kind']}/{measured['adjacent']} vs truth {stations['boundary_kind']}/{adj}")
        if abs(measured["fraction"] - stations["kind_fraction"]) > 1e-12 or measured["stations"] != stations["valid_stations"]:
            problems.append(f"{what}: measured {measured['fraction']:.4f} of {measured['stations']} stations vs truth "
                            f"{stations['kind_fraction']:.4f} of {stations['valid_stations']}")


def check_nominal_step(truth: dict[str, Any], step: StepModel) -> list[str]:
    """part.step against truth nominals: sharps, included angles, flange lengths."""
    problems = step.identify({b["bend_uid"]: b["nominal"]["loci_final"]["oml_virtual_sharp"] for b in truth["bends"]})
    if problems:
        return problems
    for b in truth["bends"]:
        uid = b["bend_uid"]
        if abs(step.bend(uid)["included"] - b["nominal"]["included_deg"]) > TOL_DEG:
            problems.append(f"{uid}: nominal included {step.bend(uid)['included']} vs {b['nominal']['included_deg']}")
        for letter in "AB":
            c = truth["characteristics"][f"{uid}/FLANGE_LENGTH/{letter}"]
            got = step.far_boundary(uid, _out_in_pose(truth, b, letter, "final", None))
            _compare(problems, f"{uid}/{letter} nominal flange", got, c["nominal"], c["nominal_stations"])
    return problems


def check_as_built_step(truth: dict[str, Any], step: StepModel) -> list[str]:
    """as_built.step against truth actuals: formed set, angles, flange truths, backgauge truth."""
    lines = {b["bend_uid"]: b["actual"]["loci"]["oml_virtual_sharp"] for b in truth["bends"] if "loci" in b["actual"]}
    lines.update({x["key"]: x["oml_virtual_sharp"] for x in truth["extra_bends"]})
    problems = step.identify(lines)
    if problems:
        return problems
    k = _stage_index(truth)
    for b in truth["bends"]:
        uid = b["bend_uid"]
        if "loci" not in b["actual"]:
            continue
        if abs(step.bend(uid)["included"] - b["actual"]["included_deg"]) > TOL_DEG:
            problems.append(f"{uid}: as-built included {step.bend(uid)['included']} vs {b['actual']['included_deg']}")
        for letter in "AB":
            c = truth["characteristics"][f"{uid}/FLANGE_LENGTH/{letter}"]
            if c["truth"] is None:
                continue
            if any(g["boundary_kind"] == "ADJACENT_BEND_NOT_FORMED" for g in c["truth_stations"]["groups"]):
                continue  # a still-flat bend leaves no edge in the BRep: such stations are not measurable here
            got = step.far_boundary(uid, _out_in_pose(truth, b, letter, "actual", k))
            _compare(problems, f"{uid}/{letter} as-built flange", got, c["truth"], c["truth_stations"])
    for x in truth["extra_bends"]:
        if abs(step.bend(x["key"])["included"] - x["included_deg"]) > TOL_DEG:
            problems.append(f"{x['key']}: crease included {step.bend(x['key'])['included']} vs {x['included_deg']}")
    bg = truth["backgauge"]
    if bg is not None:
        b = next(x for x in truth["bends"] if x["bend_uid"] == bg["bend_uid"])
        for letter, side in bg["sides"].items():
            got = step.far_boundary(bg["bend_uid"], _out_in_pose(truth, b, letter, "actual", k))
            _compare(problems, f"backgauge {letter} truth", got, side["truth_mm"], side["truth_stations"])
    return problems


def check_stage_step(truth: dict[str, Any], stage: StepModel, as_built: StepModel) -> list[str]:
    """stages/S<k>.step (the reference): stage bend lines, backgauge nominals, bend-line position."""
    k = _stage_index(truth)
    model = truth["stage_models_nominal"][k]
    problems = stage.identify({uid: model["bend_lines"][uid] for uid in model["formed_bend_uids"]})
    if problems:
        return problems
    bg = truth["backgauge"]
    if bg is not None:
        b = next(x for x in truth["bends"] if x["bend_uid"] == bg["bend_uid"])
        for letter, side in bg["sides"].items():
            got = stage.far_boundary(bg["bend_uid"], _out_in_pose(truth, b, letter, "stage", k))
            _compare(problems, f"backgauge {letter} nominal", got, side["nominal_mm"], side["nominal_stations"])
    for b in truth["bends"]:
        uid = b["bend_uid"]
        c = truth["characteristics"][f"{uid}/BEND_LINE_POSITION"]
        if c["truth"] is None:
            continue
        ref, act = stage.bend(uid), as_built.bend(uid)
        _, out_a = stage.side(uid, _out_in_pose(truth, b, "A", "stage", k))
        mid = ref["p"] + 0.5 * (ref["span"][0] + ref["span"][1]) * ref["d"]
        s = float((mid - act["p"]) @ ref["d"]) / float(act["d"] @ ref["d"])
        off = act["p"] + s * act["d"] - mid
        if abs(float(off @ out_a) - c["truth"]) > TOL_MM:
            problems.append(f"{uid}: bend-line position measured {float(off @ out_a)} vs truth {c['truth']}")
        if np.linalg.norm(off - np.asarray(b["actual"]["bend_line_position"]["offset_vector_mm"])) > TOL_MM:
            problems.append(f"{uid}: bend-line offset vector differs")
    return problems


# ---------------------------------------------------------------------------
# DXF development
# ---------------------------------------------------------------------------

_NOTE = re.compile(r"(UP|DOWN) (\d+(?:\.\d+)?)° R(\d+(?:\.\d+)?)")


def read_dxf(path: Path) -> dict[str, Any]:
    import ezdxf

    msp = ezdxf.readfile(str(path)).modelspace()
    outlines = [np.asarray([(p[0], p[1]) for p in e.get_points("xy")]) for e in msp.query("LWPOLYLINE") if e.dxf.layer == "OUTLINE"]
    lines = [(np.array([e.dxf.start.x, e.dxf.start.y]), np.array([e.dxf.end.x, e.dxf.end.y])) for e in msp.query("LINE") if e.dxf.layer == "BEND"]
    notes = [(e.dxf.text, np.array([e.dxf.insert.x, e.dxf.insert.y])) for e in msp.query("TEXT") if e.dxf.layer == "BEND"]
    assert len(outlines) == 1, "one outer loop"
    return {"outline": outlines[0], "lines": lines, "notes": notes}


def _bend_deduction(sweep_deg: float, ri: float, t: float, k: float) -> float:
    a = math.radians(sweep_deg)
    return 2.0 * (ri + t) * math.tan(a / 2.0) - a * (ri + k * t)


def _flat_dir(truth: dict[str, Any], bend: dict[str, Any], letter: str) -> np.ndarray:
    """The side's outward direction in the flat (DXF) frame: final -> S0 (truth panel
    transform) -> flat (inverse CAD placement)."""
    out_s0 = _out_in_pose(truth, bend, letter, "stage", 0)
    flat = _rotation(truth["part"]["cad_placement_flat_to_cad"]).T @ out_s0
    assert abs(flat[2]) < 1e-9
    return flat[:2] / np.linalg.norm(flat[:2])


def check_flat_dxf(truth: dict[str, Any], draft: dict[str, Any], dxf: dict[str, Any]) -> list[str]:
    """flat.dxf develops to truth: nominal flange lengths at the final pose and the
    nominal backgauge of a prefix-stage case (module docstring)."""
    problems: list[str] = []
    t, kf = truth["part"]["thickness_mm"], truth["part"]["k_factor"]
    by_uid: dict[str, dict[str, Any]] = {}
    for fl in draft["summary"]["flat_bend_lines"]:
        a, b = np.asarray(fl["start_xy_mm"]), np.asarray(fl["end_xy_mm"])
        hits = [ln for ln in dxf["lines"] if (np.linalg.norm(ln[0] - a) < TOL_MM and np.linalg.norm(ln[1] - b) < TOL_MM)
                or (np.linalg.norm(ln[0] - b) < TOL_MM and np.linalg.norm(ln[1] - a) < TOL_MM)]
        if len(hits) != 1:
            problems.append(f"{fl['bend_uid']}: {len(hits)} DXF bend lines at the drafted flat line")
            continue
        # the DXF's own full-precision coordinates (the draft rounds to 1e-6), in the drafted direction
        p, q = hits[0] if np.linalg.norm(hits[0][0] - a) < TOL_MM else hits[0][::-1]
        by_uid[fl["bend_uid"]] = {"a": p, "b": q}
    if problems or len(by_uid) != len(truth["bends"]) or len(dxf["lines"]) != len(truth["bends"]) or len(dxf["notes"]) != len(by_uid):
        return problems or [f"{len(dxf['lines'])} DXF bend lines and {len(dxf['notes'])} notes for {len(truth['bends'])} bends"]
    # each note sits next to its own line's midpoint: the assignment of least total distance
    # (a nearest-midpoint rule can pick a neighbour's note when two lines are a few mm apart)
    uids = sorted(by_uid)
    mids = [0.5 * (by_uid[u]["a"] + by_uid[u]["b"]) for u in uids]
    best = min(itertools.permutations(range(len(uids))),
               key=lambda perm: sum(float(np.linalg.norm(dxf["notes"][j][1] - mids[i])) for i, j in enumerate(perm)))
    for i, j in enumerate(best):
        text = dxf["notes"][j][0]
        m = _NOTE.fullmatch(text)
        if m is None:
            problems.append(f"{uids[i]}: unreadable bend note {text!r}")
            continue
        by_uid[uids[i]].update(sweep=float(m.group(2)), ri=float(m.group(3)), dir=m.group(1))
    if problems:
        return problems
    for b in truth["bends"]:
        info = by_uid[b["bend_uid"]]
        if abs(info["sweep"] - b["nominal"]["sweep_deg"]) > 1e-9 or abs(info["ri"] - b["nominal"]["inner_radius_mm"]) > 1e-9:
            problems.append(f"{b['bend_uid']}: DXF note {info['dir']} {info['sweep']} R{info['ri']} vs truth sweep/Ri")
        info["bd"] = _bend_deduction(info["sweep"], info["ri"], t, kf)
    outline = dxf["outline"]

    def develop(uid: str, letter: str, formed: set[str]) -> dict[str, Any]:
        info = by_uid[uid]
        bend = next(x for x in truth["bends"] if x["bend_uid"] == uid)
        u = _flat_dir(truth, bend, letter)
        d = (info["b"] - info["a"]) / np.linalg.norm(info["b"] - info["a"])
        rows: list[tuple[str, str | None, float]] = []
        for v in _stations(0.0, float(np.linalg.norm(info["b"] - info["a"]))):
            q = info["a"] + v * d
            exits = []
            for i in range(len(outline)):  # outline crossings beyond the bend line
                p0, p1 = outline[i], outline[(i + 1) % len(outline)]
                w0, w1 = (p0 - q) @ np.array([-u[1], u[0]]), (p1 - q) @ np.array([-u[1], u[0]])
                if (w0 >= 0.0) != (w1 >= 0.0):
                    lam = float((p0 - q) @ u + w0 / (w0 - w1) * ((p1 - p0) @ u))
                    if lam > 0:
                        exits.append(lam)
            crossings = []
            for other, oi in by_uid.items():
                if other == uid or other not in formed:
                    continue  # unformed bend lines are crossed (flat island)
                e = oi["b"] - oi["a"]
                den = u[0] * e[1] - u[1] * e[0]
                if abs(den) < 1e-12:
                    continue
                rel = oi["a"] - q
                lam = (rel[0] * e[1] - rel[1] * e[0]) / den
                s = (rel[0] * u[1] - rel[1] * u[0]) / den
                if lam > 0 and 0.0 <= s <= 1.0:
                    crossings.append((float(lam), other))
            if crossings:
                lam, other = min(crossings)
                rows.append(("ADJACENT_BEND", other, lam + 0.5 * info["bd"] + 0.5 * by_uid[other]["bd"]))
            elif exits:
                rows.append(("FREE_EDGE", None, max(exits) + 0.5 * info["bd"]))
        return _majority(rows)

    every = set(by_uid)
    for b in truth["bends"]:
        for letter in "AB":
            c = truth["characteristics"][f"{b['bend_uid']}/FLANGE_LENGTH/{letter}"]
            _compare(problems, f"{b['bend_uid']}/{letter} developed flange", develop(b["bend_uid"], letter, every),
                     c["nominal"], c["nominal_stations"])
    bg = truth["backgauge"]
    if bg is not None:
        formed = set(truth["stage_models_nominal"][bg["stage_index"]]["formed_bend_uids"])
        for letter, side in bg["sides"].items():
            _compare(problems, f"backgauge {letter} developed", develop(bg["bend_uid"], letter, formed),
                     side["nominal_mm"], side["nominal_stations"])
    return problems


# ---------------------------------------------------------------------------
# fixture and tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cases(tmp_path_factory) -> Path:
    """The CASES of the DEV full plan, with every STEP the generator can write
    (the scan density is lowered: the scan is not under test here)."""
    root = tmp_path_factory.mktemp("e2xcheck")
    specs = {s.case_id: s for s in plan_cases("full", "DEV")}
    for cid in CASES:
        generate_case(replace(specs[cid], point_target=8_000), root, write_as_built_step=True, write_stage_steps=True, ray_threads=1)
    return root


def _load(root: Path, cid: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    d = root / cid
    return (json.loads((d / "truth.json").read_text(encoding="utf-8")), json.loads((d / "case.json").read_text(encoding="utf-8")),
            json.loads((d / "recipe_draft.json").read_text(encoding="utf-8")))


def test_the_selected_cases_cover_what_they_claim(cases):
    seen = set()
    for cid in CASES:
        truth, case, _ = _load(cases, cid)
        seen.add(case["family"])
        kinds = {i["kind"] for i in truth["injections"]}
        if "ERRORS" in case["scenario"]:
            assert kinds & {"ANGLE", "FLANGE_LENGTH", "BEND_LINE_SHIFT"}, cid
        if case["scenario"] in ("STAGE_1", "STAGE_K"):
            assert truth["backgauge"] is not None and truth["backgauge"]["sides"], cid
    assert seen == set(FAMILIES)
    scen = {_load(cases, c)[1]["scenario"] for c in CASES}
    assert {"COMPLETE_ERRORS", "COMPLETE_SPRINGBACK", "STAGE_1", "STAGE_K", "PARTIAL_BEND", "EXTRA_BEND"} <= scen


@pytest.mark.parametrize("cid", list(CASES))
def test_truth_matches_the_written_step_files(cases, cid):
    truth, case, _ = _load(cases, cid)
    t = truth["part"]["thickness_mm"]
    d = cases / cid
    assert check_nominal_step(truth, StepModel(d / "part.step", t)) == []
    as_built = StepModel(d / "as_built.step", t)
    assert check_as_built_step(truth, as_built) == []
    if _stage_index(truth) is not None:  # every case here is a prefix stage (S_N = the complete part)
        assert check_stage_step(truth, StepModel(d / "stages" / f"S{_stage_index(truth)}.step", t), as_built) == []
    # the extra STEP files are part of the hash-bound case
    assert {"as_built.step", f"stages/S{len(truth['bends'])}.step"} <= set(case["files"])


@pytest.mark.parametrize("cid", list(CASES))
def test_flat_dxf_develops_to_the_truth_lengths(cases, cid):
    truth, _, draft = _load(cases, cid)
    assert check_flat_dxf(truth, draft, read_dxf(cases / cid / "flat.dxf")) == []


def test_crosschecks_report_mutations(cases, tmp_path):
    """Mutate-and-check: each check fails on a 1e-3 truth error, a wrong file or a shifted line."""
    import copy

    import ezdxf

    cid = "E2-DEV-HT-005"  # stage k with a backgauge and a bend-line shift on the just-formed bend
    truth, _, draft = _load(cases, cid)
    d = cases / cid
    t = truth["part"]["thickness_mm"]
    part, as_built = StepModel(d / "part.step", t), StepModel(d / "as_built.step", t)
    stage = StepModel(d / "stages" / f"S{_stage_index(truth)}.step", t)
    assert check_nominal_step(truth, part) == check_as_built_step(truth, as_built) == check_stage_step(truth, stage, as_built) == []
    bg_uid = truth["backgauge"]["bend_uid"]
    flange = next(char_id for char_id, c in truth["characteristics"].items() if c["kind"] == "FLANGE_LENGTH" and c["truth"] is not None
                  and not any(g["boundary_kind"] == "ADJACENT_BEND_NOT_FORMED" for g in c["truth_stations"]["groups"]))
    bg_side = next(iter(truth["backgauge"]["sides"]))

    def bumped(path: tuple, delta: float = 1e-3) -> dict[str, Any]:
        bad = copy.deepcopy(truth)
        node = bad
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] += delta
        return bad

    bend_index = next(i for i, b in enumerate(truth["bends"]) if b["bend_uid"] == bg_uid)
    for path, check in (
        (("characteristics", flange, "nominal"), lambda tr: check_nominal_step(tr, part)),
        (("characteristics", flange, "truth"), lambda tr: check_as_built_step(tr, as_built)),
        (("bends", bend_index, "actual", "included_deg"), lambda tr: check_as_built_step(tr, as_built)),
        (("backgauge", "sides", bg_side, "truth_mm"), lambda tr: check_as_built_step(tr, as_built)),
        (("backgauge", "sides", bg_side, "nominal_mm"), lambda tr: check_stage_step(tr, stage, as_built)),
        (("characteristics", f"{bg_uid}/BEND_LINE_POSITION", "truth"), lambda tr: check_stage_step(tr, stage, as_built)),
    ):
        assert check(bumped(path)), f"a 1e-3 error in truth {path} went unnoticed"
    # a wrong file: the nominal BRep is not the as-built part (the injected shift shows)
    shift = next(i for i in truth["injections"] if i["kind"] == "BEND_LINE_SHIFT")
    assert abs(shift["params"]["delta_mm"]) >= 0.5
    assert check_as_built_step(truth, part)
    # the DXF: a bend line moved by 0.01 mm (identification kept), and a wrong bend note, are reported
    dxf = read_dxf(d / "flat.dxf")
    assert check_flat_dxf(truth, draft, dxf) == []
    a, b = dxf["lines"][0]
    normal = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
    moved = {**dxf, "lines": [(a + 0.01 * normal, b + 0.01 * normal)] + dxf["lines"][1:]}
    draft_moved = copy.deepcopy(draft)
    for fl in draft_moved["summary"]["flat_bend_lines"]:
        ends = (np.asarray(fl["start_xy_mm"]), np.asarray(fl["end_xy_mm"]))
        if min(np.linalg.norm(e - a) for e in ends) < TOL_MM:
            fl["start_xy_mm"], fl["end_xy_mm"] = [[float(x) for x in e + 0.01 * normal] for e in ends]
    assert check_flat_dxf(truth, draft_moved, moved)
    doc = ezdxf.readfile(str(d / "flat.dxf"))
    note = next(e for e in doc.modelspace().query("TEXT") if e.dxf.layer == "BEND")
    m = _NOTE.fullmatch(note.dxf.text)
    note.dxf.text = f"{m.group(1)} {float(m.group(2)) + 1.0:.2f}° R{m.group(3)}"
    doc.saveas(str(tmp_path / "wrong_note.dxf"))
    assert check_flat_dxf(truth, draft, read_dxf(tmp_path / "wrong_note.dxf"))


# ---------------------------------------------------------------------------
# ZEISS-like names: every planned SKU, every name shape, the C01 parser
# ---------------------------------------------------------------------------


def _planned_skus() -> dict[str, str]:
    """Case id (or part-set id) -> SKU for every DEV plan (no geometry)."""
    out: dict[str, str] = {}
    for tier in TIERS:
        for s in plan_cases(tier, "DEV"):
            assert out.setdefault(s.case_id, s.sku) == s.sku  # regression and full share ids and SKUs
    for fam in FAMILIES:
        for index in range(3):
            seed = case_seed("DEV", fam, "partset", index)
            out[f"E2-DEV-PS{FAMILY_CODES[fam]}-{index:02d}"] = opaque_sku("DEV", seed, "partset")
    return out


def test_every_planned_sku_parses_in_every_zeiss_name_shape():
    skus = _planned_skus()
    assert len(set(skus.values())) == len(skus)  # one SKU per case: names never collide
    rng = np.random.default_rng(7)
    shapes = (Scenario("x", None, "FINAL"), Scenario("x", 0, "IN_PROCESS"), Scenario("x", 1, "IN_PROCESS"),
              Scenario("x", 3, "IN_PROCESS"), Scenario("x", 2, "FINAL"))
    suffixes = Counter()
    for sku in skus.values():
        for rev in REVISIONS:
            for sc in shapes:
                name = zeiss_scan_filename(sku, rev, sc, rng)
                suffixes["(1)" if name.endswith(" (1).ply") else " 1" if name.endswith(" 1.ply") else "none"] += 1
                parsed = parse_scan_filename(name, {sku: rev})
                assert (parsed.status, parsed.sku, parsed.revision_status) == ("OK", sku, "MATCH"), (name, parsed)
                wrong = "B" if rev != "B" else "A"  # mutate-and-check: another expected revision is a mismatch
                assert parse_scan_filename(name, {sku: wrong}).status == "REVISION_MISMATCH", name
    assert set(suffixes) == {"(1)", " 1", "none"}  # every export-counter shape was exercised
    # with every planned SKU known at once (one line, many released parts) a name still names one part
    known = {sku: "A" for sku in skus.values()}
    for i, sku in enumerate(sorted(skus.values())):
        if i % 7:
            continue
        for sc in shapes:
            parsed = parse_scan_filename(zeiss_scan_filename(sku, "A", sc, rng), known)
            assert (parsed.status, parsed.sku) == ("OK", sku)
    # the part sets' unknown-SKU drop names no known part
    for fam in FAMILIES:
        seed = case_seed("DEV", fam, "partset", 0)
        name = f"{opaque_sku('DEV', seed, 'partset-unknown')}_A_FULL_SCAN 1.ply"
        assert parse_scan_filename(name, known).status == "UNMATCHED", name
