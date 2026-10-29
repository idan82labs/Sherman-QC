"""Analytic truth of an E2 case (``truth.json``, schema ``sherman_synthetic_truth_v1``).

Everything here is computed from the part parameters through
:mod:`.geometry` / :mod:`.analytic` -- never by measuring generated meshes or
scans (plan §10 amendment 5). Conventions are the contract's
(``contracts/production/ENGINE_IO.md`` §1): included angles, OML flange
lengths, side A by the +X rule on the nominal final CAD pose, bend-line
position as the signed offset of the actual OML sharp from the nominal stage
model's OML sharp along side A's outward direction, backgauge at sequence
prefix stages for the just-formed bend. Characteristics are keyed by the
contract ids (``<bend_uid>/FLANGE_LENGTH/A`` ...), using the case recipe's
``bend_uid`` values, so E4 can join truth to E3 results directly.

Frame of all geometry: ``cad_model_mm`` (the STEP frame) with the as-built
datum panel's cut features (plane, free edges, holes) at their nominal CAD
position. ``scan.raw_to_cad`` is the exact registration of the raw scan.

Undefined truths are explicit (``truth: null`` plus a reason), never a
nominal copied into an actual:

- ``WRONG_PART_SCANNED`` / ``MIRRORED_PART``: the physical part is not the
  recipe part (or is its mirror), so no characteristic of the recipe exists;
- ``BEND_NOT_FORMED``: angle/position/flanges of a flat bend;
- ``ADJACENT_BEND_NOT_FORMED``: a flange whose far boundary is an unformed
  bend (the stage backgauge covers it);
- ``NO_FAR_BOUNDARY``: no station reaches a far boundary (degenerate).

**Well-definedness** (``well_defined`` on flange and backgauge sides): the
nominal and the true far boundary both have a dominant level holding >= 80 %
of the stations (:attr:`.analytic.FarBoundary.well_defined`). An ill-defined
side keeps its station medians for diagnostics, but its value depends on
station placement, so it is excluded from every expectation list and must not
be scored (the recipe never makes such a flange side REQUIRED; a backgauge
cannot be excluded per side, hence the flag).

**Expectation lists** (``expected``):

- ``characteristics_never_pass``: applicable, well-defined characteristics
  whose true deviation is out of tolerance, plus every characteristic of a
  wrong or mirrored part. A backgauge entry is ``<characteristic id>#<side>``
  (``#A`` / ``#B``): truth gives both sides, E1 attaches the characteristic to
  one of them;
- ``characteristics_never_fail``: characteristics of NOT_YET_FORMED bends;
- ``characteristics_must_be_unknown``: applicable characteristics without a
  defined truth (``ADJACENT_BEND_NOT_FORMED``, ``NO_FAR_BOUNDARY``): the engine
  must report them NOT_MEASURED (never PASS or FAIL);
- ``characteristics_ill_defined``: well-definedness failures (never scored;
  backgauge sides as ``#side``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .analytic import WELL_DEFINED_LEVEL_FRACTION, FarBoundary, bend_line_offset, far_boundary, flat_island
from .geometry import FoldedPart, FormState, PartDesign, assign_sides_rule, rigid_inverse
from .injection import AsBuiltPart, Scenario, order_consistent, out_of_order_formed

__all__ = [
    "TRUTH_SCHEMA",
    "NominalModel",
    "build_nominal",
    "build_truth",
    "observability",
    "LABEL_LEGEND",
]

TRUTH_SCHEMA = "sherman_synthetic_truth_v1"
LABEL_LEGEND = {
    "segment_kind": {"0": "panel skin", "1": "bend skin", "2": "panel edge wall (cut edge or hole)", "3": "bend side wall", "255": "outlier"},
    "segment_index": "index into truth.scan.label_panels (kinds 0, 2) or truth.scan.label_bends (kinds 1, 3); -1 outlier",
    "skin": {"0": "contract skin 0 of the physical part (ENGINE_IO §1 rule on the scanned part's datum; for a mirrored part "
                  "the reflected datum normal, for a wrong part its own datum)", "1": "contract skin 1", "2": "edge wall",
             "255": "outlier"},
    "view": "robot view index; -1 outlier",
    "setup": "scan setup index; -1 outlier",
    "flare": "edge-flare (mixed pixel) point",
    "ideal_cad": "noise-free surface point in cad_model_mm (NaN for outliers)",
}


def _f(x: float | None) -> float | None:
    if x is None:
        return None
    v = float(x)
    if not math.isfinite(v):
        raise ValueError("non-finite truth value")
    return 0.0 if v == 0.0 else v


def _vec(v) -> list[float]:
    return [_f(x) for x in np.asarray(v, dtype=float).ravel()]


def _line(a: np.ndarray) -> dict[str, list[float]]:
    return {"start_mm": _vec(a[0]), "end_mm": _vec(a[1])}


def _mat(m: np.ndarray) -> list[list[float]]:
    return [[_f(x) for x in row] for row in np.asarray(m, dtype=float)]


def _fb_stats(fb: FarBoundary, uid_by_key: dict[str, str] | None = None) -> dict[str, Any]:
    """Station statistics; bends are named by ``bend_uid`` (design key for an extra,
    unplanned bend), panels by E2 panel key (see ``truth.part.panels``)."""
    m = uid_by_key or {}
    return {
        "median_mm": _f(fb.value_mm),
        "min_mm": _f(fb.min_mm),
        "max_mm": _f(fb.max_mm),
        "boundary_kind": fb.boundary_kind,
        "kind_fraction": _f(fb.kind_fraction),
        "stations": fb.stations,
        "valid_stations": fb.valid_stations,
        "dominant_level_fraction": _f(fb.dominant_level_fraction),
        "dominant_level_mm": _f(fb.dominant_level_mm),
        "well_defined": fb.well_defined,
        "ill_defined_reason": fb.ill_defined_reason,
        "groups": [
            {"boundary_kind": k, "adjacent_bend": m.get(a, a) if a else None, "fraction": _f(fr), "median_mm": _f(md),
             "min_mm": _f(lo), "max_mm": _f(hi)}
            for k, a, fr, md, lo, hi in fb.groups
        ],
        "levels": [
            {"boundary_kind": k, "adjacent_bend": m.get(a, a) if a else None, "fraction": _f(fr), "median_mm": _f(md),
             "min_mm": _f(lo), "max_mm": _f(hi)}
            for k, a, fr, md, lo, hi in fb.levels
        ],
        "island_panels": list(fb.island_panels),
        "crossed_unformed_bends": [m.get(b, b) for b in fb.crossed_bends],
    }


# ---------------------------------------------------------------------------
# nominal model
# ---------------------------------------------------------------------------


@dataclass
class NominalModel:
    design: PartDesign
    datum: str
    final: FoldedPart
    #: bend key -> (side A panel, side B panel), +X rule at the final CAD pose
    sides: dict[str, tuple[str, str]]
    #: (bend key, side letter) -> flange-length far boundary at the final pose
    flange: dict[tuple[str, str], FarBoundary]
    #: prefix stage models S_0..S_N
    stages: list[FoldedPart]
    #: k -> side letter -> backgauge far boundary of bend seq[k-1] at S_k
    backgauge: dict[int, dict[str, FarBoundary]]

    def side_panel(self, key: str, side: str) -> str:
        a, b = self.sides[key]
        return a if side == "A" else b


def build_nominal(design: PartDesign) -> NominalModel:
    """Nominal truth model: final pose, sides, flange lengths, prefix stages, backgauge."""
    datum = design.default_datum()
    final = FoldedPart(design, FormState.nominal(design), datum=datum)
    sides: dict[str, tuple[str, str]] = {}
    flange: dict[tuple[str, str], FarBoundary] = {}
    for b in design.bends:
        oa = final.outward_cad(b.key, b.panel_a)
        ob = final.outward_cad(b.key, b.panel_b)
        ia, ib = assign_sides_rule(oa, ob)
        pair = (b.panel_a, b.panel_b)
        sides[b.key] = (pair[ia], pair[ib])
        for letter, panel in (("A", pair[ia]), ("B", pair[ib])):
            flange[(b.key, letter)] = far_boundary(final, b.key, panel, across_unformed=False)
    seq = list(design.sequence)
    stages = [FoldedPart(design, FormState.nominal(design, seq[:k]), datum=datum) for k in range(len(seq) + 1)]
    backgauge: dict[int, dict[str, FarBoundary]] = {}
    for k in range(1, len(seq)):
        key = seq[k - 1]
        part = stages[k]
        per_side: dict[str, FarBoundary] = {}
        for letter in ("A", "B"):
            panel = sides[key][0] if letter == "A" else sides[key][1]
            _, crossed = flat_island(part, key, panel)
            if crossed:
                per_side[letter] = far_boundary(part, key, panel, across_unformed=True)
        backgauge[k] = per_side
    return NominalModel(design, datum, final, sides, flange, stages, backgauge)


# ---------------------------------------------------------------------------
# observability (scan truth aid)
# ---------------------------------------------------------------------------


def observability(
    actual: FoldedPart,
    key: str,
    panel: str,
    fb: FarBoundary | None,
    labels: dict[str, np.ndarray],
    ideal_cad: np.ndarray,
    spacing_mm: float,
    panel_index: dict[str, int],
    *,
    island: tuple[str, ...] | None = None,
    min_support: int = 200,
    min_edge_coverage: float = 0.6,
) -> dict[str, Any]:
    """How well the scan sees one flange side (or backgauge island) of a formed bend.

    ``oml_support_points``: scan points on the side panel's OML skin inside the
    corridor; ``edge_coverage``: fraction of axial bins (``max(5, span/5)``)
    whose farthest point lies within ``max(1 mm, 2*spacing)`` of where the
    material ends; ``observable`` applies the recipe coverage defaults
    (200 points, 0.6). A heuristic truth aid, not the engine's rule."""
    g = actual.bend_geo[key]
    panels = island or (panel,)
    idx = np.asarray([panel_index[p] for p in panels])
    kinds = labels["segment_kind"]
    seg = labels["segment_index"]
    flat_skin = labels["flat_skin"]
    on_panels = np.isin(seg, idx) & ((kinds == 0) | (kinds == 2))
    w_inv = rigid_inverse(actual.world[panel])
    pts = ideal_cad[on_panels]
    if len(pts) == 0:
        return {"oml_support_points": 0, "edge_coverage": 0.0, "observable": False, "panel_points": 0}
    flat = pts @ w_inv[:3, :3].T + w_inv[:3, 3]
    rel = flat[:, :2] - g.line_point
    v = rel @ g.d
    u = rel @ g.outward(panel)
    e0, e1 = g.extent
    tau = min(2.0, 0.05 * (e1 - e0))
    in_corr = (v >= e0 + tau) & (v <= e1 - tau) & (u >= g.ba_actual / 2.0 - 1e-6)
    oml = in_corr & (kinds[on_panels] == 0) & (seg[on_panels] == panel_index[panel]) & (flat_skin[on_panels] == g.oml_flat_skin)
    support = int(oml.sum())
    coverage = 0.0
    if fb is not None and fb.exit_lambda_mm is not None:
        nb = max(5, int((e1 - e0) / 5.0))
        edges = np.linspace(e0 + tau, e1 - tau, nb + 1)
        tol = max(1.0, 2.0 * spacing_mm)
        covered = 0
        sel_v, sel_u = v[in_corr], u[in_corr]
        for i in range(nb):
            m = (sel_v >= edges[i]) & (sel_v < edges[i + 1])
            if m.any() and float(sel_u[m].max()) >= fb.exit_lambda_mm - tol:
                covered += 1
        coverage = covered / nb
    return {
        "oml_support_points": support,
        "panel_points": int(in_corr.sum()),
        "edge_coverage": _f(coverage),
        "observable": bool(support >= min_support and coverage >= min_edge_coverage),
    }


def _hinge_support(actual: FoldedPart, key: str, labels: dict[str, np.ndarray], ideal_cad: np.ndarray, panel_index: dict[str, int]) -> dict[str, Any]:
    g = actual.bend_geo[key]
    out = {}
    for role, panel in (("parent", g.parent), ("child", g.child)):
        m = (labels["segment_index"] == panel_index[panel]) & (labels["segment_kind"] == 0)
        pts = ideal_cad[m]
        if len(pts) == 0:
            out[f"{role}_points"] = 0
            continue
        w_inv = rigid_inverse(actual.world[panel])
        flat = pts @ w_inv[:3, :3].T + w_inv[:3, 3]
        rel = flat[:, :2] - g.line_point
        u = rel @ g.outward(panel)
        v = rel @ g.d
        band = (u >= g.ba_actual / 2.0) & (u <= g.ba_actual / 2.0 + 25.0) & (v >= g.extent[0]) & (v <= g.extent[1])
        out[f"{role}_points"] = int(band.sum())
    out["observable"] = bool(min(out.get("parent_points", 0), out.get("child_points", 0)) >= 200)
    return out


# ---------------------------------------------------------------------------
# truth document
# ---------------------------------------------------------------------------


def _tol_dict(tol) -> dict[str, float] | None:
    return None if tol is None else {"plus": _f(tol.plus), "minus": _f(tol.minus)}


def _in_tol(dev: float | None, tol) -> bool | None:
    if dev is None or tol is None:
        return None
    return bool(-tol.minus - 1e-12 <= dev <= tol.plus + 1e-12)


def build_truth(
    *,
    case_id: str,
    nominal: NominalModel,
    recipe,  # contracts.production.part_recipe.PartRecipeV1
    uid_by_key: dict[str, str],
    as_built: AsBuiltPart,
    scenario: Scenario,
    cad_to_raw: np.ndarray,
    labels: dict[str, np.ndarray] | None,
    ideal_cad: np.ndarray | None,
    spacing_mm: float,
    symmetry: dict[str, Any] | None,
    scan_meta: dict[str, Any],
) -> dict[str, Any]:
    design = nominal.design
    seq = list(design.sequence)
    n = len(seq)
    rbend = {b.bend_uid: b for b in recipe.bends}
    panel_index = {p.key: i for i, p in enumerate(design.panels)}
    defined = not (as_built.wrong_part or as_built.mirror is not None)
    undefined_reason = "WRONG_PART_SCANNED" if as_built.wrong_part else ("MIRRORED_PART" if as_built.mirror is not None else None)
    actual = FoldedPart(as_built.design, as_built.state, datum=as_built.datum, mirror=as_built.mirror)
    if defined:
        as_panel_index = {p.key: i for i, p in enumerate(as_built.design.panels)}
    formed = set(as_built.formed_nominal)
    reference = FoldedPart(design, FormState.nominal(design, [k for k in seq if k in formed]), datum=nominal.datum)
    k_formed = len(formed)
    # the formed set is a physical fact for a mirrored part, undefined for a wrong part
    is_prefix = (not as_built.wrong_part) and formed == set(seq[:k_formed])
    final = nominal.final

    bends_out = []
    chars: dict[str, dict[str, Any]] = {}
    for key in seq:
        uid = uid_by_key[key]
        rb = rbend[uid]
        g = final.bend_geo[key]
        a_panel, b_panel = nominal.sides[key]
        nominal_rec = {
            "sweep_deg": _f(g.sweep_deg),
            "included_deg": _f(180.0 - g.sweep_deg),
            "inner_radius_mm": _f(g.inner_radius),
            "thickness_mm": _f(g.thickness),
            "k_factor": _f(design.k_factor),
            "neutral_length_mm": _f(g.ba_nominal),
            "direction": final.contract_direction(key),
            "oml_skin": final.contract_skin(g.oml_flat_skin),
            "parent_panel": g.parent,
            "child_panel": g.child,
            "axis": {"point_mm": _vec(final.axis_cad(key)[0]), "direction": _vec(final.axis_cad(key)[1])},
            "loci_final": {name: _line(line) for name, line in final.loci(key).items()},
            "bend_length_mm": _f(g.extent[1] - g.extent[0]),
            "sides": {},
        }
        for letter, panel in (("A", a_panel), ("B", b_panel)):
            fb = nominal.flange[(key, letter)]
            nominal_rec["sides"][letter] = {
                "panel": panel,
                "outward_dir": _vec(final.outward_cad(key, panel)),
                "boundary_kind": fb.boundary_kind,
                "adjacent_bend_uid": uid_by_key.get(fb.adjacent_bend) if fb.adjacent_bend else None,
                "flange_length_mm": _f(fb.value_mm),
                "stations": _fb_stats(fb, uid_by_key),
            }
        rec: dict[str, Any] = {
            "bend_uid": uid,
            "key": key,
            "label": rb.label,
            "sequence_index": rb.sequence_index,
            "machine_move_index": rb.machine_move_index,
            "nominal": nominal_rec,
        }
        actual_rec: dict[str, Any] = {"defined": defined, "undefined_reason": undefined_reason}
        ga = actual.bend_geo[key] if defined else None
        if defined:
            state = "NOT_YET_FORMED" if not ga.formed else ("PARTIALLY_FORMED" if key in as_built.partial else "FORMED")
            actual_rec.update(
                state=state,
                sweep_deg=_f(ga.sweep_deg),
                included_deg=_f(180.0 - ga.sweep_deg),
                formed_fraction=_f(ga.sweep_deg / g.sweep_deg),
                bend_line_shift_mm=_f(as_built.state.bends[key].shift_mm),
            )
            if ga.formed:
                loci_a = actual.loci(key)
                actual_rec["loci"] = {nm: _line(loci_a[nm]) for nm in ("oml_virtual_sharp", "outer_apex", "inner_apex", "cylinder_axis")}
                actual_rec["bend_line_position"] = bend_line_offset(actual, reference, key, a_panel)
            else:
                actual_rec["strip_centre_oml"] = _line(actual.loci(key)["strip_centre_oml"])
        rec["actual"] = actual_rec
        bends_out.append(rec)

        # ---- characteristics ------------------------------------------------
        for letter, panel in (("A", a_panel), ("B", b_panel)):
            cid = f"{uid}/FLANGE_LENGTH/{letter}"
            fb_nom = nominal.flange[(key, letter)]
            tol = recipe.resolved_tolerance(uid, "flange_a_mm" if letter == "A" else "flange_b_mm")
            req = rb.flange_inspection.side_a if letter == "A" else rb.flange_inspection.side_b
            entry = _char_base("FLANGE_LENGTH", uid, letter, None, "mm", fb_nom.value_mm, tol, req)
            entry["nominal_boundary_kind"] = fb_nom.boundary_kind
            entry["nominal_adjacent_bend_uid"] = uid_by_key.get(fb_nom.adjacent_bend) if fb_nom.adjacent_bend else None
            entry["nominal_stations"] = _fb_stats(fb_nom, uid_by_key)
            entry["well_defined"] = None
            entry["ill_defined_reason"] = None
            if not defined:
                _undefined(entry, undefined_reason)
            elif not ga.formed:
                _not_applicable(entry, "BEND_NOT_YET_FORMED", "BEND_NOT_FORMED")
            else:
                fb_act = far_boundary(actual, key, panel, across_unformed=False)
                entry["truth_stations"] = _fb_stats(fb_act, uid_by_key)
                if fb_act.boundary_kind in ("ADJACENT_BEND_NOT_FORMED", "UNDEFINED"):
                    entry["truth"] = None
                    entry["truth_undefined_reason"] = (
                        "ADJACENT_BEND_NOT_FORMED" if fb_act.boundary_kind == "ADJACENT_BEND_NOT_FORMED" else "NO_FAR_BOUNDARY")
                    entry["truth_strip_centre_distance_mm"] = _f(fb_act.value_mm)
                    entry.pop("_tol", None)
                else:
                    _set_truth(entry, fb_act.value_mm)
                    entry["boundary_changed"] = (fb_act.boundary_kind, fb_act.adjacent_bend) != (fb_nom.boundary_kind, fb_nom.adjacent_bend)
                    entry["well_defined"] = bool(fb_nom.well_defined and fb_act.well_defined)
                    entry["ill_defined_reason"] = None if entry["well_defined"] else (
                        f"NOMINAL_{fb_nom.ill_defined_reason}" if not fb_nom.well_defined else f"TRUTH_{fb_act.ill_defined_reason}")
                if req == "NOT_INSPECTED":
                    entry["applicable"] = False
                    entry["not_applicable_reason"] = "NOT_INSPECTED_BY_RECIPE"
                if labels is not None and ideal_cad is not None:
                    entry["observability"] = observability(actual, key, panel, fb_act, labels, ideal_cad, spacing_mm, as_panel_index)
            chars[cid] = entry
        tol_a = recipe.resolved_tolerance(uid, "angle_deg")
        entry = _char_base("INCLUDED_ANGLE", uid, None, None, "deg", 180.0 - g.sweep_deg, tol_a, "REQUIRED")
        if not defined:
            _undefined(entry, undefined_reason)
        elif not ga.formed:
            _not_applicable(entry, "BEND_NOT_YET_FORMED", "BEND_NOT_FORMED")
        else:
            _set_truth(entry, 180.0 - ga.sweep_deg)
            if labels is not None and ideal_cad is not None:
                entry["observability"] = _hinge_support(actual, key, labels, ideal_cad, as_panel_index)
        chars[f"{uid}/INCLUDED_ANGLE"] = entry
        tol_p = recipe.resolved_tolerance(uid, "bend_line_position_mm")
        entry = _char_base("BEND_LINE_POSITION", uid, None, None, "mm", 0.0, tol_p, "REQUIRED" if tol_p is not None else "OPTIONAL")
        entry["nominal_line_stage"] = _line(reference.bend_line_cad(key)) if key in formed else None
        if not defined:
            _undefined(entry, undefined_reason)
        elif not ga.formed:
            _not_applicable(entry, "BEND_NOT_YET_FORMED", "BEND_NOT_FORMED")
        else:
            off = actual_rec["bend_line_position"]
            _set_truth(entry, off["value_mm"])
            entry["normal_component_mm"] = _f(off["normal_component_mm"])
            entry["line_angle_deg"] = _f(off["line_angle_deg"])
            if labels is not None and ideal_cad is not None:
                entry["observability"] = _hinge_support(actual, key, labels, ideal_cad, as_panel_index)
        chars[f"{uid}/BEND_LINE_POSITION"] = entry

    for c in chars.values():
        c.pop("_tol", None)

    # ---- backgauge at an order-consistent prefix stage -------------------------
    backgauge_rec = None
    if is_prefix and 1 <= k_formed <= n - 1:
        key = seq[k_formed - 1]
        uid = uid_by_key[key]
        tol_b = recipe.resolved_tolerance(uid, "backgauge_mm")
        backgauge_rec = {"stage_index": k_formed, "bend_uid": uid, "sides": {}}
        for letter, fb_nom in nominal.backgauge.get(k_formed, {}).items():
            panel = nominal.side_panel(key, letter)
            fb_act = far_boundary(actual, key, panel, across_unformed=True)
            side_ok = bool(fb_nom.well_defined and fb_act.well_defined)
            side_rec = {
                "side": letter,
                "panel": panel,
                "well_defined": side_ok,
                "ill_defined_reason": None if side_ok else (
                    f"NOMINAL_{fb_nom.ill_defined_reason}" if not fb_nom.well_defined else f"TRUTH_{fb_act.ill_defined_reason}"),
                "nominal_mm": _f(fb_nom.value_mm),
                "truth_mm": _f(fb_act.value_mm),
                "true_deviation_mm": _f(fb_act.value_mm - fb_nom.value_mm) if fb_act.value_mm is not None and fb_nom.value_mm is not None else None,
                "tolerance": _tol_dict(tol_b),
                "truth_in_tolerance": _in_tol(
                    None if fb_act.value_mm is None or fb_nom.value_mm is None else fb_act.value_mm - fb_nom.value_mm, tol_b
                ),
                "nominal_stations": _fb_stats(fb_nom, uid_by_key),
                "truth_stations": _fb_stats(fb_act, uid_by_key),
            }
            if labels is not None and ideal_cad is not None:
                side_rec["observability"] = observability(
                    actual, key, panel, fb_act, labels, ideal_cad, spacing_mm, as_panel_index, island=fb_act.island_panels
                )
            backgauge_rec["sides"][letter] = side_rec
        # the contract characteristic: E1 attaches it to one side; truth gives both
        cid = f"{uid}/BACKGAUGE_LENGTH/S{k_formed}"
        chars[cid] = {
            "kind": "BACKGAUGE_LENGTH",
            "bend_uid": uid,
            "stage_index": k_formed,
            "unit": "mm",
            "tolerance": _tol_dict(tol_b),
            "recipe_requirement": "REQUIRED" if tol_b is not None else "OPTIONAL",
            "applicable": True,
            "per_side": backgauge_rec["sides"],
            "note": "E1 attaches the backgauge to the side leading toward the unformed bends; match on side. "
                    "Score a side only when per_side[side].well_defined.",
        }

    # ---- stage & expectations -------------------------------------------------
    formed_uids = [uid_by_key[k] for k in seq if k in formed]
    out_of_order = [uid_by_key[k] for k in out_of_order_formed(design, formed)] if defined else []
    states = {b["bend_uid"]: b["actual"].get("state") for b in bends_out} if defined else {}
    # contract vocabulary: a formed bend the order prior does not expect is UNEXPECTED_FORMED
    states_contract = {u: ("UNEXPECTED_FORMED" if (st == "FORMED" and u in out_of_order) else st) for u, st in states.items()}
    never_pass = []
    never_fail = []
    must_be_unknown = []
    ill_defined = []
    for cid, c in chars.items():
        if c.get("kind") == "BACKGAUGE_LENGTH":
            for side in c["per_side"].values():
                if not side.get("well_defined"):
                    ill_defined.append(f"{cid}#{side['side']}")
                elif side.get("truth_in_tolerance") is False:
                    never_pass.append(f"{cid}#{side['side']}")
            continue
        if not defined:
            never_pass.append(cid)
            continue
        if c.get("applicable") and c.get("well_defined") is False:
            ill_defined.append(cid)
        elif c.get("truth_in_tolerance") is False and c.get("applicable"):
            never_pass.append(cid)
        if c.get("applicable") and c.get("truth") is None:
            must_be_unknown.append(cid)
        if c.get("not_applicable_reason") == "BEND_NOT_YET_FORMED":
            never_fail.append(cid)
    summary_never = set()
    if never_pass or not defined:
        summary_never |= {"ALL_MEASURED_IN_TOLERANCE", "STAGE_IN_TOLERANCE"}
    missing_final = [uid_by_key[k] for k in seq if k not in formed] if (defined and scenario.inspection_stage == "FINAL") else []
    if missing_final or as_built.extra_bends or as_built.partial or k_formed < n or not defined:
        summary_never.add("ALL_MEASURED_IN_TOLERANCE")
    sym_groups = []
    if symmetry:
        sym_groups = [[uid_by_key.get(k, k) for k in grp] for grp in symmetry.get("ambiguous_bend_groups", [])]
    stage_models = []
    for k, part in enumerate(nominal.stages):
        stage_models.append(
            {
                "stage_model_id": f"S{k}",
                "formed_bend_uids": [uid_by_key[x] for x in seq[:k]],
                "panel_transforms_final_to_stage": {
                    p.key: _mat(part.world[p.key] @ rigid_inverse(final.world[p.key])) for p in design.panels
                },
                "bend_lines": {uid_by_key[x]: _line(part.bend_line_cad(x)) for x in seq},
            }
        )
    extra_out = []
    for xk in as_built.extra_bends:
        loc = actual.loci(xk)
        gx = actual.bend_geo[xk]
        extra_out.append(
            {"key": xk, "sweep_deg": _f(gx.sweep_deg), "included_deg": _f(180.0 - gx.sweep_deg),
             "oml_virtual_sharp": _line(loc["oml_virtual_sharp"]), "outer_apex": _line(loc["outer_apex"]), "split_panel": list(as_built.split_panel or ())}
        )
    raw_to_cad = rigid_inverse(cad_to_raw)
    doc = {
        "schema": TRUTH_SCHEMA,
        "case_id": case_id,
        "frame": "cad_model_mm",
        "units": {"length": "mm", "angle": "deg"},
        "sku": recipe.sku,
        "revision": recipe.part_revision,
        "family": design.family,
        "conventions": {
            "angle": "INCLUDED_DEG (180 = flat)",
            "flange_length": "OUTSIDE_MOLD_LINE; per station the farthest crossing of the material (per-bin maximum extent); "
                             "median over the stations of the majority boundary kind along the trimmed bend corridor",
            "well_defined": "a 0.05 mm level holds >= 80 % of the stations (nominal and truth); score only well-defined sides",
            "side_naming": "side A has the larger +X (then +Y, +Z) in-plane outward unit vector at the nominal final pose",
            "bend_line_position": "actual OML sharp minus nominal stage-model OML sharp, perpendicular to the axis at nominal mid-span, along side A outward_dir",
            "backgauge": "OML of the just-formed bend k at S_k across unformed bends to the far boundary",
            "datum_frame": "as-built datum panel cut features at their nominal CAD position (registration truth)",
            "k_factor": "flat developed with K; stage models (Method A) use the same K",
        },
        "part": {
            "thickness_mm": _f(design.thickness),
            "k_factor": _f(design.k_factor),
            "datum_panel": nominal.datum,
            "datum_rule": "largest panel by area (ties by key)",
            "cad_placement_flat_to_cad": _mat(design.placement_matrix),
            "contract_skin0_is_flat_skin0": final.contract_skin0_is_flat_skin0(),
            "panels": [
                {"index": i, "key": p.key, "role": p.role, "area_mm2": _f(p.area()), "is_datum": p.key == nominal.datum,
                 "holes": [{"center_flat": list(h.center), "radius_mm": _f(h.radius)} for h in p.holes],
                 "skin0_normal_final": _vec(final.panel_normal_skin0(p.key))}
                for i, p in enumerate(design.panels)
            ],
            "bends_geometric": [
                {"index": i, "key": b.key, "bend_uid": uid_by_key.get(b.key)} for i, b in enumerate(design.bends)
            ],
            "sequence": [uid_by_key[k] for k in seq],
            "machine_moves": [[uid_by_key[k] for k in grp] for grp in design.groups],
            "params": _jsonable(dict(design.params)),
        },
        "bends": bends_out,
        "characteristics": chars,
        "backgauge": backgauge_rec,
        "stage": {
            "inspection_stage": scenario.inspection_stage,
            "scenario": scenario.name,
            "formed_bend_uids": None if as_built.wrong_part else formed_uids,
            "formed_count": None if as_built.wrong_part else k_formed,
            "partially_formed_bend_uids": [uid_by_key[k] for k in seq if k in as_built.partial],
            "is_sequence_prefix": None if as_built.wrong_part else bool(is_prefix),
            "order_consistent": None if as_built.wrong_part else bool(order_consistent(design, formed)),
            "out_of_order_formed_bend_uids": out_of_order,
            "expected_stage_model_id": f"S{k_formed}" if (is_prefix and defined) else None,
            "missing_bend_uids_final": missing_final,
        },
        "stage_models_nominal": stage_models,
        "injections": [i.to_dict() for i in scenario.injections],
        "extra_bends": extra_out,
        "mirrored": as_built.mirror is not None,
        "mirror_matrix_cad": _mat(as_built.mirror) if as_built.mirror is not None else None,
        "wrong_sku": as_built.wrong_part,
        "expected": {
            "bend_states": states,
            "bend_states_contract": states_contract,
            "validity_note": (
                "MIRRORED_PART_SUSPECTED expected (never a PASS)" if as_built.mirror is not None
                else "wrong part: never PASS" if as_built.wrong_part else "VALID registration expected when observable"
            ),
            "summary_never": sorted(summary_never),
            "characteristics_never_pass": sorted(never_pass),
            "characteristics_never_fail": sorted(never_fail),
            "characteristics_must_be_unknown": sorted(must_be_unknown),
            "characteristics_ill_defined": sorted(ill_defined),
            "well_defined_rule": f"dominant 0.05 mm level holds >= {WELL_DEFINED_LEVEL_FRACTION:.0%} of the stations (nominal and truth)",
            # Bend identity is ambiguous only through a proper self-map (rotation) or a
            # non-trivial bend orbit; a mirror plane perpendicular to the bends maps
            # every bend onto itself and makes nothing ambiguous.
            "symmetric": bool(symmetry and (symmetry.get("rotation_symmetric") or sym_groups)),
            "contract_is_symmetric": bool(symmetry and symmetry.get("is_symmetric")),
            "ambiguous_bend_groups": sym_groups,
            "mirror_image_rms_mm": _f(min((c["rms_mm"] for c in symmetry["candidates"] if c["kind"] == "MIRROR"), default=0.0))
            if symmetry else None,
            # a mirrored part whose mirror image is within the symmetry threshold differs
            # from the design only by small chiral features (holes, notches)
            "mirror_distinguishable_by_rms": (
                bool(min((c["rms_mm"] for c in symmetry["candidates"] if c["kind"] == "MIRROR"), default=0.0) > symmetry["threshold_rms_mm"])
                if symmetry else None
            ),
        },
        "symmetry": _symmetry_out(symmetry, uid_by_key),
        "scan": {
            **scan_meta,
            "cad_to_raw": _mat(cad_to_raw),
            "raw_to_cad": _mat(raw_to_cad),
            "label_legend": LABEL_LEGEND,
            "label_panels": [p.key for p in as_built.design.panels],
            "label_bends": [
                {"key": b.key, "bend_uid": uid_by_key.get(b.key) if not as_built.wrong_part else None}
                for b in as_built.design.bends
            ],
        },
    }
    if defined:
        doc["actual_panel_transforms_reference_to_actual"] = {
            p.key: _mat(actual.world[p.key] @ rigid_inverse(reference.world[p.key])) for p in design.panels
        }
    return doc


def _char_base(kind, uid, side, stage, unit, nominal, tol, requirement) -> dict[str, Any]:
    return {
        "kind": kind,
        "bend_uid": uid,
        "side": side,
        "stage_index": stage,
        "unit": unit,
        "nominal": _f(nominal),
        "tolerance": _tol_dict(tol),
        "recipe_requirement": requirement,
        "applicable": True,
        "not_applicable_reason": None,
        "truth": None,
        "true_deviation": None,
        "truth_in_tolerance": None,
        "truth_undefined_reason": None,
        "_tol": tol,
    }


def _set_truth(entry: dict[str, Any], value: float | None) -> None:
    tol = entry.pop("_tol", None)
    entry["truth"] = _f(value)
    if value is not None and entry["nominal"] is not None:
        dev = float(value) - float(entry["nominal"])
        entry["true_deviation"] = _f(dev)
        entry["truth_in_tolerance"] = _in_tol(dev, tol)


def _not_applicable(entry: dict[str, Any], reason: str, truth_reason: str) -> None:
    entry.pop("_tol", None)
    entry["applicable"] = False
    entry["not_applicable_reason"] = reason
    entry["truth_undefined_reason"] = truth_reason


def _undefined(entry: dict[str, Any], reason: str | None) -> None:
    entry.pop("_tol", None)
    entry["applicable"] = False
    entry["not_applicable_reason"] = None
    entry["truth_undefined_reason"] = reason


def _symmetry_out(sym: dict[str, Any] | None, uid_by_key: dict[str, str]) -> dict[str, Any] | None:
    if sym is None:
        return None
    out = dict(sym)
    out["candidates"] = [
        {**c, "bend_permutation": {uid_by_key.get(a, a): uid_by_key.get(b, b) for a, b in c["bend_permutation"].items()},
         "rms_mm": _f(c["rms_mm"]), "transform": _mat(np.asarray(c["transform"]))}
        for c in sym["candidates"]
    ]
    out["ambiguous_bend_groups"] = [[uid_by_key.get(k, k) for k in g] for g in sym["ambiguous_bend_groups"]]
    return out


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return _f(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    return x
