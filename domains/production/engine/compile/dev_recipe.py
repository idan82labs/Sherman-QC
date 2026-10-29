"""DEV_AUTO_CONFIRMED recipes for tests and development compile runs (plan §10 amendment 4d).

A DEV recipe stands in for operator confirmation: **every** CAD candidate bend
becomes an active, confirmed mark in candidate order, both flange sides are
REQUIRED, and nothing is excluded except through the pre-written per-feature
UNSUPPORTED rule of the compile. It is built through the real C01 builder
(:func:`domains.production.recipe.builder.build_part_recipe`) from a synthetic
Vendor Setup draft, so its bend identities and hashes are exactly what a
release would produce.

Candidates come from

- ``native``: ``scripts/extract_step_topology_bend_axes.py`` (the extractor the
  CAD upload uses through ``backend/vendor_unfold_preview._read_native``), or
- ``v10``: ``scripts/extract_step_topology_bend_axes_v10.py`` (V10 boundary
  chains; imports the scripts v2/v7/v9 modules and
  ``backend/cad_finite_native_curve_graph_domain_v2/v3``), or
- explicit candidates (tests: the analytic mark lines of a fixture).

Every candidate carries the native line convention (mid-radius half angular
turn), ``target_angle`` = the native sweep (NATIVE_SWEEP convention) and
``target_radius`` = the inner radius.

DEV candidate policy (:func:`dev_candidates`): the **union** of the V10 and the
native candidates ("every V10/native bend"), with (1) geometric duplicates (the
same line found by both extractors) dropped, (2) candidates feasible for the
same flange-graph bend (a bend interrupted by relief cutouts, which the
extractors report piecewise, or two extractors' markers on one bend) merged
into one mark spanning them - the resolution an operator makes in the wizard
for MARK_AMBIGUOUS duplicates - and (3) candidates whose own cylinder faces are
an unsupported feature under the frozen per-feature rule excluded (amendment
4d: "exclusions only through a pre-written per-feature UNSUPPORTED rule").
A candidate that matches no graph bend *and* is not a rule-declared feature
stays in the recipe and blocks the compile (MARK_NOT_IN_FLANGE_GRAPH), so the
compile still tests the agreement of the bend inventories.

E1-DEV-POLICY-2 (2026-10-19) adds (4): a *supported* flange-graph bend that no
extractor candidate is feasible for gets the identity mark an operator would
add for the wizard's UNMARKED_BEND (the extractors miss some bends: a joggle on
a narrow strip, skins that are not coaxial); the report lists them under
``added_from_e1_graph``. It also applies rule (3) when the CAD has no planar
flange panel at all (a rolled part or a seamed tube): candidates on
rule-declared features (ROLLED, ...) are excluded, so a part whose only
"bends" are rolled features yields no recipe (NO_CANDIDATE_BENDS) instead of
a recipe that can only block.

Steps (2) and (3) use E1's own flange graph and feasibility test, so a compile
rate measured under this policy is *with E1-graph dedupe*: MARK_AMBIGUOUS
cannot arise from extractor duplicates. The amendment-4d figure is the strict
single-source rate (every V10 bend as reported); both are reported. The
policy is pinned by :data:`DEV_POLICY_VERSION` and :func:`dev_policy_sha256`
(this file plus the UNSUPPORTED rule version), which ``dev_candidates``
records so frozen 4d recipes name the policy that built them.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from contracts.production.part_recipe import (
    CadRef,
    DrawingRef,
    FlatBindingV1,
    FlatRef,
    PartRecipeV1,
    ReleasedBy,
    ReleaseRequestV1,
)
from domains.production.recipe.builder import build_part_recipe

__all__ = [
    "DEV_POLICY_VERSION",
    "dev_policy_sha256",
    "Candidate",
    "native_candidates",
    "union_candidates",
    "merge_graph_duplicates",
    "dev_candidates",
    "dev_recipe",
    "DEFAULT_DEV_TOLERANCES",
]

#: Shop-standard development tolerances (not a quality claim; DEV recipes only).
DEFAULT_DEV_TOLERANCES: dict[str, Any] = {
    "angle_deg": {"plus": 1.0, "minus": 1.0},
    "flange_mm": {"plus": 0.5, "minus": 0.5},
    "bend_line_position_mm": {"plus": 0.5, "minus": 0.5},
    "backgauge_mm": {"plus": 0.3, "minus": 0.3},
    "general_mm": {"plus": 0.8, "minus": 0.8},
    "surface_profile_mm": 0.5,
    "source": "SHOP_STANDARD",
    "note": "DEV_AUTO_CONFIRMED development tolerances",
}

Candidate = dict  # keys: line_start, line_end, target_angle (sweep deg), target_radius (Ri mm), label?

#: Version of the DEV candidate policy (union, geometric dedupe, E1-graph merge, rule exclusion).
DEV_POLICY_VERSION = "E1-DEV-POLICY-2"


def dev_policy_sha256() -> str:
    """sha256 pinning the DEV policy: this module's source plus the UNSUPPORTED rule version."""
    from domains.production.engine.compile.unsupported import UNSUPPORTED_RULE_VERSION

    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(f"|{DEV_POLICY_VERSION}|{UNSUPPORTED_RULE_VERSION}".encode("utf-8"))
    return h.hexdigest()


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _round_point(p: Sequence[float]) -> list[float]:
    return [round(float(c), 6) + 0.0 for c in p]


def native_candidates(cad_path: str | Path, source: str = "v10") -> list[Candidate]:
    """CAD candidate bends from the native (upload) or V10 extractor; [] when none."""
    path = Path(cad_path)
    if source == "native":
        from scripts.extract_step_topology_bend_axes import extract_step_bend_axes
    elif source == "v10":
        from scripts.extract_step_topology_bend_axes_v10 import extract_step_bend_axes
    else:
        raise ValueError("source must be 'native' or 'v10'")
    payload = extract_step_bend_axes(path)
    out = []
    for i, axis in enumerate(payload.get("bend_axes", []), start=1):
        sweeps = axis.get("angular_sweeps_deg") or []
        radii = axis.get("radii_mm") or []
        start, end = axis.get("canonical_line_start"), axis.get("canonical_line_end")
        if not sweeps or not radii or start is None or end is None:
            continue
        sweep = float(sum(sweeps) / len(sweeps))
        if not 0 < sweep < 180:
            continue
        out.append({"line_start": _round_point(start), "line_end": _round_point(end), "target_angle": round(sweep, 6),
                    "target_radius": round(float(min(radii)), 6), "label": axis.get("bend_id") or f"STEP_BEND_{i:03d}"})
    return out


def _unit3(v) -> list[float]:
    n = math.sqrt(sum(float(c) * float(c) for c in v))
    return [float(c) / n for c in v]


def _same_line(a: Candidate, b: Candidate, tol_mm: float = 0.5, tol_deg: float = 2.0) -> bool:
    da = _unit3([y - x for x, y in zip(a["line_start"], a["line_end"])])
    db = _unit3([y - x for x, y in zip(b["line_start"], b["line_end"])])
    if abs(sum(x * y for x, y in zip(da, db))) < math.cos(math.radians(tol_deg)):
        return False
    ma = [(x + y) / 2 for x, y in zip(a["line_start"], a["line_end"])]
    mb = [(x + y) / 2 for x, y in zip(b["line_start"], b["line_end"])]
    rel = [y - x for x, y in zip(ma, mb)]
    along = sum(x * y for x, y in zip(rel, da))
    perp = math.sqrt(max(0.0, sum(x * x for x in rel) - along * along))
    if perp > tol_mm:
        return False
    sa = sorted(sum((p - q) * d for p, q, d in zip(pt, a["line_start"], da)) for pt in (a["line_start"], a["line_end"]))
    sb = sorted(sum((p - q) * d for p, q, d in zip(pt, a["line_start"], da)) for pt in (b["line_start"], b["line_end"]))
    overlap = min(sa[1], sb[1]) - max(sa[0], sb[0])
    return overlap >= 0.5 * min(sa[1] - sa[0], sb[1] - sb[0])


def union_candidates(*sources: Sequence[Candidate]) -> list[Candidate]:
    """Concatenate candidate lists, dropping geometric duplicates (first source wins)."""
    out: list[Candidate] = []
    for src in sources:
        for cand in src:
            if not any(_same_line(cand, kept) for kept in out):
                out.append(dict(cand))
    return out


def _candidate_cylinders(brep, cand: Candidate, thickness: float) -> list[int]:
    """Cylinder faces whose axis is parallel to the candidate and whose skin lies within t of it."""
    import numpy as np

    a = np.asarray(cand["line_start"], dtype=float)
    b = np.asarray(cand["line_end"], dtype=float)
    d = b - a
    d = d / np.linalg.norm(d)
    mid = 0.5 * (a + b)
    out = []
    for f in brep.faces:
        if f.kind != "cylinder" or abs(float(f.axis @ d)) < math.cos(math.radians(2.0)):
            continue
        rel = mid - f.origin
        radial = float(np.linalg.norm(rel - float(rel @ f.axis) * f.axis))
        if abs(radial - float(f.radius)) <= thickness / 2.0 + 0.25:
            pts = np.concatenate([u.xyz for lp in f.loops for u in lp.uses if len(u.xyz)])
            z = (pts - f.origin) @ f.axis
            zm = float(rel @ f.axis)
            if z.min() - 1.0 <= zm <= z.max() + 1.0:
                out.append(f.index)
    return out


def merge_graph_duplicates(cad_path: str | Path, candidates: Sequence[Candidate]) -> tuple[list[Candidate], dict]:
    """Merge candidates feasible for the same flange-graph bend; exclude candidates on rule-declared features.

    Returns ``(candidates, report)``. A candidate feasible for no graph bend is
    excluded only when its own cylinder faces are an unsupported feature under
    the frozen rule (a graph feature, or wall faces the compile's thin-wall test
    declares OTHER); otherwise it is kept and will block the compile.
    """
    from domains.production.engine.compile.brep import load_step
    from domains.production.engine.compile.config import CompileConfig
    from domains.production.engine.compile.flange_graph import build_flange_graph
    from domains.production.engine.compile.part_model import build_part_model
    from domains.production.engine.compile.reconcile import Mark, pair_metrics
    from domains.production.engine.compile.sampling import choose_spacing, sample_final
    from domains.production.engine.compile.unsupported import wall_is_thin

    cfg = CompileConfig()
    brep = load_step(cad_path, linear_deflection=cfg.mesh_linear_deflection_mm,
                     angular_deflection=cfg.mesh_angular_deflection_rad)
    graph = build_flange_graph(brep, cfg)
    if graph.panels:
        model = build_part_model(graph)
        thickness = model.thickness
        model_bends = model.bends
    else:
        # no planar flange joined by a bend (a rolled part, a tube with a seam): no graph bend can carry a mark,
        # so every candidate is tested against the unsupported-feature rule alone (rule (3) below); a candidate
        # that is not a rule-declared feature stays and the compile reports DATUM_PANEL_UNDETERMINED as before
        model, model_bends = None, {}
        thickness = graph.thickness if graph.thickness is not None else _min_skin_gap(brep)
    groups: dict[Any, list[int]] = {}
    order: list[Any] = []
    for i, cand in enumerate(candidates):
        mark = Mark("bu_" + "0" * 20, tuple(cand["line_start"]), tuple(cand["line_end"]))
        best = None
        for gk in sorted(model_bends):
            ok, cost, _ = pair_metrics(model_bends[gk], mark, cfg)
            if ok and (best is None or cost < best[0]):
                best = (cost, gk)
        key = ("graph", best[1]) if best else ("solo", i)
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(i)
    wall_far = None
    out, notes, excluded = [], [], []
    for key in order:
        idx = groups[key]
        if key[0] == "solo":
            cand = candidates[idx[0]]
            faces = _candidate_cylinders(brep, cand, thickness) if thickness else []
            roles = [graph.face_role.get(f, ("wall",)) for f in faces]
            kinds = sorted({graph.features[r[1]].kind for r in roles if r[0] == "feature"})
            if faces and not kinds and model is not None and all(r[0] == "wall" for r in roles):
                if wall_far is None:
                    spacing = choose_spacing(brep, cfg.sample_spacing_mm, cfg.max_points_per_stage)
                    wall_far = sample_final(model, brep, set(model.bends), spacing).wall_far_fraction
                if all(not wall_is_thin(wall_far.get(f, 0.0)) for f in faces):
                    kinds = ["OTHER"]
            if faces and not kinds and model is None and thickness:
                # no flange panel: the frozen ROLLED (a) radius rule applied to the candidate's own skins
                from domains.production.engine.compile.unsupported import ROLLED_RI_OVER_T

                ri = cand.get("target_radius")
                if ri is not None and float(ri) > ROLLED_RI_OVER_T * thickness:
                    kinds = ["ROLLED"]
            if faces and kinds:
                excluded.append({"candidate": idx[0], "rule": ",".join(kinds), "faces": faces,
                                 "line_start": cand["line_start"], "line_end": cand["line_end"]})
                continue
            out.append(dict(cand))
            continue
        if len(idx) == 1:
            out.append(dict(candidates[idx[0]]))
            continue
        first = candidates[idx[0]]
        d = _unit3([y - x for x, y in zip(first["line_start"], first["line_end"])])
        base = first["line_start"]
        proj = [sum((p - q) * c for p, q, c in zip(pt, base, d)) for i in idx
                for pt in (candidates[i]["line_start"], candidates[i]["line_end"])]
        lo, hi = min(proj), max(proj)
        merged = dict(first)
        merged["line_start"] = _round_point([q + lo * c for q, c in zip(base, d)])
        merged["line_end"] = _round_point([q + hi * c for q, c in zip(base, d)])
        merged["target_angle"] = round(sum(candidates[i]["target_angle"] for i in idx) / len(idx), 6)
        radii = [candidates[i]["target_radius"] for i in idx if candidates[i].get("target_radius") is not None]
        merged["target_radius"] = round(sum(radii) / len(radii), 6) if radii else None
        out.append(merged)
        notes.append(f"merged {len(idx)} pieces of graph bend {model.bends[key[1]].graph_bend_id}")
    # (4) E1's own inventory: a supported graph bend that no extractor candidate is feasible for gets the mark an
    # operator would add for the wizard's UNMARKED_BEND (an extractor gap, e.g. a joggle on a narrow strip or a
    # bend whose skins are not coaxial). Unsupported graph bends stay unmarked, as the compile's rule allows.
    added = []
    marked = {k[1] for k in order if k[0] == "graph"}
    for gk in sorted(model_bends):
        if gk in marked or model_bends[gk].graph.unsupported:
            continue
        cand = _graph_bend_candidate(model_bends[gk].graph)
        if cand is not None:
            out.append(cand)
            added.append(cand["label"])
    return out, {"merge_notes": notes, "excluded_by_unsupported_rule": excluded, "added_from_e1_graph": added}


def _graph_bend_candidate(gb) -> Candidate | None:
    """The identity mark of a flange-graph bend in the native convention: the mid-radius line at the bend's
    angular middle, over its axial extent; ``target_angle`` = sweep, ``target_radius`` = Ri."""
    import numpy as np

    gid = gb.graph_bend_id
    if gb.mid_radial is None or not (0.0 < gb.sweep_deg < 180.0):
        return None
    axis = np.asarray(gb.axis_dir, dtype=float)
    p = np.asarray(gb.axis_point, dtype=float) + np.asarray(gb.mid_radial, dtype=float) * 0.5 * (gb.ri + gb.ro)
    return {"line_start": _round_point(p + axis * gb.z0), "line_end": _round_point(p + axis * gb.z1),
            "target_angle": round(float(gb.sweep_deg), 6), "target_radius": round(float(gb.ri), 6),
            "label": f"E1_GRAPH_{gid}"}


def _min_skin_gap(brep) -> float | None:
    """Thickness of a part without flange panels: the smallest radius difference of a coaxial convex/concave
    cylinder pair (the two skins of a roll); None when there is none."""
    import numpy as np

    cyls = [f for f in brep.faces if f.kind == "cylinder" and f.axis is not None]
    best = None
    for a in cyls:
        for b in cyls:
            if not (a.convex and not b.convex) or abs(float(a.axis @ b.axis)) < math.cos(math.radians(0.25)):
                continue
            rel = b.origin - a.origin
            if float(np.linalg.norm(rel - float(rel @ a.axis) * a.axis)) > 0.02:
                continue
            gap = float(a.radius - b.radius)
            if gap > 1e-3 and (best is None or gap < best):
                best = gap
    return best


def dev_candidates(cad_path: str | Path, sources: Sequence[str] = ("v10", "native")) -> tuple[list[Candidate], dict]:
    """The DEV candidate policy (module docstring): union, geometric dedupe, graph-piece merge."""
    per_source = {}
    for src in sources:
        try:
            per_source[src] = native_candidates(cad_path, src)
        except Exception as exc:  # an extractor refusing a file is evidence, not a crash
            per_source[src] = []
            per_source[f"{src}_error"] = f"{type(exc).__name__}: {exc}"[:300]
    union = union_candidates(*[per_source[s] for s in sources])
    report = {"policy": {"version": DEV_POLICY_VERSION, "sha256": dev_policy_sha256(), "sources": list(sources)},
              "counts": {s: len(per_source[s]) for s in sources}, "union": len(union),
              **{k: v for k, v in per_source.items() if k.endswith("_error")}}
    try:
        merged, extra = merge_graph_duplicates(cad_path, union)
    except Exception as exc:  # the compile will report the graph failure itself
        report["merge_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return union, report
    report.update(extra)
    report["final"] = len(merged)
    return merged, report


def dev_recipe(
    cad_path: str | Path,
    candidates: Sequence[Candidate],
    *,
    sku: str = "DEV-PART",
    part_name: str | None = None,
    inspection_stage: str = "FINAL",
    tolerances: dict[str, Any] | None = None,
    machine_moves: Sequence[Sequence[int]] = (),
    general_measurements: Sequence[dict[str, Any]] = (),
    flat_path: str | Path | None = None,
    flat_format: str | None = None,
    flat_binding: dict[str, Any] | None = None,
    thickness_mm: float | None = None,
    bend_overrides: dict[int, dict[str, Any]] | None = None,
    released_at: str = "2026-10-17T00:00:00Z",
    confirmation: str = "DEV_AUTO_CONFIRMED",
) -> PartRecipeV1:
    """A finalized PartRecipeV1 whose marks are ``candidates`` (module docstring).

    ``machine_moves`` and ``bend_overrides`` use 1-based candidate numbers;
    ``flat_binding`` is a FlatBindingV1 dict whose map values are 1-based
    candidate numbers (converted to bend_uids here).
    """
    cad_bytes = Path(cad_path).read_bytes()
    cad_sha = hashlib.sha256(cad_bytes).hexdigest()
    generation = _sha_text(json.dumps(list(candidates), sort_keys=True))
    drawing_sha = _sha_text(f"DEV_AUTO_CONFIRMED drawing placeholder for {cad_sha}")
    specs, marks = [], []
    for i, cand in enumerate(candidates, start=1):
        fid = f"dev-{i:04d}"
        label = str(cand.get("label") or f"B{i}")
        start, end = _round_point(cand["line_start"]), _round_point(cand["line_end"])
        angle, radius = float(cand["target_angle"]), cand.get("target_radius")
        radius = None if radius is None else float(radius)
        specs.append({"id": fid, "bend_id": label, "target_angle": angle, "target_radius": radius,
                      "bend_line_start": json.dumps(start), "bend_line_end": json.dumps(end)})
        marks.append({"feature_id": fid, "label": label, "source": "cad_candidate", "status": "active",
                      "review_status": "confirmed", "issue": None, "corrects_feature_id": None, "note": "",
                      "line_start": start, "line_end": end, "target_angle": angle, "target_radius": radius})
    fids = [m["feature_id"] for m in marks]
    moves = [[fids[j - 1] for j in group] for group in machine_moves]
    gms = []
    for gm in general_measurements:
        gms.append({"id": gm["id"], "label": gm.get("label", gm["id"]), "kind": gm.get("kind", "points"),
                    "bend_ids": [fids[j - 1] for j in gm.get("bend_numbers", [])],
                    "point_start": list(gm["point_start"]), "point_end": list(gm["point_end"]),
                    "cad_distance": math.dist(gm["point_start"], gm["point_end"]),
                    "target_mm": gm.get("target_mm"), "note": gm.get("note", "")})
    draft = {"schema_version": 1, "cad_file_hash": cad_sha, "drawing_file_hash": drawing_sha,
             "candidate_generation": generation, "marks": marks, "sequence": fids,
             "machine_moves": moves, "machine_moves_explicit": bool(moves), "general_measurements": gms,
             "operator_sequence_confirmed": True, "draft_revision": 1, "is_stale": False}
    overrides = {fids[j - 1]: v for j, v in (bend_overrides or {}).items()}
    release = ReleaseRequestV1(expected_draft_revision=1, default_tolerances=tolerances or DEFAULT_DEV_TOLERANCES,
                               bend_overrides=overrides, thickness_mm=thickness_mm,
                               inspection_stage=inspection_stage, confirmation=confirmation,
                               release_note="DEV recipe for compile development")
    cad = CadRef(sha256=cad_sha, original_filename=Path(cad_path).name[:255],
                 stored_rel_path=f"recipes/cad/{cad_sha}.step", candidate_generation=generation,
                 candidate_angle_convention="NATIVE_SWEEP")
    drawing = DrawingRef(sha256=drawing_sha, original_filename=None, stored_rel_path=f"recipes/drawings/{drawing_sha}.pdf")
    flat = None
    if flat_path is not None:
        flat_bytes = Path(flat_path).read_bytes()
        flat_sha = hashlib.sha256(flat_bytes).hexdigest()
        fmt = flat_format or ("DXF" if str(flat_path).lower().endswith(".dxf") else "STEP")
        ext = ".dxf" if fmt == "DXF" else ".step"
        binding = dict(flat_binding or {"state": "UNBOUND"})
        flat = {"sha256": flat_sha, "original_filename": Path(flat_path).name[:255],
                "stored_rel_path": f"recipes/flats/{flat_sha}{ext}", "format": fmt, "binding": binding}
    # uids are only known after the build; build once to learn them, then attach the flat map
    recipe = _build(sku, part_name, specs, draft, release, cad, drawing, released_at, None)
    if flat is not None:
        uid_by_number = {i: b.bend_uid for i, b in enumerate(recipe.bends, start=1)}
        binding = dict(flat["binding"])
        if binding.get("bend_line_map"):
            binding["bend_line_map"] = {lid: uid_by_number[int(n)] for lid, n in binding["bend_line_map"].items()}
        flat_ref = FlatRef(**{**flat, "binding": FlatBindingV1(**binding)})
        recipe = _build(sku, part_name, specs, draft, release, cad, drawing, released_at, flat_ref)
    return recipe


def _build(sku, part_name, specs, draft, release, cad, drawing, released_at, flat) -> PartRecipeV1:
    return build_part_recipe(
        part={"id": f"dev-{sku}", "part_number": sku, "part_name": part_name or sku, "revision": None,
              "material": None},
        spec_rows=specs, draft=draft, release=release, cad=cad, drawing=drawing, version=1,
        supersedes_version=None, released_by=ReleasedBy(user_id="dev", username="dev-auto", role="admin"),
        released_at=released_at, flat=flat)
