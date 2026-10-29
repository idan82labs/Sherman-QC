"""Pre-release helpers for the onboarding wizard and the C05 flat intake (E1 library API).

These run the same flange-graph, reconciliation and flat-binding code as the
compile, on a draft (marks not yet released), and return plain JSON-ready
dicts. Nothing is written to disk. Units mm/deg, frame ``cad_model_mm``.

- :func:`analyze_cad` - flange graph summary (panels, graph bends with OML
  sharp lines (the outer apex for a hem), sweep, Ri, UNSUPPORTED class,
  features, thickness).
- :func:`reconcile_draft` - MARK_NOT_IN_FLANGE_GRAPH / UNMARKED_BEND /
  MARK_AMBIGUOUS preview for a list of draft marks.
- :func:`propose_flat_binding` - parse a flat (DXF / flat STEP) and propose the
  ``bend_line_map`` (flat line id -> bend_uid), mirror flag, confidence,
  ``needs_operator_review``, K from the flat and the developed-length check,
  plus ``fold``: whether a release with that map would fold the flat with its
  own per-bend strips (Method B) and, if not, why. Unmarked unsupported bends
  stay rigid and restrict the datum exactly as in the compile, so the proposal
  is fitted to the development the compile will check.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from domains.production.engine.compile.brep import load_step
from domains.production.engine.compile.config import DEFAULT_SHOP_K, CompileConfig
from domains.production.engine.compile.flange_graph import GraphError, build_flange_graph
from domains.production.engine.compile.flat import parse_flat
from domains.production.engine.compile.flat_binding import fold_flat_or_reason, propose_binding
from domains.production.engine.compile.part_model import PartModel, build_part_model
from domains.production.engine.compile.reconcile import Mark, reconcile_marks

__all__ = ["load_part_model", "analyze_cad", "reconcile_draft", "propose_flat_binding"]


def _round(v) -> list[float]:
    return [round(float(c), 6) + 0.0 for c in v]


def load_part_model(cad_path: str | Path, cfg: CompileConfig | None = None) -> PartModel:
    """STEP -> rooted flange tree (raises CadReadError / GraphError)."""
    cfg = cfg or CompileConfig()
    brep = load_step(cad_path, linear_deflection=cfg.mesh_linear_deflection_mm,
                     angular_deflection=cfg.mesh_angular_deflection_rad)
    return build_part_model(build_flange_graph(brep, cfg))


def analyze_cad(cad_path: str | Path, cfg: CompileConfig | None = None) -> dict[str, Any]:
    model = load_part_model(cad_path, cfg)
    panels = [{"panel_id": p.panel_id, "area_mm2": round(p.area, 3), "is_datum": p.key == model.datum,
               "thickness_mm": None if p.graph.thickness is None else round(p.graph.thickness, 6)}
              for p in sorted(model.panels.values(), key=lambda p: int(p.panel_id[1:]))]
    bends = []
    for b in sorted(model.bends.values(), key=lambda b: b.graph_bend_id):
        a, e = b.oml_sharp()
        bends.append({"graph_bend_id": b.graph_bend_id, "parent_panel_id": model.panels[b.parent].panel_id,
                      "child_panel_id": model.panels[b.child].panel_id, "sweep_deg": round(b.sweep_deg, 6),
                      # a reflex (teardrop) hem has no included angle in [0, 180]
                      "included_deg": round(180.0 - b.sweep_deg, 6) if b.sweep_deg <= 180.0 + 1e-6 else None,
                      "inner_radius_mm": round(b.ri, 6),
                      "thickness_mm": round(b.thickness, 6), "length_mm": round(b.length, 6),
                      "direction": b.direction, "oml_virtual_sharp": {"start_mm": _round(a), "end_mm": _round(e)},
                      "unsupported": b.unsupported, "unsupported_reason": b.unsupported_reason})
    features = [{"kind": f.kind, "reason": f.reason, "panel_id": (model.panels[f.panel_key].panel_id
                                                                  if f.panel_key in model.panels else None)}
                for f in model.graph.features]
    return {"thickness_mm": round(model.thickness, 6), "datum_panel_id": model.panels[model.datum].panel_id,
            "panels": panels, "graph_bends": bends, "features": features}


def reconcile_draft(cad_path: str | Path, marks: Sequence[Mark], cfg: CompileConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CompileConfig()
    model = load_part_model(cad_path, cfg)
    rec = reconcile_marks(model, list(marks), cfg)
    gid = {k: b.graph_bend_id for k, b in model.bends.items()}
    return {
        "matches": {uid: gid[k] for uid, k in sorted(rec.matches.items())},
        "mark_not_in_flange_graph": rec.unmatched_marks,
        "mark_ambiguous": [{"bend_uid": u, "graph_bend_ids": g} for u, g in rec.ambiguous_marks]
        + [{"graph_bend_id": gid[k], "bend_uids": u} for k, u in rec.ambiguous_graph],
        "unmarked_bend": [gid[k] for k in rec.unmarked_supported],
        "unmarked_unsupported": [gid[k] for k in rec.unmarked_unsupported],
        "blocked": rec.blocked,
    }


def propose_flat_binding(cad_path: str | Path, marks: Sequence[Mark], flat_path: str | Path, fmt: str | None = None,
                         cfg: CompileConfig | None = None, shop_k: float = DEFAULT_SHOP_K) -> dict[str, Any]:
    """Binding proposal for draft ``marks`` (sequence order) and a flat file.

    ``shop_k`` (with ``cfg.shop_k_uncertainty``) sets the plausible K band the
    proposal's K and developed-length check are held to, as in the compile.
    """
    cfg = cfg or CompileConfig()
    model = load_part_model(cad_path, cfg)
    rec = reconcile_marks(model, list(marks), cfg)
    if rec.blocked or len(rec.matches) != len(marks):
        return {"status": "RECONCILIATION_BLOCKED", "reconciliation": {
            "mark_not_in_flange_graph": rec.unmatched_marks, "unmarked_bend": len(rec.unmarked_supported)}}
    if rec.unmarked_unsupported:
        # exactly as the compile: unmarked unsupported bends stay rigid, which restricts the datum
        # (and so the development the flat is fitted to) to the component holding every mark
        try:
            model = build_part_model(model.graph, set(rec.unmarked_unsupported))
        except GraphError as exc:
            return {"status": "RECONCILIATION_BLOCKED", "reconciliation": {
                "mark_not_in_flange_graph": [], "unmarked_bend": 0, "flange_graph": f"{exc.code}: {exc.detail}"}}
    parsed = parse_flat(flat_path, fmt)
    order = [rec.matches[m.bend_uid] for m in marks]
    uid_of = {gk: m.bend_uid for gk, m in zip(order, marks)}
    proposal = propose_binding(model, order, uid_of, parsed, cfg, shop_k=shop_k)
    # what a release with this map would stage: Method B (the flat's own strips) or why not
    fold = fold_flat_or_reason(model, order, uid_of, parsed, proposal, proposal.bend_line_map, cfg, shop_k=shop_k)
    return {"status": "PROPOSED", "flat": parsed.summary(), "proposal": proposal.to_json(),
            "fold": fold.to_json(uid_of)}
