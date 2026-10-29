"""Deterministic generator of the contract examples for the synthetic part SYN-HAT-1.

SYN-HAT-1 is an asymmetric 4-bend hat section, t = 2 mm, Ri = 2 mm, 100 mm
long (Y), all bends 90 deg included. Outer (skin 0) top at z = 30 between the
B2 and B3 OML sharps (x = 0 and x = 40); walls 30 mm; left flange 20 mm,
right flange 30 mm (so the part is not self-symmetric). Bend order B1..B4,
one מהלך grouping B3+B4. Datum panel P2 (top).

``build_examples()`` is pure and returns ``{file name: model}``; the test
suite checks the committed JSON equals it. ``python -m
contracts.production.examples.generate_examples --out DIR`` writes the files.
All hashes/paths of binary assets are illustrative: the ``.npz``/``.ply``
files themselves are not part of the examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from contracts.production.characteristics import make_characteristic_id
from contracts.production.compiled_part import (
    AxisV1,
    BendLociV1,
    CharacteristicNominalV1,
    CompiledBendV1,
    CompiledPartV1,
    FlangeSideV1,
    FreeEdgeV1,
    LineLocusV1,
    MeshRefV1,
    PanelV1,
    PlaneV1,
    SampleCloudRefV1,
    StageModelV1,
    SymmetryCandidateV1,
    SymmetryTestV1,
)
from contracts.production.common import ToleranceRef
from contracts.production.inspection_result import (
    BendDisplayV1,
    BendRowV1,
    BendStateEvidenceV1,
    CharacteristicEvidenceV1,
    CharacteristicResultV1,
    CoverageV1,
    EngineInfoV1,
    GeneralMeasurementRowV1,
    InspectionValidityV1,
    RecipeRefV1,
    RegistrationV1,
    ResultContextV1,
    ScanInfoV1,
    ShopfloorInspectionResultV1,
    StageDecisionV1,
    StationLatestCompletedV1,
    StationFeedV1,
    StationStatsV1,
    ViewerAssetsV1,
    derive_stage_decision,
    derive_summary,
)
from contracts.production.line_job import JobFailureV1, JobResultSummaryV1, LineJobSummaryV1, LineJobV1
from contracts.production.part_recipe import (
    CadRef,
    DrawingRef,
    PartRecipeV1,
    ReleasedBy,
    ReleaseRequestV1,
)

__all__ = ["build_examples", "write_examples", "EXAMPLE_FILES"]

T = 2.0
RI = 2.0
RO = RI + T
K = 0.33
LENGTH_Y = 100.0
SWEEP = 90.0
NEUTRAL = math.radians(SWEEP) * (RI + K * T)
BEND_DEDUCTION = 2 * RO * math.tan(math.radians(SWEEP / 2)) - NEUTRAL

EXAMPLE_FILES = (
    "part_recipe_v1.example.json",
    "compiled_part_v1.example.json",
    "shopfloor_result_v1.example.json",
    "shopfloor_result_v1.in_process.example.json",
    "shopfloor_result_v1.incomplete.example.json",
    "shopfloor_result_v1.invalid.example.json",
    "station_feed_v1.example.json",
    "line_job_v1.example.json",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _r(v: float) -> float:
    out = round(v, 6)
    return 0.0 if out == 0 else out


def _v(*xs: float) -> tuple[float, float, float]:
    return (_r(xs[0]), _r(xs[1]), _r(xs[2]))


def _add(a, b, s=1.0):
    return tuple(x + s * y for x, y in zip(a, b))


def _unit(a):
    n = math.sqrt(sum(x * x for x in a))
    return tuple(x / n for x in a)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _line_xz(x: float, z: float) -> LineLocusV1:
    return LineLocusV1(start_mm=_v(x, 0, z), end_mm=_v(x, LENGTH_Y, z))


# --------------------------------------------------------------------------
# Part geometry (final pose, cad_model_mm)
# --------------------------------------------------------------------------

#: name -> (oml sharp (x, z), oml-skin normal of parent, of child, parent panel, child panel,
#:          parent outward (away from bend), child outward)
_BENDS = {
    "B1": ((2.0, 0.0), (1, 0, 0), (0, 0, -1), "P1", "P0", (0, 0, 1), (-1, 0, 0)),
    "B2": ((0.0, 30.0), (0, 0, 1), (-1, 0, 0), "P2", "P1", (1, 0, 0), (0, 0, -1)),
    "B3": ((40.0, 30.0), (0, 0, 1), (1, 0, 0), "P2", "P3", (-1, 0, 0), (0, 0, -1)),
    "B4": ((38.0, 0.0), (-1, 0, 0), (0, 0, -1), "P3", "P4", (0, 0, 1), (1, 0, 0)),
}
_OML_SKIN = {"B1": 1, "B2": 0, "B3": 0, "B4": 1}
#: (side A panel, side B panel, A boundary, B boundary, A adjacent, B adjacent, A len, B len)
_SIDES = {
    "B1": ("P1", "P0", "ADJACENT_BEND", "FREE_EDGE", "B2", None, 30.0, 20.0),
    "B2": ("P2", "P1", "ADJACENT_BEND", "ADJACENT_BEND", "B3", "B1", 40.0, 30.0),
    "B3": ("P3", "P2", "ADJACENT_BEND", "ADJACENT_BEND", "B4", "B2", 30.0, 40.0),
    "B4": ("P4", "P3", "FREE_EDGE", "ADJACENT_BEND", None, "B3", 30.0, 30.0),
}
_LABELS = ("B1", "B2", "B3", "B4")
_PANELS = {
    # id: (skin0 point, skin0 normal, skin1 point, skin1 normal, boundary corners (x,z) pairs at y=0/100)
    "P0": ((-18, 0, 2), (0, 0, 1), (-18, 0, 0), (0, 0, -1), [(-18, 2), (0, 2)]),
    "P1": ((0, 0, 15), (-1, 0, 0), (2, 0, 15), (1, 0, 0), [(0, 2), (0, 30)]),
    "P2": ((20, 0, 30), (0, 0, 1), (20, 0, 28), (0, 0, -1), [(0, 30), (40, 30)]),
    "P3": ((40, 0, 15), (1, 0, 0), (38, 0, 15), (-1, 0, 0), [(40, 30), (40, 2)]),
    "P4": ((53, 0, 2), (0, 0, 1), (53, 0, 0), (0, 0, -1), [(38, 2), (68, 2)]),
}


def _center(name: str):
    (x, z), n_p, n_c, *_ = _BENDS[name]
    s = (x, 0.0, z)
    return _add(s, _add(n_p, n_c), -RO)


def _bend_mark_line(name: str) -> tuple[list[float], list[float]]:
    """Native mid-radius half-angular-turn line, like vendor_native_step_upload_v1."""
    (x, z), n_p, n_c, *_ = _BENDS[name]
    c = _center(name)
    apex_dir = _unit(_add(n_p, n_c))
    p = _add(c, apex_dir, RI + T / 2)
    return [_r(p[0]), 0.0, _r(p[2])], [_r(p[0]), LENGTH_Y, _r(p[2])]


def _loci(name: str) -> BendLociV1:
    (x, z), n_p, n_c, *_ = _BENDS[name]
    c = _center(name)
    apex = _unit(_add(n_p, n_c))

    def at(point):
        return _line_xz(point[0], point[2])

    sharp = (x, 0.0, z)
    return BendLociV1(
        cylinder_axis=at(c),
        inner_tangent_parent=at(_add(c, n_p, RI)),
        inner_tangent_child=at(_add(c, n_c, RI)),
        outer_tangent_parent=at(_add(c, n_p, RO)),
        outer_tangent_child=at(_add(c, n_c, RO)),
        iml_virtual_sharp=at(_add(sharp, _add(n_p, n_c), -T)),
        oml_virtual_sharp=at(sharp),
        inner_apex=at(_add(c, apex, RI)),
        outer_apex=at(_add(c, apex, RO)),
    )


def _axis(name: str) -> AxisV1:
    *_, p_out, c_out = _BENDS[name]
    d = _unit(_cross(tuple(-v for v in p_out), c_out))
    c = _center(name)
    return AxisV1(point_mm=_v(c[0], 0.0, c[2]), direction=_v(*d))


# 4x4 helpers (row-major nested tuples)
def _rot(axis: AxisV1, angle_deg: float):
    d = axis.direction
    p = axis.point_mm
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = d
    r = [
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ]
    t = [p[i] - sum(r[i][j] * p[j] for j in range(3)) for i in range(3)]
    return [r[0] + [t[0]], r[1] + [t[1]], r[2] + [t[2]], [0.0, 0.0, 0.0, 1.0]]


def _eye():
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _apply(m, p):
    return tuple(sum(m[i][j] * (list(p) + [1.0])[j] for j in range(4)) for i in range(3))


def _apply_dir(m, d):
    return tuple(sum(m[i][j] * d[j] for j in range(3)) for i in range(3))


def _mat(m):
    return tuple(tuple(_r(v) for v in row) for row in m)


#: path of bends from the datum (P2) to each panel, datum side first
_PATH = {"P2": [], "P1": ["B2"], "P0": ["B2", "B1"], "P3": ["B3"], "P4": ["B3", "B4"]}


def _stage_transforms(formed: set[str]):
    out = {}
    for panel, path in _PATH.items():
        m = _eye()
        for name in path:  # U_b1 . U_b2 ... (deepest applied first)
            if name not in formed:
                # Method A: rotate the child back by the sweep about the bend axis
                # (child tangent lands on the parent tangent), then shift it by the
                # neutral strip length away from the parent panel.
                p_out = _BENDS[name][5]
                shift = _eye()
                for i in range(3):
                    shift[i][3] = -p_out[i] * NEUTRAL
                m = _mul(m, _mul(shift, _rot(_axis(name), -SWEEP)))
        out[panel] = m
    return out


def _stage_lines(uids: dict[str, str], formed: set[str], transforms) -> dict[str, LineLocusV1]:
    lines = {}
    for name in _LABELS:
        _, n_p, n_c, parent, child, p_out, c_out = _BENDS[name]
        m = transforms[parent]
        loci = _loci(name)
        if name in formed:
            base = loci.oml_virtual_sharp
            a, b = _apply(m, base.start_mm), _apply(m, base.end_mm)
        else:
            tangent = loci.outer_tangent_parent
            shift = _apply_dir(m, tuple(-v for v in p_out))
            a = _add(_apply(m, tangent.start_mm), shift, NEUTRAL / 2)
            b = _add(_apply(m, tangent.end_mm), shift, NEUTRAL / 2)
        lines[uids[name]] = LineLocusV1(start_mm=_v(*a), end_mm=_v(*b))
    return lines


# --------------------------------------------------------------------------
# Recipe
# --------------------------------------------------------------------------

CAD_SHA = _sha("SYN-HAT-1 synthetic folded STEP")
DRAWING_SHA = _sha("SYN-HAT-1 synthetic drawing PDF")
GENERATION = _sha("SYN-HAT-1 candidate generation")
RELEASED_AT = "2026-10-17T06:00:00Z"


def _draft() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    marks, specs = [], []
    for i, name in enumerate(_LABELS, start=1):
        start, end = _bend_mark_line(name)
        fid = f"spec-{i:04d}"
        specs.append({
            "id": fid, "part_id": "part-syn-hat-1", "bend_id": name, "target_angle": SWEEP, "target_radius": RI,
            "tolerance_angle": None, "tolerance_radius": None,
            "bend_line_start": json.dumps(start), "bend_line_end": json.dumps(end), "notes": None,
        })
        marks.append({
            "feature_id": fid, "label": name, "source": "cad_candidate", "status": "active",
            "review_status": "confirmed", "issue": None, "corrects_feature_id": None, "note": "",
            "line_start": start, "line_end": end, "target_angle": SWEEP, "target_radius": RI,
        })
    gm = {
        "id": "gm-overall-width", "label": "Overall width", "kind": "points", "bend_ids": [],
        "point_start": [-18.0, 50.0, 2.0], "point_end": [68.0, 50.0, 2.0], "cad_distance": 86.0,
        "target_mm": None, "note": "free edge to free edge",
    }
    draft = {
        "schema_version": 1, "cad_file_hash": CAD_SHA, "cad_uploaded_at": "2026-10-17T05:00:00",
        "part_revision": "A", "drawing_file_hash": DRAWING_SHA, "candidate_generation": GENERATION,
        "expected_draft_revision": 2, "marks": marks, "sequence": [m["feature_id"] for m in marks],
        "machine_moves": [["spec-0003", "spec-0004"]], "machine_moves_explicit": True,
        "general_measurements": [gm], "operator_sequence_confirmed": True, "completed_steps": 0,
        "matching_search": {"search_radius_mm": None, "search_angle_deg": None},
        "draft_revision": 3, "is_stale": False, "updated_at": "2026-10-17T05:30:00",
    }
    return draft, specs


def build_recipe() -> PartRecipeV1:
    from domains.production.recipe.builder import build_part_recipe  # domain builds the example

    draft, specs = _draft()
    release = ReleaseRequestV1(
        expected_draft_revision=3,
        default_tolerances={
            "angle_deg": {"plus": 1.0, "minus": 1.0},
            "flange_mm": {"plus": 0.5, "minus": 0.5},
            "bend_line_position_mm": {"plus": 0.5, "minus": 0.5},
            "backgauge_mm": {"plus": 0.3, "minus": 0.3},
            "general_mm": {"plus": 0.8, "minus": 0.8},
            "surface_profile_mm": 0.5,
            "source": "SHOP_STANDARD",
            "note": "synthetic example",
        },
        thickness_mm=T,
        inspection_stage="IN_PROCESS",
        release_note="Synthetic contract example",
    )
    return build_part_recipe(
        part={"id": "part-syn-hat-1", "part_number": "SYN-HAT-1", "part_name": "Synthetic hat section",
              "revision": "A", "material": "S235 2.0 mm"},
        spec_rows=specs,
        draft=draft,
        release=release,
        cad=CadRef(sha256=CAD_SHA, original_filename="SYN-HAT-1.step",
                   stored_rel_path=f"recipes/cad/{CAD_SHA}.step", candidate_generation=GENERATION,
                   candidate_angle_convention="NATIVE_SWEEP"),
        drawing=DrawingRef(sha256=DRAWING_SHA, original_filename="SYN-HAT-1.pdf",
                           stored_rel_path=f"recipes/drawings/{DRAWING_SHA}.pdf"),
        version=1,
        supersedes_version=None,
        released_by=ReleasedBy(user_id="u-admin", username="admin", role="admin"),
        released_at=RELEASED_AT,
    )


# --------------------------------------------------------------------------
# Compiled part
# --------------------------------------------------------------------------


def _tol(recipe: PartRecipeV1, uid: str, kind: str) -> ToleranceRef | None:
    tol = recipe.resolved_tolerance(uid, kind)
    if tol is None:
        return None
    override = getattr(recipe.bend(uid).tolerances, kind) is not None
    return ToleranceRef(plus=tol.plus, minus=tol.minus,
                        origin="RECIPE_BEND_OVERRIDE" if override else "RECIPE_DEFAULT",
                        standard=recipe.default_tolerances.source)


def _backgauge_nominals() -> dict[int, float]:
    """Developed length from the just-formed bend's OML to the far free edge (right side)."""
    return {
        1: 30 + 40 + 30 + 30 - 3 * BEND_DEDUCTION,
        2: 40 + 30 + 30 - 2 * BEND_DEDUCTION,
        3: 30 + 30 - 1 * BEND_DEDUCTION,
    }


def build_compiled(recipe: PartRecipeV1) -> CompiledPartV1:
    uids = {b.label: b.bend_uid for b in recipe.bends}
    by_label = {b.label: b for b in recipe.bends}
    panels = []
    for pid, (p0, n0, p1, n1, corners) in _PANELS.items():
        (xa, za), (xb, zb) = corners
        boundary = (_v(xa, 0, za), _v(xb, 0, zb), _v(xb, LENGTH_Y, zb), _v(xa, LENGTH_Y, za))
        width = math.dist((xa, za), (xb, zb))
        free = []
        if pid == "P0":
            free.append(FreeEdgeV1(edge_id="E0", polyline_mm=(_v(-18, 0, 2), _v(-18, LENGTH_Y, 2))))
        if pid == "P4":
            free.append(FreeEdgeV1(edge_id="E1", polyline_mm=(_v(68, 0, 2), _v(68, LENGTH_Y, 2))))
        panels.append(PanelV1(
            panel_id=pid, skin0_plane=PlaneV1(point_mm=_v(*p0), normal=_v(*n0)),
            skin1_plane=PlaneV1(point_mm=_v(*p1), normal=_v(*n1)), thickness_mm=T,
            area_mm2=_r(width * LENGTH_Y), centroid_mm=_v((xa + xb) / 2, LENGTH_Y / 2, (za + zb) / 2),
            outer_boundary_mm=boundary, free_edges=tuple(free), is_datum=(pid == "P2"),
        ))
    bends = []
    for name in _LABELS:
        rb = by_label[name]
        _, n_p, n_c, parent, child, p_out, c_out = _BENDS[name]
        a_panel, b_panel, a_kind, b_kind, a_adj, b_adj, _, _ = _SIDES[name]
        out_by_panel = {parent: p_out, child: c_out}
        bends.append(CompiledBendV1(
            bend_uid=rb.bend_uid, sequence_index=rb.sequence_index, label=rb.label,
            machine_move_index=rb.machine_move_index, graph_bend_id=f"G{rb.sequence_index}",
            parent_panel_id=parent, child_panel_id=child, inner_radius_mm=RI, thickness_mm=T,
            sweep_deg=SWEEP, nominal_included_deg=180 - SWEEP, direction="DOWN" if _OML_SKIN[name] == 0 else "UP",
            oml_skin=_OML_SKIN[name], axis=_axis(name), loci=_loci(name), bend_length_mm=LENGTH_Y,
            neutral_length_mm=_r(NEUTRAL),
            sides=(
                FlangeSideV1(side="A", panel_id=a_panel, outward_dir=_v(*out_by_panel[a_panel]),
                             boundary_kind=a_kind, adjacent_bend_uid=uids[a_adj] if a_adj else None),
                FlangeSideV1(side="B", panel_id=b_panel, outward_dir=_v(*out_by_panel[b_panel]),
                             boundary_kind=b_kind, adjacent_bend_uid=uids[b_adj] if b_adj else None),
            ),
        ))
    chars = []
    for name in _LABELS:
        uid = uids[name]
        _, _, a_kind, b_kind, a_adj, b_adj, a_len, b_len = _SIDES[name]
        for side, kind_b, adj, length, tol_key in (("A", a_kind, a_adj, a_len, "flange_a_mm"),
                                                   ("B", b_kind, b_adj, b_len, "flange_b_mm")):
            chars.append(CharacteristicNominalV1(
                characteristic_id=make_characteristic_id("FLANGE_LENGTH", bend_uid=uid, side=side),
                kind="FLANGE_LENGTH", bend_uid=uid, side=side, unit="mm", nominal=length,
                convention="OML_SHARP_TO_FREE_EDGE" if kind_b == "FREE_EDGE" else "OML_SHARP_TO_OML_SHARP",
                boundary_kind=kind_b, adjacent_bend_uid=uids[adj] if adj else None,
                tolerance=_tol(recipe, uid, tol_key), recipe_requirement="REQUIRED"))
        chars.append(CharacteristicNominalV1(
            characteristic_id=make_characteristic_id("INCLUDED_ANGLE", bend_uid=uid), kind="INCLUDED_ANGLE",
            bend_uid=uid, unit="deg", nominal=180 - SWEEP, convention="INCLUDED_DEG",
            tolerance=_tol(recipe, uid, "angle_deg"), recipe_requirement="REQUIRED"))
        chars.append(CharacteristicNominalV1(
            characteristic_id=make_characteristic_id("BEND_LINE_POSITION", bend_uid=uid),
            kind="BEND_LINE_POSITION", bend_uid=uid, unit="mm", nominal=0.0,
            convention="OML_SHARP_OFFSET_ALONG_SIDE_A", tolerance=_tol(recipe, uid, "bend_line_position_mm"),
            recipe_requirement="REQUIRED"))
    for k, length in _backgauge_nominals().items():
        uid = uids[_LABELS[k - 1]]
        chars.append(CharacteristicNominalV1(
            characteristic_id=make_characteristic_id("BACKGAUGE_LENGTH", bend_uid=uid, stage_index=k),
            kind="BACKGAUGE_LENGTH", bend_uid=uid, side="A", stage_index=k, unit="mm", nominal=_r(length),
            convention="OML_SHARP_TO_FAR_BOUNDARY_AT_STAGE", boundary_kind="FREE_EDGE",
            tolerance=_tol(recipe, uid, "backgauge_mm"), recipe_requirement="REQUIRED"))
    for gm in recipe.general_measurements:
        tol = gm.tolerance or recipe.default_tolerances.general_mm or recipe.default_tolerances.flange_mm
        chars.append(CharacteristicNominalV1(
            characteristic_id=make_characteristic_id("GENERAL_MEASUREMENT", measurement_id=gm.measurement_id),
            kind="GENERAL_MEASUREMENT", measurement_id=gm.measurement_id, unit="mm", nominal=gm.nominal_mm,
            convention="CAD_STRAIGHT_LINE",
            tolerance=ToleranceRef(plus=tol.plus, minus=tol.minus,
                                   origin="RECIPE_MEASUREMENT_OVERRIDE" if gm.tolerance else "RECIPE_DEFAULT",
                                   standard=recipe.default_tolerances.source),
            recipe_requirement="REQUIRED" if gm.required else "OPTIONAL"))
    stages = []
    order = [uids[n] for n in _LABELS]
    for k in range(len(_LABELS) + 1):
        formed = set(_LABELS[:k])
        transforms = _stage_transforms(formed)
        sid = f"S{k}"
        stages.append(StageModelV1(
            stage_model_id=sid, stage_index=k, kind="SEQUENCE_PREFIX", formed_bend_uids=tuple(order[:k]),
            method="METHOD_A_ROTATE_BACK", k_factor_used=K,
            panel_transforms={p: _mat(m) for p, m in transforms.items()},
            bend_lines=_stage_lines(uids, formed, transforms),
            sample_cloud=SampleCloudRefV1(rel_path=f"stages/{sid}.npz", sha256=_sha(f"cloud {sid}"),
                                          point_count=74000 + 10 * k),
            display_mesh=MeshRefV1(rel_path=f"stages/{sid}.ply", sha256=_sha(f"mesh {sid}"),
                                   vertex_count=4200, face_count=8300),
            # The flat blank S0 is a plain rectangle: it registers onto itself
            # (symmetric) although no bend line maps onto another bend line.
            symmetry=SymmetryTestV1(
                method="SELF_REGISTRATION_ICP_V1",
                candidates=(SymmetryCandidateV1(
                    kind="ROTATION",
                    transform=_mat(_rot(AxisV1(point_mm=(25.0, 50.0, 15.0), direction=(0.0, 0.0, 1.0)), 180)),
                    rms_mm=0.0 if k == 0 else _r(3.1 + 0.4 * k),
                ),),
                is_symmetric=(k == 0),
            ),
        ))
    return CompiledPartV1.finalize(
        status="COMPILED",
        recipe={"recipe_id": recipe.recipe_id, "version": recipe.recipe_version,
                "recipe_sha256": recipe.recipe_sha256, "content_sha256": recipe.content_sha256},
        cad_sha256=recipe.cad.sha256,
        compiler={"version": "e1-example"},
        compiled_at="2026-10-17T06:00:05Z",
        thickness={"t_mm": T, "source": "MEASURED_BOTH_SKINS"},
        k_factor={"value": K, "source": "SHOP_DEFAULT", "uncertainty": 0.05},
        datum_panel_id="P2",
        panels=panels,
        bends=bends,
        characteristics=chars,
        stages=stages,
        issues=[{"code": "K_FACTOR_DEFAULTED", "severity": "WARNING", "detail": "no flat bound; shop K 0.33"},
                {"code": "FLAT_NOT_BOUND", "severity": "WARNING"}],
        diagnostics={"compile_seconds": 1.4},
    )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

IDENTITY = tuple(tuple(1.0 if i == j else 0.0 for j in range(4)) for i in range(4))
RAW_TO_FRAME = ((0.0, -1.0, 0.0, 412.25), (1.0, 0.0, 0.0, -87.5), (0.0, 0.0, 1.0, 3.125), (0.0, 0.0, 0.0, 1.0))
SCAN_SHA = _sha("SYN-HAT-1 example scan bytes")
JOB_ID = "line_" + _sha("job example")[:32]


def _char(
    compiled: CompiledPartV1, cid: str, *, actual: float | None = None, u: float | None = None,
    state: str = "MEASURED", reason: str | None = None, stations: int = 7,
) -> CharacteristicResultV1:
    nom = compiled.nominal(cid)
    source = ("STAGE_MODEL_NOMINAL" if nom.kind == "BACKGAUGE_LENGTH"
              else "NOMINAL_LOCUS_REFERENCE" if nom.kind == "BEND_LINE_POSITION"
              else "CAD_DISTANCE" if nom.kind == "GENERAL_MEASUREMENT" else "COMPILED_NOMINAL")
    if state == "NOT_APPLICABLE":
        return CharacteristicResultV1.not_applicable(nom, reason=reason, expected_source=source)
    if state == "NOT_MEASURED":
        return CharacteristicResultV1.not_measured(nom, reason=reason, expected_source=source,
                                                   required=nom.tolerance is not None)
    spread = max(u, 0.02)
    return CharacteristicResultV1.measured(
        nom, actual=actual, uncertainty=u, expected_source=source, required=True,
        stations=StationStatsV1(count=stations, median=actual, min=actual - spread / 2, max=actual + spread / 2),
        evidence=CharacteristicEvidenceV1(method="SECTION_METROLOGY_V1", support_points=1800, coverage_ratio=0.93,
                                          fit_rms_mm=0.03),
    )


def _row(compiled: CompiledPartV1, label: str, state: str, chars: list[CharacteristicResultV1],
         *, measured_included: float | None = None, reasons: tuple[str, ...] = (), stage_id: str = "S4"):
    bend = next(b for b in compiled.bends if b.label == label)
    stage = compiled.stage(stage_id)
    line = stage.bend_lines[bend.bend_uid]
    position_measured = chars[3].state == "MEASURED"
    offset = chars[3].deviation if position_measured else 0.0
    side_a = bend.sides[0].outward_dir
    measured = (tuple(_v(*_add(p, side_a, offset)) for p in (line.start_mm, line.end_mm))
                if position_measured else None)
    mid = tuple((a + b) / 2 for a, b in zip(line.start_mm, line.end_mm))
    return BendRowV1(
        bend_uid=bend.bend_uid, label=bend.label, sequence_index=bend.sequence_index,
        machine_move_index=bend.machine_move_index, state=state, state_reasons=reasons,
        state_evidence=BendStateEvidenceV1(
            nominal_included_deg=bend.nominal_included_deg, measured_included_deg=measured_included,
            formed_fraction=(None if measured_included is None
                             else _r((180 - measured_included) / bend.sweep_deg)),
            coverage_ratio=0.9 if state != "UNKNOWN" else 0.12,
        ),
        characteristics=tuple(chars),
        display=BendDisplayV1(
            nominal_polyline_mm=(line.start_mm, line.end_mm), measured_polyline_mm=measured,
            label_anchor_mm=_v(mid[0], mid[1], mid[2] + 6.0),
            side_a_anchor_mm=_v(*_add(mid, side_a, 10.0)), side_b_anchor_mm=_v(*_add(mid, bend.sides[1].outward_dir, 10.0)),
        ),
        diagnostics={"inner_radius_mm": 2.04} if state == "FORMED" else {},
    )


def _ids(compiled: CompiledPartV1, label: str) -> list[str]:
    uid = next(b.bend_uid for b in compiled.bends if b.label == label)
    return [make_characteristic_id("FLANGE_LENGTH", bend_uid=uid, side="A"),
            make_characteristic_id("FLANGE_LENGTH", bend_uid=uid, side="B"),
            make_characteristic_id("INCLUDED_ANGLE", bend_uid=uid),
            make_characteristic_id("BEND_LINE_POSITION", bend_uid=uid)]


def _result(recipe: PartRecipeV1, compiled: CompiledPartV1, *, rows, gms=(), validity=None, declared=None,
            stage_model_id="S4", viewer=None, serial="SN0042") -> ShopfloorInspectionResultV1:
    validity = validity or InspectionValidityV1(
        state="VALID",
        registration=RegistrationV1(method="DATUM_FPFH_RANSAC_TRIMMED_ICP_V1", raw_to_frame=RAW_TO_FRAME,
                                    datum_fitness=0.94, rmse_mm=0.06, pose_candidates_considered=8,
                                    pose_margin=2.7),
        coverage=CoverageV1(surface_coverage_ratio=0.86, datum_coverage_ratio=0.97, scan_point_count=812345),
    )
    stage_fields = derive_stage_decision(
        inspection_stage=recipe.inspection_stage, declared_stage=declared, bends=rows,
        pose_ambiguous=validity.registration.pose_ambiguous, validity_state=validity.state)
    stage = StageDecisionV1(
        inspection_stage=recipe.inspection_stage, declared_stage=declared,
        declared_stage_source="SIDECAR" if declared is not None else None,
        stage_model_id=stage_model_id, **stage_fields)
    summary = derive_summary(validity_state=validity.state, bends=rows, general_measurements=gms,
                             stage_decision=stage.decision, stage_status=stage.status)
    viewer = viewer or ViewerAssetsV1(
        registered_scan="registered_scan.ply", point_count=812345, deviations="deviations.f32",
        deviation_reference_stage=stage_model_id, heatmap_band_mm=recipe.default_tolerances.surface_profile_mm,
        segment_labels="segment_labels.u16", stage_mesh="display/stage_mesh.ply",
        final_mesh="display/final_mesh.ply", bend_geometry="bend_geometry.json")
    return ShopfloorInspectionResultV1(
        created_at="2026-10-17T07:15:42Z",
        context=ResultContextV1(job_id=JOB_ID, station_id="cell-1", scan_file_name="SYN-HAT-1_A_SN0042.ply",
                                serial=serial, received_at="2026-10-17T07:15:00Z"),
        sku=recipe.sku, part_name=recipe.part_name,
        recipe=RecipeRefV1(recipe_id=recipe.recipe_id, version=recipe.recipe_version,
                           recipe_sha256=recipe.recipe_sha256, content_sha256=recipe.content_sha256),
        scan=ScanInfoV1(sha256=SCAN_SHA, format="PLY_BINARY_LE", vertex_count=812345, face_count=1620011),
        engine=EngineInfoV1(engine_version="e3-example", compiled_part_sha256=compiled.compiled_part_sha256,
                            bundle_id="bundle-example", algorithm_versions={"registration": "datum_v1",
                                                                            "metrology": "section_v1"},
                            duration_s=23.4, peak_rss_bytes=1_900_000_000),
        inspection_validity=validity, stage_decision=stage, summary=summary, bends=tuple(rows),
        general_measurements=tuple(gms), viewer_assets=viewer,
        warnings=("SYNTHETIC CONTRACT EXAMPLE",),
    )


def _gm_row(compiled: CompiledPartV1, recipe: PartRecipeV1, *, actual=None, u=None, state="MEASURED", reason=None):
    gm = recipe.general_measurements[0]
    cid = make_characteristic_id("GENERAL_MEASUREMENT", measurement_id=gm.measurement_id)
    ch = _char(compiled, cid, actual=actual, u=u, state=state, reason=reason)
    measured = state == "MEASURED"
    return GeneralMeasurementRowV1(
        measurement_id=gm.measurement_id, label=gm.label, characteristic=ch, nominal_start_mm=gm.point_start_mm,
        nominal_end_mm=gm.point_end_mm,
        measured_start_mm=_v(-18.05, 50.0, 2.0) if measured else None,
        measured_end_mm=_v(68.1, 50.0, 2.0) if measured else None)


def build_results(recipe: PartRecipeV1, compiled: CompiledPartV1) -> dict[str, ShopfloorInspectionResultV1]:
    out: dict[str, ShopfloorInspectionResultV1] = {}
    nom = {c.characteristic_id: c.nominal for c in compiled.characteristics}

    # 1) Final scan, OUT_OF_TOLERANCE_FOUND: PASS / FAIL / REVIEW / UNKNOWN(EDGE_NOT_SEEN).
    b1, b2, b3, b4 = (_ids(compiled, n) for n in _LABELS)
    rows = [
        _row(compiled, "B1", "FORMED", [
            _char(compiled, b1[0], actual=nom[b1[0]] + 0.08, u=0.09),
            _char(compiled, b1[1], actual=nom[b1[1]] - 0.12, u=0.10),
            _char(compiled, b1[2], actual=90.3, u=0.15),
            _char(compiled, b1[3], actual=0.11, u=0.12)], measured_included=90.3),
        _row(compiled, "B2", "FORMED", [
            _char(compiled, b2[0], actual=nom[b2[0]] + 0.05, u=0.09),
            _char(compiled, b2[1], actual=nom[b2[1]] + 0.10, u=0.09),
            _char(compiled, b2[2], actual=88.2, u=0.2),
            _char(compiled, b2[3], actual=-0.06, u=0.12)], measured_included=88.2),
        _row(compiled, "B3", "FORMED", [
            _char(compiled, b3[0], actual=nom[b3[0]] + 0.45, u=0.10),
            _char(compiled, b3[1], actual=nom[b3[1]] - 0.02, u=0.09),
            _char(compiled, b3[2], actual=89.6, u=0.15),
            _char(compiled, b3[3], actual=0.20, u=0.12)], measured_included=89.6),
        _row(compiled, "B4", "FORMED", [
            _char(compiled, b4[0], state="NOT_MEASURED", reason="EDGE_NOT_SEEN"),
            _char(compiled, b4[1], actual=nom[b4[1]] + 0.07, u=0.09),
            _char(compiled, b4[2], actual=90.4, u=0.15),
            _char(compiled, b4[3], actual=0.04, u=0.12)], measured_included=90.4),
    ]
    out["shopfloor_result_v1.example.json"] = _result(
        recipe, compiled, rows=rows, gms=[_gm_row(compiled, recipe, actual=86.15, u=0.15)])

    # 2) In-process stage 2 (B1, B2 formed; B3+B4 move not yet formed), all PASS incl. backgauge.
    bg2 = make_characteristic_id("BACKGAUGE_LENGTH", bend_uid=rows[1].bend_uid, stage_index=2)
    nyf = lambda ids: [_char(compiled, i, state="NOT_APPLICABLE", reason="BEND_NOT_YET_FORMED") for i in ids]
    rows_ip = [
        _row(compiled, "B1", "FORMED", [
            _char(compiled, b1[0], actual=nom[b1[0]] + 0.06, u=0.09),
            _char(compiled, b1[1], actual=nom[b1[1]] - 0.10, u=0.10),
            _char(compiled, b1[2], actual=90.2, u=0.15),
            _char(compiled, b1[3], actual=0.08, u=0.12)], measured_included=90.2, stage_id="S2"),
        _row(compiled, "B2", "FORMED", [
            _char(compiled, b2[0], actual=nom[b2[0]] + 0.04, u=0.09),
            _char(compiled, b2[1], actual=nom[b2[1]] + 0.03, u=0.09),
            _char(compiled, b2[2], actual=89.7, u=0.18),
            _char(compiled, b2[3], actual=-0.05, u=0.12),
            _char(compiled, bg2, actual=nom[bg2] + 0.11, u=0.12)], measured_included=89.7, stage_id="S2"),
        _row(compiled, "B3", "NOT_YET_FORMED", nyf(b3), measured_included=179.6, stage_id="S2"),
        _row(compiled, "B4", "NOT_YET_FORMED", nyf(b4), measured_included=179.8, stage_id="S2"),
    ]
    out["shopfloor_result_v1.in_process.example.json"] = _result(
        recipe, compiled, rows=rows_ip, declared=2, stage_model_id="S2",
        gms=[_gm_row(compiled, recipe, state="NOT_APPLICABLE", reason="NOT_APPLICABLE_AT_STAGE")])

    # 3) Final scan with occlusion: B4 UNKNOWN (OCCLUDED) -> stage UNDETERMINED -> INCOMPLETE.
    rows_inc = list(rows[:3])
    rows_inc[2] = _row(compiled, "B3", "FORMED", [
        _char(compiled, b3[0], actual=nom[b3[0]] + 0.10, u=0.10),
        _char(compiled, b3[1], actual=nom[b3[1]] - 0.02, u=0.09),
        _char(compiled, b3[2], actual=89.6, u=0.15),
        _char(compiled, b3[3], actual=0.20, u=0.12)], measured_included=89.6)
    rows_inc[1] = _row(compiled, "B2", "FORMED", [
        _char(compiled, b2[0], actual=nom[b2[0]] + 0.05, u=0.09),
        _char(compiled, b2[1], actual=nom[b2[1]] + 0.10, u=0.09),
        _char(compiled, b2[2], actual=89.5, u=0.2),
        _char(compiled, b2[3], actual=-0.06, u=0.12)], measured_included=89.5)
    rows_inc.append(_row(compiled, "B4", "UNKNOWN", [
        _char(compiled, i, state="NOT_MEASURED", reason="OCCLUDED") for i in b4], reasons=("OCCLUDED",)))
    out["shopfloor_result_v1.incomplete.example.json"] = _result(
        recipe, compiled, rows=rows_inc, gms=[_gm_row(compiled, recipe, state="NOT_MEASURED", reason="OCCLUDED")])

    # 4) Untrusted registration: INVALID_INSPECTION, every characteristic NOT_MEASURED, no heatmap.
    invalid_validity = InspectionValidityV1(
        state="UNTRUSTED_REGISTRATION", reasons=("DATUM_FITNESS_BELOW_POLICY",),
        registration=RegistrationV1(method="DATUM_FPFH_RANSAC_TRIMMED_ICP_V1", datum_fitness=0.31, rmse_mm=1.8,
                                    pose_candidates_considered=8, abstained=True),
        coverage=CoverageV1(surface_coverage_ratio=0.41, scan_point_count=402113))
    rows_inv = [
        _row(compiled, label, "UNKNOWN",
             [_char(compiled, i, state="NOT_MEASURED", reason="INVALID_INSPECTION") for i in _ids(compiled, label)],
             reasons=("REGISTRATION_UNTRUSTED",))
        for label in _LABELS
    ]
    out["shopfloor_result_v1.invalid.example.json"] = _result(
        recipe, compiled, rows=rows_inv, validity=invalid_validity, stage_model_id=None,
        gms=[_gm_row(compiled, recipe, state="NOT_MEASURED", reason="INVALID_INSPECTION")],
        viewer=ViewerAssetsV1(registered_scan=None, heatmap_band_mm=0.5,
                              heatmap_unavailable_reason="REGISTRATION_UNTRUSTED",
                              final_mesh="display/final_mesh.ply", bend_geometry="bend_geometry.json"))
    return out


def _job(recipe: PartRecipeV1, compiled: CompiledPartV1, result: ShopfloorInspectionResultV1) -> LineJobV1:
    return LineJobV1(
        job_id=JOB_ID, kind="inspect", station_id="cell-1", scan_file_id=f"scan_{SCAN_SHA[:24]}",
        scan_sha256=SCAN_SHA, sku=recipe.sku, recipe_id=recipe.recipe_id, recipe_version=recipe.recipe_version,
        recipe_sha256=recipe.recipe_sha256, compiled_part_sha256=compiled.compiled_part_sha256, priority=100,
        dedupe_key=f"{SCAN_SHA}:{recipe.recipe_sha256}", execution_status="completed", stage="done", progress=100,
        attempts=1, max_attempts=2, created_at="2026-10-17T07:15:01Z", started_at="2026-10-17T07:15:02Z",
        finished_at="2026-10-17T07:15:44Z", engine_bundle_id="bundle-example", workspace_rel=f"jobs/{JOB_ID}",
        result_summary=JobResultSummaryV1(
            inspection_validity=result.inspection_validity.state, summary_state=result.summary.state,
            bends_total=result.summary.bends_total, required_fail=result.summary.required_fail,
            required_review=result.summary.required_review, required_unknown=result.summary.required_unknown,
            stage_status=result.stage_decision.status),
        peak_rss_bytes=1_900_000_000, duration_s=42.0)


def _summary_of(job: LineJobV1, recipe: PartRecipeV1, result: ShopfloorInspectionResultV1 | None,
                failure: JobFailureV1 | None = None) -> LineJobSummaryV1:
    return LineJobSummaryV1(
        job_id=job.job_id, kind=job.kind, station_id=job.station_id, sku=job.sku, part_name=recipe.part_name,
        recipe_id=job.recipe_id, recipe_version=job.recipe_version, scan_file_name="SYN-HAT-1_A_SN0042.ply",
        serial="SN0042", execution_status=job.execution_status, stage=job.stage, progress=job.progress,
        created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at,
        failure_code=failure.code if failure else None,
        summary_state=result.summary.state if result else None,
        inspection_validity=result.inspection_validity.state if result else None,
        stage_status=result.stage_decision.status if result else None)


def build_examples() -> dict[str, Any]:
    recipe = build_recipe()
    compiled = build_compiled(recipe)
    results = build_results(recipe, compiled)
    main = results["shopfloor_result_v1.example.json"]
    job = _job(recipe, compiled, main)
    job_summary = _summary_of(job, recipe, main)
    feed = StationFeedV1(
        station_id="cell-1", etag='"' + _sha("feed")[:16] + '"', server_time="2026-10-17T07:15:50Z",
        latest_completed=StationLatestCompletedV1(
            job=job_summary, result=main,
            artifacts={name: f"/api/line/jobs/{job.job_id}/artifacts/{name}"
                       for name in ("viewer_scan.ply", "viewer_deviations.f32", "stage_mesh.ply", "final_mesh.ply",
                                    "bend_geometry.json", "report.pdf")}),
        latest_issue={"kind": "QUARANTINE", "at": "2026-10-17T07:01:12Z", "file_name": "SYN-HAT-1_C_SN0040.ply",
                      "reason_code": "REVISION_MISMATCH", "message": "File revision C differs from recipe revision A"},
        recent=(job_summary,),
    )
    return {
        "part_recipe_v1.example.json": recipe,
        "compiled_part_v1.example.json": compiled,
        **results,
        "station_feed_v1.example.json": feed,
        "line_job_v1.example.json": job,
    }


def render(model: Any) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def write_examples(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, model in build_examples().items():
        path = out / name
        path.write_text(render(model), encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the SYN-HAT-1 contract examples.")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    for path in write_examples(args.out):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
