"""Onboarding compile orchestration: recipe + folded STEP (+ flat) -> CompiledPartV1 (E1).

``compile_part`` is the library entry point used by the CLI
(``python -m domains.production.engine.compile``) and by tests. It

1. verifies the input bytes against the recipe (CAD and flat sha256),
2. reads the STEP solid (:mod:`.brep`) and builds the flange graph
   (:mod:`.flange_graph`) and the rooted part model (:mod:`.part_model`),
3. reconciles the recipe marks with the graph (:mod:`.reconcile`),
4. parses and binds the flat when the recipe has one (:mod:`.flat`,
   :mod:`.flat_binding`), which fixes K (bound flat, else the shop K with an
   explicit uncertainty) and the stage method (Method B, the flat folded with
   its own per-bend strips, when a bound flat determines every strip; else
   Method A),
5. derives the nominals with resolved tolerances (:mod:`.nominals` and
   ``contracts.production.compiled_part.recipe_policy_for``),
6. builds every prefix stage model S_0..S_N with its sample cloud, display
   mesh and symmetry test (:mod:`.stages`, :mod:`.sampling`, :mod:`.symmetry`),
7. writes ``stages/*.npz|*.ply`` and, last, ``compiled_part.json`` (canonical
   JSON, tmp + rename).

A BLOCKED compile still writes ``compiled_part.json`` (the wizard needs the
issues and ``unmarked_graph_bends``); stage models are only built when no
blocker was found before step 6. Input-integrity failures raise
:class:`CompileInputError` and the CLI writes ``compile_error.json`` instead.

An unexpected exception inside a compile phase (geometry the code does not
handle) is not an exit-3 crash: it becomes a BLOCKER naming the phase and the
exception (``FLANGE_GRAPH_NOT_A_TREE`` up to reconciliation, ``FLAT_PARSE_FAILED``
in the flat phase, ``STAGE_MODEL_FAILED`` afterwards), the traceback goes to
``compile.log`` and ``compiled_part.json`` is still written. Only the soft time
budget (``CompileTimeout``) and a second, assembly-level failure are internal
errors.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from contracts.production.canonical import canonical_json_bytes
from contracts.production.characteristics import make_characteristic_id
from contracts.production.common import Tolerance, ToleranceRef, round6
from contracts.production.compiled_part import (
    COMPILE_WARNING_CODES,
    MAX_BENDS,
    MAX_PANELS,
    CharacteristicNominalV1,
    CompiledPartV1,
    assign_sides,
    recipe_policy_for,
)
from contracts.production.layout import COMPILE_OUTPUT_FILES, stage_cloud_rel_path, stage_mesh_rel_path
from contracts.production.part_recipe import PartRecipeV1
from domains.production.engine.compile.brep import CadReadError, load_step
from domains.production.engine.compile.config import COMPILER_REVISION, COMPILER_VERSION, DEFAULT_SHOP_K, CompileConfig
from domains.production.engine.compile.flange_graph import LABEL_FREE, FlangeGraph, GraphError, build_flange_graph
from domains.production.engine.compile.geometry import mat4_tuple, rotation_matrix, unit, vec3_tuple
from domains.production.engine.compile.nominals import FlangeNominal, backgauge_nominal, flange_length_nominal
from domains.production.engine.compile.part_model import ModelBend, PartModel, build_part_model
from domains.production.engine.compile.reconcile import Mark, Reconciliation, reconcile_marks
from domains.production.engine.compile.sampling import (
    SEG_BEND,
    SEG_PANEL,
    SEG_WALL,
    choose_spacing,
    final_mesh,
    sample_final,
    stage_mesh,
    stage_samples,
    write_npz,
    write_ply,
)
from domains.production.engine.compile.stages import RIGID_FOLLOWER_NOTE_PREFIX, stage_bend_line, stage_panel_transforms
from domains.production.engine.compile.symmetry import symmetry_test
from domains.production.engine.compile.unsupported import (
    MIN_REPRESENTABLE_RI_MM,
    UNSUPPORTED_RULE_VERSION,
    wall_is_thin,
)

__all__ = [
    "CompileInputError",
    "CompileTimeout",
    "CompileResult",
    "COMPILE_ERROR_CODES",
    "compile_part",
    "sha256_file",
]

#: Codes of ``compile_error.json`` (invalid input, exit 2; no compiled_part.json).
COMPILE_ERROR_CODES = frozenset({
    "RECIPE_INVALID",
    "CAD_MISSING",
    "CAD_HASH_MISMATCH",
    "FLAT_MISSING",
    "FLAT_HASH_MISMATCH",
    "FLAT_UNEXPECTED",
    "CONFIG_INVALID",
    "SHOP_K_INVALID",
    "OUT_DIR_NOT_EMPTY",
})


class CompileInputError(Exception):
    """Invalid compile input (exit 2, ``compile_error.json``)."""

    def __init__(self, code: str, message: str) -> None:
        if code not in COMPILE_ERROR_CODES:
            raise ValueError(f"unknown compile error code {code}")
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class CompileTimeout(Exception):
    """The soft wall-clock budget ran out (internal error, exit 3)."""


@dataclass
class CompileResult:
    compiled: CompiledPartV1
    out_dir: Path
    files: dict[str, str] = field(default_factory=dict)  # rel path -> sha256

    @property
    def status(self) -> str:
        return self.compiled.status


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class _Ctx:
    def __init__(self, cfg: CompileConfig, progress: Callable[[str, int], None] | None, log) -> None:
        self.cfg = cfg
        self.t0 = time.monotonic()
        self.progress_cb = progress
        self.log_fh = log
        self.issues: list[dict[str, Any]] = []
        self.diag: dict[str, Any] = {"timings_s": {}, "notes": []}
        self._last = self.t0
        self._phase = "inputs"

    def log(self, text: str) -> None:
        if self.log_fh is not None:
            self.log_fh.write(text.rstrip() + "\n")
            self.log_fh.flush()

    def step(self, stage: str, pct: int, *, check_budget: bool = True) -> None:
        """Enter phase ``stage``; the time since the previous step is booked to the previous phase."""
        now = time.monotonic()
        self.diag["timings_s"][self._phase] = round(now - self._last, 3)
        self._last, self._phase = now, stage
        if check_budget and now - self.t0 > self.cfg.time_budget_s:
            raise CompileTimeout(f"time budget {self.cfg.time_budget_s:.0f}s exceeded at {stage}")
        self.log(f"[{now - self.t0:7.2f}s] {stage}")
        if self.progress_cb is not None:
            self.progress_cb(stage, pct)

    def issue(self, code: str, detail: str = "", bend_uid: str | None = None, graph_bend_id: str | None = None) -> None:
        self.issues.append({
            "code": code,
            "severity": "WARNING" if code in COMPILE_WARNING_CODES else "BLOCKER",
            "bend_uid": bend_uid,
            "graph_bend_id": graph_bend_id,
            "detail": detail[:1000],
        })

    @property
    def blocked(self) -> bool:
        return any(i["severity"] == "BLOCKER" for i in self.issues)

    @property
    def phase(self) -> str:
        return self._phase


def _r(v: float) -> float:
    return round6(float(v))


def _v(p) -> tuple[float, float, float]:
    return tuple(_r(c) for c in vec3_tuple(p))  # type: ignore[return-value]


def _unit_v(p) -> tuple[float, float, float]:
    u = unit(np.asarray(p, dtype=np.float64))
    return _v(u)


def _line(a, b) -> dict[str, Any]:
    return {"start_mm": _v(a), "end_mm": _v(b)}


def _mat(m) -> tuple:
    return tuple(tuple(_r(v) for v in row) for row in mat4_tuple(m))


def _verify_inputs(recipe: PartRecipeV1, cad_path: Path, flat_path: Path | None) -> tuple[str, str | None]:
    if not cad_path.is_file():
        raise CompileInputError("CAD_MISSING", f"CAD file not found: {cad_path.name}")
    cad_sha = sha256_file(cad_path)
    if cad_sha != recipe.cad.sha256:
        raise CompileInputError("CAD_HASH_MISMATCH", "CAD bytes do not hash to recipe.cad.sha256")
    flat_sha = None
    if recipe.flat is not None:
        if flat_path is None:
            raise CompileInputError("FLAT_MISSING", "the recipe has a flat but --flat was not given")
        if not flat_path.is_file():
            raise CompileInputError("FLAT_MISSING", f"flat file not found: {flat_path.name}")
        flat_sha = sha256_file(flat_path)
        if flat_sha != recipe.flat.sha256:
            raise CompileInputError("FLAT_HASH_MISMATCH", "flat bytes do not hash to recipe.flat.sha256")
    elif flat_path is not None:
        raise CompileInputError("FLAT_UNEXPECTED", "a flat was given but the recipe has none")
    return cad_sha, flat_sha


def _tolerance_ref(recipe: PartRecipeV1, probe: CharacteristicNominalV1, tol: Tolerance | None) -> ToleranceRef | None:
    if tol is None:
        return None
    if probe.kind == "GENERAL_MEASUREMENT":
        gm = recipe.general_measurement(probe.measurement_id or "")
        origin = "RECIPE_MEASUREMENT_OVERRIDE" if gm.tolerance is not None else "RECIPE_DEFAULT"
    else:
        key = {
            "INCLUDED_ANGLE": "angle_deg",
            "BEND_LINE_POSITION": "bend_line_position_mm",
            "BACKGAUGE_LENGTH": "backgauge_mm",
        }.get(probe.kind)
        if probe.kind == "FLANGE_LENGTH":
            key = "flange_a_mm" if probe.side == "A" else "flange_b_mm"
        override = getattr(recipe.bend(probe.bend_uid or "").tolerances, key) is not None
        origin = "RECIPE_BEND_OVERRIDE" if override else "RECIPE_DEFAULT"
    return ToleranceRef(plus=tol.plus, minus=tol.minus, origin=origin, standard=recipe.default_tolerances.source)


def _nominal(recipe: PartRecipeV1, *, kind: str, bend_uid: str | None = None, side: str | None = None,
             stage_index: int | None = None, measurement_id: str | None = None, nominal: float | None,
             unavailable_reason: str | None, convention: str, boundary_kind: str | None = None,
             adjacent_bend_uid: str | None = None) -> CharacteristicNominalV1:
    cid = make_characteristic_id(kind, bend_uid=bend_uid, side=side, stage_index=stage_index,
                                 measurement_id=measurement_id)
    unit_name = "deg" if kind == "INCLUDED_ANGLE" else "mm"
    probe = CharacteristicNominalV1.model_construct(
        characteristic_id=cid, kind=kind, bend_uid=bend_uid, side=side, stage_index=stage_index,
        measurement_id=measurement_id, unit=unit_name)
    tol, requirement = recipe_policy_for(recipe, probe)
    return CharacteristicNominalV1(
        characteristic_id=cid, kind=kind, bend_uid=bend_uid, side=side, stage_index=stage_index,
        measurement_id=measurement_id, unit=unit_name,
        nominal=None if nominal is None else _r(nominal),
        unavailable_reason=None if nominal is not None else (unavailable_reason or "NOMINAL_UNAVAILABLE"),
        convention=convention, boundary_kind=boundary_kind, adjacent_bend_uid=adjacent_bend_uid,
        tolerance=_tolerance_ref(recipe, probe, tol), recipe_requirement=requirement)


def _panel_record(model: PartModel, key: int, is_datum: bool) -> dict[str, Any]:
    panel = model.panels[key]
    s0, s1 = panel.side(0), panel.side(1)
    t = model.thickness
    if s0 is not None:
        p0, n0 = s0.point, s0.normal
        loops = s0.loops
        area = s0.area
    else:  # only skin 1 exists: skin 0 is t away on the other side
        p0, n0 = s1.point - s1.normal * t, -s1.normal
        loops = [type(lp)(points=lp.points - s1.normal * t, labels=lp.labels, refs=lp.refs, is_outer=lp.is_outer)
                 for lp in s1.loops]
        area = s1.area
    outer = next((lp for lp in loops if lp.is_outer), loops[0] if loops else None)
    if outer is None or len(outer.points) < 3:
        raise GraphError("FLANGE_GRAPH_NOT_A_TREE", f"panel {panel.panel_id} has no boundary polygon")
    hole_loops, region_loops = _split_holes(outer, [lp for lp in loops if lp is not outer], n0)
    outer_pts, labels = outer.points, outer.labels
    for region in region_loops:  # a panel cut into pieces by a window: keyhole-bridge the pieces
        outer_pts, labels = _bridge(outer_pts, labels, region.points, region.labels)
    free_edges = []
    n = len(outer_pts)
    if np.all(labels == LABEL_FREE):
        runs = [list(range(n)) + [0]]
    else:
        start = int(np.nonzero(labels != LABEL_FREE)[0][0])
        runs, cur = [], []
        for step in range(1, n + 1):
            i = (start + step) % n
            if labels[i] == LABEL_FREE:
                if not cur:
                    cur = [i]
                cur.append((i + 1) % n)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
    for idx, run in enumerate(runs):
        pts = [_v(outer_pts[i]) for i in run]
        dedup = [pts[0]] + [p for a, p in zip(pts, pts[1:]) if p != a]
        if len(dedup) >= 2:
            free_edges.append({"edge_id": f"{panel.panel_id}.E{idx}", "polyline_mm": tuple(dedup),
                               "registration_usable": True})
    holes = [tuple(_v(p) for p in lp.points) for lp in hole_loops if len(lp.points) >= 3]
    dominant = panel.graph.dominant_feature
    record = {
        "panel_id": panel.panel_id,
        "skin0_plane": {"point_mm": _v(p0), "normal": _unit_v(n0)},
        "skin1_plane": None if s1 is None else {"point_mm": _v(s1.point), "normal": _unit_v(s1.normal)},
        "thickness_mm": None if panel.graph.thickness is None else _r(panel.graph.thickness),
        "area_mm2": _r(area),
        "centroid_mm": _v(p0),
        "outer_boundary_mm": tuple(_v(p) for p in outer_pts),
        "free_edges": tuple(free_edges),
        "holes_mm": tuple(holes),
        "is_datum": is_datum,
        "supported": dominant is None,
        "unsupported_feature": dominant,
    }
    return record


_LABEL_BRIDGE = -1


def _split_holes(outer, others, normal: np.ndarray):
    """Loops inside the outer loop are holes; loops outside it are further pieces of the panel."""
    from matplotlib.path import Path as MplPath

    from domains.production.engine.compile.geometry import orthonormal_frame

    x, y = orthonormal_frame(normal)
    poly = np.stack([outer.points @ x, outer.points @ y], axis=1)
    path = MplPath(poly)
    holes, regions = [], []
    for lp in others:
        q = np.stack([lp.points @ x, lp.points @ y], axis=1)
        inside = path.contains_points(q)
        (holes if inside.mean() >= 0.5 else regions).append(lp)
    return holes, regions


def _bridge(pts: np.ndarray, labels: np.ndarray, other: np.ndarray, other_labels: np.ndarray):
    """Keyhole polygon joining two loops through their closest vertices (zero-width slit)."""
    d = np.linalg.norm(pts[:, None, :] - other[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    seq_pts = np.concatenate([pts[: i + 1], np.roll(other, -j, axis=0), other[j:j + 1], pts[i:]])
    seq_lab = np.concatenate([labels[:i], [_LABEL_BRIDGE], np.roll(other_labels, -j), [_LABEL_BRIDGE], labels[i:]])
    return seq_pts, seq_lab.astype(np.int64)


def _sides(bend: ModelBend, model: PartModel) -> list[tuple[str, int, np.ndarray]]:
    p_out, c_out = np.array(_unit_v(bend.p_out)), np.array(_unit_v(bend.c_out))
    ia, _ = assign_sides(tuple(p_out), tuple(c_out))
    pairs = [(bend.parent, p_out), (bend.child, c_out)]
    a, b = pairs[ia], pairs[1 - ia]
    return [("A", a[0], a[1]), ("B", b[0], b[1])]


def _unsupported_far_bend(model: PartModel, nom: FlangeNominal, uid_of: dict[int, str], gid: dict[int, str],
                          affected: dict[tuple, list[str]], characteristic_id: str) -> FlangeNominal:
    """A far boundary on an unsupported bend (marked or rigid) or a feature gives no nominal.

    The characteristic is recorded against the feature record that explains it:
    ``("rigid", key)`` for an unmarked unsupported bend, ``("bend", key)`` for a
    marked one and ``("feature", index)`` for a panel feature.
    """
    adj = nom.adjacent_bend_key
    if adj is not None and (adj not in uid_of or model.bends[adj].unsupported is not None):
        affected[("rigid", adj) if adj not in uid_of else ("bend", adj)].append(characteristic_id)
        b = model.bends[adj]
        return FlangeNominal(value=None, boundary_kind=None, adjacent_bend_key=None,
                             unavailable_reason="UNSUPPORTED_FEATURE",
                             detail=f"far boundary is {'unmarked ' if adj not in uid_of else ''}unsupported graph bend "
                                    f"{gid[adj]} ({b.unsupported})", stations_valid=nom.stations_valid,
                             plurality=nom.plurality)
    if nom.feature_ref is not None:
        affected[("feature", nom.feature_ref)].append(characteristic_id)
    return nom


def _side_boundary(nom: FlangeNominal, uid_of: dict[int, str]) -> dict[str, Any]:
    """``FlangeSideV1`` boundary fields: the nominal's class, else the stations' plurality class.

    BoundaryKind has no UNDETERMINED value, so a side without a nominal names
    the class most of its stations end on (FREE_EDGE, or ADJACENT_BEND with the
    compiled bend). A plurality on an unmarked bend, which has no bend_uid, or
    no class at all reads FREE_EDGE; the characteristic nominal carries the truth.
    """
    if nom.value is not None and nom.boundary_kind is not None:
        kind, adj = nom.boundary_kind, nom.adjacent_bend_key
    elif nom.plurality is not None:
        kind, adj = ("ADJACENT_BEND" if nom.plurality[0] == "ADJ" else "FREE_EDGE"), nom.plurality[1]
    else:
        kind, adj = "FREE_EDGE", None
    uid = uid_of.get(adj) if kind == "ADJACENT_BEND" and adj is not None else None
    if kind == "ADJACENT_BEND" and uid is None:
        kind = "FREE_EDGE"
    return {"boundary_kind": kind, "adjacent_bend_uid": uid}


def compile_part(
    recipe: PartRecipeV1,
    cad_path: str | Path,
    out_dir: str | Path,
    *,
    flat_path: str | Path | None = None,
    shop_k: float = DEFAULT_SHOP_K,
    config: CompileConfig | None = None,
    progress: Callable[[str, int], None] | None = None,
    compiled_at: str | None = None,
) -> CompileResult:
    """Compile ``recipe`` against its CAD (module docstring). Writes into ``out_dir``."""
    cfg = config or CompileConfig()
    if not (isinstance(shop_k, (int, float)) and math.isfinite(shop_k) and 0 < shop_k < 1):
        raise CompileInputError("SHOP_K_INVALID", "--shop-k must lie strictly between 0 and 1")
    cad_path, out = Path(cad_path), Path(out_dir)
    flat_p = Path(flat_path) if flat_path is not None else None
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise CompileInputError("OUT_DIR_NOT_EMPTY", f"output path exists and is not an empty directory: {out}")
    cad_sha, flat_sha = _verify_inputs(recipe, cad_path, flat_p)
    out.mkdir(parents=True, exist_ok=True)
    (out / COMPILE_OUTPUT_FILES["stages_dir"]).mkdir(exist_ok=True)
    with (out / COMPILE_OUTPUT_FILES["log"]).open("w", encoding="utf-8") as log:
        ctx = _Ctx(cfg, progress, log)
        ctx.log(f"sherman_production_compile {COMPILER_VERSION} recipe={recipe.recipe_id} cad={cad_sha[:12]}")
        fields = _base_fields(recipe, recipe.cad.sha256, cfg)
        try:
            _compile(ctx, recipe, cad_path, out, flat_p, flat_sha, float(shop_k), fields)
        except CompileTimeout:
            raise
        except Exception as exc:
            # geometry the compile does not handle must reach the wizard as a reason, not a crash
            ctx.log(traceback.format_exc())
            phase = ctx.phase
            ctx.issue(_INTERNAL_ERROR_CODES.get(phase, "STAGE_MODEL_FAILED"),
                      f"internal error in compile phase '{phase}': {type(exc).__name__}: {exc}")
        ctx.step("assemble", 98, check_budget=False)
        fields["unsupported_features"] = tuple({k: v for k, v in f.items() if k != "_slot"}
                                               for f in fields.get("unsupported_features", ()))
        fields["compiled_at"] = compiled_at or _utc_now()
        fields["diagnostics"] = _jsonable(ctx.diag)
        fields["issues"] = sorted(ctx.issues, key=lambda i: (i["severity"], i["code"], i["bend_uid"] or "",
                                                            i["graph_bend_id"] or "", i["detail"]))
        fields["status"] = "BLOCKED" if ctx.blocked else "COMPILED"
        if fields["status"] == "BLOCKED":
            # a BLOCKED part carries no stage models (and no partial inventories that would not validate)
            fields["stages"] = ()
        try:
            compiled = CompiledPartV1.finalize(**fields)
        except ValidationError as exc:
            # an internal inconsistency must not look like a valid part: record it as a
            # blocker and keep only what validates (a second failure is an internal error)
            first = str(exc).splitlines()
            ctx.log(f"contract validation failed: {exc}")
            ctx.issue("STAGE_MODEL_FAILED", "internal validation: " + " | ".join(first[:3])[:900])
            fields["issues"] = sorted(ctx.issues, key=lambda i: (i["severity"], i["code"], i["bend_uid"] or "",
                                                                i["graph_bend_id"] or "", i["detail"]))
            fields.update(status="BLOCKED", stages=(), bends=(), characteristics=(), unmarked_graph_bends=())
            compiled = CompiledPartV1.finalize(**fields)
        files = {rel: sha for rel, sha in ctx.diag.get("files", {}).items()}
        payload = canonical_json_bytes(compiled)
        tmp = out / (COMPILE_OUTPUT_FILES["compiled_part"] + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, out / COMPILE_OUTPUT_FILES["compiled_part"])
        ctx.log(f"status={compiled.status} blockers={list(compiled.blocker_codes)} sha={compiled.compiled_part_sha256}")
        ctx.step("written", 100, check_budget=False)
    return CompileResult(compiled=compiled, out_dir=out, files=files)


#: Blocker code of an unexpected exception, by the compile phase it happened in (default STAGE_MODEL_FAILED).
_INTERNAL_ERROR_CODES = {
    "inputs": "CAD_READ_FAILED",
    "parse_flat": "FLAT_PARSE_FAILED",
    "read_cad": "CAD_READ_FAILED",
    "flange_graph": "FLANGE_GRAPH_NOT_A_TREE",
    "reconcile": "FLANGE_GRAPH_NOT_A_TREE",
    "flat": "FLAT_PARSE_FAILED",
}
#: Largest sweep CompiledBendV1 can carry (sweep_deg < 180). A marked bend at exactly 180 deg (an
#: open or closed hem) is published, and staged, at this sweep (0.001 deg off: ~0.3 um over a 15 mm
#: return flange), its child outward direction rotated to match; a marked reflex bend (teardrop,
#: > 180 deg) cannot be represented and blocks the compile.
_MAX_PUBLISHED_SWEEP_DEG = 179.999
_REFLEX_SWEEP_DEG = 180.0 + 1e-6
#: K step of the finite-difference d(backgauge nominal)/dK (nominals are piecewise linear in K).
_DK = 0.01


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def _base_fields(recipe: PartRecipeV1, cad_sha: str, cfg: CompileConfig) -> dict[str, Any]:
    return {
        "recipe": {"recipe_id": recipe.recipe_id, "version": recipe.recipe_version,
                   "recipe_sha256": recipe.recipe_sha256, "content_sha256": recipe.content_sha256},
        "cad_sha256": cad_sha,
        "compiler": {"version": COMPILER_VERSION, "config_sha256": cfg.sha256()},
    }


def _compile(ctx: _Ctx, recipe: PartRecipeV1, cad_path: Path, out: Path, flat_path: Path | None,
             flat_sha: str | None, shop_k: float, fields: dict[str, Any]) -> dict[str, Any]:
    """Run the compile phases, filling ``fields`` in place (also on an early return)."""
    from domains.production.engine.compile.flat_binding import FlatOutcome, compile_flat, parse_recipe_flat

    cfg = ctx.cfg
    ctx.diag["unsupported_rule"] = UNSUPPORTED_RULE_VERSION
    # outside compiled_part_sha256: names the code revision without changing any frozen compile hash
    ctx.diag["compiler_revision"] = COMPILER_REVISION
    flat_outcome: FlatOutcome | None = None
    parsed_flat = None
    flat_record: dict[str, Any] | None = None
    if recipe.flat is not None and flat_path is not None:
        ctx.step("parse_flat", 3)
        parsed_flat, flat_record, flat_issues = parse_recipe_flat(recipe, flat_path)
        fields["flat_sha256"] = flat_sha
        fields["flat"] = flat_record
        for code, detail in flat_issues:
            ctx.issue(code, detail)
        if parsed_flat is not None:
            ctx.diag["flat"] = {"parsed": parsed_flat.summary()}
    ctx.step("read_cad", 5)
    try:
        brep = load_step(cad_path, linear_deflection=cfg.mesh_linear_deflection_mm,
                         angular_deflection=cfg.mesh_angular_deflection_rad)
    except CadReadError as exc:
        ctx.issue("CAD_READ_FAILED", f"{exc.reason}: {exc.detail}" if exc.detail else exc.reason)
        return fields
    except Exception as exc:  # OCC exceptions surface as RuntimeError/Standard_Failure
        ctx.issue("CAD_READ_FAILED", f"{type(exc).__name__}: {exc}")
        return fields
    ctx.diag["brep"] = {"faces": len(brep.faces), "edges": len(brep.edges), "unified": brep.unified,
                        "volume_mm3": brep.volume, "notes": brep.notes[:50]}
    ctx.step("flange_graph", 15)
    try:
        graph = build_flange_graph(brep, cfg)
    except GraphError as exc:
        ctx.issue(exc.code, exc.detail)
        return fields
    ctx.diag["graph"] = {
        "panels": len(graph.panels), "bends": len(graph.bends), "walls": len(graph.walls),
        "features": [{"kind": f.kind, "reason": f.reason, "faces": list(f.face_indices), "area_mm2": f.area_mm2}
                     for f in graph.features],
        "thickness_mm": graph.thickness, "thickness_spread_mm": graph.thickness_spread, "notes": graph.notes,
    }
    if len(graph.panels) > MAX_PANELS or len(graph.bends) > MAX_BENDS:
        ctx.issue("FLANGE_GRAPH_NOT_A_TREE", f"{len(graph.panels)} panels / {len(graph.bends)} bends exceed the caps")
        return fields
    if graph.thickness is not None:
        tol = cfg.thickness_tol_mm + cfg.thickness_tol_rel * graph.thickness
        if graph.thickness_spread > tol:
            ctx.issue("THICKNESS_INCONSISTENT",
                      f"skin separation spread {graph.thickness_spread:.4f} mm > {tol:.4f} mm (t={graph.thickness:.4f})")
    try:
        model = build_part_model(graph)
    except GraphError as exc:
        ctx.issue(exc.code, exc.detail)
        return fields
    t_tol = cfg.thickness_tol_mm + cfg.thickness_tol_rel * model.thickness
    if recipe.thickness_mm is not None and abs(recipe.thickness_mm - model.thickness) > t_tol:
        # a wrong-thickness recipe means the wrong material or the wrong CAD: the wizard must see it
        ctx.issue("THICKNESS_INCONSISTENT",
                  f"recipe thickness {recipe.thickness_mm:g} mm differs from the CAD skin separation "
                  f"{model.thickness:.4f} mm by more than {t_tol:.4f} mm (wrong material or wrong CAD)")
    fields["thickness"] = {"t_mm": _r(model.thickness), "source": "MEASURED_BOTH_SKINS",
                           "spread_mm": _r(graph.thickness_spread)}
    fields["datum_panel_id"] = model.panels[model.datum].panel_id
    ordered_keys = sorted(model.panels, key=lambda k: int(model.panels[k].panel_id[1:]))
    try:
        fields["panels"] = tuple(_panel_record(model, k, k == model.datum) for k in ordered_keys)
    except GraphError as exc:
        ctx.issue(exc.code, exc.detail)
        return fields
    ctx.step("reconcile", 25)
    marks = [Mark(b.bend_uid, b.line_start_mm, b.line_end_mm) for b in recipe.bends]
    rec = reconcile_marks(model, marks, cfg)
    ctx.diag["reconciliation"] = {"candidates": rec.candidates, "costs": rec.costs}
    _reconcile_issues(ctx, model, rec)
    gid = {k: b.graph_bend_id for k, b in model.bends.items()}
    fields["unmarked_graph_bends"] = tuple(
        {"graph_bend_id": gid[k], "oml_virtual_sharp": _line(*model.bends[k].oml_sharp()),
         "sweep_deg": _r(min(model.bends[k].sweep_deg, 179.999)),
         "inner_radius_mm": _r(max(model.bends[k].ri, MIN_REPRESENTABLE_RI_MM))}
        for k in sorted(rec.unmarked_supported, key=lambda k: gid[k]))
    all_matched = len(rec.matches) == len(recipe.bends)
    # rigid (unmarked unsupported) graph bends are always formed
    rigid = set(rec.unmarked_unsupported)
    if rigid and all_matched and not rec.blocked:
        try:
            model = build_part_model(graph, rigid)
        except GraphError as exc:
            ctx.issue(exc.code, exc.detail)
            return fields
        fields["datum_panel_id"] = model.panels[model.datum].panel_id
        ordered_keys = sorted(model.panels, key=lambda k: int(model.panels[k].panel_id[1:]))
        fields["panels"] = tuple(_panel_record(model, k, k == model.datum) for k in ordered_keys)
    features = _feature_records(ctx, model, rec, rigid)
    if not all_matched:
        fields["unsupported_features"] = tuple(features)
        return fields
    order = [rec.matches[b.bend_uid] for b in recipe.bends]  # graph keys in sequence order
    uid_of = {gk: b.bend_uid for gk, b in zip(order, recipe.bends)}
    for gk, uid in uid_of.items():
        model.bends[gk].bend_uid = uid
    clamped = []
    for gk in order:
        mb = model.bends[gk]
        if mb.sweep_deg > _REFLEX_SWEEP_DEG:
            ctx.issue("STAGE_MODEL_FAILED",
                      f"marked bend {mb.graph_bend_id} sweeps {mb.sweep_deg:.3f} deg (a reflex teardrop hem): a "
                      "compiled bend's sweep must stay below 180 deg, so it cannot be staged or published; remove "
                      "the mark (an unmarked hem is kept rigid) or correct the CAD", bend_uid=uid_of[gk],
                      graph_bend_id=mb.graph_bend_id)
        elif mb.sweep_deg > _MAX_PUBLISHED_SWEEP_DEG:
            # staged exactly as published; the child's outward direction follows the published sweep,
            # which also gives the side-naming rule two distinct flange directions (a 180 deg hem's
            # flanges point the same way)
            clamped.append(f"{mb.graph_bend_id} {mb.sweep_deg:.6f} -> {_MAX_PUBLISHED_SWEEP_DEG} deg")
            mb.sweep_deg = _MAX_PUBLISHED_SWEEP_DEG
            mb.c_out = unit(rotation_matrix(mb.axis_dir, mb.sweep_rad) @ (-mb.p_out))
    if clamped:
        ctx.diag["notes"].append("180 deg hems published and staged at the largest representable sweep: "
                                 + "; ".join(clamped))
    if ctx.blocked:
        fields["unsupported_features"] = tuple(features)
        return fields

    # -- flat: parse, bind, K ------------------------------------------------------
    ctx.step("flat", 35)
    # Without a bound flat the stages are Method A (rotate back, strips sweep (Ri + K t) at the shop
    # K). A bound flat supplies K (k_factor.source BOUND_FLAT) and, when it determines every strip
    # (flat_binding.fold_flat), its own per-bend strip lengths: the stages then fold the flat
    # (METHOD_B_FOLD_FLAT) and each bend's neutral_length_mm is its strip from the flat.
    k_value, k_source, k_unc, method = shop_k, "SHOP_DEFAULT", cfg.shop_k_uncertainty, "METHOD_A_ROTATE_BACK"
    neutral_override: dict[int, float] | None = None
    if parsed_flat is not None and flat_record is not None:
        flat_outcome = compile_flat(model, recipe, parsed_flat, flat_record, order, uid_of, cfg, shop_k=shop_k)
        fields["flat"] = flat_outcome.record
        ctx.diag["flat"] = flat_outcome.diagnostics
        for code, detail in flat_outcome.issues:
            ctx.issue(code, detail)
        if flat_outcome.bound and flat_outcome.k_factor is not None:
            k_value, k_source, k_unc = flat_outcome.k_factor, "BOUND_FLAT", flat_outcome.k_uncertainty
            neutral_override = flat_outcome.neutral_lengths
            method = flat_outcome.method
    if k_source == "SHOP_DEFAULT":
        ctx.issue("K_FACTOR_DEFAULTED", f"no bound flat; shop K {k_value:g} +- {k_unc:g}")
    if recipe.flat is None or flat_outcome is None or not flat_outcome.bound:
        reason = ("no flat is bound to the recipe bends" if recipe.flat is None
                  else (flat_outcome.unbound_reason if flat_outcome is not None and flat_outcome.unbound_reason
                        else f"flat binding state {recipe.flat.binding.state}"))
        ctx.issue("FLAT_NOT_BOUND", reason)
    fields["k_factor"] = {"value": _r(k_value), "source": k_source, "uncertainty": _r(k_unc)}
    neutral = {k: b.neutral_length(k_value) for k, b in model.bends.items()}
    if neutral_override:
        neutral.update(neutral_override)

    # -- compiled bends + nominals --------------------------------------------------
    ctx.step("nominals", 45)
    bends_out, chars = [], []
    flange_diag: dict[str, Any] = {}
    from collections import defaultdict

    affected: dict[tuple, list[str]] = defaultdict(list)
    for rb, gk in zip(recipe.bends, order):
        mb = model.bends[gk]
        sides = _sides(mb, model)
        side_records = []
        side_noms: dict[str, FlangeNominal] = {}
        for side, panel_key, out_dir_v in sides:
            nom = flange_length_nominal(model, mb, panel_key, cfg.flange_stations)
            nom = _unsupported_far_bend(model, nom, uid_of, gid, affected, f"{rb.bend_uid}/FLANGE_LENGTH/{side}")
            side_noms[side] = nom
            side_records.append({
                "side": side, "panel_id": model.panels[panel_key].panel_id, "outward_dir": tuple(out_dir_v),
                **_side_boundary(nom, uid_of),
            })
            flange_diag[f"{rb.bend_uid}/{side}"] = {"value": nom.value, "kind": nom.boundary_kind,
                                                     "detail": nom.detail, "stations": nom.stations_valid,
                                                     "spread_mm": nom.spread_mm}
        loci = mb.loci()
        sweep = _r(min(mb.sweep_deg, 179.999))
        bends_out.append({
            "bend_uid": rb.bend_uid, "sequence_index": rb.sequence_index, "label": rb.label,
            "machine_move_index": rb.machine_move_index, "graph_bend_id": mb.graph_bend_id,
            "parent_panel_id": model.panels[mb.parent].panel_id, "child_panel_id": model.panels[mb.child].panel_id,
            "inner_radius_mm": _r(max(mb.ri, MIN_REPRESENTABLE_RI_MM)), "thickness_mm": _r(mb.thickness),
            "sweep_deg": sweep, "nominal_included_deg": _r(180.0 - sweep), "direction": mb.direction,
            "oml_skin": mb.oml_skin,
            "axis": {"point_mm": _v(mb.axis_point + mb.z0 * mb.axis_dir), "direction": _unit_v(mb.axis_dir)},
            "loci": {name: _line(*line) for name, line in loci.items()},
            "bend_length_mm": _r(mb.length), "neutral_length_mm": _r(neutral[gk]),
            "sides": tuple(side_records),
            "supported": mb.unsupported is None, "unsupported_feature": mb.unsupported,
        })
        unsupported = mb.unsupported is not None
        for side in ("A", "B"):
            nom = side_noms[side]
            reason = "UNSUPPORTED_FEATURE" if unsupported else nom.unavailable_reason
            value = None if unsupported else nom.value
            conv = ("OML_SHARP_TO_OML_SHARP" if nom.boundary_kind == "ADJACENT_BEND"
                    else "OML_SHARP_TO_FREE_EDGE" if nom.boundary_kind == "FREE_EDGE" else "OML_SHARP_TO_FAR_BOUNDARY")
            adj = uid_of.get(nom.adjacent_bend_key) if nom.boundary_kind == "ADJACENT_BEND" else None
            chars.append(_nominal(recipe, kind="FLANGE_LENGTH", bend_uid=rb.bend_uid, side=side, nominal=value,
                                  unavailable_reason=reason, convention=conv,
                                  boundary_kind=nom.boundary_kind if value is not None else None,
                                  adjacent_bend_uid=adj if value is not None else None))
            if value is None and reason == "NOMINAL_UNAVAILABLE":
                ctx.issue("NOMINAL_UNAVAILABLE", f"flange {side}: {nom.detail}", bend_uid=rb.bend_uid)
        chars.append(_nominal(recipe, kind="INCLUDED_ANGLE", bend_uid=rb.bend_uid,
                              nominal=None if unsupported else 180.0 - sweep,
                              unavailable_reason="UNSUPPORTED_FEATURE", convention="INCLUDED_DEG"))
        chars.append(_nominal(recipe, kind="BEND_LINE_POSITION", bend_uid=rb.bend_uid,
                              nominal=None if unsupported else 0.0, unavailable_reason="UNSUPPORTED_FEATURE",
                              convention="OML_SHARP_OFFSET_ALONG_SIDE_A"))
        recipe_included = rb.nominal_angle.included_deg
        if abs(recipe_included - (180.0 - sweep)) > cfg.angle_disagreement_deg:
            ctx.issue("ANGLE_CONVENTION_DISAGREES_WITH_RECIPE",
                      f"compiled included {180.0 - sweep:.3f} deg vs recipe {recipe_included:.3f} deg "
                      f"({rb.nominal_angle.source})", bend_uid=rb.bend_uid)
    ctx.diag["flange_nominals"] = flange_diag
    n = len(order)
    formed_base = set(rigid)
    stage_formed = [formed_base | set(order[:k]) for k in range(n + 1)]
    stage_transforms = [stage_panel_transforms(model, f, neutral) for f in stage_formed]
    # the same stages with every strip at K + dK: backgauge nominals are linear in K across strips
    # (a strip folded from the flat (Method B) grows by the same sweep t dK)
    if method == "METHOD_B_FOLD_FLAT":
        neutral_dk = {key: neutral[key] + b.sweep_rad * b.thickness * _DK for key, b in model.bends.items()}
    else:
        neutral_dk = {key: b.neutral_length(k_value + _DK) for key, b in model.bends.items()}
    bg_diag = {}
    for k in range(1, n):
        gk = order[k - 1]
        mb = model.bends[gk]
        rb = recipe.bends[k - 1]
        side_panels = tuple(pk for _, pk, _ in _sides(mb, model))
        bg = backgauge_nominal(model, mb, side_panels, stage_formed[k], stage_transforms[k], neutral,
                               cfg.flange_stations)
        cid = f"{rb.bend_uid}/BACKGAUGE_LENGTH/S{k}"
        nom = _unsupported_far_bend(model, bg.nominal, uid_of, gid, affected, cid)
        side = "A" if bg.side_panel == side_panels[0] else "B"
        unsupported = mb.unsupported is not None
        if unsupported:
            affected[("bend", gk)].append(cid)
        value = None if unsupported else nom.value
        reason = "UNSUPPORTED_FEATURE" if unsupported else nom.unavailable_reason
        d_dk = None
        if value is not None:
            bg_dk = backgauge_nominal(model, mb, side_panels, stage_formed[k],
                                      stage_panel_transforms(model, stage_formed[k], neutral_dk), neutral_dk,
                                      cfg.flange_stations)
            if (bg_dk.side_panel == bg.side_panel and bg_dk.nominal.value is not None
                    and bg_dk.nominal.boundary_kind == nom.boundary_kind
                    and bg_dk.nominal.adjacent_bend_key == nom.adjacent_bend_key):
                d_dk = (bg_dk.nominal.value - value) / _DK
        chars.append(_nominal(recipe, kind="BACKGAUGE_LENGTH", bend_uid=rb.bend_uid, side=side, stage_index=k,
                              nominal=value, unavailable_reason=reason,
                              convention="OML_SHARP_TO_FAR_BOUNDARY_AT_STAGE",
                              boundary_kind=nom.boundary_kind if value is not None else None,
                              adjacent_bend_uid=(uid_of.get(nom.adjacent_bend_key)
                                                 if value is not None and nom.boundary_kind == "ADJACENT_BEND" else None)))
        if value is None and reason == "NOMINAL_UNAVAILABLE":
            ctx.issue("NOMINAL_UNAVAILABLE", f"backgauge S{k}: {nom.detail}", bend_uid=rb.bend_uid)
        bg_diag[f"S{k}"] = {"bend_uid": rb.bend_uid, "side": side, "value": nom.value, "kind": nom.boundary_kind,
                            "unformed_in_region": bg.unformed_in_region,
                            "crossed_bend_uids": [uid_of.get(b, gid[b]) for b in bg.crossed],
                            "d_nominal_d_k_mm": d_dk,
                            "k_sigma_mm": None if d_dk is None else abs(d_dk) * k_unc,
                            "detail": nom.detail}
    ctx.diag["backgauge_nominals"] = bg_diag
    for gm in recipe.general_measurements:
        chars.append(_nominal(recipe, kind="GENERAL_MEASUREMENT", measurement_id=gm.measurement_id,
                              nominal=gm.nominal_mm, unavailable_reason=None, convention="CAD_STRAIGHT_LINE"))
    fields["bends"] = tuple(bends_out)
    fields["characteristics"] = tuple(chars)
    for record in features:
        extra = affected.get(record.get("_slot"), [])
        if extra:
            record["affected_characteristic_ids"] = tuple(sorted(set(record["affected_characteristic_ids"]) | set(extra)))
    fields["unsupported_features"] = tuple(features)
    if ctx.blocked:
        return fields

    # -- stage models ------------------------------------------------------------------
    ctx.step("sampling", 55)
    compiled_keys = set(order)
    spacing = choose_spacing(brep, cfg.sample_spacing_mm, cfg.max_points_per_stage)
    samples = sample_final(model, brep, compiled_keys, spacing)
    _wall_features(ctx, model, samples, fields, features)
    # an early STAGE_MODEL_FAILED return below must still list the wall features its warnings name
    fields["unsupported_features"] = tuple(features)
    mesh = final_mesh(model, brep, compiled_keys, samples)
    panel_index = {k: i for i, k in enumerate(ordered_keys)}
    bend_index = {gk: i for i, gk in enumerate(order)}
    seg_index = np.where(samples.kind == SEG_BEND,
                         np.array([bend_index.get(int(b), -1) for b in samples.bend_key], dtype=np.int64),
                         np.array([panel_index.get(int(p), -1) for p in samples.panel_key], dtype=np.int64))
    if np.any(seg_index < 0):
        ctx.issue("STAGE_MODEL_FAILED", "sample labelling left points without a segment")
        return fields
    seg_index = seg_index.astype(np.int32)
    ctx.diag["sampling"] = {"spacing_mm": spacing, "points": int(len(samples.points)),
                            "mesh_vertices": int(len(mesh.vertices)), "mesh_triangles": int(len(mesh.triangles)),
                            "points_by_kind": {str(k): int(np.sum(samples.kind == k)) for k in (SEG_PANEL, SEG_BEND, SEG_WALL)}}
    stages = []
    files: dict[str, str] = {}
    sym_diag = {}
    for k in range(n + 1):
        ctx.step(f"stage_S{k}", 55 + int(40 * (k + 1) / (n + 1)))
        formed = stage_formed[k]
        transforms = stage_transforms[k]
        sid = f"S{k}"
        pts, nrm = stage_samples(model, samples, formed, transforms, neutral)
        if k == 0:
            flat_error = _flat_blank_error(model, samples, pts, rigid)
            ctx.diag["s0_flatness_max_mm"] = flat_error
            if flat_error > _S0_FLATNESS_TOL_MM:
                ctx.issue("STAGE_MODEL_FAILED", f"S0 is not flat: skin points {flat_error:.4f} mm off the datum skin "
                          f"planes (> {_S0_FLATNESS_TOL_MM} mm)")
                return fields
        cloud_rel, mesh_rel = stage_cloud_rel_path(sid), stage_mesh_rel_path(sid)
        write_npz(out / cloud_rel, {
            "points": np.round(pts, 5).astype("<f4"), "normals": nrm.astype("<f4"),
            "segment_kind": samples.kind.astype(np.uint8), "segment_index": seg_index,
            "skin": samples.skin.astype(np.uint8)})
        verts = stage_mesh(model, mesh, formed, transforms, neutral)
        write_ply(out / mesh_rel, np.round(verts, 5), mesh.triangles)
        cloud_sha, mesh_sha = sha256_file(out / cloud_rel), sha256_file(out / mesh_rel)
        files[cloud_rel], files[mesh_rel] = cloud_sha, mesh_sha
        lines = {}
        for gk in order:
            mb = model.bends[gk]
            a, b = stage_bend_line(mb, gk in formed, transforms[mb.parent], neutral[gk])
            lines[mb.bend_uid] = (a, b)
        formed_uids = [model.bends[gk].bend_uid for gk in order[:k]]
        sym = symmetry_test(pts, nrm, {u: lines[u] for u in formed_uids}, target_spacing=cfg.symmetry_target_spacing_mm,
                            source_points=cfg.symmetry_source_points, iterations=cfg.symmetry_icp_iterations)
        sym_diag[sid] = [(c["kind"], round(c["rms_mm"], 4)) for c in sym["candidates"]]
        stages.append({
            "stage_model_id": sid, "stage_index": k, "kind": "SEQUENCE_PREFIX",
            "formed_bend_uids": tuple(formed_uids), "method": method, "k_factor_used": _r(k_value),
            "panel_transforms": {model.panels[pk].panel_id: _mat(t) for pk, t in sorted(transforms.items())},
            "bend_lines": {u: _line(a, b) for u, (a, b) in lines.items()},
            "sample_cloud": {"rel_path": cloud_rel, "sha256": cloud_sha, "point_count": int(len(pts)),
                             "spacing_mm": _r(spacing)},
            "display_mesh": {"rel_path": mesh_rel, "sha256": mesh_sha, "vertex_count": int(len(verts)),
                             "face_count": int(len(mesh.triangles))},
            "symmetry": {
                "method": sym["method"], "threshold_rms_mm": sym["threshold_rms_mm"],
                "candidates": tuple({"kind": c["kind"], "transform": _mat(c["transform"]),
                                     "rms_mm": _r(c["rms_mm"]), "bend_permutation": c["bend_permutation"]}
                                    for c in sym["candidates"]),
                "is_symmetric": bool(any(_r(c["rms_mm"]) <= sym["threshold_rms_mm"] + 1e-12 for c in sym["candidates"])),
                "ambiguous_bend_groups": tuple(sym["ambiguous_bend_groups"]),
            },
        })
    ctx.diag["symmetry"] = sym_diag
    ctx.diag["files"] = files
    fields["stages"] = tuple(stages)
    fields["unsupported_features"] = tuple(features)
    return fields


#: S_0 self-check: every skin point reachable through recipe bends lies on the datum's skin planes.
_S0_FLATNESS_TOL_MM = 0.05


def _flat_blank_error(model: PartModel, samples, pts: np.ndarray, rigid: set[int]) -> float:
    """Largest distance of an S_0 skin point (panels and strips of recipe bends) from its flat skin plane.

    Panels carried through a rigid (unmarked unsupported, always formed) bend are
    not part of the flat blank and are skipped.
    """
    datum = model.panels[model.datum]
    side = datum.side(0)
    if side is not None:
        p0, n0 = side.point, side.normal
    else:
        s1 = datum.side(1)
        p0, n0 = s1.point - s1.normal * model.thickness, -s1.normal
    flat_panels = {k for k in model.panels if not (set(model.path_bends(k)) & rigid)}
    on_panel = (samples.kind == SEG_PANEL) & np.isin(samples.panel_key, list(flat_panels))
    on_strip = (samples.kind == SEG_BEND) & (samples.skin != 2)
    if on_strip.any():
        strip_ok = np.array([model.bends[int(b)].parent in flat_panels and model.bends[int(b)].child in flat_panels
                             for b in samples.bend_key[on_strip]])
        idx = np.nonzero(on_strip)[0][strip_ok]
        on_strip = np.zeros_like(on_strip)
        on_strip[idx] = True
    sel = on_panel | on_strip
    if not sel.any():
        return 0.0
    height = (pts[sel] - p0) @ n0
    expected = np.where(samples.skin[sel] == 0, 0.0, -model.thickness)
    return float(np.max(np.abs(height - expected)))


def _reconcile_issues(ctx: _Ctx, model: PartModel, rec: Reconciliation) -> None:
    gid = {k: b.graph_bend_id for k, b in model.bends.items()}
    for uid in rec.unmatched_marks:
        cands = rec.candidates.get(uid) or []
        ctx.issue("MARK_NOT_IN_FLANGE_GRAPH",
                  "no flange-graph bend contains this mark" + (f" (feasible: {cands})" if cands else ""), bend_uid=uid)
    for uid, gids in rec.ambiguous_marks:
        ctx.issue("MARK_AMBIGUOUS", f"mark matches several graph bends {gids}", bend_uid=uid)
    for gk, uids in rec.ambiguous_graph:
        ctx.issue("MARK_AMBIGUOUS", f"graph bend matched by several marks {uids}", graph_bend_id=gid[gk])
    for gk in rec.unmarked_supported:
        b = model.bends[gk]
        ctx.issue("UNMARKED_BEND", f"sweep {b.sweep_deg:.2f} deg, Ri {b.ri:.3f} mm, length {b.length:.1f} mm",
                  graph_bend_id=gid[gk])


def _feature_records(ctx: _Ctx, model: PartModel, rec: Reconciliation, rigid: set[int]) -> list[dict[str, Any]]:
    """UnsupportedFeatureV1 records + one UNSUPPORTED_FEATURE warning per feature.

    Records ``0..len(graph.features)-1`` are the graph features in order; then one
    per rigid (unmarked unsupported) bend, then one per marked unsupported bend.
    Each record carries a private ``_slot`` key used to attach the characteristics
    it makes unavailable; it is removed before the part is assembled.
    """
    out = []
    graph: FlangeGraph = model.graph
    for fi, feat in enumerate(graph.features):
        panel_id = model.panels[feat.panel_key].panel_id if feat.panel_key in model.panels else None
        record = {"feature_kind": feat.kind, "panel_id": panel_id, "bend_uid": None,
                  "affected_characteristic_ids": (), "note": f"{feat.reason}; faces {list(feat.face_indices)[:20]}"[:500],
                  "_slot": ("feature", fi)}
        out.append(record)
        ctx.issue("UNSUPPORTED_FEATURE", f"{feat.kind} on {panel_id or 'part'}: {feat.reason}")
    for gk in sorted(rigid, key=lambda k: model.bends[k].graph_bend_id):
        b = model.bends[gk]
        child = model.panels[b.child].panel_id
        parent = model.panels[b.parent].panel_id
        out.append({"feature_kind": b.unsupported, "panel_id": child, "bend_uid": None, "_slot": ("rigid", gk),
                    "affected_characteristic_ids": (),
                    "note": f"{RIGID_FOLLOWER_NOTE_PREFIX}{parent} unmarked graph bend {b.graph_bend_id} "
                            f"({b.unsupported_reason}); rigid in every stage"[:500]})
        ctx.issue("UNSUPPORTED_FEATURE", f"unmarked {b.unsupported} bend {b.graph_bend_id} between {parent} and "
                  f"{child}: {b.unsupported_reason}; kept rigid", graph_bend_id=b.graph_bend_id)
    for uid, gk in sorted(rec.matches.items()):
        b = model.bends[gk]
        if b.unsupported is None:
            continue
        affected = tuple(sorted([f"{uid}/FLANGE_LENGTH/A", f"{uid}/FLANGE_LENGTH/B", f"{uid}/INCLUDED_ANGLE",
                                 f"{uid}/BEND_LINE_POSITION"]))
        note = b.unsupported_reason + ("" if b.has_virtual_sharp else
                                       "; loci: the inner/outer apex lines stand in for the IML/OML virtual sharps")
        out.append({"feature_kind": b.unsupported, "panel_id": None, "bend_uid": uid, "_slot": ("bend", gk),
                    "affected_characteristic_ids": affected, "note": note[:500]})
        ctx.issue("UNSUPPORTED_FEATURE", f"{b.unsupported} bend: {b.unsupported_reason}", bend_uid=uid,
                  graph_bend_id=b.graph_bend_id)
    return out


def _wall_features(ctx: _Ctx, model: PartModel, samples, fields: dict[str, Any], features: list) -> None:
    """Wall faces that are not thin (far from every skin) become OTHER features."""
    from domains.production.engine.compile.unsupported import THIN_WALL_ALLOWANCE_MM, FeatureRecord

    for face_index, frac in sorted(samples.wall_far_fraction.items()):
        if wall_is_thin(frac):
            continue
        sel = samples.face == face_index
        owners = samples.panel_key[sel]
        owners = owners[owners >= 0]
        panel_key = int(np.bincount(owners).argmax()) if len(owners) else model.datum
        rec = FeatureRecord(kind="OTHER", reason=f"wall face {face_index} is not thin ({frac:.0%} of its points are "
                                                   f"farther than t + {THIN_WALL_ALLOWANCE_MM:g} mm from every skin)",
                            face_indices=(face_index,), panel_key=panel_key)
        model.graph.features.append(rec)
        features.append({"feature_kind": "OTHER", "panel_id": model.panels[panel_key].panel_id, "bend_uid": None,
                         "affected_characteristic_ids": (), "note": rec.reason[:500],
                         "_slot": ("feature", len(model.graph.features) - 1)})
        ctx.issue("UNSUPPORTED_FEATURE", f"OTHER on {model.panels[panel_key].panel_id}: {rec.reason}")

