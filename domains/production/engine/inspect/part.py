"""Inputs of the per-scan engine: integrity checks and a geometric view of the CompiledPart.

:func:`load_inputs` performs the ENGINE_IO §4 integrity checks before any
measurement and raises :class:`InspectInputError` (CLI exit 2,
``engine_error.json``) with one of the contract codes:

- ``COMPILED_PART_MISSING``: ``--compiled`` names no ``compiled_part.json``;
- ``COMPILED_PART_HASH_MISMATCH``: the file does not validate (its recomputed
  ``compiled_part_sha256`` differs), it differs from ``--compiled-sha256``,
  ``check_compiled_against_recipe`` reports a problem, or a stage file is
  missing or altered;
- ``RECIPE_SHA_MISMATCH``: the recipe does not validate (hashes recomputed)
  or differs from ``--recipe-sha256``;
- ``COMPILE_BLOCKED``: the compile is not ``COMPILED``;
- ``SCAN_MISSING`` / ``SCAN_HASH_MISMATCH``: the scan bytes.

:class:`PartView` exposes the flange tree (datum, BFS bend order, parent and
child panels), per-bend hinge geometry in the final pose, the nominals and the
stage models, with lazily loaded sample clouds (``np.load(allow_pickle=False)``).
Everything is in ``cad_model_mm``.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from scipy.spatial import cKDTree

from contracts.production.compiled_part import (
    CharacteristicNominalV1,
    CompiledBendV1,
    CompiledPartV1,
    PanelV1,
    StageModelV1,
    check_compiled_against_recipe,
)
from contracts.production.layout import COMPILE_OUTPUT_FILES
from contracts.production.part_recipe import PartRecipeV1

__all__ = [
    "INPUT_ERROR_CODES",
    "InspectInputError",
    "Inputs",
    "BendView",
    "PanelView",
    "StageCloud",
    "PartView",
    "load_inputs",
    "sha256_file",
]

#: ``engine_error.json`` codes (ENGINE_IO §4, exit 2).
INPUT_ERROR_CODES = frozenset({
    "COMPILED_PART_MISSING",
    "COMPILED_PART_HASH_MISMATCH",
    "COMPILE_BLOCKED",
    "RECIPE_SHA_MISMATCH",
    "SCAN_MISSING",
    "SCAN_HASH_MISMATCH",
})


class InspectInputError(Exception):
    """Invalid engine input (exit 2)."""

    def __init__(self, code: str, message: str) -> None:
        if code not in INPUT_ERROR_CODES:
            raise ValueError(f"unknown engine input error code {code}")
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Inputs:
    compiled: CompiledPartV1
    compiled_dir: Path
    recipe: PartRecipeV1
    scan_path: Path
    scan_sha256: str


def load_inputs(compiled_arg: str | Path, recipe_path: str | Path, scan_path: str | Path, scan_sha256: str, *,
                compiled_sha256: str | None = None, recipe_sha256: str | None = None) -> Inputs:
    """ENGINE_IO §4 integrity checks (module docstring); returns the validated inputs."""
    carg = Path(compiled_arg)
    cpath = carg / COMPILE_OUTPUT_FILES["compiled_part"] if carg.is_dir() else carg
    if not cpath.is_file():
        raise InspectInputError("COMPILED_PART_MISSING", f"no compiled part at {cpath.name}")
    try:
        compiled = CompiledPartV1.model_validate_json(cpath.read_bytes())
    except (ValidationError, ValueError) as exc:
        raise InspectInputError("COMPILED_PART_HASH_MISMATCH", f"compiled part does not validate: {_first(exc)}") from None
    if compiled_sha256 is not None and compiled.compiled_part_sha256 != compiled_sha256.lower():
        raise InspectInputError("COMPILED_PART_HASH_MISMATCH", "compiled_part_sha256 differs from --compiled-sha256")
    rpath = Path(recipe_path)
    if not rpath.is_file():
        raise InspectInputError("RECIPE_SHA_MISMATCH", f"recipe file not found: {rpath.name}")
    try:
        recipe = PartRecipeV1.model_validate_json(rpath.read_bytes())
    except (ValidationError, ValueError) as exc:
        raise InspectInputError("RECIPE_SHA_MISMATCH", f"recipe does not validate: {_first(exc)}") from None
    if recipe_sha256 is not None and recipe.recipe_sha256 != recipe_sha256.lower():
        raise InspectInputError("RECIPE_SHA_MISMATCH", "recipe_sha256 differs from --recipe-sha256")
    problems = check_compiled_against_recipe(compiled, recipe)
    if problems:
        raise InspectInputError("COMPILED_PART_HASH_MISMATCH",
                                "compiled part disagrees with the recipe: " + ", ".join(problems))
    if compiled.status != "COMPILED":
        raise InspectInputError("COMPILE_BLOCKED", "compile is BLOCKED: " + ", ".join(compiled.blocker_codes))
    cdir = cpath.parent
    for stage in compiled.stages:
        for ref in (stage.sample_cloud, stage.display_mesh):
            target = cdir / ref.rel_path
            if not target.is_file() or sha256_file(target) != ref.sha256:
                raise InspectInputError("COMPILED_PART_HASH_MISMATCH", f"stage file {ref.rel_path} missing or altered")
    spath = Path(scan_path)
    if not spath.is_file():
        raise InspectInputError("SCAN_MISSING", "scan file not found")
    if sha256_file(spath) != str(scan_sha256).lower():
        raise InspectInputError("SCAN_HASH_MISMATCH", "scan bytes do not hash to --scan-sha256")
    return Inputs(compiled=compiled, compiled_dir=cdir, recipe=recipe, scan_path=spath, scan_sha256=str(scan_sha256).lower())


def _first(exc: Exception) -> str:
    text = str(exc).strip().splitlines()
    return " | ".join(text[:3])[:400]


# --------------------------------------------------------------------------- geometric view
def _a(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


@dataclass
class PanelView:
    index: int
    panel_id: str
    model: PanelV1
    n0: np.ndarray  # skin-0 outward normal (final pose)
    p0: np.ndarray  # point on skin 0
    area: float
    parent_bend: int | None = None  # bend index (into PartView.bends)
    children: list[int] = field(default_factory=list)  # bend indices
    depth: int = 0


@dataclass
class BendView:
    index: int
    uid: str
    seq: int
    label: str
    model: CompiledBendV1
    parent: int  # panel index
    child: int
    axis_point: np.ndarray
    axis_dir: np.ndarray
    z0: float  # axial extent along axis_dir from axis_point
    z1: float
    ri: float
    t: float
    sweep: float  # rad
    p_out: np.ndarray  # parent outward (away from the bend, into the parent)
    c_out: np.ndarray
    r_parent: np.ndarray  # radial unit from the axis to the parent's tangent lines
    oml_skin: int
    side_panel: dict[str, int]  # "A"/"B" -> panel index
    side_out: dict[str, np.ndarray]  # "A"/"B" -> outward_dir
    rank: int = 0

    @property
    def ro(self) -> float:
        return self.ri + self.t

    @property
    def length(self) -> float:
        return self.z1 - self.z0

    def side_of_panel(self, panel_index: int) -> str:
        for s, p in self.side_panel.items():
            if p == panel_index:
                return s
        raise KeyError(panel_index)

    def locus(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        line = getattr(self.model.loci, name)
        return _a(line.start_mm), _a(line.end_mm)


@dataclass
class StageCloud:
    stage_model_id: str
    points: np.ndarray
    normals: np.ndarray
    segment_kind: np.ndarray
    segment_index: np.ndarray
    skin: np.ndarray
    _tree: cKDTree | None = None

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(self.points, balanced_tree=False, compact_nodes=False)
        return self._tree


class PartView:
    """Flange tree, hinge geometry, nominals and stage models of a COMPILED part."""

    def __init__(self, compiled: CompiledPartV1, compiled_dir: Path) -> None:
        if compiled.status != "COMPILED" or compiled.datum_panel_id is None or compiled.thickness is None:
            raise ValueError("PartView needs a COMPILED part")
        self.compiled = compiled
        self.dir = Path(compiled_dir)
        self.t = float(compiled.thickness.t_mm)
        self.k_factor = float(compiled.k_factor.value) if compiled.k_factor else 0.33
        self.k_sigma = float(compiled.k_factor.uncertainty) if compiled.k_factor else 0.05
        self.panel_index = {p.panel_id: i for i, p in enumerate(compiled.panels)}
        self.panels: list[PanelView] = []
        for i, p in enumerate(compiled.panels):
            self.panels.append(PanelView(i, p.panel_id, p, _a(p.skin0_plane.normal), _a(p.skin0_plane.point_mm),
                                         float(p.area_mm2)))
        self.datum = self.panel_index[compiled.datum_panel_id]
        first_of_move: dict[int, int] = {}
        for b in compiled.bends:
            if b.machine_move_index is not None:
                first_of_move.setdefault(b.machine_move_index, b.sequence_index)
        self.bends: list[BendView] = []
        for j, b in enumerate(compiled.bends):
            ap, ad = _a(b.axis.point_mm), _a(b.axis.direction)
            cs, ce = _a(b.loci.cylinder_axis.start_mm), _a(b.loci.cylinder_axis.end_mm)
            z0, z1 = sorted((float((cs - ap) @ ad), float((ce - ap) @ ad)))
            sides = {s.side: s for s in b.sides}
            parent, child = self.panel_index[b.parent_panel_id], self.panel_index[b.child_panel_id]
            p_out = next(_a(s.outward_dir) for s in b.sides if s.panel_id == b.parent_panel_id)
            c_out = next(_a(s.outward_dir) for s in b.sides if s.panel_id == b.child_panel_id)
            tangent = _a(b.loci.outer_tangent_parent.start_mm)
            r = tangent - cs
            r = r - float(r @ ad) * ad
            r_parent = r / max(float(np.linalg.norm(r)), 1e-12)
            rank = first_of_move[b.machine_move_index] if b.machine_move_index is not None else b.sequence_index
            self.bends.append(BendView(
                index=j, uid=b.bend_uid, seq=b.sequence_index, label=b.label, model=b, parent=parent, child=child,
                axis_point=ap, axis_dir=ad, z0=z0, z1=z1, ri=float(b.inner_radius_mm), t=float(b.thickness_mm),
                sweep=math.radians(b.sweep_deg), p_out=p_out, c_out=c_out, r_parent=r_parent, oml_skin=int(b.oml_skin),
                side_panel={k: self.panel_index[v.panel_id] for k, v in sides.items()},
                side_out={k: _a(v.outward_dir) for k, v in sides.items()}, rank=rank))
        self.bend_index = {b.uid: b.index for b in self.bends}
        for b in self.bends:
            self.panels[b.child].parent_bend = b.index
            self.panels[b.parent].children.append(b.index)
        # breadth-first bend order from the datum (parents before children)
        self.bfs: list[int] = []
        queue = [self.datum]
        depth = {self.datum: 0}
        while queue:
            pk = queue.pop(0)
            for bj in sorted(self.panels[pk].children, key=lambda j: self.bends[j].seq):
                self.bfs.append(bj)
                c = self.bends[bj].child
                depth[c] = depth[pk] + 1
                queue.append(c)
        for pk, d in depth.items():
            self.panels[pk].depth = d
        self.nominals: dict[str, CharacteristicNominalV1] = {c.characteristic_id: c for c in compiled.characteristics}
        self.stages: dict[str, StageModelV1] = {s.stage_model_id: s for s in compiled.stages}
        self.n = len(self.bends)
        self._clouds: dict[str, StageCloud] = {}
        self.rigid_followers: set[int] = set()
        for feat in compiled.unsupported_features:
            if feat.panel_id and (feat.note or "").startswith("RIGID_FOLLOWER_OF:"):
                self.rigid_followers.add(self.panel_index[feat.panel_id])

    # -- stage models ---------------------------------------------------------------
    def prefix_stage(self, k: int) -> StageModelV1:
        return self.stages[f"S{int(k)}"]

    def stage_for_formed(self, formed_uids) -> StageModelV1 | None:
        """The compiled stage model whose formed set equals ``formed_uids`` (any kind), else None."""
        want = set(formed_uids)
        for s in self.compiled.stages:
            if set(s.formed_bend_uids) == want:
                return s
        return None

    def cloud(self, stage_model_id: str) -> StageCloud:
        if stage_model_id not in self._clouds:
            stage = self.stages[stage_model_id]
            with np.load(self.dir / stage.sample_cloud.rel_path, allow_pickle=False) as z:
                self._clouds[stage_model_id] = StageCloud(
                    stage_model_id, np.asarray(z["points"], dtype=np.float64), np.asarray(z["normals"], dtype=np.float64),
                    np.asarray(z["segment_kind"]), np.asarray(z["segment_index"]), np.asarray(z["skin"]))
        return self._clouds[stage_model_id]

    def mesh_path(self, stage_model_id: str) -> Path:
        return self.dir / self.stages[stage_model_id].display_mesh.rel_path

    def stage_transform(self, stage_model_id: str, panel_index: int) -> np.ndarray:
        """Final-pose -> stage-pose transform of a panel (contract ``panel_transforms``)."""
        m = self.stages[stage_model_id].panel_transforms[self.panels[panel_index].panel_id]
        return np.asarray(m, dtype=np.float64)

    def nominal(self, cid: str) -> CharacteristicNominalV1 | None:
        return self.nominals.get(cid)

    def path_bends(self, panel_index: int) -> list[int]:
        """Bend indices on the path from the datum to ``panel_index`` (datum first)."""
        out: list[int] = []
        cur = panel_index
        while self.panels[cur].parent_bend is not None:
            b = self.panels[cur].parent_bend
            out.append(b)
            cur = self.bends[b].parent
        return out[::-1]

    def subtree_panels(self, panel_index: int) -> list[int]:
        out = [panel_index]
        for bj in self.panels[panel_index].children:
            out.extend(self.subtree_panels(self.bends[bj].child))
        return out

    def outline_extent(self) -> float:
        """Bounding-box diagonal of the final-stage sample cloud (mm)."""
        pts = self.cloud(f"S{self.n}").points
        return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

    def blank_extent(self) -> float:
        pts = self.cloud("S0").points
        return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

    # -- formed-set geometry (any formed set, e.g. out of order) -------------------
    def formed_set_transforms(self, formed_uids) -> dict[int, np.ndarray]:
        """Panel index -> final->stage transform for any formed set (E1's ``stage_transforms_from_compiled``)."""
        from domains.production.engine.compile.stages import stage_transforms_from_compiled

        by_id = stage_transforms_from_compiled(self.compiled, set(formed_uids))
        return {self.panel_index[pid]: np.asarray(m, dtype=np.float64) for pid, m in by_id.items()}

    def stage_lines(self, stage_model_id: str | None, formed_uids) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Bend line per bend at a stage: the compiled model's lines, else computed for ``formed_uids``.

        Formed bends: the OML virtual sharp; unformed bends: the centre of the flat
        strip on the OML skin (the contract's ``StageModelV1.bend_lines`` rule).
        """
        if stage_model_id is not None and stage_model_id in self.stages:
            st = self.stages[stage_model_id]
            return {u: (_a(l.start_mm), _a(l.end_mm)) for u, l in st.bend_lines.items()}
        formed = set(formed_uids)
        tr = self.formed_set_transforms(formed)
        out = {}
        for b in self.bends:
            t = tr[b.parent]
            if b.uid in formed:
                a, e = b.locus("oml_virtual_sharp")
            else:
                ta, tb = b.locus("outer_tangent_parent")
                shift = -b.p_out * (float(b.model.neutral_length_mm) / 2.0)
                a, e = ta + shift, tb + shift
            out[b.uid] = (a @ t[:3, :3].T + t[:3, 3], e @ t[:3, :3].T + t[:3, 3])
        return out

    def stage_transforms(self, stage_model_id: str | None, formed_uids) -> dict[int, np.ndarray]:
        if stage_model_id is not None and stage_model_id in self.stages:
            return {i: self.stage_transform(stage_model_id, i) for i in range(len(self.panels))}
        return self.formed_set_transforms(formed_uids)

    def flat_island(self, start_panel: int, blocked_bend: int, unformed: set[int]) -> tuple[list[int], list[int]]:
        """Panels reachable from ``start_panel`` through unformed bends (not crossing ``blocked_bend``).

        Mirrors E1's backgauge region (``nominals._flat_region``); returns (panels, crossed bend indices).
        """
        adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(self.panels))}
        for b in self.bends:
            adj[b.parent].append((b.index, b.child))
            adj[b.child].append((b.index, b.parent))
        seen, stack, crossed = {start_panel}, [start_panel], []
        while stack:
            cur = stack.pop()
            for bj, other in adj[cur]:
                if bj == blocked_bend or bj not in unformed or other in seen:
                    continue
                seen.add(other)
                crossed.append(bj)
                stack.append(other)
        return sorted(seen), sorted(set(crossed))
