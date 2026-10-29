"""Stage models S_k: per-panel rigid transforms and stage bend lines (pure numpy).

Method A (``METHOD_A_ROTATE_BACK``, always available): for every bend that is
not formed at the stage, the child subtree is rotated back by the sweep about
the bend's cylinder axis (the child's tangent lines land on the parent's), then
moved away from the parent by the bend's neutral strip length
``L_n = sweep_rad * (Ri + K * t)`` along the parent's flat continuation
(``-p_out``)::

    U_b = Shift(-p_out * L_n) @ Rot(axis_b, -sweep_b)      (final-pose coordinates)
    T(child) = T(parent) @ (U_b if b unformed else I),     T(datum) = I

Method B (``METHOD_B_FOLD_FLAT``, a bound flat) uses the same composition with
each strip length measured per bend from the bound flat
(:func:`.flat_binding.fold_flat`), i.e. the flat is folded at the formed bends;
the published ``neutral_length_mm`` is then that strip.

Stage bend lines (contract ``StageModelV1.bend_lines``): the OML virtual sharp
for formed bends; for unformed bends the centre line of the flat strip on the
bend's OML skin, ``outer_tangent_parent + (L_n / 2) * (-p_out)``. Both move with
the parent panel's transform.

Unrolling (:func:`unroll_bend_points`) maps material points of an unformed bend
region (cylinder skins and end caps) onto its flat strip: a point at radius
``rho``, angle ``phi`` from the parent end and axial position ``z`` goes to
``axis + z * a + rho * r_parent + (phi / sweep) * L_n * (-p_out)``. The map is
continuous with the parent (phi = 0) and with the rotated-back child
(phi = sweep), so stage clouds and meshes stay watertight across strips. The
angle is measured in [0, sweep] also for sweeps at or past 180 deg (hems).

:func:`stage_transforms_from_compiled` recomputes the panel transforms of *any*
formed set from a ``CompiledPartV1`` alone (for E3 out-of-order hypotheses).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np

from domains.production.engine.compile.geometry import (
    apply4,
    compose,
    rotation_about_line,
    translation,
    wrap_bend_angle,
)
from domains.production.engine.compile.part_model import ModelBend, PartModel

__all__ = [
    "bend_unfold_matrix",
    "stage_panel_transforms",
    "stage_bend_line",
    "unroll_bend_points",
    "stage_transforms_from_compiled",
    "stage_points_from_compiled",
    "RIGID_FOLLOWER_NOTE_PREFIX",
]

#: ``UnsupportedFeatureV1.note`` prefix recording a panel that is rigidly carried by
#: another panel through an unmarked unsupported bend (always formed).
RIGID_FOLLOWER_NOTE_PREFIX = "RIGID_FOLLOWER_OF:"


def bend_unfold_matrix(axis_point: np.ndarray, axis_dir: np.ndarray, sweep_rad: float, p_out: np.ndarray,
                       neutral_length: float) -> np.ndarray:
    """``U_b``: final pose of the child subtree -> its pose with this bend flat."""
    return compose(translation(-np.asarray(p_out, dtype=np.float64) * float(neutral_length)),
                   rotation_about_line(axis_point, axis_dir, -float(sweep_rad)))


def stage_panel_transforms(model: PartModel, formed: set[int], neutral: Mapping[int, float]) -> dict[int, np.ndarray]:
    """Panel key -> 4x4 final-pose -> stage-pose transform. ``formed`` holds graph bend keys."""
    out: dict[int, np.ndarray] = {model.datum: np.eye(4)}
    for key in model.order[1:]:
        panel = model.panels[key]
        bend = model.bends[panel.parent_bend]
        parent_t = out[panel.parent_panel]
        if bend.key in formed:
            out[key] = parent_t.copy()
        else:
            u = bend_unfold_matrix(bend.axis_point, bend.axis_dir, bend.sweep_rad, bend.p_out, neutral[bend.key])
            out[key] = parent_t @ u
    return out


def stage_bend_line(bend: ModelBend, formed: bool, parent_t: np.ndarray, neutral_length: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """The stage bend line of ``bend`` (module docstring) at the stage pose."""
    if formed:
        a, b = bend.oml_sharp()
    else:
        loci = bend.loci()
        ta, tb = loci["outer_tangent_parent"]
        shift = -bend.p_out * (neutral_length / 2.0)
        a, b = ta + shift, tb + shift
    return apply4(parent_t, a), apply4(parent_t, b)


def _rodrigues_batch(axis: np.ndarray, angles: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate each vector by its own angle about one unit axis (vectorised Rodrigues)."""
    k = axis / np.linalg.norm(axis)
    c = np.cos(angles)[:, None]
    s = np.sin(angles)[:, None]
    kv = np.cross(np.broadcast_to(k, vectors.shape), vectors)
    kd = (vectors @ k)[:, None] * k
    return vectors * c + kv * s + kd * (1.0 - c)


def unroll_bend_points(bend: ModelBend, points: np.ndarray, normals: np.ndarray, neutral_length: float
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Map final-pose points/normals of an unformed bend region onto its flat strip (parent frame).

    Normals rotate back by ``-phi`` about the axis, so radial skin normals become
    ``+-r_parent`` and end-cap (axial) normals are unchanged.
    """
    a = bend.axis_dir
    rel = points - bend.axis_point
    z = rel @ a
    radial = rel - np.outer(z, a)
    rho = np.linalg.norm(radial, axis=1)
    safe = np.where(rho > 1e-9, rho, 1.0)
    rdir = radial / safe[:, None]
    if np.any(rho <= 1e-9):
        rdir[rho <= 1e-9] = bend.r_parent
    g = getattr(bend, "graph", None)
    if g is not None and getattr(g, "outer_radius", None) is not None and bend.ro > bend.ri:
        # offset-skin bend: the outer arc is not coaxial, so the material between the inner arc and the outer arc
        # along each radial is mapped onto [Ri, Ri + t] (the inner skin stays at Ri, the outer skin lands on the
        # flat OML skin)
        rho_out = _ray_circle_distance(bend.axis_point, rdir, np.asarray(g.outer_axis_point, dtype=np.float64), a,
                                       float(g.outer_radius))
        span = np.maximum(rho_out - bend.ri, 1e-9)
        rho = np.where(rho > bend.ri, bend.ri + (rho - bend.ri) * (bend.ro - bend.ri) / span, rho)
    cos_phi = np.clip(rdir @ bend.r_parent, -1.0, 1.0)
    sin_phi = np.cross(np.broadcast_to(bend.r_parent, rdir.shape), rdir) @ a
    phi = wrap_bend_angle(np.arctan2(sin_phi, cos_phi), bend.sweep_rad)  # identity below 180 deg
    frac = phi / bend.sweep_rad
    strip = (bend.axis_point + np.outer(z, a) + np.outer(rho, bend.r_parent)
             - np.outer(frac * neutral_length, bend.p_out))
    return strip, _rodrigues_batch(a, -phi, normals)


def _ray_circle_distance(origin: np.ndarray, dirs: np.ndarray, centre: np.ndarray, axis: np.ndarray,
                         radius: float) -> np.ndarray:
    """Distance along each in-plane unit ray from ``origin`` (an axis point) to a circle of ``radius`` about the
    parallel axis through ``centre`` (the far intersection; the circle encloses the origin's axis)."""
    c = centre - origin
    c = c - float(c @ axis) * axis
    b = dirs @ c
    disc = np.maximum(b * b - (float(c @ c) - radius * radius), 0.0)
    return b + np.sqrt(disc)


def stage_transforms_from_compiled(compiled, formed_bend_uids: Iterable[str]) -> dict[str, np.ndarray]:
    """Panel id -> 4x4 stage transform for any formed set, from a ``CompiledPartV1`` alone.

    Uses each compiled bend's oriented axis, sweep, neutral strip length and the
    parent's outward direction; panels carried through an unmarked unsupported
    bend follow their carrier (``RIGID_FOLLOWER_OF:`` feature notes).
    """
    formed = set(formed_bend_uids)
    by_child = {b.child_panel_id: b for b in compiled.bends}
    followers: dict[str, str] = {}
    for feat in compiled.unsupported_features:
        note = feat.note or ""
        if feat.panel_id and note.startswith(RIGID_FOLLOWER_NOTE_PREFIX):
            followers[feat.panel_id] = note[len(RIGID_FOLLOWER_NOTE_PREFIX):].split()[0]
    cache: dict[str, np.ndarray] = {compiled.datum_panel_id: np.eye(4)}

    def solve(pid: str, depth: int = 0) -> np.ndarray:
        if pid in cache:
            return cache[pid]
        if depth > 2000:
            raise ValueError("panel parent chain does not reach the datum")
        if pid in by_child:
            bend = by_child[pid]
            parent_t = solve(bend.parent_panel_id, depth + 1)
            if bend.bend_uid in formed:
                t = parent_t.copy()
            else:
                p_out = next(s.outward_dir for s in bend.sides if s.panel_id == bend.parent_panel_id)
                u = bend_unfold_matrix(np.asarray(bend.axis.point_mm), np.asarray(bend.axis.direction),
                                       math.radians(bend.sweep_deg), np.asarray(p_out), bend.neutral_length_mm)
                t = parent_t @ u
        elif pid in followers:
            t = solve(followers[pid], depth + 1).copy()
        else:
            raise ValueError(f"panel {pid} has no parent bend and is not a rigid follower")
        cache[pid] = t
        return t

    return {p.panel_id: solve(p.panel_id) for p in compiled.panels}


class _CompiledBendView:
    """The fields :func:`unroll_bend_points` needs, read from a ``CompiledBendV1``."""

    def __init__(self, bend) -> None:
        self.axis_dir = np.asarray(bend.axis.direction, dtype=np.float64)
        self.axis_point = np.asarray(bend.axis.point_mm, dtype=np.float64)
        tangent = np.asarray(bend.loci.outer_tangent_parent.start_mm, dtype=np.float64)
        axis_start = np.asarray(bend.loci.cylinder_axis.start_mm, dtype=np.float64)
        r = tangent - axis_start
        r = r - float(r @ self.axis_dir) * self.axis_dir
        self.r_parent = r / np.linalg.norm(r)
        self.p_out = np.asarray(next(s.outward_dir for s in bend.sides if s.panel_id == bend.parent_panel_id),
                                dtype=np.float64)
        self.sweep_rad = math.radians(bend.sweep_deg)
        self.neutral_length = float(bend.neutral_length_mm)
        self.parent = bend.parent_panel_id


def stage_points_from_compiled(compiled, points: np.ndarray, normals: np.ndarray, segment_kind: np.ndarray,
                               segment_index: np.ndarray, formed_bend_uids: Iterable[str]
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Map the final-stage (S_N) sample cloud to any formed set, from a ``CompiledPartV1`` alone.

    Panel and wall points (``segment_kind`` 0/2) move with their panel's
    transform; bend points (kind 1, skins and end caps) move with the parent
    panel when the bend is formed, else are unrolled onto the flat strip first.
    Applied to the S_N cloud with a prefix formed set it reproduces that prefix
    stage's cloud; it also builds out-of-order hypotheses (E3).
    """
    formed = set(formed_bend_uids)
    transforms = stage_transforms_from_compiled(compiled, formed)
    pts = np.asarray(points, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    out_p = np.empty_like(pts)
    out_n = np.empty_like(nrm)
    panel_ids = [p.panel_id for p in compiled.panels]
    for i, pid in enumerate(panel_ids):
        sel = (segment_kind != 1) & (segment_index == i)
        if np.any(sel):
            t = transforms[pid]
            out_p[sel] = apply4(t, pts[sel])
            out_n[sel] = nrm[sel] @ t[:3, :3].T
    for j, bend in enumerate(compiled.bends):
        sel = (segment_kind == 1) & (segment_index == j)
        if not np.any(sel):
            continue
        view = _CompiledBendView(bend)
        p, n = pts[sel], nrm[sel]
        if bend.bend_uid not in formed:
            p, n = unroll_bend_points(view, p, n, view.neutral_length)
        t = transforms[view.parent]
        out_p[sel] = apply4(t, p)
        out_n[sel] = n @ t[:3, :3].T
    norms = np.linalg.norm(out_n, axis=1)
    out_n = out_n / np.where(norms > 1e-12, norms, 1.0)[:, None]
    return out_p, out_n
