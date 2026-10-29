"""Face-labelled sample clouds and display meshes of the stage models (E1, ENGINE_IO §2).

**Final-pose samples.** Every face is sampled on a regular grid of its own
surface parameters with metric spacing ``h`` (0.5 mm by default): planes in
``(u, v)`` (already metric), cylinders in ``(h / r, h)``, other surfaces with
the metric estimated from the first derivatives at the face centre. A grid
point is kept when it lies inside the face's trimmed parameter domain (even-odd
over all boundary loops). Points and outward normals of planes and cylinders
are evaluated analytically; other surfaces through OCC. Labels:

- panel skins: ``segment_kind`` 0, ``segment_index`` = panel, ``skin`` 0/1;
- bend skins of compiled bends: kind 1, index = bend, skin = the cylinder's
  skin (outer cylinder = the bend's ``oml_skin``);
- walls (thickness / cut faces): each point inherits the segment of its nearest
  skin point: kind 2 / skin 2 for a panel, or kind 1 / skin 2 for a bend
  region's end cap;
- unsupported-feature faces and skins of unmarked unsupported bends: kind 2 /
  skin 2 of their owning panel (never part of a skin plane fit).

**Stage clouds.** Panel and wall points move with their panel's stage
transform; bend points of formed bends move with the parent panel; bend points
of unformed bends are unrolled onto the flat strip (:func:`.stages.unroll_bend_points`)
and then moved with the parent. The ``.npz`` is written deterministically
(fixed zip timestamps, deflate level 1, ``allow_pickle=False`` compatible).

**Display meshes.** The OCC tessellation, with the same per-vertex stage map,
written as binary little-endian PLY (float32 ``x y z``, ``list uchar int``).
"""
from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from domains.production.engine.compile.brep import BrepFace, BrepModel
from domains.production.engine.compile.geometry import apply4, apply_dir4
from domains.production.engine.compile.part_model import PartModel
from domains.production.engine.compile.stages import unroll_bend_points
from domains.production.engine.compile.unsupported import THIN_WALL_ALLOWANCE_MM

__all__ = [
    "FinalSamples",
    "FinalMesh",
    "choose_spacing",
    "sample_final",
    "final_mesh",
    "stage_samples",
    "stage_mesh",
    "write_npz",
    "write_ply",
    "SEG_PANEL",
    "SEG_BEND",
    "SEG_WALL",
]

SEG_PANEL, SEG_BEND, SEG_WALL = 0, 1, 2
SKIN_WALL = 2
_GENERIC_FACE_POINT_CAP = 60_000
_NPZ_DATE = (1980, 1, 1, 0, 0, 0)


@dataclass
class FinalSamples:
    points: np.ndarray
    normals: np.ndarray
    kind: np.ndarray  # uint8
    skin: np.ndarray  # uint8
    panel_key: np.ndarray  # int64, owner panel (kind 0/2) else -1
    bend_key: np.ndarray  # int64, graph bend (kind 1) else -1
    face: np.ndarray  # int64 source face (diagnostics)
    spacing: float
    #: wall face -> fraction of its samples farther than t + THIN_WALL_ALLOWANCE_MM from every skin
    wall_far_fraction: dict[int, float] = field(default_factory=dict)


@dataclass
class FinalMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    panel_key: np.ndarray  # per vertex owner panel (-1 when bend)
    bend_key: np.ndarray  # per vertex bend (-1 when panel)


def choose_spacing(brep: BrepModel, target: float, max_points: int) -> float:
    """``target`` unless the total surface area would exceed ``max_points`` samples."""
    area = sum(f.area for f in brep.faces)
    if area <= 0:
        return target
    needed = math.sqrt(area / max_points)
    return float(max(target, round(needed + 0.005, 2)))


def _inside(uv: np.ndarray, loops: list[np.ndarray]) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    inside = np.zeros(len(uv), dtype=bool)
    for poly in loops:
        if len(poly) < 3:
            continue
        inside ^= MplPath(poly).contains_points(uv)
    return inside


def _grid(lo: float, hi: float, step: float) -> np.ndarray:
    k0, k1 = math.floor(lo / step), math.ceil(hi / step)
    vals = (np.arange(k0, k1 + 1) + 0.5) * step
    return vals[(vals >= lo) & (vals <= hi)]


def _sample_face(face: BrepFace, h: float) -> tuple[np.ndarray, np.ndarray]:
    u0, u1, v0, v1 = face.uv_bounds
    loops = [lp.uv_polygon() for lp in face.loops]
    if face.kind == "plane" and face.x_dir is not None and face.y_dir is not None:
        # planes (analytic or recognised free-form) are sampled in their own plane frame,
        # trimmed by the 3D boundary loops projected into it (exact, any parametrisation)
        x_dir, y_dir = face.x_dir, face.y_dir
        loops2 = []
        for lp in face.loops:
            pts3 = [u.xyz[:-1] for u in lp.uses if len(u.xyz) >= 2]
            if pts3:
                q = np.concatenate(pts3) - face.origin
                loops2.append(np.stack([q @ x_dir, q @ y_dir], axis=1))
        if not loops2:
            return np.zeros((0, 3)), np.zeros((0, 3))
        allp = np.concatenate(loops2)
        us, vs = _grid(float(allp[:, 0].min()), float(allp[:, 0].max()), h), _grid(
            float(allp[:, 1].min()), float(allp[:, 1].max()), h)
        if not len(us) or not len(vs):
            return np.zeros((0, 3)), np.zeros((0, 3))
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
        uv = uv[_inside(uv, loops2)]
        pts = face.origin + np.outer(uv[:, 0], x_dir) + np.outer(uv[:, 1], y_dir)
        return pts, np.broadcast_to(face.normal, pts.shape).copy()
    if face.analytic_uv and face.kind == "cylinder":
        r = float(face.radius)
        us, vs = _grid(u0, u1, h / r), _grid(v0, v1, h)
        if not len(us) or not len(vs):
            return np.zeros((0, 3)), np.zeros((0, 3))
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
        uv = uv[_inside(uv, loops)]
        radial = np.outer(np.cos(uv[:, 0]), face.x_dir) + np.outer(np.sin(uv[:, 0]), face.y_dir)
        pts = face.origin + r * radial + np.outer(uv[:, 1], face.axis)
        normals = radial if face.convex else -radial
        return pts, normals
    # generic surface: metric from the centre derivatives
    um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    eps_u, eps_v = max(1e-6, 1e-4 * abs(u1 - u0)), max(1e-6, 1e-4 * abs(v1 - v0))
    p0 = face.point_at(um, vm)
    su = float(np.linalg.norm(face.point_at(um + eps_u, vm) - p0)) / eps_u
    sv = float(np.linalg.norm(face.point_at(um, vm + eps_v) - p0)) / eps_v
    if not (su > 1e-9 and sv > 1e-9):
        return _sample_mesh(face, h)
    du, dv = h / su, h / sv
    n_est = ((u1 - u0) / du) * ((v1 - v0) / dv)
    if n_est > _GENERIC_FACE_POINT_CAP:
        scale = math.sqrt(n_est / _GENERIC_FACE_POINT_CAP)
        du, dv = du * scale, dv * scale
    us, vs = _grid(u0, u1, du), _grid(v0, v1, dv)
    if not len(us) or not len(vs):
        return _sample_mesh(face, h)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
    uv = uv[_inside(uv, loops)]
    if not len(uv):
        return _sample_mesh(face, h)
    pts = np.array([face.point_at(float(a), float(b)) for a, b in uv])
    normals = np.array([face.outward_normal_at(float(a), float(b)) for a, b in uv])
    return pts, normals


def _sample_mesh(face: BrepFace, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: deterministic area-proportional samples on the face tessellation."""
    v, t = face.mesh_vertices, face.mesh_triangles
    if not len(t):
        return np.zeros((0, 3)), np.zeros((0, 3))
    tri = v[t]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    counts = np.maximum(1, np.round(areas / (h * h)).astype(int))
    rng = np.random.default_rng(12345 + face.index)
    pts, normals = [], []
    for k in range(len(t)):
        r1, r2 = rng.random(counts[k]), rng.random(counts[k])
        s = np.sqrt(r1)
        bary = np.stack([1 - s, s * (1 - r2), s * r2], axis=1)
        pts.append(bary @ tri[k])
        n = cross[k] / max(np.linalg.norm(cross[k]), 1e-12)
        normals.append(np.broadcast_to(n, (counts[k], 3)))
    return np.concatenate(pts), np.concatenate(normals)


def _role_labels(model: PartModel, role: tuple, compiled_bends: set[int]) -> tuple[int, int, int, int]:
    """(kind, skin, panel_key, bend_key) of a face role; walls are resolved later."""
    if role[0] == "panel":
        panel = model.panels[role[1]]
        return SEG_PANEL, panel.skin_of_side(role[2]), role[1], -1
    if role[0] == "bend":
        bend = model.bends[role[1]]
        skin = bend.oml_skin if role[2] == "outer" else 1 - bend.oml_skin
        if bend.key in compiled_bends:
            return SEG_BEND, skin, -1, bend.key
        return SEG_WALL, SKIN_WALL, bend.parent, -1
    if role[0] == "feature":
        feat = model.graph.features[role[1]]
        owner = feat.panel_key
        if owner is None and feat.bend_key is not None and feat.bend_key in model.bends:
            owner = model.bends[feat.bend_key].parent
        if owner is None:
            owner = model.datum
        return SEG_WALL, SKIN_WALL, owner, -1
    return -1, SKIN_WALL, -1, -1


def sample_final(model: PartModel, brep: BrepModel, compiled_bends: set[int], spacing: float) -> FinalSamples:
    """Final-pose labelled samples of every face (module docstring)."""
    from scipy.spatial import cKDTree

    chunks = {"p": [], "n": [], "k": [], "s": [], "pk": [], "bk": [], "f": []}
    wall_chunks = []
    for face in brep.faces:
        role = model.graph.face_role.get(face.index, ("wall",))
        pts, nrm = _sample_face(face, spacing)
        if not len(pts):
            continue
        kind, skin, pk, bk = _role_labels(model, role, compiled_bends)
        if kind < 0:
            wall_chunks.append((face.index, pts, nrm))
            continue
        m = len(pts)
        chunks["p"].append(pts)
        chunks["n"].append(nrm)
        chunks["k"].append(np.full(m, kind, dtype=np.uint8))
        chunks["s"].append(np.full(m, skin, dtype=np.uint8))
        chunks["pk"].append(np.full(m, pk, dtype=np.int64))
        chunks["bk"].append(np.full(m, bk, dtype=np.int64))
        chunks["f"].append(np.full(m, face.index, dtype=np.int64))
    if not chunks["p"]:
        raise ValueError("no skin samples")
    # every non-wall face (panel, bend, rigid bend and feature skins) is a reference
    # for the walls' segment inheritance and for the thin-wall test
    skin_p = np.concatenate(chunks["p"])
    src_k = np.concatenate(chunks["k"])
    src_pk = np.concatenate(chunks["pk"])
    src_bk = np.concatenate(chunks["bk"])
    tree = cKDTree(skin_p)
    far_fraction: dict[int, float] = {}
    # the frozen thin-wall test (unsupported.THIN_WALL_*): a fixed allowance, not the configured spacing
    limit = model.thickness + THIN_WALL_ALLOWANCE_MM
    for fi, pts, nrm in wall_chunks:
        dist, idx = tree.query(pts)
        far_fraction[fi] = float(np.mean(dist > limit))
        m = len(pts)
        k = np.where(src_k[idx] == SEG_BEND, SEG_BEND, SEG_WALL).astype(np.uint8)
        chunks["p"].append(pts)
        chunks["n"].append(nrm)
        chunks["k"].append(k)
        chunks["s"].append(np.full(m, SKIN_WALL, dtype=np.uint8))
        chunks["pk"].append(np.where(k == SEG_BEND, -1, src_pk[idx]).astype(np.int64))
        chunks["bk"].append(np.where(k == SEG_BEND, src_bk[idx], -1).astype(np.int64))
        chunks["f"].append(np.full(m, fi, dtype=np.int64))
    return FinalSamples(
        points=np.concatenate(chunks["p"]),
        normals=np.concatenate(chunks["n"]),
        kind=np.concatenate(chunks["k"]),
        skin=np.concatenate(chunks["s"]),
        panel_key=np.concatenate(chunks["pk"]),
        bend_key=np.concatenate(chunks["bk"]),
        face=np.concatenate(chunks["f"]),
        spacing=spacing,
        wall_far_fraction=far_fraction,
    )


def final_mesh(model: PartModel, brep: BrepModel, compiled_bends: set[int], samples: FinalSamples) -> FinalMesh:
    """Tessellation with per-vertex owner segments (walls: nearest skin sample)."""
    from scipy.spatial import cKDTree

    is_skin = samples.kind != SEG_WALL
    skin_like = is_skin | (samples.skin != SKIN_WALL)
    tree = cKDTree(samples.points[skin_like])
    src_pk, src_bk = samples.panel_key[skin_like], samples.bend_key[skin_like]
    verts, tris, pks, bks = [], [], [], []
    offset = 0
    for face in brep.faces:
        if not len(face.mesh_triangles):
            continue
        v = face.mesh_vertices
        role = model.graph.face_role.get(face.index, ("wall",))
        kind, _, pk, bk = _role_labels(model, role, compiled_bends)
        if kind < 0:
            _, idx = tree.query(v)
            pk_v, bk_v = src_pk[idx], src_bk[idx]
        else:
            pk_v = np.full(len(v), pk if kind != SEG_BEND else -1, dtype=np.int64)
            bk_v = np.full(len(v), bk if kind == SEG_BEND else -1, dtype=np.int64)
        verts.append(v)
        tris.append(face.mesh_triangles + offset)
        pks.append(pk_v)
        bks.append(bk_v)
        offset += len(v)
    return FinalMesh(np.concatenate(verts), np.concatenate(tris), np.concatenate(pks), np.concatenate(bks))


def _stage_map(model: PartModel, points: np.ndarray, normals: np.ndarray, panel_key: np.ndarray,
               bend_key: np.ndarray, formed: set[int], transforms: dict[int, np.ndarray],
               neutral: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    out_p = np.empty_like(points)
    out_n = np.empty_like(normals)
    for pk, t in transforms.items():
        sel = panel_key == pk
        if np.any(sel):
            out_p[sel] = apply4(t, points[sel])
            out_n[sel] = apply_dir4(t, normals[sel])
    for bk in np.unique(bend_key[bend_key >= 0]):
        bend = model.bends[int(bk)]
        sel = bend_key == bk
        t = transforms[bend.parent]
        p, n = points[sel], normals[sel]
        if bend.key not in formed:
            p, n = unroll_bend_points(bend, p, n, neutral[bend.key])
        out_p[sel] = apply4(t, p)
        out_n[sel] = apply_dir4(t, n)
    return out_p, out_n


def stage_samples(model: PartModel, samples: FinalSamples, formed: set[int], transforms: dict[int, np.ndarray],
                  neutral: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    """Stage-pose points and unit normals of the final samples."""
    p, n = _stage_map(model, samples.points, samples.normals, samples.panel_key, samples.bend_key, formed,
                      transforms, neutral)
    norms = np.linalg.norm(n, axis=1)
    n = n / np.where(norms > 1e-12, norms, 1.0)[:, None]
    return p, n


def stage_mesh(model: PartModel, mesh: FinalMesh, formed: set[int], transforms: dict[int, np.ndarray],
               neutral: dict[int, float]) -> np.ndarray:
    """Stage-pose vertices of the display mesh."""
    p, _ = _stage_map(model, mesh.vertices, np.zeros_like(mesh.vertices), mesh.panel_key, mesh.bend_key, formed,
                      transforms, neutral)
    return p


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Deterministic ``.npz``: fixed member timestamps, deflate level 1, no pickles.

    Grid samples with per-face constant normals and labels compress ~10x; zlib
    output is deterministic for identical input, so the file sha256 is stable.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for name, arr in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_NPZ_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
    path.write_bytes(buf.getvalue())


def write_ply(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    """Binary little-endian PLY: float32 x y z, ``list uchar int vertex_indices``."""
    v = np.ascontiguousarray(vertices, dtype="<f4")
    t = np.ascontiguousarray(triangles, dtype="<i4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {len(t)}\nproperty list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    face_dtype = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    faces = np.empty(len(t), dtype=face_dtype)
    faces["n"] = 3
    faces["i"] = t
    path.write_bytes(header + v.tobytes() + faces.tobytes())
