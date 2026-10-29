"""Analytic sheet-metal test parts (L, U, hat with holes) with exact bend truth.

A part is a 2D mid-surface profile (straight segments joined by fillets of
radius Ri + t/2) extruded along width W. The two skins are the profile offset
by +-t/2, so every bend has inner radius Ri and outer radius Ri + t. Truth per
bend and skin: the plane-intersection sharp and the arc apex (bisector point).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = ("L", "U", "hat")


@dataclass
class Part:
    vertices: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    bends: list           # per bend: {"included_deg", "outer_sign", "skins": {+1/-1: {sharp, apex}}}
    faces: list           # flat faces: {"segment", "skin", "normal", "probe"}
    thickness: float
    ri: float
    width: float


def _spec(kind):
    if kind == "L":
        return [40.0, 30.0], [60.0], 1.5, 1.5, 60.0, []
    if kind == "U":
        return [30.0, 50.0, 30.0], [90.0, 90.0], 2.0, 2.0, 70.0, []
    if kind == "hat":
        holes = [(0, 12.0, 20.0, 3.0), (2, 20.0, 40.0, 5.0), (4, 12.0, 60.0, 3.0), (1, 15.0, 30.0, 2.5)]
        return [28.0, 32.0, 44.0, 32.0, 28.0], [80.0, -80.0, -80.0, 80.0], 1.2, 1.0, 80.0, holes
    raise ValueError(kind)


def _left(v):
    return np.array([-v[1], v[0]])


def _profile(lengths, turns_deg, rm):
    """Polyline corners and fillet pieces of the mid-surface (x, z) profile."""
    heading = 0.0
    corners = [np.zeros(2)]
    dirs = []
    for k, length in enumerate(lengths):
        d = np.array([np.cos(heading), np.sin(heading)])
        dirs.append(d)
        corners.append(corners[-1] + length * d)
        if k < len(turns_deg):
            heading += np.radians(turns_deg[k])
    tangent_len = [rm * np.tan(np.radians(abs(t)) / 2.0) for t in turns_deg]
    pieces = []
    for k, d in enumerate(dirs):
        a = corners[k] + (tangent_len[k - 1] if k > 0 else 0.0) * d
        b = corners[k + 1] - (tangent_len[k] if k < len(turns_deg) else 0.0) * d
        pieces.append(("line", a, b, d, k))
        if k < len(turns_deg):
            sign = np.sign(turns_deg[k])
            centre = b + sign * rm * _left(d)
            a0 = np.arctan2(*(b - centre)[::-1])
            pieces.append(("arc", centre, a0, np.radians(turns_deg[k]), k))
    return corners, dirs, pieces


def _sample(pieces, rm, step):
    """Mid-curve samples: (pos, tangent, segment index or -1 for arcs)."""
    pos, tan, seg = [], [], []
    for piece in pieces:
        if piece[0] == "line":
            _, a, b, d, k = piece
            n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
            for u in np.linspace(0, 1, n):
                pos.append(a + u * (b - a))
                tan.append(d)
                seg.append(k)
        else:
            _, centre, a0, sweep, k = piece
            n = max(3, int(np.ceil(abs(sweep) * rm / step)) + 1)
            for u in np.linspace(0, 1, n)[1:-1]:
                ang = a0 + u * sweep
                p = centre + rm * np.array([np.cos(ang), np.sin(ang)])
                tvec = np.sign(sweep) * np.array([-np.sin(ang), np.cos(ang)])
                pos.append(p)
                tan.append(tvec)
                seg.append(-1)
    return np.array(pos), np.array(tan), np.array(seg)


def random_pose(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return rot, rng.uniform(-300, 300, size=3)


def make_part(kind, *, step=0.5, noise=0.02, seed=0):
    lengths, turns, t, ri, width, holes = _spec(kind)
    rm = ri + t / 2.0
    corners, dirs, pieces = _profile(lengths, turns, rm)
    pos, tan, seg = _sample(pieces, rm, step)
    nrm2 = np.array([_left(v) for v in tan])
    arclen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    ys = np.linspace(0.0, width, int(np.ceil(width / step)) + 1)
    rot, trans = random_pose(seed)
    rng = np.random.default_rng(seed + 1)

    def to3(p2, y):
        x, z, y = np.broadcast_arrays(p2[..., 0], p2[..., 1], np.asarray(y, dtype=float))
        return np.stack([x, y, z], axis=-1)

    verts, norms, tris = [], [], []
    seg_start = {k: arclen[seg == k].min() for k in range(len(lengths))}
    for skin in (+1, -1):
        p2 = pos + skin * (t / 2.0) * nrm2
        grid = to3(p2[:, None, :], ys[None, :])                 # (S, Y, 3)
        n3 = to3(skin * nrm2[:, None, :], np.zeros((1, 1)))     # outward = skin * left normal
        n3 = np.broadcast_to(n3, grid.shape)
        jitter = rng.normal(0.0, noise, size=grid.shape[:2])[..., None] * n3
        base = sum(len(v) for v in verts)
        S, Y = grid.shape[:2]
        verts.append((grid + jitter).reshape(-1, 3))
        norms.append(np.array(n3).reshape(-1, 3))
        ii, jj = np.meshgrid(np.arange(S - 1), np.arange(Y - 1), indexing="ij")
        v00 = base + ii * Y + jj
        v10, v01, v11 = v00 + Y, v00 + 1, v00 + Y + 1
        quad_u = (arclen[ii] + arclen[ii + 1]) / 2.0
        quad_y = (ys[jj] + ys[jj + 1]) / 2.0
        quad_seg = np.where(seg[ii] == seg[ii + 1], seg[ii], -1)
        keep = np.ones(ii.shape, dtype=bool)
        for (hk, hu, hy, hr) in holes:
            local_u = quad_u - seg_start[hk]
            keep &= ~((quad_seg == hk) & ((local_u - hu) ** 2 + (quad_y - hy) ** 2 < hr ** 2))
        t1 = np.stack([v00, v10, v11], axis=-1)[keep]
        t2 = np.stack([v00, v11, v01], axis=-1)[keep]
        tris.append(np.concatenate([t1, t2]))
    vertices = np.concatenate(verts)
    normals = np.concatenate(norms)
    triangles = np.concatenate(tris).astype(np.int64)
    # orient triangle winding to the outward normal
    e = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                 vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    flip = np.einsum("ij,ij->i", e, normals[triangles[:, 0]]) < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]

    def pose_pt(p):
        return np.asarray(p) @ rot.T + trans

    bends = []
    for k, turn in enumerate(turns):
        a, b = dirs[k], dirs[k + 1]
        sign = np.sign(turn)
        centre = corners[k + 1] - rm * np.tan(np.radians(abs(turn)) / 2.0) * a + sign * rm * _left(a)
        skins = {}
        for skin in (+1, -1):
            p1 = corners[k + 1] + skin * (t / 2.0) * _left(a)
            p2 = corners[k + 1] + skin * (t / 2.0) * _left(b)
            mat = np.stack([a, -b], axis=1)
            lam = np.linalg.solve(mat, p2 - p1)
            sharp2 = p1 + lam[0] * a
            radius = rm - skin * sign * t / 2.0
            v = corners[k + 1] - centre
            apex2 = centre + radius * v / np.linalg.norm(v)
            skins[skin] = {
                "sharp": [pose_pt(to3(sharp2, 0.0)), pose_pt(to3(sharp2, width))],
                "apex": [pose_pt(to3(apex2, 0.0)), pose_pt(to3(apex2, width))],
            }
        bends.append({"included_deg": 180.0 - abs(turn), "outer_sign": int(-sign),
                      "segments": (k, k + 1), "skins": skins})
    faces = []
    for k, d in enumerate(dirs):
        mid2 = (corners[k] + corners[k + 1]) / 2.0
        for skin in (+1, -1):
            n2 = skin * _left(d)
            probe2 = mid2 + skin * (t / 2.0) * _left(d)
            faces.append({"segment": k, "skin": skin, "normal": (to3(n2, 0.0) @ rot.T),
                          "probe": pose_pt(to3(probe2, width * 0.15))})
    return Part(vertices=vertices @ rot.T + trans, normals=normals @ rot.T, triangles=triangles,
                bends=bends, faces=faces, thickness=t, ri=ri, width=width)


def write_part(part, path, *, as_point_cloud=False, with_normals=True):
    import open3d as o3d

    path = str(path)
    if as_point_cloud:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(part.vertices))
        if with_normals:
            cloud.normals = o3d.utility.Vector3dVector(part.normals)
        o3d.io.write_point_cloud(path, cloud, write_ascii=False)
        return path
    used = np.unique(part.triangles)
    remap = -np.ones(len(part.vertices), dtype=np.int64)
    remap[used] = np.arange(len(used))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(part.vertices[used]),
                                     o3d.utility.Vector3iVector(remap[part.triangles].astype(np.int32)))
    if path.lower().endswith(".stl"):
        mesh.compute_triangle_normals()
    elif with_normals:
        mesh.vertex_normals = o3d.utility.Vector3dVector(part.normals[used])
    o3d.io.write_triangle_mesh(path, mesh, write_ascii=False)
    return path
