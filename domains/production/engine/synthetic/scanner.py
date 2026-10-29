"""Virtual ZEISS-like structured-light scan cell (raycasting with self-occlusion).

The scanner looks at a :class:`~.brep.TessellatedSurface` (the as-built part,
already posed in the raw scanner frame) from robot viewpoints grouped in
*setups* (a ZEISS cell scans one side, the part is turned, the other side is
scanned, and the software merges the setups). Per view:

1. Pinhole camera rays over the part's projected bounding box (Open3D
   ``RaycastingScene``, Embree): first hits only, so self-occlusion is exact.
2. **Triangulation shadowing**: the projector sits on a baseline at
   ``triangulation_angle_deg``; a point must also be visible from it
   (shadow ray via ``test_occlusions``).
3. **Grazing dropout**: keep probability falls linearly from 1 at
   ``grazing_start_deg`` to 0 at ``grazing_cutoff_deg`` (worse of camera and
   projector incidence).
4. **Specular dropout**: where the projector's mirror reflection points into
   the camera within ``specular_cone_deg`` (washed-out fringes on bright
   sheet metal), points drop with ``specular_dropout_probability``.
5. Random dropout; **edge erosion** and **edge flare** at depth
   discontinuities of the scene (silhouettes and jumps between surfaces in the
   pre-dropout range image -- a dropout hole is not an edge); flare = mixed
   pixels pushed away from the camera by up to ``edge_flare_mm``.
6. **Range noise** along the ray, ``sigma = noise_sigma_mm * (1 + gain *
   (1/cos(theta) - 1))``.
7. A small per-view residual registration jitter.

Merging (like the cell software fusing all measurements into one cloud):

- per setup, overlapping views are merged by voxel *ownership* at the voxel
  size (the best-incidence view keeps the voxel, like a merge keeping the best
  measurement);
- across setups the same rule applies, so overlapping setups do **not** double
  the surface, except inside *duplicate-skin patches*: coarse cells of
  ``duplicate_skin_patch_mm`` chosen with probability
  ``duplicate_skin_fraction``, where the other setup's points are kept too.
  Every setup after the first carries a rigid misregistration
  (``duplicate_skin_offset_mm`` / ``duplicate_skin_rotation_deg``), so those
  patches show the doubled, offset skins seen on real exports;
- sparse **outliers** are added last, and the point order is randomly
  permuted (file order carries no label: outliers, setups and views are
  interleaved; mesh faces are re-indexed).

Output: points (and optionally per-view range-grid triangles) in the raw
scanner frame, plus index-aligned truth labels (segment, skin, view, setup,
flare/outlier flags and the noise-free hit point in the CAD frame), and
``stats["nn_spacing_mm"]``, the median nearest-neighbour distance of the
exported points (:func:`median_nn_spacing`, the model-card statistic).

Density. ``ScannerModel.point_spacing_mm`` is the **median nearest-neighbour
distance of the exported points**, the statistic a model card measures on a
real export. :func:`scan_surface` itself works at an explicit voxel size; the
case generator calibrates the voxel per case so the exported cloud has the
model's spacing (``domains.production.engine.synthetic.cases``).

The default :class:`ScannerModel` values are **assumptions** (``provenance =
"ASSUMED"``) except the point spacing, whose 0.46 mm default follows the
median nearest-neighbour spacing documented for the local physical references
in ``scripts/generate_f1_calibrated_raw_range_synthetic_corpus_v1.py`` (source
comment, 0.463 mm; no data read). The amendment-4b model card measured by E4
on authorized real scans replaces it through :meth:`ScannerModel.from_json`; a
``MODEL_CARD`` must state every parameter.

Threads. Open3D/Embree raycasting uses ``ray_threads`` threads (0 = all
cores). :func:`default_ray_threads` follows ``OMP_NUM_THREADS`` so a generator
run does not disturb a concurrent runtime measurement; results do not depend
on the thread count.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from .brep import TessellatedSurface
from .geometry import apply_points, rotation_about_axis

__all__ = [
    "SCANNER_MODEL_SCHEMA",
    "SCAN_PLAN_SCHEMA",
    "ScannerModel",
    "Viewpoint",
    "ScanSetup",
    "ScanPlan",
    "SCAN_PLANS",
    "ScanResult",
    "scan_surface",
    "random_pose",
    "median_nn_spacing",
    "default_ray_threads",
    "LABEL_OUTLIER",
]

SCANNER_MODEL_SCHEMA = "sherman_synthetic_scanner_model_v1"
SCAN_PLAN_SCHEMA = "sherman_synthetic_scan_plan_v1"
LABEL_OUTLIER = 255
#: Query points used by :func:`median_nn_spacing` (an evenly spaced subset).
_NN_SAMPLE = 50_000


def _sha_json(doc: Any) -> str:
    data = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def default_ray_threads() -> int:
    """Raycasting threads: ``OMP_NUM_THREADS`` when it is a positive integer, else 0 (all cores)."""
    raw = os.environ.get("OMP_NUM_THREADS", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 0
    return value if value > 0 else 0


def median_nn_spacing(points: np.ndarray, sample: int = _NN_SAMPLE) -> float:
    """Median distance from a point to its nearest other point (the model-card
    spacing statistic). Queries an evenly spaced subset of at most ``sample``
    points against all points (deterministic; the scanner's output order is
    already random, so the subset is a random sample)."""
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("need at least two points")
    tree = cKDTree(pts)
    q = pts[np.unique(np.linspace(0, len(pts) - 1, min(len(pts), sample)).astype(np.int64))]
    dist, _ = tree.query(q, k=2)
    return float(np.median(dist[:, 1]))


@dataclass(frozen=True)
class ScannerModel:
    """Sensor + merge error model. All lengths mm, angles degrees.

    ``point_spacing_mm`` is the median nearest-neighbour distance of the
    exported points (the statistic a model card measures on a real export).
    ``provenance`` is ``ASSUMED`` (defaults), ``MODEL_CARD`` (measured on
    authorized real scans, amendment 4b) or ``IDEALISED`` (diagnostics only,
    never G1)."""

    name: str = "zeiss_atos_like_assumed_v1"
    provenance: str = "ASSUMED"
    point_spacing_mm: float = 0.46
    noise_sigma_mm: float = 0.025
    noise_incidence_gain: float = 1.0
    edge_flare_mm: float = 0.4
    edge_flare_probability: float = 0.25
    edge_erosion_probability: float = 0.2
    depth_jump_mm: float = 1.5
    grazing_start_deg: float = 65.0
    grazing_cutoff_deg: float = 82.0
    specular_cone_deg: float = 6.0
    specular_dropout_probability: float = 0.6
    random_dropout_probability: float = 0.002
    outlier_fraction: float = 0.0015
    outlier_range_mm: float = 4.0
    duplicate_skin_offset_mm: float = 0.04
    duplicate_skin_rotation_deg: float = 0.01
    #: Fraction of cross-setup overlap (in coarse patches) where the merge keeps both
    #: setups' points: the doubled, offset skins of a real multi-setup export.
    duplicate_skin_fraction: float = 0.2
    duplicate_skin_patch_mm: float = 25.0
    view_jitter_mm: float = 0.008
    triangulation_angle_deg: float = 20.0
    working_distance_mm: float = 650.0
    notes: str = "Assumed defaults pending the amendment-4b model card (E4)."

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.provenance not in ("ASSUMED", "MODEL_CARD", "IDEALISED"):
            raise ValueError("provenance must be ASSUMED, MODEL_CARD or IDEALISED")
        positive = ("point_spacing_mm", "depth_jump_mm", "working_distance_mm", "duplicate_skin_patch_mm")
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0")
        non_negative = (
            "noise_sigma_mm", "noise_incidence_gain", "edge_flare_mm", "outlier_range_mm",
            "duplicate_skin_offset_mm", "duplicate_skin_rotation_deg", "view_jitter_mm", "specular_cone_deg",
        )
        for name in non_negative:
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be finite and >= 0")
        probs = (
            "edge_flare_probability", "edge_erosion_probability", "specular_dropout_probability",
            "random_dropout_probability", "outlier_fraction", "duplicate_skin_fraction",
        )
        for name in probs:
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 < self.grazing_start_deg < self.grazing_cutoff_deg <= 90.0:
            raise ValueError("need 0 < grazing_start_deg < grazing_cutoff_deg <= 90")
        if not 0.0 <= self.triangulation_angle_deg < 60.0:
            raise ValueError("triangulation_angle_deg must be in [0, 60)")

    # -- JSON ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCANNER_MODEL_SCHEMA, **asdict(self)}

    def sha256(self) -> str:
        """Identity of the model (sha256 of its sorted-key JSON, exact floats)."""
        return _sha_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScannerModel":
        data = dict(data)
        schema = data.pop("schema", SCANNER_MODEL_SCHEMA)
        if schema != SCANNER_MODEL_SCHEMA:
            raise ValueError(f"unsupported scanner model schema {schema!r}")
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown scanner model fields: {sorted(unknown)}")
        if data.get("provenance") == "MODEL_CARD":
            # a measured card states every parameter; nothing silently falls back to ASSUMED
            missing = sorted(known - set(data))
            if missing:
                raise ValueError(f"a MODEL_CARD scanner model must state every field; missing: {missing}")
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> "ScannerModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | os.PathLike) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def idealised(cls) -> "ScannerModel":
        """Noise-free, artefact-free scanner (diagnostics only; never for G1)."""
        return cls(
            name="idealised", provenance="IDEALISED", noise_sigma_mm=0.0, noise_incidence_gain=0.0,
            edge_flare_mm=0.0, edge_flare_probability=0.0, edge_erosion_probability=0.0,
            specular_dropout_probability=0.0, random_dropout_probability=0.0, outlier_fraction=0.0,
            duplicate_skin_offset_mm=0.0, duplicate_skin_rotation_deg=0.0, duplicate_skin_fraction=0.0,
            view_jitter_mm=0.0, grazing_start_deg=85.0, grazing_cutoff_deg=89.9,
            notes="Idealised sensor; diagnostics only.",
        )


@dataclass(frozen=True)
class Viewpoint:
    """Sensor position on a sphere around the part: azimuth/elevation in the raw
    scanner frame (+Z up); ``distance_mm`` defaults to the working distance."""

    azimuth_deg: float
    elevation_deg: float
    distance_mm: float | None = None


@dataclass(frozen=True)
class ScanSetup:
    name: str
    viewpoints: tuple[Viewpoint, ...]


@dataclass(frozen=True)
class ScanPlan:
    """Robot program: setups of viewpoints (JSON-loadable via :meth:`from_dict`)."""

    name: str
    setups: tuple[ScanSetup, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCAN_PLAN_SCHEMA,
            "name": self.name,
            "setups": [
                {"name": s.name, "viewpoints": [asdict(v) for v in s.viewpoints]} for s in self.setups
            ],
        }

    def sha256(self) -> str:
        return _sha_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScanPlan":
        if data.get("schema", SCAN_PLAN_SCHEMA) != SCAN_PLAN_SCHEMA:
            raise ValueError("unsupported scan plan schema")
        setups = tuple(
            ScanSetup(str(s["name"]), tuple(Viewpoint(**v) for v in s["viewpoints"])) for s in data["setups"]
        )
        if not setups or not all(s.viewpoints for s in setups):
            raise ValueError("a scan plan needs at least one setup with viewpoints")
        return cls(str(data["name"]), setups)

    @property
    def view_count(self) -> int:
        return sum(len(s.viewpoints) for s in self.setups)


def _ring(elev: float, n: int, phase: float = 0.0) -> tuple[Viewpoint, ...]:
    return tuple(Viewpoint(phase + i * 360.0 / n, elev) for i in range(n))


#: Named robot programs. FULL is a two-setup (top + flipped) program; the others
#: progressively remove views to raise occlusion.
SCAN_PLANS: dict[str, ScanPlan] = {
    "FULL": ScanPlan(
        "FULL",
        (
            ScanSetup("TOP", _ring(35.0, 6) + _ring(65.0, 4, 45.0) + (Viewpoint(0.0, 89.0),)),
            ScanSetup("BOTTOM", _ring(-35.0, 6, 30.0) + _ring(-65.0, 4) + (Viewpoint(90.0, -89.0),)),
        ),
    ),
    "TOP_ONLY": ScanPlan("TOP_ONLY", (ScanSetup("TOP", _ring(35.0, 6) + _ring(65.0, 4, 45.0) + (Viewpoint(0.0, 89.0),)),)),
    "SPARSE": ScanPlan("SPARSE", (ScanSetup("TOP", _ring(45.0, 4, 20.0)),)),
    "SINGLE": ScanPlan("SINGLE", (ScanSetup("TOP", (Viewpoint(10.0, 80.0),)),)),
}


def random_pose(rng: np.random.Generator, max_translation_mm: float = 150.0) -> np.ndarray:
    """Uniformly random rotation and a translation in a +-``max_translation_mm`` box."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = rng.uniform(-max_translation_mm, max_translation_mm, size=3)
    return m


def _small_rigid(rng: np.random.Generator, trans_sigma: float, rot_sigma_deg: float, center: np.ndarray) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = math.radians(float(rng.normal(0.0, rot_sigma_deg))) if rot_sigma_deg > 0 else 0.0
    m = rotation_about_axis(center, axis, angle)
    m[:3, 3] += rng.normal(0.0, trans_sigma, size=3) if trans_sigma > 0 else 0.0
    return m


@dataclass
class ScanResult:
    """Merged scan in the raw frame plus index-aligned truth labels."""

    points: np.ndarray  # (N, 3) float64 raw frame
    faces: np.ndarray | None  # (M, 3) int64 or None (points mode)
    seg_kind: np.ndarray  # (N,) uint8 (255 = outlier)
    seg_index: np.ndarray  # (N,) int16 (-1 = outlier)
    skin: np.ndarray  # (N,) uint8 flat-frame skin (2 = edge wall, 255 = outlier)
    view: np.ndarray  # (N,) int16 (-1 = outlier)
    setup: np.ndarray  # (N,) int8 (-1 = outlier)
    flare: np.ndarray  # (N,) bool
    ideal_raw: np.ndarray  # (N, 3) noise-free surface point (raw frame), NaN for outliers
    spacing_mm: float
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def outlier(self) -> np.ndarray:
        return self.seg_kind == LABEL_OUTLIER


def _camera(target: np.ndarray, vp: Viewpoint, distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    az, el = math.radians(vp.azimuth_deg), math.radians(vp.elevation_deg)
    direction = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    c = target + distance * direction
    f = -direction
    ref = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    r = np.cross(f, ref)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return c, f, r, u


def scan_surface(
    surface_raw: TessellatedSurface,
    model: ScannerModel,
    plan: ScanPlan,
    rng: np.random.Generator,
    *,
    spacing_mm: float | None = None,
    output: str = "points",
    ray_threads: int | None = None,
) -> ScanResult:
    """Scan ``surface_raw`` (already in the raw frame) with ``model`` and ``plan``.

    ``spacing_mm`` is the merge **voxel size** (default: the model's point
    spacing). The exported cloud's median nearest-neighbour spacing is somewhat
    smaller than the voxel; it is reported as ``stats["nn_spacing_mm"]``, and
    the case generator calibrates the voxel so that it equals
    ``model.point_spacing_mm``. ``output`` is ``"points"`` (one point per
    voxel) or ``"mesh"`` (per-view range-grid triangles over the owned pixels).
    ``ray_threads``: raycasting threads (``None``: :func:`default_ray_threads`)."""
    import open3d as o3d

    if output not in ("points", "mesh"):
        raise ValueError("output must be 'points' or 'mesh'")
    spacing = float(spacing_mm if spacing_mm is not None else model.point_spacing_mm)
    if not spacing > 0:
        raise ValueError("spacing must be > 0")
    if ray_threads is None:
        ray_threads = default_ray_threads()
    if ray_threads < 0:
        raise ValueError("ray_threads must be >= 0")
    verts = surface_raw.vertices
    tris = surface_raw.triangles
    tri_normals = surface_raw.triangle_normals
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(verts.astype(np.float32)), o3d.core.Tensor(tris.astype(np.uint32)))
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    target = 0.5 * (lo + hi)
    radius = 0.5 * float(np.linalg.norm(hi - lo))
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    # Merge-level randomness is drawn first, so it does not depend on how many
    # per-view draws the voxel size causes: the per-setup misregistrations and the
    # seed of the duplicate-skin patch hash (patches are a function of position).
    misregs = {
        setup_id: _small_rigid(rng, model.duplicate_skin_offset_mm, model.duplicate_skin_rotation_deg, target)
        for setup_id in range(1, len(plan.setups))
    }
    patch_seed = int(rng.integers(0, 2**62))
    # points: oversample then keep one point per voxel; mesh: native grid density
    ray_spacing = 0.85 * spacing if output == "points" else spacing
    view_records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"views": 0, "rays": 0, "hits": 0, "shadowed": 0, "grazing_dropped": 0, "specular_dropped": 0,
                             "random_dropped": 0, "eroded": 0, "flared": 0, "cameras_raw": []}
    view_id = 0
    for setup_id, setup in enumerate(plan.setups):
        for vp in setup.viewpoints:
            distance = max(float(vp.distance_mm or model.working_distance_mm), 2.5 * radius)
            stats["cameras_raw"].append([float(x) for x in _camera(target, vp, distance)[0]])
            rec = _scan_view(scene, tri_normals, model, rng, target, corners, vp, distance, ray_spacing, ray_threads, stats)
            if rec is not None:
                rec["view"] = view_id
                rec["setup"] = setup_id
                view_records.append(rec)
            view_id += 1
            stats["views"] += 1
    # 1) within-setup merge: voxel ownership by best incidence
    keeps: dict[int, np.ndarray] = {}
    setup_ids = sorted({r["setup"] for r in view_records})
    for setup_id in setup_ids:
        idx = [i for i, r in enumerate(view_records) if r["setup"] == setup_id]
        for i, mask in zip(idx, _ownership([view_records[i] for i in idx], spacing, one_per_voxel=(output == "points"))):
            keeps[i] = mask
    # 2) cross-setup merge: one owner setup per voxel, except in duplicate-skin patches
    stats["cross_setup_dropped"] = 0
    stats["duplicate_skin_points"] = 0
    if len(setup_ids) > 1:
        keeps = _merge_setups(view_records, keeps, spacing, model, patch_seed, stats)
    # 3) assemble, with a rigid misregistration for every setup after the first
    pts_all, ideal_all, prim_all, view_all, setup_all, flare_all = [], [], [], [], [], []
    face_blocks: list[np.ndarray] = []
    offset = 0
    for setup_id in range(len(plan.setups)):
        recs = [(i, r) for i, r in enumerate(view_records) if r["setup"] == setup_id]
        if not recs:
            continue
        misreg = misregs.get(setup_id, np.eye(4))
        for i, rec in recs:
            keep = keeps[i]
            idx = np.flatnonzero(keep)
            if len(idx) == 0:
                continue
            p = apply_points(misreg, rec["points"][idx])
            pts_all.append(p)
            ideal_all.append(rec["ideal"][idx])
            prim_all.append(rec["prim"][idx])
            view_all.append(np.full(len(idx), rec["view"], dtype=np.int16))
            setup_all.append(np.full(len(idx), setup_id, dtype=np.int8))
            flare_all.append(rec["flare"][idx])
            if output == "mesh":
                local = np.full(len(keep), -1, dtype=np.int64)
                local[idx] = np.arange(len(idx)) + offset
                f = _grid_faces(rec, keep, local, model.depth_jump_mm)
                if len(f):
                    face_blocks.append(f)
            offset += len(idx)
    if not pts_all:
        raise ValueError("the scan captured no points (check the plan and pose)")
    points = np.vstack(pts_all)
    ideal = np.vstack(ideal_all)
    prim = np.concatenate(prim_all)
    seg_kind = surface_raw.seg_kind[prim].astype(np.uint8)
    seg_index = surface_raw.seg_index[prim].astype(np.int16)
    skin = surface_raw.skin[prim].astype(np.uint8)
    view_arr = np.concatenate(view_all)
    setup_arr = np.concatenate(setup_all)
    flare = np.concatenate(flare_all)
    faces = np.vstack(face_blocks) if (output == "mesh" and face_blocks) else (np.zeros((0, 3), dtype=np.int64) if output == "mesh" else None)
    # outliers
    n_out = int(round(model.outlier_fraction * len(points)))
    if n_out > 0:
        half = n_out // 2
        sky = rng.uniform(lo - model.outlier_range_mm, hi + model.outlier_range_mm, size=(half, 3))
        base = points[rng.integers(0, len(points), size=n_out - half)]
        dirs = rng.normal(size=(n_out - half, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        near = base + dirs * rng.uniform(0.3, max(0.31, model.outlier_range_mm), size=(n_out - half, 1))
        outl = np.vstack([sky, near])
        points = np.vstack([points, outl])
        ideal = np.vstack([ideal, np.full((n_out, 3), np.nan)])
        seg_kind = np.concatenate([seg_kind, np.full(n_out, LABEL_OUTLIER, dtype=np.uint8)])
        seg_index = np.concatenate([seg_index, np.full(n_out, -1, dtype=np.int16)])
        skin = np.concatenate([skin, np.full(n_out, LABEL_OUTLIER, dtype=np.uint8)])
        view_arr = np.concatenate([view_arr, np.full(n_out, -1, dtype=np.int16)])
        setup_arr = np.concatenate([setup_arr, np.full(n_out, -1, dtype=np.int8)])
        flare = np.concatenate([flare, np.zeros(n_out, dtype=bool)])
    # 4) random point order: file order must not reveal setups, views or outliers
    perm = rng.permutation(len(points))
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    points, ideal = points[perm], ideal[perm]
    seg_kind, seg_index, skin = seg_kind[perm], seg_index[perm], skin[perm]
    view_arr, setup_arr, flare = view_arr[perm], setup_arr[perm], flare[perm]
    if faces is not None and len(faces):
        faces = inverse[faces]
    stats["points"] = int(len(points))
    stats["outliers"] = int(n_out)
    stats["faces"] = None if faces is None else int(len(faces))
    stats["voxel_mm"] = spacing
    stats["ray_spacing_mm"] = ray_spacing
    stats["nn_spacing_mm"] = median_nn_spacing(points) if len(points) >= 2 else None
    return ScanResult(points, faces, seg_kind, seg_index, skin, view_arr, setup_arr, flare, ideal, spacing, stats)


def _scan_view(
    scene,
    tri_normals: np.ndarray,
    model: ScannerModel,
    rng: np.random.Generator,
    target: np.ndarray,
    corners: np.ndarray,
    vp: Viewpoint,
    distance: float,
    ray_spacing: float,
    ray_threads: int,
    stats: dict[str, Any],
) -> dict[str, Any] | None:
    """Capture one view; returns compact per-point arrays (or None when empty)."""
    import open3d as o3d

    c, f, r, u = _camera(target, vp, distance)
    focal = distance / ray_spacing
    rel = corners - c
    depth_c = rel @ f
    if np.any(depth_c <= 1e-6):
        return None
    xs = focal * (rel @ r) / depth_c
    ys = focal * (rel @ u) / depth_c
    x0, x1 = int(math.floor(xs.min())) - 2, int(math.ceil(xs.max())) + 2
    y0, y1 = int(math.floor(ys.min())) - 2, int(math.ceil(ys.max())) + 2
    gx = np.arange(x0, x1 + 1, dtype=np.float64)
    gy = np.arange(y0, y1 + 1, dtype=np.float64)
    h, w = len(gy), len(gx)
    dirs = f[None, None, :] + (gx[None, :, None] / focal) * r[None, None, :] + (gy[:, None, None] / focal) * u[None, None, :]
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    rays = np.empty((h, w, 6), dtype=np.float32)
    rays[..., :3] = c.astype(np.float32)
    rays[..., 3:] = dirs.astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays), nthreads=ray_threads)
    del rays
    t_hit = ans["t_hit"].numpy().ravel()
    stats["rays"] += int(h * w)
    hit_idx = np.flatnonzero(np.isfinite(t_hit))
    if len(hit_idx) == 0:
        return None
    prim = ans["primitive_ids"].numpy().ravel()[hit_idx].astype(np.int64)
    d_h = dirs.reshape(-1, 3)[hit_idx]
    del dirs
    t_h = t_hit[hit_idx].astype(np.float64)
    n_h = tri_normals[prim]
    cos_c = -np.einsum("ij,ij->i", n_h, d_h)
    keep = cos_c > 0.02
    p_h = c[None, :] + d_h * t_h[:, None]
    proj = c + r * distance * math.tan(math.radians(model.triangulation_angle_deg))
    to_proj = proj[None, :] - p_h
    dist_p = np.linalg.norm(to_proj, axis=1)
    lp = to_proj / np.where(dist_p > 0, dist_p, 1.0)[:, None]
    cos_p = np.einsum("ij,ij->i", n_h, lp)
    keep &= cos_p > 0.02
    stats["hits"] += int(keep.sum())
    if model.triangulation_angle_deg > 0 and keep.any():
        vi = np.flatnonzero(keep)
        origin = p_h[vi] + 2e-3 * n_h[vi]
        sray = np.hstack([origin, proj[None, :] - origin]).astype(np.float32)
        occ = scene.test_occlusions(o3d.core.Tensor(sray), tnear=1e-5, tfar=1.0 - 1e-6, nthreads=ray_threads).numpy().astype(bool)
        stats["shadowed"] += int(occ.sum())
        keep[vi[occ]] = False
    theta = np.degrees(np.arccos(np.clip(np.minimum(cos_c, cos_p), -1.0, 1.0)))
    keep_p = np.clip((model.grazing_cutoff_deg - theta) / (model.grazing_cutoff_deg - model.grazing_start_deg), 0.0, 1.0)
    graze = keep & (rng.random(len(hit_idx)) >= keep_p)
    stats["grazing_dropped"] += int(graze.sum())
    keep &= ~graze
    if model.specular_dropout_probability > 0 and model.specular_cone_deg > 0:
        refl = 2.0 * cos_p[:, None] * n_h - lp
        cos_s = -np.einsum("ij,ij->i", refl, d_h)
        spec = keep & (cos_s > math.cos(math.radians(model.specular_cone_deg))) & (
            rng.random(len(hit_idx)) < model.specular_dropout_probability
        )
        stats["specular_dropped"] += int(spec.sum())
        keep &= ~spec
    if model.random_dropout_probability > 0:
        rnd = keep & (rng.random(len(hit_idx)) < model.random_dropout_probability)
        stats["random_dropped"] += int(rnd.sum())
        keep &= ~rnd
    if not keep.any():
        return None
    # Mixed pixels happen at true depth discontinuities of the scene (silhouettes
    # against the background, jumps between surfaces), so the boundary comes from
    # all first hits *before* any dropout: a dropout hole is not an edge.
    hit2d = np.isfinite(t_hit).reshape(h, w)
    depth_all = np.where(hit2d, t_hit.reshape(h, w).astype(np.float64), np.nan)
    boundary = _boundary_mask(hit2d, depth_all, model.depth_jump_mm).ravel()[hit_idx]
    if model.edge_erosion_probability > 0:
        er = keep & boundary & (rng.random(len(hit_idx)) < model.edge_erosion_probability)
        stats["eroded"] += int(er.sum())
        keep &= ~er
        boundary &= ~er
    sigma = model.noise_sigma_mm * (1.0 + model.noise_incidence_gain * (1.0 / np.clip(cos_c, 0.05, 1.0) - 1.0))
    noise = rng.normal(0.0, 1.0, size=len(hit_idx)) * sigma
    flare = np.zeros(len(hit_idx), dtype=bool)
    if model.edge_flare_probability > 0 and model.edge_flare_mm > 0:
        flare = keep & boundary & (rng.random(len(hit_idx)) < model.edge_flare_probability)
        noise = noise + flare * rng.uniform(0.0, model.edge_flare_mm, size=len(hit_idx))
        stats["flared"] += int(flare.sum())
    sel = np.flatnonzero(keep)
    ideal = p_h[sel]
    noisy = ideal + d_h[sel] * noise[sel, None]
    if model.view_jitter_mm > 0:
        noisy = apply_points(_small_rigid(rng, model.view_jitter_mm, 0.0, target), noisy)
    return {
        "points": noisy,
        "ideal": ideal,
        "prim": prim[sel],
        "flare": flare[sel],
        "cos": cos_c[sel],
        "grid_index": hit_idx[sel],
        "shape": (h, w),
        "depth": t_h[sel],
        "camera": c,
    }


def _boundary_mask(valid: np.ndarray, depth: np.ndarray, jump: float) -> np.ndarray:
    h, w = valid.shape
    pad_v = np.zeros((h + 2, w + 2), dtype=bool)
    pad_v[1:-1, 1:-1] = valid
    pad_d = np.full((h + 2, w + 2), np.nan)
    pad_d[1:-1, 1:-1] = depth
    boundary = np.zeros((h, w), dtype=bool)
    centre = pad_d[1:-1, 1:-1]
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb_v = pad_v[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
        nb_d = pad_d[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
        with np.errstate(invalid="ignore"):
            jumpy = np.abs(nb_d - centre) > jump
        boundary |= ~nb_v | (nb_v & jumpy)
    return boundary & valid


def _linear_keys(keys: np.ndarray) -> np.ndarray:
    keys = keys - keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    return (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]


def _cell_uniform(cells: np.ndarray, seed: int) -> np.ndarray:
    """A deterministic uniform [0, 1) value per integer cell (splitmix64 mixing of the
    cell coordinates and ``seed``): the same cell always gets the same value, no
    matter which other cells are present."""
    c = cells.astype(np.int64).view(np.uint64)
    with np.errstate(over="ignore"):
        x = (c[:, 0] * np.uint64(0x9E3779B97F4A7C15)) ^ (c[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)) ^ (
            c[:, 2] * np.uint64(0x165667B19E3779F9)) ^ np.uint64(seed)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _merge_setups(
    recs: list[dict[str, Any]],
    keeps: dict[int, np.ndarray],
    spacing: float,
    model: ScannerModel,
    patch_seed: int,
    stats: dict[str, Any],
) -> dict[int, np.ndarray]:
    """Cross-setup voxel ownership: in a voxel seen by several setups only the
    setup of the best-incidence point keeps its points, unless the voxel lies in
    a duplicate-skin patch (cells of ``duplicate_skin_patch_mm`` in the raw
    frame, each doubled with probability ``duplicate_skin_fraction`` through a
    position hash), where every setup keeps them."""
    order = sorted(keeps)
    sel = {i: np.flatnonzero(keeps[i]) for i in order}
    ideal = np.vstack([recs[i]["ideal"][sel[i]] for i in order])
    if len(ideal) == 0:
        return keeps
    cos = np.concatenate([recs[i]["cos"][sel[i]] for i in order])
    setup = np.concatenate([np.full(len(sel[i]), recs[i]["setup"], dtype=np.int64) for i in order])
    lin = _linear_keys(np.floor(ideal / spacing).astype(np.int64))
    best = np.lexsort((np.arange(len(lin)), -cos, lin))
    first = np.ones(len(best), dtype=bool)
    first[1:] = lin[best][1:] != lin[best][:-1]
    _, inverse = np.unique(lin, return_inverse=True)
    owner_setup = np.empty(int(inverse.max()) + 1, dtype=np.int64)
    owner_setup[inverse[best[first]]] = setup[best[first]]
    keep = setup == owner_setup[inverse]
    n_owner = int(keep.sum())
    if model.duplicate_skin_fraction > 0.0:
        cells = np.floor(ideal / model.duplicate_skin_patch_mm).astype(np.int64)
        keep |= _cell_uniform(cells, patch_seed) < model.duplicate_skin_fraction
    stats["duplicate_skin_points"] += int(keep.sum()) - n_owner
    stats["cross_setup_dropped"] += int(len(keep) - keep.sum())
    out: dict[int, np.ndarray] = {}
    start = 0
    for i in order:
        n = len(sel[i])
        mask = np.zeros(len(keeps[i]), dtype=bool)
        mask[sel[i][keep[start:start + n]]] = True
        out[i] = mask
        start += n
    return out


def _ownership(recs: list[dict[str, Any]], spacing: float, one_per_voxel: bool) -> list[np.ndarray]:
    """Voxel ownership at ``spacing``: each voxel belongs to the view of its
    best-incidence point (largest cos; ties by input order). ``one_per_voxel``
    keeps only that best point (uniform point density); otherwise every point of
    the owner view in the voxel survives (range-grid meshes stay connected)."""
    all_ideal = np.vstack([r["ideal"] for r in recs])
    all_cos = np.concatenate([r["cos"] for r in recs])
    owner = np.concatenate([np.full(len(r["ideal"]), i) for i, r in enumerate(recs)])
    lin = _linear_keys(np.floor(all_ideal / spacing).astype(np.int64))
    order = np.lexsort((np.arange(len(lin)), -all_cos, lin))
    first = np.ones(len(order), dtype=bool)
    first[1:] = lin[order][1:] != lin[order][:-1]
    winners = order[first]
    if one_per_voxel:
        keep = np.zeros(len(lin), dtype=bool)
        keep[winners] = True
    else:
        _, inverse = np.unique(lin, return_inverse=True)
        voxel_owner = np.empty(inverse.max() + 1, dtype=np.int64)
        voxel_owner[inverse[winners]] = owner[winners]
        keep = owner == voxel_owner[inverse]
    out = []
    start = 0
    for r in recs:
        n = len(r["ideal"])
        out.append(keep[start:start + n])
        start += n
    return out


def _grid_faces(rec: dict[str, Any], keep: np.ndarray, local: np.ndarray, jump: float) -> np.ndarray:
    """Triangles between kept, depth-continuous neighbouring pixels of one view."""
    h, w = rec["shape"]
    grid = np.full(h * w, -1, dtype=np.int64)
    grid[rec["grid_index"]] = local
    depth = np.full(h * w, np.nan)
    depth[rec["grid_index"]] = rec["depth"]
    grid = grid.reshape(h, w)
    depth = depth.reshape(h, w)
    a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]
    da, db, dc, dd = depth[:-1, :-1], depth[:-1, 1:], depth[1:, :-1], depth[1:, 1:]
    faces = []
    # NaN (missing pixel) propagates through maximum/minimum and fails the test
    m1 = (a >= 0) & (b >= 0) & (c >= 0) & ((np.maximum(np.maximum(da, db), dc) - np.minimum(np.minimum(da, db), dc)) < jump)
    m2 = (b >= 0) & (d >= 0) & (c >= 0) & ((np.maximum(np.maximum(db, dd), dc) - np.minimum(np.minimum(db, dd), dc)) < jump)
    # (x right, y up) is counter-clockwise seen from the camera: normals face it
    if m1.any():
        faces.append(np.stack([a[m1], b[m1], c[m1]], axis=1))
    if m2.any():
        faces.append(np.stack([b[m2], d[m2], c[m2]], axis=1))
    if not faces:
        return np.zeros((0, 3), dtype=np.int64)
    return np.vstack(faces)
