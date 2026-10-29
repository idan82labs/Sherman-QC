"""Gold raw-to-reference transform recovery (plan §10 amendment 6a), done before any scoring.

Amendment 6a: "Before scoring, E4 recovers each Gold case's raw-to-reference
transform from the time the labels were drawn: either the stored alignment, or
the aligned scan matched vertex-for-vertex to the raw scan, with residual
<= 0.01 mm. Scoring happens in the raw-scan frame. Cases with no recoverable
transform are listed and excluded before any scoring."

**Which cases.** Gold11 is the 11-scan, 71-line set of the historical line-F1
0.458 (``docs/research/2026-09-09_gold11_historical_comparison_binding.md``).
:data:`GOLD11` pins the 11 records. Nine are label jobs the Gold script's
``CASE_JOBS`` locates (``output/<job>/bend_instance_labels.json``, scans in
``../source-assets/02_CAD_QC_3D``). The two ``shaked_xv`` records (C05/C06,
38172000 X/V, Codex manual multiview, drawing-backed, labelled in the raw-scan
frame) are Gold11 line references the script does not locate; their scans are
CONSUMED benchmark scans under ``data/bend_corpus/incoming`` (owner directions
2026-10-17 authorize CONSUMED benchmark scans and the Gold11 dev cases + line
refs for wave 4; the first E4b pass, under plan §9 alone, had excluded them as
e-mail attachments). They are tagged ``AUTHORIZED_GOLD11_LINE_REF`` and flagged
``GOLD11_LINES_ARE_CODEX_MULTIVIEW_NOT_HUMAN``. :data:`HUMAN_NON_GOLD11` pins
the two other script-located human label jobs whose scans are BROAD cases with
STEP CAD (04_15651004, 09_47959001); they are BROAD gold, never G2(a) gold.

**Recovery, per case** (:func:`recover_case`), in order; the first method whose
residual is <= 0.01 mm wins and every attempt is recorded:

1. ``SHA256_BOUND_SCAN_RAW``: the labels declare ``scan_raw`` and the raw scan's
   sha256 equals the sha256 the labels are bound to (identity transform);
2. ``STORED_ALIGNMENT``: a 4x4 ``raw_scan_to_reference_transform`` stored with
   the label job; residual = max distance from every display-cloud point to the
   transformed raw vertices;
3. ``DISPLAY_CLOUD_IS_RAW_VERTEX_SUBSET_IDENTITY``: the job's display cloud
   (``aligned_scan.ply``, the points the labeller saw) under the identity:
   residual = max nearest-raw-vertex distance of every display point;
4. ``SAME_ORDER_KABSCH``: the display cloud has exactly the raw vertex count;
   the rigid fit of raw vertex *i* to display point *i*; residual = max error.

**Reuse where valid.** A prior recovery (the first E4b pass's
``gold_transforms.json`` rows, the broad baseline's ``gold11_frame_recovery.json``
and ``broad_frame_records.json``) is reused instead of re-reading the raw scan
only when the scan sha256, the display-cloud sha256 and, where the prior
recorded one, the labels sha256 all equal today's files, and its method,
residual and transform pass the same rules (``reused_from`` names the source and
its sha256). Otherwise the case is recovered again from the files.

A raw-scan sha256 that differs from the labels' (or the pinned one) is
``SCAN_SHA_MISMATCH``; a missing display cloud with no other method is
``NO_RECOVERABLE_TRANSFORM``; a best residual above the limit is
``TRANSFORM_RESIDUAL_TOO_LARGE``. Excluded cases stay listed.

**Gold11 audit relabel** (:func:`relabel_snapshot`): the parallel AI relabel
``ai_bend_lines_gold11_relabel_v1`` (another workflow; human Gold11 is never
overwritten, replacing it for a gate needs the owner's sign-off) is recorded as
it stands at the freeze (file sha256, bound scan sha256, line count), mapped to
the Gold11 cases by the provenance ``asset_sha256``. G2(a) is reported against
**both** the human Gold11 and the relabel; the human set stays the gate of record.

**Output.** :func:`recover_all` returns the transform list and one
``GoldCase``-shaped dict per recovered case (:class:`..line_metric.GoldCase`
fields; ``frame`` ``reference`` with the recovered ``raw_to_reference`` and
residual, or ``raw_scan``), keyed by the opaque physical case reference of its
scan. Per-line nearest-vertex distances are diagnostics only, never an
exclusion rule. The full list (every Gold case, written before any scoring) is
``gold_transforms_devcheck_SEALED.json``: once the DEV-CHECK draw is made it
names DEV-CHECK Gold cases, their scans and label files, so A1 never reads it.
:func:`public_gold_transforms` derives the A1 part ``gold_transforms.json``
from it after the draw: the rows of A1 families, every count, the excluded list
and the sha256 of the full list (so "listed before any scoring" stays checkable
from the A1 side), and the relabel snapshot restricted to the A1 cases.

**A1-only relabel reading** (``only_files``): at A1 scoring time the dual report
reads only the relabel files of the A1 cases (listed in ``cases_A1.json``); the
other files of the relabel directory are counted by name, never opened.
"""
from __future__ import annotations

import ast
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..thresholds import G2_LINE_THRESHOLDS
from .common import (
    GOLD11_XV_LABELS,
    RELABEL_DIR,
    REPO_ROOT,
    SOURCE_ASSETS,
    case_ref_for,
    custody_check,
    now_utc,
    read_json,
    read_scan,
    rel_to_repo,
    sha256_file,
    sha256_hex,
)

__all__ = [
    "GOLD_TRANSFORMS_SCHEMA",
    "GoldRecord",
    "GOLD11",
    "HUMAN_NON_GOLD11",
    "ALL_RECORDS",
    "script_case_jobs",
    "kabsch",
    "recover_case",
    "recover_all",
    "prior_records",
    "relabel_snapshot",
    "relabel_gold_cases",
    "public_gold_transforms",
    "relabel_file_split",
]

GOLD_TRANSFORMS_SCHEMA = "sherman_e4b_gold_transforms_v2"
SCRIPT_REL = "scripts/evaluate_f1_against_manual_gold.py"
RESIDUAL_LIMIT_MM = float(G2_LINE_THRESHOLDS["transform_residual_max_mm"])
_METHODS = ("SHA256_BOUND_SCAN_RAW", "STORED_ALIGNMENT", "DISPLAY_CLOUD_IS_RAW_VERTEX_SUBSET_IDENTITY",
            "SAME_ORDER_KABSCH")


@dataclass(frozen=True)
class GoldRecord:
    """One pinned Gold record (``docs/research/2026-09-09_gold11_historical_comparison_binding.md``)."""

    gold11_id: str | None
    catalog_case_id: str
    family: str
    label_job: str | None  # output/<job>/ (script-located), None otherwise
    scan_rel: str | None  # relative to ../source-assets/02_CAD_QC_3D, None otherwise
    scan_sha256: str
    declared_frame: str
    gold_line_count: int
    custody: str  # AUTHORIZED_SCRIPT_LOCATED | AUTHORIZED_GOLD11_LINE_REF | CUSTODY_EMAIL_ATTACHMENT_NOT_AUTHORIZED
    labels_rel: str | None = None  # repo-relative label file of an AUTHORIZED_GOLD11_LINE_REF record
    scan_path: str | None = None  # repo-relative scan of an AUTHORIZED_GOLD11_LINE_REF record
    tier: str = "GOLD11"  # GOLD11 | HUMAN_NON_GOLD11
    flags: tuple[str, ...] = ()


_XV_SCANS = "data/bend_corpus/incoming/2026-07-29_shaked_defective_good_pairs/38172000_01/scans/"
GOLD11: tuple[GoldRecord, ...] = (
    GoldRecord("CG84C8FA12D2", "corpus/01_10839000", "10839000", "bend_label_01_10839000",
               "10839000_01_full_scan.ply", "71cf0b216f43284e401c891091cd56c10321aa3eebccde204f8d206683999215",
               "reference_mesh_raw", 8, "AUTHORIZED_SCRIPT_LOCATED"),
    GoldRecord("CGE4B0549E41", "corpus/02_10839000", "10839000", "bend_label_02_10839000",
               "10839000/10839000_01_scan_1.ply", "08686dc1d5d4c89725393907082e8cc4e4c0f5a21427697760ff44d29c7939fc",
               "reference_mesh_raw", 4, "AUTHORIZED_SCRIPT_LOCATED"),
    GoldRecord("CG08506AE97D", "corpus/03_10839000", "10839000", "bend_label_03_10839000",
               "10839000/10839000_01_scan_2.ply", "cec08dfca9ab309ff3eef42cd3dc72eb33168e1d43029f3a0658f2923e4e171b",
               "reference_mesh_raw", 6, "AUTHORIZED_SCRIPT_LOCATED"),
    GoldRecord("CG4BDD1371B6", "corpus/05_37361005", "37361005", "bend_label_05_37361005",
               "37361005_01.ply", "d1732de180230bdbf0caa6eaf454a33071c206060aea276dce67a9abcf13b038",
               "reference_mesh_raw", 4, "AUTHORIZED_SCRIPT_LOCATED"),
    GoldRecord("CG65CD575B1C", "corpus/06_38172000", "38172000", "bend_label_06_38172000",
               "38172000_01_SCAN_1.ply", "07062eee17aa4e41721b93cdb69ddb4b752e6713a16c5aab0a10df9c86d07880",
               "reference_mesh_raw", 6, "AUTHORIZED_SCRIPT_LOCATED"),
    GoldRecord("CGEBA9A869AC", "corpus/07_38172000", "38172000", "bend_label_07_38172000",
               "38172000_01_SCAN_2 1.ply", "f2475286312b7e11be773166afae5d7eaacd826f8b59cee84289d05b4a222b63",
               "reference_mesh_raw", 8, "AUTHORIZED_SCRIPT_LOCATED"),
    GoldRecord("CG64C412CC73", "corpus/08_40500000", "40500000", "bend_label_08_40500000",
               "40500000_0_SCAN2 1.ply", "152f1b1aacb20920a601f7a8197d4c89a111688325bcf557a6a731641f306614",
               "reference_mesh_raw", 7, "AUTHORIZED_SCRIPT_LOCATED"),
    GoldRecord("CG191DEC7F81", "corpus/10_49001000", "49001000", "bend_label_10_49001000",
               "49001000_A-PRELIM.ply", "7a1773eca9fab616afef498a9f9831e67eacb17fcc0d170607490a9e9db37020",
               "reference_mesh_raw", 5, "AUTHORIZED_SCRIPT_LOCATED"),
    GoldRecord("CG4FAF273093", "corpus/11_49024000", "49024000", "bend_label_11_49024000",
               "49024000_A.ply", "70b39f530f00a72dd99acf00b4a065f45bfe4f50185e3579c3ba26143f7d2ade",
               "reference_mesh_raw", 3, "AUTHORIZED_SCRIPT_LOCATED"),
    GoldRecord("CG1E0BBA2933", "shaked_xv/C05", "38172000", None, None,
               "4f8188b4d1a844305134801c3bb5364270250416d2f4dc43f0b05d37cc7962b9", "scan_raw", 10,
               "AUTHORIZED_GOLD11_LINE_REF", labels_rel=GOLD11_XV_LABELS[0], scan_path=_XV_SCANS + "38172000_01_X.ply",
               flags=("GOLD11_LINES_ARE_CODEX_MULTIVIEW_NOT_HUMAN",)),
    GoldRecord("CGF41F49066A", "shaked_xv/C06", "38172000", None, None,
               "2714fb7e0c0b681196da62bf368e60c0635b97000f34df3241ca1b70ee56faba", "scan_raw", 10,
               "AUTHORIZED_GOLD11_LINE_REF", labels_rel=GOLD11_XV_LABELS[1], scan_path=_XV_SCANS + "38172000_01_V.ply",
               flags=("GOLD11_LINES_ARE_CODEX_MULTIVIEW_NOT_HUMAN",)),
)
#: Script-located human label jobs outside Gold11 whose scans are BROAD cases with STEP CAD.
HUMAN_NON_GOLD11: tuple[GoldRecord, ...] = (
    GoldRecord(None, "corpus/04_15651004", "15651004", "bend_label_04_15651004", "15651004_01_SECOND_SCAN.stl",
               "8dfdab44605dfc1b4ee03a5bb72b082ea674e8027d6376bcb6a2cbc33ccbae9b", "reference_mesh_raw", 1,
               "AUTHORIZED_SCRIPT_LOCATED", tier="HUMAN_NON_GOLD11"),
    GoldRecord(None, "corpus/09_47959001", "47959001", "bend_label_09_47959001", "47959001_01PREL_scan1.stl",
               "a0305a6cf78729b2ea4be5505139e43f96667850e2ff8e02e99964ea03b7bd1e", "reference_mesh_raw", 5,
               "AUTHORIZED_SCRIPT_LOCATED", tier="HUMAN_NON_GOLD11"),
)
ALL_RECORDS: tuple[GoldRecord, ...] = GOLD11 + HUMAN_NON_GOLD11


def script_case_jobs(repo_root: str | os.PathLike = REPO_ROOT) -> dict[str, str]:
    """``CASE_JOBS`` of the Gold script, read with :func:`ast.literal_eval` (the script is never imported)."""
    src = Path(repo_root) / SCRIPT_REL
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CASE_JOBS" for t in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict):
                return {str(k): str(v) for k, v in value.items()}
    raise ValueError(f"{SCRIPT_REL}: CASE_JOBS not found")


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Proper rigid 4x4 transform minimising ``|T src - dst|`` over paired points."""
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = cb - r @ ca
    return t


def _apply(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ t[:3, :3].T + t[:3, 3]


def _is_rigid(t: np.ndarray, tol: float = 2e-5) -> bool:
    if t.shape != (4, 4) or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-9) or not np.isfinite(t).all():
        return False
    r = t[:3, :3]
    return bool(np.allclose(r.T @ r, np.eye(3), atol=tol) and np.linalg.det(r) > 0)


def _nn_residual(display: np.ndarray, raw: np.ndarray) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    tree = cKDTree(raw)
    dist, idx = tree.query(display, k=1)
    return {"max_mm": float(dist.max()) if len(dist) else math.inf,
            "p99_mm": float(np.percentile(dist, 99)) if len(dist) else math.inf,
            "unique_raw_vertices_hit": int(len(np.unique(idx))), "display_points": int(len(display))}


def _stored_alignment(job_dir: Path) -> np.ndarray | None:
    """A 4x4 raw-to-reference transform stored with the label job, if any."""
    for name in ("bend_inspection_report.json", "bend_inspection_meta.json"):
        p = job_dir / name
        if not p.is_file():
            continue
        doc = read_json(custody_check(p, "gold label job"))
        for holder in (doc.get("alignment"), (doc.get("seed_case") or {}).get("alignment"), doc):
            if isinstance(holder, dict) and holder.get("raw_scan_to_reference_transform") is not None:
                t = np.asarray(holder["raw_scan_to_reference_transform"], dtype=np.float64)
                if t.shape == (4, 4):
                    return t
    return None


def _line_diagnostics(lines: list[dict[str, Any]], raw: np.ndarray, t_raw_to_ref: np.ndarray) -> list[dict[str, Any]]:
    """Median nearest-raw-vertex distance of 41 samples per gold line (diagnostic only)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(_apply(t_raw_to_ref, raw))
    out = []
    for line in lines:
        a, b = np.asarray(line["line_start"], float), np.asarray(line["line_end"], float)
        samples = a[None, :] + np.linspace(0.0, 1.0, 41)[:, None] * (b - a)[None, :]
        d, _ = tree.query(samples, k=1)
        out.append({"bend_id": line["bend_id"], "length_mm": round(float(np.linalg.norm(b - a)), 3),
                    "median_nearest_vertex_mm": round(float(np.median(d)), 4),
                    "max_nearest_vertex_mm": round(float(d.max()), 4)})
    return out


def _lines_of(labels: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"bend_id": str(b["bend_id"]), "line_start": [float(v) for v in b["line_start"]],
             "line_end": [float(v) for v in b["line_end"]]} for b in labels.get("manual_bends") or []]


def _valid_prior(prior: Mapping[str, Any] | None, *, scan_sha: str, display_sha: str | None, labels_sha: str,
                 limit_mm: float) -> bool:
    if not prior or prior.get("status") != "RECOVERED" or prior.get("scan_sha256") != scan_sha:
        return False
    if prior.get("labels_sha256") is not None and prior["labels_sha256"] != labels_sha:
        return False
    method = prior.get("method")
    if method not in _METHODS:
        return False
    if method != "SHA256_BOUND_SCAN_RAW" and (display_sha is None or prior.get("display_sha256") != display_sha):
        return False
    try:
        t = np.asarray(prior["raw_to_reference"], dtype=np.float64)
        residual = float(prior["residual_max_mm"])
    except (KeyError, TypeError, ValueError):
        return False
    return residual <= limit_mm and _is_rigid(t)


def _finish(out: dict[str, Any], rec: GoldRecord, lines: list[dict[str, Any]], method: str, t_ref: np.ndarray,
            residual: float, scan_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    out.update({"status": "RECOVERED", "reason": None, "method": method, "residual_max_mm": residual,
                "raw_to_reference": t_ref.tolist()})
    gold_case = {"case_id": out["case_ref"], "gold11_id": rec.gold11_id, "catalog_case_id": rec.catalog_case_id,
                 "family": rec.family, "tier": rec.tier, "flags": list(rec.flags),
                 "frame": "raw_scan" if method == "SHA256_BOUND_SCAN_RAW" else "reference",
                 "gold_lines": lines, "raw_to_reference": None if method == "SHA256_BOUND_SCAN_RAW" else t_ref.tolist(),
                 "residual_mm": residual, "transform_method": method, "labels_sha256": out["labels_sha256"],
                 "scan_sha256": scan_sha}
    return out, gold_case


def recover_case(rec: GoldRecord, *, repo_root: str | os.PathLike = REPO_ROOT,
                 source_root: str | os.PathLike = SOURCE_ASSETS, limit_mm: float = RESIDUAL_LIMIT_MM,
                 case_jobs: dict[str, str] | None = None, prior: Mapping[str, Any] | None = None
                 ) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Recover one Gold case (module docstring). Returns ``(record, gold_case | None)``.

    ``prior`` is a previous recovery of the same case (:func:`prior_records`); it is reused only
    when it is still valid for today's files."""
    out: dict[str, Any] = {"gold11_id": rec.gold11_id, "catalog_case_id": rec.catalog_case_id, "family": rec.family,
                           "tier": rec.tier, "flags": list(rec.flags),
                           "declared_frame": rec.declared_frame, "pinned_scan_sha256": rec.scan_sha256,
                           "pinned_gold_line_count": rec.gold_line_count, "custody": rec.custody,
                           "attempts": [], "status": "EXCLUDED", "reason": None}
    repo = Path(repo_root)
    if rec.custody == "AUTHORIZED_GOLD11_LINE_REF" and rec.labels_rel and rec.scan_path:
        labels_path = custody_check(repo / rec.labels_rel, "gold labels")
        labels = read_json(labels_path)
        lines = _lines_of(labels)
        scan_path = Path(rec.scan_path) if Path(rec.scan_path).is_absolute() else repo / rec.scan_path
        scan_sha = sha256_file(custody_check(scan_path, "gold raw scan"))
        out.update({"labels_rel": rel_to_repo(labels_path), "labels_sha256": sha256_file(labels_path),
                    "label_frame": labels.get("coordinate_frame"), "label_review_status": labels.get("review_status"),
                    "label_kind": labels.get("kind"), "gold_line_count": len(lines),
                    "scan_rel": rel_to_repo(scan_path), "scan_sha256": scan_sha, "case_ref": case_ref_for(scan_sha),
                    "label_bound_scan_sha256": labels.get("scan_sha256"),
                    "authoritative_ground_truth": labels.get("authoritative_ground_truth")})
        if labels.get("coordinate_frame") != rec.declared_frame:
            out["reason"] = "LABEL_FRAME_DECLARATION_MISMATCH"
            return out, None
        if scan_sha != rec.scan_sha256 or labels.get("scan_sha256") not in (None, scan_sha):
            out["reason"] = "SCAN_SHA_MISMATCH"
            return out, None
        if len(lines) != rec.gold_line_count:
            out["reason"] = "GOLD_LINE_COUNT_DIFFERS_FROM_PINNED"
            return out, None
        if rec.declared_frame != "scan_raw" or labels.get("scan_sha256") != scan_sha:
            out["reason"] = "NO_RECOVERABLE_TRANSFORM"
            return out, None
        out["attempts"].append({"method": "SHA256_BOUND_SCAN_RAW", "residual_max_mm": 0.0, "bound_sha256": scan_sha})
        return _finish(out, rec, lines, "SHA256_BOUND_SCAN_RAW", np.eye(4), 0.0, scan_sha)
    if rec.custody != "AUTHORIZED_SCRIPT_LOCATED" or rec.label_job is None or rec.scan_rel is None:
        out["reason"] = rec.custody if rec.custody != "AUTHORIZED_SCRIPT_LOCATED" else "NOT_SCRIPT_LOCATED"
        out["note"] = "listed and excluded before scoring; neither the scan nor the labels were read"
        return out, None
    jobs = case_jobs if case_jobs is not None else script_case_jobs(repo_root)
    if rec.label_job not in jobs.values():
        out["reason"] = "NOT_SCRIPT_LOCATED"
        return out, None
    out["script_case_id"] = next(k for k, v in jobs.items() if v == rec.label_job)
    job_dir = repo / "output" / rec.label_job
    labels_path = custody_check(job_dir / "bend_instance_labels.json", "gold labels")
    meta_path = job_dir / "bend_inspection_meta.json"
    labels = read_json(labels_path)
    meta = read_json(custody_check(meta_path, "gold label job")) if meta_path.is_file() else {}
    seed = meta.get("seed_case") or {}
    lines = _lines_of(labels)
    out.update({
        "label_job": rec.label_job, "labels_rel": rel_to_repo(labels_path), "labels_sha256": sha256_file(labels_path),
        "label_frame": labels.get("coordinate_frame"), "label_review_status": labels.get("review_status"),
        "label_job_frame": seed.get("coordinate_frame"),
        "label_job_frame_provenance": seed.get("coordinate_frame_provenance"),
        "labeling_mode": seed.get("labeling_mode"), "gold_line_count": len(lines),
    })
    scan_path = Path(source_root) / rec.scan_rel
    scan_sha = sha256_file(custody_check(scan_path, "gold raw scan"))
    out.update({"scan_rel": rel_to_repo(scan_path), "scan_sha256": scan_sha, "case_ref": case_ref_for(scan_sha),
                "label_job_source_sha256": seed.get("source_sha256")})
    if labels.get("coordinate_frame") != rec.declared_frame or (seed and seed.get("coordinate_frame") not in
                                                                 (None, rec.declared_frame)):
        out["reason"] = "LABEL_FRAME_DECLARATION_MISMATCH"
        return out, None
    if scan_sha != rec.scan_sha256 or (seed.get("source_sha256") not in (None, scan_sha)):
        out["reason"] = "SCAN_SHA_MISMATCH"
        return out, None
    if len(lines) != rec.gold_line_count:
        out["reason"] = "GOLD_LINE_COUNT_DIFFERS_FROM_PINNED"
        return out, None
    display_path = job_dir / "aligned_scan.ply"
    display_sha = None
    if display_path.is_file():
        display_sha = sha256_file(custody_check(display_path, "gold display cloud"))
        out["display_cloud"] = {"rel": rel_to_repo(display_path), "sha256": display_sha}
    if _valid_prior(prior, scan_sha=scan_sha, display_sha=display_sha, labels_sha=out["labels_sha256"],
                    limit_mm=limit_mm):
        t_ref = np.asarray(prior["raw_to_reference"], dtype=np.float64)
        out["attempts"].append({"method": prior["method"], "residual_max_mm": float(prior["residual_max_mm"]),
                                "reused": True})
        out["reused_from"] = dict(prior.get("source") or {})
        out["provenance_consistent"] = (prior["method"] != "DISPLAY_CLOUD_IS_RAW_VERTEX_SUBSET_IDENTITY"
                                        or seed.get("coordinate_frame_provenance") == "unaligned_source_scan_raw")
        if prior.get("line_diagnostics"):
            out["line_diagnostics"] = prior["line_diagnostics"]
        return _finish(out, rec, lines, prior["method"], t_ref, float(prior["residual_max_mm"]), scan_sha)
    raw = read_scan(scan_path, with_faces=False, purpose="gold raw scan").points
    out["raw_vertex_count"] = int(len(raw))
    best: tuple[str, np.ndarray, float] | None = None

    def consider(method: str, t: np.ndarray, residual: float, detail: dict[str, Any]) -> None:
        nonlocal best
        out["attempts"].append({"method": method, "residual_max_mm": residual, **detail})
        if residual <= limit_mm and best is None and _is_rigid(t):
            best = (method, t, residual)

    if rec.declared_frame == "scan_raw":
        consider("SHA256_BOUND_SCAN_RAW", np.eye(4), 0.0, {"bound_sha256": seed.get("source_sha256") or rec.scan_sha256})
    stored = _stored_alignment(job_dir)
    display = None
    if display_path.is_file():
        display = read_scan(display_path, with_faces=False, purpose="gold display cloud").points
        out["display_cloud"]["points"] = int(len(display))
    if stored is not None and display is not None:
        res = _nn_residual(display, _apply(stored, raw))
        consider("STORED_ALIGNMENT", stored, res["max_mm"], res)
    if display is not None:
        res = _nn_residual(display, raw)
        consider("DISPLAY_CLOUD_IS_RAW_VERTEX_SUBSET_IDENTITY", np.eye(4), res["max_mm"], res)
        if len(display) == len(raw):
            t = kabsch(raw, display)
            err = np.linalg.norm(_apply(t, raw) - display, axis=1)
            consider("SAME_ORDER_KABSCH", t, float(err.max()), {"points": int(len(raw))})
    if best is None:
        out["reason"] = ("NO_RECOVERABLE_TRANSFORM" if not out["attempts"] else "TRANSFORM_RESIDUAL_TOO_LARGE")
        return out, None
    method, t_ref, residual = best
    out["provenance_consistent"] = (method != "DISPLAY_CLOUD_IS_RAW_VERTEX_SUBSET_IDENTITY"
                                    or seed.get("coordinate_frame_provenance") == "unaligned_source_scan_raw")
    out["line_diagnostics"] = _line_diagnostics(lines, raw, t_ref)
    return _finish(out, rec, lines, method, t_ref, residual, scan_sha)


def prior_records(sources: Sequence[Path], *, own_roots: Sequence[Path] = ()) -> dict[str, dict[str, Any]]:
    """``catalog_case_id -> prior recovery`` from earlier transform lists (first source wins).

    Accepted shapes: the first E4b pass's ``gold_transforms.json`` (``cases`` rows with
    ``labels_sha256``, ``display_cloud.sha256``) and the broad baseline's frame records
    (``cases`` rows with ``display_cloud.sha256``, no labels sha256). A source under one of
    ``own_roots`` (this chunk's work directory, holding its own earlier run log) is read
    directly; every other source passes the custody guard."""
    out: dict[str, dict[str, Any]] = {}
    for src in sources:
        p = Path(src)
        if not p.is_file():
            continue
        own = any(p.resolve().is_relative_to(Path(r).resolve()) for r in own_roots)
        doc = read_json(p if own else custody_check(p, "prior gold transform list"))
        file_sha = sha256_file(p)
        for row in doc.get("cases") or []:
            cid = row.get("catalog_case_id")
            if not cid or cid in out or row.get("status") != "RECOVERED":
                continue
            out[cid] = {"status": "RECOVERED", "scan_sha256": row.get("scan_sha256"),
                        "labels_sha256": row.get("labels_sha256"),
                        "display_sha256": (row.get("display_cloud") or {}).get("sha256"),
                        "method": row.get("method"), "residual_max_mm": row.get("residual_max_mm"),
                        "raw_to_reference": row.get("raw_to_reference"),
                        "line_diagnostics": row.get("line_diagnostics"),
                        "source": {"path": rel_to_repo(p), "sha256": file_sha, "schema": doc.get("schema")}}
    return out


def recover_all(records: tuple[GoldRecord, ...] = ALL_RECORDS, *, repo_root: str | os.PathLike = REPO_ROOT,
                source_root: str | os.PathLike = SOURCE_ASSETS, limit_mm: float = RESIDUAL_LIMIT_MM,
                priors: Mapping[str, Mapping[str, Any]] | None = None
                ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Every pinned record, one scan at a time. Returns ``(gold_transforms doc, gold cases)``."""
    jobs = script_case_jobs(repo_root)
    rows, cases = [], []
    for rec in records:
        row, case = recover_case(rec, repo_root=repo_root, source_root=source_root, limit_mm=limit_mm, case_jobs=jobs,
                                 prior=(priors or {}).get(rec.catalog_case_id))
        rows.append(row)
        if case is not None:
            cases.append(case)
    g11 = [c for c in cases if c["tier"] == "GOLD11"]
    doc = {
        "schema": GOLD_TRANSFORMS_SCHEMA,
        "created_at": now_utc(),
        "rule": "plan §10 amendment 6a: recovered before any scoring; residual <= "
                f"{limit_mm} mm; excluded cases listed; scoring in the raw-scan frame",
        "residual_limit_mm": limit_mm,
        "locator": SCRIPT_REL + " CASE_JOBS (AST-read, never imported) + the pinned Gold11 records "
                   "(C05/C06 label files named by the broad baseline's gold11_frame_recovery.json)",
        "gold11_records": sum(1 for r in records if r.tier == "GOLD11"),
        "recovered": len(g11),
        "excluded": [{"gold11_id": r["gold11_id"], "catalog_case_id": r["catalog_case_id"], "reason": r["reason"]}
                     for r in rows if r["status"] != "RECOVERED"],
        "gold_lines_recovered": sum(len(c["gold_lines"]) for c in g11),
        "human_non_gold11_recovered": sum(1 for c in cases if c["tier"] == "HUMAN_NON_GOLD11"),
        "reused_priors": sum(1 for r in rows if r.get("reused_from")),
        "cases": rows,
        "scored_before_this_list": False,
    }
    return doc, cases


def relabel_snapshot(gold_cases: Sequence[Mapping[str, Any]], relabel_dir: Path = RELABEL_DIR, *,
                     only_files: Sequence[str] | None = None) -> dict[str, Any]:
    """The Gold11 audit relabel as it stands now (module docstring); nothing is scored here.

    ``only_files`` (file names or repo-relative paths) restricts the reading to those files: the other
    ``*.json`` names of the directory are only counted (``other_files``), never opened."""
    by_scan = {c["scan_sha256"]: c for c in gold_cases if c.get("tier") == "GOLD11"}
    rows: list[dict[str, Any]] = []
    other_files = None
    missing_files: list[str] = []
    if only_files is not None:
        wanted = sorted({Path(str(f)).name for f in only_files})
        present = {p.name for p in relabel_dir.glob("*.json")} if relabel_dir.is_dir() else set()
        other_files = len(present - set(wanted))
        missing_files = [n for n in wanted if n not in present]
        paths = [relabel_dir / n for n in wanted if n in present]
    else:
        paths = sorted(relabel_dir.glob("*.json")) if relabel_dir.is_dir() else []
    if paths:
        for p in paths:
            lab = read_json(custody_check(p, "gold11 relabel"))
            prov_p = relabel_dir / "provenance" / p.name
            prov = read_json(custody_check(prov_p, "gold11 relabel provenance")) if prov_p.is_file() else {}
            sha = prov.get("asset_sha256")
            g = by_scan.get(sha)
            rows.append({"file": rel_to_repo(p), "file_sha256": sha256_file(p),
                         "provenance_sha256": sha256_file(prov_p) if prov_p.is_file() else None,
                         "asset_id": lab.get("asset_id"), "asset_sha256": sha,
                         "case_ref": None if g is None else g["case_id"],
                         "gold11_id": None if g is None else g.get("gold11_id"),
                         "coordinate_frame": lab.get("coordinate_frame"), "annotator": lab.get("annotator"),
                         "review_status": lab.get("review_status"), "revision": lab.get("revision"),
                         "no_bends_present": bool(lab.get("no_bends_present")),
                         "lines": len(lab.get("bends") or []),
                         "usable": bool(g is not None and lab.get("coordinate_frame") == "source_ply_raw"
                                        and sha is not None)})
    from contracts.production.canonical import sha256_hex

    covered = sorted({r["case_ref"] for r in rows if r["usable"]})
    out = {"dir": rel_to_repo(relabel_dir), "present": bool(rows), "files": rows,
           "snapshot_sha256": sha256_hex([[r["file"], r["file_sha256"]] for r in rows]),
           "gold11_cases_covered": covered, "gold11_cases_total": len(by_scan),
           "status": ("COMPLETE_PENDING_OWNER_SIGNOFF" if len(covered) == len(by_scan) and by_scan else
                      "IN_PROGRESS" if rows else "ABSENT"),
           "rule": "reported next to human Gold11 (G2(a) dual report); human Gold11 stays the gate of record "
                   "until the owner signs off; never replaced silently"}
    if only_files is not None:
        out.update(read_scope="LISTED_FILES_ONLY", other_files=other_files, missing_files=missing_files)
    return out


def relabel_file_split(snapshot: Mapping[str, Any], public_case_ids: set[str]
                       ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(A1 files, other files)`` of a full relabel snapshot: a file mapped to an A1 Gold11 case is
    public; a file of a DEV-CHECK case or mapped to no Gold11 case is sealed (never listed publicly)."""
    keep = ("file", "file_sha256", "provenance_sha256", "case_ref")
    pub, other = [], []
    for r in snapshot.get("files") or []:
        (pub if r.get("case_ref") in public_case_ids else other).append({k: r.get(k) for k in keep})
    return pub, other


def public_gold_transforms(full: Mapping[str, Any], sealed_families: set[str], *, full_sha256: str,
                           full_path: str) -> dict[str, Any]:
    """The A1 part of the full amendment-6a transform list (module docstring).

    Rows of DEV-CHECK families are left out (counted); every count, the excluded rows of A1 families
    (an excluded DEV-CHECK row is only counted: its catalog id names the family) and the full list's
    sha256 stay, so the recovery stays 'listed before any scoring' from the A1 side. The relabel
    snapshot keeps only the files of A1 cases."""
    rows = [dict(r) for r in full.get("cases") or [] if r.get("family") not in sealed_families]
    out = {k: v for k, v in full.items() if k not in ("cases", "gold11_relabel_v1", "excluded", "audience")}
    pub_ids = {r["case_ref"] for r in rows if r.get("case_ref")}
    pub_catalog = {r.get("catalog_case_id") for r in rows}
    excluded = list(full.get("excluded") or [])
    out.update({
        "audience": "A1 (accuracy loop) and reviewers",
        "cases": rows,
        "cases_listed": len(rows),
        "cases_sealed": len(full.get("cases") or []) - len(rows),
        "excluded": [dict(e) for e in excluded if e.get("catalog_case_id") in pub_catalog],
        "excluded_sealed_count": sum(1 for e in excluded if e.get("catalog_case_id") not in pub_catalog),
        "full_list": {"path": full_path, "sha256": full_sha256, "created_at": full.get("created_at"),
                      "note": "the full list (every Gold case, written before any scoring) is DEV-CHECK material: "
                              "A1 never reads it; its sha256 is recorded here and in the ledger"},
        "split_rule": "rows of the DEV-CHECK families are sealed in the full list; counts and the excluded list "
                      "stay public (amendment 6a: every case listed before any scoring)",
    })
    rel = full.get("gold11_relabel_v1")
    if rel:
        pub_files, other = relabel_file_split(rel, pub_ids)
        out["gold11_relabel_v1"] = {
            **{k: v for k, v in rel.items() if k not in ("files", "gold11_cases_covered", "snapshot_sha256")},
            "files": pub_files, "files_sealed": len(other),
            "gold11_cases_covered_A1": sorted(c for c in rel.get("gold11_cases_covered") or [] if c in pub_ids),
            "snapshot_sha256_A1": sha256_hex([[r["file"], r["file_sha256"]] for r in pub_files])}
    return out


def relabel_gold_cases(gold_cases: Sequence[Mapping[str, Any]], relabel_dir: Path = RELABEL_DIR, *,
                       only_files: Sequence[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """GoldCase dicts (raw-scan frame) of the relabel for the Gold11 cases in ``gold_cases``, read now.

    Returns ``(cases, snapshot)``; a Gold11 case without a usable relabel file is absent (the report
    says which), so the dual report compares the two gold sets on the cases both cover. ``only_files``
    restricts the reading (:func:`relabel_snapshot`)."""
    snap = relabel_snapshot(gold_cases, relabel_dir, only_files=only_files)
    keep = {c["case_id"] for c in gold_cases}
    family = {c["case_id"]: c.get("family") for c in gold_cases}
    out = []
    for r in snap["files"]:
        if not r["usable"] or r["case_ref"] not in keep:
            continue
        lab = read_json(custody_check(REPO_ROOT / r["file"], "gold11 relabel"))
        lines = [{"bend_id": str(b.get("bend_id") or f"R{i:02d}"), "line_start": [float(v) for v in b["line_start"]],
                  "line_end": [float(v) for v in b["line_end"]]} for i, b in enumerate(lab.get("bends") or [], 1)]
        out.append({"case_id": r["case_ref"], "gold11_id": r["gold11_id"], "family": family.get(r["case_ref"]),
                    "tier": "GOLD11_RELABEL_V1",
                    "frame": "raw_scan", "gold_lines": lines, "raw_to_reference": None, "residual_mm": 0.0,
                    "transform_method": "SOURCE_PLY_RAW_IS_SCAN_RAW", "labels_sha256": r["file_sha256"],
                    "scan_sha256": r["asset_sha256"], "zero_bend_control": r["no_bends_present"] and not lines})
    return out, snap
