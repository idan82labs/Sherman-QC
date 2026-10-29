"""Shared constants and helpers of the E4b pre-A1 freeze tools (plan §9 custody, §10 amendments 4, 6-8).

- **Paths**: the repository root, the authorized source tree
  (``../source-assets/02_CAD_QC_3D``), the run-log directories E4b owns
  (``docs/production/runlog/{freeze,legacy_control,g3_truth_material}``) and the
  sealed-name convention (``*SEALED*``: material A1 must never read).
- **Custody guard** (:func:`custody_check`): every read of a real input goes
  through it. It admits only the inputs this chunk is authorized for (plan §9 as
  widened by the owner's 2026-10-17 directions for wave 4): the source-asset
  CAD/scans; the Gold11 development cases and their line references (the label
  jobs the Gold script's ``CASE_JOBS`` locates, plus the two ``shaked_xv``
  label files of Gold11 C05/C06); the STEP CAD and benchmark scans of parts
  marked CONSUMED in ``BENCHMARK_V2.json`` (exact listed paths only; a scan the
  manifest reserves as ``FRESH_SCAN_OF_CONSUMED_PART`` is not admitted); the AI
  gold of ``ai_bend_lines_v1`` and the Gold11 audit relabel
  ``ai_bend_lines_gold11_relabel_v1`` (label + provenance files); and the corpus
  ``metadata.json`` stage keys (scoring only). Everything else under ``data/``,
  ``output/``, ``outputs/``, ``artifacts/``, ``uploads/`` or ``live_scans/`` is
  refused, and so is any path naming a holdout, sealed truth, N001-N016, an
  EHO4 part, the FRESH codex fetch, another workflow's ``*SEALED*`` material,
  authorization/one-shot material or the rhino-preview face scans. A refused
  read raises :class:`CustodyError`; it is never retried with another path.
- **Files**: atomic JSON writing (sorted keys, UTF-8, trailing newline),
  canonical hashing (contracts ``canonical.py``), PLY and STL vertex reading
  (``read_scan``: E4a's PLY reader plus a binary/ASCII STL reader that merges
  the triangle soup's repeated corners).
- **Ids**: opaque case references derived from the scan sha256 (``pc_<12 hex>``),
  opaque DEV SKUs from the CAD sha256 (``DEV-<10 HEX>``), so no file name, part
  number or stage token reaches an engine or a reviewer image.
"""
from __future__ import annotations

import json
import os
import re
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from contracts.production.canonical import sha256_hex

from ..plyio import read_ply, sha256_file

__all__ = [
    "FREEZE_TOOLS_VERSION",
    "REPO_ROOT",
    "SOURCE_ASSETS",
    "RUNLOG",
    "FREEZE_DIR",
    "LEGACY_DIR",
    "G3_DIR",
    "SEALED_TAG",
    "WORK_SEALED_DIR",
    "WORK_BUILDER_DIR",
    "GOLD_TRANSFORMS_FULL_NAME",
    "GOLD_LABEL_JOBS",
    "HUMAN_NON_GOLD11_JOBS",
    "GOLD11_XV_LABELS",
    "BENCHMARK_MANIFEST",
    "BROAD_CASE_LIST",
    "AI_GOLD_DIR",
    "RELABEL_DIR",
    "EHO4_PARTS",
    "CustodyError",
    "custody_check",
    "consumed_allowlist",
    "expand_home",
    "is_sealed_path",
    "now_utc",
    "write_json",
    "read_json",
    "sha256_file",
    "sha256_hex",
    "family_of",
    "case_ref_for",
    "dev_sku_for",
    "ScanGeometry",
    "read_scan",
    "read_stl",
    "rel_to_repo",
    "stable_rows",
    "doc_sha256",
    "compact_png",
]

FREEZE_TOOLS_VERSION = "e4b-freeze-1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE_ASSETS = (REPO_ROOT.parent / "source-assets" / "02_CAD_QC_3D").resolve()
RUNLOG = REPO_ROOT / "docs" / "production" / "runlog"
FREEZE_DIR = RUNLOG / "freeze"
LEGACY_DIR = RUNLOG / "legacy_control"
G3_DIR = RUNLOG / "g3_truth_material"
#: Any path component containing this tag is DEV-CHECK material that A1 must never read.
SEALED_TAG = "SEALED"
#: Work-directory layout (``prefreeze``). An evaluation work directory ``W`` (A1's own, fresh) holds A1's
#: products directly (staged inputs, compiled parts, E3 outputs) and, for the final DEV-CHECK run only,
#: the DEV-CHECK audience's under ``W/devcheck_SEALED/``. The freeze builder's work area is
#: ``W/builder_SEALED/`` of the builder's own work directory: every product made before or across the
#: public/sealed split (full gold list, full compile report, recipe builds and compiles of every STEP,
#: the legacy worker's outputs, reviewer-blind placements, earlier passes, logs, dry runs). Both carry
#: :data:`SEALED_TAG`, so A1 never reads them; the A1-facing commands refuse a work directory holding
#: ``builder_SEALED/`` and, without ``--include-sealed``, one holding ``devcheck_SEALED/``.
WORK_SEALED_DIR = "devcheck_SEALED"
WORK_BUILDER_DIR = "builder_SEALED"
#: The full amendment-6a transform list (every Gold case, written before any scoring); it names DEV-CHECK
#: Gold cases once the draw is made, so it is sealed. ``gold_transforms.json`` is its A1 part.
GOLD_TRANSFORMS_FULL_NAME = "gold_transforms_devcheck_SEALED.json"

#: The Gold11 label jobs located through ``scripts/evaluate_f1_against_manual_gold.py``
#: (its ``CASE_JOBS``) that belong to Gold11.
GOLD_LABEL_JOBS: tuple[str, ...] = (
    "bend_label_01_10839000", "bend_label_02_10839000", "bend_label_03_10839000", "bend_label_05_37361005",
    "bend_label_06_38172000", "bend_label_07_38172000", "bend_label_08_40500000", "bend_label_10_49001000",
    "bend_label_11_49024000",
)
#: Script-located human label jobs outside Gold11 whose scans are BROAD cases with STEP CAD
#: (``case_list_broad_v3.json`` tier GOLD_LINES_HUMAN_NON_GOLD11). 12_49125000 and 13_ysherman
#: have no STEP, so they are never read here.
HUMAN_NON_GOLD11_JOBS: tuple[str, ...] = ("bend_label_04_15651004", "bend_label_09_47959001")
#: The two Gold11 line references that the Gold script does not locate (``shaked_xv`` C05/C06,
#: Codex manual multiview, drawing-backed, labelled in the raw-scan frame); the broad baseline's
#: ``gold11_frame_recovery.json`` names them.
GOLD11_XV_LABELS: tuple[str, ...] = (
    "output/research/f1-new-shaked-pairs-20260730-v1/manual_adjudication/labels/C05/bend_instance_labels.json",
    "output/research/f1-new-shaked-pairs-20260730-v1/manual_adjudication/labels/C06/bend_instance_labels.json",
)
_GOLD_JOB_FILES = ("bend_instance_labels.json", "bend_inspection_meta.json", "bend_inspection_report.json",
                   "aligned_scan.ply")
#: The owner's CONSUMED/FRESH split (``parts.<id>.split``) and the broad baseline's case list.
BENCHMARK_MANIFEST = RUNLOG / "benchmark_v2" / "BENCHMARK_V2.json"
BROAD_CASE_LIST = RUNLOG / "broad_baseline" / "case_list_broad_v3.json"
AI_GOLD_DIR = REPO_ROOT / "data" / "bend_corpus" / "annotations" / "ai_bend_lines_v1"
RELABEL_DIR = REPO_ROOT / "data" / "bend_corpus" / "annotations" / "ai_bend_lines_gold11_relabel_v1"
#: EHO4 (forbidden): no file naming one of these part numbers is ever read.
EHO4_PARTS: tuple[str, ...] = ("16231000", "38168001", "38173001", "26687013", "31409000", "47215000", "42637000",
                               "50476000", "31367000", "30820000")

_RESEARCH_ROOTS = ("data", "output", "outputs", "artifacts", "uploads", "live_scans")
#: Path fragments that are never readable here (forbidden list), case-insensitive, any location.
_FORBIDDEN_FRAGMENTS = ("holdout", "sealed_truth", "eho4", "one_shot", "oneshot", "authorization",
                        "incoming_codex_fetch_2026-10-17", "rhino-preview", "rhino_preview", "fresh_sealed")
_N_OPAQUE = re.compile(r"(?i)(^|[^a-z0-9])n0(0[1-9]|1[0-6])([^0-9]|$)")
_EHO4 = re.compile(r"(?<!\d)(" + "|".join(EHO4_PARTS) + r")(?!\d)")
_OWN_RUNLOGS = ("docs/production/runlog/freeze", "docs/production/runlog/legacy_control",
                "docs/production/runlog/g3_truth_material")
_LABEL_FILE = re.compile(r"^[0-9a-f]{24}\.json$")
_SCAN_SUFFIXES = (".ply", ".stl", ".obj")
_STEP_SUFFIXES = (".stp", ".step")


class CustodyError(PermissionError):
    """A read outside the inputs this chunk is authorized for (plan §9)."""


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def expand_home(path: str) -> Path:
    """``~/...`` paths of the benchmark manifest, resolved."""
    return Path(os.path.expanduser(str(path))).resolve()


_ALLOWLIST_CACHE: dict[str, dict[str, dict[str, Any]]] = {}


def consumed_allowlist(manifest: str | os.PathLike | None = None,
                       case_list: str | os.PathLike | None = None) -> dict[str, dict[str, Any]]:
    """``resolved path -> record`` of the files of CONSUMED parts in ``BENCHMARK_V2.json`` that this
    chunk may read: every listed path of a ``benchmark_scan`` (eval tier CONSUMED; a scan reserved as
    ``FRESH_SCAN_OF_CONSUMED_PART`` is left out) and every STEP CAD file; plus the scan and CAD paths
    of ``case_list_broad_v3.json`` (CONSUMED scans only, by construction of that list; it names the
    in-repo copies of some ``~/Downloads`` scans). Each record carries the listed sha256, which the
    caller compares with the bytes it reads."""
    mpath = Path(manifest) if manifest is not None else BENCHMARK_MANIFEST
    cpath = Path(case_list) if case_list is not None else BROAD_CASE_LIST
    key = f"{mpath.resolve()}|{cpath.resolve()}"
    if key in _ALLOWLIST_CACHE:
        return _ALLOWLIST_CACHE[key]
    doc = json.loads(mpath.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    if cpath.is_file():
        cl = json.loads(cpath.read_text(encoding="utf-8"))
        for c in cl.get("cases") or []:
            if c.get("scan_path") and str(c["scan_path"]).lower().endswith(_SCAN_SUFFIXES):
                out[str(expand_home(c["scan_path"]))] = {"kind": "SCAN", "part_id": c.get("family_id"),
                                                         "scan_id": c.get("case_id"), "sha256": c.get("scan_sha256")}
            if c.get("cad_path") and str(c["cad_path"]).lower().endswith(_STEP_SUFFIXES):
                out[str(expand_home(c["cad_path"]))] = {"kind": "STEP", "part_id": c.get("family_id"),
                                                        "sha256": c.get("cad_sha256")}
    for part_id, part in (doc.get("parts") or {}).items():
        if part.get("split") != "CONSUMED":
            continue
        for s in part.get("scans") or []:
            if s.get("role") != "benchmark_scan" or s.get("eval_tier") not in (None, "CONSUMED"):
                continue
            for raw in s.get("paths") or []:
                if str(raw).lower().endswith(_SCAN_SUFFIXES):
                    out[str(expand_home(raw))] = {"kind": "SCAN", "part_id": part_id, "scan_id": s.get("scan_id"),
                                                  "sha256": s.get("sha256")}
        for f in (part.get("cad") or {}).get("files") or []:
            if str(f.get("path", "")).lower().endswith(_STEP_SUFFIXES):
                out[str(expand_home(f["path"]))] = {"kind": "STEP", "part_id": part_id, "sha256": f.get("sha256")}
    _ALLOWLIST_CACHE[key] = out
    return out


def _forbidden(p: Path) -> str | None:
    low = str(p).lower()
    for frag in _FORBIDDEN_FRAGMENTS:
        if frag in low:
            return frag
    for part in p.parts:
        if _N_OPAQUE.search(part):
            return "N001-N016"
        if _EHO4.search(part):
            return "EHO4 part"
    try:
        rel = p.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        rel = None
    own = rel is not None and rel.startswith(_OWN_RUNLOGS)
    if not own and any("sealed" in part.lower() for part in p.parts):
        return "another workflow's sealed material"
    return None


def custody_check(path: str | os.PathLike, purpose: str) -> Path:
    """Return the resolved ``path`` if this chunk may read it, else raise :class:`CustodyError`.

    Admitted (module docstring): files under ``../source-assets/02_CAD_QC_3D``; the four files of
    the Gold11 and the two human non-Gold11 script-located label jobs (``output/<job>/``) and the two
    Gold11 ``shaked_xv`` label files; the CONSUMED benchmark scans and STEP CAD files listed in
    ``BENCHMARK_V2.json`` (:func:`consumed_allowlist`); label and provenance files of
    ``ai_bend_lines_v1`` and ``ai_bend_lines_gold11_relabel_v1``; ``data/bend_corpus/parts/*/
    metadata.json`` (stage keys, scoring only); and anything under the repository that is not a
    research root (source, docs, this chunk's own run logs). The forbidden list wins everywhere."""
    p = Path(path).expanduser().resolve()
    why = _forbidden(p)
    if why is not None:
        raise CustodyError(f"{purpose}: {p} names forbidden material ({why})")
    if _is_under(p, SOURCE_ASSETS):
        return p
    try:
        allow = consumed_allowlist()
    except (OSError, ValueError):  # pragma: no cover - the manifest is part of the run log
        allow = {}
    if str(p) in allow:
        return p
    repo = REPO_ROOT.resolve()
    if not _is_under(p, repo):
        raise CustodyError(f"{purpose}: {p} is outside the authorized inputs")
    rel = p.relative_to(repo)
    parts = rel.parts
    if parts and parts[0] in _RESEARCH_ROOTS:
        if (parts[0] == "output" and len(parts) == 3 and parts[1] in (*GOLD_LABEL_JOBS, *HUMAN_NON_GOLD11_JOBS)
                and parts[2] in _GOLD_JOB_FILES):
            return p
        if rel.as_posix() in GOLD11_XV_LABELS:
            return p
        if (parts[0] == "data" and len(parts) == 5 and parts[1] == "bend_corpus" and parts[2] == "parts"
                and parts[4] == "metadata.json"):
            return p
        if parts[:3] == ("data", "bend_corpus", "annotations") and len(parts) in (5, 6) and parts[3] in (
                AI_GOLD_DIR.name, RELABEL_DIR.name) and _LABEL_FILE.match(parts[-1]) and (
                len(parts) == 5 or parts[4] == "provenance"):
            return p
        raise CustodyError(f"{purpose}: {rel} is a research root path this chunk is not authorized to read")
    return p


def is_sealed_path(path: str | os.PathLike) -> bool:
    """True when any component of ``path`` carries the :data:`SEALED_TAG`."""
    return any(SEALED_TAG in part for part in Path(path).parts)


def rel_to_repo(path: str | os.PathLike) -> str:
    """Repository-relative POSIX path when possible, else the source-assets-relative
    ``../source-assets/02_CAD_QC_3D/...`` form, else ``~/...`` under the home directory, else the
    absolute path."""
    p = Path(path).resolve()
    if _is_under(p, REPO_ROOT.resolve()):
        return p.relative_to(REPO_ROOT.resolve()).as_posix()
    if _is_under(p, SOURCE_ASSETS):
        return "../source-assets/02_CAD_QC_3D/" + p.relative_to(SOURCE_ASSETS).as_posix()
    home = Path(os.path.expanduser("~")).resolve()
    if _is_under(p, home):
        return "~/" + p.relative_to(home).as_posix()
    return str(p)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_json(path: str | os.PathLike, doc: Any, *, indent: int | None = 1) -> str:
    """Write ``doc`` atomically (tmp + rename), sorted keys, UTF-8; returns the file's sha256."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=indent, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       default=_json_default) + "\n").encode("utf-8")
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    import hashlib

    return hashlib.sha256(data).hexdigest()


def read_json(path: str | os.PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


_FAMILY = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def family_of(name: str) -> str | None:
    """The 8-digit Sherman part number in a file name, or None (``Default.stp``, ``part 1.ply``)."""
    found = _FAMILY.findall(Path(name).name)
    return found[0] if found else None


def case_ref_for(scan_sha256: str) -> str:
    """Opaque physical case reference: never a file name, part number or stage token."""
    return "pc_" + scan_sha256[:12]


def dev_sku_for(cad_sha256: str) -> str:
    """Opaque DEV SKU of the DEV_AUTO_CONFIRMED recipe built from one STEP."""
    return "DEV-" + cad_sha256[:10].upper()


@dataclass(frozen=True)
class ScanGeometry:
    """Vertices of a scan file (mm, file frame) and its triangles when it is a mesh."""

    points: np.ndarray  # (N, 3) float64
    faces: np.ndarray | None  # (M, 3) int64
    normals: np.ndarray | None
    fmt: str  # PLY_* | STL_BINARY | STL_ASCII
    raw_vertex_count: int  # vertices as stored (an STL stores 3 per triangle)


def read_stl(path: str | os.PathLike) -> ScanGeometry:
    """Binary or ASCII STL; repeated triangle corners are merged (exact float32 equality)."""
    raw = Path(path).read_bytes()
    tris: np.ndarray
    fmt = "STL_BINARY"
    if len(raw) >= 84:
        (n,) = struct.unpack_from("<I", raw, 80)
        if 84 + 50 * n == len(raw):
            rec = np.frombuffer(raw, dtype=np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            tris = rec["v"].astype(np.float32)
        else:
            tris = _ascii_stl(raw)
            fmt = "STL_ASCII"
    else:
        tris = _ascii_stl(raw)
        fmt = "STL_ASCII"
    corners = tris.reshape(-1, 3)
    if not len(corners):
        raise ValueError(f"{path}: STL without triangles")
    uniq, inverse = np.unique(corners, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int64)
    return ScanGeometry(uniq.astype(np.float64), faces, None, fmt, int(len(corners)))


def _ascii_stl(raw: bytes) -> np.ndarray:
    text = raw.decode("ascii", errors="replace")
    if not text.lstrip().lower().startswith("solid"):
        raise ValueError("not an STL file (neither binary-sized nor ASCII 'solid')")
    vals = [line.split()[1:4] for line in text.splitlines() if line.strip().lower().startswith("vertex")]
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size == 0 or len(arr) % 3:
        raise ValueError("ASCII STL with an incomplete triangle")
    return arr.reshape(-1, 3, 3)


def read_scan(path: str | os.PathLike, *, with_faces: bool = True, purpose: str = "scan read") -> ScanGeometry:
    """Vertices of a PLY (E4a reader) or STL scan after the custody check."""
    p = custody_check(path, purpose)
    suffix = p.suffix.lower()
    if suffix == ".ply":
        data = read_ply(p, with_faces=with_faces)
        from ..plyio import read_ply_header

        header = read_ply_header(p)
        fmt = {"ascii": "PLY_ASCII", "binary_little_endian": "PLY_BINARY_LE",
               "binary_big_endian": "PLY_BINARY_BE"}.get(header.fmt, header.fmt)
        return ScanGeometry(np.asarray(data.vertices, dtype=np.float64), data.faces, data.normals, fmt,
                            int(len(data.vertices)))
    if suffix == ".stl":
        return read_stl(p)
    raise ValueError(f"{p.name}: unsupported scan format {suffix!r} (the line accepts PLY and STL)")


def compact_png(path: str | os.PathLike, colors: int = 192) -> str:
    """Re-encode a PNG in place with an adaptive palette (shaded point clouds shrink ~4x with no
    visible loss); returns the new file's sha256."""
    from PIL import Image

    p = Path(path)
    img = Image.open(p).convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    tmp = p.with_name(f".{p.name}.tmp")
    img.save(tmp, format="PNG", optimize=True)
    os.replace(tmp, p)
    return sha256_file(p)


def stable_rows(rows: Iterable[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    """Rows sorted by ``key`` (deterministic output files)."""
    return sorted((dict(r) for r in rows), key=lambda r: str(r.get(key)))


def doc_sha256(doc: Any) -> str:
    """Canonical sha256 of a JSON-able document (contracts ``canonical.py``)."""
    return sha256_hex(doc)
