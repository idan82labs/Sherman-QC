"""Filesystem layout of the line data root, job workspaces and engine outputs.

Pure path math only: nothing here touches the filesystem.

Data root (``SHERMAN_LINE_DATA_ROOT``, absolute, outside the repo)::

    db/                      line.sqlite3, recipes.sqlite3, security.sqlite3
    recipes/cad/             <sha256>.step|.stp   (content addressed, frozen at release)
    recipes/drawings/        <sha256><ext>
    recipes/flats/           <sha256>.dxf|.step|.stp
    recipes/compiled/<recipe_sha256>/   E1 output: compiled_part.json, stages/, compile.log
    archive/scans/YYYY/MM/   <sha256><ext>
    quarantine/YYYY-MM-DD/
    jobs/<job_id>/           job workspace (below)
    engine/ logs/ secrets/ backups/ config/

Job workspace ``<data_root>/jobs/<job_id>/``::

    job.json                 LineJobV1
    recipe.json              canonical bytes of the PartRecipeV1 used
    compiled/                copy of recipes/compiled/<recipe_sha256>/ (hash-checked)
    input/scan.ply|.stl      the scan, staged under an opaque name
    input/context.json       ResultContextV1 for C06/the runner (never passed to E3)
    engine/                  E3 --out  (ENGINE_OUTPUT_FILES)
    result/                  C06 postprocess output (RESULT_FILES)

E3 output directory (``--out``), every geometry file in ``cad_model_mm``::

    result.json              ShopfloorInspectionResultV1 (canonical JSON)
    registered_scan.ply      binary LE PLY, float32 xyz, all ingested scan points in the frame
    deviations.f32           float32 LE, index-aligned with registered_scan vertices, NaN = none
    segment_labels.u16       uint16 LE per point (optional; 0 = unassigned, see ENGINE_IO.md)
    display/stage_mesh.ply   display mesh of the decoded stage S_k
    display/final_mesh.ply   display mesh of the final stage S_N
    bend_geometry.json       BendGeometryFileV1: polylines + label anchors
    engine.log               human-readable log (never parsed)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePath, PurePosixPath
from typing import TypeVar

__all__ = [
    "JOB_ID_RE",
    "FORBIDDEN_DATA_ROOT_PARENTS",
    "DATA_ROOT_DIRS",
    "WORKSPACE_NAMES",
    "ENGINE_OUTPUT_FILES",
    "COMPILE_OUTPUT_FILES",
    "RESULT_FILES",
    "ARTIFACT_ALLOWLIST",
    "WorkspacePaths",
    "workspace_paths",
    "compiled_part_rel_dir",
    "archive_scan_rel_path",
    "stage_cloud_rel_path",
    "stage_mesh_rel_path",
]

JOB_ID_RE = re.compile(r"^line_[0-9a-f]{32}$")

#: A data root (or schema export dir) must not resolve under these repo directories.
FORBIDDEN_DATA_ROOT_PARENTS: tuple[str, ...] = ("data", "output", "outputs", "artifacts", "uploads", "live_scans")

DATA_ROOT_DIRS: tuple[str, ...] = (
    "db",
    "recipes/cad",
    "recipes/drawings",
    "recipes/flats",
    "recipes/compiled",
    "archive/scans",
    "quarantine",
    "jobs",
    "engine",
    "logs",
    "secrets",
    "backups",
    "config",
)

#: Top-level names inside a job workspace.
WORKSPACE_NAMES: dict[str, str] = {
    "job": "job.json",
    "recipe": "recipe.json",
    "compiled": "compiled",
    "input": "input",
    "context": "input/context.json",
    "engine": "engine",
    "result": "result",
}

#: E3 ``--out`` contents (relative to the out dir).
ENGINE_OUTPUT_FILES: dict[str, str] = {
    "result": "result.json",
    "registered_scan": "registered_scan.ply",
    "deviations": "deviations.f32",
    "segment_labels": "segment_labels.u16",
    "stage_mesh": "display/stage_mesh.ply",
    "final_mesh": "display/final_mesh.ply",
    "bend_geometry": "bend_geometry.json",
    "log": "engine.log",
}

#: E1 ``--out`` contents (relative to the out dir). Stage files are named per stage model id.
COMPILE_OUTPUT_FILES: dict[str, str] = {
    "compiled_part": "compiled_part.json",
    "stages_dir": "stages",
    "log": "compile.log",
}

#: C06 postprocess output (relative to <workspace>/result).
RESULT_FILES: dict[str, str] = {
    "shopfloor_result": "shopfloor_result.json",
    "postprocess_error": "postprocess_error.json",
    "viewer_scan": "viewer_scan.ply",
    "viewer_deviations": "viewer_deviations.f32",
    "stage_mesh": "stage_mesh.ply",
    "final_mesh": "final_mesh.ply",
    "bend_geometry": "bend_geometry.json",
    "overview_png": "overview.png",
    "report_pdf": "report.pdf",
    "log": "postprocess.log",
}

#: Served artifact name -> path relative to the job workspace. Nothing else is served.
ARTIFACT_ALLOWLIST: dict[str, str] = {
    "viewer_scan.ply": "result/viewer_scan.ply",
    "viewer_deviations.f32": "result/viewer_deviations.f32",
    "stage_mesh.ply": "result/stage_mesh.ply",
    "final_mesh.ply": "result/final_mesh.ply",
    "bend_geometry.json": "result/bend_geometry.json",
    "report.pdf": "result/report.pdf",
    "overview.png": "result/overview.png",
}

P = TypeVar("P", bound=PurePath)


@dataclass(frozen=True)
class WorkspacePaths:
    root: PurePath
    job: PurePath
    recipe: PurePath
    compiled_dir: PurePath
    input_dir: PurePath
    context: PurePath
    engine_dir: PurePath
    result_dir: PurePath

    def scan(self, ext: str) -> PurePath:
        """Staged scan path; ``ext`` is ``.ply`` or ``.stl``."""
        ext = ext.lower()
        if ext not in (".ply", ".stl"):
            raise ValueError("scan extension must be .ply or .stl")
        return self.input_dir / f"scan{ext}"

    def engine_file(self, key: str) -> PurePath:
        return self.engine_dir / ENGINE_OUTPUT_FILES[key]

    def result_file(self, key: str) -> PurePath:
        return self.result_dir / RESULT_FILES[key]

    def artifact(self, name: str) -> PurePath:
        """Path of a served artifact; ``KeyError`` for anything not allow-listed."""
        return self.root / ARTIFACT_ALLOWLIST[name]


def workspace_paths(jobs_root: P | str, job_id: str) -> WorkspacePaths:
    """Workspace paths for ``job_id`` under ``jobs_root`` (``<data_root>/jobs``)."""
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        raise ValueError("job_id must match ^line_[0-9a-f]{32}$")
    base = jobs_root if isinstance(jobs_root, PurePath) else PurePath(jobs_root)
    root = base / job_id
    return WorkspacePaths(
        root=root,
        job=root / WORKSPACE_NAMES["job"],
        recipe=root / WORKSPACE_NAMES["recipe"],
        compiled_dir=root / WORKSPACE_NAMES["compiled"],
        input_dir=root / WORKSPACE_NAMES["input"],
        context=root / WORKSPACE_NAMES["context"],
        engine_dir=root / WORKSPACE_NAMES["engine"],
        result_dir=root / WORKSPACE_NAMES["result"],
    )


def compiled_part_rel_dir(recipe_sha256: str) -> str:
    """Data-root-relative directory of a recipe's compile output."""
    if not re.fullmatch(r"[0-9a-f]{64}", recipe_sha256 or ""):
        raise ValueError("recipe_sha256 must be 64 lowercase hex characters")
    return f"recipes/compiled/{recipe_sha256}"


def archive_scan_rel_path(sha256: str, ext: str, year: int, month: int) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", sha256 or ""):
        raise ValueError("sha256 must be 64 lowercase hex characters")
    ext = ext.lower()
    if ext not in (".ply", ".stl"):
        raise ValueError("scan extension must be .ply or .stl")
    if not (1 <= month <= 12 and 1970 <= year <= 9999):
        raise ValueError("invalid archive year/month")
    return str(PurePosixPath("archive", "scans", f"{year:04d}", f"{month:02d}", f"{sha256}{ext}"))


def stage_cloud_rel_path(stage_model_id: str) -> str:
    """Path of a stage sample cloud, relative to the compile out dir."""
    return f"{COMPILE_OUTPUT_FILES['stages_dir']}/{stage_model_id}.npz"


def stage_mesh_rel_path(stage_model_id: str) -> str:
    """Path of a stage display mesh, relative to the compile out dir."""
    return f"{COMPILE_OUTPUT_FILES['stages_dir']}/{stage_model_id}.ply"
