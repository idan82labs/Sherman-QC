"""One-shot, five-source fresh CAD upload -> native bends -> V4 flat evaluation.

This is a consumed-development *selected-skin visual preview* campaign.  It
never reads a physical CAD in --freeze mode.  --execute requires a separately
written, independent GREEN PRE record.  The five source bindings are copied
from the released, source-only demo declaration; no fixture candidates or
Gold labels are loaded.  A terminal attempt is never resumed.

The live legacy server module is intentionally never imported: its route AST
is compiled with an isolated PartCatalog and UPLOAD_DIR.  Thus the exact route
body runs, but HTTP middleware/auth dispatch is outside this campaign.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace
from typing import Any
import uuid


ROOT = Path(__file__).resolve().parents[1]
# Direct `python scripts/...py --freeze` sets sys.path[0] to scripts/, unlike
# pytest/module execution. Resolve product imports from this exact Git root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROTOCOL = ROOT / "docs/operations/vendor-cad-upload-unfold-five-part-successor-protocol-2026-09-29.md"
CAD_ROOT = ROOT.parent / "source-assets/02_CAD_QC_3D"
ROUTE_SOURCE = ROOT / "backend/_server_legacy_impl.py"
SETUP_SERVICE_SOURCE = ROOT / "apps/api/services/vendor_setup_service.py"
PROFILE = "VENDOR_FIVE_FRESH_UPLOAD_UNFOLD_V1"
PRE_PROFILE = "VENDOR_FIVE_FRESH_UPLOAD_UNFOLD_PRE_V1"
CASE_SOURCES = (
    ("37361005", "37361005_01.stp", "81c428a0a55c531d67fcd68d74cac905c8b7b00e88277d836a932e954c7d636e"),
    ("49001000", "49001000_A-PRELIM.stp", "d8c1042f1673258976fa2b9ef812930c1be81ea3a60ab3f6357f966f7f452fd5"),
    ("42288005", "42288005_01.stp", "7afb000cdb1f18eddb4d270c5b8c2c5854d822d67adfd3cfebc5e7ad9e126837"),
    ("38172000", "38172000_01.stp", "4d512b5c565ab0e9e4d02e73fcaafd164ae82bcd977eba69198c8b0356f5644f"),
    ("10839000", "10839000_01.stp", "ea3e4820dff568916164c79417925ead79f307fba8aff11c7b3aeda7375e0b37"),
)
WALL_SECONDS = 60.0
RSS_KIB = 4 * 1024 * 1024
PREPARATION_SECONDS = 30.0
CAMPAIGN_SECONDS = 5 * WALL_SECONDS + PREPARATION_SECONDS
MAX_CAD_BYTES = 5_000_000
MAX_RESULT_BYTES = 40_000_000
MAX_LOG_BYTES = 1_000_000
POLL_SECONDS = 0.1
OWNED_SOURCE_FILES = (
    "scripts/evaluate_vendor_cad_upload_unfold_v1.py",
    "scripts/build_vendor_unfold_demo.py",
    "backend/_server_legacy_impl.py",
    "backend/part_catalog.py",
    "backend/vendor_native_step_upload_v1.py",
    "backend/vendor_unfold_worker_v3.py",
    "backend/vendor_unfold_preview_v4.py",
    "backend/vendor_unfold_contract_v3.py",
    "apps/api/services/vendor_setup_service.py",
    "docs/operations/vendor-cad-upload-unfold-five-part-successor-protocol-2026-09-29.md",
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read_bounded(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if not 0 < len(data) <= limit:
        raise ValueError(f"FILE_BUDGET_OR_EMPTY:{path.name}")
    return data


def _write_x(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_result(path: Path, value: Any) -> None:
    """No campaign result artifact may exceed the frozen 40 MB result cap."""
    data = _json(value)
    if len(data) > MAX_RESULT_BYTES:
        raise ValueError("CAMPAIGN_RESULT_BUDGET_EXCEEDED")
    _write_x(path, data)


def _source_pins() -> dict[str, str]:
    """Pin declared product/route source plus the native workers' own closure."""
    from backend import vendor_native_step_upload_v1 as native
    from backend import vendor_unfold_worker_v3 as flat

    pins = native._source_pins()
    pins.update(flat._source_pins())
    for relative in OWNED_SOURCE_FILES:
        path = (ROOT / relative).resolve()
        pins[str(path)] = _digest(path.read_bytes())
    return dict(sorted(pins.items()))


def _bindings() -> list[dict[str, str]]:
    return [{"part": part, "filename": filename, "sha256": sha}
            for part, filename, sha in CASE_SOURCES]


def _archive_sources(root: Path, pins: dict[str, str]) -> dict[str, Any]:
    """Freeze every declared source payload; no physical CAD is opened."""
    members = []
    with tarfile.open(root / "SOURCE.tar", "x") as archive:
        for index, (name, digest) in enumerate(sorted(pins.items())):
            path = Path(name)
            data = path.read_bytes()
            if _digest(data) != digest:
                raise ValueError("SOURCE_CHANGED_DURING_ARCHIVE")
            member = f"sources/{index:04d}-{digest[:16]}"
            info = tarfile.TarInfo(member)
            info.size = len(data)
            info.mode = 0o444
            info.mtime = 0
            import io
            archive.addfile(info, io.BytesIO(data))
            members.append({"source_path": name, "member": member,
                            "sha256": digest, "bytes": len(data)})
    manifest = {"profile": PROFILE, "members": members}
    _write_x(root / "SOURCE_MANIFEST.json", _json(manifest))
    archive_hash = hashlib.sha256()
    with (root / "SOURCE.tar").open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            archive_hash.update(block)
    return {"source_archive_sha256": archive_hash.hexdigest(),
            "source_manifest_sha256": _digest(_json(manifest))}


def freeze(root: Path) -> dict[str, Any]:
    """Create an exclusive proposal for independent PRE, without raw-file access."""
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    pins = _source_pins()
    archive = _archive_sources(root, pins)
    frozen = {"profile": PROFILE, "state": "FROZEN_AWAITING_INDEPENDENT_PRE",
        "protocol_sha256": _digest(PROTOCOL.read_bytes()),
        "runner_sha256": _digest(Path(__file__).read_bytes()),
        "source_pins": pins, "source_pins_sha256": _digest(_json(pins)),
        **archive, "case_sources": _bindings(),
        "cad_root": str(CAD_ROOT), "route_method": "AST_ISOLATED_EXACT_UPLOAD_PART_CAD",
        "runtime": "v3-development", "candidate_origin": "FRESH_UPLOAD_NATIVE_STEP_ONLY",
        "case_order": [part for part, _, _ in CASE_SOURCES],
        "limits": {"whole_case_wall_seconds": WALL_SECONDS, "whole_case_process_tree_rss_kib": RSS_KIB,
                   "campaign_seconds": CAMPAIGN_SECONDS, "preparation_cleanup_allowance_seconds": PREPARATION_SECONDS,
                   "source_bytes": MAX_CAD_BYTES, "result_bytes": MAX_RESULT_BYTES,
                   "log_bytes": MAX_LOG_BYTES, "cases_serial": True},
        "score_rule": "ALL_FIVE_DENOMINATOR_AVAILABLE_ONLY_WITH_EXACT_UPLOAD_ID_BIJECTION",
        "visual_rule": "SUPPORTED_TOP_BOTTOM_FOUR_OBLIQUE_AND_ALL_BEND_SELECTIONS_SEPARATE_POST",
        "manufacturing_ready": False, "units": "assumed_mm", "independent_truth": None}
    _write_x(root / "FROZEN.json", _json(frozen))
    for index, (part, _, expected) in enumerate(CASE_SOURCES):
        case = root / f"case-{index + 1:02d}-{part}"
        case.mkdir(mode=0o700)
        _write_x(case / "RESERVED.json", _json({"part": part, "expected_sha256": expected,
            "state": "NOT_EXECUTED", "case_index": index, "profile": PROFILE}))
    return frozen


def _validate_frozen(root: Path) -> tuple[dict[str, Any], str]:
    raw = _read_bounded(root / "FROZEN.json", 4_000_000)
    frozen = json.loads(raw)
    if (frozen.get("profile") != PROFILE or frozen.get("case_sources") != _bindings()
            or frozen.get("case_order") != [part for part, _, _ in CASE_SOURCES]
            or frozen.get("limits", {}).get("whole_case_wall_seconds") != WALL_SECONDS
            or frozen.get("limits", {}).get("whole_case_process_tree_rss_kib") != RSS_KIB
            or frozen.get("cad_root") != str(CAD_ROOT)
            or frozen.get("protocol_sha256") != _digest(PROTOCOL.read_bytes())
            or frozen.get("runner_sha256") != _digest(Path(__file__).read_bytes())
            or frozen.get("source_pins") != _source_pins()):
        raise ValueError("FROZEN_SOURCE_OR_CONFIG_MISMATCH")
    if _digest(_json(frozen["source_pins"])) != frozen["source_pins_sha256"]:
        raise ValueError("FROZEN_PIN_DIGEST_MISMATCH")
    manifest_raw = _read_bounded(root / "SOURCE_MANIFEST.json", 4_000_000)
    if _digest(manifest_raw) != frozen["source_manifest_sha256"]:
        raise ValueError("FROZEN_MANIFEST_MISMATCH")
    archive = root / "SOURCE.tar"
    h = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != frozen["source_archive_sha256"]:
        raise ValueError("FROZEN_ARCHIVE_MISMATCH")
    for index, (part, _, digest) in enumerate(CASE_SOURCES):
        case = root / f"case-{index + 1:02d}-{part}"
        if json.loads((case / "RESERVED.json").read_bytes()) != {
                "part": part, "expected_sha256": digest, "state": "NOT_EXECUTED",
                "case_index": index, "profile": PROFILE}:
            raise ValueError("FIVE_CASE_RESERVATION_MISMATCH")
        if any(path.name not in {"RESERVED.json"} for path in case.iterdir()):
            raise ValueError("ATTEMPT_ALREADY_STARTED_OR_DIRTY")
    return frozen, _digest(raw)


def _validate_pre(root: Path, frozen: dict[str, Any], frozen_sha: str) -> dict[str, Any]:
    record = json.loads(_read_bounded(root / "PRE_RUN_AUDIT.json", 100_000))
    if (record.get("profile") != PRE_PROFILE
            or record.get("decision") != "GREEN_FOR_FROZEN_FIVE_PART_UPLOAD_UNFOLD_RUN"
            or record.get("blocking_findings") != []
            or record.get("frozen_sha256") != frozen_sha
            or record.get("source_manifest_sha256") != frozen["source_manifest_sha256"]
            or record.get("source_archive_sha256") != frozen["source_archive_sha256"]
            or record.get("verified_case_sources") != _bindings()
            or record.get("heldouts_excluded") is not True
            or record.get("isolated_route_checked") is not True
            or record.get("whole_case_budget_checked") is not True
            or record.get("independent_reviewer") in (None, "")):
        raise ValueError("INDEPENDENT_PRE_NOT_GREEN_FOR_EXACT_FREEZE")
    return record


def _route_isolated(catalog: Any, upload_dir: Path):
    """Compile the *actual* endpoint body; do not import the live server."""
    from fastapi import HTTPException
    import aiofiles

    source = ROUTE_SOURCE.read_text()
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
               and node.name == "upload_part_cad"]
    if len(matches) != 1:
        raise ValueError("UPLOAD_ROUTE_NOT_UNIQUE")
    endpoint = matches[0]
    endpoint.decorator_list = []
    def forbidden(*_args, **_kwargs):
        raise RuntimeError("LEGACY_MESH_PATH_FORBIDDEN_IN_STEP_CAMPAIGN")
    scope = {"UploadFile": object, "File": lambda *args, **kwargs: None,
        "Form": lambda *args, **kwargs: None, "Path": Path, "uuid": uuid,
        "MAX_MESH_FILE_SIZE": 100_000_000, "HTTPException": HTTPException,
        "aiofiles": aiofiles, "UPLOAD_DIR": upload_dir,
        "get_catalog": lambda: catalog, "asyncio": asyncio,
        "logger": logging.getLogger("vendor-five-upload"),
        "load_bend_runtime_config": forbidden, "load_cad_geometry": forbidden,
        "extract_cad_bend_specs": forbidden}
    exec(compile(ast.Module(body=[endpoint], type_ignores=[]), str(ROUTE_SOURCE), "exec"), scope)
    return scope["upload_part_cad"]


def _vendor_get_isolated(catalog: Any):
    """Compile maintained GET dispatch with the same isolated catalog.

    Controlled mode remains explicit and no live legacy/server import occurs.
    Its production stale-generation check is executed, not reimplemented.
    """
    from backend.vendor_unfold_contract_v3 import unavailable as unavailable_v4
    from backend.vendor_unfold_preview import unavailable_unfold_preview
    from backend.vendor_unfold_worker_v3 import build_unfold_preview_bounded
    from fastapi import HTTPException

    tree = ast.parse(SETUP_SERVICE_SOURCE.read_text())
    matches = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
               and node.name == "get_vendor_setup"]
    if len(matches) != 1:
        raise ValueError("VENDOR_GET_ROUTE_NOT_UNIQUE")
    async def threadpool(function, *args):
        return await asyncio.to_thread(function, *args)
    def forbidden(*_args):
        raise RuntimeError("UNCONTROLLED_FOLDED_FLANGE_PATH_FORBIDDEN")
    def line(value):
        return json.loads(value) if isinstance(value, str) else value
    scope = {"legacy": SimpleNamespace(get_catalog=lambda: catalog),
        "copy": copy, "os": SimpleNamespace(environ={"SHERMAN_VENDOR_UNFOLD_RUNTIME": "v3-development"}),
        "HTTPException": HTTPException, "_line": line, "run_in_threadpool": threadpool,
        "build_flange_previews": forbidden, "FlangePreviewUnavailable": ValueError,
        "build_unfold_preview_bounded": forbidden,
        "build_unfold_preview_v2_bounded": forbidden,
        "build_unfold_preview_v3_bounded": build_unfold_preview_bounded,
        "unavailable_unfold_preview": unavailable_unfold_preview,
        "unavailable_unfold_preview_v4": unavailable_v4}
    exec(compile(ast.Module(body=[matches[0]], type_ignores=[]),
                 str(SETUP_SERVICE_SOURCE), "exec"), scope)
    return scope["get_vendor_setup"]


def _actual_upload_result(case_dir: Path, source_raw: bytes, part: str,
                          expected_sha: str) -> dict[str, Any]:
    """Child-only isolated route + catalog + controlled V4, all from one CAD."""
    from backend.part_catalog import PartCatalogManager
    from backend.vendor_unfold_contract_v2 import canonical_candidates
    from backend.vendor_unfold_worker_v3 import validate_preview

    isolated = case_dir / "isolated-upload"
    isolated.mkdir(mode=0o700, exist_ok=False)
    catalog = PartCatalogManager(str(isolated / "parts.sqlite"))
    row = catalog.create_part(part, f"Five-part consumed-development {part}")
    route = _route_isolated(catalog, isolated)

    class FileLike:
        filename = next(filename for number, filename, _ in CASE_SOURCES if number == part)
        async def read(self, _limit):
            return source_raw

    response = asyncio.run(route(row.id, FileLike(), True))
    _write_result(case_dir / "UPLOAD_RESPONSE.json", response)
    published = catalog.get_part(row.id)
    if response.get("status") != "CAD_PROCESSED":
        return {"status": "UNAVAILABLE", "reason": response.get("bend_analysis_error") or response.get("status"),
                "upload_status": response.get("status"), "candidate_count": 0,
                "preview_status": "NOT_RUN_NO_PUBLISHED_UPLOAD"}
    if (published.cad_file_hash != expected_sha or not published.cad_file_path
            or not published.preview_mesh_path
            or not Path(published.cad_file_path).resolve().is_relative_to((isolated / "parts_cad").resolve())
            or not Path(published.preview_mesh_path).resolve().is_relative_to((isolated / "parts_previews").resolve())
            or _digest(_read_bounded(Path(published.cad_file_path), MAX_CAD_BYTES)) != expected_sha):
        raise ValueError("UPLOAD_PUBLICATION_SOURCE_MISMATCH")
    folded = _read_bounded(Path(published.preview_mesh_path), MAX_RESULT_BYTES)
    specs = catalog.get_bend_specs_for_part(row.id)
    candidates = [{"feature_id": spec.id, "label": spec.bend_id,
        "source": "cad_candidate", "status": "active", "issue": None,
        "review_status": "pending", "note": "",
        "line_start": json.loads(spec.bend_line_start),
        "line_end": json.loads(spec.bend_line_end),
        "target_angle": spec.target_angle, "target_radius": spec.target_radius}
        for spec in specs]
    bound = canonical_candidates(candidates)
    if len(bound) != len(specs) or len({row["feature_id"] for row in bound}) != len(bound):
        raise ValueError("UPLOAD_CANDIDATE_IDENTITY_MISMATCH")
    generation = catalog.candidate_generation(row.id)
    source_meta = response.get("cad_source")
    if (type(source_meta) is not dict or source_meta.get("source_sha256") != expected_sha
            or source_meta.get("folded_mesh_sha256") != _digest(folded)
            or response.get("bends_detected") != len(candidates)
            or {item.get("bend_id") for item in response.get("bends", [])}
                != {item["label"] for item in candidates}):
        raise ValueError("UPLOAD_NATIVE_PROVENANCE_MISMATCH")
    candidate_receipt = {"part": part, "source_sha256": expected_sha,
        "candidate_generation": generation, "candidate_input_sha256": _digest(_json(bound)),
        "folded_mesh_sha256": _digest(folded), "folded_mesh_bytes": len(folded),
        "candidates": candidates}
    _write_result(case_dir / "CANDIDATES.json", candidate_receipt)
    _write_result(case_dir / "FOLDED_ASSET.json", {"relative_path": str(
        Path(published.preview_mesh_path).relative_to(case_dir)),
        "source_sha256": expected_sha, "candidate_generation": generation,
        "sha256": _digest(folded), "bytes": len(folded)})
    setup = asyncio.run(_vendor_get_isolated(catalog)(row.id))
    if (setup.get("part", {}).get("cad_file_hash") != expected_sha
            or setup.get("part", {}).get("cad_file_path") != published.cad_file_path
            or setup.get("part", {}).get("preview_mesh_path") != published.preview_mesh_path
            or setup.get("candidate_generation") != generation
            or setup.get("candidates") != candidates
            or setup.get("flange_surfaces") != []
            or setup.get("flange_surface_status") != "CONTROLLED_PATH_FOLDED_FLANGE_NOT_QUALIFIED"):
        raise ValueError("VENDOR_GET_UPLOAD_SNAPSHOT_MISMATCH")
    flat = setup["unfold_preview"]
    validate_preview(flat, expected_sha, candidates)
    # A source edit or concurrent publication is a refusal, never a mixed view.
    latest = catalog.get_part(row.id)
    if (latest.cad_file_hash != expected_sha or latest.cad_file_path != published.cad_file_path
            or catalog.candidate_generation(row.id) != generation):
        raise ValueError("CAD_CANDIDATES_CHANGED_DURING_PREVIEW")
    flat_bytes, setup_bytes = _json(flat), _json(setup)
    if len(flat_bytes) > MAX_RESULT_BYTES or len(setup_bytes) > MAX_RESULT_BYTES:
        raise ValueError("VENDOR_SETUP_RESULT_BUDGET_EXCEEDED")
    _write_x(case_dir / "FLAT_PREVIEW.json", flat_bytes)
    _write_x(case_dir / "VENDOR_SETUP_RESPONSE.json", setup_bytes)
    if flat["status"] != "AVAILABLE":
        return {"status": "UNAVAILABLE", "reason": flat["reason"],
            "upload_status": response["status"], "candidate_count": len(candidates),
            "candidate_generation": generation,
            "candidate_input_sha256": candidate_receipt["candidate_input_sha256"],
            "folded_mesh_sha256": candidate_receipt["folded_mesh_sha256"],
            "preview_status": flat["status"]}
    lines = flat["bend_lines"]
    overlays = flat["flange_surfaces"]
    expected_ids = {row["feature_id"] for row in candidates}
    if (len(lines) != len(expected_ids) or len(overlays) != len(expected_ids)
            or {line["feature_id"] for line in lines} != expected_ids
            or {surface["feature_id"] for surface in overlays} != expected_ids
            or flat["diagnostics"].get("candidate_input_sha256")
                != candidate_receipt["candidate_input_sha256"]):
        raise ValueError("FLAT_UPLOAD_BEND_BIJECTION_FAILED")
    return {"status": "AVAILABLE", "reason": None, "upload_status": response["status"],
        "candidate_count": len(candidates), "flat_bend_count": len(lines),
        "candidate_generation": generation,
        "candidate_input_sha256": candidate_receipt["candidate_input_sha256"],
        "folded_mesh_sha256": candidate_receipt["folded_mesh_sha256"],
        "flat_preview_sha256": _digest(_json(flat)), "preview_status": flat["status"],
        "visual_review_status": "PENDING_INDEPENDENT_POST",
        "units_verified": False, "manufacturing_ready": False}


def _case_worker(root: Path, index: int, nonce: str) -> None:
    """Private child entry: re-check freeze/PRE before physical byte access."""
    if os.environ.get("SHERMAN_VENDOR_FIVE_CASE_NONCE") != nonce:
        raise ValueError("PRIVATE_CASE_ENTRY_REQUIRED")
    frozen, frozen_sha = _validate_frozen_started(root)
    _validate_pre(root, frozen, frozen_sha)
    part, filename, expected = CASE_SOURCES[index]
    case = root / f"case-{index + 1:02d}-{part}"
    started_record = json.loads(_read_bounded(root / "EXECUTION_STARTED.json", 10_000))
    if (type(started_record) is not dict
            or set(started_record) != {"profile", "frozen_sha256", "started_unix_ns"}
            or started_record["profile"] != PROFILE
            or started_record["frozen_sha256"] != frozen_sha
            or type(started_record["started_unix_ns"]) is not int
            or started_record["started_unix_ns"] <= 0):
        raise ValueError("PARENT_EXECUTION_SENTINEL_REQUIRED")
    launch = json.loads(_read_bounded(case / "CASE_LAUNCH.json", 10_000))
    if launch != {"profile": PROFILE, "case_index": index, "part": part,
                  "frozen_sha256": frozen_sha, "nonce_sha256": _digest(nonce.encode()),
                  "parent_pid": os.getppid()}:
        raise ValueError("PARENT_OWNED_CASE_LAUNCH_REQUIRED")
    # This exclusive claim is *before* raw access. A second direct entry with
    # the same nonce is a terminal replay, not a retry of the source case.
    _write_result(case / "CASE_CLAIM.json", {"profile": PROFILE,
        "case_index": index, "part": part, "claim_pid": os.getpid(),
        "nonce_sha256": _digest(nonce.encode())})
    try:
        if os.getppid() != launch["parent_pid"]:
            raise ValueError("PARENT_DIED_BEFORE_RAW_ACCESS")
        source = _read_bounded(CAD_ROOT / filename, MAX_CAD_BYTES)
        if _digest(source) != expected:
            raise ValueError("RAW_CAD_HASH_MISMATCH")
        if os.getppid() != launch["parent_pid"]:
            raise ValueError("PARENT_DIED_AFTER_RAW_ACCESS")
        _write_x(case / "UPLOAD_SOURCE.step", source)
        outcome = _actual_upload_result(case, source, part, expected)
        _write_result(case / "CHILD_OUTCOME.json", outcome)
    except BaseException as exc:
        _write_result(case / "CHILD_FAILURE.json", {"status": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}:{str(exc)[:400]}"})
        raise


def _validate_frozen_started(root: Path) -> tuple[dict[str, Any], str]:
    """Same frozen pins, but case reservation has now been parent-started."""
    raw = _read_bounded(root / "FROZEN.json", 4_000_000)
    frozen = json.loads(raw)
    if (frozen.get("profile") != PROFILE or frozen.get("case_sources") != _bindings()
            or frozen.get("runner_sha256") != _digest(Path(__file__).read_bytes())
            or frozen.get("protocol_sha256") != _digest(PROTOCOL.read_bytes())
            or frozen.get("source_pins") != _source_pins()):
        raise ValueError("FROZEN_SOURCE_OR_CONFIG_MISMATCH")
    return frozen, _digest(raw)


def _ps_rows() -> dict[int, tuple[int, int, int]]:
    """Return pid -> (ppid, pgid, RSS KiB); zombies excluded."""
    try:
        completed = subprocess.run(["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss=,stat="],
                                   capture_output=True, text=True, check=True, timeout=2)
        rows = {}
        for line in completed.stdout.splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            pid, ppid, pgid, rss = map(int, fields[:4])
            if not fields[4].startswith("Z"):
                rows[pid] = (ppid, pgid, rss)
        return rows
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError("PROCESS_TREE_MONITOR_UNAVAILABLE") from exc


def _tree(rows: dict[int, tuple[int, int, int]], root_pid: int) -> set[int]:
    found = {root_pid} if root_pid in rows else set()
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _, _) in rows.items():
            if ppid in found and pid not in found:
                found.add(pid)
                changed = True
    return found


def _admission_quiescent(*, allow_absent: bool = False) -> tuple[bool, str]:
    """Check the shared native-worker journal; PPID samples alone miss orphans.

    Both inner workers write ADMITTING/RUNNING before/after their private PGID
    spawn and QUIESCENT only after whole-group cleanup. An orphan between our
    outer monitor samples leaves a non-quiescent durable journal and blocks the
    next source case. This is still not a continuous memory HWM proof.
    """
    from backend.vendor_unfold_worker_v3 import ADMISSION_LOCK_PATH
    path = Path(ADMISSION_LOCK_PATH)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                return False, "ADMISSION_JOURNAL_OWNERSHIP_UNVERIFIED"
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False, "ADMISSION_JOURNAL_LOCK_BUSY"
            raw = stream.read(1025)
            if not 0 < len(raw) <= 1024:
                return False, "ADMISSION_JOURNAL_SIZE_INVALID"
            state = json.loads(raw)
            if state == {"state": "QUIESCENT", "pgid": None}:
                return True, "QUIESCENT"
            return False, str(state.get("state", "INVALID_JOURNAL"))[:80]
    except FileNotFoundError:
        return (True, "NO_NATIVE_ADMISSION_YET") if allow_absent else (False, "ADMISSION_JOURNAL_MISSING_AFTER_CASE")
    except (OSError, ValueError, TypeError, KeyError):
        return False, "ADMISSION_JOURNAL_UNREADABLE"


def _kill_seen_groups(groups: set[int]) -> bool:
    """Best-effort cleanup; caller must refuse further work if not quiescent."""
    own_group = os.getpgrp()
    for pgid in sorted(groups, reverse=True):
        if pgid == own_group or pgid <= 0:
            continue
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        rows = _ps_rows()
        if not any(pgid in groups for _, pgid, _ in rows.values()):
            return True
        time.sleep(.05)
    return False


def _run_case(root: Path, index: int, campaign_deadline: float) -> dict[str, Any]:
    part, _, expected = CASE_SOURCES[index]
    case = root / f"case-{index + 1:02d}-{part}"
    started = time.monotonic()
    if started + WALL_SECONDS > campaign_deadline:
        return {"part": part, "status": "NOT_EXECUTED", "reason": "CAMPAIGN_RESERVATION_EXHAUSTED"}
    nonce = uuid.uuid4().hex
    frozen_sha = _digest(_read_bounded(root / "FROZEN.json", 4_000_000))
    _write_x(case / "CASE_LAUNCH.json", _json({"profile": PROFILE,
        "case_index": index, "part": part, "frozen_sha256": frozen_sha,
        "nonce_sha256": _digest(nonce.encode()), "parent_pid": os.getpid()}))
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "SHERMAN_VENDOR_FIVE_CASE_NONCE": nonce})
    groups: set[int] = set()
    peak = 0
    process = None
    reason = None
    monitor_failed = False
    with (case / "stdout.log").open("xb") as out, (case / "stderr.log").open("xb") as err:
        try:
            if not _ps_rows():
                raise RuntimeError("PROCESS_TREE_MONITOR_UNAVAILABLE_BEFORE_SPAWN")
            ready, admission_before = _admission_quiescent(allow_absent=True)
            if not ready:
                raise RuntimeError(f"NATIVE_ADMISSION_NOT_QUIESCENT_BEFORE_SPAWN:{admission_before}")
            if (time.monotonic() - started >= WALL_SECONDS
                    or time.monotonic() + WALL_SECONDS > campaign_deadline):
                raise TimeoutError("WHOLE_CASE_RESERVATION_EXPIRED_BEFORE_SPAWN")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                "--case-worker", str(root), str(index), nonce], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
            groups.add(process.pid)
            while process.poll() is None:
                if time.monotonic() - started >= WALL_SECONDS:
                    raise TimeoutError("WHOLE_CASE_WALL_BUDGET_EXCEEDED")
                rows = _ps_rows()
                live = _tree(rows, process.pid)
                if not live and process.poll() is None:
                    raise RuntimeError("PROCESS_TREE_MONITOR_LOST_CHILD")
                groups.update(rows[pid][1] for pid in live)
                peak = max(peak, sum(rows[pid][2] for pid in live))
                if peak > RSS_KIB:
                    raise MemoryError("WHOLE_CASE_RSS_BUDGET_EXCEEDED")
                if out.tell() + err.tell() > MAX_LOG_BYTES:
                    raise RuntimeError("WHOLE_CASE_LOG_BUDGET_EXCEEDED")
                time.sleep(POLL_SECONDS)
            if time.monotonic() - started >= WALL_SECONDS:
                raise TimeoutError("WHOLE_CASE_WALL_BUDGET_EXCEEDED")
            if process.returncode != 0:
                reason = f"CASE_CHILD_EXIT_{process.returncode}"
            if out.tell() + err.tell() > MAX_LOG_BYTES:
                reason = "WHOLE_CASE_LOG_BUDGET_EXCEEDED"
        except (OSError, ValueError, RuntimeError, TimeoutError, MemoryError,
                subprocess.SubprocessError) as exc:
            reason = str(exc)
            monitor_failed = "MONITOR" in reason
        finally:
            try:
                quiescent = _kill_seen_groups(groups) if process is not None else True
            except (OSError, RuntimeError, subprocess.SubprocessError):
                quiescent = False
            if process is not None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    quiescent = False
    elapsed = time.monotonic() - started
    admission_ok, admission_after = _admission_quiescent()
    primary_reason = reason
    if elapsed >= WALL_SECONDS:
        reason = reason or "WHOLE_CASE_WALL_BUDGET_EXCEEDED"
    if not quiescent:
        reason = "PROCESS_TREE_CLEANUP_UNVERIFIED"
    if not admission_ok:
        reason = "INNER_NATIVE_PROCESS_CLEANUP_UNVERIFIED"
    outcome_path = case / "CHILD_OUTCOME.json"
    failure_path = case / "CHILD_FAILURE.json"
    if reason and failure_path.is_file() and reason.startswith("CASE_CHILD_EXIT_"):
        failure = json.loads(_read_bounded(failure_path, 10_000))
        reason = failure.get("reason") or reason
    if reason is None and outcome_path.is_file():
        outcome = json.loads(_read_bounded(outcome_path, MAX_RESULT_BYTES))
    else:
        outcome = {"status": "UNAVAILABLE", "reason": reason or "CHILD_OUTCOME_MISSING"}
    if outcome.get("status") not in {"AVAILABLE", "UNAVAILABLE"}:
        outcome = {"status": "UNAVAILABLE", "reason": "INVALID_CHILD_OUTCOME"}
    return {"part": part, "expected_sha256": expected, "case_index": index,
        "status": outcome["status"], "reason": outcome.get("reason"),
        "primary_failure_reason": primary_reason,
        "child_outcome": outcome, "parent_observed_exit": process.returncode if process else None,
        "process_tree_quiescent": quiescent, "monitor_failed": monitor_failed,
        "native_admission_quiescent": admission_ok,
        "native_admission_after": admission_after,
        "sampled_peak_process_tree_rss_kib": peak, "continuous_hwm_verified": False,
        "elapsed_seconds": elapsed}


def execute(root: Path) -> list[dict[str, Any]]:
    frozen, frozen_sha = _validate_frozen(root)
    _validate_pre(root, frozen, frozen_sha)
    # The exclusive sentinel is created before any physical access.  No retry.
    _write_x(root / "EXECUTION_STARTED.json", _json({"profile": PROFILE,
        "frozen_sha256": frozen_sha, "started_unix_ns": time.time_ns()}))
    campaign_started = time.monotonic()
    campaign_deadline = campaign_started + CAMPAIGN_SECONDS
    results = []
    halt_reason = None
    for index, (part, _, expected) in enumerate(CASE_SOURCES):
        if halt_reason:
            row = {"part": part, "expected_sha256": expected, "case_index": index,
                   "status": "NOT_EXECUTED", "reason": halt_reason}
        else:
            try:
                row = _run_case(root, index, campaign_deadline)
            except BaseException as exc:
                row = {"part": part, "expected_sha256": expected, "case_index": index,
                       "status": "UNAVAILABLE", "reason": f"PARENT_EXCEPTION_{type(exc).__name__}"}
                halt_reason = "PARENT_EXCEPTION_STOPS_FURTHER_NATIVE_WORK"
        case = root / f"case-{index + 1:02d}-{part}"
        _write_result(case / "TERMINAL.json", row)
        results.append(row)
        if (row.get("process_tree_quiescent") is False or row.get("monitor_failed") is True
                or row.get("native_admission_quiescent") is False):
            halt_reason = "PREVIOUS_CASE_CLEANUP_OR_MONITOR_UNVERIFIED"
        if time.monotonic() > campaign_deadline:
            halt_reason = "CAMPAIGN_DEADLINE_EXCEEDED"
    state = ("FAILED_SAFETY_GATE_PENDING_POST" if any(
        row["status"] == "NOT_EXECUTED" or row.get("process_tree_quiescent") is False
        or row.get("monitor_failed") is True
        or row.get("native_admission_quiescent") is False for row in results)
        else "COMPLETE_PENDING_INDEPENDENT_POST")
    summary = {"profile": PROFILE, "state": state,
        "frozen_sha256": frozen_sha, "all_five_terminal": len(results) == 5,
        "available_count": sum(row["status"] == "AVAILABLE" for row in results),
        "denominator": 5, "rows": results, "elapsed_seconds": time.monotonic() - campaign_started,
        "manufacturing_ready": False, "independent_post": "PENDING"}
    if summary["elapsed_seconds"] > CAMPAIGN_SECONDS:
        summary["state"] = "FAILED_CAMPAIGN_DEADLINE_PENDING_POST"
    _write_result(root / "CAMPAIGN_TERMINAL.json", summary)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--case-worker", nargs=3, metavar=("ROOT", "INDEX", "NONCE"))
    parser.add_argument("--attempt-root", type=Path)
    args = parser.parse_args()
    if args.case_worker is not None:
        root, index, nonce = args.case_worker
        _case_worker(Path(root), int(index), nonce)
        return
    if args.attempt_root is None:
        parser.error("--attempt-root is required")
    if args.freeze:
        frozen = freeze(args.attempt_root.resolve())
        print(_json({"state": frozen["state"], "attempt_root": str(args.attempt_root.resolve()),
                     "case_count": 5, "source_pin_count": len(frozen["source_pins"])}).decode(), flush=True)
    else:
        rows = execute(args.attempt_root.resolve())
        terminal = json.loads(_read_bounded(args.attempt_root.resolve() / "CAMPAIGN_TERMINAL.json",
                                            MAX_RESULT_BYTES))
        print(_json({"state": terminal["state"], "terminal_rows": len(rows),
                     "available_count": sum(row["status"] == "AVAILABLE" for row in rows)}).decode(), flush=True)


if __name__ == "__main__":
    main()
