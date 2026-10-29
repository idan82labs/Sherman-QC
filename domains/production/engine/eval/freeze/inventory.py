"""Inventory of the authorized development inputs (plan §9 and the owner's 2026-10-17 wave-4 directions).

Two parts, every file hashed (sha256) and classified:

1. **Source tree** ``../source-assets/02_CAD_QC_3D/**`` (plan §9): every file,
   as the first E4b pass did (the same rules, so the same entries):

   - ``STEP`` (``.stp``/``.step``): a candidate for a DEV_AUTO_CONFIRMED recipe;
   - ``SCAN`` (``.ply``/``.stl``: the formats the line accepts);
   - ``OTHER`` (``.obj``, Parasolid ``.X_T``/``.X_B``, ``.xdl``, ``.dwg``, ...): listed,
     never used (a scan the line cannot ingest is not a case).

   Byte-identical copies collapse to one entry per sha256 with the other paths as
   ``aliases``; the canonical path is the shortest (then lexicographically first)
   one. The part family is the 8-digit Sherman part number in the file name.
   Scan headers (vertex / face / triangle counts, exporter comment) are read; no
   geometry is loaded here.

2. **External CONSUMED files** (``external_entries``): the STEP CAD of every part
   marked CONSUMED in ``BENCHMARK_V2.json`` that lives outside the source tree
   (e-mail attachments, ``~/Downloads``), and the scans of
   ``case_list_broad_v3.json`` outside the source tree whose part has STEP CAD
   (the BROAD candidates, including the Gold11 38172000 X/V scans). Each is
   hashed and compared with the manifest's / list's sha256 (``sha_matches``);
   a mismatching file is listed and never used. Its family is the manifest part
   id. Files whose bytes equal a source-tree entry become aliases of it.

The inventory hash (canonical sha256 of the sorted ``(sha256, canonical path)``
pairs of both parts) seeds the DEV-CHECK draw (:mod:`.cases`), so the draw is a
function of inputs fixed before it.
"""
from __future__ import annotations

import os
import re
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..plyio import read_ply_header
from .common import (
    BENCHMARK_MANIFEST,
    BROAD_CASE_LIST,
    SOURCE_ASSETS,
    consumed_allowlist,
    custody_check,
    doc_sha256,
    expand_home,
    family_of,
    now_utc,
    read_json,
    rel_to_repo,
    sha256_file,
)

__all__ = ["INVENTORY_SCHEMA", "STEP_SUFFIXES", "SCAN_SUFFIXES", "build_inventory", "scan_header", "unique_entries",
           "external_entries", "all_entries", "entry_path"]

INVENTORY_SCHEMA = "sherman_e4b_source_inventory_v2"
STEP_SUFFIXES = (".stp", ".step")
SCAN_SUFFIXES = (".ply", ".stl")
_PART = re.compile(r"^\d{8}$")


def scan_header(path: Path) -> dict[str, Any]:
    """Cheap header facts of a PLY or STL scan (no geometry is read)."""
    suffix = path.suffix.lower()
    if suffix == ".ply":
        h = read_ply_header(path)
        return {"format": {"ascii": "PLY_ASCII", "binary_little_endian": "PLY_BINARY_LE",
                           "binary_big_endian": "PLY_BINARY_BE"}.get(h.fmt, h.fmt),
                "vertex_count": h.count("vertex"), "face_count": h.count("face"),
                "comments": list(h.comments)[:4]}
    with open(path, "rb") as fh:
        head = fh.read(84)
    size = path.stat().st_size
    if len(head) == 84:
        (n,) = struct.unpack_from("<I", head, 80)
        if 84 + 50 * n == size:
            label = head[:80].split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
            return {"format": "STL_BINARY", "vertex_count": None, "face_count": int(n), "comments": [label][:1]}
    return {"format": "STL_ASCII", "vertex_count": None, "face_count": None, "comments": []}


def _kind(path: Path) -> str:
    s = path.suffix.lower()
    if s in STEP_SUFFIXES:
        return "STEP"
    if s in SCAN_SUFFIXES:
        return "SCAN"
    return "OTHER"


def _source_entries(root: Path) -> tuple[list[dict[str, Any]], int]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    by_sha: dict[str, list[Path]] = {}
    for p in files:
        custody_check(p, "inventory")
        by_sha.setdefault(sha256_file(p), []).append(p)
    entries: list[dict[str, Any]] = []
    for sha, paths in by_sha.items():
        rels = sorted((p.relative_to(root).as_posix() for p in paths), key=lambda r: (len(r), r))
        canonical = root / rels[0]
        entry: dict[str, Any] = {
            "sha256": sha, "rel_path": rels[0], "aliases": rels[1:], "kind": _kind(canonical),
            "suffix": canonical.suffix.lower(), "bytes": canonical.stat().st_size, "family": family_of(canonical.name),
            "origin": "SOURCE_ASSETS",
        }
        if entry["kind"] == "SCAN":
            try:
                entry["header"] = scan_header(canonical)
            except (OSError, ValueError) as exc:
                entry["header"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        entries.append(entry)
    entries.sort(key=lambda e: e["rel_path"])
    return entries, len(files)


def external_entries(source_entries: list[dict[str, Any]], *, manifest: Path = BENCHMARK_MANIFEST,
                     case_list: Path = BROAD_CASE_LIST, source_root: Path = SOURCE_ASSETS) -> list[dict[str, Any]]:
    """The external CONSUMED files (module docstring), hashed; source-tree duplicates become aliases."""
    by_sha = {e["sha256"]: e for e in source_entries}
    out: dict[str, dict[str, Any]] = {}
    wanted: list[tuple[Path, str, dict[str, Any]]] = []
    for path, rec in sorted(consumed_allowlist(manifest).items()):
        p = Path(path)
        if rec["kind"] == "STEP" and not str(p).startswith(str(source_root)):
            wanted.append((p, "STEP", {"family": rec["part_id"] if _PART.match(str(rec["part_id"])) else None,
                                       "part_id": rec["part_id"], "expected_sha256": rec.get("sha256"),
                                       "source": "BENCHMARK_V2 CONSUMED part CAD"}))
    cl = read_json(case_list)
    for c in cl.get("cases") or []:
        if not c.get("cad_path") or not str(c["cad_path"]).lower().endswith(STEP_SUFFIXES):
            continue
        p = expand_home(c["scan_path"])
        if str(p).startswith(str(source_root)):
            continue
        wanted.append((p, "SCAN" if p.suffix.lower() in SCAN_SUFFIXES else "OTHER",
                       {"family": c["family_id"], "part_id": c["family_id"], "expected_sha256": c["scan_sha256"],
                        "broad_case_id": c["case_id"], "source": "case_list_broad_v3 CONSUMED scan with STEP CAD"}))
    for p, kind, meta in wanted:
        if not p.is_file():
            out[str(p)] = {"sha256": None, "rel_path": rel_to_repo(p), "aliases": [], "kind": kind,
                           "suffix": p.suffix.lower(), "origin": "EXTERNAL_CONSUMED", "missing": True,
                           "sha_matches": False, **meta}
            continue
        sha = sha256_file(custody_check(p, "inventory (external CONSUMED)"))
        if sha in by_sha:
            alias = rel_to_repo(p)
            if alias not in by_sha[sha].setdefault("external_aliases", []):
                by_sha[sha]["external_aliases"].append(alias)
            continue
        entry = {"sha256": sha, "rel_path": rel_to_repo(p), "aliases": [], "kind": kind, "suffix": p.suffix.lower(),
                 "bytes": p.stat().st_size, "origin": "EXTERNAL_CONSUMED", "abs_path": str(p),
                 "sha_matches": sha == meta.get("expected_sha256"), **meta}
        if kind == "SCAN":
            try:
                entry["header"] = scan_header(p)
            except (OSError, ValueError) as exc:
                entry["header"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        prev = next((e for e in out.values() if e.get("sha256") == sha), None)
        if prev is not None:
            prev["aliases"].append(entry["rel_path"])
            continue
        out[str(p)] = entry
    return sorted(out.values(), key=lambda e: e["rel_path"])


def build_inventory(source_root: str | os.PathLike = SOURCE_ASSETS, *, include_external: bool = True,
                    manifest: Path = BENCHMARK_MANIFEST, case_list: Path = BROAD_CASE_LIST) -> dict[str, Any]:
    """Hash and classify the authorized inputs (module docstring)."""
    root = Path(source_root).resolve()
    entries, files_on_disk = _source_entries(root)
    ext = external_entries(entries, manifest=manifest, case_list=case_list, source_root=root) if include_external \
        else []
    pairs = [[e["sha256"], e["rel_path"]] for e in entries]
    pairs += [[e["sha256"], e["rel_path"]] for e in ext if e.get("sha256")]
    digest = doc_sha256(sorted(pairs))
    counts = {k: sum(1 for e in entries if e["kind"] == k) for k in ("STEP", "SCAN", "OTHER")}
    ext_counts = {k: sum(1 for e in ext if e["kind"] == k and e.get("sha_matches")) for k in ("STEP", "SCAN", "OTHER")}
    return {
        "schema": INVENTORY_SCHEMA,
        "created_at": now_utc(),
        "source_root": "../source-assets/02_CAD_QC_3D",
        "custody": "plan §9 source tree + the owner's wave-4 CONSUMED benchmark files (consumed development); "
                   "files hashed, scan headers only",
        "files_on_disk": files_on_disk,
        "unique_files": len(entries),
        "unique_counts": counts,
        "source_inventory_sha256": doc_sha256(sorted([e["sha256"], e["rel_path"]] for e in entries)),
        "external_counts": ext_counts,
        "external_mismatched_or_missing": [e["rel_path"] for e in ext if not e.get("sha_matches")],
        "inventory_sha256": digest,
        "inventory_sha256_definition": "canonical sha256 of sorted [sha256, canonical rel_path] pairs of the source "
                                       "entries and the hashed external entries",
        "inputs": {"benchmark_manifest": {"path": rel_to_repo(manifest), "sha256": sha256_file(manifest)},
                   "broad_case_list": {"path": rel_to_repo(case_list), "sha256": sha256_file(case_list)}},
        "entries": entries,
        "external_entries": ext,
    }


def unique_entries(inventory: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in inventory["entries"] if e["kind"] == kind]


def all_entries(inventory: Mapping[str, Any], kind: str | None = None) -> list[dict[str, Any]]:
    """Source and usable external entries (external ones only when their sha256 matched)."""
    out = list(inventory["entries"]) + [e for e in inventory.get("external_entries") or [] if e.get("sha_matches")]
    return [e for e in out if kind is None or e["kind"] == kind]


def entry_path(entry: Mapping[str, Any], source_root: Path = SOURCE_ASSETS) -> Path:
    """Absolute path of an inventory entry."""
    if entry.get("origin") == "EXTERNAL_CONSUMED":
        return Path(entry["abs_path"])
    return Path(source_root) / entry["rel_path"]
