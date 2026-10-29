#!/usr/bin/env python3
"""Discover and optionally seal a daily raw-scan delivery.

This is a custody boundary, not a downloader.  It reads only explicitly named
local/sync roots, explicitly named known-corpus roots, a sealed experiment
freeze, and (optionally) a caller-supplied external message/link inventory.
It never opens a URL or contacts mail/cloud services.

Eligible PLY/STL bytes are ordered by SHA-256, byte size, and a deterministic
sealed member ordinal, then copied unchanged to opaque ``Txxx`` filenames.
The emitted scan-only manifest contains no source name, message, link, part,
label, expected-count, CAD, or prediction fields.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, datetime, time as datetime_time
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DISCOVERY_SCHEMA = "SHERMAN_QC_DAILY_RAW_SCAN_DELIVERY_DISCOVERY_V2"
EXTERNAL_KIND = "sherman_qc_external_daily_delivery_evidence_v1"
SCAN_ONLY_KIND = "sherman_qc_f1_v51_opaque_scan_only_development_input"
SCAN_SUFFIXES = frozenset({".ply", ".stl"})
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
EXTERNAL_TOP_KEYS = frozenset(
    {"schema_version", "kind", "cohort_date", "timezone", "inventory", "observations"}
)
INVENTORY_KEYS = frozenset({"captured_at", "inventory_sha256", "members"})
INVENTORY_MEMBER_KEYS = frozenset(
    {"member_name_sha256", "link_sha256", "content_sha256", "local_path"}
)
OBSERVATION_KEYS = frozenset(
    {"observed_at", "message_id_sha256", "link_sha256", "local_path"}
)

# Exact directory components are pruned.  Filename stems are tokenized before
# comparison, so a legitimate substring such as "golden" is not rejected.
DERIVED_PATH_COMPONENTS = frozenset(
    {
        "artifact",
        "artifacts",
        "cache",
        "caches",
        "derived",
        "evidence",
        "fixture",
        "fixtures",
        "generated",
        "gold",
        "label",
        "labels",
        "output",
        "outputs",
        "prediction",
        "predictions",
        "render",
        "renders",
        "synthetic",
        "synthetics",
        "visualization",
        "visualizations",
    }
)
DERIVED_FILENAME_TOKENS = frozenset(
    {
        "aligned",
        "augmented",
        "derived",
        "downsampled",
        "filtered",
        "fixture",
        "generated",
        "gold",
        "label",
        "normalized",
        "prediction",
        "reconstructed",
        "registered",
        "render",
        "synthetic",
    }
)
CONTENT_MARKERS = (
    b"synthetic",
    b"fixture",
    b"ground truth",
    b"prediction",
    b"derived artifact",
    b"aligned scan",
    b"registered scan",
    b"normalized scan",
    b"downsampled scan",
    b"augmented scan",
    b"reconstructed scan",
)


class DiscoveryError(RuntimeError):
    """The discovery/materialization contract could not be satisfied."""


@dataclass(frozen=True)
class ExternalReference:
    source: str
    message_id_sha256: str | None = None
    member_name_sha256: str | None = None
    link_sha256: str | None = None
    declared_content_sha256: str | None = None


@dataclass
class SourceCandidate:
    path: Path
    root_id: str
    relative_path: str
    locator_sha256: str
    member_ordinal: int = 0
    filesystem_mtime_in_window: bool = False
    references: list[ExternalReference] = field(default_factory=list)
    scan_sha256: str | None = None
    byte_count: int | None = None
    extension: str | None = None
    stat_identity: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class KnownCorpus:
    content_sha256s: frozenset[str]
    name_to_content_sha256s: Mapping[str, frozenset[str]]
    geometry_file_count: int


@dataclass
class DiscoveryPlan:
    manifest: dict[str, Any]
    eligible: list[SourceCandidate]
    freeze_path: Path
    freeze_sha256: str
    external_manifest_path: Path | None
    external_manifest_sha256: str | None
    blocking_issue_count: int


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DiscoveryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiscoveryError(f"cannot read canonical JSON input: {path}") from exc


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DiscoveryError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64_RE.fullmatch(value) is not None


def _atomic_write(path: Path, data: bytes, *, replace: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if not path.is_file() or path.read_bytes() != data:
            raise DiscoveryError(f"existing artifact conflicts with sealed bytes: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise DiscoveryError("cohort_date must be YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise DiscoveryError("cohort_date must be canonical YYYY-MM-DD")
    return parsed


def _parse_timezone(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except (TypeError, ZoneInfoNotFoundError) as exc:
        raise DiscoveryError(f"unknown IANA timezone: {value}") from exc


def _parse_aware_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise DiscoveryError(f"{field_name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DiscoveryError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DiscoveryError(f"{field_name} must include a UTC offset")
    return parsed


def _normalize_roots(values: Sequence[Path], field_name: str) -> list[Path]:
    if not values:
        raise DiscoveryError(f"at least one {field_name} is required")
    resolved: list[Path] = []
    for value in values:
        path = value.expanduser().resolve()
        if not path.is_dir():
            raise DiscoveryError(f"{field_name} is not a directory: {value}")
        if path not in resolved:
            resolved.append(path)
    resolved.sort(key=lambda item: str(item))
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if _is_under(left, right) or _is_under(right, left):
                raise DiscoveryError(f"overlapping {field_name} values are forbidden")
    return resolved


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _ensure_disjoint(discovery_roots: Sequence[Path], known_roots: Sequence[Path]) -> None:
    for discovery in discovery_roots:
        for known in known_roots:
            if _is_under(discovery, known) or _is_under(known, discovery):
                raise DiscoveryError(
                    "discovery roots and known-corpus roots must be disjoint"
                )


def _walk_geometry(root: Path, *, prune_derived: bool) -> tuple[list[Path], int, int]:
    files: list[Path] = []
    pruned_derived = 0
    pruned_symlink = 0
    for directory, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        filenames.sort()
        kept: list[str] = []
        for name in directories:
            candidate = Path(directory) / name
            if candidate.is_symlink():
                pruned_symlink += 1
                continue
            if prune_derived and name.casefold() in DERIVED_PATH_COMPONENTS:
                pruned_derived += 1
                continue
            kept.append(name)
        directories[:] = kept
        for name in filenames:
            if Path(name).suffix.casefold() in SCAN_SUFFIXES:
                files.append(Path(directory) / name)
    files.sort(key=lambda item: item.relative_to(root).as_posix())
    return files, pruned_derived, pruned_symlink


def _path_has_derived_marker(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    components = [root.name.casefold(), *(part.casefold() for part in relative.parts[:-1])]
    if any(component in DERIVED_PATH_COMPONENTS for component in components):
        return True
    stem_tokens = {
        token for token in re.split(r"[^a-z0-9]+", relative.stem.casefold()) if token
    }
    return bool(stem_tokens & DERIVED_FILENAME_TOKENS)


def _path_has_symlink_component(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _root_for_path(path: Path, roots: Sequence[Path]) -> tuple[str, Path, str] | None:
    for index, root in enumerate(roots, start=1):
        if _is_under(path, root):
            return f"R{index:03d}", root, path.relative_to(root).as_posix()
    return None


def _source_locator(root_id: str, relative_path: str) -> str:
    return _sha_bytes(
        canonical_json_bytes({"root_id": root_id, "relative_path": relative_path})
    )


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _validate_ply(path: Path) -> tuple[bool, str | None]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(1024 * 1024 + 1)
    except OSError:
        return False, "UNREADABLE"
    if len(prefix) > 1024 * 1024:
        return False, "PLY_HEADER_TOO_LARGE"
    normalized = prefix.replace(b"\r\n", b"\n")
    marker = b"end_header\n"
    end = normalized.find(marker)
    if end < 0 and normalized.endswith(b"end_header"):
        end = len(normalized) - len(b"end_header")
        header_bytes = normalized
    elif end >= 0:
        header_bytes = normalized[: end + len(marker)]
    else:
        return False, "INVALID_PLY_HEADER"
    try:
        lines = header_bytes.decode("ascii").splitlines()
    except UnicodeDecodeError:
        return False, "NON_ASCII_PLY_HEADER"
    if not lines or lines[0].strip() != "ply":
        return False, "INVALID_PLY_MAGIC"
    format_rows = [line.split() for line in lines if line.startswith("format ")]
    if len(format_rows) != 1 or format_rows[0] not in (
        ["format", "ascii", "1.0"],
        ["format", "binary_little_endian", "1.0"],
        ["format", "binary_big_endian", "1.0"],
    ):
        return False, "UNSUPPORTED_PLY_FORMAT"
    vertex_count: int | None = None
    current_element: str | None = None
    vertex_properties: set[str] = set()
    header_metadata: list[bytes] = []
    for line in lines[1:]:
        fields = line.split()
        if not fields:
            continue
        if fields[0] in {"comment", "obj_info"}:
            header_metadata.append(line.casefold().encode("ascii", errors="ignore"))
        if fields[0] == "element" and len(fields) == 3:
            current_element = fields[1]
            if current_element == "vertex":
                try:
                    vertex_count = int(fields[2])
                except ValueError:
                    return False, "INVALID_PLY_VERTEX_COUNT"
        elif fields[0] == "property" and current_element == "vertex" and len(fields) >= 3:
            vertex_properties.add(fields[-1])
    if vertex_count is None or vertex_count <= 0:
        return False, "EMPTY_OR_MISSING_PLY_VERTICES"
    if not {"x", "y", "z"}.issubset(vertex_properties):
        return False, "PLY_MISSING_XYZ_PROPERTIES"
    metadata = b"\n".join(header_metadata)
    if any(marker in metadata for marker in CONTENT_MARKERS):
        return False, "SYNTHETIC_OR_DERIVED_CONTENT_MARKER"
    if path.stat().st_size <= len(header_bytes):
        return False, "PLY_HAS_NO_VERTEX_PAYLOAD"
    return True, None


def _validate_stl(path: Path) -> tuple[bool, str | None]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(min(size, 1024 * 1024))
            tail = b""
            if size > len(prefix):
                handle.seek(max(0, size - 1024))
                tail = handle.read(1024)
    except OSError:
        return False, "UNREADABLE"
    if size >= 84:
        triangle_count = struct.unpack("<I", prefix[80:84])[0]
        if triangle_count > 0 and 84 + 50 * triangle_count == size:
            if any(marker in prefix[:80].lower() for marker in CONTENT_MARKERS):
                return False, "SYNTHETIC_OR_DERIVED_CONTENT_MARKER"
            return True, None
    lowered = prefix.lower()
    ending = (tail or prefix).lower()
    if not lowered.lstrip().startswith(b"solid"):
        return False, "INVALID_STL_MAGIC_OR_LENGTH"
    header_line = lowered.lstrip().splitlines()[0] if lowered.lstrip() else b""
    if any(marker in header_line for marker in CONTENT_MARKERS):
        return False, "SYNTHETIC_OR_DERIVED_CONTENT_MARKER"
    required = (b"facet normal", b"outer loop", b"vertex", b"endfacet")
    if not all(token in lowered for token in required) or b"endsolid" not in ending:
        return False, "INVALID_ASCII_STL_STRUCTURE"
    if lowered.count(b"vertex") < 3:
        return False, "EMPTY_ASCII_STL"
    return True, None


def _validate_raw_geometry(path: Path) -> tuple[bool, str | None]:
    if path.suffix.casefold() == ".ply":
        return _validate_ply(path)
    if path.suffix.casefold() == ".stl":
        return _validate_stl(path)
    return False, "UNSUPPORTED_SCAN_EXTENSION"


def _hash_known_corpus(roots: Sequence[Path]) -> KnownCorpus:
    hashes: set[str] = set()
    names: dict[str, set[str]] = {}
    file_count = 0
    for root in roots:
        files, _pruned_derived, pruned_symlink = _walk_geometry(root, prune_derived=False)
        if pruned_symlink:
            raise DiscoveryError(
                "known-corpus root contains a pruned symlink directory; hash inventory is incomplete"
            )
        for path in files:
            if _path_has_symlink_component(root, path) or not path.is_file():
                raise DiscoveryError(
                    "known-corpus geometry must be a regular non-symlink file"
                )
            digest = _sha_path(path)
            hashes.add(digest)
            name_digest = _sha_bytes(path.name.encode("utf-8"))
            names.setdefault(name_digest, set()).add(digest)
            file_count += 1
    return KnownCorpus(
        content_sha256s=frozenset(hashes),
        name_to_content_sha256s={
            key: frozenset(value) for key, value in sorted(names.items())
        },
        geometry_file_count=file_count,
    )


def _validate_inventory_digest(members: Sequence[Mapping[str, Any]], expected: str) -> None:
    rows = sorted(
        (
            {
                "link_sha256": row["link_sha256"],
                "member_name_sha256": row["member_name_sha256"],
            }
            for row in members
        ),
        key=lambda row: (row["member_name_sha256"], row["link_sha256"]),
    )
    observed = _sha_bytes(canonical_json_bytes(rows))
    if observed != expected:
        raise DiscoveryError("external inventory_sha256 does not bind its name/link members")


def _load_external_manifest(
    path: Path | None,
    *,
    cohort_date: date,
    timezone_name: str,
    timezone: ZoneInfo,
) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    resolved = path.expanduser().resolve()
    payload = _read_json(resolved)
    if not isinstance(payload, dict) or set(payload) != EXTERNAL_TOP_KEYS:
        raise DiscoveryError("external manifest contains unknown or missing top-level fields")
    if payload["schema_version"] != 1 or payload["kind"] != EXTERNAL_KIND:
        raise DiscoveryError("external manifest schema/kind mismatch")
    if payload["cohort_date"] != cohort_date.isoformat() or payload["timezone"] != timezone_name:
        raise DiscoveryError("external manifest cohort date/timezone mismatch")
    observations = payload["observations"]
    if not isinstance(observations, list):
        raise DiscoveryError("external observations must be a list")
    seen_observations: set[bytes] = set()
    for index, row in enumerate(observations):
        if not isinstance(row, dict) or set(row) != OBSERVATION_KEYS:
            raise DiscoveryError(f"external observations[{index}] fields mismatch")
        timestamp = _parse_aware_datetime(row["observed_at"], "observed_at")
        if timestamp.astimezone(timezone).date() != cohort_date:
            raise DiscoveryError("external observation falls outside the cohort date")
        if not _is_hex64(row["message_id_sha256"]):
            raise DiscoveryError("message_id_sha256 must be lowercase hex64")
        if row["link_sha256"] is not None and not _is_hex64(row["link_sha256"]):
            raise DiscoveryError("link_sha256 must be null or lowercase hex64")
        if row["local_path"] is not None and not isinstance(row["local_path"], str):
            raise DiscoveryError("observation local_path must be null or a string")
        identity = canonical_json_bytes(row)
        if identity in seen_observations:
            raise DiscoveryError("duplicate external observation")
        seen_observations.add(identity)
    inventory = payload["inventory"]
    if inventory is not None:
        if not isinstance(inventory, dict) or set(inventory) != INVENTORY_KEYS:
            raise DiscoveryError("external inventory fields mismatch")
        captured = _parse_aware_datetime(inventory["captured_at"], "captured_at")
        if captured.astimezone(timezone).date() != cohort_date:
            raise DiscoveryError("external inventory capture falls outside the cohort date")
        if not _is_hex64(inventory["inventory_sha256"]):
            raise DiscoveryError("inventory_sha256 must be lowercase hex64")
        members = inventory["members"]
        if not isinstance(members, list):
            raise DiscoveryError("external inventory members must be a list")
        seen_members: set[tuple[str, str]] = set()
        for index, row in enumerate(members):
            if not isinstance(row, dict) or set(row) != INVENTORY_MEMBER_KEYS:
                raise DiscoveryError(f"external inventory members[{index}] fields mismatch")
            if not _is_hex64(row["member_name_sha256"]) or not _is_hex64(row["link_sha256"]):
                raise DiscoveryError("inventory name/link hashes must be lowercase hex64")
            if row["content_sha256"] is not None and not _is_hex64(row["content_sha256"]):
                raise DiscoveryError("inventory content_sha256 must be null or lowercase hex64")
            if row["local_path"] is not None and not isinstance(row["local_path"], str):
                raise DiscoveryError("inventory local_path must be null or a string")
            key = (row["member_name_sha256"], row["link_sha256"])
            if key in seen_members:
                raise DiscoveryError("duplicate external name/link inventory member")
            seen_members.add(key)
        _validate_inventory_digest(members, inventory["inventory_sha256"])
    before = _sha_path(resolved)
    # Re-read through the hash path so mutation during parsing is detected.
    after = _sha_path(resolved)
    if before != after:
        raise DiscoveryError("external manifest changed during validation")
    return payload, before


def _validate_freeze(path: Path, cohort_date: date, timezone_name: str) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    before = _sha_path(resolved)
    payload = _read_json(resolved)
    if not isinstance(payload, dict):
        raise DiscoveryError("freeze packet must be a JSON object")
    if payload.get("local_cohort_date") != cohort_date.isoformat():
        raise DiscoveryError("freeze packet cohort date mismatch")
    if payload.get("local_timezone") != timezone_name:
        raise DiscoveryError("freeze packet timezone mismatch")
    if not str(payload.get("status", "")).startswith("SEALED_BEFORE_"):
        raise DiscoveryError("freeze packet is not sealed before discovery")
    after = _sha_path(resolved)
    if before != after:
        raise DiscoveryError("freeze packet changed during validation")
    return payload, before


def _external_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise DiscoveryError("external local_path must be absolute")
    # abspath is lexical and deliberately does not conceal a symlink component.
    return Path(os.path.abspath(candidate))


def build_discovery_plan(
    *,
    cohort_date_value: str,
    timezone_name: str,
    discovery_roots: Sequence[Path],
    known_corpus_roots: Sequence[Path],
    freeze_path: Path,
    external_manifest_path: Path | None = None,
    known_inventory_digests: Iterable[str] = (),
    materialize_requested: bool = False,
) -> DiscoveryPlan:
    """Build a deterministic plan without copying or writing any artifact."""

    cohort_date = _parse_date(cohort_date_value)
    timezone = _parse_timezone(timezone_name)
    roots = _normalize_roots(discovery_roots, "discovery root")
    known_roots = _normalize_roots(known_corpus_roots, "known-corpus root")
    _ensure_disjoint(roots, known_roots)
    known_inventory = frozenset(known_inventory_digests)
    if any(not _is_hex64(value) for value in known_inventory):
        raise DiscoveryError("known inventory digests must be lowercase hex64")

    _freeze_payload, freeze_sha = _validate_freeze(
        freeze_path, cohort_date, timezone_name
    )
    external, external_sha = _load_external_manifest(
        external_manifest_path,
        cohort_date=cohort_date,
        timezone_name=timezone_name,
        timezone=timezone,
    )
    known = _hash_known_corpus(known_roots)
    start = datetime.combine(cohort_date, datetime_time.min, tzinfo=timezone)
    end = datetime.combine(cohort_date.fromordinal(cohort_date.toordinal() + 1), datetime_time.min, tzinfo=timezone)

    candidates: dict[Path, SourceCandidate] = {}
    rejected: list[dict[str, Any]] = []
    pruned_derived_total = 0
    pruned_symlink_total = 0
    geometry_file_count = 0

    def add_rejection(locator: str, reason: str, *, blocking: bool = False) -> None:
        rejected.append(
            {
                "blocking": blocking,
                "reason": reason,
                "source_locator_sha256": locator,
            }
        )

    for root_index, root in enumerate(roots, start=1):
        root_id = f"R{root_index:03d}"
        files, pruned_derived, pruned_symlink = _walk_geometry(root, prune_derived=True)
        pruned_derived_total += pruned_derived
        pruned_symlink_total += pruned_symlink
        geometry_file_count += len(files)
        for path in files:
            relative = path.relative_to(root).as_posix()
            locator = _source_locator(root_id, relative)
            if _path_has_symlink_component(root, path) or not path.is_file():
                add_rejection(locator, "SYMLINK_OR_NONREGULAR")
                continue
            if _path_has_derived_marker(root, path):
                add_rejection(locator, "DERIVED_OR_SYNTHETIC_PATH")
                continue
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone)
            except OSError:
                add_rejection(locator, "UNREADABLE")
                continue
            candidates[path] = SourceCandidate(
                path=path,
                root_id=root_id,
                relative_path=relative,
                locator_sha256=locator,
                filesystem_mtime_in_window=start <= mtime < end,
            )

    external_classifications: list[dict[str, Any]] = []
    external_candidate_paths: set[Path] = set()
    inventory_status = "NOT_SUPPLIED"
    inventory_sha: str | None = None
    if external is not None:
        for row in external["observations"]:
            if row["local_path"] is None:
                continue
            path = _external_path(row["local_path"])
            root_match = _root_for_path(path, roots)
            locator = _sha_bytes(canonical_json_bytes({"external_local_path": str(path)}))
            if root_match is None:
                add_rejection(locator, "EXTERNAL_CANDIDATE_OUTSIDE_ROOTS", blocking=True)
                continue
            root_id, root, relative = root_match
            locator = _source_locator(root_id, relative)
            external_candidate_paths.add(path)
            candidate = candidates.get(path)
            if candidate is None:
                candidate = SourceCandidate(path, root_id, relative, locator)
                candidates[path] = candidate
            candidate.references.append(
                ExternalReference(
                    source="message_observation",
                    message_id_sha256=row["message_id_sha256"],
                    link_sha256=row["link_sha256"],
                )
            )
        inventory = external["inventory"]
        if inventory is not None:
            inventory_sha = inventory["inventory_sha256"]
            inventory_status = (
                "UNCHANGED_KNOWN_NAME_LINK_INVENTORY"
                if inventory_sha in known_inventory
                else "UNRECOGNIZED_NAME_LINK_INVENTORY"
            )
            for row in inventory["members"]:
                classification: dict[str, Any] = {
                    "content_sha256": row["content_sha256"],
                    "link_sha256": row["link_sha256"],
                    "member_name_sha256": row["member_name_sha256"],
                }
                declared = row["content_sha256"]
                if inventory_sha in known_inventory:
                    classification["classification"] = "UNCHANGED_KNOWN_NAME_LINK_MEMBER"
                elif declared in known.content_sha256s:
                    classification["classification"] = "DECLARED_KNOWN_CORPUS_CONTENT_SHA256"
                elif row["member_name_sha256"] in known.name_to_content_sha256s:
                    classification["classification"] = "KNOWN_NAME_CONTENT_UNVERIFIED"
                else:
                    classification["classification"] = "NOVEL_OR_UNVERIFIED_EXTERNAL_MEMBER"
                if row["local_path"] is not None:
                    path = _external_path(row["local_path"])
                    root_match = _root_for_path(path, roots)
                    locator = _sha_bytes(canonical_json_bytes({"external_local_path": str(path)}))
                    if root_match is None:
                        add_rejection(locator, "EXTERNAL_CANDIDATE_OUTSIDE_ROOTS", blocking=True)
                    else:
                        root_id, root, relative = root_match
                        locator = _source_locator(root_id, relative)
                        external_candidate_paths.add(path)
                        candidate = candidates.get(path)
                        if candidate is None:
                            candidate = SourceCandidate(path, root_id, relative, locator)
                            candidates[path] = candidate
                        candidate.references.append(
                            ExternalReference(
                                source="name_link_inventory",
                                member_name_sha256=row["member_name_sha256"],
                                link_sha256=row["link_sha256"],
                                declared_content_sha256=declared,
                            )
                        )
                elif inventory_sha not in known_inventory and declared not in known.content_sha256s:
                    reason = (
                        "EXTERNAL_KNOWN_NAME_CONTENT_UNVERIFIED"
                        if row["member_name_sha256"] in known.name_to_content_sha256s
                        else "EXTERNAL_NOVEL_MEMBER_NOT_MATERIALIZED"
                    )
                    locator = _sha_bytes(
                        canonical_json_bytes(
                            {
                                "link_sha256": row["link_sha256"],
                                "member_name_sha256": row["member_name_sha256"],
                            }
                        )
                    )
                    add_rejection(locator, reason, blocking=True)
                external_classifications.append(classification)

    # Member ordinal is independent of mtimes, external manifest row order, and
    # later content inspection.  It is used only as an exact-byte tie breaker.
    for ordinal, candidate in enumerate(
        sorted(candidates.values(), key=lambda row: (row.root_id, row.relative_path)),
        start=1,
    ):
        candidate.member_ordinal = ordinal

    eligible: list[SourceCandidate] = []
    already_rejected_locators = {row["source_locator_sha256"] for row in rejected}
    for candidate in sorted(candidates.values(), key=lambda row: row.member_ordinal):
        if candidate.locator_sha256 in already_rejected_locators:
            continue
        externally_attested = bool(candidate.references)
        if not candidate.filesystem_mtime_in_window and not externally_attested:
            add_rejection(candidate.locator_sha256, "OUTSIDE_COHORT_WINDOW")
            continue
        root = roots[int(candidate.root_id[1:]) - 1]
        if not candidate.path.exists():
            add_rejection(
                candidate.locator_sha256,
                "EXTERNAL_CANDIDATE_MISSING",
                blocking=externally_attested,
            )
            continue
        if _path_has_symlink_component(root, candidate.path) or not candidate.path.is_file():
            add_rejection(
                candidate.locator_sha256,
                "SYMLINK_OR_NONREGULAR",
                blocking=externally_attested,
            )
            continue
        if _path_has_derived_marker(root, candidate.path):
            add_rejection(
                candidate.locator_sha256,
                "DERIVED_OR_SYNTHETIC_PATH",
                blocking=externally_attested,
            )
            continue
        valid, invalid_reason = _validate_raw_geometry(candidate.path)
        if not valid:
            add_rejection(
                candidate.locator_sha256,
                invalid_reason or "INVALID_RAW_GEOMETRY",
                blocking=externally_attested,
            )
            continue
        try:
            identity_before = _stat_identity(candidate.path)
            digest = _sha_path(candidate.path)
            identity_after = _stat_identity(candidate.path)
        except (OSError, DiscoveryError):
            add_rejection(
                candidate.locator_sha256,
                "UNREADABLE",
                blocking=externally_attested,
            )
            continue
        if identity_before != identity_after:
            add_rejection(candidate.locator_sha256, "SOURCE_CHANGED_DURING_HASH", blocking=True)
            continue
        declared = {
            reference.declared_content_sha256
            for reference in candidate.references
            if reference.declared_content_sha256 is not None
        }
        if declared and declared != {digest}:
            add_rejection(
                candidate.locator_sha256,
                "DECLARED_CONTENT_HASH_MISMATCH",
                blocking=True,
            )
            continue
        candidate.scan_sha256 = digest
        candidate.byte_count = identity_after[2]
        candidate.extension = candidate.path.suffix.casefold()
        candidate.stat_identity = identity_after
        if digest in known.content_sha256s:
            add_rejection(candidate.locator_sha256, "KNOWN_CORPUS_DUPLICATE")
            continue
        eligible.append(candidate)

    eligible.sort(
        key=lambda row: (
            row.scan_sha256 or "",
            row.byte_count or 0,
            row.member_ordinal,
        )
    )
    if len(eligible) > 999:
        raise DiscoveryError("daily opaque intake supports at most 999 eligible scans")

    public_eligible = []
    for ordinal, candidate in enumerate(eligible, start=1):
        reference_rows = sorted(
            (
                {
                    "declared_content_sha256": ref.declared_content_sha256,
                    "link_sha256": ref.link_sha256,
                    "member_name_sha256": ref.member_name_sha256,
                    "message_id_sha256": ref.message_id_sha256,
                    "source": ref.source,
                }
                for ref in candidate.references
            ),
            key=lambda row: canonical_json_bytes(row),
        )
        public_eligible.append(
            {
                "byte_count": candidate.byte_count,
                "case_id": f"T{ordinal:03d}",
                "discovery_channels": sorted(
                    [
                        *( ["filesystem_mtime"] if candidate.filesystem_mtime_in_window else [] ),
                        *( ["external_manifest"] if candidate.references else [] ),
                    ]
                ),
                "extension": candidate.extension,
                "external_references": reference_rows,
                "opaque_filename": f"T{ordinal:03d}{candidate.extension}",
                "scan_sha256": candidate.scan_sha256,
                "sealed_delivery_member_ordinal": candidate.member_ordinal,
                "source_locator_sha256": candidate.locator_sha256,
            }
        )

    rejected.sort(
        key=lambda row: (
            row["source_locator_sha256"], row["reason"], row["blocking"]
        )
    )
    external_classifications.sort(
        key=lambda row: (row["member_name_sha256"], row["link_sha256"])
    )
    blocking_count = sum(bool(row["blocking"]) for row in rejected)
    if blocking_count:
        status = "BLOCKED_FAIL_CLOSED"
        materialization_status = "NOT_AUTHORIZED_BLOCKING_DISCOVERY_ISSUES"
    elif not eligible:
        status = "ZERO_ELIGIBLE_SCANS"
        materialization_status = "NOT_RUN_EMPTY_COHORT"
    elif materialize_requested:
        status = "ELIGIBLE_SCANS_READY"
        materialization_status = "AUTHORIZED_PENDING_ATOMIC_COPY"
    else:
        status = "ELIGIBLE_SCANS_DRY_RUN"
        materialization_status = "NOT_REQUESTED"

    manifest: dict[str, Any] = {
        "cohort_date": cohort_date.isoformat(),
        "eligibility_policy": {
            "allowed_extensions": sorted(SCAN_SUFFIXES),
            "derived_output_synthetic_rejected": True,
            "external_access_performed": False,
            "known_corpus_content_hash_comparison": True,
            "ordering": ["scan_sha256", "byte_count", "sealed_delivery_member_ordinal"],
            "source_scope": "EXPLICIT_ROOTS_AND_OPTIONAL_SUPPLIED_MANIFEST_ONLY",
        },
        "eligible_scan_count": len(public_eligible),
        "eligible_scans": public_eligible,
        "experiment_freeze": {
            "path": str(freeze_path.expanduser().resolve()),
            "sha256": freeze_sha,
        },
        "external_evidence": {
            "inventory": {
                "classifications": external_classifications,
                "inventory_sha256": inventory_sha,
                "status": inventory_status,
            },
            "manifest_sha256": external_sha,
            "observation_count": len(external["observations"]) if external else 0,
            "supplied": external is not None,
        },
        "materialization": {
            "requested": materialize_requested,
            "status": materialization_status,
        },
        "rejected_scan_count": len(rejected),
        "rejections": rejected,
        "schema": DISCOVERY_SCHEMA,
        "schema_version": 2,
        "scope": {
            "discovery_roots": [str(root) for root in roots],
            "known_corpus_roots": [str(root) for root in known_roots],
            "known_inventory_digests": sorted(known_inventory),
            "time_window": f"{start.isoformat()}/{end.isoformat()}",
        },
        "status": status,
        "summary": {
            "blocking_issue_count": blocking_count,
            "geometry_file_count": geometry_file_count,
            "known_corpus_geometry_file_count": known.geometry_file_count,
            "known_corpus_unique_sha256_count": len(known.content_sha256s),
            "pruned_derived_directory_count": pruned_derived_total,
            "pruned_symlink_directory_count": pruned_symlink_total,
        },
        "timezone": timezone_name,
    }
    return DiscoveryPlan(
        manifest=manifest,
        eligible=eligible,
        freeze_path=freeze_path.expanduser().resolve(),
        freeze_sha256=freeze_sha,
        external_manifest_path=(
            external_manifest_path.expanduser().resolve()
            if external_manifest_path is not None
            else None
        ),
        external_manifest_sha256=external_sha,
        blocking_issue_count=blocking_count,
    )


def _verify_plan_inputs_unchanged(plan: DiscoveryPlan) -> None:
    if _sha_path(plan.freeze_path) != plan.freeze_sha256:
        raise DiscoveryError("freeze packet changed after discovery")
    if plan.external_manifest_path is not None:
        if _sha_path(plan.external_manifest_path) != plan.external_manifest_sha256:
            raise DiscoveryError("external manifest changed after discovery")
    for candidate in plan.eligible:
        if candidate.stat_identity is None or candidate.scan_sha256 is None:
            raise DiscoveryError("eligible source is missing its sealed identity")
        try:
            identity = _stat_identity(candidate.path)
        except OSError as exc:
            raise DiscoveryError("eligible source disappeared after discovery") from exc
        if identity != candidate.stat_identity or _sha_path(candidate.path) != candidate.scan_sha256:
            raise DiscoveryError("eligible source changed after discovery")


def _validate_materialization_targets(
    plan: DiscoveryPlan,
    *,
    opaque_root: Path,
    scan_only_manifest: Path,
    discovery_roots: Sequence[Path],
) -> tuple[Path, Path, bytes]:
    opaque = opaque_root.expanduser().resolve()
    scan_manifest = scan_only_manifest.expanduser().resolve()
    for root in discovery_roots:
        resolved = root.expanduser().resolve()
        if _is_under(opaque, resolved) or _is_under(resolved, opaque):
            raise DiscoveryError("opaque root must be disjoint from discovery roots")
    if _is_under(scan_manifest, opaque):
        raise DiscoveryError("scan-only manifest must not be inside the opaque scan root")
    expected_names = {
        f"T{ordinal:03d}{candidate.extension}"
        for ordinal, candidate in enumerate(plan.eligible, start=1)
    }
    if opaque.exists():
        if not opaque.is_dir():
            raise DiscoveryError("opaque root exists but is not a directory")
        observed_names = {entry.name for entry in opaque.iterdir()}
        unexpected = observed_names - expected_names
        if unexpected:
            raise DiscoveryError("opaque root contains unexpected stale artifacts")
    scan_rows = []
    for ordinal, candidate in enumerate(plan.eligible, start=1):
        target = opaque / f"T{ordinal:03d}{candidate.extension}"
        if target.exists():
            if not target.is_file() or target.is_symlink() or _sha_path(target) != candidate.scan_sha256:
                raise DiscoveryError(f"opaque target conflict: {target.name}")
        scan_rows.append(
            {"scan_path": str(target), "scan_sha256": candidate.scan_sha256}
        )
    scan_bytes = canonical_json_bytes(
        {"cases": scan_rows, "kind": SCAN_ONLY_KIND, "schema_version": 1}
    )
    if scan_manifest.exists():
        if not scan_manifest.is_file() or scan_manifest.read_bytes() != scan_bytes:
            raise DiscoveryError("existing scan-only manifest conflicts with discovery")
    return opaque, scan_manifest, scan_bytes


def materialize_plan(
    plan: DiscoveryPlan,
    *,
    opaque_root: Path,
    scan_only_manifest: Path,
    discovery_roots: Sequence[Path],
) -> None:
    """Atomically stage and seal all eligible opaque copies, or copy none."""

    if plan.blocking_issue_count:
        raise DiscoveryError("blocking discovery issues forbid materialization")
    if not plan.eligible:
        opaque = opaque_root.expanduser().resolve()
        scan_manifest = scan_only_manifest.expanduser().resolve()
        if opaque.exists() or scan_manifest.exists():
            raise DiscoveryError("empty discovery conflicts with existing materialized cohort")
        return
    _verify_plan_inputs_unchanged(plan)
    opaque, scan_manifest, scan_bytes = _validate_materialization_targets(
        plan,
        opaque_root=opaque_root,
        scan_only_manifest=scan_only_manifest,
        discovery_roots=discovery_roots,
    )
    opaque.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for ordinal, candidate in enumerate(plan.eligible, start=1):
            target = opaque / f"T{ordinal:03d}{candidate.extension}"
            if target.exists():
                continue
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=opaque
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            shutil.copyfile(candidate.path, temporary)
            if _sha_path(temporary) != candidate.scan_sha256:
                raise DiscoveryError("staged opaque copy hash mismatch")
            staged.append((temporary, target))
        _verify_plan_inputs_unchanged(plan)
        for temporary, target in staged:
            os.replace(temporary, target)
        _atomic_write(scan_manifest, scan_bytes, replace=False)
    finally:
        for temporary, _target in staged:
            if temporary.exists():
                temporary.unlink()


def run_discovery(
    *,
    cohort_date_value: str,
    timezone_name: str,
    discovery_roots: Sequence[Path],
    known_corpus_roots: Sequence[Path],
    freeze_path: Path,
    output_manifest: Path,
    external_manifest_path: Path | None = None,
    known_inventory_digests: Iterable[str] = (),
    materialize: bool = False,
    opaque_root: Path | None = None,
    scan_only_manifest: Path | None = None,
) -> dict[str, Any]:
    plan = build_discovery_plan(
        cohort_date_value=cohort_date_value,
        timezone_name=timezone_name,
        discovery_roots=discovery_roots,
        known_corpus_roots=known_corpus_roots,
        freeze_path=freeze_path,
        external_manifest_path=external_manifest_path,
        known_inventory_digests=known_inventory_digests,
        materialize_requested=materialize,
    )
    if materialize and not plan.blocking_issue_count:
        if opaque_root is None or scan_only_manifest is None:
            raise DiscoveryError(
                "--materialize requires --opaque-root and --scan-only-manifest"
            )
        materialize_plan(
            plan,
            opaque_root=opaque_root,
            scan_only_manifest=scan_only_manifest,
            discovery_roots=discovery_roots,
        )
        if plan.eligible and not plan.blocking_issue_count:
            plan.manifest["status"] = "ELIGIBLE_SCANS_MATERIALIZED"
            plan.manifest["materialization"] = {
                "opaque_root": str(opaque_root.expanduser().resolve()),
                "requested": True,
                "scan_only_manifest": str(scan_only_manifest.expanduser().resolve()),
                "status": "MATERIALIZED_OR_ALREADY_IDENTICAL",
            }
    _verify_plan_inputs_unchanged(plan)
    _atomic_write(
        output_manifest.expanduser().resolve(),
        canonical_json_bytes(plan.manifest),
        replace=True,
    )
    return plan.manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed daily raw scan discovery and opaque materialization"
    )
    parser.add_argument("--cohort-date", required=True)
    parser.add_argument("--timezone", required=True)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--known-corpus-root", action="append", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument("--known-inventory-digest", action="append", default=[])
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--opaque-root", type=Path)
    parser.add_argument("--scan-only-manifest", type=Path)
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=0.0,
        help="sleep between zero-result polls; 0 runs once",
    )
    parser.add_argument("--max-polls", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.poll_interval_seconds < 0 or args.max_polls < 1:
        raise SystemExit("poll interval must be nonnegative and max-polls must be positive")
    if args.max_polls > 1 and args.poll_interval_seconds <= 0:
        raise SystemExit("multiple polls require a positive poll interval")
    if args.materialize and (args.opaque_root is None or args.scan_only_manifest is None):
        raise SystemExit("--materialize requires --opaque-root and --scan-only-manifest")
    try:
        result: dict[str, Any] | None = None
        for poll in range(args.max_polls):
            result = run_discovery(
                cohort_date_value=args.cohort_date,
                timezone_name=args.timezone,
                discovery_roots=args.root,
                known_corpus_roots=args.known_corpus_root,
                freeze_path=args.freeze,
                external_manifest_path=args.external_manifest,
                known_inventory_digests=args.known_inventory_digest,
                output_manifest=args.output_manifest,
                materialize=args.materialize,
                opaque_root=args.opaque_root,
                scan_only_manifest=args.scan_only_manifest,
            )
            if result["eligible_scan_count"] > 0 or result["status"] == "BLOCKED_FAIL_CLOSED":
                break
            if poll + 1 < args.max_polls:
                time.sleep(args.poll_interval_seconds)
        assert result is not None
    except DiscoveryError as exc:
        print(
            json.dumps(
                {"error": str(exc), "schema": DISCOVERY_SCHEMA, "status": "FAIL_CLOSED"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "blocking_issue_count": result["summary"]["blocking_issue_count"],
                "eligible_scan_count": result["eligible_scan_count"],
                "manifest": str(args.output_manifest.expanduser().resolve()),
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
