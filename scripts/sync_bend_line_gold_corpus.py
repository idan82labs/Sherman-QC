#!/usr/bin/env python3
"""Consolidate Sherman bend-line truth and synchronize PLY labels to the studio.

The source label sets intentionally retain their original review tier.  Human-
verified labels are promotion eligible; assistant multiview labels are useful
for development scoring but remain review candidates until a human completes
them in the browser studio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bend_training_labels import (
    LABEL_KIND,
    SCHEMA_VERSION,
    default_asset_roots,
    list_ply_assets,
    load_labels,
    save_labels,
)
from build_additional_line_label_review import _scan_points, line_scan_support


LEGACY_ROOT = ROOT / "output"
SHAKED_MANIFEST = (
    ROOT
    / "output/manual_labels/shaked-count-backed-12-20260717/adjudicated_v1/manifest.json"
)
XV_MANIFEST = (
    ROOT
    / "output/research/f1-new-shaked-pairs-20260730-v1/manual_adjudication/labels/manifest.json"
)
CATALOG_PATH = ROOT / "data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"
STUDIO_LABELS_ROOT = ROOT / "data/bend_corpus/annotations/human_bend_lines_v1"
PLY_PROXY_ROOT = ROOT / "data/bend_corpus/gold_source_ply_v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _point(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three coordinates")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite coordinate")
    return result


def _normalize_bends(rows: Iterable[dict[str, Any]], case_id: str) -> list[dict[str, Any]]:
    bends: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        bend_id = str(row.get("bend_id") or f"B{index:02d}")
        if bend_id in ids:
            raise ValueError(f"{case_id}: duplicate bend id {bend_id}")
        ids.add(bend_id)
        start = _point(row.get("line_start"), f"{case_id}.{bend_id}.line_start")
        end = _point(row.get("line_end"), f"{case_id}.{bend_id}.line_end")
        if start == end:
            raise ValueError(f"{case_id}.{bend_id} is a zero-length line")
        confidence = str(row.get("confidence") or "high").lower()
        bends.append(
            {
                "bend_id": bend_id,
                "line_start": start,
                "line_end": end,
                "confidence": confidence,
                "source_candidate_id": row.get("source_candidate_id"),
            }
        )
    if not bends:
        raise ValueError(f"{case_id}: labels contain no bend lines")
    return bends


def _part_id(value: Any) -> str:
    raw = str(value or "").strip()
    match = re.match(r"^(\d{8})", raw)
    return match.group(1) if match else raw


def _legacy_entries(legacy_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for label_path in sorted(legacy_root.glob("bend_label_*/bend_instance_labels.json")):
        meta_path = label_path.with_name("bend_inspection_meta.json")
        if not meta_path.is_file():
            continue
        labels = _read(label_path)
        metadata = _read(meta_path)
        seed = dict(metadata.get("seed_case") or {})
        source_case_id = str(seed.get("case_id") or labels.get("job_id") or label_path.parent.name)
        case_id = f"corpus/{source_case_id}"
        scan_path = Path(str(seed.get("source_scan") or "")).expanduser().resolve()
        if not scan_path.is_file():
            raise FileNotFoundError(f"{case_id}: source scan missing: {scan_path}")
        bends = _normalize_bends(labels.get("manual_bends") or [], case_id)
        entries.append(
            {
                "case_id": case_id,
                "part_id": _part_id(metadata.get("part_id") or source_case_id.split("_", 1)[-1]),
                "scan_role": "corpus",
                "scan_path": str(scan_path),
                "scan_name": scan_path.name,
                "scan_sha256": _sha256(scan_path),
                "coordinate_frame": str(labels.get("coordinate_frame") or "reference_mesh_raw"),
                "bend_count": len(bends),
                "bends": bends,
                "truth_tier": "authoritative_user_verified",
                "review_status": str(labels.get("review_status") or "complete"),
                "scoring_eligible": True,
                "promotion_eligible": True,
                "source_labels_path": str(label_path.resolve()),
                "source_manifest_path": None,
            }
        )
    return entries


def _manifest_entries(
    manifest_path: Path,
    *,
    tier: str,
    case_key: str,
    part_key: str,
    role_key: str | None,
) -> list[dict[str, Any]]:
    manifest = _read(manifest_path)
    entries: list[dict[str, Any]] = []
    for row in manifest.get("cases") or []:
        label_path = Path(str(row["labels_path"])).expanduser().resolve()
        labels = _read(label_path)
        source_case_id = str(row[case_key])
        namespace = "shaked12" if tier == "assistant_multiview_count_backed" else "shaked_xv"
        case_id = f"{namespace}/{source_case_id}"
        scan_path = Path(str(labels["scan_path"])).expanduser().resolve()
        if not scan_path.is_file():
            raise FileNotFoundError(f"{case_id}: source scan missing: {scan_path}")
        bends = _normalize_bends(labels.get("manual_bends") or [], case_id)
        expected = int(labels.get("sender_bend_count") or row.get("sender_bend_count") or 0)
        if len(bends) != expected:
            raise ValueError(f"{case_id}: {len(bends)} bend lines do not match expected {expected}")
        entries.append(
            {
                "case_id": case_id,
                "part_id": _part_id(row[part_key]),
                "scan_role": str(row.get(role_key) or "scan") if role_key else "scan",
                "scan_path": str(scan_path),
                "scan_name": scan_path.name,
                "scan_sha256": str(labels.get("scan_sha256") or _sha256(scan_path)),
                "coordinate_frame": str(labels.get("coordinate_frame") or "scan_raw"),
                "bend_count": len(bends),
                "bends": bends,
                "truth_tier": (
                    tier.replace("assistant_multiview", "codex_manual_multiview")
                    if labels.get("review_status") == "codex_manual_multiview_complete"
                    else tier
                ),
                "review_status": str(labels.get("review_status") or manifest.get("review_status") or ""),
                "scoring_eligible": True,
                "promotion_eligible": False,
                "source_labels_path": str(label_path),
                "source_manifest_path": str(manifest_path.resolve()),
            }
        )
    return entries


def collect_entries(
    *,
    legacy_root: Path = LEGACY_ROOT,
    shaked_manifest: Path = SHAKED_MANIFEST,
    xv_manifest: Path = XV_MANIFEST,
) -> list[dict[str, Any]]:
    entries = _legacy_entries(legacy_root)
    entries.extend(
        _manifest_entries(
            shaked_manifest,
            tier="assistant_multiview_count_backed",
            case_key="case_id",
            part_key="part_number",
            role_key=None,
        )
    )
    entries.extend(
        _manifest_entries(
            xv_manifest,
            tier="assistant_multiview_drawing_backed",
            case_key="case_token",
            part_key="part_id",
            role_key="role",
        )
    )
    case_ids = [str(entry["case_id"]) for entry in entries]
    if len(case_ids) != len(set(case_ids)):
        duplicates = sorted(case for case, count in Counter(case_ids).items() if count > 1)
        raise ValueError(f"duplicate case ids: {duplicates}")
    return entries


def _asset_paths(project_root: Path, assets: list[dict[str, Any]]) -> dict[Path, dict[str, Any]]:
    roots = default_asset_roots(project_root)
    return {
        (roots[asset["source"]] / asset["relative_path"]).resolve(): asset
        for asset in assets
    }


def _proxy_path(source: Path) -> Path:
    return PLY_PROXY_ROOT / f"{source.stem}__source_frame_proxy.ply"


def materialize_ply_proxies(entries: list[dict[str, Any]]) -> list[str]:
    """Create source-frame PLY proxies only when a gold source is not already PLY."""
    import trimesh

    created: list[str] = []
    for entry in entries:
        source = Path(str(entry["scan_path"]))
        if source.suffix.lower() == ".ply" or source.with_suffix(".ply").is_file():
            continue
        target = _proxy_path(source)
        if target.is_file():
            continue
        loaded = trimesh.load(source, process=False)
        if isinstance(loaded, trimesh.Scene):
            meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
            if not meshes:
                raise ValueError(f"{entry['case_id']}: source scene contains no mesh")
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise ValueError(f"{entry['case_id']}: unsupported mesh type {type(loaded).__name__}")
        target.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(target, file_type="ply", encoding="binary_little_endian")
        created.append(str(target.resolve()))
    return created


def _validated_ply_alias(entry: dict[str, Any]) -> Path | None:
    source = Path(str(entry["scan_path"]))
    if source.suffix.lower() == ".ply":
        return source
    same_stem = source.with_suffix(".ply")
    candidate = same_stem if same_stem.is_file() else _proxy_path(source)
    if not candidate.is_file():
        return None
    points = _scan_points(candidate)
    for bend in entry["bends"]:
        support = line_scan_support(points, bend["line_start"], bend["line_end"])
        if not support["frame_integrity_pass"]:
            return None
    return candidate.resolve()


def attach_studio_assets(
    entries: list[dict[str, Any]],
    *,
    project_root: Path = ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assets = list_ply_assets(project_root, STUDIO_LABELS_ROOT)
    by_path = _asset_paths(project_root, assets)
    mirrors: list[dict[str, Any]] = []
    for entry in entries:
        source = Path(str(entry["scan_path"]))
        studio_path = _validated_ply_alias(entry)
        asset = None if studio_path is None else by_path.get(studio_path.resolve())
        entry["studio"] = {
            "eligible": asset is not None,
            "asset_id": None if asset is None else asset["asset_id"],
            "source": None if asset is None else asset["source"],
            "relative_path": None if asset is None else asset["relative_path"],
            "path": None if studio_path is None else str(studio_path),
            "frame_transfer": (
                "exact_source_ply"
                if studio_path is not None and studio_path == source
                else "lossless_stl_to_ply_source_frame_proxy"
                if studio_path is not None and studio_path == _proxy_path(source)
                else "validated_same_stem_ply_alias"
                if studio_path is not None
                else "unavailable_non_ply_source"
            ),
        }
        if asset is None:
            continue
        if asset["source"] != "source-assets":
            continue
        size = studio_path.stat().st_size
        source_hash = _sha256(studio_path)
        for candidate_path, candidate_asset in by_path.items():
            if candidate_path == studio_path or candidate_asset["source"] != "source-assets":
                continue
            if candidate_path.name != studio_path.name or candidate_path.stat().st_size != size:
                continue
            if _sha256(candidate_path) != source_hash:
                continue
            mirrors.append(
                {
                    "case_id": entry["case_id"],
                    "asset_id": candidate_asset["asset_id"],
                    "source": candidate_asset["source"],
                    "relative_path": candidate_asset["relative_path"],
                    "path": str(candidate_path),
                    "reason": "byte_identical_source_asset_copy",
                }
            )
    return entries, mirrors


def _studio_bends(entry: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for bend in entry["bends"]:
        confidence = "uncertain" if bend.get("confidence") in {"low", "medium"} else "certain"
        output.append(
            {
                "bend_id": bend["bend_id"],
                "line_start": bend["line_start"],
                "line_end": bend["line_end"],
                "confidence": confidence,
                "notes": f"Imported from {entry['truth_tier']}",
            }
        )
    return output


def _same_geometry(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("source", "source_relative_path", "coordinate_frame")
    if not all(left.get(key) == right.get(key) for key in keys):
        return False
    geometry_keys = ("bend_id", "line_start", "line_end")
    left_bends = [tuple(bend.get(key) if key == "bend_id" else tuple(bend.get(key) or ()) for key in geometry_keys) for bend in left.get("bends") or []]
    right_bends = [tuple(bend.get(key) if key == "bend_id" else tuple(bend.get(key) or ()) for key in geometry_keys) for bend in right.get("bends") or []]
    return left_bends == right_bends


def sync_studio(
    entries: list[dict[str, Any]],
    mirrors: list[dict[str, Any]],
    *,
    project_root: Path = ROOT,
    labels_root: Path = STUDIO_LABELS_ROOT,
) -> dict[str, Any]:
    assets = list_ply_assets(project_root, labels_root)
    by_id = {asset["asset_id"]: asset for asset in assets}
    jobs: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    entries_by_case = {str(entry["case_id"]): entry for entry in entries}
    for entry in entries:
        asset_id = entry["studio"]["asset_id"]
        if asset_id:
            jobs.append((entry, by_id[asset_id], "canonical"))
    for mirror in mirrors:
        jobs.append(
            (
                entries_by_case[str(mirror["case_id"])],
                by_id[str(mirror["asset_id"])],
                "mirror",
            )
        )

    saved = 0
    unchanged = 0
    seen_assets: set[str] = set()
    for entry, asset, sync_kind in jobs:
        asset_id = str(asset["asset_id"])
        if asset_id in seen_assets:
            raise ValueError(f"multiple gold cases resolve to studio asset {asset_id}")
        seen_assets.add(asset_id)
        desired = {
            "schema_version": SCHEMA_VERSION,
            "kind": LABEL_KIND,
            "asset_id": asset_id,
            "source": asset["source"],
            "source_relative_path": asset["relative_path"],
            "coordinate_frame": "source_ply_raw",
            "review_status": (
                "complete"
                if entry["promotion_eligible"]
                or entry["review_status"] == "codex_manual_multiview_complete"
                else "needs_review"
            ),
            "annotator": "Codex multiview gold import",
            "revision": int(asset.get("revision") or 0),
            "no_bends_present": False,
            "bends": _studio_bends(entry),
            "notes": (
                f"Gold case {entry['case_id']}; tier={entry['truth_tier']}; "
                f"sync={sync_kind}; provenance={entry['source_labels_path']}"
            ),
        }
        path = labels_root / f"{asset_id}.json"
        if path.is_file():
            current = load_labels(labels_root, asset)
            if not _same_geometry(current, desired):
                raise ValueError(f"refusing to overwrite differing studio labels for {asset_id}")
            managed = str(current.get("annotator") or "").startswith("Codex multiview")
            metadata_changed = any(
                current.get(key) != desired.get(key)
                for key in ("review_status", "annotator", "notes", "bends")
            )
            if managed and metadata_changed:
                desired["revision"] = current["revision"]
                save_labels(labels_root, desired, asset)
                saved += 1
            else:
                unchanged += 1
            continue
        save_labels(labels_root, desired, asset)
        saved += 1
    return {"eligible_jobs": len(jobs), "saved": saved, "unchanged": unchanged}


def build_catalog(entries: list[dict[str, Any]], mirrors: list[dict[str, Any]]) -> dict[str, Any]:
    tier_counts = Counter(str(entry["truth_tier"]) for entry in entries)
    part_ids = sorted({str(entry["part_id"]) for entry in entries})
    studio_entries = [entry for entry in entries if entry["studio"]["eligible"]]
    return {
        "schema_version": 1,
        "kind": "sherman_qc_bend_line_gold_catalog",
        "source_snapshot_sha256": hashlib.sha256(
            "\n".join(
                f"{entry['case_id']}:{entry['scan_sha256']}:{entry['bend_count']}"
                for entry in entries
            ).encode("utf-8")
        ).hexdigest(),
        "semantic_contract": {
            "unit": "one physical conventional bend line",
            "line_geometry": "finite start/end segment in the unaligned source mesh frame",
            "matching": "one-to-one angle, transverse-offset, and finite-span overlap",
            "promotion_policy": (
                "Only authoritative_user_verified is enterprise-promotion eligible. "
                "Assistant multiview tiers are development scoring truth pending human review."
            ),
        },
        "summary": {
            "case_count": len(entries),
            "unique_part_count": len(part_ids),
            "bend_count": sum(int(entry["bend_count"]) for entry in entries),
            "promotion_case_count": sum(bool(entry["promotion_eligible"]) for entry in entries),
            "promotion_bend_count": sum(
                int(entry["bend_count"]) for entry in entries if entry["promotion_eligible"]
            ),
            "development_case_count": sum(not entry["promotion_eligible"] for entry in entries),
            "development_bend_count": sum(
                int(entry["bend_count"]) for entry in entries if not entry["promotion_eligible"]
            ),
            "studio_canonical_case_count": len(studio_entries),
            "studio_canonical_bend_count": sum(int(entry["bend_count"]) for entry in studio_entries),
            "studio_mirror_asset_count": len(mirrors),
            "non_ply_case_count": len(entries) - len(studio_entries),
            "tier_case_counts": dict(sorted(tier_counts.items())),
        },
        "part_ids": part_ids,
        "cases": entries,
        "studio_mirrors": mirrors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--write-studio", action="store_true")
    parser.add_argument("--materialize-ply-proxies", action="store_true")
    args = parser.parse_args()

    entries = collect_entries()
    proxies = materialize_ply_proxies(entries) if args.materialize_ply_proxies else []
    entries, mirrors = attach_studio_assets(entries)
    catalog = build_catalog(entries, mirrors)
    catalog_path = args.catalog.resolve()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    result: dict[str, Any] = {"catalog": str(catalog_path), **catalog["summary"]}
    result["created_ply_proxies"] = proxies
    if args.write_studio:
        result["studio_sync"] = sync_studio(entries, mirrors)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
