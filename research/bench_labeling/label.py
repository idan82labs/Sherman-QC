"""``write-label``: Bend Label Studio data contract, AI-adjudicated variant.

Labels go to ``data/bend_corpus/annotations/ai_bend_lines_v1/<asset_id>.json``
(never ``human_bend_lines_v1``). The payload is the studio's contract
(``backend.bend_training_labels``) with ``review_status='ai_adjudicated'`` and
``annotator='claude-vision-ai-v1'``; metrics and image hashes go to a sidecar
``provenance/<asset_id>.json`` because the contract forbids extra keys.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import AI_ANNOTATOR, AI_REVIEW_STATUS, TOOL_VERSION
from .custody import check_path
from .scanio import load_scan, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.bend_training_labels import (  # noqa: E402
    ASSET_SOURCES,
    COORDINATE_FRAME,
    LABEL_KIND,
    SCHEMA_VERSION,
    BendTrainingLabelError,
    asset_id_for,
    default_asset_roots,
    validate_labels,
)

AI_LABELS_DIRNAME = "ai_bend_lines_v1"
HUMAN_LABELS_DIRNAME = "human_bend_lines_v1"
DEFAULT_LABELS_ROOT = PROJECT_ROOT / "data" / "bend_corpus" / "annotations" / AI_LABELS_DIRNAME


class LabelWriteError(ValueError):
    pass


def resolve_asset_identity(asset_path, project_root=PROJECT_ROOT):
    """Studio asset identity: (source, relative_path, asset_id), same rules as the service."""
    path = check_path(asset_path)
    if path.suffix.lower() != ".ply":
        raise LabelWriteError("the studio inventory indexes .ply files only; label a PLY asset")
    roots = default_asset_roots(Path(project_root))
    for source in ASSET_SOURCES:
        root = roots[source]
        try:
            relative = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        return {"source": source, "relative_path": relative,
                "asset_id": asset_id_for(source, relative), "path": path}
    allowed = ", ".join(f"{s}={roots[s]}" for s in ASSET_SOURCES)
    raise LabelWriteError(
        f"{path.name} is not under a studio asset root ({allowed}); copy it into e.g. "
        "data/bend_corpus/incoming/<batch>/ first so the label has a stable asset id")


def validate_ai_label(payload, asset):
    if payload.get("review_status") != AI_REVIEW_STATUS:
        raise LabelWriteError(f"review_status must be {AI_REVIEW_STATUS}")
    if payload.get("annotator") != AI_ANNOTATOR:
        raise LabelWriteError(f"annotator must be {AI_ANNOTATOR}")
    if not payload.get("bends") and not payload.get("no_bends_present"):
        raise LabelWriteError("an adjudicated label needs bend lines or no_bends_present=true")
    probe = dict(payload, review_status="needs_review")
    try:
        validated = validate_labels(probe, {"asset_id": asset["asset_id"], "source": asset["source"],
                                            "relative_path": asset["relative_path"]})
    except BendTrainingLabelError as exc:
        raise LabelWriteError(f"contract violation: {exc}") from exc
    validated["review_status"] = AI_REVIEW_STATUS
    return validated


def _coordinates_match(lines_doc, asset_path):
    asset_sha = sha256_file(asset_path)
    if asset_sha == lines_doc["scan_sha256"]:
        return "sha256", asset_sha
    from scipy.spatial import cKDTree

    a = load_scan(lines_doc["scan_path"]).points
    b = load_scan(asset_path).points
    if len(a) != len(b):
        raise LabelWriteError("asset is a different file from the labelled scan (point counts differ)")
    dist, _ = cKDTree(a).query(b, k=1)
    if float(dist.max()) > 1e-4:
        raise LabelWriteError("asset coordinates differ from the labelled scan; labels would be "
                              "in the wrong frame")
    return "vertex_set", asset_sha


def _atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _side_text(side):
    return side if isinstance(side, str) else "+".join(map(str, side))


def _bend_note(bend):
    parts = [
        f"patches {_side_text(bend['patches'][0])}|{_side_text(bend['patches'][1])}",
        f"locus {bend['label_locus']}",
        f"incl {bend['included_angle_deg']:.2f} deg",
        f"len {bend['length_mm']:.1f} mm",
        f"skin {bend['skin']}",
        f"cov {bend['coverage']:.2f}",
    ]
    if bend.get("extent_source") == "override":
        parts.append(f"extent override (fold support {bend['override_support_frac']:.2f})")
    if bend.get("confidence_reasons"):
        parts.append("flags: " + "; ".join(bend["confidence_reasons"]))
    if bend.get("note"):
        parts.append("labeler: " + bend["note"])
    return ("; ".join(parts))[:1000]


def build_payload(lines_doc, asset, *, confirm_no_bends=False, include_uncertain=True):
    ok = [b for b in lines_doc["bends"] if b.get("status") == "ok"]
    failed = [b for b in lines_doc["bends"] if b.get("status") != "ok"]
    if failed:
        raise LabelWriteError(
            f"{len(failed)} bend(s) failed in lines ({', '.join(b['bend_id'] for b in failed)}); "
            "fix the pairs and re-run lines before writing a label")
    if not include_uncertain:
        ok = [b for b in ok if b["confidence"] == "certain"]
    no_bends = bool(lines_doc.get("no_bends_present"))
    if no_bends and not confirm_no_bends:
        raise LabelWriteError("zero-bend label requires --confirm-no-bends")
    if not ok and not no_bends:
        raise LabelWriteError("no bend lines to write (and no_bends_present is false)")
    bends = [{
        "bend_id": b["bend_id"],
        "line_start": [round(float(v), 5) for v in b["label_line"]["start"]],
        "line_end": [round(float(v), 5) for v in b["label_line"]["end"]],
        "confidence": b["confidence"],
        "notes": _bend_note(b),
    } for b in ok]
    notes = (f"AI-vision label ({AI_ANNOTATOR}, {TOOL_VERSION}); locus={lines_doc['locus']} "
             f"(apex = visible bend-surface apex, sharp = plane-intersection); "
             f"t={lines_doc['thickness_mm']:.2f} mm Ri={lines_doc['ri_mm']:.2f} mm; "
             f"scan={lines_doc['scan_name']} sha256={lines_doc['scan_sha256'][:16]}; "
             f"provenance=provenance/{asset['asset_id']}.json")
    if lines_doc.get("notes"):
        notes += "; labeler notes: " + lines_doc["notes"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": LABEL_KIND,
        "asset_id": asset["asset_id"],
        "source": asset["source"],
        "source_relative_path": asset["relative_path"],
        "coordinate_frame": COORDINATE_FRAME,
        "review_status": AI_REVIEW_STATUS,
        "annotator": AI_ANNOTATOR,
        "revision": 0,
        "no_bends_present": no_bends and not ok,
        "bends": bends,
        "notes": notes[:4000],
    }


def write_label(lines_dir, asset_path, *, labels_root=None, project_root=PROJECT_ROOT,
                confirm_no_bends=False, include_uncertain=True):
    lines_dir = Path(lines_dir)
    lines_doc = json.loads((lines_dir / "lines.json").read_text())
    labels_root = Path(labels_root or DEFAULT_LABELS_ROOT).resolve()
    if HUMAN_LABELS_DIRNAME in labels_root.parts:
        raise LabelWriteError("refusing to write AI labels into human_bend_lines_v1")
    asset = resolve_asset_identity(asset_path, project_root)
    match, asset_sha = _coordinates_match(lines_doc, asset["path"])
    payload = build_payload(lines_doc, asset, confirm_no_bends=confirm_no_bends,
                            include_uncertain=include_uncertain)
    label_path = labels_root / f"{asset['asset_id']}.json"
    now = datetime.now(timezone.utc).isoformat()
    created = now
    revision = 0
    if label_path.is_file():
        previous = json.loads(label_path.read_text())
        revision = int(previous.get("revision", 0))
        created = previous.get("created_at", now)
    payload.update({"revision": revision + 1, "created_at": created, "updated_at": now})
    validated = validate_ai_label(payload, asset)
    _atomic_write(label_path, json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n")
    human = PROJECT_ROOT / "data" / "bend_corpus" / "annotations" / HUMAN_LABELS_DIRNAME / label_path.name
    provenance = {
        "tool": TOOL_VERSION,
        "asset_id": asset["asset_id"],
        "source": asset["source"],
        "source_relative_path": asset["relative_path"],
        "asset_sha256": asset_sha,
        "coordinate_match": match,
        "label_revision": validated["revision"],
        "label_sha256": sha256_file(label_path),
        "human_label_exists": human.is_file(),
        "lines_dir": str(lines_dir.resolve()),
        "lines": lines_doc,
        "written_at": now,
    }
    prov_path = labels_root / "provenance" / f"{asset['asset_id']}.json"
    _atomic_write(prov_path, json.dumps(provenance, indent=2, allow_nan=False, default=_json_default) + "\n")
    return {"label_path": str(label_path), "provenance_path": str(prov_path), "label": validated}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))
