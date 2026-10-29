"""Derived-only native-face comparison panels; never opens CAD/native inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

COLORS = {"retained": "#269363", "recovered": "#147fc1", "missing": "#d24343",
          "unmatched": "#884fb2", "unknown_prediction": "#db9029", "context": "#c6cbd1"}
VIEWS = [(1, 0, 0), (0, 1, 0), (1, 1, 1), (-1, -1, -1)]


def marker_line(prediction):
    """Validate supplied actual marker endpoints without synthesizing extent."""
    start, end = prediction.get("line_start"), prediction.get("line_end")
    if start is None and end is None:
        return None
    for vector in (start, end):
        if not isinstance(vector, (list, tuple)) or len(vector) != 3 or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in vector):
            raise ValueError("Marker endpoints must be finite3vectors")
    if all(a == b for a, b in zip(start, end)):
        raise ValueError("Marker extent cannot be zero")
    semantics = prediction.get("marker_extent_semantics")
    if not isinstance(semantics, str) or not semantics.strip():
        raise ValueError("Marker extent semantics required")
    return {"prediction_id": prediction["prediction_id"], "line_start": list(start),
            "line_end": list(end), "marker_extent_semantics": semantics}


def _group(ids, census):
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 1 for i in ids):
        raise ValueError("Face groups require positive integer native IDs")
    if len(ids) != len(set(ids)) or not set(ids) <= census:
        raise ValueError("Duplicate or out-of-census face IDs")
    return frozenset(ids)


def classify_part(part, census):
    """Exact one-to-one face identity, explicitly separated from geometry accuracy."""
    references = {}
    owned = set()
    for bend in part["reference_bends"]:
        name = bend["bend_id"]
        inner = _group(bend["inner_face_ids"], census)
        outer = _group(bend["outer_face_ids"], census)
        group = inner | outer
        if name in references or inner & outer or group & owned:
            raise ValueError("Invalid reference ownership")
        references[name] = group
        owned.update(group)
    unknown = set(part["unknown_face_ids"])
    if any(type(i) is not int or i < 1 for i in unknown) or not unknown <= census or unknown & owned:
        raise ValueError("Invalid unknown domain")
    if part["reference_status"] not in ("complete", "partial"):
        raise ValueError("Invalid reference status")
    if part["reference_status"] == "complete" and unknown:
        raise ValueError("Complete part cannot contain unknowns")
    stages = {}
    for stage in ("baseline", "challenger"):
        value = part[stage]
        if value["status"] not in ("ok", "failed") or value["status"] == "failed" and value["predictions"]:
            raise ValueError("Invalid extraction status")
        available = set(references)
        predictions = {}
        matched, unmatched, markers = {}, [], []
        for prediction in value["predictions"]:
            name = prediction["prediction_id"]
            if not isinstance(name, str) or not name or name in predictions:
                raise ValueError("Prediction IDs must be unique text")
            predictions[name] = _group(prediction["face_ids"], census)
            marker = marker_line(prediction)
            if marker:
                markers.append(marker)
        for name in sorted(predictions):
            matches = sorted(ref for ref in available if references[ref] == predictions[name])
            if matches:
                matched[matches[0]] = name
                available.remove(matches[0])
            else:
                unmatched.append({"prediction_id": name, "face_ids": sorted(predictions[name]),
                                  "category": "unknown_prediction" if predictions[name] & unknown else "unmatched"})
        stages[stage] = {"matched": matched, "unmatched": unmatched, "prediction_count": len(predictions),
                         "actual_marker_lines": markers}
    regions = [{"bend_id": name, "face_ids": sorted(group),
                "category": ("retained" if name in stages["baseline"]["matched"] else "recovered")
                if name in stages["challenger"]["matched"] else "missing"}
               for name, group in sorted(references.items())]
    return {"regions": regions, "stages": stages,
            "counts": {category: sum(r["category"] == category for r in regions)
                       for category in ("retained", "recovered", "missing")}}


def _read_pinned(path, digest, limit):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Invalid or oversized pinned input")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Pinned input hash mismatch")
    return json.loads(raw)


def render(evidence_dir, comparison_json, comparison_sha256, out):
    bundle = _read_pinned(comparison_json, comparison_sha256, 5_000_000)
    parts = bundle["parts"]
    names = {f"case-{i:02d}" for i in range(1, 21)}
    if len(parts) != 20 or {p["part_id"] for p in parts} != names:
        raise ValueError("Exactly all20 anonymous cases required")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    out.mkdir(parents=False, exist_ok=False)
    manifest = []
    for part in sorted(parts, key=lambda p: p["part_id"]):
        name = part["part_id"]
        evidence = _read_pinned(evidence_dir / name / "EVIDENCE.json", part["evidence_sha256"], 40_000_000)
        if evidence["input_sha256"] != part["source_sha256"]:
            raise ValueError("Source binding mismatch")
        faces = evidence["faces"]
        census = {f["id"] for f in faces}
        if len(census) != len(faces) or census != set(part["face_ids"]):
            raise ValueError("Full native face census mismatch")
        result = classify_part(part, census)
        face_categories = {i: r["category"] for r in result["regions"] for i in r["face_ids"]}
        for item in result["stages"]["challenger"]["unmatched"]:
            for i in item["face_ids"]:
                face_categories[i] = item["category"]
        meshes, unrendered, total_triangles = {}, [], 0
        for face in faces:
            mesh = face.get("display_mesh")
            if not mesh:
                unrendered.append(face["id"])
                continue
            vertices = np.asarray(mesh["vertices"], dtype=float)
            raw_triangles = mesh["triangles_one_based"]
            if any(type(i) is not int for tri in raw_triangles for i in tri):
                raise ValueError("Noninteger mesh index")
            triangles = np.asarray(raw_triangles, dtype=int) - 1
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError("Invalid mesh vertices")
            if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles) or triangles.min() < 0 or triangles.max() >= len(vertices):
                raise ValueError("Invalid mesh triangles")
            total_triangles += len(triangles)
            if len(triangles) > 200_000 or total_triangles > 2_000_000:
                raise ValueError("Display mesh budget exceeded")
            meshes[face["id"]] = (vertices, triangles)
        if not meshes:
            raise ValueError("No display mesh")
        points = np.concatenate([v for v, _ in meshes.values()])
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        fig, axes = plt.subplots(2, 2, figsize=(22, 18))
        for ax, direction in zip(axes.flat, VIEWS):
            z = np.array(direction, dtype=float); z /= np.linalg.norm(z)
            seed = np.eye(3)[np.argmin(abs(z))]
            x = np.cross(seed, z); x /= np.linalg.norm(x)
            basis = np.stack([x, np.cross(z, x), z], axis=1)
            polygons, depths, colors = [], [], []
            for face_id, (vertices, triangles) in meshes.items():
                tri = ((vertices - center) @ basis)[triangles]
                normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                lengths = np.linalg.norm(normals, axis=1)
                shade = .63 + .34 * np.divide(abs(normals[:, 2]), lengths, out=np.zeros(len(lengths)), where=lengths > 0)
                base = np.array(to_rgb(COLORS[face_categories.get(face_id, "context")]))
                polygons.extend(tri[:, :, :2]); depths.extend(tri[:, :, 2].mean(axis=1))
                colors.extend(base[None, :] * shade[:, None])
            order = np.argsort(depths, kind="stable")
            ax.add_collection(PolyCollection(np.asarray(polygons)[order], facecolors=np.asarray(colors)[order], edgecolors="none"))
            projection = (points - center) @ basis
            lo, hi = projection[:, :2].min(axis=0), projection[:, :2].max(axis=0)
            span = max(hi - lo); pad = span * .10 + 1e-6
            ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
            lookup = {f["id"]: f for f in faces}
            for index, region in enumerate(result["regions"]):
                centers = [lookup[i].get("area_centroid") for i in region["face_ids"]]
                if any(c is None for c in centers):
                    continue
                c = (np.mean(centers, axis=0) - center) @ basis
                ax.annotate(f"R{index+1} [{','.join(map(str,region['face_ids']))}]", c[:2],
                            xytext=(8, 8 if index % 2 else -14), textcoords="offset points", fontsize=8,
                            color=COLORS[region["category"]], bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=1),
                            arrowprops=dict(arrowstyle="-", color=COLORS[region["category"]], lw=.6))
            retained_predictions = {result["stages"]["challenger"]["matched"][r["bend_id"]]
                                    for r in result["regions"] if r["category"] == "retained"}
            for marker in result["stages"]["challenger"]["actual_marker_lines"]:
                endpoints = (np.asarray([marker["line_start"], marker["line_end"]]) - center) @ basis
                color = COLORS["retained"] if marker["prediction_id"] in retained_predictions else "#00acc1"
                ax.plot(endpoints[:, 0], endpoints[:, 1], color=color, linestyle="--", linewidth=2,
                        zorder=5, solid_capstyle="butt")
                midpoint = endpoints[:, :2].mean(axis=0)
                ax.annotate(f"M:{marker['prediction_id']}", midpoint, xytext=(0, 5),
                            textcoords="offset points", fontsize=7, color=color, zorder=6,
                            bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=.6))
            ax.set_aspect("equal"); ax.axis("off"); ax.set_title(f"Orthographic direction {direction}", fontsize=12)
        counts = result["counts"]
        fig.suptitle(f"{name} — geometric bend-face comparison | reference {part['reference_status']}\n"
                     f"Baseline {part['baseline']['status']}: {result['stages']['baseline']['prediction_count']} predictions; "
                     f"challenger {part['challenger']['status']}: {result['stages']['challenger']['prediction_count']} | "
                     f"known reference {len(result['regions'])}; retained {counts['retained']}, recovered {counts['recovered']}, missing {counts['missing']}", fontsize=16)
        legend = [("retained", "Reference matched by baseline AND challenger"), ("recovered", "Reference newly matched by challenger"),
                  ("missing", "Known reference NOT matched by challenger (reference overlay)"),
                  ("unmatched", "Unmatched challenger prediction — not a verified TP"),
                  ("unknown_prediction", "Unmatched prediction intersects unknown reference domain"), ("context", "Full native surface context")]
        fig.legend(handles=[Patch(color=COLORS[k], label=label) for k, label in legend], loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(.5, .035))
        unmatched = result["stages"]["challenger"]["unmatched"]
        details = "; ".join(f"{u['prediction_id']} faces{u['face_ids']}" for u in unmatched[:6])
        if len(unmatched) > 6:
            details += f"; +{len(unmatched)-6} groups (full IDs in MANIFEST)"
        unknown_preview = str(part["unknown_face_ids"][:16])
        if len(part["unknown_face_ids"]) > 16:
            unknown_preview += " … (full IDs in MANIFEST/input bundle)"
        fig.text(.02, .018, "Exact native face-set detection only; reference overlays are NOT inferred output. Mesh projection is approximate, labels may mark occluded faces.\n"
                 f"Unknown reference faces ({len(part['unknown_face_ids'])}): {unknown_preview} | Unrendered faces: {unrendered}\nUnmatched challenger groups: {details or 'none'}", fontsize=8)
        marker_semantics = sorted({m["marker_extent_semantics"] for m in result["stages"]["challenger"]["actual_marker_lines"]})
        fig.text(.02, .105, "Dashed green/cyan M-lines: actual challenger marker endpoints only (may be occluded); no synthesized full-width span. "
                 f"Extent semantics: {', '.join(marker_semantics) or 'no marker endpoints supplied'}. Whole-face reference colors do not certify marker location/extent.", fontsize=8)
        fig.tight_layout(rect=(0, .14, 1, .94))
        target = out / f"{name}.png"
        fig.savefig(target, dpi=140); plt.close(fig)
        manifest.append({"part_id": name, "evidence_sha256": part["evidence_sha256"], "source_sha256": part["source_sha256"],
                         "image": target.name, "image_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                         "classification": result, "unrendered_face_ids": unrendered,
                         "unknown_reference_face_ids": part["unknown_face_ids"]})
    with (out / "MANIFEST.json").open("x") as stream:
        json.dump({"schema": "cad_bend_score_comparison_gallery_v1", "comparison_sha256": comparison_sha256,
                   "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "panels": manifest,
                   "native_execution": False, "reference_overlays_not_inference": True}, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--comparison-json", required=True, type=Path)
    parser.add_argument("--comparison-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    render(args.evidence_dir, args.comparison_json, args.comparison_sha256, args.out)
