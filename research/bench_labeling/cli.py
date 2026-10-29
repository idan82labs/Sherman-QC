"""CLI: ``python -m research.bench_labeling {prep,lines,write-label}``."""

from __future__ import annotations

import argparse
import json
import sys


def _parser():
    parser = argparse.ArgumentParser(prog="python -m research.bench_labeling",
                                     description="AI-vision bend labeling tool")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prep", help="segment a scan into numbered planar patches + renders")
    prep.add_argument("--scan", required=True, help="PLY/STL scan (mesh or point cloud)")
    prep.add_argument("--out", required=True, help="output prep directory")
    prep.add_argument("--voxel", type=float, default=None, help="voxel size mm (default auto)")
    prep.add_argument("--max-points", type=int, default=200_000)
    prep.add_argument("--angle", type=float, default=8.0, help="region normal tolerance deg")
    prep.add_argument("--local-dist", type=float, default=0.3, help="neighbour plane tol mm")
    prep.add_argument("--min-area", type=float, default=30.0, help="ignore patches below mm^2")
    prep.add_argument("--min-width", type=float, default=2.0, help="ignore slivers below mm")
    prep.add_argument("--thickness", type=float, default=None, help="sheet thickness mm")
    prep.add_argument("--ri", type=float, default=None, help="inner bend radius mm (default t)")
    prep.add_argument("--title", default=None)

    lines = sub.add_parser("lines", help="compute bend lines from labeler patch pairs")
    lines.add_argument("--prep", required=True, help="prep directory")
    lines.add_argument("--pairs", required=True, help="pairs JSON")
    lines.add_argument("--out", required=True, help="output lines directory")
    lines.add_argument("--locus", choices=("apex", "sharp"), default="apex")
    lines.add_argument("--thickness", type=float, default=None)
    lines.add_argument("--ri", type=float, default=None)
    lines.add_argument("--scan", default=None, help="override scan path (sha256 must match)")
    lines.add_argument("--margin", type=float, default=None,
                       help="default support-zone margin mm (LineParams.margin_mm, default 3.0); "
                            "per-bend 'margin_mm' in the pairs file wins")

    write = sub.add_parser("write-label", help="write an ai_bend_lines_v1 label")
    write.add_argument("--lines", required=True, help="lines directory")
    write.add_argument("--asset-path", required=True, help="studio PLY asset (same coordinates)")
    write.add_argument("--labels-root", default=None, help="override labels dir (tests)")
    write.add_argument("--project-root", default=None, help="override project root (tests)")
    write.add_argument("--confirm-no-bends", action="store_true")
    write.add_argument("--certain-only", action="store_true", help="drop uncertain bends")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "prep":
        from .prep import run_prep
        from .segment import SegmentParams

        params = SegmentParams(voxel_mm=args.voxel, max_points=args.max_points,
                               angle_deg=args.angle, local_dist_mm=args.local_dist,
                               min_area_mm2=args.min_area, min_width_mm=args.min_width)
        manifest = run_prep(args.scan, args.out, params=params, title=args.title,
                            thickness_mm=args.thickness, ri_mm=args.ri)
        summary = {k: manifest[k] for k in ("scan_name", "scan_sha256", "voxel_mm", "ds_points",
                                             "n_patches", "thickness_mm", "ri_mm", "images")}
    elif args.command == "lines":
        from .lines import run_lines

        doc = run_lines(args.prep, args.pairs, args.out, locus=args.locus, ri_mm=args.ri,
                        thickness_mm=args.thickness, scan_path=args.scan, margin_mm=args.margin)
        summary = {"n_ok": doc["n_ok"], "n_failed": doc["n_failed"], "images": doc["images"],
                   "bends": [{k: b.get(k) for k in ("bend_id", "patches", "status", "error",
                                                    "included_angle_deg", "length_mm", "skin",
                                                    "confidence", "confidence_reasons",
                                                    "extent_source", "t_range", "t_range_auto",
                                                    "fold_support_frac", "override_support_frac")}
                             for b in doc["bends"]]}
    else:
        from .label import PROJECT_ROOT, write_label

        result = write_label(args.lines, args.asset_path, labels_root=args.labels_root,
                             project_root=args.project_root or PROJECT_ROOT,
                             confirm_no_bends=args.confirm_no_bends,
                             include_uncertain=not args.certain_only)
        summary = {"label_path": result["label_path"], "provenance_path": result["provenance_path"],
                   "asset_id": result["label"]["asset_id"], "bends": len(result["label"]["bends"]),
                   "revision": result["label"]["revision"]}
    json.dump(summary, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0
