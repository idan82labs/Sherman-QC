#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from ai_analyzer import Model3DSnapshotRenderer
from bend_inspection_pipeline import load_bend_runtime_config
from zero_shot_bend_overlay import build_zero_shot_overlay_manifest, render_zero_shot_overlay_artifacts
from zero_shot_bend_profile import run_zero_shot_bend_profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Render zero-shot bend support strips as overlay artifacts.")
    parser.add_argument("--scan-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--part-id", default=None)
    args = parser.parse_args()

    runtime_config = load_bend_runtime_config(None)
    result = run_zero_shot_bend_profile(
        scan_path=args.scan_path,
        runtime_config=runtime_config,
        part_id=args.part_id,
    )
    payload = result.to_dict()
    manifest = build_zero_shot_overlay_manifest(
        result_payload=payload,
        scan_path=args.scan_path,
        renderable_bend_objects=payload.get("renderable_bend_objects"),
    )
    renderer = Model3DSnapshotRenderer()
    render_info = render_zero_shot_overlay_artifacts(
        renderer=renderer,
        manifest=manifest,
        scan_geometry_path=args.scan_path,
        output_dir=args.output_dir,
    )
    result_path = Path(args.output_dir) / "zero_shot_result.json"
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "result_path": str(result_path),
                "manifest_path": render_info["manifest_path"],
                "overview_path": render_info["overview_path"],
                "focus_paths": render_info["focus_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
