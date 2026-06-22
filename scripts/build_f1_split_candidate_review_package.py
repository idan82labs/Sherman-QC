#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


SPLIT_LABELS = ("keep_single", "split_into_two_features", "unclear")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _focus_path_by_bend_id(render_dir: Path) -> Dict[str, Path]:
    mapped: Dict[str, Path] = {}
    for path in sorted(render_dir.glob("patch_graph_owned_region_OB*.png")):
        bend_id = path.stem.rsplit("_", 1)[-1]
        if bend_id.startswith("OB"):
            mapped[bend_id] = path
    return mapped


def _copy_images(*, output_dir: Path, render_dir: Path, candidates: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    focus_by_id = _focus_path_by_bend_id(render_dir)
    copied: Dict[str, str] = {}
    for candidate in candidates:
        bend_id = str(candidate.get("bend_id") or "")
        source = focus_by_id.get(bend_id)
        if source and source.exists():
            destination = image_dir / f"{_safe_name(bend_id)}_{source.name}"
            shutil.copy2(source, destination)
            copied[bend_id] = str(destination)
    for name in (
        "patch_graph_owned_region_overview.png",
        "patch_graph_owned_region_overview_side.png",
        "patch_graph_owned_region_overview_top.png",
        "patch_graph_owned_region_atom_projection.png",
    ):
        source = render_dir / name
        if source.exists():
            destination = image_dir / name
            shutil.copy2(source, destination)
            copied[source.stem] = str(destination)
    return copied


def _write_contact_sheet(*, output_dir: Path, candidates: Sequence[Mapping[str, Any]], copied_images: Mapping[str, str]) -> Optional[str]:
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not candidates:
        return None
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    fig.suptitle("10839000 full scan - split candidate review", fontsize=18, fontweight="bold", color="#0f172a")
    cols = max(1, len(candidates))
    for index, candidate in enumerate(candidates, start=1):
        ax = fig.add_subplot(1, cols, index)
        ax.axis("off")
        bend_id = str(candidate.get("bend_id") or "")
        image_path = copied_images.get(bend_id)
        if image_path and Path(image_path).exists():
            ax.imshow(mpimg.imread(image_path))
        title = (
            f"{bend_id} | {candidate.get('split_candidate_rank')}\n"
            f"gap={candidate.get('max_projection_gap_mm')}mm ratio={candidate.get('max_gap_to_median_gap_ratio')} "
            f"L/R={candidate.get('largest_gap_left_atom_count')}/{candidate.get('largest_gap_right_atom_count')}"
        )
        ax.set_title(title, fontsize=11, fontweight="bold", color="#111827")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), pad=0.5)
    output = output_dir / "split_candidate_review_1080p.png"
    fig.savefig(output, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    return str(output)


def build_split_candidate_review_package(
    *,
    split_diagnostics_json: Path,
    render_dir: Path,
    output_dir: Path,
    max_candidates: int = 2,
) -> Dict[str, Any]:
    diagnostics = _load_json(split_diagnostics_json)
    candidates = [
        row for row in list(diagnostics.get("regions") or ())
        if row.get("split_candidate")
    ][: max(0, int(max_candidates))]
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_images(output_dir=output_dir, render_dir=render_dir, candidates=candidates)
    contact_sheet = _write_contact_sheet(output_dir=output_dir, candidates=candidates, copied_images=copied)
    template = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "part_key": diagnostics.get("part_id") or "10839000",
        "scan_path": diagnostics.get("scan_path"),
        "capacity_deficit": diagnostics.get("capacity_deficit"),
        "split_label_choices": list(SPLIT_LABELS),
        "instructions": [
            "Label whether each split candidate is one feature or should be split into two counted features.",
            "Use split_into_two_features only when the single OB marker visibly spans two real bends/features.",
            "If neither candidate explains the missing feature, mark both keep_single or unclear and note missing feature location.",
        ],
        "contact_sheet": contact_sheet,
        "candidate_labels": [
            {
                "bend_id": row.get("bend_id"),
                "suggested_split_rank": row.get("split_candidate_rank"),
                "human_split_label": None,
                "notes": "",
                "focus_image": copied.get(str(row.get("bend_id") or "")),
                "metrics": row,
            }
            for row in candidates
        ],
        "missing_feature_notes": "",
    }
    _write_json(output_dir / "split_candidate_labels.template.json", template)
    _write_json(output_dir / "split_candidate_diagnostics.json", diagnostics)
    readme = [
        "# Split Candidate Review",
        "",
        "Open `split_candidate_review_1080p.png` first.",
        "",
        "For each candidate, choose one:",
        "",
        *[f"- `{label}`" for label in SPLIT_LABELS],
        "",
        "If one candidate should split, tell Codex for example:",
        "",
        "```text",
        "OB2=split_into_two_features, OB1=keep_single",
        "```",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "candidate_count": len(candidates),
        "contact_sheet": contact_sheet,
        "label_template": str(output_dir / "split_candidate_labels.template.json"),
        "candidates": [row.get("bend_id") for row in candidates],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build focused review package for F1 owned-region split candidates.")
    parser.add_argument("--split-diagnostics-json", required=True, type=Path)
    parser.add_argument("--render-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-candidates", type=int, default=2)
    args = parser.parse_args()

    payload = build_split_candidate_review_package(
        split_diagnostics_json=args.split_diagnostics_json,
        render_dir=args.render_dir,
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
