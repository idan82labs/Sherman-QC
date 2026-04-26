#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


LABEL_CHOICES = (
    "conventional_bend",
    "raised_form_feature",
    "fragment_or_duplicate",
    "rolled_transition",
    "unclear",
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _scan_name(value: Any) -> str:
    return Path(str(value or "")).name


def _case_key(*, part_key: Any, scan_name: Any) -> Tuple[str, str]:
    return (str(part_key or ""), _scan_name(scan_name))


def _index_validation_results(summary_paths: Sequence[Path]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    rows: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for summary_path in summary_paths:
        summary = _load_json(summary_path)
        for result in list(summary.get("results") or ()):
            rows[_case_key(part_key=result.get("part_key"), scan_name=result.get("scan_path"))] = result
    return rows


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _focus_path_by_bend_id(render: Mapping[str, Any]) -> Dict[str, str]:
    mapped: Dict[str, str] = {}
    for raw_path in list(render.get("focus_paths") or ()):
        path = Path(str(raw_path))
        stem = path.stem
        # Current renderer emits patch_graph_owned_region_OB12.png.
        bend_id = stem.rsplit("_", 1)[-1]
        if bend_id:
            mapped[bend_id] = str(path)
    return mapped


def _copy_case_images(
    *,
    case_dir: Path,
    render: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    image_dir = case_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, Any] = {"focus_paths": {}}
    for key in ("overview_path", "scan_context_path", "atom_projection_path", "contact_sheet_1080p_path"):
        source = render.get(key)
        if source and Path(str(source)).exists():
            destination = image_dir / Path(str(source)).name
            shutil.copy2(str(source), destination)
            copied[key] = str(destination)

    focus_by_id = _focus_path_by_bend_id(render)
    for row in diagnostics:
        bend_id = str(row.get("bend_id") or "")
        source = focus_by_id.get(bend_id)
        if source and Path(source).exists():
            destination = image_dir / f"{_safe_name(bend_id)}_{Path(source).name}"
            shutil.copy2(source, destination)
            copied["focus_paths"][bend_id] = str(destination)
    return copied


def _write_feature_label_contact_sheet(
    *,
    case_dir: Path,
    case: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    copied_images: Mapping[str, Any],
) -> Optional[str]:
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception:
        return None

    focus_paths = dict(copied_images.get("focus_paths") or {})
    if not focus_paths:
        return None
    rows = list(diagnostics)
    cols = 3
    grid_rows = max(1, (len(rows) + cols - 1) // cols)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    fig.suptitle(
        f"{case.get('part_key')} | {case.get('scan_name')} | label each owned region",
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
        y=0.985,
    )
    for index, row in enumerate(rows, start=1):
        ax = fig.add_subplot(grid_rows, cols, index)
        ax.axis("off")
        bend_id = str(row.get("bend_id") or "")
        image_path = focus_paths.get(bend_id)
        if image_path and Path(str(image_path)).exists():
            ax.imshow(mpimg.imread(str(image_path)))
        title = (
            f"{bend_id} | {row.get('provisional_role')}\n"
            f"atoms={row.get('owned_atom_count')} mass={row.get('support_mass')} pair={row.get('incident_flange_ids')}"
        )
        ax.set_title(title, fontsize=8.5, fontweight="bold", color="#111827", pad=4)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955), pad=0.35)
    output_path = case_dir / "feature_label_contact_sheet_1080p.png"
    fig.savefig(output_path, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    return str(output_path)


def _label_template_for_case(
    *,
    case: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    copied_images: Mapping[str, Any],
    contact_sheet: Optional[str],
) -> Dict[str, Any]:
    focus_paths = dict(copied_images.get("focus_paths") or {})
    return {
        "part_key": case.get("part_key"),
        "scan_name": case.get("scan_name"),
        "truth_conventional_bends": case.get("truth_conventional_bends"),
        "truth_raised_form_features": case.get("truth_raised_form_features"),
        "truth_total_features": case.get("truth_total_features"),
        "algorithm_range": case.get("algorithm_range"),
        "algorithm_raw_exact_field": case.get("algorithm_raw_exact_field"),
        "label_choices": list(LABEL_CHOICES),
        "contact_sheet": contact_sheet,
        "instructions": [
            "Assign one label per owned region.",
            "Use conventional_bend only for a normal sheet-metal bend counted in the conventional target.",
            "Use raised_form_feature for raised/form features that should not count as conventional bends.",
            "Use fragment_or_duplicate for duplicate support, tiny fragments, or cluster artifacts.",
            "Use unclear if the image is insufficient.",
        ],
        "region_labels": [
            {
                "bend_id": row.get("bend_id"),
                "suggested_role": row.get("provisional_role"),
                "diagnostic_tags": list(row.get("diagnostic_tags") or ()),
                "human_feature_label": None,
                "confidence": None,
                "notes": "",
                "focus_image": focus_paths.get(str(row.get("bend_id") or "")),
                "geometry": {
                    "angle_deg": row.get("angle_deg"),
                    "support_mass": row.get("support_mass"),
                    "owned_atom_count": row.get("owned_atom_count"),
                    "span_length_mm": row.get("span_length_mm"),
                    "incident_flange_ids": row.get("incident_flange_ids"),
                    "anchor": row.get("anchor"),
                },
            }
            for row in diagnostics
        ],
    }


def _write_case_readme(case_dir: Path, template: Mapping[str, Any]) -> None:
    lines = [
        "# F1 Feature Label Review",
        "",
        f"Part: `{template.get('part_key')}`",
        f"Scan: `{template.get('scan_name')}`",
        "",
        "## Goal",
        "",
        "Label each owned F1 region by feature type so we can separate conventional bends from raised/form or duplicate fragments.",
        "",
        "## Files",
        "",
        "- `feature_label_contact_sheet_1080p.png`: overview of all owned regions.",
        "- `feature_region_labels.template.json`: fillable labels.",
        "- `images/`: copied source renders.",
        "",
        "## Labels",
        "",
    ]
    for label in LABEL_CHOICES:
        lines.append(f"- `{label}`")
    lines.extend(["", "## Regions", ""])
    for row in template.get("region_labels") or []:
        lines.append(f"- `{row.get('bend_id')}` suggested `{row.get('suggested_role')}` tags `{row.get('diagnostic_tags')}`")
    (case_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_feature_label_review_package(
    *,
    blocker_diagnostics_json: Path,
    validation_summary_jsons: Sequence[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    blockers = _load_json(blocker_diagnostics_json)
    validation = _index_validation_results(validation_summary_jsons)
    output_dir.mkdir(parents=True, exist_ok=True)
    packages: List[Dict[str, Any]] = []
    for index, case in enumerate(list(blockers.get("blockers") or ()), start=1):
        part_key = str(case.get("part_key") or f"case_{index}")
        scan_name = _scan_name(case.get("scan_name"))
        case_dir = output_dir / f"{index:02d}_{_safe_name(part_key)}_{_safe_name(Path(scan_name).stem)}"
        case_dir.mkdir(parents=True, exist_ok=True)
        validation_row = validation.get(_case_key(part_key=part_key, scan_name=scan_name))
        render = dict((validation_row or {}).get("render_artifacts") or {})
        diagnostics = list(case.get("owned_region_diagnostics") or ())
        copied = _copy_case_images(case_dir=case_dir, render=render, diagnostics=diagnostics)
        contact_sheet = _write_feature_label_contact_sheet(
            case_dir=case_dir,
            case=case,
            diagnostics=diagnostics,
            copied_images=copied,
        )
        template = _label_template_for_case(case=case, diagnostics=diagnostics, copied_images=copied, contact_sheet=contact_sheet)
        _write_json(case_dir / "feature_region_labels.template.json", template)
        _write_json(case_dir / "feature_region_diagnostics.json", dict(case))
        _write_case_readme(case_dir, template)
        packages.append(
            {
                "part_key": part_key,
                "scan_name": scan_name,
                "case_dir": str(case_dir),
                "contact_sheet": contact_sheet,
                "label_template": str(case_dir / "feature_region_labels.template.json"),
                "region_count": len(diagnostics),
            }
        )
    index_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Per-owned-region feature labels for F1 semantic blockers.",
        "label_choices": list(LABEL_CHOICES),
        "packages": packages,
    }
    _write_json(output_dir / "feature_label_review_index.json", index_payload)
    return index_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-owned-region F1 feature-label review packages.")
    parser.add_argument("--blocker-diagnostics-json", required=True, type=Path)
    parser.add_argument("--validation-summary-json", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    payload = build_feature_label_review_package(
        blocker_diagnostics_json=args.blocker_diagnostics_json,
        validation_summary_jsons=args.validation_summary_json,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "package_count": len(payload["packages"]),
                "packages": payload["packages"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
