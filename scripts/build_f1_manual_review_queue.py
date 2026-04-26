#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _range_width(count_range: Sequence[Any]) -> Optional[int]:
    if len(count_range) < 2:
        return None
    try:
        return int(count_range[1]) - int(count_range[0])
    except (TypeError, ValueError):
        return None


def _priority_for_review(review: Mapping[str, Any]) -> int:
    track = str(review.get("track") or "")
    algorithm = dict(review.get("algorithm") or {})
    manual = dict(review.get("manual_review") or {})
    reasons = set(str(value) for value in manual.get("reason_codes") or ())
    width = _range_width(list(algorithm.get("accepted_bend_count_range") or ()))
    if track == "partial" and width is not None and width > 2:
        return 1
    if "raw_f1_manual_promotion_candidate" in reasons:
        return 1
    if manual.get("required"):
        return 2
    return 3


def _review_case_row(review: Mapping[str, Any], *, priority: Optional[int] = None) -> Dict[str, Any]:
    algorithm = dict(review.get("algorithm") or {})
    render = dict(review.get("render_artifacts") or {})
    manual = dict(review.get("manual_review") or {})
    scan_path = str(review.get("scan_path") or "")
    return {
        "priority": _priority_for_review(review) if priority is None else int(priority),
        "track": review.get("track"),
        "part_key": review.get("part_key"),
        "scan_name": Path(scan_path).name,
        "scan_path": scan_path,
        "range": list(algorithm.get("accepted_bend_count_range") or ()),
        "observed_prediction": algorithm.get("observed_bend_count_prediction"),
        "exact": algorithm.get("accepted_exact_bend_count"),
        "known_total_context": review.get("known_total_bend_count_context"),
        "reasons": list(manual.get("reason_codes") or ()),
        "contact_sheet": render.get("contact_sheet_1080p_path"),
        "review_json": None,
    }


def _collect_review_rows(summary_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in summary_paths:
        summary = _load_json(summary_path)
        for review in list(summary.get("results") or ()):
            row = _review_case_row(review)
            row["source_summary_json"] = str(summary_path)
            rows.append(row)
    rows.sort(key=lambda row: (int(row["priority"]), row.get("track") != "partial", str(row.get("part_key")), str(row.get("scan_name"))))
    return rows


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def _copy_contact_sheets(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    contact_dir = output_dir / "priority_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        src = Path(str(row.get("contact_sheet") or ""))
        if not src.exists():
            continue
        filename = _safe_filename(
            f"P{row['priority']}_{index:02d}_{row.get('track')}_{row.get('part_key')}_{Path(str(row.get('scan_name') or 'scan')).stem}.png"
        )
        dst = contact_dir / filename
        shutil.copy2(src, dst)
        row["review_queue_contact_sheet"] = str(dst)


def _write_p1_overview(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> Optional[str]:
    priority = [row for row in rows if int(row.get("priority") or 0) == 1 and row.get("review_queue_contact_sheet")]
    if not priority:
        return None
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception:
        return None

    cols = 2
    grid_rows = max(1, (len(priority) + cols - 1) // cols)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    for index, row in enumerate(priority, start=1):
        ax = fig.add_subplot(grid_rows, cols, index)
        ax.axis("off")
        ax.imshow(mpimg.imread(str(row["review_queue_contact_sheet"])))
        ax.set_title(
            f"P1 {row.get('track')} {row.get('part_key')} {row.get('scan_name')} range {row.get('range')}",
            fontsize=9,
            fontweight="bold",
        )
    fig.tight_layout(pad=0.4)
    output_path = output_dir / "P1_overview_contact_sheet_1080p.png"
    fig.savefig(output_path, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    return str(output_path)


def _labels_template(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "label_semantics": "observed visible bends for partial scans; known full-scan count verification for known scans",
        "labels": [
            {
                "part_key": row.get("part_key"),
                "scan_name": row.get("scan_name"),
                "track": row.get("track"),
                "algorithm_range": list(row.get("range") or ()),
                "algorithm_observed_prediction": row.get("observed_prediction"),
                "algorithm_exact": row.get("exact"),
                "human_observed_bend_count": None,
                "human_full_part_bend_count": row.get("known_total_context"),
                "algo_correct": None,
                "false_positive_bend_ids": [],
                "missed_bend_notes": [],
                "notes": "",
                "contact_sheet": row.get("review_queue_contact_sheet") or row.get("contact_sheet"),
            }
            for row in rows
        ],
    }


def _write_readme(output_dir: Path, rows: Sequence[Mapping[str, Any]], generated_at: str) -> None:
    lines = [
        "# F1 Manual Review Queue",
        "",
        f"Generated: {generated_at}",
        "",
        "## Files",
        "",
        "- `manual_review_queue.json`: machine-readable queue with render paths.",
        "- `manual_validation_labels.template.json`: fillable labels/corrections template.",
        "- `P1_overview_contact_sheet_1080p.png`: one-page overview of highest-priority cases when available.",
        "- `priority_contact_sheets/`: copied 1080p contact sheets for every review case.",
        "",
        "## Instructions",
        "",
        "- Validate visible observed bends only for partial scans.",
        "- Start with Priority 1.",
        "- If you tell Codex corrections in plain text, use: `part scan_name observed=N notes=...`.",
        "",
        "## Cases",
        "",
    ]
    for row in rows:
        contact = row.get("review_queue_contact_sheet") or row.get("contact_sheet")
        if contact:
            try:
                rel = Path(str(contact)).relative_to(output_dir)
            except ValueError:
                rel = Path(str(contact))
            link = f"[contact sheet]({rel})"
        else:
            link = "no contact sheet"
        lines.append(
            f"- P{row.get('priority')} `{row.get('track')}` `{row.get('part_key')}` `{row.get('scan_name')}` "
            f"range `{row.get('range')}` observed `{row.get('observed_prediction')}` - {link}"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_review_queue(*, summary_paths: Sequence[Path], output_dir: Path, copy_contact_sheets: bool = True) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _collect_review_rows(summary_paths)
    if copy_contact_sheets:
        _copy_contact_sheets(rows, output_dir)
    overview_path = _write_p1_overview(rows, output_dir)
    generated_at = datetime.now().isoformat(timespec="seconds")
    queue = {
        "generated_at": generated_at,
        "purpose": "Manual validation queue for F1 rendered bend-count evidence.",
        "instructions": [
            "Use the 1080p contact sheet first.",
            "For partial scans, validate visible observed bends only, not whole-part total.",
            "Fill observed counts in manual_validation_labels.template.json or send corrections to Codex.",
        ],
        "overview_path": overview_path,
        "cases": rows,
    }
    _write_json(output_dir / "manual_review_queue.json", queue)
    _write_json(output_dir / "manual_validation_labels.template.json", _labels_template(rows))
    _write_readme(output_dir, rows, generated_at)
    return queue


def _default_output_dir() -> Path:
    icloud_root = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/Sherman QC"
    if icloud_root.exists():
        return icloud_root / f"F1 Accuracy Validation Review Queue {datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    return Path.cwd() / "data/bend_corpus/evaluation" / f"f1_accuracy_validation_review_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a prioritized manual-review queue from F1 validation summaries.")
    parser.add_argument("--validation-summary-json", action="append", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-copy", action="store_true", help="Do not copy contact-sheet images into the queue folder.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir()
    queue = build_review_queue(
        summary_paths=[Path(path).expanduser().resolve() for path in args.validation_summary_json],
        output_dir=output_dir,
        copy_contact_sheets=not args.no_copy,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "case_count": len(queue["cases"]),
                "priority_1": sum(1 for row in queue["cases"] if int(row["priority"]) == 1),
                "priority_2": sum(1 for row in queue["cases"] if int(row["priority"]) == 2),
                "priority_3": sum(1 for row in queue["cases"] if int(row["priority"]) == 3),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
