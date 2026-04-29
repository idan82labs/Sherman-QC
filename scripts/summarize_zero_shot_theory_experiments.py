#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score(entry: Dict[str, Any]) -> List[Any]:
    return [
        0 if entry["decision"] == "keep" else 1,
        -len(entry.get("improvements") or []),
        len(entry.get("regressions") or []),
        entry["method_family"],
    ]


def _load_experiment(folder: Path) -> Dict[str, Any]:
    manifest = _load_json(folder / "experiment_manifest.json")
    decision = _load_json(folder / "decision.json")
    return {
        "folder": str(folder),
        "method_family": manifest["method_family"],
        "hypothesis_id": manifest["hypothesis_id"],
        "decision": decision["decision"],
        "rejection_reason": decision.get("rejection_reason"),
        "improvements": list(decision.get("improvements") or []),
        "regressions": list(decision.get("regressions") or []),
        "watch_cases": dict(decision.get("watch_cases") or {}),
        "professor_brief": str(folder / "PROFESSOR_BRIEF.md"),
    }


def _render_markdown(rankings: List[Dict[str, Any]]) -> str:
    lines = [
        "# Zero-Shot Theory Experiment Summary",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Families evaluated: `{len(rankings)}`",
        "",
        "| Rank | Family | Decision | Improvements | Regressions | Rejection reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(rankings, start=1):
        lines.append(
            f"| {index} | `{row['method_family']}` | `{row['decision']}` | "
            f"`{len(row['improvements'])}` | `{len(row['regressions'])}` | "
            f"{row.get('rejection_reason') or ''} |"
        )
    winning = next((row for row in rankings if row["decision"] == "keep"), None)
    lines.extend(
        [
            "",
            "## Outcome",
            "",
            (
                f"Smallest winning tranche: `{winning['method_family']}`"
                if winning
                else "No winning tranche in this first-pass theory-to-reality run."
            ),
            "",
            "## Professor Brief Links",
            "",
        ]
    )
    for row in rankings:
        lines.append(f"- `{row['method_family']}` -> `{row['professor_brief']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize and rank zero-shot theory experiment folders.")
    parser.add_argument("--experiment-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    experiments = [_load_experiment(Path(value).resolve()) for value in args.experiment_dir]
    rankings = sorted(experiments, key=_score)
    winning = next((row for row in rankings if row["decision"] == "keep"), None)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "theory_experiment_rankings.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "rankings": rankings,
                "winning_tranche": winning["method_family"] if winning else None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "theory_experiment_rankings.md").write_text(
        _render_markdown(rankings),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "families_evaluated": len(rankings),
                "winning_tranche": winning["method_family"] if winning else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
