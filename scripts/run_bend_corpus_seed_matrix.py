#!/opt/homebrew/bin/python3.11
"""Run repeated bend corpus evaluations across one or more seed profiles."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "data" / "bend_corpus"
EVALUATOR = ROOT / "scripts" / "evaluate_bend_corpus.py"
COMPARE = ROOT / "scripts" / "compare_bend_corpus_summaries.py"

import sys

sys.path.insert(0, str(ROOT / "scripts"))

from compare_bend_corpus_summaries import compare_summaries  # noqa: E402


def _load_summary(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("aggregate", payload)


def _normalize_profile(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    label = str(raw.get("label") or f"profile_{idx + 1}").strip() or f"profile_{idx + 1}"
    deterministic_seed = raw.get("deterministic_seed")
    retry_seeds = raw.get("retry_seeds")
    return {
        "label": label,
        "deterministic_seed": None if deterministic_seed is None else int(deterministic_seed),
        "retry_seeds": [int(seed) for seed in (retry_seeds or [])],
    }


def _seed_profiles(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.seed_matrix_json not in {None, "", "None"}:
        raw = json.loads(args.seed_matrix_json)
        if not isinstance(raw, list) or not raw:
            raise ValueError("--seed-matrix-json must be a non-empty JSON list")
        return [_normalize_profile(dict(item), idx) for idx, item in enumerate(raw)]
    retry_seeds = []
    if args.retry_seeds_json not in {None, "", "None"}:
        retry_seeds = [int(seed) for seed in json.loads(args.retry_seeds_json)]
    label = str(args.profile_label or f"seed_{args.deterministic_seed if args.deterministic_seed is not None else 'default'}")
    return [{
        "label": label,
        "deterministic_seed": args.deterministic_seed,
        "retry_seeds": retry_seeds,
    }]


def _run_cmd(
    *,
    corpus: Path,
    output_dir: Path,
    parts: List[str],
    timeout_sec: int,
    unary_model: Optional[str],
    skip_heatmap_consistency: bool,
    heatmap_cache_dir: Optional[str],
    deterministic_seed: Optional[int],
    retry_seeds: List[int],
) -> List[str]:
    cmd = [
        "/opt/homebrew/opt/python@3.11/bin/python3.11",
        str(EVALUATOR),
        "--corpus",
        str(corpus),
        "--output",
        str(output_dir),
        "--timeout-sec",
        str(timeout_sec),
    ]
    for part in parts:
        cmd.extend(["--part", part])
    if unary_model:
        cmd.extend(["--unary-model", unary_model])
    if skip_heatmap_consistency:
        cmd.append("--skip-heatmap-consistency")
    if heatmap_cache_dir:
        cmd.extend(["--heatmap-cache-dir", heatmap_cache_dir])
    if deterministic_seed is not None:
        cmd.extend(["--deterministic-seed", str(int(deterministic_seed))])
    if retry_seeds:
        cmd.extend(["--retry-seeds-json", json.dumps([int(seed) for seed in retry_seeds])])
    return cmd


def _compare_runs(reference_summary: Path, candidate_summary: Path) -> Dict[str, Any]:
    baseline = _load_summary(reference_summary)
    candidate = _load_summary(candidate_summary)
    return compare_summaries(baseline, candidate)


def _markdown_lines(manifest: Dict[str, Any]) -> Iterable[str]:
    yield "# Bend Corpus Seed Matrix"
    yield ""
    yield f"- Corpus: `{manifest['corpus']}`"
    yield f"- Repeats per profile: `{manifest['repeat_count']}`"
    yield f"- Profiles: `{len(manifest['profiles'])}`"
    yield ""
    for profile in manifest["profiles"]:
        yield f"## {profile['label']}"
        yield ""
        yield f"- Deterministic seed: `{profile['deterministic_seed']}`"
        yield f"- Retry seeds: `{profile['retry_seeds']}`"
        yield f"- Runs: `{len(profile['runs'])}`"
        if profile.get("repeat_comparisons"):
            yield ""
            yield "| Comparison | Changed metrics |"
            yield "| --- | ---: |"
            for comparison in profile["repeat_comparisons"]:
                yield f"| `{comparison['label']}` | `{comparison['changed_metric_count']}` |"
        yield ""
    if manifest.get("cross_profile_comparisons"):
        yield "## Cross-Profile"
        yield ""
        yield "| Comparison | Changed metrics |"
        yield "| --- | ---: |"
        for comparison in manifest["cross_profile_comparisons"]:
            yield f"| `{comparison['label']}` | `{comparison['changed_metric_count']}` |"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(CORPUS_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--part", action="append", default=[])
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=420)
    parser.add_argument("--unary-model", default=None)
    parser.add_argument("--skip-heatmap-consistency", action="store_true")
    parser.add_argument("--heatmap-cache-dir", default=None)
    parser.add_argument("--deterministic-seed", type=int, default=None)
    parser.add_argument("--retry-seeds-json", default=None)
    parser.add_argument("--profile-label", default=None)
    parser.add_argument("--seed-matrix-json", default=None)
    args = parser.parse_args(argv)

    if args.repeat_count < 1:
        raise ValueError("--repeat-count must be >= 1")

    corpus = Path(args.corpus)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else (corpus / "evaluation" / f"seed_matrix_{stamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    profiles = _seed_profiles(args)
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "corpus": str(corpus),
        "output_root": str(output_root),
        "repeat_count": int(args.repeat_count),
        "profiles": [],
        "cross_profile_comparisons": [],
    }

    for profile in profiles:
        profile_out = output_root / profile["label"]
        profile_out.mkdir(parents=True, exist_ok=True)
        runs: List[Dict[str, Any]] = []
        for repeat_idx in range(args.repeat_count):
            run_label = f"run_{repeat_idx + 1:02d}"
            run_dir = profile_out / run_label
            cmd = _run_cmd(
                corpus=corpus,
                output_dir=run_dir,
                parts=[part for part in args.part if part],
                timeout_sec=args.timeout_sec,
                unary_model=args.unary_model,
                skip_heatmap_consistency=bool(args.skip_heatmap_consistency),
                heatmap_cache_dir=args.heatmap_cache_dir,
                deterministic_seed=profile["deterministic_seed"],
                retry_seeds=profile["retry_seeds"],
            )
            completed = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError((completed.stderr or completed.stdout or f"seed matrix runner failed: {completed.returncode}").strip())
            runs.append({
                "label": run_label,
                "output_dir": str(run_dir),
                "summary_path": str(run_dir / "summary.json"),
                "command": cmd,
            })

        repeat_comparisons: List[Dict[str, Any]] = []
        reference_summary = Path(runs[0]["summary_path"])
        for run in runs[1:]:
            comparison = _compare_runs(reference_summary, Path(run["summary_path"]))
            repeat_comparisons.append({
                "label": f"{runs[0]['label']} vs {run['label']}",
                "baseline_summary": str(reference_summary),
                "candidate_summary": run["summary_path"],
                **comparison,
            })
        manifest["profiles"].append({
            **profile,
            "runs": runs,
            "repeat_comparisons": repeat_comparisons,
        })

    if len(manifest["profiles"]) > 1:
        reference = manifest["profiles"][0]
        reference_summary = Path(reference["runs"][0]["summary_path"])
        for profile in manifest["profiles"][1:]:
            comparison = _compare_runs(reference_summary, Path(profile["runs"][0]["summary_path"]))
            manifest["cross_profile_comparisons"].append({
                "label": f"{reference['label']} vs {profile['label']}",
                "baseline_summary": str(reference_summary),
                "candidate_summary": profile["runs"][0]["summary_path"],
                **comparison,
            })

    (output_root / "seed_matrix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_root / "seed_matrix_report.md").write_text("\n".join(_markdown_lines(manifest)) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "profile_count": len(manifest["profiles"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
