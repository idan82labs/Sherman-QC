"""Command line for the Jev bend-candidate harness (C10, v1).

Usage::

    python -m research.jev.cli export-onboarding --input RECIPES.jsonl [...] --out DIR
    python -m research.jev.cli export-scan --out DIR [--jobs-root ROOT] [--labels-root DIR]
                                           [--f1-pool POOL.jsonl --gold GOLD.json]
    python -m research.jev.cli dry-run --states states.jsonl --out DIR [--model jev-1.13.0]
    python -m research.jev.cli live --live --states states.jsonl --labels labels.jsonl
                                    --splits splits.json --out DIR       # needs TYPESAFE_API_KEY
    python -m research.jev.cli evaluate --labels labels.jsonl --splits splits.json
                                        --states states.jsonl --out DIR [--answers answers.jsonl]
    python -m research.jev.cli make-synthetic --out DIR   # synthetic inputs + exports, no real data

Exit codes: 0 ok; 2 usage, custody or policy refusal (including ``live`` without
``--live`` or without ``TYPESAFE_API_KEY``); 3 API or internal failure.

Every input path is explicit. Inputs under the repo's ``data/``, ``output/``,
``outputs/``, ``artifacts/``, ``uploads/``, ``live_scans/`` or ``research/`` (other
than ``research/jev``) are refused unless ``--authorization FILE`` lists them.
Jev output is never wired into a runtime decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from research.jev.client_v1 import ClientError, dry_run, live_run
from research.jev.evaluate_v1 import EvaluationError, evaluate
from research.jev.export_onboarding_v1 import (
    CustodyError,
    ExportError,
    export_onboarding,
    write_synthetic_inputs,
)
from research.jev.export_scan_candidates_v1 import (
    export_scan,
    synthetic_e3_result,
    synthetic_f1_pool,
    synthetic_legacy_report,
)
from research.jev.labels_v1 import LabelError
from research.jev.questions_v1 import DEFAULT_QUESTIONS
from research.jev.request_v1 import RequestInvalid
from research.jev.splits_v1 import SPLIT_MANIFEST_SCHEMA, SplitError
from research.jev.state_v1 import StateDenyError

__all__ = ["main", "build_parser", "write_synthetic_workspace"]

_POLICY_ERRORS = (CustodyError, ExportError, LabelError, SplitError, StateDenyError, RequestInvalid,
                  EvaluationError, FileNotFoundError, KeyError)


def _question_list(text: str) -> list[str]:
    return [q.strip() for q in text.split(",") if q.strip()]


def _add_request_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="jev-1.13.0", help="versioned Jev id (aliases need --allow-alias)")
    p.add_argument("--allow-alias", action="store_true")
    p.add_argument("--variant", default="v1", choices=("v1", "verbalized_v1"))
    p.add_argument("--questions", type=_question_list, default=list(DEFAULT_QUESTIONS),
                   help="comma list; is_real_bend is mandatory (candidate_kind, evidence_quality optional)")
    p.add_argument("--include-eval", action="store_true", help="also send eval-split records")
    p.add_argument("--cad-free", action="store_true", help="drop the CAD-assisted cad_relation block")
    p.add_argument("--context-free", action="store_true", help="drop the part-level context block")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research.jev.cli", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export-onboarding", help="export CAD onboarding candidates")
    p.add_argument("--input", type=Path, action="append", required=True, help="recipe JSON or bundle JSON/JSONL")
    p.add_argument("--out", type=Path, required=True, help="new output directory")
    p.add_argument("--authorization", type=Path)
    p.add_argument("--no-unmarked-graph", action="store_true", help="skip unmarked E1 flange-graph bends")

    p = sub.add_parser("export-scan", help="export scan/engine candidates")
    p.add_argument("--out", type=Path, required=True, help="new output directory")
    p.add_argument("--jobs-root", type=Path, help="directory of job workspaces")
    p.add_argument("--labels-root", type=Path, help="explicit human label documents per job")
    p.add_argument("--f1-pool", type=Path, help="F1 candidate-pool JSONL")
    p.add_argument("--gold", type=Path, help="jev_gold_lines_v1 document for the F1 pool")
    p.add_argument("--authorization", type=Path)
    p.add_argument("--provenance-flag", action="append", default=[], choices=("consumed", "retrospective"))

    p = sub.add_parser("dry-run", help="write request bodies and a cost summary (no network)")
    _add_request_options(p)
    p.add_argument("--splits", type=Path, help="split manifest (with --labels) to filter like live")
    p.add_argument("--labels", type=Path)

    p = sub.add_parser("live", help="send dev/calib requests (needs --live and TYPESAFE_API_KEY)")
    _add_request_options(p)
    p.add_argument("--live", action="store_true", help="explicitly allow network calls")
    p.add_argument("--splits", type=Path)
    p.add_argument("--labels", type=Path)
    p.add_argument("--max-records", type=int)

    p = sub.add_parser("evaluate", help="score answers and baselines")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--answers", type=Path)
    p.add_argument("--operating-point", type=Path, help="frozen operating_point.json (never refitted)")
    p.add_argument("--cad-free-baseline", action="store_true")
    p.add_argument("--bootstrap-rounds", type=int, default=1000)

    p = sub.add_parser("make-synthetic", help="write synthetic inputs and their exports (tests/acceptance)")
    p.add_argument("--out", type=Path, required=True)
    return parser


def write_synthetic_workspace(out: Path) -> dict[str, str]:
    """Synthetic onboarding + scan inputs, their exports and a split manifest under ``out``."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ExportError(f"{out} exists and is not empty")
    inputs = out / "inputs"
    onboarding_input = write_synthetic_inputs(inputs)
    jobs = inputs / "jobs"
    for i, sku in enumerate(("SYN-SCAN-A", "SYN-SCAN-B")):
        engine = jobs / f"job-{i}" / "engine"
        engine.mkdir(parents=True)
        (engine / "result.json").write_text(json.dumps(synthetic_e3_result(i, sku=sku)), encoding="utf-8")
        (engine / "bend_inspection_report.json").write_text(json.dumps(synthetic_legacy_report(i)), encoding="utf-8")
        (jobs / f"job-{i}" / "recipe.json").write_text(json.dumps({"sku": sku}), encoding="utf-8")
    rows, gold = synthetic_f1_pool(0)
    pool = inputs / "f1_pool.jsonl"
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    gold_path = inputs / "gold.json"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")
    export_onboarding([onboarding_input], out / "onboarding")
    export_scan(out / "scan", jobs_root=jobs, f1_pool=pool, gold=gold_path)
    splits = {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "families": {"FAM-A": {"split": "dev"}, "FAM-B": {"split": "calib"}, "FAM-C": {"split": "eval"},
                     "FAM-SCAN": {"split": "dev"}, "FAM-F1": {"split": "excluded", "flags": ["holdout"]}},
        "sku_to_family": {"SYN-ONB-A": "FAM-A", "SYN-ONB-B": "FAM-B", "SYN-ONB-C": "FAM-C",
                          "SYN-SCAN-A": "FAM-SCAN", "SYN-SCAN-B": "FAM-SCAN", "SYN-F1-A": "FAM-F1"},
    }
    (out / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")
    return {"onboarding_states": str(out / "onboarding" / "states.jsonl"),
            "onboarding_labels": str(out / "onboarding" / "labels.jsonl"),
            "scan_states": str(out / "scan" / "states.jsonl"),
            "splits": str(out / "splits.json")}


def _print(obj: object) -> None:
    print(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "export-onboarding":
            r = export_onboarding(args.input, args.out, authorization=args.authorization,
                                  include_unmarked_graph=not args.no_unmarked_graph)
            _print({"out": str(r.out_dir), "states": r.states, "labels": r.labels, "warnings": r.manifest["warnings"]})
        elif args.command == "export-scan":
            r = export_scan(args.out, jobs_root=args.jobs_root, labels_root=args.labels_root, f1_pool=args.f1_pool,
                            gold=args.gold, authorization=args.authorization, provenance_flags=args.provenance_flag)
            _print({"out": str(r.out_dir), "states": r.states, "labels": r.labels})
        elif args.command == "dry-run":
            _print(dry_run(args.states, args.out, model=args.model, allow_alias=args.allow_alias,
                           variant=args.variant, question_names=args.questions, splits=args.splits,
                           labels=args.labels, include_eval=args.include_eval, cad_free=args.cad_free,
                           context_free=args.context_free))
        elif args.command == "live":
            _print(live_run(args.states, args.out, live=args.live, splits=args.splits, labels=args.labels,
                            model=args.model, allow_alias=args.allow_alias, variant=args.variant,
                            question_names=args.questions, include_eval=args.include_eval,
                            cad_free=args.cad_free, context_free=args.context_free,
                            max_records=args.max_records))
        elif args.command == "evaluate":
            report = evaluate(labels=args.labels, splits=args.splits, states=args.states, out_dir=args.out,
                              answers=args.answers, operating_point=args.operating_point,
                              cad_free_baseline=args.cad_free_baseline, bootstrap_rounds=args.bootstrap_rounds)
            _print({"flags": report["flags"], "counts": report["counts"], "out": str(args.out)})
        elif args.command == "make-synthetic":
            _print(write_synthetic_workspace(args.out))
    except ClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except _POLICY_ERRORS as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
