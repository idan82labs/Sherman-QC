#!/usr/bin/env python3
"""Namespace adapter for frozen calibrated V51/V52/V60/V61 runners on V3.

The underlying route modules remain unchanged.  This adapter replaces only the
synthetic confirmation token and post-run manifest labels so the independently
locked V3 corpus retains explicit provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v3 import TOKEN


def _delegate(module, arguments: list[str]) -> int:
    previous = sys.argv
    module.TOKEN = TOKEN
    try:
        sys.argv = [str(Path(module.__file__).resolve()), *arguments]
        return int(module.main())
    finally:
        sys.argv = previous


def _relabel_manifest(output_root: Path, stage: str) -> None:
    path = output_root / "execution_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["kind"] = f"sherman_qc_f1_{stage}_calibrated_raw_synthetic_v3_execution"
    payload["corpus_kind"] = "sherman_qc_f1_calibrated_raw_range_complex_geometry_v3"
    payload["token"] = TOKEN
    payload["geometry_stress"] = True
    if stage == "v61":
        payload.pop("development_only_after_v1_truth_open", None)
        payload["threshold_provenance"] = "frozen_on_v1_before_independent_v2_and_v3"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("v51", "v52", "v60", "v61"), required=True)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--public-manifest", type=Path)
    parser.add_argument("--v9-model", type=Path)
    parser.add_argument("--v4-model", type=Path)
    parser.add_argument("--v51-execution-root", type=Path)
    parser.add_argument("--v52-root", type=Path)
    parser.add_argument("--v60-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--confirmation-token", required=True)
    args = parser.parse_args()
    if args.confirmation_token != TOKEN:
        raise ValueError("explicit V3 synthetic engineering token required")

    if args.stage == "v51":
        import scripts.run_f1_v51_calibrated_raw_synthetic_v1 as runner

        required = (args.public_manifest, args.v9_model, args.v4_model)
        if any(value is None for value in required):
            parser.error("v51 requires --public-manifest, --v9-model, and --v4-model")
        arguments = [
            "--public-manifest", str(args.public_manifest),
            "--v9-model", str(args.v9_model),
            "--v4-model", str(args.v4_model),
            "--output-root", str(args.output_root),
            "--confirmation-token", TOKEN,
        ]
    elif args.stage == "v52":
        import scripts.run_f1_v52_calibrated_raw_synthetic_v1 as runner

        if args.v51_execution_root is None:
            parser.error("v52 requires --v51-execution-root")
        arguments = [
            "--v51-execution-root", str(args.v51_execution_root),
            "--output-root", str(args.output_root),
            "--confirmation-token", TOKEN,
        ]
    elif args.stage == "v60":
        import scripts.run_f1_v60_calibrated_raw_synthetic_v1 as runner

        if args.corpus_root is None or args.v52_root is None:
            parser.error("v60 requires --corpus-root and --v52-root")
        arguments = [
            "--corpus-root", str(args.corpus_root),
            "--v52-root", str(args.v52_root),
            "--output-root", str(args.output_root),
            "--confirmation-token", TOKEN,
        ]
    else:
        import scripts.run_f1_v61_calibrated_raw_synthetic_v1 as runner

        if args.corpus_root is None or args.v52_root is None or args.v60_root is None:
            parser.error("v61 requires --corpus-root, --v52-root, and --v60-root")
        arguments = [
            "--corpus-root", str(args.corpus_root),
            "--v52-root", str(args.v52_root),
            "--v60-root", str(args.v60_root),
            "--output-root", str(args.output_root),
            "--confirmation-token", TOKEN,
        ]

    code = _delegate(runner, arguments)
    _relabel_manifest(args.output_root.resolve(), args.stage)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
