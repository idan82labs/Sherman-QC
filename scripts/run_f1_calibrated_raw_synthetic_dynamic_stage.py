#!/usr/bin/env python3
"""Namespace adapter for frozen V51/V52/V60/V61/V62 on small visual pilots.

The adapter changes only corpus namespace and the opaque case-id list read from
the supplied public manifest.  Detector implementations and thresholds remain
frozen.  It exists so a visually accepted pilot need not be padded to the old
30-case runner assumption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _delegate(module, arguments: list[str], *, token: str, case_ids: tuple[str, ...], corpus_kind: str) -> int:
    previous = sys.argv
    module.TOKEN = token
    module.CASE_IDS = case_ids
    if hasattr(module, "CORPUS_KIND"):
        module.CORPUS_KIND = corpus_kind
    try:
        sys.argv = [str(Path(module.__file__).resolve()), *arguments]
        return int(module.main())
    finally:
        sys.argv = previous


def _rewrite_manifest(output_root: Path, stage: str, *, token: str, corpus_kind: str, case_ids: tuple[str, ...]) -> None:
    path = output_root / "execution_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update({
        "kind": f"sherman_qc_f1_{stage}_calibrated_raw_synthetic_dynamic_execution",
        "token": token,
        "corpus_kind": corpus_kind,
        "case_ids": list(case_ids),
        "case_count": len(case_ids),
        "visual_gate_passed_before_inference": True,
        "truth_opened_by_inference": False,
        "confirmatory_eligible": False,
    })
    payload.pop("development_only_after_v1_truth_open", None)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("v51","v52","v60","v61","v62"), required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--v9-model", type=Path)
    parser.add_argument("--v4-model", type=Path)
    parser.add_argument("--v51-execution-root", type=Path)
    parser.add_argument("--v52-root", type=Path)
    parser.add_argument("--v60-root", type=Path)
    parser.add_argument("--v61-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    public = json.loads(args.public_manifest.read_text(encoding="utf-8"))
    token = str(public["token"])
    corpus_kind = str(public["kind"])
    case_ids = tuple(str(row["opaque_case_id"]) for row in public["cases"])
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("public manifest case IDs must be nonempty and unique")

    common = ["--output-root", str(args.output_root), "--confirmation-token", token]
    if args.stage == "v51":
        import scripts.run_f1_v51_calibrated_raw_synthetic_v1 as runner
        if args.v9_model is None or args.v4_model is None:
            parser.error("v51 requires --v9-model and --v4-model")
        arguments = ["--public-manifest",str(args.public_manifest),"--v9-model",str(args.v9_model),"--v4-model",str(args.v4_model),*common]
    elif args.stage == "v52":
        import scripts.run_f1_v52_calibrated_raw_synthetic_v1 as runner
        if args.v51_execution_root is None: parser.error("v52 requires --v51-execution-root")
        arguments = ["--v51-execution-root",str(args.v51_execution_root),*common]
    elif args.stage == "v60":
        import scripts.run_f1_v60_calibrated_raw_synthetic_v1 as runner
        if args.v52_root is None: parser.error("v60 requires --v52-root")
        arguments = ["--corpus-root",str(args.public_manifest.parent),"--v52-root",str(args.v52_root),*common]
    elif args.stage == "v61":
        import scripts.run_f1_v61_calibrated_raw_synthetic_v1 as runner
        if args.v52_root is None or args.v60_root is None: parser.error("v61 requires --v52-root and --v60-root")
        arguments = ["--corpus-root",str(args.public_manifest.parent),"--v52-root",str(args.v52_root),"--v60-root",str(args.v60_root),*common]
    else:
        import scripts.run_f1_v62_calibrated_raw_synthetic_v4 as runner
        if args.v61_root is None: parser.error("v62 requires --v61-root")
        arguments = ["--corpus-root",str(args.public_manifest.parent),"--v61-root",str(args.v61_root),*common]

    code = _delegate(runner, arguments, token=token, case_ids=case_ids, corpus_kind=corpus_kind)
    _rewrite_manifest(args.output_root.resolve(), args.stage, token=token, corpus_kind=corpus_kind, case_ids=case_ids)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
