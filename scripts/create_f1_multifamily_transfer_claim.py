#!/usr/bin/env python3
"""Create a fresh one-shot transfer claim bound to one frozen submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--case-token", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to replace an existing transfer claim")
    payload = {
        "schema_version": 1,
        "kind": "sherman_qc_f1_multifamily_transfer_claim",
        "status": "created",
        "case_token": args.case_token,
        "lock_sha256": sha(args.lock),
        "submission_sha256": sha(args.submission),
        "c006_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
