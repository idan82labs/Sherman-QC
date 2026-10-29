#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_organized_frame_intake_v1 import validate_bundle
from scripts.prepare_f1_physical_intake_v2 import _atomic


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate authentic organized depth/range observations")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--intake-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    receipt = validate_bundle(payload, intake_root=args.intake_root)
    _atomic(args.output, receipt)
    print(json.dumps({"status": receipt["status"], "cases": receipt["case_count"], "frames": receipt["frame_count"], "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
