#!/usr/bin/env python3
"""Prepare split public/private manifests for untouched physical scan intake."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.f1_physical_intake_v2 import prepare


def _atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _consumed_hashes(registry: dict[str, Any], root: Path) -> set[str]:
    consumed = set()
    for row in registry.get("intakes") or []:
        if not str(row.get("status") or "").startswith("DEVELOPMENT_CONSUMED"):
            continue
        explicit = row.get("scan_sha256s") or []
        if not isinstance(explicit, list):
            raise ValueError("scan_sha256s must be a list")
        for digest in explicit:
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("scan_sha256s entries must be lowercase hex64")
            consumed.add(digest)
        relative_root = row.get("root")
        if relative_root:
            path = (root / str(relative_root)).resolve()
            for source in sorted(path.rglob("*.ply")) if path.is_dir() else ():
                consumed.add(hashlib.sha256(source.read_bytes()).hexdigest())
    return consumed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-input", type=Path, required=True)
    parser.add_argument("--intake-root", type=Path, required=True)
    parser.add_argument("--lineage-registry", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--custodian-output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.lineage_registry.read_text(encoding="utf-8"))
    result = prepare(
        json.loads(args.private_input.read_text(encoding="utf-8")),
        intake_root=args.intake_root,
        consumed_scan_hashes=_consumed_hashes(registry, args.workspace_root.resolve()),
    )
    _atomic(args.public_output, result["public_manifest"])
    _atomic(args.custodian_output, result["custodian_receipt"])
    print(json.dumps({
        "status": result["public_manifest"]["status"],
        "scan_count": result["public_manifest"]["scan_count"],
        "public_and_custodian_outputs_separated": True,
    }, sort_keys=True))
    return 0 if result["public_manifest"]["status"] == "READY_FOR_CUSTODIAN_SEAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
