#!/usr/bin/env python3
"""Extract ZIP attachments from a locally preserved RFC822 message."""

from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence


def extract_zip_attachments(eml_path: Path, output_dir: Path) -> list[dict[str, object]]:
    """Extract ZIP MIME parts without trusting attachment directory components."""
    message = BytesParser(policy=policy.default).parsebytes(eml_path.read_bytes())
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, object]] = []
    for part in message.iter_attachments():
        filename = Path(str(part.get_filename() or "")).name
        if not filename.lower().endswith(".zip"):
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            raise ValueError(f"empty ZIP attachment payload: {filename}")
        destination = output_dir / filename
        if destination.exists() and destination.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different attachment: {destination}")
        destination.write_bytes(payload)
        extracted.append(
            {
                "filename": filename,
                "path": str(destination.resolve()),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not extracted:
        raise ValueError(f"no ZIP attachment found in {eml_path}")
    return extracted


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eml", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = extract_zip_attachments(args.eml.resolve(), args.output_dir.resolve())
    print(json.dumps({"attachments": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
