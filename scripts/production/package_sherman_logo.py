#!/usr/bin/env python3
"""Package the official Sherman logo for the inspection report.

Usage::

    .venv/bin/python scripts/production/package_sherman_logo.py \
        /Users/idantal/ShermanQC/brand/www.shermanep.com_Images_logo.png

Writes ``domains/production/reporting/brand_logo.py`` (a Python module, so the
engine bundle carries it like any other source file) holding:

- ``SOURCE_PNG``: the official file, byte for byte (221 x 46 px), and
- ``REPORT_PNG``: the same image resampled 4x (Lanczos) so PDF viewers draw
  smooth edges at print size. Nothing is redrawn or recoloured.

Why not vector outlines: an iso-line trace of the 221 px raster (tried on
2026-10-19: per-colour coverage, 8x cubic upsampling, contour at 0.5) loses the
sub-pixel white strokes of the "i" mark and makes the tagline letters lumpy, i.e.
it changes the logo. Until a vector master (SVG/AI/EPS) is supplied, the
faithful option is the original pixels, embedded losslessly.
"""
from __future__ import annotations

import base64
import hashlib
import io
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "domains" / "production" / "reporting" / "brand_logo.py"
SCALE = 4


def main(argv: list[str]) -> int:
    from PIL import Image

    src = Path(argv[0])
    raw = src.read_bytes()
    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    big = im.resize((im.width * SCALE, im.height * SCALE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    big.save(buf, format="PNG", optimize=True)
    report = buf.getvalue()

    def b64(data: bytes) -> str:
        return "\n".join(f'    "{line}"' for line in textwrap.wrap(base64.b64encode(data).decode("ascii"), 96))

    text = f'''"""Official Sherman logo for the inspection report (generated; do not edit by hand).

Made by ``scripts/production/package_sherman_logo.py`` (see its docstring).
Source: {src.name}, {im.width}x{im.height} px, sha256 {hashlib.sha256(raw).hexdigest()}.
"""
from __future__ import annotations

import base64

__all__ = ["SIZE", "SOURCE_SHA256", "REPORT_SHA256", "NAVY", "BLUE", "source_png", "report_png"]

SIZE = ({im.width}, {im.height})
SOURCE_SHA256 = "{hashlib.sha256(raw).hexdigest()}"
REPORT_SHA256 = "{hashlib.sha256(report).hexdigest()}"
#: The logo's two flat colours (sampled from the source file).
NAVY = "#01416F"
BLUE = "#00AEEF"

_SOURCE_B64 = (
{b64(raw)}
)

_REPORT_B64 = (
{b64(report)}
)


def source_png() -> bytes:
    return base64.b64decode(_SOURCE_B64)


def report_png() -> bytes:
    return base64.b64decode(_REPORT_B64)
'''
    OUT.write_text(text, encoding="utf-8")
    print(OUT, len(raw), len(report), OUT.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
