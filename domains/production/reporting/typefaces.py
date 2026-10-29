"""Bundled report typeface: Heebo (SIL Open Font License 1.1).

Heebo covers Hebrew and Latin, so one family typesets both languages of the
report.

Upstream files (unmodified, in ``fonts/``), the static instances at the commit
Google Fonts ships (``ofl/heebo/METADATA.pb`` source commit), downloaded on
2026-10-17:

- https://github.com/OdedEzer/heebo/tree/76c7d538568d3e6d7f052a12c5675c84430df25f/fonts/ttf
  ``Heebo-Regular.ttf`` and ``Heebo-Bold.ttf`` (Version 3.100);
- ``OFL.txt`` from https://github.com/google/fonts/blob/main/ofl/heebo/OFL.txt
  (identical to the upstream ``OFL.txt``). It declares no Reserved Font Name.

Report build (``HeeboReport-*.ttf``, a Modified Version under the OFL): the
upstream files with the ``liga`` GSUB feature emptied (:func:`derive_report_font`).
Heebo's ``liga`` holds only the Latin f-ligatures (ff, fi, fl, ffi, ffl); its
``fl`` glyph has no Unicode mapping, so with ligatures on, text extracted
from the PDF read "flange" as "ýange". Hebrew shaping (``ccmp``, ``locl``,
marks) is untouched; the alef-lamed ligature lives in ``dlig``, which is off
by default anyway. The derivation is deterministic (no timestamp update), and
a test re-derives the report build from the upstream files and compares bytes.

The OFL allows bundling and embedding (the PDF embeds the glyphs it uses);
the licence file travels with the fonts, which the engine bundle does
(``engine_bundle.RESOURCE_SUFFIXES`` / ``RESOURCE_NAMES_PREFIX``).
:func:`verify_fonts` checks every file against the sha256 values below, so a
swapped or truncated font can never silently change the report.

Known glyph gaps (checked with fontTools): no Greek (Δ), arrows, ≤/≥ or
geometric shapes. The report therefore writes "Dev." and plain ASCII/Latin-1
symbols only; U+2212 (minus), ±, °, ×, —, …, · and U+200F (RLM) are present.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = [
    "FONT_DIR",
    "FONT_FAMILY",
    "FONT_FILES",
    "UPSTREAM_FILES",
    "LICENSE_FILE",
    "verify_fonts",
    "font_face_css",
    "font_path",
    "derive_report_font",
]

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FAMILY = "Heebo"
#: weight -> (report build file name, sha256): what the PDF and snapshots use.
FONT_FILES: dict[str, tuple[str, str]] = {
    "regular": ("HeeboReport-Regular.ttf", "2f819e4c26e1b16b3c16afffe4a8d86d33f1af921b403c020073a891dc95e96e"),
    "bold": ("HeeboReport-Bold.ttf", "44e641cc825206eb614a1fa1b3a14bdeacf7e43d0803ca7237a8956f4abc10ab"),
}
#: weight -> (upstream file name, sha256): the unmodified sources of the report build.
UPSTREAM_FILES: dict[str, tuple[str, str]] = {
    "regular": ("Heebo-Regular.ttf", "e9dc1958e6a5aec129ee79f01e9cfb442680b73bf6c72964137daa581a3afd62"),
    "bold": ("Heebo-Bold.ttf", "54d93bf35b8272df73180f687879bda00284a3343d85c3f46ec04d39c89ed5a7"),
}
LICENSE_FILE = ("OFL.txt", "866747cbccae229433e946b90fcfc457b81c33ac89e593855770851354f9e6c3")


def font_path(weight: str = "regular") -> Path:
    return FONT_DIR / FONT_FILES[weight][0]


def verify_fonts() -> None:
    """Raise ``RuntimeError`` unless every bundled font and the licence match their sha256."""
    for name, sha in (*FONT_FILES.values(), *UPSTREAM_FILES.values(), LICENSE_FILE):
        path = FONT_DIR / name
        if not path.is_file():
            raise RuntimeError(f"bundled report font file missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha:
            raise RuntimeError(f"bundled report font file {name} has sha256 {digest}, expected {sha}")


def font_face_css() -> str:
    """``@font-face`` rules resolving against an archive rooted at :data:`FONT_DIR`."""
    regular, bold = FONT_FILES["regular"][0], FONT_FILES["bold"][0]
    return (
        f"@font-face {{font-family: {FONT_FAMILY}; src: url({regular});}}\n"
        f"@font-face {{font-family: {FONT_FAMILY}; font-weight: bold; src: url({bold});}}\n"
    )


def derive_report_font(src: str | Path) -> bytes:
    """The report build of an upstream Heebo file: every ``liga`` feature
    record keeps its tag but references no lookup. Nothing else changes
    (``head.modified`` is not touched), so the output is deterministic."""
    import io

    from fontTools.ttLib import TTFont

    font = TTFont(str(src), recalcTimestamp=False, recalcBBoxes=False)
    for record in font["GSUB"].table.FeatureList.FeatureRecord:
        if record.FeatureTag == "liga":
            record.Feature.LookupListIndex = []
            record.Feature.LookupCount = 0
    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()
