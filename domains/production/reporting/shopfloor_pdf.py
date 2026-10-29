"""Formal dimensional inspection report (PDF) of one inspection result.

Rendered **only** from a ``ShopfloorInspectionResultV1`` (plus, optionally,
the released recipe for header fields the result does not carry, such as the
drawing revision, and the viewer assets for the 3D pictures). The report never
computes or changes a value or a decision: every number, decision and reason is
the stored result's, printed through the contract vocabulary
(``contracts.production.vocabulary.DISPLAY_TOKENS``) and :mod:`.labels`.

Layout (A4 portrait; structure after AS9102 FAIR forms and CALYPSO / PiWeb
protocols, see ``docs/production/runlog/pdf_v2/RESEARCH.md``):

Page 1
    Title, a title block (part number, revision, part name, material, job,
    serial, recipe, tolerance standard, scan file, station, scan / inspection
    times, a blank operator field, document number), the summary (verdict,
    stage "<formed>/<total> bends formed", counts of required PASS / REVIEW /
    FAIL / undetermined characteristics, worst characteristic, stage status,
    validity), one clean 3D overview with the bends numbered in shop order,
    and the inspector sign-off box (the software is an operator assist; the
    disposition is the inspector's).
1. Characteristic results
    One numbered row per characteristic (``<shop order>.<n>``), bends in shop
    order, grouped under their machine move (מהלך), flange lengths first and
    then the angle (then bend-line position and backgauge). Studio (CAD) bend
    labels (``context.bend_display_labels``). Only the Result cell is coloured.
    Bends that are not yet formed are listed in a separate light table, never
    as failures.
2. Measurements
    General measurements, then unplanned creases.
Appendix
    A: heatmap views; B: method, uncertainty and traceability notes (decision
    rule, conventions, reason legend, stage and validity details, hashes,
    warnings); C: per-characteristic evidence (boundary, stations, method).

Every page has a running header (logo, title, part number, revision, job) and
footer (document number, page X of Y, generated-by line, confidentiality, and
the operator-assist note in both languages).

Language: ``lang="he"`` (default) typesets a Hebrew right-to-left document
(tables mirrored: first column on the right); ``lang="en"`` an English one.
Field labels and column heads carry the other language in small type. Hebrew
is stored in logical order inside ``dir="rtl"`` blocks (MuPDF applies the
Unicode bidi algorithm and HarfBuzz shaping); numbers sit in left-to-right
cells or LRM-guarded runs, so they are never reordered.

Honesty: a NOT_MEASURED value prints "Undetermined" plus its reason; it never
prints PASS or FAIL. A NOT_YET_FORMED bend is never coloured red or green.

Determinism: identical inputs give identical bytes (fixed metadata dates from
``result.created_at``, no PDF id, deterministic snapshots, embedded logo).
"""
from __future__ import annotations

import html
import io
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from domains.production.reporting.labels import (
    BOUNDARY_LABELS,
    INSPECTION_STAGE_LABELS,
    STAGE_STATUS_LABELS,
    TEXT,
    VALIDITY_LABELS,
    Bi,
    bend_display_label,
    kind_label,
    reason_label,
    token,
)

if TYPE_CHECKING:  # pragma: no cover
    from contracts.production.inspection_result import (
        BendRowV1,
        CharacteristicResultV1,
        ShopfloorInspectionResultV1,
    )
    from contracts.production.part_recipe import PartRecipeV1

__all__ = [
    "PAGE_RECT", "CONTENT_RECT", "LANGS", "build_report_flow", "build_report_html", "render_pdf", "write_pdf",
    "pdf_date", "document_number", "char_number", "OVERVIEW_PX",
]

#: A4 portrait, points.
PAGE_RECT = (0.0, 0.0, 595.0, 842.0)
_MARGIN_X = 42.0
CONTENT_RECT = (_MARGIN_X, 82.0, 595.0 - _MARGIN_X, 784.0)
_CONTENT_W = CONTENT_RECT[2] - CONTENT_RECT[0]
LANGS = ("he", "en")
#: Page-1 overview picture size (px); the adapter renders it with the same size.
OVERVIEW_PX = (1800, 620)
_MAX_PAGES = 400
_MINUS = "−"
_MISSING = "—"

# Palette: black / greys, Sherman navy (the logo's) as the only accent, and
# green / amber / red only in verdict (Result) cells.
_INK = "#111827"
_GREY = "#4B5563"
_MUTED = "#6B7280"
_FAINT = "#9CA3AF"
_RULE = "#D1D5DB"
_FILL = "#F3F4F6"
_ACCENT = "#01416F"
_VERDICT = {"pass": "#15803D", "fail": "#B91C1C", "review": "#D97706"}

_CSS = f"""
body {{font-family: Heebo; font-size: 8pt; color: {_INK}; line-height: 1.3; margin: 0;}}
p {{margin: 0 0 3pt 0;}}
h1 {{font-size: 16pt; font-weight: bold; color: {_INK}; margin: 0;}}
h2 {{font-size: 10.5pt; font-weight: bold; color: {_ACCENT}; margin: 14pt 0 5pt 0; padding: 0 0 2pt 0;
     border-bottom: 0.9pt solid {_ACCENT};}}
h3 {{font-size: 8.6pt; font-weight: bold; color: {_INK}; margin: 9pt 0 3pt 0;}}
.sub {{color: {_GREY}; font-size: 8.5pt; margin: 1pt 0 8pt 0;}}
.muted {{color: {_MUTED};}}
.faint {{color: {_FAINT};}}
.small {{font-size: 6.6pt;}}
.tiny {{font-size: 5.8pt; color: {_MUTED};}}
.b {{font-weight: bold;}}
table {{border-collapse: collapse;}}
td {{padding: 2.6pt 4pt; vertical-align: top; border-bottom: 0.4pt solid {_RULE};}}
th {{padding: 0 4pt; vertical-align: bottom; border-bottom: 0.9pt solid {_ACCENT}; font-weight: bold; text-align: left;}}
th div.h {{padding: 2pt 0 2.5pt 0; color: {_ACCENT}; font-size: 7pt;}}
td.num {{text-align: right;}}
td.ctr {{text-align: center;}}
td.grp {{padding: 0; border-bottom: 0.4pt solid {_RULE};}}
div.grp {{background-color: {_FILL}; padding: 2.4pt 4pt; font-weight: bold; color: {_INK};}}
tr.nyf td {{color: {_MUTED};}}
td.res {{padding: 1.6pt 4pt;}}
div.badge {{font-weight: bold; text-align: center; padding: 1pt 2pt; font-size: 7pt;}}
div.d-PASS {{background-color: {_VERDICT['pass']}; color: #FFFFFF;}}
div.d-FAIL {{background-color: {_VERDICT['fail']}; color: #FFFFFF;}}
div.d-REVIEW {{background-color: {_VERDICT['review']}; color: #FFFFFF;}}
div.d-UNKNOWN {{background-color: #E5E7EB; color: #374151;}}
div.d-NOT_APPLICABLE {{color: {_FAINT};}}
table.tb td {{border: 0.5pt solid {_RULE}; padding: 3pt 4pt 3.5pt 4pt;}}
table.tb td.sz {{border: none; padding: 0 4pt; height: 0;}}
table.tb td.k {{background-color: #FFFFFF;}}
div.k {{color: {_GREY}; font-size: 6.6pt;}}
div.v {{font-size: 8.4pt; font-weight: bold; color: {_INK};}}
table.sum td {{border: none; padding: 0; vertical-align: top;}}
div.verdict {{padding: 8pt 9pt; color: #FFFFFF;}}
table.counts td {{border: 0.5pt solid {_RULE}; padding: 4pt 4pt; text-align: center;}}
div.cnum {{font-size: 13pt; font-weight: bold;}}
table.kv td {{border: none; border-bottom: 0.4pt solid {_RULE}; padding: 2.4pt 4pt;}}
table.sign td {{border: 0.5pt solid {_RULE}; padding: 4pt 4pt; vertical-align: top;}}
div.box {{border: 0.7pt solid {_GREY}; width: 7pt; height: 7pt;}}
table.chk td {{border: none; padding: 1pt 3pt 1pt 0; vertical-align: middle;}}
div.auth {{border-left: 1.6pt solid {_ACCENT}; padding: 3pt 7pt; color: {_GREY}; margin: 8pt 0 0 0; font-size: 7pt;}}
table.grid td {{border: 0.4pt solid {_RULE};}}
table.grid th {{border: 0.4pt solid {_RULE}; border-bottom: 0.9pt solid {_ACCENT};}}
table.grid td.grp {{border: 0.4pt solid {_RULE};}}
table.views td {{border: none; padding: 2pt; text-align: center;}}
table.legend td {{font-size: 6.8pt;}}
div.app p {{font-size: 6.8pt; color: #374151;}}
"""


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


#: RIGHT-TO-LEFT MARK. MuPDF treats ``dir="rtl"`` on an inline span as a hint,
#: not as a bidi embedding, so neutral characters at the edges of a Hebrew run
#: would resolve to the surrounding direction. Strong RLMs at both edges keep
#: them inside the RTL run.
_RLM = "‏"
#: LEFT-TO-RIGHT MARK: closes a Hebrew run before a digit-led identifier (a
#: numeric SKU such as ``10839000`` or ``10839000@v1``), which would otherwise
#: join the RTL run and be drawn reversed.
_LRM = "‎"


def _ltr(text: Any) -> str:
    """An identifier or number kept left-to-right as one run whatever surrounds it."""
    return f"{_LRM}{_e(text)}{_LRM}"


def _he(text: str) -> str:
    """Inline Hebrew run (inside a left-to-right line)."""
    return f'<span dir="rtl">{_RLM}{_e(text)}{_RLM}</span>'


def _has_hebrew(text: str) -> bool:
    return any("\u0590" <= c <= "\u05ff" for c in text)


def _sha8(sha: str | None) -> str:
    return sha[:8] if sha else _MISSING


def _num(value: float | None, unit: str, *, signed: bool = False, digits: int = 2) -> str:
    if value is None:
        return _MISSING
    text = f"{abs(value):.{digits}f}"
    if signed:
        sign = "+" if value > 0 else (_MINUS if value < 0 else "")
        if float(text) == 0.0:
            sign = ""
        text = sign + text
    elif value < 0:
        text = _MINUS + text
    return text + ("°" if unit == "deg" else "")


def _tol(tol: Any, unit: str) -> str:
    if tol is None:
        return _MISSING
    if abs(tol.plus - tol.minus) < 1e-12:
        return "±" + _num(tol.plus, unit)
    return f"+{_num(tol.plus, unit)} / {_MINUS}{_num(tol.minus, unit)}"


def pdf_date(iso: str) -> str:
    """``2026-10-17T07:15:42Z`` -> ``D:20261017071542Z`` (PDF date string)."""
    digits = re.sub(r"[^0-9]", "", iso)[:14].ljust(14, "0")
    return f"D:{digits}Z"


def document_number(result: "ShopfloorInspectionResultV1") -> str:
    """Report number: SKU plus the first 8 hex digits of the scan's sha256
    (one report per scan; stable across re-renders)."""
    return f"IR-{result.sku}-{_sha8(result.scan.sha256)}"


#: Display order of a bend's characteristics: flange lengths, angle, position, backgauge.
_KIND_ORDER = {"FLANGE_LENGTH": 0, "INCLUDED_ANGLE": 2, "BEND_LINE_POSITION": 3, "BACKGAUGE_LENGTH": 4}


def _ordered(chars: Sequence["CharacteristicResultV1"]) -> list["CharacteristicResultV1"]:
    def key(item: tuple[int, "CharacteristicResultV1"]) -> tuple[int, int, int]:
        i, c = item
        side = 1 if (c.kind == "FLANGE_LENGTH" and c.side == "B") else 0
        return (_KIND_ORDER.get(c.kind, 5), side, i)

    return [c for _, c in sorted(enumerate(chars), key=key)]


def char_number(row: "BendRowV1", index: int) -> str:
    """Characteristic number: ``<shop-order sequence>.<position in the bend>``."""
    return f"{row.sequence_index}.{index + 1}"


# --------------------------------------------------------------------------
# Language
# --------------------------------------------------------------------------


class _L:
    """Primary-language typesetting helpers (Hebrew right-to-left or English)."""

    def __init__(self, lang: str) -> None:
        if lang not in LANGS:
            raise ValueError(f"unknown report language {lang!r}; expected one of {LANGS}")
        self.lang = lang
        self.rtl = lang == "he"
        self.i = 1 if self.rtl else 0

    # strings
    def s(self, pair: Bi) -> str:
        return pair[self.i]

    def other(self, pair: Bi) -> str:
        return pair[1 - self.i]

    def t(self, key: str) -> str:
        return self.s(TEXT[key])

    @property
    def sep(self) -> str:
        """Separator between runs: RLM-guarded in Hebrew so neighbouring
        left-to-right items (numbers, labels) do not merge into one run."""
        return f" {_RLM}·{_RLM} " if self.rtl else " · "

    # inline html of a primary-language string (Hebrew guarded by RLMs)
    def x(self, text: str) -> str:
        return f"{_RLM}{_e(text)}{_RLM}" if self.rtl else _e(text)

    def xo(self, text: str) -> str:
        """Inline html of a secondary-language string."""
        return _e(text) if self.rtl else _he(text)

    # blocks
    def block(self, inner_html: str, cls: str = "", style: str = "") -> str:
        attrs = (f' class="{cls}"' if cls else "") + (f' style="{style}"' if style else "")
        if self.rtl:
            return f'<div dir="rtl"{attrs}>{_RLM}{inner_html}{_RLM}</div>'
        return f"<div{attrs}>{inner_html}</div>"

    def other_block(self, text: str, cls: str = "tiny") -> str:
        """A secondary-language line (always laid out in its own direction)."""
        if self.rtl:  # English line inside a right-to-left document: right-aligned, left-to-right
            return f'<div dir="rtl" class="{cls}">{_RLM}{_LRM}{_e(text)}{_LRM}{_RLM}</div>'
        return f'<div dir="rtl" class="{cls}">{_RLM}{_e(text)}{_RLM}</div>'

    def label(self, pair: Bi, cls: str = "k") -> str:
        """Field label: primary language, the other language underneath in small type."""
        return self.block(self.x(self.s(pair)), cls) + self.other_block(self.other(pair))

    # tables: the html column order is the visual left-to-right order, so a
    # Hebrew table lists its columns reversed (first column on the right).
    def row(self, cells: Sequence[str], cls: str = "") -> str:
        seq = list(reversed(cells)) if self.rtl else list(cells)
        return f"<tr{f' class={chr(34)}{cls}{chr(34)}' if cls else ''}>{''.join(seq)}</tr>"

    def td(self, inner_html: str, *, width: float | None = None, cls: str = "", rowspan: int = 1, colspan: int = 1) -> str:
        """Text cell (right-to-left block in Hebrew)."""
        attrs = ""
        if cls:
            attrs += f' class="{cls}"'
        if rowspan > 1:
            attrs += f' rowspan="{rowspan}"'
        if colspan > 1:
            attrs += f' colspan="{colspan}"'
        style = f"width:{width:.1f}pt;" if width is not None else ""
        if self.rtl:
            if _has_hebrew(inner_html):
                attrs += ' dir="rtl"'
            else:  # identifiers, numbers, codes: a left-to-right paragraph, aligned to the right edge
                style += "text-align:right;"
        if style:
            attrs += f' style="{style}"'
        return f"<td{attrs}>{inner_html}</td>"


_CELL_INSET_PT = 8.8  # horizontal padding (2 x 4) + rules


def _w(points: float) -> float:
    """CSS content width of a column ``points`` wide (MuPDF ignores % widths)."""
    return max(points - _CELL_INSET_PT, 4.0)


def _num_td(text_html: str, *, width: float | None = None, cls: str = "num", rowspan: int = 1) -> str:
    style = f' style="width:{width:.1f}pt"' if width is not None else ""
    span = f' rowspan="{rowspan}"' if rowspan > 1 else ""
    return f'<td class="{cls}"{span}{style}>{text_html}</td>'


def _head(lng: _L, cols: Sequence[tuple[str, float, bool]]) -> str:
    """Header row; ``cols`` = (TEXT key, width pt, numeric)."""
    cells = []
    for key, width, numeric in cols:
        pair = TEXT[key]
        align = "right" if numeric else ("right" if lng.rtl else "left")
        if lng.rtl and not numeric:
            inner = f'<div dir="rtl" class="h">{_RLM}{_e(pair[1])}{_RLM}<br/><span class="tiny">{_LRM}{_e(pair[0])}{_LRM}</span></div>'
        elif lng.rtl:
            inner = f'<div class="h" style="text-align:{align}">{_he(pair[1])}<br/><span class="tiny">{_e(pair[0])}</span></div>'
        else:
            inner = f'<div class="h" style="text-align:{align}">{_e(pair[0])}<br/><span class="tiny">{_he(pair[1])}</span></div>'
        cells.append(f'<th style="width:{_w(width):.1f}pt">{inner}</th>')
    return lng.row(cells)


def _decision_td(lng: _L, char: "CharacteristicResultV1", width: float | None = None) -> str:
    style = f' style="width:{width:.1f}pt"' if width is not None else ""
    if char.decision == "NOT_APPLICABLE":
        word = lng.s(("n/a", "לא רלוונטי"))
    else:
        word = lng.s((token(char.decision)["en"], token(char.decision)["he"]))
    inner = (f'<div dir="rtl" class="badge d-{_e(char.decision)}">{_RLM}{_e(word)}{_RLM}</div>' if lng.rtl
             else f'<div class="badge d-{_e(char.decision)}">{_e(word)}</div>')
    # backgrounds go on a <div>: MuPDF Story repaints <td>/<tr> backgrounds on later pages
    return f'<td class="res"{style}>{inner}</td>'


def _value_tds(char: "CharacteristicResultV1", widths: Sequence[float] | None = None) -> list[str]:
    """Nominal, tolerance, actual, deviation, uncertainty (left-to-right number cells)."""
    unit = char.unit
    signed_pos = char.kind == "BEND_LINE_POSITION"
    nominal = _num(char.expected, unit, signed=signed_pos)
    if char.state == "MEASURED":
        actual = _num(char.actual, unit, signed=signed_pos)
        dev = _num(char.deviation, unit, signed=True)
        unc = "±" + _num(char.uncertainty, unit)
    else:
        actual = dev = unc = _MISSING
    vals = [nominal, _tol(char.tolerance, unit), actual, dev, unc]
    cls = ["num", "num", "num b" if char.decision == "FAIL" else "num", "num", "num muted"]
    ws = list(widths) if widths is not None else [None] * 5
    return [_num_td(v, width=w, cls=c) for v, w, c in zip(vals, ws, cls)]


def _kind_text(lng: _L, char: "CharacteristicResultV1") -> str:
    pair = kind_label(char.kind, char.side)
    extra = f" S{char.stage_index}" if char.kind == "BACKGAUGE_LENGTH" and char.stage_index is not None else ""
    return lng.x(lng.s(pair)) + (" " + _ltr(extra.strip()) if extra else "")


def _note_refs(chars: Iterable["CharacteristicResultV1"]) -> dict[str, int]:
    """Reason code -> note number, in order of first appearance."""
    refs: dict[str, int] = {}
    for c in chars:
        if c.reason and c.reason not in refs:
            refs[c.reason] = len(refs) + 1
    return refs


def _note_text(lng: _L, char: "CharacteristicResultV1", refs: Mapping[str, int]) -> str:
    """Notes cell: a numbered reference to the reason (listed under the table) and a reference marker."""
    bits = []
    if char.reason:
        bits.append(f'<span class="small">{_ltr(f"({refs[char.reason]})")}</span>')
    if not char.required:
        bits.append(f'<span class="small muted">{lng.x(lng.s(("ref.", "ייחוס")))}</span>')
    return " ".join(bits)


# --------------------------------------------------------------------------
# Flow items and pagination
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Block:
    """Atomic HTML fragment (moved whole to the next page when it does not fit)."""

    html: str


@dataclass(frozen=True)
class _Table:
    """A table whose row groups may continue on the next page; the title and
    the header row are repeated there (``cont_title`` instead of ``title``)."""

    title: str
    head: str
    rows: tuple[str, ...]
    cont_title: str = ""
    cls: str = ""

    def html(self, rows: Sequence[str], *, first: bool) -> str:
        cls = f' class="{self.cls}"' if self.cls else ""
        return f"{self.title if first else self.cont_title}<table{cls}>{self.head}{''.join(rows)}</table>"


@dataclass(frozen=True)
class _PageBreak:
    pass


_Flow = _Block | _Table | _PageBreak


def _paginate(items: Sequence[_Flow], fits: Callable[[str], bool]) -> list[str]:
    """Greedy pagination into per-page HTML bodies.

    Every page is later laid out by its *own* Story: MuPDF repaints table-cell
    backgrounds of earlier pages onto later pages of one Story, and one Story
    per page also lets tables repeat their header row.
    """
    pages: list[str] = []
    cur: list[str] = []

    def flush() -> None:
        if cur:
            pages.append("".join(cur))
            cur.clear()

    for item in items:
        if isinstance(item, _PageBreak):
            flush()
        elif isinstance(item, _Block):
            if cur and not fits("".join(cur) + item.html):
                flush()
            cur.append(item.html)  # an oversized block overflows onto extra pages at render time
        else:
            i, first = 0, True
            rows = item.rows
            while i < len(rows):
                taken: list[str] = []
                while i < len(rows) and fits("".join(cur) + item.html([*taken, rows[i]], first=first)):
                    taken.append(rows[i])
                    i += 1
                if taken:
                    cur.append(item.html(taken, first=first))
                    first = False
                    if i < len(rows):
                        flush()
                elif cur:
                    flush()
                else:  # one row group taller than a page: give it a page of its own
                    cur.append(item.html([rows[i]], first=first))
                    i += 1
                    first = False
                    flush()
    flush()
    return pages


# --------------------------------------------------------------------------
# Page 1
# --------------------------------------------------------------------------


def _title(lng: _L) -> str:
    return (lng.block(lng.x(lng.t("doc_title")), style="font-size:16pt;font-weight:bold")
            + lng.block(lng.x(lng.t("doc_subtitle")), "sub"))


def _title_block(lng: _L, result: "ShopfloorInspectionResultV1", recipe: "PartRecipeV1 | None") -> str:
    """Four-column title block grid (AS9102 Form 1 style): label over value."""
    ctx = result.context
    material = _MISSING
    if recipe is not None:
        thick = f"{recipe.thickness_mm:.2f} mm" if recipe.thickness_mm is not None else _MISSING
        material = f"{recipe.material or _MISSING} · {thick}"
    revision = recipe.part_revision if recipe is not None and recipe.part_revision else _MISSING
    stage = INSPECTION_STAGE_LABELS[result.stage_decision.inspection_stage]
    #: rows of (label, value html, colspan)
    grid: list[list[tuple[Bi, str, int]]] = [
        [(TEXT["part_no"], _ltr(result.sku), 1), (TEXT["revision"], _ltr(revision), 1),
         (TEXT["part_name"], _ltr(result.part_name or _MISSING), 2)],
        [(TEXT["material"], _ltr(material), 1),
         (TEXT["tol_standard"], _ltr(recipe.default_tolerances.source if recipe is not None else _MISSING), 1),
         (TEXT["recipe"], _ltr(f"{result.recipe.recipe_id} (v{result.recipe.version})"), 1),
         (TEXT["inspection_stage"], lng.x(lng.s(stage)), 1)],
        [(TEXT["job"], _ltr(ctx.job_id or _MISSING), 2), (TEXT["serial"], _ltr(ctx.serial or _MISSING), 1),
         (TEXT["doc_no"], _ltr(document_number(result)), 1)],
        [(TEXT["scan_file"], _ltr(ctx.scan_file_name or _MISSING), 2),
         (TEXT["station_scanner"], _ltr(ctx.station_id or _MISSING), 1),
         (TEXT["scan_date"], _ltr(ctx.scan_timestamp or ctx.received_at or _MISSING), 1)],
        [(TEXT["inspected"], _ltr(result.created_at), 1), (TEXT["operator"], "", 2),  # operator: written in by hand
         (TEXT["bundle"], _ltr(result.engine.bundle_id or _MISSING), 1)],
    ]
    col = _CONTENT_W / 4.0
    sizer = lng.row([f'<td class="sz" style="width:{_w(col):.1f}pt"></td>'] * 4)
    rows = [sizer]
    for line in grid:
        rows.append(lng.row([lng.td(lng.label(pair) + lng.block(value or "&#160;", "v"), colspan=span)
                             for pair, value, span in line]))
    return f'<table class="tb">{"".join(rows)}</table>'


def _verdict_colour(state: str) -> str:
    tone = token(state)["tone"]
    return _VERDICT.get(tone, "#4B5563" if tone in ("unknown", "inactive") else _ACCENT)


def _worst_line(lng: _L, result: "ShopfloorInspectionResultV1") -> str:
    worst = result.summary.worst
    if worst is None:
        return _MISSING
    row = next((r for r in result.bends if r.bend_uid == worst.bend_uid), None)
    label = bend_display_label(result, row) if row is not None else worst.characteristic_id
    number = ""
    if row is not None:
        for i, c in enumerate(_ordered(row.characteristics)):
            if c.characteristic_id == worst.characteristic_id:
                number = char_number(row, i)
    side = worst.characteristic_id.rsplit("/", 1)[-1] if worst.kind == "FLANGE_LENGTH" else None
    unit = "deg" if worst.kind == "INCLUDED_ANGLE" else "mm"
    tok = token(worst.decision)
    d = lng.sep
    return (f"{_ltr(number)}{d}{_ltr(label)}{d}{lng.x(lng.s(kind_label(worst.kind, side)))}{d}"
            f"{lng.x(lng.s(('dev.', 'סטייה')))} {_ltr(_num(worst.deviation, unit, signed=True))}{d}"
            f"{lng.x(lng.s((tok['en'], tok['he'])))}")


def _summary(lng: _L, result: "ShopfloorInspectionResultV1") -> str:
    s = result.summary
    tok = token(s.state)
    stage = result.stage_decision
    validity = result.inspection_validity
    verdict = (
        f'<div class="verdict" style="background-color:{_verdict_colour(s.state)}">'
        + lng.block(lng.x(lng.t("result_label")), style="font-size:7pt")
        + lng.block(lng.x(lng.s((tok["en"], tok["he"]))), style="font-size:14pt;font-weight:bold")
        + lng.block(lng.xo(lng.other((tok["en"], tok["he"]))) if lng.rtl else _he(tok["he"]), style="font-size:8pt")
        + "</div>"
    )
    progress = (lng.block(lng.x(lng.t("stage_progress")), "k")
                + lng.block(f"{_ltr(f'{s.bends_formed}/{s.bends_total}')} {lng.x(lng.t('bends_formed_of'))}",
                            style="font-size:11pt;font-weight:bold"))
    counts = [("count_pass", s.required_pass, "pass"), ("count_review", s.required_review, "review"),
              ("count_fail", s.required_fail, "fail"), ("count_unknown", s.required_unknown, None)]
    cw = (_CONTENT_W * 0.62) / 4.0
    count_cells = []
    for key, n, tone in counts:
        bar = _VERDICT[tone] if tone else "#9CA3AF"
        count_cells.append(
            f'<td style="width:{_w(cw):.1f}pt;border-top:2pt solid {bar}"><div class="cnum">{n}</div>'
            + lng.block(lng.x(lng.t(key)), "k") + "</td>")
    count_table = (f'<table class="counts">{lng.row(count_cells)}</table>'
                   + lng.block(f"{_ltr(s.characteristics_required)} {lng.x(lng.t('req_chars_of'))}", "tiny"))
    status = STAGE_STATUS_LABELS[stage.status]
    kv: list[tuple[Bi, str]] = [
        (TEXT["worst"], _worst_line(lng, result)),
        (TEXT["stage_decision"], lng.x(lng.s(status)) + f"{lng.sep}{lng.x(lng.s((token(stage.decision)['en'], token(stage.decision)['he'])))}"),
        (TEXT["validity_state"], lng.x(lng.s(VALIDITY_LABELS[validity.state]))),
    ]
    if s.unplanned_creases:
        kv.append((TEXT["creases"], _ltr(s.unplanned_creases)))
    left_w = _CONTENT_W * 0.30
    right_w = _CONTENT_W - left_w - 10
    right = progress + count_table + _kv(lng, kv, key_width=right_w * 0.32, value_width=right_w * 0.68)
    cells = [f'<td style="width:{left_w:.1f}pt">{verdict}</td>', f'<td style="width:10pt"></td>',
             f'<td style="width:{right_w:.1f}pt">{right}</td>']
    return f'<table class="sum">{lng.row(cells)}</table>'


def _kv(lng: _L, rows: Iterable[tuple[Bi, str]], *, key_width: float, value_width: float, cls: str = "kv") -> str:
    out = []
    for i, (k, v) in enumerate(rows):
        kw = _w(key_width) if i == 0 else None
        vw = _w(value_width) if i == 0 else None
        out.append(lng.row([lng.td(lng.label(k), width=kw), lng.td(v, width=vw)]))
    return f'<table class="{cls}">{"".join(out)}</table>'


def _overview(lng: _L, views: Mapping[str, bytes] | None, note: str | None) -> str:
    head = f"<h3>{lng.block(lng.x(lng.t('overview')))}</h3>"
    if views and "overview" in views:
        return head + f'<div style="text-align:center"><img src="view_overview.png" style="width:{_CONTENT_W - 20:.0f}pt"/></div>'
    reason = f" ({_ltr(note)})" if note else ""
    return head + lng.block(lng.x(lng.t("overview_none")) + reason, "muted small")


def _signoff(lng: _L) -> str:
    def check(key: str) -> str:
        return lng.row([f'<td style="width:9pt"><div class="box">&#160;</div></td>',
                        lng.td(lng.x(lng.t(key)), width=80.0)])

    boxes = "".join(check(k) for k in ("accept", "rework", "reject"))
    col = _CONTENT_W / 4.0
    cells = [
        lng.td(lng.label(TEXT["disposition"]) + f'<table class="chk">{boxes}</table>', width=_w(col)),
        lng.td(lng.label(TEXT["name"]) + "<br/><br/>", width=_w(col)),
        lng.td(lng.label(TEXT["signature"]), width=_w(col)),
        lng.td(lng.label(TEXT["date"]), width=_w(col)),
    ]
    en, he = TEXT["authority"]
    auth = (f'<div class="auth"><div class="b">{_e(en)}</div><div dir="rtl" class="b">{_RLM}{_e(he)}{_RLM}</div>'
            f"</div>")
    return (f"<h3>{lng.block(lng.x(lng.t('signoff')))}</h3>"
            f'<table class="sign">{lng.row(cells)}</table>{auth}')


# --------------------------------------------------------------------------
# 1. Characteristic results
# --------------------------------------------------------------------------

#: (TEXT key, width pt, numeric) - sums to the content width (511 pt).
_CHAR_COLS: tuple[tuple[str, float, bool], ...] = (
    ("no", 32.0, False), ("bend", 52.0, False), ("characteristic", 88.0, False), ("nominal", 50.0, True),
    ("tol", 46.0, True), ("actual", 50.0, True), ("dev", 44.0, True), ("u", 40.0, True),
    ("result_col", 58.0, False), ("notes", 51.0, False),
)


def _section(lng: _L, number: str, key: str, *, note_key: str | None = None) -> str:
    title = lng.block(f"{_ltr(number)}&#160;&#160;{lng.x(lng.t(key))}" if number else lng.x(lng.t(key)))
    out = f"<h2>{title}</h2>"
    if note_key:
        out += lng.block(lng.x(lng.t(note_key)), "muted small")
    return out


def _bend_rows(lng: _L, result: "ShopfloorInspectionResultV1", row: "BendRowV1", refs: Mapping[str, int]) -> str:
    widths = [_w(w) for _, w, _ in _CHAR_COLS]
    chars = _ordered(row.characteristics)
    n = len(chars)
    label = bend_display_label(result, row)
    bend_cell = f'<span class="b">{_ltr(label)}</span>'
    if row.state != "FORMED":
        tok = token(row.state)
        bend_cell += f'<br/><span class="small muted">{lng.x(lng.s((tok["en"], tok["he"])))}</span>'
    out = []
    for i, char in enumerate(chars):
        cells = [lng.td(_ltr(char_number(row, i)), width=widths[0] if i == 0 else None, cls="")]
        if i == 0:
            cells.append(lng.td(bend_cell, width=widths[1], rowspan=n))
        cells.append(lng.td(_kind_text(lng, char), width=widths[2] if i == 0 else None))
        cells += _value_tds(char, widths[3:8] if i == 0 else None)
        cells.append(_decision_td(lng, char, widths[8] if i == 0 else None))
        cells.append(lng.td(_note_text(lng, char, refs), width=widths[9] if i == 0 else None))
        out.append(lng.row(cells))
    return "".join(out)


def _move_groups(bends: Sequence["BendRowV1"]) -> list[tuple[int | None, list["BendRowV1"]]]:
    groups: list[tuple[int | None, list[BendRowV1]]] = []
    for row in bends:
        move = row.machine_move_index
        if groups and move is not None and groups[-1][0] == move:
            groups[-1][1].append(row)
        else:
            groups.append((move, [row]))
    return groups


def _group_row(lng: _L, result: "ShopfloorInspectionResultV1", move: int, rows: Sequence["BendRowV1"]) -> str:
    labels = ", ".join(bend_display_label(result, r) for r in rows)
    text = (f"{lng.x(lng.s(('Move', 'מהלך')))} {_ltr(move)}{lng.sep}{_ltr(labels)}" if lng.rtl
            else f"Move {move} · {_e(labels)} · {_he(f'מהלך {move}')}")
    inner = (f'<div dir="rtl" class="grp">{_RLM}{text}{_RLM}</div>' if lng.rtl else f'<div class="grp">{text}</div>')
    return f'<tr><td class="grp" colspan="{len(_CHAR_COLS)}">{inner}</td></tr>'


def _reason_notes(lng: _L, refs: Mapping[str, int]) -> str:
    if not refs:
        return ""
    r = _RLM if lng.rtl else ""
    items = "".join(
        lng.block(f'{_ltr(f"({n})")} {r}{_ltr(code)} {r}—{r} {lng.x(lng.s(reason_label(code)))}', "small muted")
        for code, n in refs.items())
    return lng.block(lng.x(lng.t("notes_title")), "small b", style="margin-top:4pt") + items


def _chars_section(lng: _L, result: "ShopfloorInspectionResultV1") -> list[_Flow]:
    formed = [r for r in result.bends if r.state != "NOT_YET_FORMED"]
    nyf = [r for r in result.bends if r.state == "NOT_YET_FORMED"]
    title = _section(lng, "1", "sec_chars", note_key="sec_chars_note")
    items: list[_Flow] = [_PageBreak()]
    if not result.bends:
        return items + [_Block(title + lng.block(lng.x(lng.t("none"))))]
    if formed:
        refs = _note_refs(c for r in formed for c in _ordered(r.characteristics))
        groups: list[str] = []
        for move, rows in _move_groups(formed):
            for k, row in enumerate(rows):
                prefix = _group_row(lng, result, move, rows) if (move is not None and k == 0) else ""
                groups.append(prefix + _bend_rows(lng, result, row, refs))
        cont = _section(lng, "1", "sec_chars")
        items.append(_Table(title=title, head=_head(lng, _CHAR_COLS), rows=tuple(groups), cont_title=cont, cls="grid"))
        notes = _reason_notes(lng, refs)
        if any(not c.required for r in formed for c in r.characteristics):
            notes += lng.block(f"{lng.x(lng.s(('ref.', 'ייחוס')))} — {lng.x(lng.t('ref_only'))}", "small muted")
        if notes:
            items.append(_Block(notes))
    else:
        items.append(_Block(title))
    if nyf:
        items += _nyf_section(lng, result, nyf)
    return items


_NYF_COLS: tuple[tuple[str, float, bool], ...] = (
    ("seq", 40.0, False), ("bend", 60.0, False), ("move", 50.0, False), ("state", 120.0, False), ("notes", 241.0, False),
)


def _nyf_section(lng: _L, result: "ShopfloorInspectionResultV1", rows: Sequence["BendRowV1"]) -> list[_Flow]:
    title = (f"<h3>{lng.block(lng.x(lng.t('sec_nyf')))}</h3>" + lng.block(lng.x(lng.t("sec_nyf_note")), "muted small"))
    widths = [_w(w) for _, w, _ in _NYF_COLS]
    body = []
    for k, row in enumerate(rows):
        tok = token(row.state)
        first = k == 0
        cells = [
            lng.td(_ltr(row.sequence_index), width=widths[0] if first else None),
            lng.td(_ltr(bend_display_label(result, row)), width=widths[1] if first else None),
            lng.td(_ltr(row.machine_move_index if row.machine_move_index is not None else _MISSING), width=widths[2] if first else None),
            lng.td(lng.x(lng.s((tok["en"], tok["he"]))), width=widths[3] if first else None),
            lng.td(lng.x(lng.s(reason_label("BEND_NOT_YET_FORMED"))), width=widths[4] if first else None),
        ]
        body.append(lng.row(cells, cls="nyf"))
    return [_Table(title=title, head=_head(lng, _NYF_COLS), rows=tuple(body), cont_title=title)]


# --------------------------------------------------------------------------
# 2. Measurements
# --------------------------------------------------------------------------

_GM_COLS: tuple[tuple[str, float, bool], ...] = (
    ("no", 32.0, False), ("measurement", 140.0, False), ("nominal", 50.0, True), ("tol", 46.0, True),
    ("actual", 50.0, True), ("dev", 44.0, True), ("u", 40.0, True), ("result_col", 58.0, False), ("notes", 51.0, False),
)


def _measurements_section(lng: _L, result: "ShopfloorInspectionResultV1") -> list[_Flow]:
    title = _section(lng, "2", "sec_measurements")
    items: list[_Flow] = []
    if not result.general_measurements:
        items.append(_Block(title + lng.block(lng.x(lng.t("none")), "muted")))
    else:
        widths = [_w(w) for _, w, _ in _GM_COLS]
        refs = _note_refs(gm.characteristic for gm in result.general_measurements)
        body = []
        for k, gm in enumerate(result.general_measurements):
            c = gm.characteristic
            first = k == 0
            cells = [lng.td(_ltr(f"M{k + 1}"), width=widths[0] if first else None),
                     lng.td(_ltr(gm.label), width=widths[1] if first else None)]
            cells += _value_tds(c, widths[2:7] if first else None)
            cells.append(_decision_td(lng, c, widths[7] if first else None))
            cells.append(lng.td(_note_text(lng, c, refs), width=widths[8] if first else None))
            body.append(lng.row(cells))
        items.append(_Table(title=title, head=_head(lng, _GM_COLS), rows=tuple(body), cont_title=title, cls="grid"))
        notes = _reason_notes(lng, refs)
        if any(not gm.characteristic.required for gm in result.general_measurements):
            notes += lng.block(f"{lng.x(lng.s(('ref.', 'ייחוס')))} — {lng.x(lng.t('ref_only'))}", "small muted")
        if notes:
            items.append(_Block(notes))
    if result.unplanned_creases:
        ctitle = f"<h3>{lng.block(lng.x(lng.t('creases')))}</h3>"
        rows = tuple(
            lng.row([lng.td(_ltr(c.crease_id)), lng.td(_ltr(c.reason)), _num_td(_num(c.included_deg, "deg")),
                     lng.td(_ltr(c.note))])
            for c in result.unplanned_creases
        )
        head = _head(lng, (("id_col", 60.0, False), ("reason", 140.0, False), ("angle_col", 70.0, True), ("notes", 241.0, False)))
        items.append(_Table(title=ctitle, head=head, rows=rows, cont_title=ctitle))
    return items


# --------------------------------------------------------------------------
# 3. Detections (unverified): the zero-shot overlay, never measurements
# --------------------------------------------------------------------------

#: (TEXT key, width pt, numeric) - sums to the content width (511 pt).
_ZS_COLS: tuple[tuple[str, float, bool], ...] = (
    ("no", 32.0, False), ("zs_kind", 160.0, False), ("bend", 60.0, False), ("zs_length", 60.0, True),
    ("angle_col", 60.0, True), ("zs_score", 50.0, True), ("result_col", 89.0, False),
)


def _zs_lowered_block(lng: _L, overlay: Any) -> str:
    """Page-1 notice when the overlay lowered a pass-like verdict to REVIEW (never green)."""
    if overlay is None or not overlay.verdict_lowered:
        return ""
    tone = _VERDICT["review"]
    return (f'<div style="border:1pt dashed {tone};padding:3pt;margin-top:4pt">'
            + lng.block(lng.x(lng.t("zs_lowered")), "small b") + "</div>")


def _zs_section(lng: _L, result: "ShopfloorInspectionResultV1", overlay: Any) -> list[_Flow]:
    """The overlay's lines in their own section (dashed, review / unverified colours, never green)."""
    if overlay is None:
        return []
    refused = overlay.inspection_validity != "VALID"
    title = _section(lng, "3", "sec_zs", note_key="sec_zs_note")
    if refused:
        title += lng.block(lng.x(lng.t("sec_zs_refused")), "muted small")
    labels = {r.bend_uid: bend_display_label(result, r) for r in result.bends}
    for f in overlay.flags:
        title += lng.block(f'{lng.x(lng.t("zs_flag_" + f.code))} {_ltr("(" + ", ".join(f.line_ids) + ")")}', "small b")
    for a in getattr(overlay, "angle_checks", ()):
        detail = f"({labels.get(a.bend_uid, a.bend_uid)}: E3 {a.e3_angle_deg:.1f}° / zs {a.zs_angle_deg:.1f}°)"
        title += lng.block(f'{lng.x(lng.t("zs_flag_" + a.code))} {_ltr(detail)}', "small b")
    if not overlay.lines:
        return [_Block(title + lng.block(lng.x(lng.t("none")), "muted"))]
    widths = [_w(w) for _, w, _ in _ZS_COLS]
    body = []
    for k, ln in enumerate(overlay.lines):
        first = k == 0
        tone = _VERDICT["review"] if ln.decision == "REVIEW" else _MUTED
        decision = (f'<span style="color:{tone};font-weight:bold;border:0.8pt dashed {tone};padding:0 2pt">'
                    f'{lng.x(lng.t("zs_decision_" + ln.decision))}</span>')
        cells = [
            lng.td(_ltr(ln.line_id), width=widths[0] if first else None),
            lng.td(lng.x(lng.t("zs_kind_" + ln.kind)), width=widths[1] if first else None),
            lng.td(_ltr(labels.get(ln.bend_uid, _MISSING) if ln.bend_uid else _MISSING), width=widths[2] if first else None),
            _num_td(_num(ln.length_mm, "mm", digits=1), width=widths[3] if first else None),
            (lng.td(lng.x(lng.t("zs_angle_joggle")), width=widths[4] if first else None)
             if ln.angle_status == "NOT_MEASURED_JOGGLE" else _num_td(_num(ln.incl_deg, "deg", digits=1), width=widths[4] if first else None)),
            _num_td(_ltr(f"{ln.score:.2f}"), width=widths[5] if first else None),
            lng.td(decision, width=widths[6] if first else None),
        ]
        body.append(lng.row(cells, cls="zs"))
    return [_Table(title=title, head=_head(lng, _ZS_COLS), rows=tuple(body), cont_title=_section(lng, "3", "sec_zs"),
                   cls="grid")]


# --------------------------------------------------------------------------
# Appendix
# --------------------------------------------------------------------------


def _views_appendix(lng: _L, views: Mapping[str, bytes] | None, result: "ShopfloorInspectionResultV1", note: str | None) -> list[_Flow]:
    out = _section(lng, "A", "app_views")
    detail = [v for v in ("iso", "front", "top", "right") if views and v in views]
    if not detail:
        suffix = f" ({_ltr(note)})" if note else ""
        return [_PageBreak(), _Block(out + lng.block(lng.x(lng.t("no_views")) + suffix, "muted"))]
    va = result.viewer_assets
    if va.deviations is not None:
        caption = (f"Scan points coloured by signed deviation from stage model {va.deviation_reference_stage or _MISSING} "
                    f"(±{va.heatmap_band_mm:.2f} mm band; + = outward). Grey dashed: nominal bend line; solid: measured "
                    f"OML line coloured by the bend's worst result. Labels: shop order, bend, uid code.",
                    f"נקודות הסריקה צבועות לפי סטייה מסומנת ממודל השלב (פס ±{va.heatmap_band_mm:.2f} מ\"מ; + כלפי חוץ). "
                    "קו מקווקו אפור: קו כיפוף נומינלי; קו רציף: קו OML שנמדד, בצבע התוצאה הגרועה של הכיפוף.")
    else:
        caption = (f"{TEXT['no_heatmap'][0]}: {va.heatmap_unavailable_reason or _MISSING}. Grey dashed: nominal bend line; "
                   "solid: measured OML line. Labels: shop order, bend, uid code.",
                   "מפת סטיות אינה זמינה. קו מקווקו אפור: קו כיפוף נומינלי; קו רציף: קו OML שנמדד.")
    cells = [f"<td>{_img(f'view_{v}.png', (_CONTENT_W - 12) / 2)}</td>" for v in detail]
    rows = "".join(f"<tr>{''.join(cells[i:i + 2])}</tr>" for i in range(0, len(cells), 2))
    text = lng.block(lng.x(lng.s(caption)), "small muted")
    if note:
        text += f'<p class="tiny">{_e(note)}</p>'
    return [_PageBreak(), _Block(out + f'<table class="views">{rows}</table>' + text)]


def _img(name: str, width_pt: float) -> str:
    return f'<img src="{_e(name)}" style="width:{width_pt:.0f}pt"/>'


_CONVENTIONS: tuple[Bi, ...] = (
    ("Decision with guard band: PASS only if |dev| + U ≤ tol; FAIL only if |dev| − U > tol; otherwise REVIEW. "
     "An undetermined value never becomes PASS or FAIL, and a nominal is never copied into the actual value.",
     "החלטה עם פס ביטחון: תקין רק אם |סטייה| + U קטן או שווה לטולרנס; חריגה רק אם |סטייה| − U גדול מהטולרנס; "
     "אחרת לבדיקה. ערך שלא הוכרע לעולם אינו הופך לתקין או לחריגה."),
    ("U is the engine's expanded measurement uncertainty of the value (±, same unit).",
     "U היא אי-הוודאות המורחבת של המדידה כפי שחישב המנוע (±, באותה יחידה)."),
    ("Angles are included angles between the two flanges: 180° is flat.",
     "הזוויות הן זוויות כלואות בין שתי הכנפיים: 180° משמעו שטוח."),
    ("Flange length (OML): from the bend's outside-mould-line virtual sharp to the far boundary of the flange "
     "(free edge, or the OML sharp of the adjacent bend), in the flange plane, perpendicular to the bend.",
     "אורך כנף (OML): מהחוד הווירטואלי של קו התבנית החיצוני של הכיפוף ועד הגבול הרחוק של הכנף (קצה חופשי או קו OML "
     "של הכיפוף השכן), במישור הכנף, בניצב לכיפוף."),
    ("Side A is the flange whose outward direction has the larger +X component (ties: +Y, then +Z).",
     "צד A הוא הכנף שכיוונה החיצוני בעל רכיב X+ גדול יותר (בשוויון: Y+, ואחר כך Z+)."),
    ("Bend-line position: signed offset of the measured OML sharp from the nominal one, along side A's outward "
     "direction; the expected value is 0.",
     "מיקום קו הכיפוף: היסט מסומן של חוד ה-OML הנמדד מהנומינלי, בכיוון החיצוני של צד A; הערך הצפוי הוא 0."),
    ("Backgauge length (mid-sequence): from the OML of the just-formed bend across still-flat material to the far "
     "boundary; the nominal comes from the stage model.",
     "אורך מעצור אחורי (באמצע הרצף): מקו ה-OML של הכיפוף האחרון שבוצע, דרך החומר השטוח, ועד הגבול הרחוק; הנומינלי "
     "ממודל השלב."),
    ("All geometry is in the CAD model frame (cad_model_mm). Heatmap deviations are signed along the stage model's "
     "outward normal.",
     "כל הגאומטריה במערכת הצירים של מודל ה-CAD. סטיות מפת החום מסומנות לאורך הנורמל החיצוני של מודל השלב."),
)


def _method_appendix(lng: _L, result: "ShopfloorInspectionResultV1", recipe: "PartRecipeV1 | None",
                     bundle_id: str | None, extra_warnings: Sequence[str]) -> list[_Flow]:
    from contracts.production.inspection_result import iter_characteristics

    items: list[_Flow] = []
    head = _section(lng, "B", "app_method")
    conv = "".join(lng.block(lng.x(lng.s(pair)), "small", style="margin:0 0 2.5pt 0") for pair in _CONVENTIONS)
    items.append(_Block(head + f"<h3>{lng.block(lng.x(lng.t('conventions')))}</h3>" + conv))

    stage = result.stage_decision
    v = result.inspection_validity
    reg = v.registration
    cov = v.coverage
    eng = result.engine

    def opt(x: float | None, fmt: str) -> str:
        return _MISSING if x is None else format(x, fmt)

    def yn(flag: bool | None) -> str:
        return {True: "yes", False: "no", None: _MISSING}[flag]

    declared = f"{stage.declared_stage} ({stage.declared_stage_source})" if stage.declared_stage is not None else _MISSING
    rows: list[tuple[Bi, str]] = [
        (TEXT["inspection_stage"], lng.x(lng.s(INSPECTION_STAGE_LABELS[stage.inspection_stage]))),
        (("Declared / decoded stage", "שלב מוצהר / שזוהה"),
         _ltr(f"{declared} / {stage.inferred_stage if stage.inferred_stage is not None else _MISSING}/{stage.bends_total}")),
        (("Stage model · status", "מודל שלב · סטטוס"),
         _ltr(stage.stage_model_id or _MISSING) + lng.sep + lng.x(lng.s(STAGE_STATUS_LABELS[stage.status]))
         + (f"{lng.sep}{_ltr(', '.join(stage.flags))}" if len(stage.flags) > 1 else "")),
        (("Part complete", "החלק הושלם"),
         lng.x(lng.s({True: ("yes", "כן"), False: ("no", "לא"), None: (_MISSING, _MISSING)}[stage.part_complete]))),
    ]
    if stage.detail:
        rows.append((("Stage detail", "פירוט שלב"), f'<span class="small">{_ltr(stage.detail)}</span>'))
    rows += [
        (TEXT["validity_state"], lng.x(lng.s(VALIDITY_LABELS[v.state])) + (f"{lng.sep}{_ltr(', '.join(v.reasons))}" if v.reasons else "")),
        (("Registration", "יישור"),
         _ltr(f"{reg.method} · datum fitness {opt(reg.datum_fitness, '.3f')} · RMSE {opt(reg.rmse_mm, '.3f')} mm · "
              f"unit factor {reg.unit_factor:g}")),
        (("Pose", "מנח"),
         _ltr(f"candidates {reg.pose_candidates_considered if reg.pose_candidates_considered is not None else _MISSING} · "
              f"margin {opt(reg.pose_margin, '.3f')} · ambiguous {yn(reg.pose_ambiguous)} · in-plane degenerate "
              f"{yn(reg.in_plane_degenerate)} · abstained {yn(reg.abstained)}")),
        (("Coverage (surface / datum / points)", "כיסוי (משטח / דאטום / נקודות)"),
         _ltr(f"{opt(cov.surface_coverage_ratio, '.2f')} / {opt(cov.datum_coverage_ratio, '.2f')} / "
              f"{cov.scan_point_count if cov.scan_point_count is not None else _MISSING}")),
    ]
    if reg.raw_to_frame is not None:
        matrix = "<br/>".join(" ".join(f"{x: .6f}" for x in r) for r in reg.raw_to_frame)
        rows.append((("Raw scan to frame (4x4)", "מטריצת מעבר"), f'<span class="tiny">{matrix}</span>'))
    rows += [
        (("Metrology qualified", "מוסמך מטרולוגית"), lng.x(lng.s(("no", "לא")))),
        (TEXT["recipe"], _ltr(f"{result.recipe.recipe_id} (v{result.recipe.version}) · sha {_sha8(result.recipe.recipe_sha256)}")),
        (TEXT["compiled"], _ltr(f"sha {_sha8(eng.compiled_part_sha256)}")),
        (TEXT["scan_file"], _ltr(f"sha {_sha8(result.scan.sha256)} · {result.scan.format} · {result.scan.vertex_count} vertices, "
                                 f"{result.scan.face_count} faces")),
        (TEXT["received"], _ltr(result.context.received_at or _MISSING)),
        (TEXT["engine"], _ltr(f"{eng.engine_name} {eng.engine_version} · config {_sha8(eng.config_sha256)}"
                              + (f" · {eng.duration_s:.1f} s" if eng.duration_s is not None else ""))),
        (TEXT["bundle"], _ltr((eng.bundle_id or bundle_id or _MISSING)
                              + (f" (runner: {bundle_id})" if bundle_id and eng.bundle_id and bundle_id != eng.bundle_id else ""))),
    ]
    if recipe is None:
        rows.append((TEXT["revision"], _ltr(_MISSING)))
    kv = _kv(lng, rows, key_width=_CONTENT_W * 0.28, value_width=_CONTENT_W * 0.72)
    items.append(_Block(f"<h3>{lng.block(lng.x(lng.t('id_hashes')))}</h3>" + f'<div class="app">{kv}</div>'))

    # warnings
    warnings = list(result.warnings) + [w for w in extra_warnings if w not in result.warnings]
    if warnings:
        wt = f"<h3>{lng.block(lng.x(lng.t('warnings')))}</h3>"
        items.append(_Table(title=wt, head="", rows=tuple(f'<tr><td class="small">{_ltr(w)}</td></tr>' for w in warnings),
                            cont_title=wt))

    # reasons legend (every UNKNOWN / NOT_APPLICABLE reason present)
    reasons: set[str] = set()
    for char in iter_characteristics(result.bends, result.general_measurements):
        if char.reason:
            reasons.add(char.reason)
    for row in result.bends:
        reasons.update(row.state_reasons)
    lt = f"<h3>{lng.block(lng.x(lng.t('legend')))}</h3>"
    if reasons:
        lrows = tuple(
            lng.row([lng.td(f'<span class="b">{_ltr(code)}</span>', width=_w(130) if i == 0 else None),
                     f'<td{f" style={chr(34)}width:{_w(190):.1f}pt{chr(34)}" if i == 0 else ""}>{_e(reason_label(code)[0])}</td>',
                     f'<td{f" style={chr(34)}width:{_w(191):.1f}pt{chr(34)}" if i == 0 else ""}><div dir="rtl">{_RLM}{_e(reason_label(code)[1])}{_RLM}</div></td>'])
            for i, code in enumerate(sorted(reasons))
        )
        items.append(_Table(title=lt, head="", rows=lrows, cont_title=lt, cls="legend"))
    else:
        items.append(_Block(lt + lng.block(lng.x(lng.t("none")), "small")))
    return items


_EV_COLS: tuple[tuple[str, float, bool], ...] = (
    ("no", 32.0, False), ("characteristic", 88.0, False), ("boundary", 100.0, False),
    ("stations_col", 96.0, False), ("method", 195.0, False),
)


def _evidence_appendix(lng: _L, result: "ShopfloorInspectionResultV1") -> list[_Flow]:
    uid_labels = {r.bend_uid: bend_display_label(result, r) for r in result.bends}
    body = []
    first = True
    widths = [_w(w) for _, w, _ in _EV_COLS]
    for row in result.bends:
        if row.state == "NOT_YET_FORMED":
            continue
        for i, c in enumerate(_ordered(row.characteristics)):
            if c.state != "MEASURED" and c.boundary_kind is None and c.evidence is None:
                continue
            boundary = _MISSING
            if c.boundary_kind:
                adj = uid_labels.get(c.adjacent_bend_uid or "", (c.adjacent_bend_uid or "")[:7]) if c.boundary_kind == "ADJACENT_BEND" else ""
                boundary = lng.x(lng.s(BOUNDARY_LABELS[c.boundary_kind])) + (" " + _ltr(adj) if adj else "")
            st = c.stations
            stations = _ltr(f"{st.count} · {_num(st.min, c.unit)} / {_num(st.max, c.unit)}") if st is not None else _MISSING
            ev = c.evidence
            method = _ltr(f"{ev.method}" + (f" · {ev.support_points}" if ev.support_points is not None else "")) if ev is not None else _MISSING
            ws = widths if first else [None] * len(widths)
            first = False
            body.append(lng.row([
                lng.td(_ltr(char_number(row, i)), width=ws[0]),
                lng.td(_kind_text(lng, c), width=ws[1]), lng.td(f'<span class="small">{boundary}</span>', width=ws[2]),
                lng.td(f'<span class="small">{stations}</span>', width=ws[3]),
                lng.td(f'<span class="tiny">{method}</span>', width=ws[4]),
            ]))
    if not body:
        return []
    title = _section(lng, "C", "app_evidence")
    return [_Table(title=title, head=_head(lng, _EV_COLS), rows=tuple(body), cont_title=title)]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_report_flow(
    result: "ShopfloorInspectionResultV1",
    *,
    recipe: "PartRecipeV1 | None" = None,
    views: Mapping[str, bytes] | None = None,
    views_note: str | None = None,
    engine_bundle_id: str | None = None,
    extra_warnings: Sequence[str] = (),
    lang: str = "he",
    overlay: Any = None,
) -> list[_Flow]:
    """The report as flow items (images referenced as ``view_<name>.png``).

    ``overlay`` is the job's validated ``ZeroshotOverlayV1`` (or None): its lines are listed in section 3
    "Detections (unverified)", apart from the characteristic results, and a lowered verdict is noted under
    the summary. E3's verdict, counts and tables are unchanged."""
    lng = _L(lang)
    items: list[_Flow] = [
        _Block(_title(lng) + _title_block(lng, result, recipe)),
        _Block(f"<h3>{lng.block(lng.x(lng.t('summary_title')))}</h3>" + _summary(lng, result)
               + _zs_lowered_block(lng, overlay)),
        _Block(_overview(lng, views, views_note)),
        _Block(_signoff(lng)),
    ]
    items += _chars_section(lng, result)
    items += _measurements_section(lng, result)
    items += _zs_section(lng, result, overlay)
    items += _views_appendix(lng, views, result, views_note)
    items += _method_appendix(lng, result, recipe, engine_bundle_id, extra_warnings)
    items += _evidence_appendix(lng, result)
    return items


def build_report_html(result: "ShopfloorInspectionResultV1", **kwargs: Any) -> str:
    """The whole report as one HTML body (unpaginated; for inspection and tests)."""
    parts: list[str] = []
    for item in build_report_flow(result, **kwargs):
        if isinstance(item, _Block):
            parts.append(item.html)
        elif isinstance(item, _Table):
            parts.append(item.html(item.rows, first=True))
    return f"<body>{''.join(parts)}</body>"


def _layout_pages(items: Sequence[_Flow], archive: Any) -> bytes:
    import fitz

    from domains.production.reporting.typefaces import font_face_css

    css = font_face_css() + _CSS
    content = fitz.Rect(*CONTENT_RECT)
    limit = content.height - 1.0
    probe = fitz.Rect(0, 0, content.width, 100_000)

    def fits(fragment: str) -> bool:
        story = fitz.Story(html=f"<body>{fragment}</body>", user_css=css, archive=archive)
        more, filled = story.place(probe)
        return not more and (fitz.Rect(filled).y1 - probe.y0) <= limit

    bodies = _paginate(items, fits)
    buf = io.BytesIO()
    writer = fitz.DocumentWriter(buf)
    page_rect = fitz.Rect(*PAGE_RECT)
    pages = 0
    for body in bodies:
        story = fitz.Story(html=f"<body>{body}</body>", user_css=css, archive=archive)
        more = 1
        while more:
            pages += 1
            if pages > _MAX_PAGES:
                raise RuntimeError("report layout did not converge (page limit reached)")
            dev = writer.begin_page(page_rect)
            more, _ = story.place(content)
            story.draw(dev)
            writer.end_page()
    writer.close()
    return buf.getvalue()


def _running_header_footer(doc: Any, result: "ShopfloorInspectionResultV1", recipe: "PartRecipeV1 | None", lng: _L) -> None:
    import fitz

    from domains.production.inspection import ADAPTER_VERSION
    from domains.production.reporting.brand_logo import SIZE, report_png
    from domains.production.reporting.typefaces import FONT_DIR, font_face_css

    css = font_face_css() + f"body {{font-family: Heebo; font-size: 6.6pt; color: {_GREY}; line-height: 1.35;}} .b {{font-weight: bold; color: {_INK};}}"
    fonts = fitz.Archive(str(FONT_DIR))
    total = doc.page_count
    x0, x1 = CONTENT_RECT[0], CONTENT_RECT[2]
    logo_h = 24.0
    logo_w = logo_h * SIZE[0] / SIZE[1]
    logo = report_png()
    logo_xref = 0
    revision = recipe.part_revision if recipe is not None and recipe.part_revision else _MISSING
    doc_no = document_number(result)
    auth_en, auth_he = TEXT["authority_short"]
    for page in doc:
        n = page.number + 1
        # logo at the start side (left in English, right in Hebrew)
        rect = fitz.Rect(x1 - logo_w, 30, x1, 30 + logo_h) if lng.rtl else fitz.Rect(x0, 30, x0 + logo_w, 30 + logo_h)
        logo_xref = page.insert_image(rect, stream=logo if not logo_xref else None, xref=logo_xref, keep_proportion=True)
        title_rect = fitz.Rect(x0, 28, x1 - logo_w - 12, 64) if lng.rtl else fitz.Rect(x0 + logo_w + 12, 28, x1, 64)
        d = lng.sep
        ids = (f"{lng.x(lng.t('part_no'))} {_ltr(result.sku)}{d}{lng.x(lng.t('revision'))} {_ltr(revision)}"
               f"{d}{lng.x(lng.t('job'))} {_ltr(result.context.job_id or _MISSING)}")
        if lng.rtl:
            header = (f'<div dir="rtl" style="text-align:left"><span class="b">{_RLM}{_e(lng.t("doc_title"))}{_RLM}</span> '
                      f'· {_LRM}{_e(TEXT["doc_title"][0])}{_LRM}</div>'
                      f'<div dir="rtl" style="text-align:left">{_RLM}{ids}{_RLM}</div>')
        else:
            header = (f'<div style="text-align:right"><span class="b">{_e(TEXT["doc_title"][0])}</span> · {_he(TEXT["doc_title"][1])}</div>'
                      f'<div style="text-align:right">{ids}</div>')
        page.insert_htmlbox(title_rect, header, css=css, archive=fonts)
        page.draw_line(fitz.Point(x0, 64), fitz.Point(x1, 64), color=(1 / 255, 65 / 255, 111 / 255), width=0.8)
        # footer
        page.draw_line(fitz.Point(x0, 794), fitz.Point(x1, 794), color=(0.82, 0.84, 0.86), width=0.5)
        page_txt = f"{lng.x(lng.t('page'))} {_ltr(n)} {lng.x(lng.t('of'))} {_ltr(total)}"
        left = (f'<div>{_e(TEXT["doc_no"][0])} <span class="b">{_ltr(doc_no)}</span> · {_e(TEXT["generated_by"][0])} '
                f'({_e(ADAPTER_VERSION)}) · {_e(TEXT["company"][0])} · {_e(TEXT["confidential"][0])}</div>'
                f'<div><span class="b">{_e(auth_en)}</span> · {_he(auth_he)}</div>')
        if lng.rtl:
            left = (f'<div dir="rtl">{_RLM}{_e(TEXT["doc_no"][1])} <span class="b">{_ltr(doc_no)}</span> · '
                    f'{_e(TEXT["generated_by"][1])} {_ltr(f"({ADAPTER_VERSION})")} · {_e(TEXT["company"][1])} · '
                    f'{_e(TEXT["confidential"][1])}{_RLM}</div>'
                    f'<div dir="rtl">{_RLM}<span class="b">{_e(auth_he)}</span> · {_LRM}{_e(auth_en)}{_LRM}{_RLM}</div>')
        page.insert_htmlbox(fitz.Rect(x0 + (70 if lng.rtl else 0), 798, x1 - (0 if lng.rtl else 70), 826), left, css=css, archive=fonts)
        pg = (f'<div dir="rtl" style="text-align:left"><span class="b">{_RLM}{page_txt}{_RLM}</span></div>' if lng.rtl
              else f'<div style="text-align:right"><span class="b">{page_txt}</span></div>')
        prect = fitz.Rect(x0, 798, x0 + 66, 812) if lng.rtl else fitz.Rect(x1 - 66, 798, x1, 812)
        page.insert_htmlbox(prect, pg, css=css, archive=fonts)


def render_pdf(
    result: "ShopfloorInspectionResultV1",
    *,
    recipe: "PartRecipeV1 | None" = None,
    assets_dir: str | Path | None = None,
    views: Mapping[str, bytes] | None = None,
    engine_bundle_id: str | None = None,
    extra_warnings: Sequence[str] = (),
    lang: str = "he",
    overlay: Any = None,
) -> bytes:
    """Render the report to PDF bytes (module docstring).

    ``views`` are PNG snapshots keyed by view name (``iso``, ``front``, ``top``,
    ``right`` and, for page 1, ``overview``); when omitted and ``assets_dir`` is
    given they are rendered from the result's viewer assets there. Views given
    without ``overview`` get one rendered from ``assets_dir`` when available.
    A snapshot failure never fails the report: the report then says why the
    pictures are unavailable.

    A ``recipe`` must be the exact recipe the result names (it supplies the
    drawing revision, material and tolerance standard in the header);
    otherwise ``PostprocessError("RECIPE_INVALID")`` is raised.
    """
    import fitz

    from domains.production.inspection import ADAPTER_VERSION
    from domains.production.inspection.engine_output import check_recipe_ref
    from domains.production.reporting.typefaces import FONT_DIR, verify_fonts

    lng = _L(lang)
    if recipe is not None:
        check_recipe_ref(result, recipe)
    verify_fonts()
    note: str | None = None
    views = dict(views) if views is not None else None
    if (views is None or "overview" not in views) and (assets_dir is not None or views is None):
        try:
            from domains.production.reporting.snapshots import load_scene, render_view, render_views

            scene = load_scene(result, assets_dir if assets_dir is not None else Path("/nonexistent-assets"))
            if views is None and assets_dir is not None:
                views = render_views(scene)
            views = dict(views or {})
            views["overview"] = render_view(scene, "iso", width_px=OVERVIEW_PX[0], height_px=OVERVIEW_PX[1], style="overview")
            if scene.warnings and assets_dir is not None:
                note = "; ".join(scene.warnings)[:300]
        except Exception as exc:  # display only: never fail the report on a snapshot
            note = f"SNAPSHOT_FAILED: {type(exc).__name__}: {exc}"[:300]
    items = build_report_flow(
        result, recipe=recipe, views=views, views_note=note, engine_bundle_id=engine_bundle_id,
        extra_warnings=extra_warnings, lang=lang, overlay=overlay,
    )
    archive = fitz.Archive(str(FONT_DIR))
    for name, png in sorted((views or {}).items()):
        archive.add(png, f"view_{name}.png")
    doc = fitz.open("pdf", _layout_pages(items, archive))
    _running_header_footer(doc, result, recipe, lng)
    date = pdf_date(result.created_at)
    doc.set_metadata({
        "title": f"Dimensional inspection report {result.sku} ({result.recipe.recipe_id}) {document_number(result)}",
        "author": TEXT["company"][0],
        "subject": f"{result.summary.state}; operator assist, not an automatic accept/reject decision",
        "keywords": f"{result.context.job_id or ''} {result.scan.sha256}".strip(),
        "creator": f"Sherman QC {ADAPTER_VERSION}",
        "producer": "Sherman QC",
        "creationDate": date,
        "modDate": date,
    })
    data = doc.tobytes(garbage=4, deflate=True, no_new_id=True)
    doc.close()
    return data


def write_pdf(path: str | Path, result: "ShopfloorInspectionResultV1", **kwargs: Any) -> int:
    """Render and write atomically; returns the byte count."""
    from domains.production.inspection.ply_io import atomic_write_bytes

    data = render_pdf(result, **kwargs)
    atomic_write_bytes(path, data)
    return len(data)
