"""C06 formal inspection report (PDF) and offscreen snapshots.

Checks (synthetic SYN-HAT-1 results only), in both report languages (Hebrew
is the default):

- page 1: title block, summary (verdict, stage "<formed>/<total> bends
  formed", counts), overview picture, sign-off box; running header with the
  Sherman logo and footer with "page X of Y" on every page;
- the characteristic table reads back (PDF table parser of the real-part E2E
  equality check) to exactly the stored values and decisions, bends in shop
  order with studio labels, flange lengths before the angle;
- honesty: an Undetermined (NOT_MEASURED) value never prints Pass/Fail, a
  NOT_YET_FORMED bend is listed apart and never gets a result badge, an
  invalid inspection prints no Pass at all;
- Hebrew: extracted in logical order and laid out right-to-left (not
  mirrored); numbers stay left-to-right;
- determinism (byte-identical renders) with a mutate-and-check counterpart;
- the bundled Heebo font is embedded and sha256-verified;
- snapshots are deterministic and actually show the heatmap.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
from pathlib import Path

import fitz
import numpy as np
import pytest

from contracts.production.characteristics import NOT_APPLICABLE_REASONS, UNKNOWN_REASONS
from contracts.production.states import STAGE_STATUSES, VALIDITY_STATES
from domains.production.inspection.engine_output import load_engine_output
from domains.production.inspection.fake_engine_output import build_fake_engine_output
from domains.production.reporting import labels
from domains.production.reporting.shopfloor_pdf import (
    CONTENT_RECT,
    PAGE_RECT,
    build_report_html,
    char_number,
    document_number,
    pdf_date,
    render_pdf,
)

_RLM = "‏"
_LRM = "‎"
_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def outputs(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = {}
    for scenario in ("final", "in_process", "invalid"):
        fake = build_fake_engine_output(tmp_path_factory.mktemp(scenario), scenario=scenario)
        eo = load_engine_output(fake.result_path, fake.recipe_path, fake.compiled_path)
        for lang in ("he", "en"):
            pdf = render_pdf(eo.result, recipe=eo.recipe, assets_dir=eo.engine_dir, engine_bundle_id="bundle-example",
                             lang=lang)
            out[(scenario, lang)] = (eo, pdf)
    return out


def _doc(pdf: bytes) -> fitz.Document:
    return fitz.open("pdf", pdf)


def _norm(text: str) -> str:
    return " ".join(text.replace(_RLM, " ").replace(_LRM, " ").split())


def _text(pdf: bytes) -> str:
    return _norm(" ".join(page.get_text() for page in _doc(pdf)))


def _parse_tables(pdf: bytes, tmp_path: Path) -> dict:
    """The PDF table parser used by the real-part E2E five-way equality check."""
    spec = importlib.util.spec_from_file_location("real_e2e_support", _REPO / "scripts/production/real_e2e_support.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    path = tmp_path / "r.pdf"
    path.write_bytes(pdf)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.pdf_bends(argparse.Namespace(pdf=str(path)))
    return json.loads(buf.getvalue())


# ---------------------------------------------------------------- page 1, header, footer
def test_title_block_summary_and_signoff(outputs) -> None:
    eo, pdf = outputs[("final", "en")]
    doc = _doc(pdf)
    first = _norm(doc[0].get_text())
    for needle in ("Dimensional inspection report", "Part number", "SYN-HAT-1", "Synthetic hat section",
                   "SYN-HAT-1@v1 (v1)", "SYN-HAT-1_A_SN0042.ply", "SN0042", "cell-1", "2026-10-17T07:15:42Z",
                   "S235 2.0 mm · 2.00 mm", "SHOP_STANDARD", "Operator", document_number(eo.result),
                   "Inspection summary", "Out of tolerance found", "bends formed", "Pass", "Review", "Fail",
                   "Undetermined", "Inspector review and disposition", "Accept", "Rework", "Reject", "Signature",
                   "Part overview"):
        assert needle in first, needle
    s = eo.result.summary
    assert f"{s.bends_formed}/{s.bends_total} bends formed" in first
    assert f"{s.characteristics_required} required characteristics" in first
    assert "Revision" in first and " A " in first  # recipe part revision
    assert doc.metadata["creationDate"] == "D:20261017071542Z"
    assert doc.metadata["author"] == "I. Sherman Industries"


@pytest.mark.parametrize("lang", ["he", "en"])
def test_every_page_has_logo_header_and_footer(outputs, lang: str) -> None:
    from domains.production.reporting.brand_logo import REPORT_SHA256

    eo, pdf = outputs[("final", lang)]
    doc = _doc(pdf)
    page_word, of_word = ("עמוד", "מתוך") if lang == "he" else ("Page", "of")
    for page in doc:
        text = _norm(page.get_text())
        assert "Operator assist — not an automatic accept/reject decision" in text
        assert "סיוע למפעיל" in text
        assert f"{page_word} {page.number + 1} {of_word} {doc.page_count}" in text, text[-300:]
        assert document_number(eo.result) in text and "SYN-HAT-1" in text
        assert ("I. Sherman Industries" in text) or ("י. שרמן תעשיות" in text)
        images = page.get_images(full=True)
        assert any(hashlib.sha256(doc.extract_image(img[0])["image"]).hexdigest() == REPORT_SHA256
                   or doc.extract_image(img[0])["width"] == 884 for img in images), "logo missing"
        logo_rects = [r for img in images for r in page.get_image_rects(img[0]) if r.y1 < CONTENT_RECT[1]]
        assert logo_rects, "logo not in the running header"
        # logo at the start side: left in English, right in Hebrew
        mid = (PAGE_RECT[2]) / 2
        assert (logo_rects[0].x0 > mid) == (lang == "he")


def test_logo_is_the_official_file() -> None:
    from PIL import Image

    from domains.production.reporting import brand_logo

    src = brand_logo.source_png()
    assert hashlib.sha256(src).hexdigest() == brand_logo.SOURCE_SHA256
    assert hashlib.sha256(brand_logo.report_png()).hexdigest() == brand_logo.REPORT_SHA256
    official = Path("/Users/idantal/ShermanQC/brand/www.shermanep.com_Images_logo.png")
    if official.is_file():
        assert official.read_bytes() == src
    small = Image.open(io.BytesIO(src)).convert("RGBA")
    big = Image.open(io.BytesIO(brand_logo.report_png())).convert("RGBA")
    assert big.size == (small.width * 4, small.height * 4)
    # the resampled logo is the same picture (no recolouring or redrawing)
    def premultiplied(im: Image.Image) -> np.ndarray:
        a = np.asarray(im, dtype=float)
        return a[..., :3] * a[..., 3:] / 255.0

    back = big.resize(small.size, Image.Resampling.BOX)
    assert np.abs(premultiplied(back) - premultiplied(small)).mean() < 6.0


# ---------------------------------------------------------------- characteristic table
@pytest.mark.parametrize("lang", ["he", "en"])
def test_tables_read_back_to_the_stored_values(outputs, lang: str, tmp_path: Path) -> None:
    eo, pdf = outputs[("final", lang)]
    parsed = _parse_tables(pdf, tmp_path)
    assert parsed["ok"]
    words = {"PASS": "Pass", "FAIL": "Fail", "REVIEW": "Review", "UNKNOWN": "Undetermined"}
    seen_order = list(parsed["bends"])
    assert seen_order == [labels.bend_display_label(eo.result, r) for r in eo.result.bends]  # shop order
    for row in eo.result.bends:
        got = parsed["bends"][labels.bend_display_label(eo.result, row)]["characteristics"]
        for char in row.characteristics:
            key = "BACKGAUGE_LENGTH" if char.kind == "BACKGAUGE_LENGTH" else (
                f"{char.kind}/{char.side}" if char.kind == "FLANGE_LENGTH" else char.kind)
            pc = got[key]
            assert pc["decision"] == words[char.decision], (row.label, key)
            assert pc["expected"] == pytest.approx(char.expected, abs=0.005 + 1e-9)
            if char.state == "MEASURED":
                assert pc["actual"] == pytest.approx(char.actual, abs=0.005 + 1e-9)
            else:
                assert pc["actual"] is None  # an undetermined value prints no number
        # flange lengths first, then the angle (characteristic numbers)
        nums = {k: v["no"] for k, v in got.items()}
        assert nums["FLANGE_LENGTH/A"] < nums["FLANGE_LENGTH/B"] < nums["INCLUDED_ANGLE"]
        assert nums["FLANGE_LENGTH/A"] == char_number(row, 0)
    for gm in eo.result.general_measurements:
        cells = parsed["general_rows"][gm.label]["cells"]
        if gm.characteristic.state == "MEASURED":
            assert any(c and f"{gm.characteristic.actual:.2f}" in c for c in cells)


def test_move_groups_and_studio_labels(outputs) -> None:
    eo, pdf = outputs[("final", "en")]
    text = _text(pdf)
    assert "Move 1 · B3, B4" in text and "מהלך 1" in text
    studio = {row.bend_uid: f"B{10 + row.sequence_index - 1}" for row in eo.result.bends}
    ctx = eo.result.context.model_copy(update={"bend_display_labels": studio})
    result = eo.result.model_copy(update={"context": ctx})
    text = _text(render_pdf(result, recipe=eo.recipe, assets_dir=eo.engine_dir, lang="en"))
    assert "Move 1 · B12, B13" in text  # move headings use the studio labels
    he = _text(render_pdf(result, recipe=eo.recipe, assets_dir=eo.engine_dir))
    assert "מהלך 1" in he and "B12, B13" in he
    from domains.production.reporting.snapshots import load_scene
    scene = load_scene(result, eo.engine_dir)
    assert [b.label for b in scene.bends] == [studio[r.bend_uid] for r in eo.result.bends]


def _rows(html_doc: str) -> list[str]:
    return re.findall(r"<tr[ >].*?</tr>", html_doc)


@pytest.mark.parametrize("scenario", ["final", "in_process", "invalid"])
def test_undetermined_never_prints_pass_or_fail(outputs, scenario: str, tmp_path: Path) -> None:
    eo, pdf = outputs[(scenario, "en")]
    html_doc = build_report_html(eo.result, recipe=eo.recipe, lang="en")
    for row in _rows(html_doc):
        if "d-UNKNOWN" in row:
            assert "d-PASS" not in row and "d-FAIL" not in row and ">Pass<" not in row and ">Fail<" not in row
    parsed = _parse_tables(pdf, tmp_path)
    for brow in eo.result.bends:
        got = parsed["bends"].get(labels.bend_display_label(eo.result, brow), {}).get("characteristics", {})
        for char in brow.characteristics:
            if char.state == "NOT_MEASURED":
                key = f"{char.kind}/{char.side}" if char.kind == "FLANGE_LENGTH" else char.kind
                assert got[key.split("/")[0] if char.kind == "BACKGAUGE_LENGTH" else key]["decision"] == "Undetermined"


def test_reason_notes_and_legend(outputs) -> None:
    _, pdf = outputs[("final", "en")]
    text = _text(pdf)
    assert "(1) EDGE_NOT_SEEN — The far edge of the flange was not captured." in text
    he = _text(outputs[("final", "he")][1])
    assert "EDGE_NOT_SEEN" in he and "הקצה הרחוק של הכנף לא נסרק." in he


def test_in_process_lists_unformed_bends_apart(outputs, tmp_path: Path) -> None:
    eo, pdf = outputs[("in_process", "en")]
    text = _text(pdf)
    s = eo.result.summary
    for needle in ("Stage in tolerance (part not complete)", "השלב בטולרנס", f"{s.bends_formed}/{s.bends_total} bends formed",
                   "Bends not yet formed at this stage", "A bend that is not yet formed is never a defect",
                   "Backgauge length S2", "Not yet formed"):
        assert needle in text, needle
    nyf = [r for r in eo.result.bends if r.state == "NOT_YET_FORMED"]
    assert nyf
    html_doc = build_report_html(eo.result, recipe=eo.recipe, lang="en")
    nyf_rows = [r for r in _rows(html_doc) if r.startswith('<tr class="nyf"')]
    assert len(nyf_rows) == len(nyf)  # one light row per unformed bend, no characteristic rows
    for row in nyf_rows:
        assert "badge" not in row
    parsed = _parse_tables(pdf, tmp_path)
    for row in nyf:  # not in the characteristic table at all
        assert labels.bend_display_label(eo.result, row) not in parsed["bends"]
    assert "Measurements" in text and "NOT_APPLICABLE_AT_STAGE" in text
    he = _text(outputs[("in_process", "he")][1])
    assert "כיפופים שטרם בוצעו בשלב זה" in he and "טרם כופף" in he


def test_invalid_inspection_prints_no_pass(outputs) -> None:
    eo, pdf = outputs[("invalid", "en")]
    text = _text(pdf)
    assert "Invalid inspection" in text and "בדיקה לא תקפה" in text
    assert "UNTRUSTED_REGISTRATION" in text or "Untrusted registration" in text
    for lang in ("en", "he"):
        html_doc = build_report_html(eo.result, recipe=eo.recipe, lang=lang)
        assert "d-PASS" not in html_doc and "d-FAIL" not in html_doc and "d-REVIEW" not in html_doc
    assert "INVALID_INSPECTION" in text


def test_unplanned_crease_is_reported_and_drawn(outputs) -> None:
    from contracts.production.inspection_result import ShopfloorInspectionResultV1, UnplannedCreaseV1, derive_summary
    from domains.production.reporting.snapshots import load_scene, render_view

    eo, _ = outputs[("final", "en")]
    crease = UnplannedCreaseV1(crease_id="X1", reason="EXTRA_BEND_SUSPECTED",
                               polyline_mm=((10.0, 0.0, 30.0), (10.0, 100.0, 30.0)), included_deg=171.5,
                               note="crease across the top panel")
    data = eo.result.model_dump(mode="json")
    data["unplanned_creases"] = [crease.model_dump(mode="json")]
    r = eo.result
    data["summary"] = derive_summary(validity_state=r.inspection_validity.state, bends=r.bends,
                                     general_measurements=r.general_measurements,
                                     stage_decision=r.stage_decision.decision, stage_status=r.stage_decision.status,
                                     unplanned_creases=1).model_dump(mode="json")
    result = ShopfloorInspectionResultV1.model_validate(data)
    text = _text(render_pdf(result, recipe=eo.recipe, lang="en"))
    assert "Unplanned creases" in text and "EXTRA_BEND_SUSPECTED" in text and "171.50°" in text
    scene = load_scene(result, eo.engine_dir)
    assert [c[0] for c in scene.creases] == ["X1"]
    assert render_view(scene, "top") != render_view(load_scene(eo.result, eo.engine_dir), "top")


# ---------------------------------------------------------------- Hebrew order (not mirrored), numbers LTR
def _rtl_ok(pdf: bytes, needle: str) -> int:
    """Count occurrences of ``needle`` whose letters are laid out right-to-left."""
    letters = [c for c in needle if "֐" <= c <= "׿"]
    found = 0
    for page in _doc(pdf):
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                chars = [c for span in line["spans"] for c in span["chars"]]
                heb = [(c["c"], c["origin"][0]) for c in chars if "֐" <= c["c"] <= "׿"]
                text = "".join(c for c, _ in heb)
                start = text.find("".join(letters))
                while start >= 0:
                    xs = [x for _, x in heb[start:start + len(letters)]]
                    assert all(a > b for a, b in zip(xs, xs[1:])), f"{needle!r} is mirrored: {xs}"
                    found += 1
                    start = text.find("".join(letters), start + 1)
    return found


@pytest.mark.parametrize(
    "scenario, needle",
    [("final", "דוח בדיקת מידות"), ("final", "סיוע למפעיל"), ("final", "לא הוכרע"), ("final", "מהלך 1"),
     ("final", "אורך כנף"), ("final", "תוצאות מאפיינים"), ("in_process", "טרם כופף"), ("invalid", "בדיקה לא תקפה")],
)
def test_hebrew_is_logical_and_right_to_left(outputs, scenario: str, needle: str) -> None:
    _, pdf = outputs[(scenario, "he")]
    assert needle in _text(pdf)  # extracted in logical order
    assert _rtl_ok(pdf, needle) >= 1


def test_hebrew_report_is_laid_out_right_to_left(outputs) -> None:
    # the first column (characteristic number) is on the right, the result column left of the values
    _, pdf = outputs[("final", "he")]
    page = next(p for p in _doc(pdf) if "תוצאות מאפיינים" in p.get_text())
    no = page.search_for("1.1")[0]
    first_value = [r for r in page.search_for("±") if r.y0 > no.y0 - 3 and r.y1 < no.y1 + 3]
    assert first_value and no.x0 > max(r.x1 for r in first_value)
    assert no.x1 > (CONTENT_RECT[2] - 40)


def test_hebrew_numbers_stay_left_to_right(outputs) -> None:
    # signed values in a right-to-left document: the minus sign is drawn left of the digits
    eo, pdf = outputs[("final", "he")]
    checked = 0
    for page in _doc(pdf):
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                chars = [c for span in line["spans"] for c in span["chars"]]
                for i, c in enumerate(chars):
                    if c["c"] == "−" and i + 1 < len(chars) and chars[i + 1]["c"].isdigit():
                        assert c["origin"][0] < chars[i + 1]["origin"][0]
                        checked += 1
    assert checked >= 3
    text = _text(pdf)
    assert "SYN-HAT-1@v1 (v1)" in text and "2026-10-17T07:15:42Z" in text  # identifiers intact


def test_hebrew_parentheses_are_mirrored_correctly(outputs) -> None:
    # The inspection stage in the title block ("בתהליך (באמצע רצף הכיפופים)" in the
    # synthetic result): both parentheses inside the RTL run, the logical ')' at its left end.
    _, pdf = outputs[("final", "he")]
    page = _doc(pdf)[0]
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            chars = [c for span in line["spans"] for c in span["chars"]]
            text = "".join(c["c"] for c in chars)
            if "בתהליך" in text:
                heb_x = [c["origin"][0] for c in chars if "֐" <= c["c"] <= "׿"]
                parens = [c["origin"][0] for c in chars if c["c"] in "()"]
                assert len(parens) == 2
                assert min(parens) < min(heb_x) and max(parens) < max(heb_x)
                return
    pytest.fail("inspection-stage field not found")


def _header_visual(page: fitz.Page) -> str:
    chars = [c for b in page.get_text("rawdict")["blocks"] for line in b.get("lines", [])
             for span in line["spans"] for c in span["chars"] if c["origin"][1] < CONTENT_RECT[1] - 15]
    text = "".join(c["c"] for c in sorted(chars, key=lambda c: (round(c["origin"][1]), c["origin"][0])))
    return " ".join(text.replace(_RLM, " ").replace(_LRM, " ").split())


def test_numeric_sku_header_keeps_the_ids_in_order(outputs) -> None:
    # Production SKUs are numeric: next to Hebrew labels they must not join the RTL run.
    eo, _ = outputs[("invalid", "en")]
    data = eo.result.model_dump(mode="json")
    data["sku"], data["recipe"]["recipe_id"] = "10839000", "10839000@v1"
    result = type(eo.result).model_validate(data)
    for lang in ("he", "en"):
        doc = _doc(render_pdf(result, assets_dir=eo.engine_dir, lang=lang))
        for page in doc:
            header = _header_visual(page)
            assert "10839000" in header and "00093801" not in header, header
            if lang == "en":
                assert "Part number 10839000 · Revision" in header, header
            else:  # drawn right-to-left: "<job> עבודה · ... · 10839000 ט"קמ"
                assert re.search(r"10839000 ט\"קמ", header), header


# ---------------------------------------------------------------- determinism, fonts
def test_render_is_byte_deterministic_and_input_sensitive(outputs) -> None:
    eo, pdf = outputs[("final", "he")]
    again = render_pdf(eo.result, recipe=eo.recipe, assets_dir=eo.engine_dir, engine_bundle_id="bundle-example")
    assert again == pdf  # Hebrew is the default
    changed = eo.result.model_copy(update={"part_name": "Synthetic hat section (rev)"})
    assert render_pdf(changed, recipe=eo.recipe, assets_dir=eo.engine_dir, engine_bundle_id="bundle-example") != pdf
    with pytest.raises(ValueError):
        render_pdf(eo.result, lang="fr")


def test_heebo_is_embedded(outputs) -> None:
    _, pdf = outputs[("final", "he")]
    fonts = {f[3] for page in _doc(pdf) for f in page.get_fonts()}
    assert any("Heebo" in name for name in fonts), fonts


def test_tampered_font_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from domains.production.reporting import typefaces

    copy = tmp_path / "fonts"
    shutil.copytree(typefaces.FONT_DIR, copy)
    data = bytearray((copy / "Heebo-Regular.ttf").read_bytes())
    data[1000] ^= 0xFF
    (copy / "Heebo-Regular.ttf").write_bytes(bytes(data))
    monkeypatch.setattr(typefaces, "FONT_DIR", copy)
    with pytest.raises(RuntimeError, match="sha256"):
        typefaces.verify_fonts()
    (copy / "Heebo-Regular.ttf").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        typefaces.verify_fonts()


def test_report_font_build_is_reproducible_and_ligature_free(outputs) -> None:
    from fontTools.ttLib import TTFont

    from domains.production.reporting import typefaces

    typefaces.verify_fonts()
    for weight, (upstream, _) in typefaces.UPSTREAM_FILES.items():
        derived = typefaces.derive_report_font(typefaces.FONT_DIR / upstream)
        assert derived == typefaces.font_path(weight).read_bytes(), weight
        gsub = TTFont(typefaces.font_path(weight))["GSUB"].table
        assert all(not r.Feature.LookupListIndex for r in gsub.FeatureList.FeatureRecord if r.FeatureTag == "liga")
        assert any(r.Feature.LookupListIndex for r in gsub.FeatureList.FeatureRecord if r.FeatureTag == "ccmp")
    _, pdf = outputs[("final", "en")]
    text = _text(pdf)
    assert "Flange length A" in text and "far edge of the flange" in text
    assert "ý" not in text and "ﬁ" not in text and "ﬂ" not in text


def test_without_assets_the_views_appendix_says_so(outputs) -> None:
    eo, _ = outputs[("final", "en")]
    text = _text(render_pdf(eo.result, recipe=eo.recipe, lang="en"))
    assert "3D views unavailable" in text
    assert "Part overview" in text  # the overview still draws the bend lines from the result
    he = _text(render_pdf(eo.result, recipe=eo.recipe))
    assert "מבטים תלת-ממדיים אינם זמינים" in he


def test_labels_cover_every_contract_code() -> None:
    assert (UNKNOWN_REASONS | NOT_APPLICABLE_REASONS) <= set(labels.REASON_LABELS)
    assert STAGE_STATUSES <= set(labels.STAGE_STATUS_LABELS)
    assert VALIDITY_STATES <= set(labels.VALIDITY_LABELS)
    for en, he in labels.REASON_LABELS.values():
        assert en and he and any("֐" <= c <= "׿" for c in he)


def test_pdf_date() -> None:
    assert pdf_date("2026-10-17T07:15:42Z") == "D:20261017071542Z"
    assert pdf_date("2026-10-17T07:15:42.123456+00:00") == "D:20261017071542Z"


@pytest.mark.parametrize("scenario", ["final", "in_process", "invalid"])
@pytest.mark.parametrize("lang", ["he", "en"])
def test_nothing_is_laid_out_beyond_the_page_margins(outputs, scenario: str, lang: str) -> None:
    # A table wider than the content box is clipped by MuPDF; every glyph and
    # rule must stay inside the side margins, A4 portrait.
    _, pdf = outputs[(scenario, lang)]
    lo, hi = CONTENT_RECT[0] - 0.5, CONTENT_RECT[2] + 0.5
    for page in _doc(pdf):
        assert (page.rect.width, page.rect.height) == (PAGE_RECT[2], PAGE_RECT[3])
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    for c in span["chars"]:
                        if c["c"].strip(" \u200e\u200f"):  # visible glyphs (bidi marks have no ink)
                            assert lo <= c["bbox"][0] and c["bbox"][2] <= hi, (page.number, c["c"], c["bbox"])
        for drawing in page.get_drawings():
            assert lo <= drawing["rect"].x0 and drawing["rect"].x1 <= hi, (page.number, drawing["rect"])


# ---------------------------------------------------------------- pagination
def test_paginate_repeats_table_headers_and_keeps_every_row() -> None:
    from domains.production.reporting.shopfloor_pdf import _Block, _PageBreak, _paginate, _Table

    rows = tuple(f"<tr>r{i:02d}</tr>" for i in range(12))
    table = _Table(title="<h3>T</h3>", head="<tr>HEAD</tr>", rows=rows, cont_title="<h3>T (cont.)</h3>")
    items = [_Block("<p>intro</p>"), table, _PageBreak(), _Block("<p>" + "x" * 500 + "</p>"), _Block("<p>tail</p>")]
    pages = _paginate(items, lambda html: len(html) <= 120)
    body = "".join(pages)
    assert [r for r in rows if r not in body] == []  # nothing lost
    assert all(body.count(r) == 1 for r in rows)  # nothing duplicated
    table_pages = [p for p in pages if "<tr>r" in p]
    assert len(table_pages) > 1
    assert all("<tr>HEAD</tr>" in p for p in table_pages)  # header repeated on every page
    assert "<h3>T</h3>" in table_pages[0] and all("(cont.)" in p for p in table_pages[1:])
    assert pages[-2] == "<p>" + "x" * 500 + "</p>"  # an oversized block gets a page of its own
    assert pages[-1] == "<p>tail</p>"
    assert all(len(p) <= 120 for p in pages if "x" * 500 not in p)


# ---------------------------------------------------------------- snapshots
def _scene(eo, **kw):
    from domains.production.reporting.snapshots import load_scene

    return load_scene(eo.result, eo.engine_dir, **kw)


def _dark_red_pixels(png: bytes) -> int:
    pix = fitz.Pixmap(png)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].astype(int)
    target = np.array([103, 0, 31])  # RdBu_r at +band
    return int((np.abs(arr - target).sum(axis=2) < 60).sum())


def test_snapshots_are_deterministic_and_show_the_heatmap(outputs) -> None:
    from domains.production.reporting.snapshots import VIEWS, render_view, render_views

    eo, _ = outputs[("final", "en")]
    scene = _scene(eo)
    assert scene.points is not None and scene.deviations is not None and scene.mesh_kind == "stage"
    views = render_views(scene)
    assert set(views) == set(VIEWS)
    for name, png in views.items():
        pix = fitz.Pixmap(png)
        assert (pix.width, pix.height) == (1400, 1000), name
        assert render_view(scene, name) == png  # deterministic
    hot = _scene(eo)
    hot.deviations = np.full_like(hot.deviations, 10.0)  # everything beyond +band
    cold = _scene(eo)
    cold.deviations = None
    assert _dark_red_pixels(render_view(hot, "top")) > 5_000
    assert _dark_red_pixels(render_view(cold, "top")) < 200
    # the page-1 overview is clean: no heatmap colours even when deviations are published
    overview = render_view(hot, "iso", width_px=1800, height_px=620, style="overview")
    assert _dark_red_pixels(overview) < 200
    assert render_view(hot, "iso", width_px=1800, height_px=620, style="overview") == overview
    with pytest.raises(ValueError):
        render_view(scene, "bottom")
    with pytest.raises(ValueError):
        render_view(scene, "iso", style="fancy")


def test_scene_reports_missing_assets(outputs, tmp_path: Path) -> None:
    from domains.production.reporting.snapshots import load_scene, render_views

    eo, _ = outputs[("final", "en")]
    scene = load_scene(eo.result, tmp_path)  # nothing there
    assert scene.points is None and scene.mesh_vertices is None
    assert any(w.startswith("SNAPSHOT_SCAN_UNAVAILABLE") for w in scene.warnings)
    assert set(render_views(scene)) == {"iso", "front", "top", "right"}  # still renders bend lines


def _fine_plane_scene(n: int):
    """A flat square plate (-50..50 mm, z=0) tessellated into 2*n*n triangles, no scan points."""
    from domains.production.reporting.snapshots import Scene

    g = np.linspace(-50.0, 50.0, n + 1)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    verts = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]).astype(np.int64)
    return Scene(mesh_vertices=verts, mesh_faces=faces, mesh_kind="final"), verts, faces


def test_fine_mesh_is_drawn_solid_not_striped(tmp_path: Path) -> None:
    from domains.production.inspection.ply_io import write_mesh_ply
    from domains.production.reporting.snapshots import load_scene, render_view

    scene, verts, faces = _fine_plane_scene(200)  # 80k triangles, above the old 40k thinning cap
    assert len(faces) == 80_000
    pix = fitz.Pixmap(render_view(scene, "top"))
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].astype(int)
    covered = (arr < 245).any(axis=2)  # anything that is not the white background
    rows, cols = np.nonzero(covered)
    r0, r1 = np.percentile(rows, [5, 95]).astype(int)
    c0, c1 = np.percentile(cols, [5, 95]).astype(int)
    inner = covered[r0 + 40 : r1 - 40, c0 + 40 : c1 - 40]
    assert inner.size > 100_000
    assert inner.mean() > 0.999, f"{(~inner).sum()} background pixels inside the plate"

    eo_dir = tmp_path / "assets"
    eo_dir.mkdir()
    write_mesh_ply(eo_dir / "final_mesh.ply", verts, faces)

    class _VA:
        registered_scan = deviations = stage_mesh = None
        final_mesh = "final_mesh.ply"
        heatmap_band_mm = 0.5
        heatmap_unavailable_reason = "NO_SCAN"

    class _SD:
        stage_model_id = None

    class _R:
        viewer_assets = _VA()
        stage_decision = _SD()
        bends: list = []
        unplanned_creases: list = []

    full = load_scene(_R(), eo_dir)  # type: ignore[arg-type]
    assert full.mesh_faces is not None and len(full.mesh_faces) == len(faces) and not full.warnings
    capped = load_scene(_R(), eo_dir, max_faces=50_000)  # type: ignore[arg-type]
    assert capped.mesh_faces is None
    assert any(w.startswith("SNAPSHOT_MESH_TOO_LARGE") for w in capped.warnings)
