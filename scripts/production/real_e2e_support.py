#!/usr/bin/env python3
"""Python helpers for the REAL-part app end-to-end run (``scripts/production/run_e2e_real.mjs``).

Each subcommand prints one JSON object on stdout; exit 0 = the check passed, 1 = it
failed, 2 = usage error. Nothing here writes into the line's data root; everything
is read-only on the job workspace and writes only into ``--out``.

``pdf-bends --pdf F``
    Parses the characteristic table of the station PDF report (PyMuPDF ``find_tables``;
    English or Hebrew report, columns found by their header): for every bend (studio
    label) and characteristic its number, expected / actual / decision, plus the
    general-measurement rows.

``compare --direct R --job-engine R --job-rendered R --api R --dom D --pdf P --out DIR``
    Equality of the per-bend values across the direct E3 CLI run, the job's engine
    output, the rendered shopfloor result, the station API, the station DOM and
    the PDF. Engine numbers are compared exactly between the JSON sources; the DOM
    (1 decimal) and the PDF (2 decimals) within half a display unit. Writes
    ``compare.json`` and ``compare.md`` into DIR.

``overlay --job-dir J --api A --dom D --pdf P``
    The job-level zero-shot overlay (``sherman_zeroshot_overlay_v1``) is the same in the job output
    (``J/zeroshot/zeroshot_overlay_v1.json``), the station API (``A``, the JSON of
    ``GET /jobs/{id}/zeroshot-overlay``), the station screen (``D``, ``station_dom.json`` with its ``zs``
    block: line ids, kinds, decisions, review reasons, lowered flag) and the PDF (``P``: the
    "detections (unverified)" section lists exactly the overlay's line ids). The overlay never
    upgrades E3's verdict: its effective state is E3's, or REVIEW_REQUIRED lowered from a pass-like one.

``heatmap --job-dir J [--direct-engine D]``
    Checks the station heatmap data: every viewer point (``result/viewer_scan.ply``)
    is a point of the engine's registered scan and its ``viewer_deviations.f32``
    value equals the engine ``deviations.f32`` value of that point; with
    ``--direct-engine`` also that the job's engine deviations equal the direct run's.
    Reports the colour-band histogram (|dev| <= b, b..2b, > 2b, NaN) with b the
    published ``heatmap_band_mm``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _emit(doc: dict[str, Any], ok: bool = True) -> int:
    doc.setdefault("ok", ok)
    print(json.dumps(doc, ensure_ascii=False, default=str))
    return 0 if ok else 1


def _labels():
    from contracts.production.vocabulary import DISPLAY_TOKENS
    from domains.production.reporting.labels import KIND_LABELS, TEXT

    return KIND_LABELS, TEXT, DISPLAY_TOKENS


def _num(text: str | None) -> float | None:
    if text is None:
        return None
    t = str(text).strip().replace("‎", "").replace("‏", "").replace("−", "-").replace("°", "").replace("±", "")
    m = re.match(r"^[+-]?\d+(?:\.\d+)?", t)
    return float(m.group(0)) if m else None


def _clean(text: str | None) -> str:
    return " ".join(str(text or "").replace("‎", " ").replace("‏", " ").split())


def _first_line(text: str | None) -> str:
    return _clean((str(text or "").strip().split("\n") or [""])[0])


def _key(text: str) -> str:
    """Comparison key: no marks, no spaces, no stage suffix (S<k>)."""
    return re.sub(r"S\d+|\d+S", "", _clean(text)).replace(" ", "")


def _keys(text: str) -> set[str]:
    """Keys of a label as drawn in either direction: PyMuPDF's table extraction
    returns right-to-left (Hebrew) cell text in visual, i.e. reversed, order."""
    k = _key(text)
    return {k, k[::-1]}


# ------------------------------------------------------------------ PDF
def pdf_bends(args: argparse.Namespace) -> int:
    """Parse the report's characteristic and measurement tables (English or Hebrew report).

    Columns are located by their header (either language; the Hebrew report
    lists its columns right-to-left), kinds by their label's words (order-free,
    so a right-to-left extraction still matches), decisions are mapped back to
    the English display word. Bends are keyed by the studio label in the Bend
    column (a row-spanned cell: following rows of the same bend leave it empty).
    """
    import fitz  # PyMuPDF

    kind_labels, text, tokens = _labels()
    kinds = {}
    for key, pair in kind_labels.items():
        for lab in pair:
            for k in _keys(lab):
                kinds[k] = key
    decision_word = {}
    for term in ("PASS", "FAIL", "REVIEW", "UNKNOWN", "NOT_APPLICABLE"):
        for lang in ("en", "he"):
            for k in _keys(tokens[term][lang]):
                decision_word[k] = tokens[term]["en"]
    for k in _keys("n/a") | _keys("לא רלוונטי"):
        decision_word[k] = "Not applicable"
    heads = {name: _keys(text[key][0]) | _keys(text[key][1]) for name, key in (
        ("no", "no"), ("bend", "bend"), ("char", "characteristic"), ("nominal", "nominal"), ("actual", "actual"),
        ("result", "result_col"), ("measurement", "measurement"))}

    def header_map(row: list[Any]) -> dict[str, int] | None:
        found: dict[str, int] = {}
        for i, cell in enumerate(row):
            lines = {_key(x) for x in str(cell or "").split("\n") if x.strip()}
            for name, names in heads.items():
                if lines & names and name not in found:
                    found[name] = i
        return found if {"nominal", "actual", "result"} <= set(found) else None

    import contextlib
    import io

    doc = fitz.open(args.pdf)
    bends: dict[str, Any] = {}
    gms: dict[str, Any] = {}
    for page_no, page in enumerate(doc, start=1):
        with contextlib.redirect_stdout(io.StringIO()):  # PyMuPDF prints an advert on stdout
            tables = page.find_tables().tables
        for tab in tables:
            rows = tab.extract()
            cols = None
            current: str | None = None
            for row in rows:
                if cols is None:
                    cols = header_map(row)
                    continue
                cells = [str(c) if c is not None else None for c in row]
                if "measurement" in cols:  # general measurements
                    label = _first_line(cells[cols["measurement"]])
                    if label:
                        gms.setdefault(label, {"page": page_no, "cells": cells})
                    continue
                if "bend" not in cols or "char" not in cols:
                    continue
                bend_cell = _first_line(cells[cols["bend"]])
                if re.match(r"^B\d+$", bend_cell):
                    current = bend_cell
                    if current not in bends:
                        lines = [_clean(x) for x in str(cells[cols["bend"]]).split("\n") if x.strip()]
                        bends[current] = {"page": page_no, "state_text": lines[1] if len(lines) > 1 else "",
                                          "characteristics": {}}
                if current is None:
                    continue
                kind = kinds.get(_key(cells[cols["char"]] or ""))
                if not kind:
                    continue
                key = kind.split("/")[0] if kind.startswith("BACKGAUGE") else kind
                dec = decision_word.get(_key(_first_line(cells[cols["result"]])), _first_line(cells[cols["result"]]) or None)
                bends[current]["characteristics"].setdefault(key, {
                    "no": _first_line(cells[cols["no"]]) if "no" in cols else None,
                    "expected": _num(cells[cols["nominal"]]), "actual": _num(cells[cols["actual"]]), "decision": dec,
                    "raw": [cells[cols["nominal"]], cells[cols["actual"]], dec],
                })
    return _emit({"pdf": args.pdf, "bends": bends, "general_rows": gms}, ok=bool(bends))


# ------------------------------------------------------------------ compare
def _engine_view(result: dict[str, Any], display: dict[str, str] | None = None) -> dict[str, Any]:
    """Per-bend view keyed by the DISPLAY label (the studio/CAD label the station and
    the PDF show, from ``context.bend_display_labels``), else the result's own label."""
    out: dict[str, Any] = {}
    for b in result.get("bends", []):
        chars = {}
        for c in b.get("characteristics", []):
            key = f"{c['kind']}{'/' + c['side'] if c.get('side') else ''}"
            chars[key] = {"state": c.get("state"), "decision": c.get("decision"), "actual": c.get("actual"), "expected": c.get("expected")}
        key = (display or {}).get(b.get("bend_uid"), b["label"])
        out[key] = {"bend_uid": b.get("bend_uid"), "seq": b.get("sequence_index"), "seq_label": b["label"], "state": b.get("state"), "characteristics": chars}
    return out


DECISION_WORD = {"PASS": "Pass", "FAIL": "Fail", "REVIEW": "Review", "UNKNOWN": "Undetermined"}
DOM_CELLS = {"FLANGE_LENGTH/A": "flange-a", "FLANGE_LENGTH/B": "flange-b", "INCLUDED_ANGLE": "angle"}


def compare(args: argparse.Namespace) -> int:
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    direct = load(args.direct)
    sources = {"direct_cli": direct, "job_engine": load(args.job_engine), "job_rendered": load(args.job_rendered), "station_api": load(args.api)}
    # The direct CLI and the engine output carry no context: the job's rendered result names
    # the display labels (studio/CAD labels) that the API, the screen and the PDF must show.
    display = (sources["job_rendered"].get("context") or {}).get("bend_display_labels") or {}
    views = {k: _engine_view(v, display) for k, v in sources.items()}
    dom = load(args.dom)
    pdf = load(args.pdf).get("bends", {})
    ref = views["direct_cli"]
    problems: list[str] = []
    if (sources["station_api"].get("context") or {}).get("bend_display_labels") != (display or None):
        problems.append("station API context.bend_display_labels != rendered result's")
    rows: list[dict[str, Any]] = []
    # Top-level verdicts.
    tops = {}
    for k, v in sources.items():
        tops[k] = {"summary": (v.get("summary") or {}).get("state"), "validity": (v.get("inspection_validity") or {}).get("state"),
                   "stage": (v.get("stage_decision") or {}).get("status"), "inferred_stage": (v.get("stage_decision") or {}).get("inferred_stage")}
    tops["station_dom"] = {"summary": dom.get("banner_state")}
    for k, v in tops.items():
        if v["summary"] != tops["direct_cli"]["summary"]:
            problems.append(f"summary state {k}={v['summary']} != direct {tops['direct_cli']['summary']}")
        for f in ("validity", "stage", "inferred_stage"):
            if f in v and v[f] != tops["direct_cli"][f]:
                problems.append(f"{f} {k}={v[f]} != direct {tops['direct_cli'][f]}")
    dom_rows = {r["label"]: r for r in dom.get("rows", [])}
    for label, b in sorted(ref.items(), key=lambda kv: kv[1]["seq"]):
        for src in ("job_engine", "job_rendered", "station_api"):
            o = views[src].get(label)
            if o is None:
                problems.append(f"{label} missing in {src}")
                continue
            if o["state"] != b["state"]:
                problems.append(f"{label} state {src}={o['state']} != direct {b['state']}")
            if o["characteristics"] != b["characteristics"]:
                problems.append(f"{label} characteristics differ in {src}")
        dr = dom_rows.get(label)
        pr = pdf.get(label)
        if dr is None:
            problems.append(f"{label} missing in station DOM")
        else:
            if dr.get("seq") is not None and dr.get("seq") != b["seq"]:
                problems.append(f"{label} DOM sequence {dr.get('seq')} != engine {b['seq']}")
            unformed_dom = dr.get("unformed", False)
            if unformed_dom != (b["state"] == "NOT_YET_FORMED"):
                problems.append(f"{label} DOM greyed={unformed_dom} but engine state {b['state']}")
        if pr is None and b["state"] != "NOT_YET_FORMED":
            problems.append(f"{label} missing in PDF")
        for key, c in b["characteristics"].items():
            row = {"bend": label, "seq": b["seq"], "seq_label": b["seq_label"], "state": b["state"], "char": key, "engine_state": c["state"], "expected": c["expected"],
                   "engine_actual": c["actual"], "engine_decision": c["decision"]}
            for src in ("job_engine", "station_api"):
                o = (views[src].get(label) or {}).get("characteristics", {}).get(key) or {}
                row[f"{src}_actual"] = o.get("actual")
                row[f"{src}_decision"] = o.get("decision")
            if dr is not None and key in DOM_CELLS and not dr.get("unformed"):
                cell = (dr.get("cells") or {}).get(DOM_CELLS[key]) or {}
                row["dom_actual_text"] = cell.get("actual")
                row["dom_decision"] = cell.get("decision")
                row["dom_state"] = cell.get("state")
                if cell.get("state") != c["state"]:
                    problems.append(f"{label} {key} DOM state {cell.get('state')} != engine {c['state']}")
                if cell.get("decision") != c["decision"]:
                    problems.append(f"{label} {key} DOM decision {cell.get('decision')} != engine {c['decision']}")
                if c["state"] == "MEASURED":
                    v = _num(cell.get("actual"))
                    if v is None or abs(v - c["actual"]) > 0.05 + 1e-6:
                        problems.append(f"{label} {key} DOM actual {cell.get('actual')} != engine {c['actual']:.3f} (1 dp)")
            if pr is not None:
                pcs = pr.get("characteristics") or {}
                pc = pcs.get(key) or pcs.get(key.split("/")[0])  # the PDF names a backgauge by stage (S<k>), not by side
                if pc is None:
                    if c["state"] != "NOT_APPLICABLE":
                        problems.append(f"{label} {key} missing in PDF")
                else:
                    row["pdf_actual"] = pc.get("actual")
                    row["pdf_decision"] = pc.get("decision")
                    want = DECISION_WORD.get(c["decision"], c["decision"])
                    if c["state"] == "MEASURED":
                        if pc.get("actual") is None or abs(pc["actual"] - c["actual"]) > 0.005 + 1e-6:
                            problems.append(f"{label} {key} PDF actual {pc.get('actual')} != engine {c['actual']:.3f} (2 dp)")
                    if c["decision"] in DECISION_WORD and pc.get("decision") != want:
                        problems.append(f"{label} {key} PDF decision {pc.get('decision')} != engine {c['decision']}")
            rows.append(row)
    # General measurements (GENERAL_MEASUREMENT): engine -> job -> API -> station screen -> PDF.
    gm_rows: list[dict[str, Any]] = []
    pdf_gm = load(args.pdf).get("general_rows", {})
    dom_gm = {g.get("label"): g for g in dom.get("general_measurements", [])}
    def gm_view(res: dict[str, Any]) -> dict[str, Any]:
        return {g["label"]: {k: g["characteristic"].get(k) for k in ("state", "decision", "actual", "expected")} for g in res.get("general_measurements", [])}
    gref = gm_view(direct)
    for src in ("job_engine", "job_rendered", "station_api"):
        if gm_view(sources[src]) != gref:
            problems.append(f"general measurements differ in {src}")
    for label, c in gref.items():
        r = {"label": label, **{f"engine_{k}": v for k, v in c.items()}}
        dg = dom_gm.get(label)
        if dg is None:
            problems.append(f"general measurement {label} missing on the station screen")
        else:
            r.update(dom_state=dg.get("state"), dom_decision=dg.get("decision"), dom_actual=dg.get("actual"))
            if dg.get("state") != c["state"] or dg.get("decision") != c["decision"]:
                problems.append(f"general measurement {label}: screen {dg.get('state')}/{dg.get('decision')} != engine {c['state']}/{c['decision']}")
            if c["state"] == "MEASURED" and (_num(dg.get("actual")) is None or abs(_num(dg.get("actual")) - c["actual"]) > 0.05 + 1e-6):
                problems.append(f"general measurement {label}: screen actual {dg.get('actual')} != engine {c['actual']}")
        pg = pdf_gm.get(label)
        if pg is None:
            problems.append(f"general measurement {label} missing in the PDF")
        else:
            r["pdf_cells"] = pg["cells"]
            if c["state"] == "MEASURED" and not any(_num(x) is not None and abs(_num(x) - c["actual"]) <= 0.005 + 1e-6 for x in pg["cells"] if x):
                problems.append(f"general measurement {label}: engine actual {c['actual']:.3f} not in the PDF row")
        gm_rows.append(r)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"tops": tops, "rows": rows, "general_measurements": gm_rows, "problems": problems, "bends": len(ref)}
    (out / "compare.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    f = lambda v, d=2: "—" if v is None else (f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v))  # noqa: E731
    md = [f"Top-level: " + "; ".join(f"{k}: {v}" for k, v in tops.items()), "",
          "| Bend | State | Char | Expected | E3 CLI actual / decision | Job engine | Station API | Station screen | PDF |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['bend']} | {r['state']} | {r['char']} | {f(r['expected'])} | {f(r['engine_actual'], 3)} {r['engine_decision']} | "
                  f"{f(r.get('job_engine_actual'), 3)} {r.get('job_engine_decision')} | {f(r.get('station_api_actual'), 3)} {r.get('station_api_decision')} | "
                  f"{r.get('dom_actual_text', '—')} {r.get('dom_decision', '')} | {f(r.get('pdf_actual'))} {r.get('pdf_decision', '')} |")
    md += ["", "General measurements:"] + [f"- {g}" for g in gm_rows]
    md += ["", f"Problems: {len(problems)}"] + [f"- {p}" for p in problems]
    (out / "compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return _emit({"bends": len(ref), "rows": len(rows), "problems": problems[:40], "n_problems": len(problems), "tops": tops}, ok=not problems)


# ------------------------------------------------------------------ heatmap
def _read_points(path: Path):
    import numpy as np
    import open3d as o3d

    pc = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pc.points)
    if len(pts) == 0:  # a mesh PLY
        m = o3d.io.read_triangle_mesh(str(path))
        pts = np.asarray(m.vertices)
    return pts


def heatmap(args: argparse.Namespace) -> int:
    import numpy as np
    from scipy.spatial import cKDTree

    job = Path(args.job_dir)
    shop = json.loads((job / "result" / "shopfloor_result.json").read_text(encoding="utf-8"))
    va = shop.get("viewer_assets") or {}
    band = float(va.get("heatmap_band_mm") or 0.5)
    if not va.get("deviations"):
        # The engine published no heatmap (contract reason); nothing may be drawn in its place.
        return _emit({"available": False, "reason": va.get("heatmap_unavailable_reason"),
                      "viewer_deviations_file_absent": not (job / "result" / "viewer_deviations.f32").exists(),
                      "engine_deviations_file_absent": not (job / "engine" / "deviations.f32").exists()}, ok=True)
    vpts = _read_points(job / "result" / "viewer_scan.ply")
    vdev = np.fromfile(job / "result" / "viewer_deviations.f32", dtype="<f4")
    epts = _read_points(job / "engine" / "registered_scan.ply")
    edev = np.fromfile(job / "engine" / "deviations.f32", dtype="<f4")
    doc: dict[str, Any] = {"available": True, "band_mm": band, "viewer_points": int(len(vpts)), "viewer_devs": int(len(vdev)),
                           "engine_points": int(len(epts)), "engine_devs": int(len(edev))}
    ok = len(vpts) == len(vdev) and len(epts) == len(edev) and len(vpts) > 0
    if ok:
        d, idx = cKDTree(epts).query(vpts, k=1)
        doc["max_point_match_mm"] = float(d.max())
        ev = edev[idx]
        both_nan = np.isnan(ev) & np.isnan(vdev)
        diff = np.abs(ev - vdev)
        diff[both_nan] = 0.0
        nan_mismatch = int((np.isnan(ev) != np.isnan(vdev)).sum())
        doc["nan_mismatch"] = nan_mismatch
        doc["max_dev_diff_mm"] = float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0
        ok = doc["max_point_match_mm"] <= 1e-3 and nan_mismatch == 0 and doc["max_dev_diff_mm"] <= 1e-4
        fin = vdev[np.isfinite(vdev)]
        a = np.abs(fin)
        doc["histogram"] = {"abs<=b": float((a <= band).mean()) if len(a) else 0, "b<abs<=2b": float(((a > band) & (a <= 2 * band)).mean()) if len(a) else 0,
                            "abs>2b": float((a > 2 * band).mean()) if len(a) else 0, "nan_fraction": float(1 - len(fin) / len(vdev)),
                            "over_frac(+)": float((fin > band).mean()) if len(fin) else 0, "under_frac(-)": float((fin < -band).mean()) if len(fin) else 0,
                            "median_abs": float(np.median(a)) if len(a) else None, "p95_abs": float(np.percentile(a, 95)) if len(a) else None}
    if args.direct_engine:
        dd = np.fromfile(Path(args.direct_engine) / "deviations.f32", dtype="<f4")
        same = len(dd) == len(edev) and bool(np.array_equal(np.isnan(dd), np.isnan(edev))) and float(np.nanmax(np.abs(dd - edev)) if len(dd) else 0) <= 1e-6
        doc["direct_engine_deviations_equal"] = same
        ok = ok and same
    return _emit(doc, ok=ok)


def overlay(args: argparse.Namespace) -> int:
    import fitz  # PyMuPDF

    from domains.production.reporting.labels import TEXT
    from domains.production.zeroshot.overlay import PASS_LIKE_STATES, ZeroshotOverlayV1

    problems: list[str] = []
    job = Path(args.job_dir)
    path = job / "zeroshot" / "zeroshot_overlay_v1.json"
    status = json.loads((job / "zeroshot" / "overlay_status.json").read_text()) if (job / "zeroshot" / "overlay_status.json").is_file() else None
    if not path.is_file():
        return _emit({"present": False, "overlay_status": status, "problems": ["OVERLAY_MISSING_IN_JOB_OUTPUT"]}, ok=False)
    file_doc = json.loads(path.read_text())
    ov = ZeroshotOverlayV1.model_validate(file_doc)
    api_doc = json.loads(Path(args.api).read_text())
    if api_doc != file_doc:
        problems.append("API_DIFFERS_FROM_JOB_OUTPUT")
    engine = json.loads((job / "engine" / "result.json").read_text())
    if ov.e3_summary_state != engine["summary"]["state"] or ov.scan_sha256 != engine["scan"]["sha256"]:
        problems.append("OVERLAY_NOT_THIS_ENGINE_RESULT")
    if ov.effective_summary_state != ov.e3_summary_state and not (
            ov.e3_summary_state in PASS_LIKE_STATES and ov.effective_summary_state == "REVIEW_REQUIRED"):
        problems.append("OVERLAY_UPGRADED_OR_CHANGED_VERDICT")
    want = [{"id": ln.line_id, "kind": ln.kind, "decision": ln.decision, "bend_uid": ln.bend_uid or ""} for ln in ov.lines]
    dom = json.loads(Path(args.dom).read_text()).get("zs") or {}
    if not dom.get("present"):
        problems.append("OVERLAY_NOT_ON_SCREEN")
    else:
        if dom.get("lines") != want:
            problems.append("SCREEN_LINES_DIFFER")
        if dom.get("lowered") != ov.verdict_lowered or dom.get("effective") != ov.effective_summary_state:
            problems.append("SCREEN_VERDICT_DIFFERS")
        if sorted(dom.get("reasons") or []) != sorted([f.code for f in ov.flags] + [a.code for a in ov.angle_checks]):
            problems.append("SCREEN_REVIEW_REASONS_DIFFER")
    text = "\n".join(page.get_text() for page in fitz.open(args.pdf))
    flat = _clean(text)
    titles = [TEXT["sec_zs"][0], TEXT["sec_zs"][1], TEXT["sec_zs"][1][::-1]]
    has_section = any(_clean(t_) in flat for t_ in titles) or "זיהויים" in flat or "םייוהיז" in flat
    pdf_ids = sorted(set(re.findall(r"(?<![A-Za-z0-9])Z\d{2}(?![0-9])", text)))
    if not has_section:
        problems.append("PDF_SECTION_MISSING")
    if pdf_ids != sorted(w["id"] for w in want):
        problems.append("PDF_LINES_DIFFER")
    doc = {"present": True, "overlay_status": status, "e3": ov.e3_summary_state, "effective": ov.effective_summary_state,
           "lowered": ov.verdict_lowered, "lines": want, "flags": [f.model_dump() for f in ov.flags],
           "withheld": [ov.withheld_bends_located, ov.withheld_bends], "lane": ov.lane.model_dump(),
           "pdf_ids": pdf_ids, "pdf_section": has_section, "dom": dom, "problems": problems}
    return _emit(doc, ok=not problems)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="real_e2e_support")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pdf-bends"); a.add_argument("--pdf", required=True)  # noqa: E702
    c = sub.add_parser("compare")
    for n in ("direct", "job-engine", "job-rendered", "api", "dom", "pdf", "out"):
        c.add_argument(f"--{n}", required=True)
    h = sub.add_parser("heatmap"); h.add_argument("--job-dir", required=True); h.add_argument("--direct-engine")  # noqa: E702
    o = sub.add_parser("overlay")
    for n in ("job-dir", "api", "dom", "pdf"):
        o.add_argument(f"--{n}", required=True)
    args = p.parse_args(argv)
    return {"pdf-bends": pdf_bends, "compare": compare, "heatmap": heatmap, "overlay": overlay}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
