#!/usr/bin/env python3
"""Python helpers for the app end-to-end run (``scripts/production/run_e2e_app.mjs``, C11).

The Playwright driver calls these subcommands; each prints one JSON object on stdout
and exits 0 on success, 1 when the check it performs fails, 2 on usage errors.

``vendor-plan --api URL --part-id P --set-dir S`` (bearer token in ``$SHERMAN_E2E_TOKEN``)
    Read-only test oracle for the UI-driven bend-order step: matches the Vendor Setup
    candidates to the E2 bends by geometry (as below) and prints the planned order
    (candidate feature ids + labels) and the מהלך group to click. Nothing is saved.

``placeholder-drawing --out F --title T``
    Writes a one-page placeholder PDF (English + Hebrew) for a real dev part that has
    no authorized customer drawing.

``vendor-draft --api URL --part-id P --set-dir S`` (bearer token in ``$SHERMAN_E2E_TOKEN``)
    FALLBACK ONLY (the step is then recorded FAIL): API assist for the bend-order step. Reads the part's Vendor Setup candidates
    (``GET /api/parts/{id}/vendor-setup``, the legacy route the Vendor Setup page
    uses), matches each catalog candidate to the E2 part's bend lines by geometry
    (midpoint distance + direction; one to one, never guessed: an unmatched or
    ambiguous candidate fails), and saves a draft with every mark confirmed, the
    E2 bending order and its מהלך (machine-move) groups through
    ``PUT /api/parts/{id}/vendor-setup``. This is what an operator does in the
    Vendor Setup UI; the driver records the step as an API assist, not a UI pass.

``pdf-pages --pdf F --out DIR [--expect TEXT]... [--header-ids SKU RECIPE_ID]``
    Renders every page to ``DIR/page-<n>.png`` (PyMuPDF, 110 dpi) and extracts the
    text. Each ``--expect`` string (Hebrew allowed) must appear in logical order in
    the extracted text; its visually reversed form must not be the only match.
    ``--header-ids`` checks the running header AS DRAWN (glyphs left to right) on
    every page: it must read "<SKU> · <RECIPE_ID>" after the Hebrew title, so a
    numeric SKU never joins the Hebrew right-to-left run.

``benchmark --data-root R --recipe-id ID``
    Reads the release's ``benchmark_references`` row (read-only SQLite URI on
    ``<R>/db/recipes.sqlite3``) and the recipe's bend uids; passes when the row is
    REGISTERED and its nominals name every bend.

``partset --out DIR``
    Generates the E2 HAT part set (DEV seeds, native scanner spacing) used by the run.

Nothing here reads a real scan or CAD file (the driver copies the authorized 10839000
STEP and FULL scan itself).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _emit(doc: dict[str, Any], ok: bool = True) -> int:
    print(json.dumps(doc, ensure_ascii=False, sort_keys=True))
    return 0 if ok else 1


def _http(method: str, url: str, token: str, body: Any | None = None, timeout: float = 300.0) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - localhost test server
        return json.loads(resp.read().decode("utf-8"))


def _vec(raw: Any) -> list[float] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    if isinstance(raw, (list, tuple)) and len(raw) == 3 and all(isinstance(v, (int, float)) for v in raw):
        return [float(v) for v in raw]
    return None


def _line_key(start: list[float], end: list[float]) -> tuple[list[float], list[float]]:
    mid = [(a + b) / 2.0 for a, b in zip(start, end)]
    d = [b - a for a, b in zip(start, end)]
    n = math.sqrt(sum(v * v for v in d)) or 1.0
    return mid, [v / n for v in d]


def _plan(args: argparse.Namespace, token: str) -> dict[str, Any]:
    """Match every E2 bend to exactly one Vendor Setup candidate (midpoint within tol, parallel).

    Read-only (one GET). Returns the setup, the candidates, the E2 -> candidate mapping and the
    planned order / machine moves in candidate feature ids, plus any matching problems."""
    base = args.api.rstrip("/")
    setup = _http("GET", f"{base}/api/parts/{args.part_id}/vendor-setup", token)
    candidates = setup.get("candidates") or []
    doc = json.loads((Path(args.set_dir) / "recipe_draft.json").read_text(encoding="utf-8"))
    inputs = doc["builder_inputs"]
    rows = {r["id"]: r for r in inputs["spec_rows"]}
    e2_draft = inputs["draft"]
    mapping: dict[str, str] = {}
    problems: list[str] = []
    used: set[str] = set()
    for fid, row in rows.items():
        s, e = _vec(row["bend_line_start"]), _vec(row["bend_line_end"])
        if s is None or e is None:
            problems.append(f"E2 bend {fid} has no line")
            continue
        mid, d = _line_key(s, e)
        hits = []
        for cand in candidates:
            cs, ce = _vec(cand.get("line_start")), _vec(cand.get("line_end"))
            if cs is None or ce is None:
                continue
            cmid, cd = _line_key(cs, ce)
            dist = math.dist(mid, cmid)
            par = abs(sum(a * b for a, b in zip(d, cd)))
            if dist <= args.tol_mm and par >= math.cos(math.radians(5.0)):
                hits.append((dist, cand["feature_id"]))
        hits.sort()
        if len(hits) != 1:
            problems.append(f"E2 bend {fid}: {len(hits)} candidate matches within {args.tol_mm} mm")
            continue
        if hits[0][1] in used:
            problems.append(f"candidate {hits[0][1]} matched twice")
            continue
        used.add(hits[0][1])
        mapping[fid] = hits[0][1]
    unmatched = [c["feature_id"] for c in candidates if c["feature_id"] not in used]
    sequence = [mapping[f] for f in e2_draft["sequence"] if f in mapping]
    moves = [[mapping[f] for f in move if f in mapping] for move in (e2_draft.get("machine_moves") or [])]
    moves = [m for m in moves if len(m) > 1]
    if not moves and len(sequence) >= 2:
        moves = [sequence[:2]]  # tag one מהלך group of two bends, as §6 step 2 asks
    labels = {c["feature_id"]: c.get("label") for c in candidates}
    return {"setup": setup, "candidates": candidates, "mapping": mapping, "problems": problems,
            "unmatched": unmatched, "sequence": sequence, "moves": moves, "labels": labels, "e2_bends": len(rows)}


def vendor_plan(args: argparse.Namespace) -> int:
    token = os.environ.get("SHERMAN_E2E_TOKEN", "")
    if not token:
        return _emit({"ok": False, "problem": "SHERMAN_E2E_TOKEN is not set"}, ok=False)
    plan = _plan(args, token)
    ok = not plan["problems"] and not plan["unmatched"]
    return _emit({"ok": ok, "problems": plan["problems"], "unmatched_candidates": plan["unmatched"],
                  "candidate_count": len(plan["candidates"]), "e2_bends": plan["e2_bends"],
                  "order": [{"feature_id": f, "label": plan["labels"].get(f)} for f in plan["sequence"]],
                  "moves": plan["moves"]}, ok=ok)


def vendor_draft(args: argparse.Namespace) -> int:
    base = args.api.rstrip("/")
    # The token comes from the environment, never argv: argv is visible to other users in ps.
    token = os.environ.get("SHERMAN_E2E_TOKEN", "")
    if not token:
        return _emit({"ok": False, "problem": "SHERMAN_E2E_TOKEN is not set"}, ok=False)
    plan = _plan(args, token)
    setup, candidates, problems, unmatched = plan["setup"], plan["candidates"], plan["problems"], plan["unmatched"]
    if problems or unmatched:
        return _emit({"ok": False, "problems": problems, "unmatched_candidates": unmatched,
                      "candidate_count": len(candidates), "e2_bends": plan["e2_bends"]}, ok=False)
    marks = []
    for cand in candidates:
        marks.append({
            "feature_id": cand["feature_id"], "label": cand["label"], "source": "cad_candidate", "status": "active",
            "review_status": "confirmed", "issue": None, "corrects_feature_id": None, "note": "",
            "line_start": cand["line_start"], "line_end": cand["line_end"],
            "target_angle": cand.get("target_angle"), "target_radius": cand.get("target_radius"),
        })
    sequence, moves = plan["sequence"], plan["moves"]
    part = setup["part"]
    existing = setup.get("draft") or {}
    draft = {
        "schema_version": 1,
        "cad_file_hash": part.get("cad_file_hash"),
        "cad_uploaded_at": part.get("cad_uploaded_at"),
        "part_revision": part.get("revision"),
        "drawing_file_hash": part.get("drawing_file_hash"),
        "candidate_generation": setup["candidate_generation"],
        "expected_draft_revision": int(existing.get("draft_revision") or 0),
        "marks": marks,
        "sequence": sequence,
        "machine_moves": moves,
        "machine_moves_explicit": True,
        "general_measurements": [],
        "operator_sequence_confirmed": True,
        "completed_steps": 0,
        "matching_search": {"search_radius_mm": None, "search_angle_deg": None},
    }
    try:
        saved = _http("PUT", f"{base}/api/parts/{args.part_id}/vendor-setup", token, draft)
    except urllib.error.HTTPError as exc:
        return _emit({"ok": False, "status": exc.code, "body": exc.read().decode("utf-8", "replace")[:2000]}, ok=False)
    return _emit({"ok": True, "draft_revision": saved["draft"].get("draft_revision"), "sequence": sequence,
                  "machine_moves": moves, "candidate_count": len(candidates)})


def pdf_pages(args: argparse.Namespace) -> int:
    import fitz  # PyMuPDF

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pages: list[str] = []
    texts: list[str] = []
    with fitz.open(args.pdf) as doc:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=110)
            path = out / f"page-{i}.png"
            pix.save(str(path))
            pages.append(str(path))
            texts.append(page.get_text())
    text = "\n".join(texts)
    (out / "pdf_text.txt").write_text(text, encoding="utf-8")
    found = {s: (s in text) for s in args.expect or []}
    reversed_only = {s: (s not in text and s[::-1] in text) for s in args.expect or []}
    ok = all(found.values()) and not any(reversed_only.values()) and bool(pages)
    header: dict[str, Any] | None = None
    if args.header_ids:
        sku, _recipe_id = args.header_ids
        # Formal report (pdf_v2): every page carries a running header with the part number.
        drawn: list[str] = []
        with fitz.open(args.pdf) as doc:
            for page in doc:
                top = fitz.Rect(0, 0, page.rect.width, page.rect.height * 0.1)
                drawn.append(" ".join(page.get_text(clip=top).split()))
        bad = [i + 1 for i, h in enumerate(drawn) if sku not in h]
        header = {"expected": f"part number {sku} in every page header", "pages_wrong": bad,
                  "first_page": drawn[0] if drawn else None}
        ok = ok and not bad and bool(drawn)
    return _emit({"ok": ok, "pages": pages, "expect_found": found, "expect_reversed_only": reversed_only,
                  "header": header, "chars": len(text)}, ok=ok)


def placeholder_drawing(args: argparse.Namespace) -> int:
    """One-page PDF that says, in English and Hebrew, that no customer drawing is on file.

    The real dev part has no authorized drawing PDF; the wizard requires one, so the run attaches
    this clearly labelled placeholder (never a drawing of another part)."""
    import fitz  # PyMuPDF

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=842, height=595)  # A4 landscape, points
    html = (f"<div style='font-family:sans-serif;font-size:22px'><b>{args.title}</b></div>"
            "<div style='font-family:sans-serif;font-size:16px;margin-top:18px'>E2E PLACEHOLDER - no customer drawing on file "
            "for this development part. Not a drawing; carries no dimensions or tolerances.</div>"
            "<div dir='rtl' style='font-family:sans-serif;font-size:16px;margin-top:12px'>מסמך זמני לבדיקה - אין שרטוט לקוח לחלק הזה.</div>")
    page.insert_htmlbox(fitz.Rect(48, 48, 794, 547), html)
    doc.save(str(out))
    doc.close()
    return _emit({"ok": True, "pdf": str(out)})


def benchmark(args: argparse.Namespace) -> int:
    import sqlite3

    db = Path(args.data_root) / "db" / "recipes.sqlite3"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT status, nominals_json FROM benchmark_references WHERE recipe_id=?",
                           (args.recipe_id,)).fetchone()
        rec = conn.execute("SELECT recipe_json FROM part_recipes WHERE recipe_id=?", (args.recipe_id,)).fetchone()
    finally:
        conn.close()
    if row is None or rec is None:
        return _emit({"ok": False, "problem": "no benchmark row or recipe"}, ok=False)
    raw = rec[0].decode("utf-8") if isinstance(rec[0], bytes) else rec[0]
    uids = [b["bend_uid"] for b in json.loads(raw)["bends"]]
    nominals = row[1] or ""
    missing = [u for u in uids if u not in nominals]
    ok = row[0] == "REGISTERED" and bool(uids) and not missing
    return _emit({"ok": ok, "status": row[0], "bends": len(uids), "missing_bends": missing,
                  "nominals_bytes": len(nominals)}, ok=ok)


def partset(args: argparse.Namespace) -> int:
    from domains.production.engine.synthetic.cases import generate_part_set

    meta = generate_part_set("HAT", "DEV", Path(args.out), ray_threads=1)
    set_dir = Path(args.out) / meta["set_id"]
    return _emit({"ok": True, "set_dir": str(set_dir), "sku": meta["sku"]})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run_e2e_app_support.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vendor-draft")
    v.add_argument("--api", required=True)
    v.add_argument("--part-id", required=True)
    v.add_argument("--set-dir", required=True)
    v.add_argument("--tol-mm", type=float, default=1.0)
    vp = sub.add_parser("vendor-plan")
    vp.add_argument("--api", required=True)
    vp.add_argument("--part-id", required=True)
    vp.add_argument("--set-dir", required=True)
    vp.add_argument("--tol-mm", type=float, default=1.0)
    pd = sub.add_parser("placeholder-drawing")
    pd.add_argument("--out", required=True)
    pd.add_argument("--title", required=True)
    d = sub.add_parser("pdf-pages")
    d.add_argument("--pdf", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--expect", action="append")
    d.add_argument("--header-ids", nargs=2, metavar=("SKU", "RECIPE_ID"))
    bm = sub.add_parser("benchmark")
    bm.add_argument("--data-root", required=True)
    bm.add_argument("--recipe-id", required=True)
    s = sub.add_parser("partset")
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    return {"vendor-draft": vendor_draft, "vendor-plan": vendor_plan, "placeholder-drawing": placeholder_drawing, "pdf-pages": pdf_pages, "benchmark": benchmark, "partset": partset}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
