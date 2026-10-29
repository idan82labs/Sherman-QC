"""The BROAD development list and its frozen gold (owner directions 2026-10-17; plan §10 amendment 6a).

Owner directions for wave 4: "aim for line-F1 >= 0.80 with NO part-specific
optimization -- generality over the parts in the mix"; "blind/broad line-F1 is
reported family-macro"; the primary production gate is per-bend correctness
given CAD. The BROAD list is "every CONSUMED scan in case_list_broad_v3.json
with CAD that compiles", for family-macro line-F1 and per-bend checks.

**Candidates** (:func:`broad_rows`): one row per case of
``docs/production/runlog/broad_baseline/case_list_broad_v3.json`` (60 CONSUMED
scans; FRESH, Eyal-50, the codex fetch and possible CAD reference meshes were
excluded when that list was built). A case leaves BROAD, listed with its reason
and before any scoring, when it has no STEP CAD (``NO_STEP_CAD``), when the line
cannot ingest its scan (``SCAN_FORMAT_NOT_ACCEPTED_BY_LINE``: the line takes PLY
and STL; an OBJ copy of a PLY scan is such a case), when its scan or CAD
bytes differ from the list's sha256 (``SCAN_SHA_MISMATCH`` / ``CAD_SHA_MISMATCH``)
or when its STEP does not compile (``CAD_DOES_NOT_COMPILE``: the STEP stays in
every compile-rate denominator). Every path is admitted by the custody guard.

**Gold** (:func:`broad_gold`), all in the raw-scan frame:

- ``gold11_catalog`` (Gold11 and the human non-Gold11 label jobs): the
  amendment-6a recovery of :mod:`.gold` (the label job / C05-C06 label file, with
  its recovered transform); a case without a recovered transform is
  ``NO_RECOVERABLE_TRANSFORM`` (listed, excluded before scoring). The catalog
  container itself is not read; its line count is checked against the list's.
- ``ai_studio`` (``ai_bend_lines_v1``): frame ``source_ply_raw`` and provenance
  ``asset_sha256`` equal to the scan sha256 (identity, ``SOURCE_PLY_RAW_IS_SCAN_RAW``);
  ``no_bends_present`` with no line is a zero-bend false-positive control.
- ``bench_lines`` (STL-only scans): ``lines.json`` with frame ``source_ply_raw``,
  ``scan_sha256`` equal to the scan's, bends with status ``ok``, apex
  ``label_line`` (the declared locus).

Label completeness and flags (``PARTIAL_LENGTH_LINE_B2_72PCT``,
``OWNER_FLAG_LABEL_MAY_MISS_A_BEND``, ``GOLD11_LINES_ARE_CODEX_MULTIVIEW_NOT_HUMAN``)
are carried from the list unchanged; a gold file whose sha256 differs from the
list's is used as it is today and flagged ``GOLD_FILE_CHANGED_SINCE_CASE_LIST``.

**Recipe-count check** (:func:`recipe_count_check`, one rule for every case, no part
or family is named): a recipe-guided engine publishes at most one line per recipe
bend, so a gold with more lines than the frozen DEV recipe has bends caps that
case's recall at ``R / G`` whatever the engine does. Such a case is flagged
``GOLD_LINES_EXCEED_RECIPE_BENDS`` with its ceiling, plus the cause E1's own
policy report shows: ``RECIPE_EXCLUDED_FEATURE`` when E1's frozen per-feature
UNSUPPORTED rule excluded candidates (amendment 4d: a rule-declared feature the
recipe does not inspect), ``GOLD_CONVENTION_CONFLICT`` when the policy merged
extractor pieces of one fold (the owner's convention: a fold interrupted by
windows or reliefs folding the same two plates is one line, so a gold that labels
the pieces separately has more lines than the recipe). A complete, non-staged
scan whose gold has fewer lines than the recipe has bends is flagged
``GOLD_LINES_FEWER_THAN_RECIPE_BENDS`` (no hard ceiling: it exposes precision when
the engine reports every bend it measures). The cases stay scored as they are
(comparable with the reference lanes); :mod:`.physical` reports the ceiling next
to the owner's aim.

**Family-macro** (scored by :mod:`.physical`): per family the micro F1 of its
complete-label line cases, then the mean over families (the broad baseline's
``family_macro_f1``); zero-bend controls count their predicted lines as false
positives of their family. Legacy references for BROAD come from the broad
baseline's own runs (``runs/runtime`` = the shipped worker, ``runs/best_f1`` =
the old best research lane) and are labelled with their run conditions.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import (
    BROAD_CASE_LIST,
    REPO_ROOT,
    case_ref_for,
    custody_check,
    expand_home,
    now_utc,
    read_json,
    rel_to_repo,
    sha256_file,
)

__all__ = [
    "BROAD_SCHEMA",
    "LINE_FORMATS",
    "broad_rows",
    "broad_gold",
    "recipe_count_check",
    "annotate_recipe_counts",
    "baseline_predictions",
]

BROAD_SCHEMA = "sherman_e4b_broad_v1"
LINE_FORMATS = ("ply", "stl")
_RUNS = REPO_ROOT / "docs" / "production" / "runlog" / "broad_baseline" / "runs"


def _abs(path: str | None) -> Path | None:
    if not path:
        return None
    p = expand_home(path)
    return p if p.is_absolute() else (REPO_ROOT / path).resolve()


def broad_rows(case_list: Mapping[str, Any], compile_by_cad: Mapping[str, Mapping[str, Any]], *,
               verify_bytes: bool = True) -> list[dict[str, Any]]:
    """One row per case of the broad case list with its BROAD status (module docstring).

    ``compile_by_cad`` maps a CAD sha256 to its compile-report row (``variants.dev.status``)."""
    rows = []
    for c in case_list["cases"]:
        scan = _abs(c.get("scan_path"))
        cad = _abs(c.get("cad_path"))
        row: dict[str, Any] = {
            "broad_case_id": c["case_id"], "case_ref": case_ref_for(c["scan_sha256"]), "family": c["family_id"],
            "tier": c["tier"], "gold_spec": c.get("gold"), "gold_file_sha256_case_list": c.get("gold_file_sha256"),
            "gold_line_count_case_list": c.get("gold_line_count"), "label_complete": bool(c.get("label_complete", True)),
            "zero_bend_control": bool(c.get("zero_bend_control")), "scan_sha256": c["scan_sha256"],
            "scan_format": c.get("format"), "scan_path": None if scan is None else str(scan),
            "scan_rel": None if scan is None else rel_to_repo(scan), "cad_sha256": c.get("cad_sha256"),
            "cad_path": None if cad is None else str(cad), "cad_rel": None if cad is None else rel_to_repo(cad),
            "stage_hints": list(c.get("stage_hints") or []), "staged_or_incomplete": bool(c.get("staged_or_incomplete")),
            "in_gold11": bool(c.get("in_gold11")), "gold11_id": c.get("gold11_id"),
            "flags": list(c.get("flags") or []), "quality_flags": list(c.get("quality_flags") or []),
            "status": "BROAD", "reason": None}
        if cad is None or cad.suffix.lower() not in (".stp", ".step") or not c.get("cad_sha256"):
            row.update(status="EXCLUDED", reason="NO_STEP_CAD")
        elif str(c.get("format", "")).lower() not in LINE_FORMATS:
            row.update(status="EXCLUDED", reason=f"SCAN_FORMAT_NOT_ACCEPTED_BY_LINE ({c.get('format')})")
        else:
            if verify_bytes:
                if sha256_file(custody_check(scan, "BROAD scan")) != c["scan_sha256"]:
                    row.update(status="EXCLUDED", reason="SCAN_SHA_MISMATCH")
                elif sha256_file(custody_check(cad, "BROAD CAD")) != c["cad_sha256"]:
                    row.update(status="EXCLUDED", reason="CAD_SHA_MISMATCH")
            if row["status"] == "BROAD":
                comp = ((compile_by_cad.get(c["cad_sha256"]) or {}).get("variants") or {}).get("dev") or {}
                row["compile_status"] = comp.get("status", "NOT_COMPILED")
                if row["compile_status"] != "COMPILED":
                    row.update(status="EXCLUDED", reason=f"CAD_DOES_NOT_COMPILE ({row['compile_status']})")
        rows.append(row)
    return rows


def _line(bend_id: str, start: Sequence[float], end: Sequence[float]) -> dict[str, Any]:
    return {"bend_id": str(bend_id), "line_start": [float(v) for v in start], "line_end": [float(v) for v in end]}


def recipe_count_check(gold_lines: int, compile_row: Mapping[str, Any] | None, *, staged: bool,
                       zero_bend_control: bool = False) -> tuple[list[str], dict[str, Any] | None]:
    """``(flags, check)`` of one gold case against its frozen DEV recipe (module docstring)."""
    dev = (((compile_row or {}).get("variants") or {}).get("dev")) or {}
    if dev.get("status") != "COMPILED" or dev.get("bends") is None or zero_bend_control:
        return [], None
    r, g = int(dev["bends"]), int(gold_lines)
    rep = (compile_row or {}).get("dev_policy_report") or {}
    check: dict[str, Any] = {"gold_lines": g, "recipe_bends": r,
                             "rule": "a recipe-guided engine publishes at most one line per recipe bend"}
    flags: list[str] = []
    if g > r:
        flags.append("GOLD_LINES_EXCEED_RECIPE_BENDS")
        causes = []
        if rep.get("excluded_by_unsupported_rule"):
            flags.append("RECIPE_EXCLUDED_FEATURE")
            causes.append(f"E1's frozen UNSUPPORTED rule excluded {len(rep['excluded_by_unsupported_rule'])} "
                          "candidate(s) (rules: " + ", ".join(sorted({str(x.get('rule')) for x in
                                                                    rep['excluded_by_unsupported_rule']})) + ")")
        if rep.get("merge_notes"):
            flags.append("GOLD_CONVENTION_CONFLICT")
            causes.append(f"E1's policy merged extractor pieces of {len(rep['merge_notes'])} fold(s) into one bend "
                          "each (owner convention: a fold interrupted by windows/reliefs is one line); a gold that "
                          "labels the pieces separately has more lines than the recipe (inferred by rule)")
        check.update(true_positive_max=r, recall_ceiling=r / g, f1_ceiling=2.0 * r / (r + g),
                     causes=causes or ["no cause in E1's policy report"])
    elif 0 < g < r and not staged:
        flags.append("GOLD_LINES_FEWER_THAN_RECIPE_BENDS")
        check["note"] = ("no hard ceiling: on a complete scan an engine that reports every recipe bend it measures "
                         "scores the extra lines as false positives unless the gold is incomplete")
    return flags, check


def annotate_recipe_counts(gold_cases: Sequence[dict[str, Any]], cases: Sequence[Mapping[str, Any]],
                           compile_by_cad: Mapping[str, Mapping[str, Any]]) -> int:
    """Add the recipe-count check (:func:`recipe_count_check`) to every gold case in place, against the
    frozen DEV recipe of the CAD its physical case uses; returns the number of flagged cases. A case is
    staged when its frozen stage class is (G3 / BROAD staged scans: unformed bends emit no line)."""
    by_ref = {c["case_ref"]: c for c in cases}
    flagged = 0
    for g in gold_cases:
        c = by_ref.get(g["case_id"])
        if c is None:
            continue
        flags, check = recipe_count_check(len(g["gold_lines"]), compile_by_cad.get(c.get("cad_sha256")),
                                          staged=c.get("stage_class") in ("STAGED_PARTIAL", "BLANK_STAGE0"),
                                          zero_bend_control=bool(g.get("zero_bend_control")))
        g["flags"] = list(g.get("flags") or []) + [f for f in flags if f not in (g.get("flags") or [])]
        g["recipe_count_check"] = check
        flagged += int(bool(flags))
    return flagged


def broad_gold(rows: Sequence[Mapping[str, Any]], human_cases: Sequence[Mapping[str, Any]]
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(gold cases, excluded)`` for the BROAD rows (module docstring); GoldCase-shaped dicts."""
    by_scan = {g["scan_sha256"]: g for g in human_cases}
    gold, excluded = [], []
    for r in rows:
        if r["status"] != "BROAD":
            continue
        spec = r.get("gold_spec") or {}
        kind = spec.get("kind")
        base = {"case_id": r["case_ref"], "broad_case_id": r["broad_case_id"], "family": r["family"],
                "tier": r["tier"], "gold_kind": kind, "label_complete": r["label_complete"],
                "flags": list(r["flags"]), "scan_sha256": r["scan_sha256"]}
        try:
            if kind == "gold11_catalog":
                g = by_scan.get(r["scan_sha256"])
                if g is None:
                    excluded.append({**base, "reason": "NO_RECOVERABLE_TRANSFORM"})
                    continue
                rec = {**base, "frame": g["frame"], "gold_lines": list(g["gold_lines"]),
                       "raw_to_reference": g.get("raw_to_reference"), "residual_mm": g.get("residual_mm"),
                       "transform_method": g.get("transform_method"), "labels_sha256": g.get("labels_sha256"),
                       "gold_source": "label job / Gold11 label file (amendment 6a recovery in gold_transforms.json)",
                       "zero_bend_control": False}
                if r.get("gold_line_count_case_list") not in (None, len(rec["gold_lines"])):
                    rec["flags"] = rec["flags"] + ["GOLD_LINE_COUNT_DIFFERS_FROM_CASE_LIST"]
                gold.append(rec)
                continue
            path = custody_check(REPO_ROOT / spec["path"], "BROAD gold")
            sha = sha256_file(path)
            flags = list(base["flags"])
            if r.get("gold_file_sha256_case_list") and sha != r["gold_file_sha256_case_list"]:
                flags.append("GOLD_FILE_CHANGED_SINCE_CASE_LIST")
            doc = read_json(path)
            if doc.get("coordinate_frame") != "source_ply_raw":
                excluded.append({**base, "reason": f"UNKNOWN_FRAME:{doc.get('coordinate_frame')}"})
                continue
            if kind == "ai_studio":
                prov_p = path.parent / "provenance" / path.name
                prov = read_json(custody_check(prov_p, "BROAD gold provenance")) if prov_p.is_file() else {}
                if prov.get("asset_sha256") != r["scan_sha256"]:
                    excluded.append({**base, "reason": "SCAN_SHA_MISMATCH", "provenance_asset_sha256":
                                     prov.get("asset_sha256")})
                    continue
                lines = [_line(b.get("bend_id") or f"B{i:02d}", b["line_start"], b["line_end"])
                         for i, b in enumerate(doc.get("bends") or [], 1)]
                zero = bool(doc.get("no_bends_present")) and not lines
            elif kind == "bench_lines":
                if doc.get("scan_sha256") != r["scan_sha256"]:
                    excluded.append({**base, "reason": "SCAN_SHA_MISMATCH"})
                    continue
                lines = [_line(b["bend_id"], b["label_line"]["start"], b["label_line"]["end"])
                         for b in doc.get("bends") or [] if b.get("status") == "ok" and b.get("label_line")]
                zero = bool(doc.get("no_bends_present")) and not lines
            else:
                excluded.append({**base, "reason": f"UNKNOWN_GOLD_KIND:{kind}"})
                continue
        except (OSError, KeyError, ValueError, TypeError) as exc:
            excluded.append({**base, "reason": f"GOLD_UNREADABLE:{type(exc).__name__}"})
            continue
        if r.get("zero_bend_control") != zero:
            flags.append("ZERO_BEND_FLAG_DIFFERS_FROM_CASE_LIST")
        if not zero and r.get("gold_line_count_case_list") not in (None, len(lines)):
            flags.append("GOLD_LINE_COUNT_DIFFERS_FROM_CASE_LIST")
        gold.append({**base, "flags": flags, "frame": "raw_scan", "gold_lines": lines, "raw_to_reference": None,
                     "residual_mm": 0.0, "transform_method": "SOURCE_PLY_RAW_IS_SCAN_RAW", "labels_sha256": sha,
                     "gold_source": rel_to_repo(path), "zero_bend_control": zero})
    return gold, excluded


def baseline_predictions(rows: Sequence[Mapping[str, Any]], *, runs_root: Path = _RUNS
                         ) -> dict[str, dict[str, Any]]:
    """The broad baseline's own line predictions per BROAD case (raw frame), for the reference lanes:
    ``runtime`` (the shipped worker, run as the API runs it: repo-default config, 9 GiB) and
    ``best_f1`` (the old best research lane). Read-only; their conditions are recorded, they are
    never re-run here."""
    lanes = {"runtime": "predictions_runtime.json", "best_f1": "predictions_best_f1.json"}
    out: dict[str, dict[str, Any]] = {lane: {} for lane in lanes}
    meta: dict[str, Any] = {}
    for lane, fname in lanes.items():
        for r in rows:
            if r["status"] != "BROAD":
                continue
            p = runs_root / lane / r["broad_case_id"] / fname
            if not p.is_file():
                continue
            doc = read_json(custody_check(p, "broad baseline predictions"))
            scan_sha = (doc.get("scan") or {}).get("sha256")
            if scan_sha not in (None, r["scan_sha256"]):
                continue
            lines = []
            for i, row in enumerate(doc.get("owned_bend_regions") or [], 1):
                ends = row.get("span_endpoints")
                if ends:
                    start, end = ends[0], ends[1]
                elif row.get("line_start") is not None and row.get("line_end") is not None:
                    start, end = row["line_start"], row["line_end"]
                else:
                    continue
                lines.append(_line(row.get("bend_id") or f"P{i:02d}", start, end))
            out[lane][r["case_ref"]] = {"lines": lines, "status": doc.get("status"), "file": rel_to_repo(p),
                                        "sha256": sha256_file(p)}
        meta[lane] = {"cases": len(out[lane]), "file": fname,
                      "conditions": ("backend/bend_analysis_worker.py as the API invokes it (repo-default config "
                                     "resolution, 9 GiB, opaque inputs); not the E4b legacy-arm conditions"
                                     if lane == "runtime" else
                                     "old best research lane (A0 typed union + gated CAD fusion), research code")}
    return {"lanes": out, "meta": meta, "created_at": now_utc()}
