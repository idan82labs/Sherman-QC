"""E4b pre-A1 freeze: command line and work layout (plan §10 amendments 4, 6, 7, 8; §9 custody as widened
by the owner's 2026-10-17 wave-4 directions).

``python -m domains.production.engine.eval.freeze.prefreeze <command> ...`` with
``PYTHONPATH=.:backend OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1``. One heavy
process at a time (every heavy command runs under the shared lock
``/private/tmp/claude-501/sherman_geom.lock``); every child (E1 compile, recipe
build, legacy worker, E3) runs under :func:`..procmon.run_monitored` with a
process-group RSS limit (4 GiB; the legacy worker 12 GiB, its line-contract limit).

**Builder commands** (E4b, in freeze order). ``--work W`` is the builder's own work
directory; everything they produce outside the run logs goes to its work area
``W/builder_SEALED/`` (pre-split and superseded material: A1 never reads it):

1. ``inventory``           hash and classify the authorized inputs (:mod:`.inventory`)
2. ``gold --work W``        Gold transform recovery, before any scoring (:mod:`.gold`): the full
   list ``freeze/gold_transforms_devcheck_SEALED.json`` (it names the DEV-CHECK Gold cases once
   the draw is made) and ``gold_cases_ALL.json`` in the work area
3. ``recipes --work W``     DEV_AUTO_CONFIRMED recipes + E1 compiles of every unique STEP (:mod:`.recipes`)
4. ``cases --work W``       case lists, BROAD list + gold (recipe-count flags and ceilings), DEV-CHECK
   draw, public/sealed split, the A1 part ``freeze/gold_transforms.json`` (:mod:`.cases`, :mod:`.broad`)
5. ``legacy --work W``      the legacy control arm, serial, + its G4 renders (:mod:`.legacy`)
6. ``g3-material --work W`` the G3 truth-table material, no engine output (:mod:`.g3_material`); the
   reviewer-blind display placements go to the work area, never into the reviewer-facing tree
7. ``ledger``               the FREEZE entry + anchor (:mod:`.ledger`); ``verify`` re-checks it

**A1-facing commands** (A1, the final DEV-CHECK run, R1). ``--work W`` is a fresh directory of the
caller's own outside the repository: the builder's work directory (it holds ``builder_SEALED/``) is
refused, and without ``--include-sealed`` so is a ``*SEALED*`` directory or one holding
``devcheck_SEALED/``. They read the A1 audience only (``cases_A1.json``, ``recipes/``); the DEV-CHECK
material needs ``--include-sealed``, which only the final DEV-CHECK run and R1 pass, and its products
go to ``W/devcheck_SEALED/``:

- ``compile-frozen --work W`` recompiles the frozen recipes into ``W/compiled/dev/<sku>`` and checks
  every compiled-part sha256 against the freeze;
- ``run-physical --work W --cases C`` runs E3 on a frozen case document. Amendment-7 gate: E3 never
  runs on a truth-relevant case (:func:`.physical.truth_relevant`) before a frozen visual truth table
  that covers its template rows is recorded in the ledger; refused cases are listed (exit 2), the
  others run;
- ``score-physical --work W --cases C [--truth-table T] [--ledger-anchor A] --out O`` scores G2, G3
  and the BROAD report (:mod:`.physical`);
- ``record-truth-table`` validates the frozen truth table(s) against the templates and records them
  in one FREEZE entry of the ledger (the DEV-CHECK table by sha256 only).

Run-log layout (E4b owns ``docs/production/runlog/{freeze,legacy_control,g3_truth_material}``):

- ``freeze/``: ``README.md``, ``inventory.json``, ``gold_transforms.json`` (the A1 part),
  ``compile_report.json``, ``cases_A1.json`` and the A1 id lists (``g2_complete_A1``,
  ``g3_staged_A1``, ``known_good_A1``, ``repeatability_A1``, ``broad_A1``, ``broad_staged_A1``), the
  A1 gold (``gold_g2a_A1``, ``gold_g2a_relabel_A1``, ``gold_broad_A1``),
  ``broad_reference_lanes_A1.json``, ``recipes/``, ``thresholds_plan_verbatim.md``,
  ``thresholds_documents.json``, ``owner_directions.json``, ``LEDGER.json``, ``ANCHOR.txt``; sealed:
  ``devcheck_SEALED.json``, ``gold_transforms_devcheck_SEALED.json``,
  ``broad_reference_lanes_devcheck_SEALED.json``, ``recipes_devcheck_SEALED/``;
- ``legacy_control/``: ``README.md``, ``SUMMARY.json`` and per case ``run.json``,
  ``legacy_case.json``, ``renders/``; sealed: ``devcheck_SEALED/``;
- ``g3_truth_material/``: ``REVIEWER_INSTRUCTIONS.md``, ``MANIFEST.json``,
  ``truth_table_TEMPLATE.json``, per case images; sealed: ``devcheck_SEALED/``.

A1 never reads a ``*SEALED*`` path (DEV-CHECK or builder material).

Exit codes: 0 done, 2 usage / input / custody / policy error (including a case refused by the
amendment-7 gate), 3 internal error.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from .common import (
    BENCHMARK_MANIFEST,
    BROAD_CASE_LIST,
    FREEZE_DIR,
    FREEZE_TOOLS_VERSION,
    G3_DIR,
    GOLD_TRANSFORMS_FULL_NAME,
    LEGACY_DIR,
    REPO_ROOT,
    RUNLOG,
    SOURCE_ASSETS,
    WORK_BUILDER_DIR,
    WORK_SEALED_DIR,
    CustodyError,
    compact_png,
    custody_check,
    is_sealed_path,
    now_utc,
    read_json,
    rel_to_repo,
    sha256_file,
    write_json,
)

__all__ = ["main", "build_parser", "LEGACY_LISTS", "AUDIENCES"]

#: Plan lists the legacy control arm runs on (§5 G2, G3; amendment 6f).
LEGACY_LISTS = ("G2_COMPLETE", "G3_STAGED", "REPEATABILITY")
#: Lists whose cases get G3 truth-table material (every listed case; sections for the staged ones).
G3_MATERIAL_LISTS = ("G2_COMPLETE", "G3_STAGED", "REPEATABILITY", "BROAD")
#: The two audiences of the freeze: A1 (public run logs) and the sealed DEV-CHECK subset.
AUDIENCES = ("A1", "DEVCHECK")
LEDGER_PATH = FREEZE_DIR / "LEDGER.json"


def _print(doc: Any) -> None:
    sys.stdout.write(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")


def _progress(msg: str) -> None:
    sys.stdout.write("PROGRESS\t" + json.dumps({"message": msg}) + "\n")
    sys.stdout.flush()


def _work(a) -> Path:
    if a.work is None:
        raise ValueError("--work DIR is required (large artifacts never go into the repository)")
    w = Path(a.work).resolve()
    if w.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"--work {w} is inside the repository")
    w.mkdir(parents=True, exist_ok=True)
    return w


def _builder(a) -> Path:
    """The freeze builder's work area ``W/builder_SEALED`` (module docstring)."""
    b = _work(a) / WORK_BUILDER_DIR
    b.mkdir(parents=True, exist_ok=True)
    return b


def _eval_root(a, *, sealed: bool) -> Path:
    """The work root of an A1-facing command: ``W`` (A1) or ``W/devcheck_SEALED`` (the final DEV-CHECK
    run, ``--include-sealed``). The builder's work directory is always refused; without
    ``--include-sealed`` a ``*SEALED*`` directory or one holding ``devcheck_SEALED/`` is refused too."""
    if a.work is None:
        raise ValueError("--work DIR is required (large artifacts never go into the repository)")
    include = bool(getattr(a, "include_sealed", False))
    w = Path(a.work).resolve()
    if (w / WORK_BUILDER_DIR).exists():
        raise ValueError(f"--work {w} is the freeze builder's work directory (it holds {WORK_BUILDER_DIR}/): "
                         "run A1-facing commands in a fresh work directory of your own")
    if not include and (is_sealed_path(w) or (w / WORK_SEALED_DIR).exists()):
        raise ValueError(f"--work {w} holds DEV-CHECK material: A1 uses a fresh work directory of its own "
                         "(only the final DEV-CHECK run and R1 pass --include-sealed)")
    if sealed and not include:
        raise ValueError("DEV-CHECK material needs --include-sealed (the final DEV-CHECK run and R1 only)")
    root = _work(a)
    if sealed:
        root = root / WORK_SEALED_DIR
        root.mkdir(parents=True, exist_ok=True)
    return root


def _case_doc(a) -> tuple[dict[str, Any], bool]:
    """``(frozen case document, is DEV-CHECK)`` of an A1-facing command; a ``*SEALED*`` path is refused
    before it is opened unless ``--include-sealed``."""
    from .cases import DEVCHECK_SCHEMA

    p = Path(a.cases)
    include = bool(getattr(a, "include_sealed", False))
    if is_sealed_path(p) and not include:
        raise ValueError(f"{p.name} is DEV-CHECK material (a *SEALED* path): only the final DEV-CHECK run and R1 "
                         "pass --include-sealed")
    doc = read_json(p)
    sealed = is_sealed_path(p) or doc.get("schema") == DEVCHECK_SCHEMA
    if sealed and not include:
        raise ValueError(f"{p.name} is the DEV-CHECK case document: only the final DEV-CHECK run and R1 pass "
                         "--include-sealed")
    return doc, sealed


def _audience_dir(base: Path, audience: str) -> Path:
    return base if audience == "A1" else base / "devcheck_SEALED"


def _frozen_table(g3_base: Path) -> Path | None:
    p = g3_base / "truth_table_FROZEN.json"
    return p if p.is_file() else None


def _not_inside(src: Path | None, target: Path, flag: str) -> None:
    if src is not None and Path(src).resolve().is_relative_to(target.resolve()):
        raise ValueError(f"{flag} {src} lies inside {target}, which this command rewrites: pass a copy "
                         f"(e.g. W/{WORK_BUILDER_DIR}/prev_passN/...)")


def scan_path_of(case: dict[str, Any]) -> Path:
    """Absolute scan path of a frozen case (older case documents carry only ``scan_rel``)."""
    if case.get("scan_path"):
        return Path(case["scan_path"])
    return SOURCE_ASSETS / str(case["scan_rel"]).split("02_CAD_QC_3D/", 1)[1]


def cad_path_of(case: dict[str, Any]) -> Path:
    if case.get("cad_path"):
        return Path(case["cad_path"])
    return SOURCE_ASSETS / str(case["cad_rel"]).split("02_CAD_QC_3D/", 1)[1]


def _case_docs(fd: Path) -> list[tuple[str, dict[str, Any]]]:
    """Both audiences' case documents: freeze builder commands only (A1-facing commands use :func:`_case_doc`)."""
    return [("A1", read_json(fd / "cases_A1.json")), ("DEVCHECK", read_json(fd / "devcheck_SEALED.json"))]


def _compiled_dir(root: Path, case: dict[str, Any]) -> Path:
    return root / "compiled" / "dev" / case["recipe_sku"]


def _prev_compiled(path: str | Path | None) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    if path and Path(path).is_file():
        for row in read_json(path)["rows"]:
            out[row["sku"]] = ((row.get("variants") or {}).get("dev") or {}).get("compiled_part_sha256")
    return out


# --------------------------------------------------------------------------------------------
# builder commands
# --------------------------------------------------------------------------------------------


def cmd_inventory(a) -> int:
    from .inventory import build_inventory

    inv = build_inventory(SOURCE_ASSETS)
    write_json(Path(a.freeze_dir) / "inventory.json", inv)
    _print({"unique_files": inv["unique_files"], "counts": inv["unique_counts"],
            "external_counts": inv["external_counts"], "external_problems": inv["external_mismatched_or_missing"],
            "sha256": inv["inventory_sha256"]})
    return 0


def cmd_gold(a) -> int:
    from .gold import prior_records, recover_all, relabel_snapshot

    work, b = _work(a), _builder(a)
    fd = Path(a.freeze_dir)
    sources = [Path(p) for p in a.prior] if a.prior else [
        b / "prev_pass1" / "freeze" / "gold_transforms.json",
        RUNLOG / "broad_baseline" / "gold11_frame_recovery.json",
        RUNLOG / "broad_baseline" / "broad_frame_records.json"]
    priors = prior_records(sources, own_roots=[work])
    doc, cases = recover_all(priors=priors)
    doc["prior_sources"] = [{"path": str(p), "sha256": sha256_file(p)} for p in sources if p.is_file()]
    doc["gold11_relabel_v1"] = relabel_snapshot(cases)
    doc["audience"] = ("the freeze builder, the final DEV-CHECK run and R1: every Gold case (it names the DEV-CHECK "
                       "Gold cases once the draw is made); A1 reads its A1 part gold_transforms.json")
    write_json(fd / GOLD_TRANSFORMS_FULL_NAME, doc)
    write_json(b / "gold_cases_ALL.json", {"created_at": now_utc(), "cases": cases})
    _print({"recovered_gold11": doc["recovered"], "excluded": len(doc["excluded"]), "gold_lines": doc["gold_lines_recovered"],
            "human_non_gold11": doc["human_non_gold11_recovered"], "reused_priors": doc["reused_priors"],
            "relabel": {"status": doc["gold11_relabel_v1"]["status"],
                        "gold11_cases_covered": len(doc["gold11_relabel_v1"]["gold11_cases_covered"])}})
    return 0


def cmd_recipes(a) -> int:
    from .recipes import build_and_compile_all

    b = _builder(a)
    inv = read_json(Path(a.freeze_dir) / "inventory.json")
    report = build_and_compile_all(inv, b, only_sha=a.only or None, progress=_progress)
    write_json(b / "compile_report_full.json", report)
    _print({"rates": report["rates"], "rates_source_assets": report["rates_source_assets"], "gate": report["gate"],
            "meets_floor": report["meets_floor"], "meets_floor_source_assets": report["meets_floor_source_assets"]})
    return 0


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _id_list(fd: Path, fname: str, lst: str, rule: str, cases: list[str]) -> None:
    write_json(fd / f"{fname}.json", {"schema": "sherman_e4b_case_id_list_v1", "list": lst, "rule": rule,
                                      "cases": cases, "source": "docs/production/runlog/freeze/cases_A1.json"})


def _built_recipe(b: Path, row: dict[str, Any], variant: str) -> Path | None:
    """The builder's recipe file of one compile-report row (the work area, whatever path the report
    recorded), checked against the recorded sha256."""
    v = (row.get("variants") or {}).get(variant) or {}
    if not v.get("recipe_path"):
        return None
    p = b / "recipes" / row["sku"] / f"{variant}.recipe.json"
    if not p.is_file():
        p = Path(v["recipe_path"])
    if v.get("recipe_file_sha256") and sha256_file(p) != v["recipe_file_sha256"]:
        raise ValueError(f"{row['sku']} {variant}: the recipe file changed since the compile report")
    return p


def _recipe_ceilings(doc: dict[str, Any]) -> dict[str, Any]:
    """Report-only, before any run: the best line-F1 a recipe-guided engine can reach on this document's
    frozen gold (:func:`.physical.recipe_line_ceiling`; ``R`` = the frozen DEV recipe's compiled bends)."""
    from ..thresholds import G2_LINE_THRESHOLDS
    from .physical import BROAD_OWNER_TARGET, recipe_line_ceiling

    pops = {c["case_ref"]: {"source": "COMPILED" if c.get("compile_status") == "COMPILED" else c.get("compile_status"),
                            "bends": [None] * int(c.get("planned_bends") or 0)} for c in doc["cases"]}
    return {"rule": "a recipe-guided engine publishes at most one line per recipe bend: min(gold lines, recipe "
                    "bends) true positives per case, no false positive; complete-label line cases only",
            "use": "report-only (owner aim and plan §5 G2(a) shown as targets); the scorer recomputes it from the "
                   "compiled parts",
            "broad": recipe_line_ceiling(doc.get("broad_gold_cases") or [], pops,
                                         target=BROAD_OWNER_TARGET["family_macro_line_f1_min"]),
            "gold11": recipe_line_ceiling(doc.get("gold_cases") or [], pops, target=G2_LINE_THRESHOLDS["f1_min"])}


def cmd_cases(a) -> int:
    from .broad import BROAD_SCHEMA, annotate_recipe_counts, baseline_predictions, broad_gold, broad_rows
    from .cases import benchmark_scan_index, build_cases, draw_devcheck, split_public_sealed
    from .gold import public_gold_transforms, relabel_gold_cases
    from .physical import compile_gate_fields
    from .recipes import FROZEN_VARIANT, public_row

    b = _builder(a)
    fd = Path(a.freeze_dir)
    full_path = fd / GOLD_TRANSFORMS_FULL_NAME
    if not full_path.is_file():
        raise ValueError(f"{full_path.name} is missing: run `gold` first (amendment 6a: recovered before any scoring)")
    inv = read_json(fd / "inventory.json")
    report = read_json(b / "compile_report_full.json")
    gold_all = read_json(b / "gold_cases_ALL.json")["cases"]
    gold11 = [g for g in gold_all if g.get("tier", "GOLD11") == "GOLD11"]
    bm = benchmark_scan_index(read_json(BENCHMARK_MANIFEST))
    case_list = read_json(BROAD_CASE_LIST)
    compile_by_cad = {r["cad_sha256"]: r for r in report["rows"]}
    rows = broad_rows(case_list, compile_by_cad)
    bgold, bexcl = broad_gold(rows, gold_all)
    built = build_cases(inv, report, gold11, broad_rows=rows, benchmark_scans=bm)
    prev_path = Path(a.prev_devcheck) if a.prev_devcheck else b / "prev_pass1" / "freeze" / "devcheck_SEALED.json"
    prev = read_json(prev_path).get("draw") if prev_path.is_file() else None
    dev = draw_devcheck(built["cases"], inv["inventory_sha256"], built["repeatability_sets"], superseded=prev)
    relabel, snap = relabel_gold_cases(gold11)
    # one rule for every gold case: what the frozen DEV recipe allows a recipe-guided engine (flags, report-only)
    flagged = {name: annotate_recipe_counts(g, built["cases"], compile_by_cad)
               for name, g in (("broad_gold", bgold), ("gold11", gold11), ("relabel", relabel))}
    public, sealed = split_public_sealed(built, dev, gold11, broad_gold=bgold, broad_excluded=bexcl,
                                         relabel_cases=relabel, relabel_snapshot=snap)
    sealed_fams = set(dev["families"])
    # recipes: frozen variant only, A1 families public, DEV-CHECK families sealed
    pub_dir, sea_dir = fd / "recipes", fd / "recipes_devcheck_SEALED"
    for d in (pub_dir, sea_dir):
        if d.exists():
            shutil.rmtree(d)
    rows_pub, rows_sea = [], []
    for row in report["rows"]:
        sealed_row = (row.get("family") or row.get("part_id")) in sealed_fams
        (rows_sea if sealed_row else rows_pub).append(public_row(row))
        src = _built_recipe(b, row, FROZEN_VARIANT)
        if src is not None:
            _copy(src, (sea_dir if sealed_row else pub_dir) / f"{row['sku']}.recipe.json")
    drop = ("rows", "work_dir", "gate", "gate_source_assets", "amendment_4d_figure", "policy_figure", "gate_rule",
            "meets_floor", "meets_floor_source_assets", "figures")
    compile_public = {k: v for k, v in report.items() if k not in drop}
    compile_public.update(compile_gate_fields(report.get("rates") or {}, report.get("rates_source_assets") or {}))
    compile_public.update(rows=rows_pub, sealed_rows=len(rows_sea),
                          note="rates are over every authorized unique STEP (DEV-CHECK rows included); the "
                               "DEV-CHECK rows themselves are in devcheck_SEALED.json")
    write_json(fd / "compile_report.json", compile_public)
    # BROAD reference lanes (the broad baseline's own runs), split like everything else
    lanes = baseline_predictions(rows)

    def lane_part(sealed_part: bool) -> dict[str, Any]:
        fams = {r["case_ref"]: r["family"] for r in rows}
        return {"schema": BROAD_SCHEMA + ".reference_lanes", "created_at": lanes["created_at"], "meta": lanes["meta"],
                "lanes": {lane: {k: v for k, v in per.items() if (fams.get(k) in sealed_fams) == sealed_part}
                          for lane, per in lanes["lanes"].items()}}

    pub_lanes, sea_lanes = fd / "broad_reference_lanes_A1.json", fd / "broad_reference_lanes_devcheck_SEALED.json"
    write_json(pub_lanes, lane_part(False))
    write_json(sea_lanes, lane_part(True))
    broad_rows_pub = [r for r in rows if r["family"] not in sealed_fams]
    broad_rows_sea = [r for r in rows if r["family"] in sealed_fams]
    # amendment 6a, A1 side: the A1 part of the full transform list (written before any scoring, sealed)
    write_json(fd / "gold_transforms.json", public_gold_transforms(
        read_json(full_path), sealed_fams, full_sha256=sha256_file(full_path), full_path=rel_to_repo(full_path)))
    public["recipe_line_ceiling"] = _recipe_ceilings(public)
    sealed["recipe_line_ceiling"] = _recipe_ceilings(sealed)
    sealed.update(compile_rows=rows_sea, recipes_dir="docs/production/runlog/freeze/recipes_devcheck_SEALED",
                  legacy_dir="docs/production/runlog/legacy_control/devcheck_SEALED",
                  g3_material_dir="docs/production/runlog/g3_truth_material/devcheck_SEALED",
                  broad_reference_lanes_path="docs/production/runlog/freeze/broad_reference_lanes_devcheck_SEALED.json",
                  gold_transforms_path=rel_to_repo(full_path), broad_rows=broad_rows_sea)
    write_json(fd / "devcheck_SEALED.json", sealed)
    public.update(recipes_dir="docs/production/runlog/freeze/recipes",
                  broad_reference_lanes_path="docs/production/runlog/freeze/broad_reference_lanes_A1.json",
                  gold_transforms_path="docs/production/runlog/freeze/gold_transforms.json", broad_rows=broad_rows_pub)
    write_json(fd / "cases_A1.json", public)
    rules = {
        "g2_complete_A1": ("G2_COMPLETE", "plan §5 G2 complete scans (provisional ones decided by the G3 truth table)"),
        "g3_staged_A1": ("G3_STAGED", "plan §5 G3 staged scans (truth from the frozen visual truth table)"),
        "known_good_A1": ("KNOWN_GOOD", "amendment 4e: ANGLE_TRUTH_CONTROL_SET stable + secondary controls"),
        "repeatability_A1": ("REPEATABILITY", "amendment 6f repeatability members"),
        "broad_A1": ("BROAD", "owner directions 2026-10-17: every CONSUMED scan of case_list_broad_v3.json with CAD "
                              "that compiles (family-macro line-F1 + per-bend checks; report-only)"),
        "broad_staged_A1": ("BROAD_STAGED", "BROAD scans outside the G3 list whose stage hints say partial / stage 0 "
                                            "(per-bend state truth from the frozen visual truth table)"),
    }
    for fname, (lst, rule) in rules.items():
        _id_list(fd, fname, lst, rule, public["lists"].get(lst, []))
    write_json(fd / "gold_g2a_A1.json", {"schema": "sherman_e4b_gold_cases_v2",
                                         "rule": "amendment 6a GoldCase records (raw-frame scoring), human Gold11: "
                                                 "the G2(a) gate of record",
                                         "recipe_ceiling": public["recipe_line_ceiling"]["gold11"],
                                         "cases": public["gold_cases"]})
    pub_ids = {g["case_id"] for g in public["gold_cases"]}
    write_json(fd / "gold_g2a_relabel_A1.json", {
        "schema": "sherman_e4b_gold_cases_v2", "rule": "Gold11 audit relabel (ai_bend_lines_gold11_relabel_v1) as it "
        "stood at the freeze; report-only next to the human Gold11 (scoring re-reads the A1 relabel files listed "
        "here and records whether they changed)",
        "snapshot": {"dir": snap["dir"], "status": snap["status"], "rule": snap["rule"],
                     "files": public["relabel_files"], "other_files": public["relabel_other_files"],
                     "snapshot_sha256_A1": public["relabel_snapshot_sha256"],
                     "scope": public["relabel_snapshot_scope"],
                     "gold11_cases_covered_A1": [c for c in snap["gold11_cases_covered"] if c in pub_ids],
                     "gold11_cases_A1": len(pub_ids)},
        "cases": public["relabel_gold_cases"]})
    write_json(fd / "gold_broad_A1.json", {"schema": "sherman_e4b_gold_cases_v2",
                                           "rule": "BROAD gold (raw frame; human label jobs via the amendment-6a "
                                                   "recovery, AI gold sha-bound to the scan)",
                                           "recipe_ceiling": public["recipe_line_ceiling"]["broad"],
                                           "cases": public["broad_gold_cases"],
                                           "excluded_before_scoring": public["broad_gold_excluded_before_scoring"]})
    _print({"cases_A1": len(public["cases"]), "cases_devcheck": len(sealed["cases"]),
            "families_devcheck": len(dev["families"]), "families_total": dev["families_total"],
            "lists_A1": {k: len(v) for k, v in public["lists"].items()},
            "lists_devcheck": {k: len(v) for k, v in sealed["lists"].items()},
            "broad_rows": {s: sum(1 for r in rows if r["status"] == s) for s in ("BROAD", "EXCLUDED")},
            "broad_gold": {"A1": len(public["broad_gold_cases"]), "devcheck": len(sealed["broad_gold_cases"]),
                           "excluded": len(bexcl)},
            "recipe_count_flagged": flagged,
            "relabel": {"status": snap["status"], "A1": len(public["relabel_gold_cases"]),
                        "devcheck": len(sealed["relabel_gold_cases"]), "A1_files": len(public["relabel_files"])},
            "excluded_scans": len(public["excluded_scans"]) + len(sealed["excluded_scans"])})
    return 0


def _prev_cases(prev_freeze: Path | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if prev_freeze is None:
        return out
    for name in ("cases_A1.json", "devcheck_SEALED.json"):
        p = Path(prev_freeze) / name
        if p.is_file():
            for c in read_json(p).get("cases") or []:
                out[c["case_ref"]] = c
    return out


def _prev_runs(prev_legacy: Path | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if prev_legacy is None:
        return out
    for base in (Path(prev_legacy), Path(prev_legacy) / "devcheck_SEALED"):
        for d in sorted(base.glob("pc_*")):
            if (d / "run.json").is_file():
                out[d.name] = read_json(d / "run.json")
    return out


def cmd_legacy(a) -> int:
    from ..render import plan_closeups, render_closeup, render_global_views
    from ..runner import code_fingerprint
    from .legacy import LEGACY_RSS_LIMIT_BYTES, legacy_render_scene, parse_legacy_report, reusable_run, run_legacy_case

    b = _builder(a)
    fd = Path(a.freeze_dir)
    _not_inside(a.reuse_from, LEGACY_DIR, "--reuse-from")
    legacy_fp = code_fingerprint(REPO_ROOT, ("backend", "domains/bend"))  # the code the worker runs
    prev_cases = _prev_cases(Path(a.prev_freeze) if a.prev_freeze else None)
    prev_runs = _prev_runs(Path(a.reuse_from) if a.reuse_from else None)
    prev_compiled = _prev_compiled(a.prev_compile)
    summaries: dict[str, list[dict[str, Any]]] = {aud: [] for aud in AUDIENCES}
    if not a.case:
        for d in (LEGACY_DIR, LEGACY_DIR / "devcheck_SEALED"):
            for sub in d.glob("pc_*"):
                shutil.rmtree(sub)
            if (d / "SUMMARY.json").is_file():
                (d / "SUMMARY.json").unlink()
    for audience, doc in _case_docs(fd):
        base = _audience_dir(LEGACY_DIR, audience)
        todo = [c for c in doc["cases"] if any(l.split(":")[0] in LEGACY_LISTS for l in c["lists"])
                and (not a.case or c["case_ref"] in a.case)]
        for i, c in enumerate(todo, start=1):
            out = base / c["case_ref"]
            work_case = b / "legacy" / c["case_ref"]
            prev = prev_runs.get(c["case_ref"])
            reuse = reusable_run(prev, c, legacy_fp, prev_case=prev_cases.get(c["case_ref"]), work_case=work_case)
            if reuse["reusable"] and not a.rerun:
                _progress(f"legacy {audience} [{i}/{len(todo)}] {c['case_ref']} REUSED ({reuse['why']})")
                origin = (prev.get("reused") or {}).get("from") or f"E4b legacy run recorded {prev.get('recorded_at')}"
                run = {**prev, "reused": {"from": origin, "why": reuse["why"], "checked_at": now_utc()}}
                run.setdefault("scan_sha256", c["scan_sha256"])
                run.setdefault("cad_sha256", c["cad_sha256"])
                run["work_dir"] = str(work_case)
                run["outputs"] = {k: (str(work_case / "out" / k) if v is not None else None)
                                  for k, v in (prev.get("outputs") or {}).items()}
            else:
                _progress(f"legacy {audience} [{i}/{len(todo)}] {c['case_ref']} RUN"
                          + ("" if prev is None else f" (not reused: {reuse['why']})"))
                run = run_legacy_case(c, scan_path_of(c), cad_path_of(c), b)
                run["legacy_code_fingerprint"] = legacy_fp
            cdir = _compiled_dir(b, c)
            compiled = None
            if (cdir / "compiled_part.json").is_file():
                from ..engine_output import load_compiled

                compiled = load_compiled(cdir / "compiled_part.json")
            parsed = parse_legacy_report(run, compiled, cdir if compiled is not None else None)
            parsed["compiled_part_sha256"] = None if compiled is None else compiled.compiled_part_sha256
            write_json(out / "run.json", {**run, "work_dir": f"<work>/{WORK_BUILDER_DIR}/legacy/" + c["case_ref"]})
            write_json(out / "legacy_case.json", parsed)
            renders: list[dict[str, Any]] = []
            prev_rdir = None
            if a.reuse_from and run.get("reused"):
                for base_prev in (Path(a.reuse_from), Path(a.reuse_from) / "devcheck_SEALED"):
                    cand = base_prev / c["case_ref"] / "renders"
                    if (cand / "render_manifest.json").is_file():
                        prev_rdir = cand
            same_compiled = (compiled is not None and prev_compiled.get(c["recipe_sku"]) is not None
                             and prev_compiled.get(c["recipe_sku"]) == compiled.compiled_part_sha256)
            if compiled is not None and compiled.status == "COMPILED" and not a.no_render and prev_rdir is not None \
                    and same_compiled:
                # a reused run on the same compiled part: the renders are those of the earlier pass (same code)
                rdir = out / "renders"
                if rdir.exists():
                    shutil.rmtree(rdir)
                shutil.copytree(prev_rdir, rdir)
                renders = read_json(rdir / "render_manifest.json")
            elif compiled is not None and compiled.status == "COMPILED" and not a.no_render:
                scene = legacy_render_scene(parsed, run, compiled, cdir, title=f"legacy control {c['case_ref']}")
                rdir = out / "renders"
                if rdir.exists():
                    shutil.rmtree(rdir)
                renders = render_global_views(scene, rdir, show_values=False)
                if scene.points is not None:
                    for plan in plan_closeups(scene):
                        renders.append(render_closeup(scene, plan, rdir / f"closeup_{plan.bend_uid}_{plan.tile_index}.png"))
                for rec in renders:  # palette re-encode (run-log size); the manifest keeps the final sha256
                    rec["sha256"] = compact_png(rdir / rec["file"])
                write_json(rdir / "render_manifest.json", renders)
            cap = int((run.get("conditions") or {}).get("rss_limit_bytes") or 0)
            summaries[audience].append({
                "case_ref": c["case_ref"], "family": c["family"], "lists": c["lists"], "status": run["status"],
                "worker_error": run.get("worker_error"), "wall_s": run["wall_s"], "peak_rss_bytes": run["peak_rss_bytes"],
                "rss_limit_gib": round(cap / 1024**3, 3) if cap else None,
                "alignment_available": bool((parsed.get("alignment") or {}).get("available")),
                "frame_check": (parsed.get("frame_check") or {}).get("status"), "counts": parsed.get("counts"),
                "unary_count_model_applied": parsed.get("unary_count_model_applied"),
                "planned_bends": c.get("planned_bends"), "renders": len(renders),
                "reused": bool(run.get("reused")), "legacy_code_fingerprint": run.get("legacy_code_fingerprint")})
    import numpy as np

    for audience, rows in summaries.items():
        if a.case and not rows:
            continue
        base = _audience_dir(LEGACY_DIR, audience)
        prev = read_json(base / "SUMMARY.json")["cases"] if (a.case and (base / "SUMMARY.json").is_file()) else []
        merged = {r["case_ref"]: r for r in prev}
        merged.update({r["case_ref"]: r for r in rows})
        cases = sorted(merged.values(), key=lambda r: r["case_ref"])
        walls = [r["wall_s"] for r in cases]
        ok_reports = [r for r in cases if r["status"] == "OK"]
        write_json(base / "SUMMARY.json", {
            "schema": "sherman_e4b_legacy_control_summary_v3", "created_at": now_utc(),
            "arm": "LEGACY (backend/bend_analysis_worker.py, unchanged, serial)", "audience": audience,
            "lists": list(LEGACY_LISTS),
            "conditions": {
                "tolerance_angle_deg": 1.0, "tolerance_radius_mm": 0.5, "timeout_s": 600.0,
                "rss_limit_gib": LEGACY_RSS_LIMIT_BYTES / 1024**3,
                "rss_limit_gib_of_runs": {f"{g:g}": n for g, n in sorted(Counter(
                    r.get("rss_limit_gib") for r in cases if r.get("rss_limit_gib") is not None).items())},
                "rss_rule": "the line contract's 12 GiB (SHERMAN_LINE_RSS_LIMIT_MB), under the 16 GiB machine rule; a "
                            "run of an earlier pass that its 4 GiB cap did not kill and that peaked below 12 GiB is "
                            "the run a 12 GiB cap gives (the cap only kills; legacy.reusable_run); a run a lower cap "
                            "killed is re-run at 12 GiB",
                "config": "built-in defaults (config files absent)", "blind": True,
                "unary_count_model": {
                    "source": "<repo>/output/unary_models/latest.json, loaded by the worker itself when present (no "
                              "override; outside this chunk's custody, never opened or stat'ed here)",
                    "recorded_as": "read back from every OK report (structured_context or per-match "
                                   "unary_predictions: legacy.unary_count_model_applied)",
                    "ok_reports": len(ok_reports),
                    "applied_in_reports": sum(1 for r in ok_reports if r.get("unary_count_model_applied"))},
                "legacy_code_fingerprints": sorted({r.get("legacy_code_fingerprint") or "NOT_RECORDED" for r in cases}),
                "legacy_code_fingerprint_roots": ["backend", "domains/bend"]},
            "cases": cases, "runs": len(cases), "reused_runs": sum(1 for r in cases if r.get("reused")),
            "status_counts": {s: sum(1 for r in cases if r["status"] == s) for s in sorted({r["status"] for r in cases})},
            "p95_wall_s": float(np.percentile(walls, 95)) if walls else None,
            "max_peak_rss_bytes": max((r["peak_rss_bytes"] or 0) for r in cases) if cases else None,
            "baseline": _legacy_baseline(fd, audience, b, base)})
    _print({k: len(v) for k, v in summaries.items()})
    return 0


def _legacy_baseline(fd: Path, audience: str, b: Path, base: Path) -> dict[str, Any]:
    """The legacy arm's numbers on the frozen lists of one audience (no engine involved)."""
    from .physical import legacy_baseline, planned_population

    doc = dict(_case_docs(fd))[audience]
    recipes_dir = REPO_ROOT / doc.get("recipes_dir", "docs/production/runlog/freeze/recipes")
    pops = {c["case_ref"]: planned_population(recipes_dir / f"{c['recipe_sku']}.recipe.json", _compiled_dir(b, c))
            for c in doc["cases"]}
    legacy = {c["case_ref"]: read_json(base / c["case_ref"] / "legacy_case.json") for c in doc["cases"]
              if (base / c["case_ref"] / "legacy_case.json").is_file()}
    return legacy_baseline(doc, legacy=legacy, pops=pops)


def _material_group(c: dict[str, Any]) -> tuple[str, bool]:
    if "G3_STAGED" in c["lists"]:
        return "G3_STAGED", True
    if "BROAD_STAGED" in c["lists"]:
        return "BROAD_STAGED", True
    if c.get("provisional_complete"):
        return "PROVISIONAL_COMPLETE", True
    return "STATE_CHECK", False


def cmd_g3_material(a) -> int:
    from .g3_material import MATERIAL_VERSION, REVIEWER_INSTRUCTIONS, render_case_material, truth_table_template

    b = _builder(a)
    fd = Path(a.freeze_dir)
    _not_inside(a.reuse_from, G3_DIR, "--reuse-from")
    blind_root = b / "g3_reviewer_blind"  # display placements: builder material, never in the reviewer tree
    _not_inside(a.reuse_blind_from, blind_root, "--reuse-blind-from")
    prev_entries: dict[str, dict[str, Any]] = {}
    if a.reuse_from:
        for audience in AUDIENCES:
            base = _audience_dir(Path(a.reuse_from), audience)
            if not (base / "MANIFEST.json").is_file():
                continue
            man = read_json(base / "MANIFEST.json")
            for e in man["cases"]:
                blind = [base / "_reviewer_blind" / f"{e['case_ref']}.registration.json"]  # the first passes' layout
                if a.reuse_blind_from:
                    blind.append(Path(a.reuse_blind_from) / audience / f"{e['case_ref']}.registration.json")
                prev_entries[e["case_ref"]] = {**e, "_dir": str(base / e["case_ref"]), "_blind": [str(p) for p in blind],
                                               "_manifest_created_at": man.get("created_at")}
    prev_compiled = _prev_compiled(a.prev_compile)
    if not a.case:
        for d in (G3_DIR, G3_DIR / "devcheck_SEALED", blind_root):
            if d.exists():
                for sub in list(d.iterdir()):
                    if sub.is_dir():
                        shutil.rmtree(sub)
                    else:
                        sub.unlink()
    for audience, doc in _case_docs(fd):
        base = _audience_dir(G3_DIR, audience)
        blind_dir = blind_root / audience
        todo = [c for c in doc["cases"] if any(l.split(":")[0] in G3_MATERIAL_LISTS for l in c["lists"])
                and (not a.case or c["case_ref"] in a.case)]
        entries = []
        prev = {}
        if a.case and (base / "MANIFEST.json").is_file():
            prev = {e["case_ref"]: e for e in read_json(base / "MANIFEST.json")["cases"]}
        for i, c in enumerate(todo, start=1):
            out = base / c["case_ref"]
            group, sections = _material_group(c)
            pe = prev_entries.get(c["case_ref"])
            reuse = (pe is not None and pe.get("scan_sha256") == c["scan_sha256"]
                     and pe.get("recipe_sku") == c["recipe_sku"] and pe.get("material_version") == MATERIAL_VERSION
                     and bool(pe.get("sections")) == sections and c.get("compiled_part_sha256") is not None
                     and prev_compiled.get(c["recipe_sku"]) == c.get("compiled_part_sha256")
                     and Path(pe["_dir"]).is_dir())
            if out.exists():
                shutil.rmtree(out)
            if reuse and not a.rerender:
                _progress(f"g3 material {audience} [{i}/{len(todo)}] {c['case_ref']} REUSED")
                shutil.copytree(pe["_dir"], out)
                src_blind = next((Path(p) for p in pe["_blind"] if Path(p).is_file()), None)
                if src_blind is not None:
                    _copy(src_blind, blind_dir / f"{c['case_ref']}.registration.json")
                e = {k: v for k, v in pe.items() if not k.startswith("_")}
                e["reused_from_manifest"] = pe.get("_manifest_created_at")
            else:
                _progress(f"g3 material {audience} [{i}/{len(todo)}] {c['case_ref']} RENDER")
                e = render_case_material(c, scan_path_of(c), _compiled_dir(b, c), out, blind_dir, sections=sections)
            e["group"] = group
            e["compiled_part_sha256"] = c.get("compiled_part_sha256")
            entries.append(e)
        prev.update({e["case_ref"]: e for e in entries})
        allent = sorted(prev.values(), key=lambda e: e["case_ref"])
        if not allent:
            continue
        write_json(base / "MANIFEST.json", {"schema": "sherman_e4b_g3_material_v2", "created_at": now_utc(),
                                            "rule": "plan §10 amendment 7: built before any engine output; "
                                                    "no engine, legacy or stage-key information is drawn",
                                            "cases": allent})
        write_json(base / "truth_table_TEMPLATE.json", truth_table_template(allent))
        (base / "REVIEWER_INSTRUCTIONS.md").write_text(REVIEWER_INSTRUCTIONS, encoding="utf-8")
    _print({"done": True})
    return 0


def cmd_ledger(a) -> int:
    from .ledger import verify_freeze, write_ledger

    out = write_ledger(Path(a.freeze_dir), replace_before_a1=a.replace_before_a1)
    ver = verify_freeze(Path(a.freeze_dir))
    _print({**out, "verify_ok": ver["ok"], "mismatches": ver["check_against_ledger"]["mismatches"],
            "file_drift": ver["file_drift"]})
    return 0 if ver["ok"] else 2


def cmd_verify(a) -> int:
    from .ledger import verify_freeze

    ver = verify_freeze(Path(a.freeze_dir), include_sealed=a.include_sealed)
    _print(ver)
    return 0 if ver["ok"] else 2


# --------------------------------------------------------------------------------------------
# A1-facing commands
# --------------------------------------------------------------------------------------------


def cmd_compile_frozen(a) -> int:
    """Recompile the frozen recipes of the A1 audience (``--include-sealed``: also DEV-CHECK's, into
    ``W/devcheck_SEALED``) and check every compiled-part sha256 against the freeze."""
    from ..runner import RunnerConfig, engine_env
    from .common import dev_sku_for
    from .inventory import all_entries, entry_path
    from .recipes import compile_recipe

    fd = Path(a.freeze_dir)
    plans = [("A1", fd / "cases_A1.json", fd / "recipes", False)]
    if a.include_sealed:
        plans.append(("DEVCHECK", fd / "devcheck_SEALED.json", fd / "recipes_devcheck_SEALED", True))
    roots = {aud: _eval_root(a, sealed=sealed) for aud, _, _, sealed in plans}
    inv = read_json(fd / "inventory.json")
    by_sku = {dev_sku_for(e["sha256"]): e for e in all_entries(inv, "STEP")}
    public_rows = read_json(fd / "compile_report.json").get("rows") or []
    rows = []
    for audience, cases_path, recipe_dir, sealed in plans:
        root = roots[audience]
        doc = read_json(cases_path)
        expected: dict[str, str | None] = {}
        for r in (doc.get("compile_rows") if sealed else public_rows) or []:
            expected[r["sku"]] = ((r.get("variants") or {}).get("dev") or {}).get("compiled_part_sha256")
        for c in doc["cases"]:
            if expected.get(c["recipe_sku"]) is None:
                expected[c["recipe_sku"]] = c.get("compiled_part_sha256")
        env = engine_env(RunnerConfig(tier_dir=root, out_dir=root))
        for rp in sorted(recipe_dir.glob("*.recipe.json")):
            sku = rp.name.split(".")[0]
            e = by_sku.get(sku)
            if e is None:
                rows.append({"audience": audience, "sku": sku, "status": "NO_STEP"})
                continue
            staged = root / "inputs" / f"cad_{e['sha256'][:12]}.step"
            if not staged.is_file() or sha256_file(staged) != e["sha256"]:
                staged.parent.mkdir(parents=True, exist_ok=True)
                if staged.exists():
                    staged.unlink()
                shutil.copyfile(custody_check(entry_path(e), "frozen recipe STEP"), staged)
            rec = compile_recipe(rp, staged, _compiled_dir(root, {"recipe_sku": sku}), env=env)
            exp = expected.get(sku)
            rows.append({"audience": audience, "sku": sku, "status": rec["status"],
                         "compiled_part_sha256": rec.get("compiled_part_sha256"),
                         "matches_freeze": None if exp is None else rec.get("compiled_part_sha256") == exp})
    bad = [r for r in rows if r.get("matches_freeze") is False]
    _print({"audiences": [p[0] for p in plans],
            "recompiled": {aud: sum(1 for r in rows if r["audience"] == aud) for aud, *_ in plans},
            "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
            "matches_freeze": sum(1 for r in rows if r.get("matches_freeze")),
            "not_recorded_at_freeze": sum(1 for r in rows if r.get("matches_freeze") is None),
            "mismatches": bad})
    return 0 if not bad else 2


def cmd_run_physical(a) -> int:
    from .physical import run_e3_case, truth_relevant, truth_table_gate

    doc, sealed = _case_doc(a)
    root = _eval_root(a, sealed=sealed)
    recipes_dir = REPO_ROOT / doc.get("recipes_dir", "docs/production/runlog/freeze/recipes")
    todo = [c for c in doc["cases"] if (not a.case or c["case_ref"] in a.case)
            and (not a.list or any(l.split(":")[0] in a.list for l in c["lists"]))]
    g3_base = _audience_dir(G3_DIR, "DEVCHECK" if sealed else "A1")
    gated = [c for c in todo if truth_relevant(c)]
    refused: list[str] = []
    gate: dict[str, Any] = {"ok": True, "reason": None}
    if gated:
        template = g3_base / "truth_table_TEMPLATE.json"
        if not template.is_file():
            raise ValueError(f"{template} is missing: the frozen truth-table template names the rows that must be "
                             "frozen before E3 runs on a truth-relevant case (amendment 7)")
        rows_by_case: dict[str, list[str]] = {}
        for r in read_json(template).get("rows") or []:
            rows_by_case.setdefault(str(r["case_ref"]), []).append(str(r["bend_uid"]))
        table = a.truth_table if a.truth_table is not None else _frozen_table(g3_base)
        gate = truth_table_gate(table, {c["case_ref"]: rows_by_case.get(c["case_ref"], []) for c in gated},
                                ledger=a.ledger)
        if not gate["ok"]:
            refused = sorted(c["case_ref"] for c in gated if rows_by_case.get(c["case_ref"]))
    run_list = [c for c in todo if c["case_ref"] not in set(refused)]
    out = []
    for i, c in enumerate(run_list, start=1):
        _progress(f"E3 [{i}/{len(run_list)}] {c['case_ref']}")
        rec = run_e3_case(c, scan_path_of(c), recipes_dir / f"{c['recipe_sku']}.recipe.json", _compiled_dir(root, c),
                          root, engine_config=a.engine_config)
        out.append({"case_ref": c["case_ref"], "status": rec["status"], "wall_s": rec["wall_s"]})
    report: dict[str, Any] = {"runs": out}
    if refused:
        report["refused_by_amendment7_gate"] = {
            "cases": refused, "reason": gate["reason"],
            "rule": "plan §10 amendment 7: the visual truth table of a truth-relevant case (G3 / BROAD staged, "
                    "provisional-complete G2) is frozen and recorded in the ledger before any engine output on it "
                    "exists (prefreeze record-truth-table)"}
    _print(report)
    return 2 if refused else 0


def cmd_score_physical(a) -> int:
    from .physical import score_physical

    doc, sealed = _case_doc(a)
    root = _eval_root(a, sealed=sealed)
    recipes_dir = REPO_ROOT / doc.get("recipes_dir", "docs/production/runlog/freeze/recipes")
    report_path = Path(a.freeze_dir) / "compile_report.json"
    g3_base = _audience_dir(G3_DIR, "DEVCHECK" if sealed else "A1")
    table = a.truth_table if a.truth_table is not None else _frozen_table(g3_base)
    card = score_physical(doc, e3_dir=root / "physical" / "E3",
                          legacy_dir=_audience_dir(LEGACY_DIR, "DEVCHECK" if sealed else "A1"),
                          recipes_dir=recipes_dir, compiled_root=root / "compiled" / "dev",
                          compile_report=read_json(report_path) if report_path.is_file() else None,
                          truth_table=table, ledger=a.ledger, ledger_anchor=a.ledger_anchor)
    write_json(a.out, card)
    _print({"verdict": card["verdict"], "final": card["final"], "qualifying": card["qualifying"],
            "G2": card["G2"]["gates"], "G3": card["G3"].get("gates", card["G3"].get("verdict")),
            "truth_table": card["truth_table"].get("status"),
            "BROAD_family_macro_f1": ((card.get("BROAD") or {}).get("line_metric") or {}).get("engine", {}).get(
                "family_macro_f1")})
    return 0


def cmd_record_truth_table(a) -> int:
    """Validate the frozen visual truth table(s) against the templates and record them in one FREEZE
    entry of the ledger: the A1 table by path and sha256, the DEV-CHECK table (``--include-sealed``,
    final DEV-CHECK run / R1) by sha256 only."""
    from datetime import datetime, timezone

    from .. import freeze as e4a_ledger
    from .physical import _ts, load_truth_table

    plans = [("A1", Path(a.truth_table) if a.truth_table else G3_DIR / "truth_table_FROZEN.json",
              G3_DIR / "truth_table_TEMPLATE.json")]
    if a.include_sealed:
        plans.append(("DEVCHECK", Path(a.sealed_truth_table) if a.sealed_truth_table
                      else G3_DIR / "devcheck_SEALED" / "truth_table_FROZEN.json",
                      G3_DIR / "devcheck_SEALED" / "truth_table_TEMPLATE.json"))
    items: dict[str, Any] = {"files": {}, "devcheck_sealed": {}}
    now = datetime.now(timezone.utc)
    rows_a1 = None
    for audience, table, template in plans:
        if not table.is_file():
            raise ValueError(f"{audience} truth table {table} does not exist")
        truth = load_truth_table(table)  # refuses unfrozen tables, unknown labels, a wrong final, duplicates
        frozen = _ts(read_json(table).get("frozen_at"))
        if frozen is None or frozen > now:
            raise ValueError(f"{audience} truth table: frozen_at is unparseable or in the future")
        want = {(str(r["case_ref"]), str(r["bend_uid"])) for r in read_json(template).get("rows") or []}
        missing, extra = len(want - set(truth)), len(set(truth) - want)
        if missing or extra:
            raise ValueError(f"{audience} truth table: {missing} template row(s) missing, {extra} row(s) not in the "
                             "template")
        if audience == "A1":
            rows_a1 = len(truth)
            items["files"]["g3_truth_table_A1"] = {"path": str(table), "sha256": sha256_file(table),
                                                   "bytes": table.stat().st_size}
            items["files"]["g3_truth_table_template_A1"] = {"path": str(template), "sha256": sha256_file(template),
                                                            "bytes": template.stat().st_size}
        else:
            items["devcheck_sealed"]["g3_truth_table_SEALED"] = {"sha256": sha256_file(table)}
            items["devcheck_sealed"]["g3_truth_table_template_SEALED"] = {"sha256": sha256_file(template)}
    if not items["devcheck_sealed"]:
        del items["devcheck_sealed"]
    items["g3_truth_table_rule"] = {
        "rule": "plan §10 amendment 7: two independent reviewers, a tie-break, then frozen, before any engine output "
                "is seen; recorded here before the first E3 run on a truth-relevant case",
        "enforced_by": "prefreeze run-physical (refuses truth-relevant cases until this entry exists) and the scorer "
                       "(rejects a table frozen or recorded after the first E3 record of such a case)"}
    entry = e4a_ledger.append_entry(Path(a.ledger), label=a.label, items=items, kind="FREEZE",
                                    recorded_by="e4b-record-truth-table")
    _print({"entry_id": entry["entry_id"], "recorded_at": entry["recorded_at"], "seq": entry["seq"],
            "a1_rows": rows_a1, "devcheck_recorded": "devcheck_sealed" in items})
    return 0


def cmd_build_recipes(argv: list[str]) -> int:
    from .recipes import main_build

    return main_build(argv)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m domains.production.engine.eval.freeze.prefreeze",
                                 description=f"E4b pre-A1 freeze tools ({FREEZE_TOOLS_VERSION})")
    ap.add_argument("--freeze-dir", default=str(FREEZE_DIR), help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn, help_: str, work: bool = True):
        p = sub.add_parser(name, help=help_)
        if work:
            p.add_argument("--work", type=Path, help="work directory outside the repository (large artifacts)")
        p.set_defaults(fn=fn)
        return p

    def sealed_flag(p, help_: str = "also the DEV-CHECK material (the final DEV-CHECK run and R1 only, never A1)"):
        p.add_argument("--include-sealed", action="store_true", help=help_)

    add("inventory", cmd_inventory, "hash the authorized inputs", work=False)
    gd = add("gold", cmd_gold, "Gold transform recovery (before any scoring)")
    gd.add_argument("--prior", action="append", default=[], help="earlier transform list to reuse where valid")
    r = add("recipes", cmd_recipes, "DEV_AUTO_CONFIRMED recipes + E1 compiles")
    r.add_argument("--only", action="append", default=[], help="only this STEP sha256 (repeatable)")
    cs = add("cases", cmd_cases, "case lists, BROAD list, DEV-CHECK draw, public/sealed split")
    cs.add_argument("--prev-devcheck", type=Path, help="the superseded draw's sealed file (recorded, never used; "
                                                       "default W/builder_SEALED/prev_pass1/freeze/devcheck_SEALED.json)")
    lg = add("legacy", cmd_legacy, "legacy control arm (serial) + G4 renders")
    lg.add_argument("--case", action="append", default=[])
    lg.add_argument("--no-render", action="store_true")
    lg.add_argument("--rerun", action="store_true", help="never reuse an earlier run")
    lg.add_argument("--reuse-from", type=Path, help="a copy of an earlier legacy_control directory (run records)")
    lg.add_argument("--prev-freeze", type=Path, help="earlier freeze directory (case records of those runs)")
    lg.add_argument("--prev-compile", type=Path, help="earlier full compile report (renders of a reused run on an "
                                                      "unchanged compiled part are copied, not redrawn)")
    g3 = add("g3-material", cmd_g3_material, "G3 truth-table material (no engine output)")
    g3.add_argument("--case", action="append", default=[])
    g3.add_argument("--reuse-from", type=Path, help="a copy of an earlier g3_truth_material directory")
    g3.add_argument("--reuse-blind-from", type=Path, help="a copy of an earlier W/builder_SEALED/g3_reviewer_blind")
    g3.add_argument("--prev-compile", type=Path, help="earlier full compile report (compiled-part sha256)")
    g3.add_argument("--rerender", action="store_true")
    led = add("ledger", cmd_ledger, "append the FREEZE entry and write the anchor", work=False)
    led.add_argument("--replace-before-a1", action="store_true",
                     help="replace an existing ledger that holds no RUN/WAIVER entry")
    vf = add("verify", cmd_verify, "verify the ledger, anchor, frozen code and recorded files", work=False)
    sealed_flag(vf, "also re-hash the DEV-CHECK material (final DEV-CHECK run / R1 only, never A1)")
    cf = add("compile-frozen", cmd_compile_frozen, "recompile the frozen A1 recipes and check their sha256")
    sealed_flag(cf, "also the DEV-CHECK recipes, into W/devcheck_SEALED (final DEV-CHECK run / R1 only)")
    rp = add("run-physical", cmd_run_physical, "run E3 on a frozen case document (A1 / final DEV-CHECK)")
    rp.add_argument("--cases", required=True, type=Path)
    rp.add_argument("--case", action="append", default=[])
    rp.add_argument("--list", action="append", default=[], help="only cases in this list (e.g. G2_COMPLETE, BROAD)")
    rp.add_argument("--engine-config", type=Path)
    rp.add_argument("--truth-table", type=Path, help="the frozen visual truth table (default: truth_table_FROZEN.json "
                                                     "next to the template, if it exists)")
    rp.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    sealed_flag(rp)
    sp = add("score-physical", cmd_score_physical, "score G2, G3 and the BROAD report on a frozen case document")
    sp.add_argument("--cases", required=True, type=Path)
    sp.add_argument("--truth-table", type=Path, help="the frozen visual truth table (default: truth_table_FROZEN.json "
                                                     "next to the template, if it exists)")
    sp.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    sp.add_argument("--ledger-anchor", help="the first line of ANCHOR.txt, recorded outside the freeze directory")
    sp.add_argument("--out", required=True, type=Path)
    sealed_flag(sp)
    tt = add("record-truth-table", cmd_record_truth_table,
             "validate the frozen visual truth table(s) and record them in the ledger", work=False)
    tt.add_argument("--truth-table", type=Path, help="the A1 table (default g3_truth_material/truth_table_FROZEN.json)")
    tt.add_argument("--sealed-truth-table", type=Path, help="the DEV-CHECK table (default "
                                                            "g3_truth_material/devcheck_SEALED/truth_table_FROZEN.json)")
    tt.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    tt.add_argument("--label", default="G3 visual truth table (amendment 7)")
    sealed_flag(tt, "also the DEV-CHECK table (final DEV-CHECK run / R1 only)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "_build-recipes":
        return cmd_build_recipes(args[1:])
    ap = build_parser()
    a = ap.parse_args(args)
    try:
        return int(a.fn(a))
    except (CustodyError, OSError, ValueError, KeyError, PermissionError, FileExistsError) as exc:
        sys.stderr.write(f"error: {type(exc).__name__}: {exc}\n")
        return 2
    except Exception:  # internal error
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
