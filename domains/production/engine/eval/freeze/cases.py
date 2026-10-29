"""Frozen physical case lists and the hidden DEV-CHECK subset (plan §5 G2/G3, §10 amendments 4e, 4f, 6f, 7;
owner directions 2026-10-17 for the BROAD list).

**Physical cases.** One case per unique scan (sha256). Two sources:

1. the scans of the authorized source tree (:mod:`.inventory`) whose part family
   has an authorized STEP (source tree first, else an external CONSUMED STEP);
2. the BROAD rows of :mod:`.broad` (every CONSUMED scan of
   ``case_list_broad_v3.json`` whose STEP compiles), including scans outside the
   source tree (the Gold11 38172000 X/V scans, the e-mail / download scans).

Excluded scans are listed with a reason: ``NO_PART_NUMBER`` (``part 1.ply``),
``NO_AUTHORIZED_STEP`` (49125000: Parasolid only),
``RESERVED_FRESH_SCAN_OF_CONSUMED_PART`` (``BENCHMARK_V2.json`` eval tier of
17139000_01: kept for a fresh evaluation; the source-tree inventory hashed it and
read its PLY header like every source file, nothing else was read or used),
``BENCHMARK_ROLE_FORMAT_ALTERNATE`` (the manifest's role of 37126000_D.stl, a
CAD-recipe mesh), format alternates in no list, and the BROAD exclusions of
:mod:`.broad`; byte-identical copies are aliases of one case.

**Lists** (plan text, frozen here, matched on the canonical path or an alias):

- ``G3_STAGED`` -- plan §5 G3: 10839000 full + partials, 23553000_B FULL +
  SCAN_1/2, 28870000_A FULL + SCAN_1, 38172000 SCAN_1/2, 40500000 SCAN1/2 and the
  11979003 BEFORE BENDING blank (stage 0).
- ``G2_COMPLETE`` -- every other source-tree case plus the G3 full scans and the
  two Gold11 38172000 X/V scans (complete parts, defective X and good V); of a
  format pair (same stem as PLY and STL) only the PLY counts, the STL is
  ``FORMAT_ALTERNATE`` (still a repeatability member). A case whose corpus stage
  keys disagree with "complete" is ``provisional_complete``: it stays in G2 only
  if the frozen visual truth table (amendment 7) marks every planned bend FORMED;
  otherwise the scorer moves it to the staged population. The engine never sees
  the stage keys.
- ``KNOWN_GOOD`` -- amendment 4e: the stable (15651004, 10839000, 37361005) and
  secondary (11979003, 48991006, 48963008) controls of
  ``docs/research/ANGLE_TRUTH_CONTROL_SET_20260404.md``; their G2 cases.
- ``REPEATABILITY`` -- amendment 6f: 15651004 x3, 42288005 x2, 47959001 PLY + STL.
  A set whose members are byte-identical copies is ``NOT_EVALUABLE``; 47959001
  is a format pair (one scan exported twice), recorded as such.
- ``GOLD_G2A`` -- the recovered Gold11 cases (:mod:`.gold`), human references
  (the relabel is reported next to them, never instead of them).
- ``BROAD`` -- every BROAD row (:mod:`.broad`): family-macro line-F1 on its gold
  and per-bend checks; ``BROAD_STAGED`` -- the BROAD scans outside the G3 list
  whose stage hints say partial (e.g. a SCAN_1 / SCAN_2 pair) or stage 0: their per-bend
  state truth comes from the same visual truth table.

**Stage keys** (scoring only, never an engine input): per case, the
``data/bend_corpus/parts/<family>/metadata.json`` entries for the scan file and
the expected total bend count. ``expected_completed`` = ``completed_bends`` where
present; the blank is stage 0 by the plan.

**DEV-CHECK** (amendment 4f: "about a third of the part families, including one
staged family. A1 never sees it; it runs once at the end, and 'gates met'
requires G2 and G3 to hold on it as well"), rule ``E4B-DEVCHECK-V4``: the
families of every listed case (G2, G3, repeatability, BROAD) except those the
plan uses outside the accuracy evaluation (10839000: the real dev part of the
app E2E, §6 / amendment 10, because DEV-CHECK must run exactly once) are
stratified, in order, into GOLD (families with a Gold11 G2(a) case), STAGED (the
G3 families and the BROAD staged families that are not 6f repeatability
families), REPEATABILITY (families with an evaluable 6f set) and OTHER; from each
stratum ``round(n/3)`` families are drawn (at least one GOLD and one STAGED) with
a generator seeded by ``sha256("E4B-DEVCHECK-V4|" + inventory sha256 + "|" +
canonical strata)``: a draw fixed by inputs that existed before it, which anyone
can re-derive. Superseded before the freeze, with no E3 run on any physical case
(the legacy control arm had run in the first pass; their draws are recorded in
the sealed file only): rule V2 (the first pass, 18 source-tree families) because
the owner's wave-4 directions added the BROAD families; rule V3 (V2's strata over
the widened population) because its draw held no Gold11 family, which would leave
G2(a) NOT_EVALUABLE on DEV-CHECK and make amendment 4f unsatisfiable. Every case,
Gold case, Gold transform row, relabel file entry, recipe, compiled part, legacy
output and G3 truth material of a DEV-CHECK family is written only to ``*SEALED*``
locations (run logs and work directories alike). The A1 lists name only the
other families. Identities are inferable by omission from the public tree and
from ``inventory.json`` (which hashes every authorized file, A1 and DEV-CHECK
alike, and seeds the draw); the protection is procedural: A1 runs only the A1
lists, in a work directory of its own, and never reads a ``*SEALED*`` path.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    REPO_ROOT,
    SOURCE_ASSETS,
    case_ref_for,
    custody_check,
    dev_sku_for,
    doc_sha256,
    now_utc,
    read_json,
)

__all__ = [
    "CASES_SCHEMA",
    "DEVCHECK_SCHEMA",
    "DEVCHECK_SEED_LABEL",
    "G3_PLAN_STAGED",
    "BLANK_STAGE0",
    "REPEATABILITY_PLAN",
    "KNOWN_GOOD",
    "stage_keys_for",
    "benchmark_scan_index",
    "build_cases",
    "draw_devcheck",
    "split_public_sealed",
    "DEVCHECK_INELIGIBLE",
]

CASES_SCHEMA = "sherman_e4b_physical_cases_v2"
DEVCHECK_SCHEMA = "sherman_e4b_devcheck_sealed_v2"
DEVCHECK_SEED_LABEL = "E4B-DEVCHECK-V4"
#: Plan uses outside the accuracy evaluation (never DEV-CHECK): the app E2E real part (§6, amendment 10).
DEVCHECK_INELIGIBLE: dict[str, str] = {"10839000": "app E2E real dev part (plan §6, amendment 10)"}

#: Plan §5 G3, by file name in ../source-assets/02_CAD_QC_3D (canonical path or alias).
G3_PLAN_STAGED: dict[str, tuple[str, ...]] = {
    "10839000": ("10839000_01_full_scan.ply", "10839000/10839000_01_scan_1.ply", "10839000/10839000_01_scan_2.ply"),
    "23553000": ("23553000_B_FULL_SCAN 1.ply", "23553000_B_SCAN_1.ply", "23553000_B_SCAN_2.ply"),
    "28870000": ("28870000_A_FULL_SCAN.ply", "28870000_A_SCAN_1.ply"),
    "38172000": ("38172000_01_SCAN_1.ply", "38172000_01_SCAN_2 1.ply"),
    "40500000": ("40500000_0_SCAN1 1.ply", "40500000_0_SCAN2 1.ply"),
    "11979003": ("BEFORE BENDING  11979003_C.ply",),
}
BLANK_STAGE0 = "BEFORE BENDING  11979003_C.ply"
#: G3 scans that are the fully formed end of their staged sequence (also G2 complete).
G3_FULL = frozenset({"10839000_01_full_scan.ply", "23553000_B_FULL_SCAN 1.ply", "28870000_A_FULL_SCAN.ply"})
#: Amendment 6f.
REPEATABILITY_PLAN: dict[str, tuple[str, ...]] = {
    "15651004": ("15651004_01.ply", "15651004_01_SECOND_SCAN.stl", "15651004_01_3RD_SCAN.stl"),
    "42288005": ("42288005_01_FULL_SCAN.ply", "42288005_01_FULL_SCAN (1).ply"),
    "47959001": ("47959001_01PREL_scan1.ply", "47959001_01PREL_scan1.stl"),
}
#: Amendment 4e: ANGLE_TRUTH_CONTROL_SET_20260404 stable + secondary controls.
KNOWN_GOOD: dict[str, tuple[str, ...]] = {"stable": ("15651004", "10839000", "37361005"),
                                          "secondary": ("11979003", "48991006", "48963008")}
_PARTIAL_HINT = "PARTIAL_SCAN"
_BLANK_HINT = "STAGE0_BLANK"


def _names(entry: Mapping[str, Any]) -> list[str]:
    return [entry["rel_path"], *entry.get("aliases", [])]


def _match(entry: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for n in _names(entry):
        if n in names:
            return n
    return None


def stage_keys_for(family: str, file_names: Sequence[str], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Scoring-only stage keys of one scan from ``data/bend_corpus/parts/<family>/metadata.json``."""
    path = repo_root / "data" / "bend_corpus" / "parts" / family / "metadata.json"
    if not path.is_file():
        return {"source": None, "expected_completed": None, "note": "no corpus metadata for this family"}
    meta = read_json(custody_check(path, "corpus stage keys"))
    base = {Path(n).name for n in file_names}
    ut = meta.get("user_truth") or {}
    mt = meta.get("mail_truth") or {}
    rows = [{k: s.get(k) for k in ("name", "state", "kind", "sequence", "sequence_confident")}
            for s in meta.get("scan_files") or [] if s.get("name") in base]
    stages = {k: v for k, v in (ut.get("scan_stages") or {}).items() if k in base}
    truth = {k: {kk: v.get(kk) for kk in ("stage", "completed_bends")} for k, v in (ut.get("scan_truth") or {}).items()
             if k in base}
    completed = next((t["completed_bends"] for t in truth.values() if t.get("completed_bends") is not None), None)
    expected_total = ut.get("expected_total_bends") or ut.get("expected_total_discrete_bends") \
        or meta.get("expected_bends_override") or mt.get("expected_total_bends")
    return {"source": path.relative_to(repo_root).as_posix(), "scan_files": rows, "scan_stages": stages,
            "scan_truth": truth, "expected_completed": completed, "expected_total_bends": expected_total,
            "use": "scoring only (amendment 7); never an engine input"}


def benchmark_scan_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``scan sha256 -> {part_id, split, scan_id, role, eval_tier, primary_truth}`` of ``BENCHMARK_V2.json``."""
    out: dict[str, dict[str, Any]] = {}
    for pid, part in (manifest.get("parts") or {}).items():
        for s in part.get("scans") or []:
            out.setdefault(s["sha256"], {"part_id": pid, "split": part.get("split"), "scan_id": s.get("scan_id"),
                                         "role": s.get("role"), "eval_tier": s.get("eval_tier"),
                                         "primary_truth": s.get("primary_truth")})
    return out


def _stage_from_hints(hints: Sequence[str]) -> str:
    if any(str(h).startswith(_BLANK_HINT) for h in hints):
        return "BLANK_STAGE0"
    if any(str(h).startswith(_PARTIAL_HINT) for h in hints):
        return "STAGED_PARTIAL"
    return "COMPLETE"


def _step_order(s: Mapping[str, Any]) -> tuple[bool, int, str]:
    return (s.get("origin", "SOURCE_ASSETS") != "SOURCE_ASSETS", len(s["rel_path"]), s["rel_path"])


def build_cases(inventory: Mapping[str, Any], compile_report: Mapping[str, Any], gold_cases: Sequence[Mapping[str, Any]],
                *, broad_rows: Sequence[Mapping[str, Any]] = (),
                benchmark_scans: Mapping[str, Mapping[str, Any]] | None = None,
                repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Every physical case with its list memberships, plus the excluded scans (module docstring)."""
    from .inventory import all_entries

    steps_by_family: dict[str, list[dict[str, Any]]] = {}
    step_by_sha: dict[str, dict[str, Any]] = {}
    for e in all_entries(inventory, "STEP"):
        step_by_sha[e["sha256"]] = e
        if e.get("family"):
            steps_by_family.setdefault(e["family"], []).append(e)
    compile_rows = {r["cad_sha256"]: r for r in compile_report.get("rows", [])}
    gold_by_scan = {g["scan_sha256"]: g for g in gold_cases if g.get("tier", "GOLD11") == "GOLD11"}
    broad_by_scan = {r["scan_sha256"]: r for r in broad_rows}
    staged_names = {n for names in G3_PLAN_STAGED.values() for n in names}
    repeat_names = {n for names in REPEATABILITY_PLAN.values() for n in names}
    cases: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    scans = [e for e in inventory["entries"] if e["kind"] == "SCAN"]
    stems: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for e in scans:
        stems.setdefault((e.get("family") or "", str(Path(e["rel_path"]).with_suffix(""))), []).append(e)

    def recipe_fields(step: Mapping[str, Any]) -> dict[str, Any]:
        row = compile_rows.get(step["sha256"]) or {}
        dev = (row.get("variants") or {}).get("dev") or {}
        origin = step.get("origin", "SOURCE_ASSETS")
        return {"cad_sha256": step["sha256"],
                "cad_rel": ("../source-assets/02_CAD_QC_3D/" + step["rel_path"] if origin == "SOURCE_ASSETS"
                            else step["rel_path"]),
                "cad_path": str(SOURCE_ASSETS / step["rel_path"]) if origin == "SOURCE_ASSETS" else step["abs_path"],
                "cad_origin": origin, "recipe_sku": dev_sku_for(step["sha256"]),
                "compile_status": dev.get("status", "NO_RECIPE"),
                "compiled_part_sha256": dev.get("compiled_part_sha256"), "recipe_sha256": dev.get("recipe_sha256"),
                "planned_bends": dev.get("bends")}

    def broad_fields(r: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if r is None or r.get("status") != "BROAD":
            return None
        return {k: r.get(k) for k in ("broad_case_id", "tier", "stage_hints", "staged_or_incomplete", "flags",
                                      "label_complete", "zero_bend_control", "in_gold11", "gold11_id")}

    seen: set[str] = set()
    for e in scans:
        fam = e.get("family")
        seen.add(e["sha256"])
        base = {"scan_sha256": e["sha256"], "scan_rel": "../source-assets/02_CAD_QC_3D/" + e["rel_path"],
                "scan_path": str(SOURCE_ASSETS / e["rel_path"]), "aliases": e.get("aliases", []), "family": fam}
        bm = (benchmark_scans or {}).get(e["sha256"]) or {}
        if bm.get("eval_tier") == "FRESH_SCAN_OF_CONSUMED_PART":
            excluded.append({**base, "reason": "RESERVED_FRESH_SCAN_OF_CONSUMED_PART (BENCHMARK_V2 eval tier; "
                                               "only hashed and its header read by the inventory, never used)"})
            continue
        if not fam:
            excluded.append({**base, "reason": "NO_PART_NUMBER"})
            continue
        steps = sorted(steps_by_family.get(fam, []), key=_step_order)
        if not steps:
            excluded.append({**base, "reason": "NO_AUTHORIZED_STEP"})
            continue
        pair = stems.get((fam, str(Path(e["rel_path"]).with_suffix(""))), [])
        stem_paired = len(pair) > 1 and e["suffix"] == ".stl" and any(p["suffix"] == ".ply" for p in pair)
        if bm.get("role") == "FORMAT_ALTERNATE" and not stem_paired and not _match(e, list(repeat_names)):
            excluded.append({**base, "reason": f"BENCHMARK_ROLE_FORMAT_ALTERNATE (truth {bm.get('primary_truth')})"})
            continue
        step = steps[0]
        lists: list[str] = []
        staged = _match(e, list(staged_names))
        stage_class = "COMPLETE"
        if staged:
            lists.append("G3_STAGED")
            stage_class = ("BLANK_STAGE0" if staged == BLANK_STAGE0 else
                           "STAGED_FULL" if staged in G3_FULL else "STAGED_PARTIAL")
        alternate_of = None
        if stem_paired:
            alternate_of = case_ref_for(next(p for p in pair if p["suffix"] == ".ply")["sha256"])
        if stage_class in ("COMPLETE", "STAGED_FULL") and alternate_of is None:
            lists.append("G2_COMPLETE")
            for tier, fams in KNOWN_GOOD.items():
                if fam in fams:
                    lists.append(f"KNOWN_GOOD:{tier}")
        for rfam, names in REPEATABILITY_PLAN.items():
            if rfam == fam and _match(e, names):
                lists.append(f"REPEATABILITY:{rfam}")
        if e["sha256"] in gold_by_scan:
            lists.append("GOLD_G2A")
        br = broad_by_scan.get(e["sha256"])
        notes = []
        if br is not None and br.get("status") == "BROAD":
            lists.append("BROAD")
            if br.get("cad_sha256") != step["sha256"]:
                notes.append(f"BROAD_CAD_DIFFERS: the case list names CAD {str(br.get('cad_sha256'))[:12]}; "
                             f"the case uses {step['sha256'][:12]}")
        keys = stage_keys_for(fam, [Path(n).name for n in _names(e)], repo_root)
        states = {r.get("state") for r in keys.get("scan_files") or []}
        provisional = "G2_COMPLETE" in lists and stage_class == "COMPLETE" and "partial" in states
        expected_completed = 0 if stage_class == "BLANK_STAGE0" else keys.get("expected_completed")
        if len(steps) > 1:
            notes.append("MULTIPLE_STEPS_IN_FAMILY: " + ", ".join(s["rel_path"] for s in steps))
        if provisional:
            notes.append("STAGE_KEYS_SAY_PARTIAL: G2 membership is provisional on the frozen visual truth table")
        if not lists:
            excluded.append({**base, "reason": f"FORMAT_ALTERNATE_IN_NO_LIST (same mesh as {alternate_of})"
                             if alternate_of else "IN_NO_LIST"})
            continue
        cases.append({
            **base, "case_ref": case_ref_for(e["sha256"]), "scan_origin": "SOURCE_ASSETS",
            "scan_format": (e.get("header") or {}).get("format"),
            "vertex_count": (e.get("header") or {}).get("vertex_count"),
            "face_count": (e.get("header") or {}).get("face_count"),
            **recipe_fields(step),
            "stage_class": stage_class, "lists": lists, "format_alternate_of": alternate_of,
            "provisional_complete": provisional, "broad": broad_fields(br), "benchmark": bm or None,
            "stage_keys": {**keys, "expected_completed": expected_completed,
                           "expected_completed_source": ("plan §5 G3: the blank is stage 0"
                                                         if stage_class == "BLANK_STAGE0" else
                                                         ("user_truth.scan_truth.completed_bends"
                                                          if keys.get("expected_completed") is not None else None))},
            "notes": notes,
        })
    # BROAD rows whose scan is outside the source tree
    ext_by_sha = {e["sha256"]: e for e in inventory.get("external_entries") or [] if e.get("sha_matches")}
    for r in broad_rows:
        if r["scan_sha256"] in seen:
            continue
        seen.add(r["scan_sha256"])
        base = {"scan_sha256": r["scan_sha256"], "scan_rel": r.get("scan_rel"), "scan_path": r.get("scan_path"),
                "aliases": list((ext_by_sha.get(r["scan_sha256"]) or {}).get("aliases", [])), "family": r["family"]}
        if r.get("status") != "BROAD":
            excluded.append({**base, "reason": f"BROAD_EXCLUDED: {r.get('reason')}",
                             "broad_case_id": r.get("broad_case_id")})
            continue
        step = step_by_sha.get(r["cad_sha256"])
        if step is None:
            excluded.append({**base, "reason": "BROAD_CAD_NOT_IN_INVENTORY", "broad_case_id": r.get("broad_case_id")})
            continue
        stage_class = _stage_from_hints(r.get("stage_hints") or [])
        lists = ["BROAD"]
        if stage_class != "COMPLETE":
            lists.append("BROAD_STAGED")
        if r["scan_sha256"] in gold_by_scan and stage_class == "COMPLETE":
            lists.insert(0, "G2_COMPLETE")
            for tier, fams in KNOWN_GOOD.items():
                if r["family"] in fams:
                    lists.append(f"KNOWN_GOOD:{tier}")
        if r["scan_sha256"] in gold_by_scan:
            lists.append("GOLD_G2A")
        ext = ext_by_sha.get(r["scan_sha256"]) or {}
        keys = stage_keys_for(r["family"], [Path(r.get("scan_path") or "").name], repo_root)
        cases.append({
            **base, "case_ref": r["case_ref"], "scan_origin": "EXTERNAL_CONSUMED",
            "scan_format": (ext.get("header") or {}).get("format"),
            "vertex_count": (ext.get("header") or {}).get("vertex_count"),
            "face_count": (ext.get("header") or {}).get("face_count"),
            **recipe_fields(step), "stage_class": stage_class, "lists": lists, "format_alternate_of": None,
            "provisional_complete": bool(r.get("staged_or_incomplete")) and stage_class == "COMPLETE"
            and "G2_COMPLETE" in lists,
            "broad": broad_fields(r), "benchmark": (benchmark_scans or {}).get(r["scan_sha256"]),
            "stage_keys": {**keys, "expected_completed": 0 if stage_class == "BLANK_STAGE0"
                           else keys.get("expected_completed")},
            "notes": [],
        })
    cases.sort(key=lambda c: (c["family"], str(c["scan_rel"])))
    repeat_sets = []
    for rfam, names in REPEATABILITY_PLAN.items():
        members = [c for c in cases if f"REPEATABILITY:{rfam}" in c["lists"]]
        listed = [n for n in names]
        found = {n for c in members for n in [str(c["scan_rel"]).split("02_CAD_QC_3D/", 1)[-1], *c["aliases"]]}
        missing = [n for n in listed if n not in found]
        unique = sorted({c["case_ref"] for c in members})
        status = "EVALUABLE" if len(unique) >= 2 else "NOT_EVALUABLE"
        reason = None
        if status == "NOT_EVALUABLE":
            aliased = [n for c in members for n in c["aliases"] if n in listed]
            if aliased:
                reason = ("BYTE_IDENTICAL_COPIES: " + ", ".join(sorted(aliased)) + " is a copy of the same scan (sha256 "
                          + ", ".join(sorted({c["scan_sha256"][:12] for c in members})) + ")")
            elif missing:
                reason = "LISTED_FILES_MISSING: " + ", ".join(missing)
            else:
                reason = "FEWER_THAN_TWO_SCANS"
        kind = ("FORMAT_PAIR" if len(unique) == 2 and any(c["format_alternate_of"] for c in members)
                else "PHYSICAL_REPEATS")
        repeat_sets.append({"family": rfam, "listed": listed, "case_refs": unique, "status": status,
                            "reason": reason, "kind": kind if status == "EVALUABLE" else None, "missing": missing})
    return {"cases": cases, "excluded_scans": sorted(excluded, key=lambda x: str(x["scan_rel"])),
            "repeatability_sets": repeat_sets}


_STRATA_V3 = (("STAGED", True), ("REPEATABILITY", False), ("OTHER", False))
_STRATA_V4 = (("GOLD", True), ("STAGED", True), ("REPEATABILITY", False), ("OTHER", False))


def _draw(strata: Mapping[str, Sequence[str]], label: str, inventory_sha256: str,
          order: Sequence[tuple[str, bool]] = _STRATA_V3) -> dict[str, Any]:
    seed_hex = hashlib.sha256(f"{label}|{inventory_sha256}|{doc_sha256(dict(strata))}".encode()).hexdigest()
    rng = np.random.default_rng(int(seed_hex[:16], 16))
    selected: dict[str, list[str]] = {}
    for name, at_least_one in order:
        pool = list(strata.get(name, []))
        k = int(round(len(pool) / 3.0))
        if at_least_one:
            k = max(1, k) if pool else 0
        k = min(k, len(pool))
        selected[name] = sorted(str(x) for x in rng.choice(pool, size=k, replace=False)) if k else []
    return {"seed_inputs": {"label": label, "inventory_sha256": inventory_sha256, "strata": dict(strata)},
            "seed_sha256": seed_hex, "selected_by_stratum": selected,
            "families": sorted(f for v in selected.values() for f in v)}


def draw_devcheck(cases: Sequence[Mapping[str, Any]], inventory_sha256: str,
                  repeatability_sets: Sequence[Mapping[str, Any]] = (), *,
                  superseded: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The stratified, input-seeded DEV-CHECK family draw (module docstring).

    Rule V4 strata, in order: GOLD (families with a Gold11 G2(a) case: at least one is drawn, so
    G2(a) is evaluable on DEV-CHECK as amendment 4f requires), STAGED (G3 families and BROAD staged
    families that are not 6f repeatability families; at least one), REPEATABILITY (evaluable 6f
    sets) and OTHER; ``round(n/3)`` per stratum. Rule V3 (no GOLD stratum) is recomputed and recorded
    as superseded; ``superseded`` is the first pass's V2 draw block (recorded, never used)."""
    families = sorted({c["family"] for c in cases})
    eligible = [f for f in families if f not in DEVCHECK_INELIGIBLE]
    evaluable = {s["family"] for s in repeatability_sets if s.get("status") == "EVALUABLE"}
    broad_staged = {c["family"] for c in cases if "BROAD_STAGED" in c["lists"]}
    gold_fams = {c["family"] for c in cases if "GOLD_G2A" in c["lists"]}

    def is_staged(f: str) -> bool:
        return f in G3_PLAN_STAGED or (f in broad_staged and f not in REPEATABILITY_PLAN)

    # superseded rule V3 (the same draw without the GOLD stratum), recomputed for the record
    st3 = sorted(f for f in eligible if is_staged(f))
    rp3 = sorted(f for f in eligible if f in evaluable and f not in st3)
    v3 = _draw({"STAGED": st3, "REPEATABILITY": rp3,
                "OTHER": sorted(f for f in eligible if f not in st3 and f not in rp3)},
               "E4B-DEVCHECK-V3", inventory_sha256, _STRATA_V3)
    gold = sorted(f for f in eligible if f in gold_fams)
    staged = sorted(f for f in eligible if is_staged(f) and f not in gold)
    repeat = sorted(f for f in eligible if f in evaluable and f not in gold and f not in staged)
    other = sorted(f for f in eligible if f not in gold and f not in staged and f not in repeat)
    strata = {"GOLD": gold, "STAGED": staged, "REPEATABILITY": repeat, "OTHER": other}
    v4 = _draw(strata, DEVCHECK_SEED_LABEL, inventory_sha256, _STRATA_V4)
    selected_staged = [f for f in v4["families"] if is_staged(f)]
    out = {"rule": "round(n/3) eligible families per stratum (GOLD >= 1, STAGED >= 1), generator seeded by "
                   "sha256(label | inventory sha256 | canonical strata); population = families of every G2, G3, "
                   "repeatability and BROAD case",
           "rule_version": DEVCHECK_SEED_LABEL, "ineligible": dict(DEVCHECK_INELIGIBLE),
           **v4, "strata": strata, "family_fraction": (len(v4["families"]) / len(families)) if families else None,
           "families_total": len(families), "staged_families_selected": selected_staged,
           "gold11_families_selected": [f for f in v4["families"] if f in gold_fams],
           "superseded_rule_v3": {
               "families": v3["families"], "seed_sha256": v3["seed_sha256"], "strata": v3["seed_inputs"]["strata"],
               "why": "superseded before the freeze (no E3 had run on any physical case; the legacy control arm "
                      "had run in the first pass): its draw held no family with a Gold11 G2(a) case, so G2(a) "
                      "would be NOT_EVALUABLE on DEV-CHECK and amendment 4f ('gates met' requires G2 and G3 to hold "
                      "on DEV-CHECK as well) could never be met; rule V4 adds the GOLD stratum (at least one Gold11 "
                      "family)"}}
    if superseded:
        out["superseded_rule_v2"] = {
            "families": list(superseded.get("families") or []), "seed_sha256": superseded.get("seed_sha256"),
            "rule_version": superseded.get("rule_version"),
            "why": "superseded before the freeze (no E3 had run on any physical case; the legacy control arm had "
                   "run in the first pass): the owner's wave-4 directions (2026-10-17) added the BROAD families "
                   "(CONSUMED benchmark scans) to the population",
            "superseded_rule_v1": superseded.get("superseded_rule_v1")}
    return out


def _lists(cs: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    names = sorted({lab.split(":")[0] if lab.startswith(("REPEATABILITY", "KNOWN_GOOD")) else lab
                    for c in cs for lab in c["lists"]})
    out: dict[str, list[str]] = {}
    for n in names:
        out[n] = sorted(c["case_ref"] for c in cs if any(lab == n or lab.startswith(n + ":") for lab in c["lists"]))
    return out


def split_public_sealed(cases_doc: Mapping[str, Any], devcheck: Mapping[str, Any],
                        gold_cases: Sequence[Mapping[str, Any]], *,
                        broad_gold: Sequence[Mapping[str, Any]] = (),
                        broad_excluded: Sequence[Mapping[str, Any]] = (),
                        relabel_cases: Sequence[Mapping[str, Any]] = (),
                        relabel_snapshot: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(A1-visible lists, sealed DEV-CHECK document)``; nothing of a DEV-CHECK family is public.

    ``relabel_snapshot`` (the full Gold11 relabel snapshot) is split by file: each document lists only
    its own audience's relabel files (``relabel_files``) with their snapshot sha256 and the count of the
    other files, so A1 scoring re-reads the A1 files only (:func:`.gold.relabel_snapshot` ``only_files``)."""
    from .gold import relabel_file_split

    sealed_fams = set(devcheck["families"])

    def part(rows: Sequence[Mapping[str, Any]], sealed: bool) -> list[dict[str, Any]]:
        missing = [r.get("case_id") or r.get("case_ref") for r in rows if not r.get("family")]
        if missing:  # a row without a family cannot be placed: never let it go public by default
            raise ValueError(f"rows without a family cannot be split public/sealed: {missing[:5]}")
        return [dict(r) for r in rows if (r.get("family") in sealed_fams) == sealed]

    g11 = [g for g in gold_cases if g.get("tier", "GOLD11") == "GOLD11"]
    pub_cases, dev_cases = part(cases_doc["cases"], False), part(cases_doc["cases"], True)
    pub_sets = [s for s in cases_doc["repeatability_sets"] if s["family"] not in sealed_fams]
    dev_sets = [s for s in cases_doc["repeatability_sets"] if s["family"] in sealed_fams]
    created = now_utc()
    common = {"created_at": created, "case_schema": CASES_SCHEMA}
    public = {"schema": CASES_SCHEMA, **common, "audience": "A1 (accuracy loop) and reviewers",
              "rule": "A1 runs only these cases; every other authorized scan may be DEV-CHECK; never read *SEALED*",
              "cases": pub_cases, "lists": _lists(pub_cases), "repeatability_sets": pub_sets,
              "gold_cases": part(g11, False), "relabel_gold_cases": part(relabel_cases, False),
              "broad_gold_cases": part(broad_gold, False),
              "broad_gold_excluded_before_scoring": part(broad_excluded, False),
              "excluded_scans": [e for e in cases_doc["excluded_scans"] if e.get("family") not in sealed_fams],
              "known_good_families": {k: [f for f in v if f not in sealed_fams] for k, v in KNOWN_GOOD.items()}}
    sealed = {"schema": DEVCHECK_SCHEMA, **common,
              "audience": "the one final DEV-CHECK run and the G3/G4 reviewers; A1 must never read this file",
              "draw": dict(devcheck), "cases": dev_cases, "lists": _lists(dev_cases), "repeatability_sets": dev_sets,
              "excluded_scans": [e for e in cases_doc["excluded_scans"] if e.get("family") in sealed_fams],
              "gold_cases": part(g11, True), "relabel_gold_cases": part(relabel_cases, True),
              "broad_gold_cases": part(broad_gold, True),
              "broad_gold_excluded_before_scoring": part(broad_excluded, True),
              "known_good_families": {k: [f for f in v if f in sealed_fams] for k, v in KNOWN_GOOD.items()},
              "gates": "G2 and G3 must also hold on this subset (amendment 4f); it runs once, at the end"}
    if relabel_snapshot is not None:
        pub_ids = {g["case_id"] for g in public["gold_cases"]}
        pub_files, other = relabel_file_split(relabel_snapshot, pub_ids)
        for doc, mine, theirs in ((public, pub_files, other), (sealed, other, pub_files)):
            doc.update(relabel_files=mine, relabel_other_files=len(theirs),
                       relabel_snapshot_sha256=doc_sha256([[r["file"], r["file_sha256"]] for r in mine]),
                       relabel_snapshot_status=relabel_snapshot.get("status"),
                       relabel_snapshot_scope="this document's relabel files only (the other files are counted)")
    return public, sealed
