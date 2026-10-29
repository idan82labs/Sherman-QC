"""The pre-A1 freeze ledger entry (plan §10 amendment 4: "freeze before A1 starts, hashes recorded in the ledger").

One FREEZE entry is appended with E4a's hash-chained ledger
(:func:`..freeze.append_entry`) to ``docs/production/runlog/freeze/LEDGER.json``.
The file is E4a's JSONL ledger format; with its single FREEZE entry it is also a
valid JSON document, and the runner appends one RUN line per QUAL run later.
Items (:func:`freeze_entry_items`):

- **thresholds** (4a): the sha256 of every E4a threshold document (G1, G2_LINE,
  G2_PHYSICAL, G3, G4), plus ``thresholds_plan_verbatim.md`` -- plan §5 and §10
  copied byte for byte from ``PRODUCTION_LINE_PLAN.md`` (:func:`plan_sections`) --
  and ``thresholds_documents.json`` (the documents themselves);
- **seeds** (4c): the planned E2 case ids and seeds of every DEV tier and of the
  QUAL ``full`` and ``runtime`` tiers (planning builds no geometry and reads no
  output);
- **scanner model card** (4b): the E2 model sha256 of
  ``docs/production/runlog/scanner_model_card.json``;
- **recipes** (4d): every A1-visible frozen DEV_AUTO_CONFIRMED recipe file (path +
  sha256); the sealed recipes only as one aggregate sha256 (no file name, so no
  DEV SKU of a sealed family appears in the ledger);
- **case lists** (4e, 4f, 6, 7): the A1 lists and gold (G2, G3, known-good,
  repeatability, BROAD, BROAD_STAGED, the human Gold11, the relabel snapshot and
  the BROAD gold);
- **DEV-CHECK, sha256 only** (``devcheck_sealed``): ``devcheck_SEALED.json``, the
  full amendment-6a transform list ``gold_transforms_devcheck_SEALED.json`` (A1's
  ``gold_transforms.json`` names its sha256 too), the sealed BROAD reference lanes,
  the sealed recipes (aggregate), the sealed legacy summary and the sealed G3
  manifest and template -- no path of a sealed case, no count;
- **decisions** taken before A1 (``decisions``): the amendment-4d compile-rate gate
  rule (:data:`.physical.COMPILE_GATE_RULE`, with the figures at the freeze), the
  amendment-7 order rule (truth-relevant cases, the run-physical gate and the
  scorer's order check), the work-directory rule and the DEV-CHECK seal's scope;
- **owner directions** (``owner_directions.json``): the wave-4 directions that bind
  the freeze (no part-specific optimization, family-macro BROAD reporting with the
  0.80 aim, the per-bend production gate, labelling conventions, the Gold11 dual
  report), as relayed to this chunk;
- **code fingerprints** of the engine, compile, eval and contracts packages (eval
  and contracts are frozen: any later change makes scorecards non-qualifying
  unless the owner waives the exact fingerprint);
- **files**: the inventory, Gold transform list, compile report, legacy control
  summaries, G3 material manifests and truth-table templates, the plan of record
  and the freeze README.

No engine config is frozen here: the engine config is A1's final knob and E4a
compares it with the *latest* FREEZE entry that defines one (runs without a
config match a ledger that defines none). The frozen G3 visual truth table is
recorded by a later FREEZE entry (``prefreeze record-truth-table``: the A1 table
by path and sha256, the DEV-CHECK table by sha256 only).

The sha256 of the first FREEZE entry is the **anchor**: it is written to
``ANCHOR.txt`` and must also be recorded outside this directory (R1:
``docs/CURRENT_STATUS.md``); every qualifying run passes it as
``--ledger-anchor``. :func:`write_ledger` refuses to touch an existing ledger
unless ``replace_before_a1`` is given **and** the ledger holds no RUN or WAIVER
entry (i.e. nothing has been evaluated against it yet).
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import FREEZE_DIR, REPO_ROOT, RUNLOG, now_utc, read_json, sha256_file, write_json

__all__ = [
    "OWNER_DIRECTIONS",
    "sealed_items",
    "decisions",
    "LEDGER_NAME",
    "ANCHOR_NAME",
    "FROZEN_SEEDS",
    "plan_sections",
    "write_verbatim_thresholds",
    "freeze_entry_items",
    "write_ledger",
    "verify_freeze",
]

LEDGER_NAME = "LEDGER.json"
ANCHOR_NAME = "ANCHOR.txt"
PLAN_REL = "docs/production/PRODUCTION_LINE_PLAN.md"
CARD_REL = "docs/production/runlog/scanner_model_card.json"
#: (tier, seed set) pairs whose planned cases are frozen (QUAL only for the gated tiers).
FROZEN_SEEDS: tuple[tuple[str, str], ...] = (("smoke", "DEV"), ("regression", "DEV"), ("full", "DEV"),
                                             ("runtime", "DEV"), ("full", "QUAL"), ("runtime", "QUAL"))
_HEADING = re.compile(r"^## (\d+)\. ", re.MULTILINE)


def plan_sections(text: str, numbers: Sequence[str] = ("5", "10")) -> list[dict[str, Any]]:
    """The verbatim text of each top-level ``## <n>.`` section (up to the next ``## `` heading or a ``---`` rule)."""
    out = []
    heads = [(m.group(1), m.start()) for m in _HEADING.finditer(text)]
    for i, (num, start) in enumerate(heads):
        if num not in numbers:
            continue
        end = heads[i + 1][1] if i + 1 < len(heads) else len(text)
        body = text[start:end]
        rule = body.find("\n---\n")
        if rule != -1:
            body = body[: rule + 1]
        first_line = text.count("\n", 0, start) + 1
        out.append({"section": num, "first_line": first_line, "last_line": first_line + body.count("\n") - 1,
                    "text": body})
    missing = sorted(set(numbers) - {s["section"] for s in out})
    if missing:
        raise ValueError(f"plan sections not found: {missing}")
    return out


def write_verbatim_thresholds(out_path: Path, plan_path: Path | None = None) -> dict[str, Any]:
    """Copy plan §5 and §10 byte for byte into ``out_path`` with a provenance header."""
    plan = plan_path or REPO_ROOT / PLAN_REL
    text = plan.read_text(encoding="utf-8")
    secs = plan_sections(text)
    head = ["<!-- E4b pre-A1 freeze: plan §5 (accuracy gates) and §10 (binding amendments), verbatim.",
            f"     source: {PLAN_REL} (sha256 {sha256_file(plan)})"]
    for s in secs:
        head.append(f"     §{s['section']}: lines {s['first_line']}-{s['last_line']}")
    head.append("     Thresholds never change mid-run (amendment 3). -->")
    body = "\n".join(head) + "\n\n" + "".join(s["text"] for s in secs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    return {"path": str(out_path), "sha256": sha256_file(out_path), "plan_sha256": sha256_file(plan),
            "sections": [{k: s[k] for k in ("section", "first_line", "last_line")} for s in secs]}


def _existing(paths: Mapping[str, Path]) -> dict[str, Path]:
    return {k: p for k, p in paths.items() if Path(p).is_file()}


#: The owner's wave-4 directions that bind the freeze, as relayed to E4b (2026-10-17).
OWNER_DIRECTIONS: dict[str, Any] = {
    "date": "2026-10-17",
    "relayed_by": "the wave-4 workflow task text (E4b); consistent with the session's recorded owner agreement",
    "directions": [
        "Aim for line-F1 >= 0.80 with NO part-specific optimization: generality over the parts in the mix.",
        "No SKU/family/part-id/filename-dependent code, thresholds or special cases anywhere in the engine "
        "(E3 never receives filename tokens or metadata counts, plan amendment 4).",
        "Primary production gate: per-bend correctness given CAD (formed/not, flange length, angle, PASS/FAIL "
        "with ~zero false PASS).",
        "Blind/broad line-F1 is reported family-macro.",
        "Labelling conventions: a joggle is 2 bends; formed 15 deg tabs count; a fold interrupted by windows or "
        "reliefs folding the same two plates is one line; the line locus is the visible-surface apex (amendment 6a).",
        "Gold11 human labels are under audit: until the owner signs off, report G2(a) against BOTH the human "
        "Gold11 and ai_bend_lines_gold11_relabel_v1 if present; never replace silently.",
    ],
    "freeze_consequences": {
        "broad_list": "BROAD = every CONSUMED scan of case_list_broad_v3.json with CAD that compiles; family-macro "
                      "line-F1 (mean over families of the family-micro F1) and per-bend checks, report-only, owner "
                      "target 0.80",
        "g2a_dual_report": "G2(a) gate of record = human Gold11; the relabel is scored next to it (report-only)",
        "plan_gates": "plan §5 and §10 thresholds unchanged (thresholds_plan_verbatim.md)",
    },
}


def _aggregate_sha(paths: Sequence[Path]) -> str:
    from contracts.production.canonical import sha256_hex

    return sha256_hex(sorted(sha256_file(p) for p in paths))


def sealed_items(freeze_dir: Path = FREEZE_DIR, *, runlog: Path = RUNLOG) -> dict[str, Any]:
    """DEV-CHECK material by sha256 only (no path of a sealed case, no count)."""
    files = {
        "devcheck_SEALED": freeze_dir / "devcheck_SEALED.json",
        "gold_transforms_devcheck_SEALED": freeze_dir / "gold_transforms_devcheck_SEALED.json",
        "broad_reference_lanes_devcheck_SEALED": freeze_dir / "broad_reference_lanes_devcheck_SEALED.json",
        "legacy_control_summary_SEALED": runlog / "legacy_control" / "devcheck_SEALED" / "SUMMARY.json",
        "g3_material_manifest_SEALED": runlog / "g3_truth_material" / "devcheck_SEALED" / "MANIFEST.json",
        "g3_truth_table_template_SEALED": runlog / "g3_truth_material" / "devcheck_SEALED" / "truth_table_TEMPLATE.json",
    }
    out: dict[str, Any] = {name: {"sha256": sha256_file(p)} for name, p in files.items() if p.is_file()}
    rec = sorted((freeze_dir / "recipes_devcheck_SEALED").glob("*.recipe.json"))
    out["recipes_devcheck_SEALED_aggregate"] = {"sha256": _aggregate_sha(rec),
                                                "definition": "canonical sha256 of the sorted file sha256s"}
    return out


def decisions(freeze_dir: Path = FREEZE_DIR) -> dict[str, Any]:
    """The rules decided before A1 that the scorer and the A1-facing commands apply (module docstring)."""
    from .common import WORK_BUILDER_DIR, WORK_SEALED_DIR
    from .physical import COMPILE_GATE_RULE, TRUTH_RELEVANT_LISTS

    figures: dict[str, Any] = {}
    report = freeze_dir / "compile_report.json"
    if report.is_file():
        doc = read_json(report)
        for denom in ("rates", "rates_source_assets"):
            figures[denom] = {v: {"compiled": d.get("compiled"), "steps": d.get("authorized_unique_steps"),
                                  "rate": d.get("rate")} for v, d in sorted((doc.get(denom) or {}).items())}
    return {
        "compile_rate_gate_4d": {
            **COMPILE_GATE_RULE, "figures_at_freeze": figures,
            "status": "DECIDED_BEFORE_A1 (no RUN or WAIVER entry existed); flagged for the owner's / orchestrator's "
                      "confirmation before the anchor is published; never changed after A1 starts"},
        "amendment_7_order": {
            "truth_relevant_cases": f"lists {list(TRUTH_RELEVANT_LISTS)} and provisional-complete G2 cases (every case "
                                    "whose per-bend truth a score reads; STATE_CHECK rows are a completeness check)",
            "run_physical_gate": "prefreeze run-physical refuses E3 on a truth-relevant case until a frozen visual "
                                 "truth table covering its template rows is recorded in this ledger "
                                 "(prefreeze record-truth-table)",
            "scorer": "score_physical rejects a table frozen or recorded after the earliest E3 record of a "
                      "truth-relevant case (G3 NOT_EVALUABLE, final = false)"},
        "work_directories": {
            "builder": f"W/{WORK_BUILDER_DIR}/ of the freeze builder's work directory: every pre-split and superseded "
                       "product (full gold list, full compile report, recipe builds and compiles of every STEP, legacy "
                       "worker outputs, reviewer-blind placements, earlier passes, logs, dry runs); A1 never reads it",
            "a1": "a fresh work directory of A1's own; the A1-facing commands refuse the builder's work directory and, "
                  f"without --include-sealed, a *SEALED* directory or one holding {WORK_SEALED_DIR}/",
            "devcheck": f"W/{WORK_SEALED_DIR}/ of the final DEV-CHECK run (--include-sealed, once, at the end)"},
        "devcheck_seal": "every A1-facing file names only A1 cases; inventory.json hashes every authorized file (A1 and "
                         "DEV-CHECK alike) and seeds the draw, so DEV-CHECK identities are inferable by omission from "
                         "the public tree; the protection is procedural (A1 runs, reads and scores only the A1 lists, "
                         "in a work directory of its own, and never reads a *SEALED* path)",
    }


def freeze_entry_items(freeze_dir: Path = FREEZE_DIR, *, repo_root: Path = REPO_ROOT,
                       runlog: Path = RUNLOG) -> dict[str, Any]:
    """The items of the pre-A1 FREEZE entry (module docstring)."""
    from .. import freeze as e4a_ledger
    from ..thresholds import thresholds_bundle

    write_json(freeze_dir / "thresholds_documents.json", thresholds_bundle())
    write_json(freeze_dir / "owner_directions.json", OWNER_DIRECTIONS)
    verbatim = write_verbatim_thresholds(freeze_dir / "thresholds_plan_verbatim.md", repo_root / PLAN_REL)
    recipes = sorted((freeze_dir / "recipes").glob("*.recipe.json"))
    case_lists = _existing({
        "CASES_A1": freeze_dir / "cases_A1.json", "G2_COMPLETE_A1": freeze_dir / "g2_complete_A1.json",
        "G3_STAGED_A1": freeze_dir / "g3_staged_A1.json", "KNOWN_GOOD_A1": freeze_dir / "known_good_A1.json",
        "REPEATABILITY_A1": freeze_dir / "repeatability_A1.json", "GOLD_G2A_A1": freeze_dir / "gold_g2a_A1.json",
        "GOLD_G2A_RELABEL_A1": freeze_dir / "gold_g2a_relabel_A1.json",
        "BROAD_A1": freeze_dir / "broad_A1.json", "BROAD_STAGED_A1": freeze_dir / "broad_staged_A1.json",
        "GOLD_BROAD_A1": freeze_dir / "gold_broad_A1.json",
    })
    extra = _existing({
        "plan_of_record": repo_root / PLAN_REL,
        "plan_thresholds_verbatim": Path(verbatim["path"]),
        "threshold_documents": freeze_dir / "thresholds_documents.json",
        "owner_directions": freeze_dir / "owner_directions.json",
        "inventory": freeze_dir / "inventory.json",
        "gold_transforms": freeze_dir / "gold_transforms.json",
        "compile_report": freeze_dir / "compile_report.json",
        "broad_reference_lanes_A1": freeze_dir / "broad_reference_lanes_A1.json",
        "freeze_readme": freeze_dir / "README.md",
        "legacy_control_summary": runlog / "legacy_control" / "SUMMARY.json",
        "legacy_control_readme": runlog / "legacy_control" / "README.md",
        "g3_material_manifest": runlog / "g3_truth_material" / "MANIFEST.json",
        "g3_reviewer_instructions": runlog / "g3_truth_material" / "REVIEWER_INSTRUCTIONS.md",
        "g3_truth_table_template": runlog / "g3_truth_material" / "truth_table_TEMPLATE.json",
    })
    items = e4a_ledger.freeze_items(
        seeds=list(FROZEN_SEEDS), allow_qual=True, recipes=[str(p) for p in recipes],
        case_lists={k: str(v) for k, v in case_lists.items()},
        scanner_model_card=str(repo_root / CARD_REL),
        code_roots={k: list(v) for k, v in e4a_ledger.CODE_ROOTS.items()}, repo_root=repo_root,
        extra_files={k: str(v) for k, v in extra.items()})
    items["devcheck_sealed"] = sealed_items(freeze_dir, runlog=runlog)
    items["decisions"] = decisions(freeze_dir)
    card_note = ""
    try:
        card = read_json(repo_root / CARD_REL)
        if "PARTIAL" in str(card.get("notes", "")):
            card_note = ("the scanner model card is PARTIAL (grazing / specular dropout ASSUMED, not measured); "
                         "G1 qualification needs those fields measured or an owner SCANNER_MODEL_PARTIAL waiver")
    except (OSError, ValueError):  # pragma: no cover - the card is part of the run log
        card_note = "scanner model card unreadable"
    items["pre_a1"] = {"recorded_by_chunk": "E4b", "plan": PLAN_REL, "verbatim_thresholds": verbatim,
                       "engine_config": "not frozen here (A1's final knob; compared with the latest FREEZE "
                                        "that defines one)",
                       "dev_check": "sha256 only (devcheck_sealed); content sealed in devcheck_SEALED.json and "
                                    "*_SEALED/ directories; A1 never reads them",
                       "open_items": [x for x in (card_note,
                                                  "amendment-4d gate rule E4B-4D-GATE-2 (the frozen DEV recipes gated "
                                                  "on both denominators, strict V10 reported as a diagnostic) needs "
                                                  "the owner's / orchestrator's confirmation before the anchor is "
                                                  "published; the alternative (strict V10 gated too) makes G2 FAIL by "
                                                  "construction",
                                                  "G3 visual truth table not yet filled (amendment 7): two fresh "
                                                  "reviewers plus a tie-break fill and freeze it and record it with "
                                                  "prefreeze record-truth-table; until then run-physical refuses the "
                                                  "truth-relevant cases",
                                                  "Gold11 relabel (ai_bend_lines_gold11_relabel_v1) not signed off "
                                                  "by the owner: human Gold11 stays the G2(a) gate of record") if x]}
    return items


def write_ledger(freeze_dir: Path = FREEZE_DIR, *, replace_before_a1: bool = False, repo_root: Path = REPO_ROOT,
                 runlog: Path = RUNLOG, label: str = "E4b pre-A1 freeze (2026-10-17)") -> dict[str, Any]:
    """Append the FREEZE entry and write the anchor (module docstring)."""
    from .. import freeze as e4a_ledger

    ledger = freeze_dir / LEDGER_NAME
    if ledger.exists():
        entries = e4a_ledger.read_ledger(ledger)
        used = [e for e in entries if e.get("kind") in ("RUN", "WAIVER")]
        if not replace_before_a1:
            raise FileExistsError(f"{ledger} exists; pass replace_before_a1 only if nothing was evaluated against it")
        if used:
            raise PermissionError(f"{ledger} already holds {len(used)} RUN/WAIVER entries: it cannot be replaced")
        ledger.unlink()
    items = freeze_entry_items(freeze_dir, repo_root=repo_root, runlog=runlog)
    entry = e4a_ledger.append_entry(ledger, label=label, items=items, kind="FREEZE", recorded_by="e4b")
    anchor = freeze_dir / ANCHOR_NAME
    anchor.write_text(
        f"{entry['entry_sha256']}\n"
        f"# sha256 of the first FREEZE entry of {LEDGER_NAME} ({entry['entry_id']}, recorded {entry['recorded_at']}).\n"
        "# Record it outside this directory too (docs/CURRENT_STATUS.md) and pass it as --ledger-anchor.\n",
        encoding="utf-8")
    return {"entry_id": entry["entry_id"], "entry_sha256": entry["entry_sha256"], "ledger": str(ledger),
            "anchor_file": str(anchor), "items": sorted(items)}


def verify_freeze(freeze_dir: Path = FREEZE_DIR, *, repo_root: Path = REPO_ROOT,
                  include_sealed: bool = False) -> dict[str, Any]:
    """Ledger chain, anchor, frozen code and every recorded file hash, as of now.

    ``include_sealed`` also re-hashes the DEV-CHECK material (only the final DEV-CHECK run and R1
    may; A1 never reads a sealed path)."""
    from .. import freeze as e4a_ledger

    ledger = freeze_dir / LEDGER_NAME
    anchor_file = freeze_dir / ANCHOR_NAME
    chain = e4a_ledger.verify_ledger(ledger)
    anchor = anchor_file.read_text(encoding="utf-8").split()[0] if anchor_file.is_file() else None
    check = e4a_ledger.check_against_ledger(
        ledger, code_fingerprints=e4a_ledger.current_code_fingerprints(repo_root, e4a_ledger.FROZEN_CODE),
        anchor_sha256=anchor)
    first = e4a_ledger.first_freeze(ledger) or {}
    drift = []
    items = first.get("items") or {}
    for group in ("case_lists", "files"):
        for name, d in (items.get(group) or {}).items():
            p = Path(d["path"])
            if not p.is_file():
                drift.append({"item": f"{group}:{name}", "problem": "MISSING"})
            elif sha256_file(p) != d["sha256"]:
                drift.append({"item": f"{group}:{name}", "problem": "CHANGED"})
    for f in (items.get("recipes") or {}).get("files", []):
        p = Path(f["path"])
        if not p.is_file() or sha256_file(p) != f["sha256"]:
            drift.append({"item": f"recipe:{p.name}", "problem": "MISSING" if not p.is_file() else "CHANGED"})
    if include_sealed and items.get("devcheck_sealed"):
        now_sealed = sealed_items(freeze_dir, runlog=freeze_dir.parent)
        for name, d in items["devcheck_sealed"].items():
            if (now_sealed.get(name) or {}).get("sha256") != d.get("sha256"):
                drift.append({"item": f"devcheck_sealed:{name}", "problem": "CHANGED_OR_MISSING"})
    fps_now = e4a_ledger.current_code_fingerprints(repo_root)
    fps_frozen = items.get("code_fingerprints") or {}
    return {"checked_at": now_utc(), "ledger": str(ledger), "chain": chain, "anchor": anchor,
            "anchor_matches_first_freeze": anchor is not None and anchor == first.get("entry_sha256"),
            "check_against_ledger": check, "file_drift": drift,
            "code_fingerprints": {k: {"frozen": fps_frozen.get(k), "now": v, "same": fps_frozen.get(k) == v}
                                  for k, v in fps_now.items()},
            "ok": bool(chain["ok"] and check["ok"] and not drift and anchor == first.get("entry_sha256"))}
