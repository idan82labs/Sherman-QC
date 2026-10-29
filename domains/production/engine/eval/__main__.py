"""E4 evaluation command line: ``python -m domains.production.engine.eval <command> ...``.

Commands:

- ``run``        serial tier run: E1 compile + E3 inspect + score -> scorecard (:mod:`.runner`)
- ``score``      rebuild a scorecard from a run directory's ``cases/*.json``
- ``blindness``  amendment-4 renamed-copy check on some cases
- ``render``     G4 global views + close-ups for one E3 output directory
- ``packet``     G4 reviewer packet with secret canaries + sealed key (:mod:`.g4`)
- ``g4``         evaluate ``g4_verdicts.jsonl`` against a packet and its sealed key
- ``freeze``     append a FREEZE entry to a ledger (:mod:`.freeze`); ``verify-ledger`` checks it
- ``waive``      append an OWNER WAIVER entry (owner decisions only; agents never run it)
- ``model-card`` amendment-4b scanner model card from authorized real scans (:mod:`.scanner_card`)
- ``line-metric`` G2 line metric from Gold cases and prediction files (:mod:`.line_metric`)

Exit codes: 0 success (the command ran; a gate verdict is in its output), 2 usage
or input error, 3 internal error.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path


def _print(doc) -> None:
    sys.stdout.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _cmd_run(a) -> int:
    from .runner import RunnerConfig, run_tier

    cfg = RunnerConfig(tier_dir=a.tier_dir, out_dir=a.out, case_ids=tuple(a.case) or None,
                       families=tuple(a.family) or None, limit=a.limit, force=a.force, force_compile=a.force_compile,
                       engine_timeout_s=a.timeout_s, engine_rss_limit_bytes=int(a.rss_limit_mb * 1024 * 1024),
                       engine_config=a.engine_config, compile_config=a.compile_config, bootstrap=not a.no_bootstrap,
                       keep_engine_outputs=not a.drop_outputs, ledger=a.ledger, ledger_anchor=a.ledger_anchor,
                       final_qual_run=a.final_qual_run, engine_bundle_id=a.engine_bundle_id,
                       extra_env=dict(x.split("=", 1) for x in a.engine_env))
    if a.engine_cmd:
        cfg = replace(cfg, engine_cmd=tuple(a.engine_cmd.split()))

    def progress(i: int, n: int, d: dict) -> None:
        sys.stdout.write("PROGRESS\t" + json.dumps({"case": i, "of": n, **d}, sort_keys=True) + "\n")
        sys.stdout.flush()

    card = run_tier(cfg, progress=progress)
    sys.stdout.write("RESULT\t" + json.dumps({"verdict": card["verdict"], "regression_verdict": card["regression_verdict"],
                                              "qualifying": card["qualifying"],
                                              "scorecard": str(a.out / "scorecard.json")}) + "\n")
    return 0


def _cmd_score(a) -> int:
    from .runner import rescore_run

    if not (a.run_dir / "run_manifest.json").is_file():
        raise FileNotFoundError(f"{a.run_dir}: no run_manifest.json")
    card = rescore_run(a.run_dir, tier_dir=a.tier_dir, ledger=a.ledger, ledger_anchor=a.ledger_anchor)
    _print({"verdict": card["verdict"], "regression_verdict": card["regression_verdict"],
            "qualifying": card["qualifying"], "not_qualifying_reasons": card["not_qualifying_reasons"],
            "records_rescored": card["provenance"].get("records_rescored")})
    return 0


def _cmd_blindness(a) -> int:
    from .runner import RunnerConfig, blindness_check

    cfg = RunnerConfig(tier_dir=a.tier_dir, out_dir=a.out)
    if a.engine_cmd:
        cfg = replace(cfg, engine_cmd=tuple(a.engine_cmd.split()))
    _print(blindness_check(cfg, a.case))
    return 0


def _cmd_render(a) -> int:
    from .render import plan_closeups, render_closeup, render_global_views, scene_from_engine_output

    scene = scene_from_engine_output(a.engine_dir, a.compiled, a.recipe, title=a.title or "")
    recs = render_global_views(scene, a.out)
    for plan in plan_closeups(scene, px_per_mm=a.px_per_mm):
        recs.append(render_closeup(scene, plan, a.out / f"closeup_{plan.bend_uid}_{plan.tile_index}.png"))
    (a.out / "render_manifest.json").write_text(json.dumps(recs, indent=1) + "\n", encoding="utf-8")
    _print({"images": len(recs), "out": str(a.out)})
    return 0


def _cmd_packet(a) -> int:
    from .g4 import RenderJob, build_review_packet

    jobs_doc = json.loads(a.jobs.read_text(encoding="utf-8"))
    jobs = [RenderJob(engine_dir=Path(j["engine_dir"]), compiled_json=Path(j["compiled_json"]),
                      recipe_json=Path(j["recipe_json"]), case_ref=j["case_ref"], source=j.get("source", "E3"),
                      population=j.get("population", "SYNTHETIC"), incomplete=j.get("incomplete"))
            for j in jobs_doc]
    # The canary salt always comes from the OS here; reproducible test seeds exist in the Python API only.
    _print(build_review_packet(jobs, a.packet, a.sealed, canary_fraction=a.canary_fraction, px_per_mm=a.px_per_mm))
    return 0


def _cmd_g4(a) -> int:
    from .g4 import evaluate_g4

    report = evaluate_g4(a.packet, a.sealed, a.verdicts, require_synthetic_sample=not a.no_synthetic_requirement)
    if a.out:
        a.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    _print({"verdict": report["verdict"], "reasons": report["reasons"], "review_valid": report["review_valid"]})
    return 0


def _cmd_freeze(a) -> int:
    from .freeze import append_entry, freeze_items

    seeds = [tuple(x.split(":", 1)) for x in a.seeds]
    case_lists = dict(x.split("=", 1) for x in a.case_list)
    extra = dict(x.split("=", 1) for x in a.file)
    from .freeze import CODE_ROOTS

    items = freeze_items(seeds=seeds, allow_qual=a.allow_qual, recipes=a.recipe, case_lists=case_lists,
                         engine_config=a.engine_config, scanner_model_card=a.scanner_model_card,
                         code_roots={k: list(v) for k, v in CODE_ROOTS.items()}, extra_files=extra)
    entry = append_entry(a.ledger, label=a.label, items=items)
    _print({"entry_id": entry["entry_id"], "entry_sha256": entry["entry_sha256"], "seq": entry["seq"],
            "note": "record entry_sha256 of the first FREEZE entry outside the ledger (the anchor: run --ledger-anchor)"})
    return 0


def _cmd_waive(a) -> int:
    from .freeze import append_entry

    items: dict = {"scope": a.scope, "approved_by": a.approved_by, "reason": a.reason}
    if a.scope == "SCANNER_MODEL_PARTIAL":
        from domains.production.engine.synthetic.scanner import ScannerModel

        if a.scanner_model_card is None or not a.fields:
            raise ValueError("a SCANNER_MODEL_PARTIAL waiver needs --scanner-model-card and --field ...")
        items.update(scanner_model_sha256=ScannerModel.from_json(a.scanner_model_card).sha256(), fields=sorted(a.fields))
    else:
        if not a.fingerprint:
            raise ValueError("a code-change waiver needs --fingerprint (the accepted current code fingerprint)")
        items["fingerprint"] = a.fingerprint
    entry = append_entry(a.ledger, label=f"OWNER WAIVER {a.scope}", items=items, kind="WAIVER",
                         recorded_by="owner-waiver-cli")
    _print({"entry_id": entry["entry_id"], "entry_sha256": entry["entry_sha256"], "seq": entry["seq"]})
    return 0


def _cmd_verify_ledger(a) -> int:
    from .freeze import verify_ledger

    out = verify_ledger(a.ledger)
    _print(out)
    return 0 if out["ok"] else 2


def _cmd_model_card(a) -> int:
    from . import scanner_card

    argv = [str(p) for p in a.scans] + ["--out", str(a.out), "--name", a.name, "--measured-on", a.measured_on]
    if a.evidence:
        argv += ["--evidence", str(a.evidence)]
    if a.from_evidence:
        argv += ["--from-evidence", str(a.from_evidence)]
    return scanner_card.main(argv)


def _cmd_line_metric(a) -> int:
    import numpy as np

    from .line_metric import GoldCase, score_line_metric

    gold_doc = json.loads(a.gold.read_text(encoding="utf-8"))
    cases = [GoldCase(case_id=g["case_id"], gold_lines=tuple(g["gold_lines"]), frame=g["frame"],
                      raw_to_reference=None if g.get("raw_to_reference") is None else np.asarray(g["raw_to_reference"]),
                      residual_mm=g.get("residual_mm"), transform_method=g.get("transform_method"))
             for g in gold_doc["cases"]]
    preds = json.loads(a.predictions.read_text(encoding="utf-8"))
    legacy = json.loads(a.legacy.read_text(encoding="utf-8")) if a.legacy else None
    report = score_line_metric(cases, preds, legacy_predictions=legacy)
    a.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    _print({"verdict": report["verdict"], "gates": report["gates"], "f1": report["engine"]["f1"]})
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m domains.production.engine.eval", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="serial tier run: compile + inspect + score")
    r.add_argument("--tier-dir", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--case", action="append", default=[])
    r.add_argument("--family", action="append", default=[])
    r.add_argument("--limit", type=int)
    r.add_argument("--force", action="store_true")
    r.add_argument("--force-compile", action="store_true")
    r.add_argument("--timeout-s", type=float, default=180.0)
    r.add_argument("--rss-limit-mb", type=float, default=4096.0)
    r.add_argument("--engine-cmd", help="engine command prefix (default: python -m domains.production.engine.inspect)")
    r.add_argument("--engine-config", type=Path)
    r.add_argument("--engine-env", action="append", default=[],
                   help="KEY=VALUE for the engine subprocess (the harness strips SHERMAN_E4_* from its own env)")
    r.add_argument("--compile-config", type=Path)
    r.add_argument("--engine-bundle-id")
    r.add_argument("--ledger", type=Path)
    r.add_argument("--ledger-anchor", help="sha256 of the ledger's first FREEZE entry, recorded outside the ledger")
    r.add_argument("--final-qual-run", action="store_true",
                   help="the one final run on QUAL seeds (amendment 4c); needs --ledger, logged as a RUN entry")
    r.add_argument("--no-bootstrap", action="store_true")
    r.add_argument("--drop-outputs", action="store_true", help="keep only result.json per case")
    r.set_defaults(fn=_cmd_run)

    s = sub.add_parser("score", help="rebuild the scorecard of a run directory (re-scores stale records)")
    s.add_argument("--run-dir", required=True, type=Path)
    s.add_argument("--tier-dir", type=Path)
    s.add_argument("--ledger", type=Path)
    s.add_argument("--ledger-anchor")
    s.set_defaults(fn=_cmd_score)

    b = sub.add_parser("blindness", help="renamed-copy determinism check")
    b.add_argument("--tier-dir", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--case", action="append", required=True)
    b.add_argument("--engine-cmd")
    b.set_defaults(fn=_cmd_blindness)

    rd = sub.add_parser("render", help="G4 renders of one E3 output directory")
    rd.add_argument("--engine-dir", required=True, type=Path)
    rd.add_argument("--compiled", required=True, type=Path)
    rd.add_argument("--recipe", required=True, type=Path)
    rd.add_argument("--out", required=True, type=Path)
    rd.add_argument("--title")
    rd.add_argument("--px-per-mm", type=float)
    rd.set_defaults(fn=_cmd_render)

    pk = sub.add_parser("packet", help="G4 reviewer packet with secret canaries")
    pk.add_argument("--jobs", required=True, type=Path,
                    help="JSON list of {engine_dir, compiled_json, recipe_json, case_ref, source E3|LEGACY|ORACLE, "
                         "population SYNTHETIC|PHYSICAL_DEV, incomplete true|false}")
    pk.add_argument("--packet", required=True, type=Path)
    pk.add_argument("--sealed", required=True, type=Path)
    pk.add_argument("--canary-fraction", type=float)
    pk.add_argument("--px-per-mm", type=float)
    pk.set_defaults(fn=_cmd_packet)

    g = sub.add_parser("g4", help="evaluate g4_verdicts.jsonl")
    g.add_argument("--packet", required=True, type=Path)
    g.add_argument("--sealed", required=True, type=Path)
    g.add_argument("--verdicts", required=True, type=Path)
    g.add_argument("--out", type=Path)
    g.add_argument("--no-synthetic-requirement", action="store_true",
                   help="a physical-only packet (its synthetic sample is reviewed in another packet)")
    g.set_defaults(fn=_cmd_g4)

    f = sub.add_parser("freeze", help="append a FREEZE entry to the ledger")
    f.add_argument("--ledger", required=True, type=Path)
    f.add_argument("--label", required=True)
    f.add_argument("--seeds", action="append", default=[], help="tier:seedset, e.g. full:QUAL")
    f.add_argument("--allow-qual", action="store_true")
    f.add_argument("--recipe", action="append", default=[], type=Path)
    f.add_argument("--case-list", action="append", default=[], help="name=path")
    f.add_argument("--engine-config", type=Path)
    f.add_argument("--scanner-model-card", type=Path)
    f.add_argument("--file", action="append", default=[], help="name=path")
    f.set_defaults(fn=_cmd_freeze)

    v = sub.add_parser("verify-ledger", help="verify the ledger hash chain")
    v.add_argument("--ledger", required=True, type=Path)
    v.set_defaults(fn=_cmd_verify_ledger)

    w = sub.add_parser("waive", help="OWNER ONLY: append a WAIVER entry for one waivable mismatch")
    w.add_argument("--ledger", required=True, type=Path)
    w.add_argument("--scope", required=True,
                   choices=["SCANNER_MODEL_PARTIAL", "EVAL_CODE_CHANGED_SINCE_FREEZE", "CONTRACTS_CODE_CHANGED_SINCE_FREEZE"])
    w.add_argument("--approved-by", required=True, help="the owner's name; agents never approve waivers")
    w.add_argument("--reason", required=True)
    w.add_argument("--scanner-model-card", type=Path)
    w.add_argument("--field", dest="fields", action="append", default=[])
    w.add_argument("--fingerprint", help="the code fingerprint the waiver accepts")
    w.set_defaults(fn=_cmd_waive)

    mc = sub.add_parser("model-card", help="amendment-4b scanner model card")
    mc.add_argument("scans", nargs="*", type=Path)
    mc.add_argument("--out", required=True, type=Path)
    mc.add_argument("--evidence", type=Path)
    mc.add_argument("--from-evidence", type=Path, help="evidence JSON or docs/production/ENGINE_EVAL.md")
    mc.add_argument("--name", default="zeiss_geomagic_export_model_card_v1")
    mc.add_argument("--measured-on", default="")
    mc.set_defaults(fn=_cmd_model_card)

    lm = sub.add_parser("line-metric", help="G2 line metric (raw-scan frame)")
    lm.add_argument("--gold", required=True, type=Path,
                    help="{cases: [{case_id, frame raw_scan|reference, gold_lines, raw_to_reference?, residual_mm?}]}")
    lm.add_argument("--predictions", required=True, type=Path, help="{case_id: [line records, raw frame]}")
    lm.add_argument("--legacy", type=Path)
    lm.add_argument("--out", required=True, type=Path)
    lm.set_defaults(fn=_cmd_line_metric)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        return int(a.fn(a))
    except (OSError, ValueError, KeyError) as exc:
        sys.stderr.write(f"error: {type(exc).__name__}: {exc}\n")
        return 2
    except Exception:  # internal error
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
