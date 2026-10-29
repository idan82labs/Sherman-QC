"""Command line for the E2 synthetic suite.

::

    python -m domains.production.engine.synthetic generate --tier regression --seedset DEV --out DIR
    python -m domains.production.engine.synthetic plan --tier full --seedset DEV [--json]
    python -m domains.production.engine.synthetic part-set --family HAT --seedset DEV --out DIR [--index 0]
    python -m domains.production.engine.synthetic export-inbox --cases DIR --out INBOX [--case ID ...]
    python -m domains.production.engine.synthetic scanner-model --out FILE.json

``generate`` resumes: an existing case directory is skipped only when its spec,
generator version, scanner model, scan plan and options all match; any other
existing directory is refused unless ``--force``. ``--seedset QUAL`` (for
``generate``, ``plan`` and ``part-set``) requires ``--allow-qual``: QUAL seeds
are for the final G1 run only (plan §10 amendment 4c). ``part-set`` writes one
released part with several scans of it (nominal, the amendment-10 error scan,
a staged scan, a blank, an unknown-SKU drop, a wrong part) for onboarding and
E2E flows. ``export-inbox`` copies scans under their ZEISS-like names the way
the cell does (write to a temporary name, then rename); given a part-set
directory it exports that set's scans in drop order.

``--threads N`` sets the raycasting threads (default: ``OMP_NUM_THREADS``, or
all cores when it is unset); outputs do not depend on it.

Exit codes: 0 success; 2 usage or policy error (bad arguments, QUAL without
``--allow-qual``, an existing directory that does not match, a bad scanner
model card); 3 internal error (traceback on stderr), including any failure
inside geometry generation.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

__all__ = ["main", "build_parser"]


class UsageError(Exception):
    """A CLI argument or input problem (exit 2)."""


def build_parser() -> argparse.ArgumentParser:
    from .cases import SEEDSETS, TIERS
    from .families import FAMILIES

    p = argparse.ArgumentParser(prog="python -m domains.production.engine.synthetic", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="generate a tier of cases")
    g.add_argument("--tier", required=True, choices=TIERS)
    g.add_argument("--seedset", required=True, choices=sorted(SEEDSETS))
    g.add_argument("--out", required=True, type=Path)
    g.add_argument("--allow-qual", action="store_true", help="required for QUAL (final-run-only seeds)")
    g.add_argument("--families", default=None, help=f"comma list from {','.join(FAMILIES)}")
    g.add_argument("--cases", default=None, help="comma list of case ids")
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--scanner-model", type=Path, default=None, help="ScannerModel JSON (e.g. the E4 model card)")
    g.add_argument("--force", action="store_true", help="regenerate existing case directories")
    g.add_argument("--write-as-built-step", action="store_true", help="also write the as-built BRep (truth) as as_built.step")
    g.add_argument("--write-stage-steps", action="store_true", help="also write nominal stage models as stages/S<k>.step")
    g.add_argument("--threads", type=int, default=None, help="raycasting threads (default: OMP_NUM_THREADS, else all cores)")
    g.add_argument("--quiet", action="store_true")
    pl = sub.add_parser("plan", help="print the case plan of a tier (no geometry)")
    pl.add_argument("--tier", required=True, choices=TIERS)
    pl.add_argument("--seedset", required=True, choices=sorted(SEEDSETS))
    pl.add_argument("--allow-qual", action="store_true", help="required for QUAL (final-run-only seeds)")
    pl.add_argument("--json", action="store_true")
    ps = sub.add_parser("part-set", help="one released part plus several scans of it (onboarding / E2E)")
    ps.add_argument("--family", default="HAT", choices=FAMILIES)
    ps.add_argument("--seedset", required=True, choices=sorted(SEEDSETS))
    ps.add_argument("--out", required=True, type=Path)
    ps.add_argument("--index", type=int, default=0)
    ps.add_argument("--allow-qual", action="store_true")
    ps.add_argument("--target-points", type=int, default=None,
                    help="explicit point-count override (recorded); default: the scanner model's native spacing")
    ps.add_argument("--scanner-model", type=Path, default=None)
    ps.add_argument("--threads", type=int, default=None)
    ps.add_argument("--force", action="store_true")
    ex = sub.add_parser("export-inbox", help="copy scans under ZEISS-like names (temp + rename)")
    ex.add_argument("--cases", required=True, type=Path, help="a generated tier directory")
    ex.add_argument("--out", required=True, type=Path, help="inbox directory")
    ex.add_argument("--case", action="append", default=None, help="case id (repeatable); default all")
    sm = sub.add_parser("scanner-model", help="write the default ScannerModel JSON (model-card template)")
    sm.add_argument("--out", required=True, type=Path)
    return p


def _load_model(path: Path | None):
    from .scanner import ScannerModel

    if path is None:
        return None
    try:
        return ScannerModel.from_json(path)
    except (OSError, ValueError, TypeError) as exc:
        raise UsageError(f"scanner model {path}: {exc}") from exc


def _threads(value: int | None) -> int | None:
    if value is not None and value < 0:
        raise UsageError("--threads must be >= 0 (0 = all cores)")
    return value


def _generate(args: argparse.Namespace) -> int:
    from .cases import generate_tier
    from .families import FAMILIES

    families = [f.strip() for f in args.families.split(",")] if args.families else None
    if families:
        unknown = sorted(set(families) - set(FAMILIES))
        if unknown:
            raise UsageError(f"unknown families {unknown}")
    if args.limit is not None and args.limit < 0:
        raise UsageError("--limit must be >= 0")
    case_ids = [c.strip() for c in args.cases.split(",")] if args.cases else None
    model = _load_model(args.scanner_model)
    threads = _threads(args.threads)

    def progress(i: int, n: int, entry: dict) -> None:
        if args.quiet:
            return
        state = "skipped" if entry.get("skipped") else f"{entry.get('seconds', 0):.1f}s"
        print(f"PROGRESS\t{json.dumps({'case': entry['case_id'], 'done': i, 'total': n, 'points': entry['point_count'], 'status': state})}", flush=True)

    manifest = generate_tier(
        args.tier, args.seedset, args.out, families=families, case_ids=case_ids, limit=args.limit,
        scanner_model=model, force=args.force, write_as_built_step=args.write_as_built_step,
        write_stage_steps=args.write_stage_steps, allow_qual=args.allow_qual, ray_threads=threads, progress=progress,
    )
    missing = {fam: [k for k, v in cov.items() if not v] for fam, cov in manifest["coverage"].items()}
    print(f"RESULT\t{json.dumps({'cases': manifest['case_count'], 'out': str(args.out), 'manifest': str(args.out / 'manifest.json'), 'coverage_gaps': {k: v for k, v in missing.items() if v}})}")
    return 0


def _plan(args: argparse.Namespace) -> int:
    from .cases import plan_cases

    specs = plan_cases(args.tier, args.seedset, allow_qual=args.allow_qual)
    if args.json:
        print(json.dumps([s.to_dict() for s in specs], indent=1))
    else:
        for s in specs:
            density = "native" if s.point_target is None else str(s.point_target)
            print(f"{s.case_id}\t{s.family}\t{s.scenario}\t{','.join(s.features.tags()) or '-'}\t{s.plan}\t{s.output}\t{density}")
    return 0


def _safe_name(name: str) -> bool:
    return bool(name) and "/" not in name and "\\" not in name and not name.startswith(".")


def _export_inbox(args: argparse.Namespace) -> int:
    root: Path = args.cases
    wanted = set(args.case) if args.case else None
    items: list[tuple[str, Path, str]] = []  # (id, scan path, inbox name)
    if (root / "part_set.json").exists():
        doc = json.loads((root / "part_set.json").read_text(encoding="utf-8"))
        by_name = {s["name"]: s for s in doc["scans"]}
        for name in doc["drop_order"]:
            if wanted is None or name in wanted:
                s = by_name[name]
                items.append((name, root / s["dir"] / "scan.ply", s["zeiss_filename"]))
    elif (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["cases"]:
            cid = entry["case_id"]
            if wanted is None or cid in wanted:
                case = json.loads((root / cid / "case.json").read_text(encoding="utf-8"))
                items.append((cid, root / cid / "scan.ply", case["zeiss_filename"]))
    else:
        raise UsageError(f"{root} has neither manifest.json nor part_set.json")
    for _item_id, src, name in items:
        if not _safe_name(name):
            raise UsageError(f"unsafe file name {name!r}")
        if not src.is_file():
            raise UsageError(f"missing scan {src}")
    args.out.mkdir(parents=True, exist_ok=True)
    exported = []
    for item_id, src, name in items:
        tmp = args.out / f".{item_id.replace('/', '_')}.partial"
        shutil.copyfile(src, tmp)
        os.replace(tmp, args.out / name)
        exported.append({"id": item_id, "file": name})
    print(f"RESULT\t{json.dumps({'exported': exported})}")
    return 0


def _part_set(args: argparse.Namespace) -> int:
    from .cases import generate_part_set

    if args.target_points is not None and args.target_points <= 0:
        raise UsageError("--target-points must be > 0")
    model = _load_model(args.scanner_model)
    doc = generate_part_set(args.family, args.seedset, args.out, index=args.index, scanner_model=model,
                            target_points=args.target_points, force=args.force, allow_qual=args.allow_qual,
                            ray_threads=_threads(args.threads))
    print(f"RESULT\t{json.dumps({'set_id': doc['set_id'], 'sku': doc['sku'], 'dir': str(args.out / doc['set_id']), 'scans': [s['name'] for s in doc['scans']]})}")
    return 0


def _scanner_model(args: argparse.Namespace) -> int:
    from .scanner import ScannerModel

    ScannerModel().to_json(args.out)
    print(f"RESULT\t{json.dumps({'out': str(args.out)})}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from .cases import CasePolicyError

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse usage errors exit 2
        return int(exc.code or 0)
    handlers = {"generate": _generate, "plan": _plan, "part-set": _part_set, "export-inbox": _export_inbox, "scanner-model": _scanner_model}
    try:
        return handlers[args.command](args)
    except (UsageError, CasePolicyError) as exc:
        print(f"ERROR\t{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception:  # internal: geometry, I/O or a bug -- surfaced with its traceback
        traceback.print_exc()
        return 3
