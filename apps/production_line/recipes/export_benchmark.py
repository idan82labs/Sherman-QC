"""Export released benchmark references as JSONL, one row per recipe bend (C05).

::

    python -m apps.production_line.recipes.export_benchmark --data-root ROOT --out FILE.jsonl \\
        [--sku SKU] [--include-retired]

Each row (``schema = sherman_benchmark_bend_row_v1``): sku, recipe_id,
version, recipe_sha256, content_sha256, compiled_part_sha256, bend_uid,
sequence_index, label, cad_label, machine_move_index, the nominal angle (the
compiled flange-graph included angle, plus the recipe's nominal with its
source and ``convention_verified``), the resolved tolerances, the
flange_inspection policy, the compiled per-side flange nominals and
backgauge nominals, the benchmark status, and ``algo_version`` (the E1
compiler version whose READY expectations these are).

These are *nominals* for benchmark bookkeeping. There is no physical truth
in them; nothing here is a measurement.

Refuses (exit 2) an ``--out`` that resolves inside the repository's
``data/``, ``output/``, ``outputs/``, ``artifacts/``, ``uploads/`` or
``live_scans/`` (research custody), and a data root without a recipe store.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from apps.production_line.recipes.store import RecipeStore

__all__ = ["BENCHMARK_ROW_SCHEMA", "FORBIDDEN_OUT_DIRS", "out_path_refusal", "iter_rows", "export", "main"]

BENCHMARK_ROW_SCHEMA = "sherman_benchmark_bend_row_v1"
FORBIDDEN_OUT_DIRS: tuple[str, ...] = ("data", "output", "outputs", "artifacts", "uploads", "live_scans")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def out_path_refusal(out: Path, repo_root: Path | None = None) -> str | None:
    """Why ``out`` may not be written (None when allowed)."""
    root = (repo_root or _repo_root()).resolve()
    target = Path(os.path.abspath(out))
    try:
        target = target.parent.resolve() / target.name
    except OSError:
        pass
    for name in FORBIDDEN_OUT_DIRS:
        forbidden = root / name
        candidates = {forbidden}
        try:
            candidates.add(forbidden.resolve())
        except OSError:
            pass
        for base in candidates:
            try:
                target.relative_to(base)
            except ValueError:
                continue
            return f"--out may not be inside the repository's {name}/ directory"
    return None


def iter_rows(store: RecipeStore, sku: str | None = None, include_retired: bool = False) -> Iterator[dict[str, Any]]:
    for ref in store.list_benchmark_references(sku=sku, include_retired=include_retired):
        stored = store.get(ref["recipe_id"])
        if stored is None:
            continue
        recipe = stored.recipe()
        nominals = ref.get("nominals") or {}
        compiled_bends = {b["bend_uid"]: b for b in nominals.get("bends", [])}
        expectations = store.get_expectations(stored.recipe_sha256, ref.get("algo_version"))
        ready = expectations is not None and expectations.get("status") == "READY"
        for bend in recipe.bends:
            nb = compiled_bends.get(bend.bend_uid, {})
            yield {
                "schema": BENCHMARK_ROW_SCHEMA,
                "sku": recipe.sku,
                "recipe_id": recipe.recipe_id,
                "version": recipe.recipe_version,
                "recipe_sha256": recipe.recipe_sha256,
                "content_sha256": recipe.content_sha256,
                "compiled_part_sha256": ref.get("compiled_part_sha256"),
                "benchmark_status": ref.get("status"),
                "confirmation": recipe.confirmation,
                "bend_uid": bend.bend_uid,
                "sequence_index": bend.sequence_index,
                "label": bend.label,
                "cad_label": bend.cad_label,
                "machine_move_index": bend.machine_move_index,
                "nominal_angle": {
                    "compiled_included_deg": nb.get("nominal_included_deg"),
                    "recipe": bend.nominal_angle.model_dump(mode="json"),
                },
                "tolerances": {
                    kind: (None if (t := recipe.resolved_tolerance(bend.bend_uid, kind)) is None
                           else t.model_dump(mode="json"))
                    for kind in ("angle_deg", "flange_a_mm", "flange_b_mm", "bend_line_position_mm", "backgauge_mm")
                },
                "flange_inspection": bend.flange_inspection.model_dump(mode="json"),
                "flange_a": nb.get("flange_a"),
                "flange_b": nb.get("flange_b"),
                "backgauge": nb.get("backgauge", []),
                "expectations": {
                    "status": None if expectations is None else expectations.get("status"),
                    "algo_version": ref.get("algo_version"),
                    "ready": ready,
                },
            }


def export(data_root: Path, out: Path, sku: str | None = None, include_retired: bool = False) -> int:
    """Write the JSONL file (tmp + rename); returns the number of rows."""
    db = Path(data_root) / "db" / "recipes.sqlite3"
    if not db.is_file():
        raise FileNotFoundError(f"no recipe store at {db}")
    store = RecipeStore(db)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    count = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in iter_rows(store, sku, include_retired):
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    os.replace(tmp, out)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m apps.production_line.recipes.export_benchmark")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sku", default=None)
    parser.add_argument("--include-retired", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code not in (0, None) else 0
    refusal = out_path_refusal(args.out)
    if refusal is not None:
        print(f"refused: {refusal}", file=sys.stderr)
        return 2
    sku = None
    if args.sku is not None:
        from domains.production.recipe.canonical_sku import SkuError, canonical_sku

        try:
            sku = canonical_sku(args.sku)
        except SkuError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
    try:
        count = export(args.data_root, args.out, sku, args.include_retired)
    except FileNotFoundError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"rows": count, "out": str(args.out)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
