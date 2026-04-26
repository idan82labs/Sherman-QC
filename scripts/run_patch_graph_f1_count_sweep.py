#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from bend_inspection_pipeline import load_bend_runtime_config
from patch_graph_latent_decomposition import run_patch_graph_latent_decomposition
from run_patch_graph_f1_prototype import _entry_result_payload


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_caps(raw: str) -> List[int]:
    caps: List[int] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError("caps must be non-negative")
        caps.append(value)
    return list(dict.fromkeys(caps))


def _find_entry(slice_payload: Dict[str, Any], part_key: str) -> Dict[str, Any]:
    for entry in slice_payload.get("entries") or []:
        if str(entry.get("part_key") or "") == str(part_key):
            return dict(entry)
    raise SystemExit(f"part_key not found in slice: {part_key}")


def _best_energy(payload: Dict[str, Any]) -> Optional[float]:
    energies = payload.get("raw_f1_candidate", {}).get("solution_energies") or ()
    if not energies:
        return None
    return float(min(float(value) for value in energies))


def _variant_overrides(cap: Optional[int], *, uncapped: bool = False) -> Optional[Dict[str, Any]]:
    if cap is None and not uncapped:
        return None
    overrides: Dict[str, Any] = {
        "limit_birth_rescue_to_control_upper": False,
        "upper_tail_probe": True,
    }
    if uncapped:
        overrides["birth_rescue_max_active_bends"] = 0
        overrides["count_sweep_variant"] = "uncapped"
    else:
        overrides["birth_rescue_max_active_bends"] = int(cap or 0)
        overrides["count_sweep_variant"] = f"cap_{int(cap or 0)}"
    return overrides


def _variant_summary(*, variant_id: str, cap: Optional[int], payload: Dict[str, Any], base_energy: Optional[float]) -> Dict[str, Any]:
    raw = dict(payload.get("raw_f1_candidate") or {})
    candidate = dict(payload.get("candidate") or {})
    best = _best_energy(payload)
    return {
        "variant_id": variant_id,
        "birth_rescue_max_active_bends": cap,
        "raw_map_bend_count": raw.get("map_bend_count"),
        "raw_exact_bend_count": raw.get("exact_bend_count"),
        "raw_bend_count_range": raw.get("bend_count_range"),
        "raw_solution_counts": raw.get("solution_counts"),
        "raw_owned_bend_region_count": raw.get("owned_bend_region_count"),
        "best_energy": best,
        "energy_delta_vs_baseline": None if best is None or base_energy is None else round(best - base_energy, 6),
        "candidate_bend_count_range": candidate.get("bend_count_range"),
        "candidate_source": candidate.get("candidate_source"),
        "guard_reason": candidate.get("guard_reason"),
        "promotion_candidate": bool((payload.get("raw_f1_promotion_diagnostic") or {}).get("candidate")),
        "promotion_blockers": list((payload.get("raw_f1_promotion_diagnostic") or {}).get("blockers") or ()),
    }


def _write_markdown(path: Path, *, part_key: str, rows: Sequence[Dict[str, Any]], expected_count: Optional[int]) -> None:
    baseline = rows[0] if rows else {}
    lines = [
        f"# F1 Count Sweep - {part_key}",
        "",
        f"Date: {datetime.now().date().isoformat()}",
        "",
        "## Purpose",
        "",
        "Compare the normal routed F1 result against diagnostic upper-tail variants with larger interface-birth caps.",
        "This is an offline evidence artifact only; probe variants are explicitly non-promotable.",
        "",
        "## Summary",
        "",
        f"- expected known count: `{expected_count}`" if expected_count is not None else "- expected known count: unknown",
        f"- baseline raw range: `{baseline.get('raw_bend_count_range')}`",
        f"- baseline best energy: `{baseline.get('best_energy')}`",
        "",
        "## Sweep",
        "",
        "| Variant | Birth cap | Raw range | Raw MAP | Solution counts | Best energy | Delta vs baseline | Promotion candidate |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('variant_id')}`",
                    "`none`" if row.get("birth_rescue_max_active_bends") is None else f"`{row.get('birth_rescue_max_active_bends')}`",
                    f"`{row.get('raw_bend_count_range')}`",
                    f"`{row.get('raw_map_bend_count')}`",
                    f"`{row.get('raw_solution_counts')}`",
                    f"`{row.get('best_energy')}`",
                    f"`{row.get('energy_delta_vs_baseline')}`",
                    f"`{row.get('promotion_candidate')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Use this artifact as evidence for or against exact-count promotion, but do not auto-promote from probe variants.",
            "A safe exact-promotion rule still needs a formal constrained-count comparison, not only active-birth caps.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an offline F1 upper-tail count sweep for one scan.")
    parser.add_argument("--slice-json", required=True)
    parser.add_argument("--part-key", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--route-mode", choices=("off", "observe", "apply"), default="apply")
    parser.add_argument("--caps", default="5,6,8", help="Comma-separated active-bend caps to probe after the baseline run.")
    parser.add_argument("--include-uncapped", action="store_true", help="Also run an uncapped upper-tail probe.")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    slice_payload = _load_json(Path(args.slice_json).resolve())
    entry = _find_entry(slice_payload, args.part_key)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = load_bend_runtime_config(None)
    cache_root = Path(args.cache_dir).resolve() if args.cache_dir else output_dir / "_cache"
    caps = _parse_caps(args.caps)

    variants: List[Dict[str, Any]] = [{"variant_id": "baseline", "cap": None, "uncapped": False}]
    variants.extend({"variant_id": f"cap_{cap}", "cap": cap, "uncapped": False} for cap in caps)
    if args.include_uncapped:
        variants.append({"variant_id": "uncapped", "cap": None, "uncapped": True})

    payloads: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    base_energy: Optional[float] = None
    for variant in variants:
        variant_id = str(variant["variant_id"])
        print(f"[count-sweep] START {args.part_key} {variant_id}", flush=True)
        result = run_patch_graph_latent_decomposition(
            scan_path=str(entry["scan_path"]),
            runtime_config=runtime_config,
            part_id=str(args.part_key),
            cache_dir=cache_root / str(args.part_key),
            route_mode=args.route_mode,
            solver_option_overrides=_variant_overrides(variant.get("cap"), uncapped=bool(variant.get("uncapped"))),
        )
        variant_dir = output_dir / variant_id
        _write_json(variant_dir / "decomposition_result.json", result.to_dict())
        payload = _entry_result_payload(entry=entry, result=result, render_info=None)
        payload["count_sweep_variant"] = variant_id
        _write_json(variant_dir / "summary.json", payload)
        if base_energy is None:
            base_energy = _best_energy(payload)
        row = _variant_summary(variant_id=variant_id, cap=variant.get("cap"), payload=payload, base_energy=base_energy)
        rows.append(row)
        payloads.append(payload)
        print(
            f"[count-sweep] DONE {variant_id}: raw={row['raw_bend_count_range']} best_energy={row['best_energy']} delta={row['energy_delta_vs_baseline']}",
            flush=True,
        )

    expected_count = (entry.get("expectation") or {}).get("expected_bend_count")
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "part_key": str(args.part_key),
        "slice_json": str(Path(args.slice_json).resolve()),
        "scan_path": str(entry.get("scan_path") or ""),
        "expected_bend_count": expected_count,
        "variants": rows,
        "payloads": payloads,
    }
    _write_json(output_dir / "count_sweep_summary.json", summary)
    _write_markdown(output_dir / f"COUNT_SWEEP_{args.part_key}.md", part_key=str(args.part_key), rows=rows, expected_count=expected_count)
    print(json.dumps({"output_dir": str(output_dir), "variants": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
