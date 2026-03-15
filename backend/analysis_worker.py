"""
Isolated analysis worker process.

Runs heavy geometry analysis outside the API process so native crashes
cannot take down the backend server.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from qc_engine import ScanQCEngine
from pdf_generator import generate_pdf_report


def _sanitize_error_message(message: Any) -> str:
    text = str(message) if message is not None else ""
    text = re.sub(r"\x1B\[[0-9;]*[A-Za-z]", "", text)
    text = " ".join(text.split())
    return text.strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        num = float(value)
        return num if math.isfinite(num) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _emit(tag: str, payload: Dict[str, Any]) -> None:
    print(f"{tag}\t{json.dumps(payload, ensure_ascii=False)}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sherman QC isolated analysis worker")
    p.add_argument("--job-id", required=True)
    p.add_argument("--ref-path", required=True)
    p.add_argument("--scan-path", required=True)
    p.add_argument("--part-id", required=True)
    p.add_argument("--part-name", required=True)
    p.add_argument("--material", required=True)
    p.add_argument("--tolerance", type=float, required=True)
    p.add_argument("--drawing-context", default="")
    p.add_argument("--drawing-path", default="")
    p.add_argument("--dimension-path", default="")
    p.add_argument("--output-dir", required=True)
    return p


def run_worker(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _progress_callback(update):
        _emit(
            "PROGRESS",
            {
                "stage": update.stage,
                "progress": update.progress,
                "message": update.message,
            },
        )

    engine = ScanQCEngine(progress_callback=_progress_callback, output_dir=str(output_dir))

    try:
        engine.load_reference(args.ref_path)
        engine.load_scan(args.scan_path)

        report = engine.run_analysis(
            part_id=args.part_id,
            part_name=args.part_name,
            material=args.material,
            tolerance=float(args.tolerance),
            drawing_context=args.drawing_context or "",
            drawing_path=args.drawing_path or None,
            output_dir=str(output_dir),
        )

        # Run dimension analysis if XLSX provided, otherwise CAD fallback
        if args.dimension_path:
            try:
                dimension_analysis = engine.run_dimension_analysis(
                    xlsx_path=args.dimension_path,
                    part_id=args.part_id,
                    part_name=args.part_name,
                )
                report.dimension_analysis = dimension_analysis
            except Exception as e:
                report.dimension_analysis = {"error": _sanitize_error_message(e)}
        else:
            try:
                dimension_analysis = engine.run_cad_dimension_analysis(
                    part_id=args.part_id,
                    part_name=args.part_name,
                )
                if dimension_analysis:
                    report.dimension_analysis = dimension_analysis
            except Exception:
                # CAD dimension fallback is optional and should not fail the job.
                pass

        report_dict = _json_safe(report.to_dict())

        # Normalize heatmap paths for API consumption.
        if report_dict.get("heatmaps"):
            for key in list(report_dict["heatmaps"].keys()):
                p = report_dict["heatmaps"].get(key)
                if p:
                    filename = Path(p).name
                    report_dict["heatmaps"][key] = f"/api/heatmap/{args.job_id}/{filename}"

        report_path = output_dir / "report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, allow_nan=False)

        pdf_path = output_dir / f"QC_Report_{args.part_id}.pdf"
        generate_pdf_report(report_dict, str(pdf_path))

        _emit(
            "RESULT",
            {
                "report_path": str(report_path),
                "pdf_path": str(pdf_path),
            },
        )
        return 0
    except Exception as exc:
        clean_error = _sanitize_error_message(exc)
        _emit("ERROR", {"error": clean_error})
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
