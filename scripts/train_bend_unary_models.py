#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path('/Users/idant/82Labs/Sherman QC/Full Project')
sys.path.insert(0, str(ROOT / 'backend'))

from bend_unary_model import train_unary_models, save_unary_models  # noqa: E402


def _load_case_payloads(source_dirs):
    payloads = []
    seen = set()
    for source_dir in source_dirs:
        for path in sorted(source_dir.rglob('*.json')):
            if path.name == 'summary.json':
                continue
            try:
                real = path.resolve()
            except Exception:
                real = path
            if real in seen:
                continue
            seen.add(real)
            try:
                payloads.append(json.loads(path.read_text(encoding='utf-8')))
            except Exception:
                continue
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description='Train bootstrap unary GBDT models from bend corpus outputs.')
    parser.add_argument(
        '--source-dir',
        action='append',
        default=[],
        help='Source evaluation directory. Can be provided multiple times.',
    )
    parser.add_argument(
        '--output-dir',
        default='',
        help='Optional explicit output dir. Defaults to output/unary_models/<timestamp>.',
    )
    args = parser.parse_args()

    source_dirs = [Path(p) for p in args.source_dir] if args.source_dir else [
        ROOT / 'data' / 'bend_corpus' / 'evaluation' / 'full_ready_regression_20260311_metrics'
    ]
    missing = [str(p) for p in source_dirs if not p.exists()]
    if missing:
        raise SystemExit(f'source dir not found: {missing}')

    payloads = _load_case_payloads(source_dirs)
    if not payloads:
        raise SystemExit(f'no case payloads found under {source_dirs}')

    bundle, summary = train_unary_models(payloads)
    summary.source_dir = json.dumps([str(p) for p in source_dirs], ensure_ascii=False)

    out_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / 'output' / 'unary_models' / f"bootstrap_gbdt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    save_unary_models(bundle, summary, out_dir)
    print(json.dumps({
        'event': 'unary_models_trained',
        'source_dirs': [str(p) for p in source_dirs],
        'output_dir': str(out_dir),
        'visibility_samples': summary.visibility_head.sample_count if summary.visibility_head else 0,
        'state_samples': summary.state_head.sample_count if summary.state_head else 0,
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
