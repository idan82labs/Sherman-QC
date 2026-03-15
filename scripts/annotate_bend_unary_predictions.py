#!/opt/homebrew/bin/python3.11
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path('/Users/idant/82Labs/Sherman QC/Full Project')
sys.path.insert(0, str(ROOT / 'backend'))

from bend_unary_model import load_unary_models, score_case_payload  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='Annotate bend corpus case jsons with unary model predictions.')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--input-json', required=True)
    parser.add_argument('--output-json', required=True)
    args = parser.parse_args()

    bundle = load_unary_models(args.model_path)
    payload = json.loads(Path(args.input_json).read_text(encoding='utf-8'))
    annotated = score_case_payload(payload, bundle)
    Path(args.output_json).write_text(json.dumps(annotated, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'event': 'annotated', 'output_json': args.output_json}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
