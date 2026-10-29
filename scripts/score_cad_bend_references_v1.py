"""Immutable derived-only comparison of frozen technical CAD labels and predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.cad_bend_reference_scoring_v1 import score_parts
from backend.cad_bend_reference_bounds_v1 import metric_bounds


def read(path, pin):
    if not isinstance(pin, str) or len(pin) != 64 or any(c not in '0123456789abcdef' for c in pin):
        raise ValueError('Exact SHA-256 pin required')
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 40_000_000:
        raise ValueError('Unsafe or oversized derived input')
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != pin:
        raise ValueError('Input hash mismatch: ' + str(path))
    return json.loads(data), digest


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def pins():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
        'scripts/score_cad_bend_references_v1.py', 'backend/cad_bend_reference_scoring_v1.py',
        'scripts/run_cad_reference_score_v1.py', 'scripts/evaluate_vendor_cad_upload_unfold_v2.py',
        'tests/test_score_cad_bend_references_v1.py', 'tests/test_cad_bend_reference_scoring_v1.py',
        'backend/cad_bend_reference_bounds_v1.py', 'tests/test_cad_bend_reference_bounds_v1.py',
        'docs/operations/cad-reference-partial-bounds-v1-2026-09-30.md')}


def freeze(out, reference, reference_sha, predictions, post_sha):
    refs, _ = read(reference, reference_sha)
    post, _ = read(predictions/'POST_AUDIT.json', post_sha)
    if post.get('decision') != 'ACCEPT_DIAGNOSTIC_WITH_LIMITATIONS' or post.get('independent_non_author') is not True or post.get('blocking_findings'):
        raise ValueError('Prediction release invalid')
    candidate, candidate_sha = read(predictions/'FROZEN.json', post['frozen_sha256'])
    parts = refs['parts']
    if len(parts) != 20 or len(candidate['cases']) != 20 or len({c['sha256'] for c in candidate['cases']}) != 20 or {p['part_id'] for p in parts} != {f'case-{i:02d}' for i in range(1,21)}:
        raise ValueError('Complete 20-case census required')
    if refs.get('reference_frozen') is not True or refs.get('independent_reviews_complete') is not True:
        raise ValueError('Reference freeze/reviews missing')
    for i, c in enumerate(candidate['cases'], 1):
        p = next(p for p in parts if p['part_id'] == f'case-{i:02d}')
        if p['source_sha256'] != c['sha256'] or 'predictions' in p:
            raise ValueError('Reference input mismatch or predictions leaked into reference')
    score_parts([{**p, 'predictions': []} for p in parts])  # Validate reference schema only.
    out.mkdir(parents=False, exist_ok=False)
    write(out/'FROZEN.json', {
        'reference': str(reference.resolve()), 'reference_sha256': reference_sha,
        'predictions_attempt': str(predictions.resolve()), 'prediction_post_sha256': post_sha,
        'candidate_frozen_sha256': candidate_sha,
        'source_pins': pins(), 'part_denominator': 20,
        'matching': 'exact_native_face_union_one_to_one',
        'claims': 'CAD technical detection only; not physical Gold F1 or manufacturing truth',
        'seconds': 30, 'rss_kib': 1_048_576, 'raw_access': False,
    })


def execute(out, frozen_sha):
    frozen, _ = read(out/'FROZEN.json', frozen_sha)
    if frozen['source_pins'] != pins():
        raise ValueError('Scoring source changed')
    refs, _ = read(frozen['reference'], frozen['reference_sha256'])
    directory = Path(frozen['predictions_attempt'])
    post, _ = read(directory/'POST_AUDIT.json', frozen['prediction_post_sha256'])
    read(directory/'FROZEN.json', frozen['candidate_frozen_sha256'])
    write(out/'EXECUTION.json', {'frozen_sha256': frozen_sha})
    try:
        parts = []
        evidence = post['evidence']['derived_artifact_sha256']
        for ref in sorted(refs['parts'], key=lambda p: p['part_id']):
            name = ref['part_id']
            result, _ = read(directory/name/'RESULT.json', evidence[name+'/RESULT.json'])
            if result['source_sha256'] != ref['source_sha256']:
                raise ValueError('Case source mismatch')
            if result['extraction_status'] == 'COMPLETE':
                extraction, _ = read(directory/name/'EXTRACTION.json', evidence[name+'/EXTRACTION.json'])
                predictions = [{'prediction_id': row['feature_id'], 'face_ids': row['face_indices']} for row in extraction['bend_axes']]
                status, reason = 'ok', None
            else:
                predictions, status, reason = [], 'failed', result.get('extraction_reason', 'extraction refused')
            parts.append({**ref, 'predictions': predictions, 'prediction_status': status, 'failure_reason': reason})
        score = score_parts(parts)
        known = score['all_verified_positive_recall']
        unscored = sum(len(p['unscored_prediction_ids']) for p in score['parts'])
        bound_counts = dict(matched=known['matched'],
                            known_missed=known['reference_denominator']-known['matched'],
                            complete_false_positive=score['full_reference_micro']['fp'],
                            unscored_predictions=unscored,
                            all_complete=score['coverage']['fully_referenced_parts'] == 20)
        if sum(p['prediction_count'] for p in score['parts']) != bound_counts['matched']+bound_counts['complete_false_positive']+unscored:
            raise ValueError('Bound prediction partition invalid')
        score['whole_corpus_optimistic_bounds'] = metric_bounds(**bound_counts)
        score['bound_counts'] = bound_counts
        score.update({'reference_sha256': frozen['reference_sha256'], 'frozen_sha256': frozen_sha,
                      'prediction_post_sha256': frozen['prediction_post_sha256'],
                      'reference_kind': 'independently reviewed AI technical CAD labels, not human manufacturing approval',
                      'whole_corpus_fully_referenced': score['coverage']['fully_referenced_parts'] == 20})
        write(out/'SCORE.json', score)
        write(out/'TERMINAL.json', {'status': 'COMPLETE', 'score_sha256': hashlib.sha256((out/'SCORE.json').read_bytes()).hexdigest()})
    except Exception as exc:
        write(out/'TERMINAL.json', {'status': 'FAILED', 'reason': type(exc).__name__ + ':' + str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze', action='store_true'); action.add_argument('--execute', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path); parser.add_argument('--reference-sha256')
    parser.add_argument('--predictions', type=Path); parser.add_argument('--post-sha256')
    parser.add_argument('--frozen-sha256')
    args = parser.parse_args()
    if args.freeze:
        freeze(args.out, args.reference, args.reference_sha256, args.predictions, args.post_sha256)
    else:
        execute(args.out, args.frozen_sha256)
