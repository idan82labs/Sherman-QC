"""Freeze conservative consensus from preserved independent CAD reviews."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.cad_bend_reference_scoring_v1 import score_parts


def load(path, pin):
    raw = Path(path).read_bytes()
    if len(raw) > 40_000_000 or hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError('Reference review hash mismatch')
    return json.loads(raw)


def group(bend):
    return frozenset(bend['inner_face_ids'] + bend['outer_face_ids'])


def freeze(directory, r1_pin, r2_pin, a1_pin, a2_pin, final_pin):
    r1 = load(directory/'reviewer-1-initial-valid.json', r1_pin)
    r2 = load(directory/'reviewer-2.json', r2_pin)
    a1 = load(directory/'reviewer-1-adjudication.json', a1_pin)
    a2 = load(directory/'reviewer-2-adjudication.json', a2_pin)
    final = load(directory/'reviewer-1-completeness-final.json', final_pin)
    expected = {f'case-{i:02d}' for i in range(1,21)}
    for review in (r1, r2):
        if len(review['parts']) != 20 or {p['part_id'] for p in review['parts']} != expected:
            raise ValueError('Each independent review must cover exactly all 20 cases')
        score_parts([{**p, 'predictions': []} for p in review['parts']])
    complete1 = set(final['complete_parts'])
    partial1 = set(final['remaining_partial_parts'])
    complete2 = set(a2['proposed_complete_parts'])
    partial2 = set(a2['remaining_partial_parts'])
    if complete1 & partial1 or complete2 & partial2 or complete1 | partial1 != expected or complete2 | partial2 != expected:
        raise ValueError('Explicit completeness census required from both reviewers')
    # Explicitly agreed role resolutions only. A recommendation by one reviewer
    # is insufficient to turn an uncertain face into a negative.
    proposed = {x['part_id']: set(x['face_ids']) for x in a2['nonbend_resolutions']}
    accepted = {x['part_id']: set(x['accepted_nonbend_face_ids']) for x in final['decisions']}
    parts = []
    for second in r2['parts']:
        name = second['part_id']
        first = next(p for p in r1['parts'] if p['part_id'] == name)
        if first['source_sha256'] != second['source_sha256'] or set(first['face_ids']) != set(second['face_ids']):
            raise ValueError('Independent review input disagreement')
        if first['inputevidence_sha256'] != second['evidence_sha256']:
            raise ValueError('Neutral evidence hash disagreement')
        known_first = {group(b) for b in first['reference_bends']}
        for decision in a1['decisions']:
            if decision['part_id'] == name and decision['decision'].startswith('accept_positive'):
                known_first.update(group(b) for b in decision['groups'])
        known_second = {group(b) for b in second['reference_bends']}
        if known_first != known_second:
            raise ValueError('Positive groups not fully adjudicated: ' + name)
        positive_faces = set().union(*known_second) if known_second else set()
        resolved = accepted.get(name, set())
        if not resolved <= proposed.get(name, set()) or resolved & positive_faces:
            raise ValueError('Nonbend resolution not agreed or overlaps known positive')
        unknown = (set(first['unknown_face_ids']) | set(second['unknown_face_ids'])) - positive_faces - resolved
        # Include explicitly conservative downgrades recorded during adjudication.
        for decision in a1['decisions']:
            if decision['part_id'] == name:
                unknown |= set(decision.get('unknown_face_ids', [])) - positive_faces - resolved
        faces = set(second['face_ids'])
        if not unknown <= faces:
            raise ValueError('Unknown face outside census')
        complete = name in complete1 and name in complete2
        if complete and unknown or not complete and not unknown:
            raise ValueError('Completeness assertion and unknown witness disagree: ' + name)
        bends = []
        for index, bend in enumerate(second['reference_bends'], 1):
            bends.append({'bend_id': f'{name}-B{index:02d}',
                          'inner_face_ids': bend['inner_face_ids'], 'outer_face_ids': bend['outer_face_ids'],
                          'reviewed_native_face_group': sorted(group(bend))})
        parts.append({'part_id': name, 'source_sha256': second['source_sha256'],
                      'evidence_sha256': second['evidence_sha256'],
                      'face_ids': sorted(faces), 'reference_status': 'complete' if complete else 'partial',
                      'reference_bends': bends, 'unknown_face_ids': sorted(unknown),
                      'verified_nonbend_face_ids': sorted(faces-positive_faces-unknown),
                      'reference_rationale': 'Both reviewers agreed geometric positive identities; unresolved role union retained except explicit two-reviewer nonbend resolutions.'})
    if {p['part_id'] for p in parts} != {f'case-{i:02d}' for i in range(1,21)} or len(parts) != 20:
        raise ValueError('Incomplete reference census')
    score_parts([{**p, 'predictions': []} for p in parts])
    payload = {'reference_frozen': True, 'independent_reviews_complete': True,
               'reference_kind': 'AI technical geometric CAD reference, not human manufacturing truth',
               'exposure': 'Historical narratives included limited counts and face IDs; detailed current prediction artifacts withheld. Not a strictly blind trial.',
               'ontology': 'Connected finite geometric sheet-turn regions; disconnected strips separate even if one press operation; rolled regions included.',
               'review_hashes': {'reviewer-1-initial-valid.json': r1_pin, 'reviewer-2.json': r2_pin,
                                'reviewer-1-adjudication.json': a1_pin, 'reviewer-2-adjudication.json': a2_pin,
                                'reviewer-1-completeness-final.json': final_pin},
               'consensus_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'parts': sorted(parts, key=lambda p:p['part_id'])}
    with (directory/'REFERENCE_FROZEN.json').open('x') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    for name in ('r1', 'r2', 'a1', 'a2', 'final'):
        p.add_argument('--'+name+'-sha256', required=True)
    a = p.parse_args()
    freeze(a.directory, a.r1_sha256, a.r2_sha256, a.a1_sha256, a.a2_sha256, a.final_sha256)
