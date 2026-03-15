import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bend_unary_model import build_examples, extract_match_features, train_unary_models, score_case_payload


def _case_payload(scan_state='full', bend_id='CAD_B1', status='PASS', observability_state='OBSERVED_FORMED', extra=None):
    match = {
        'bend_id': bend_id,
        'bend_form': 'FOLDED',
        'target_angle': 90.0,
        'target_radius': 2.0,
        'measured_angle': 91.0,
        'measured_radius': 2.1,
        'angle_deviation': 1.0,
        'radius_deviation': 0.1,
        'confidence': 0.8,
        'observability_state': observability_state,
        'observability_confidence': 0.75,
        'visibility_score': 0.7,
        'surface_visibility_ratio': 1.0,
        'local_support_score': 0.6,
        'side_balance_score': 0.5,
        'assignment_source': 'CAD_LOCAL_NEIGHBORHOOD',
        'assignment_confidence': 0.8,
        'assignment_candidate_id': 'local_candidate_0',
        'assignment_candidate_kind': 'MEASUREMENT',
        'assignment_candidate_score': 0.82,
        'assignment_null_score': 0.14,
        'assignment_candidate_count': 2,
        'local_evidence_score': 0.72,
        'local_point_count': 120,
        'observed_surface_count': 2,
        'measurement_mode': 'cad_local_neighborhood',
        'measurement_method': 'probe_region',
        'measurement_context': {
            'assignment_candidates': [
                {
                    'candidate_id': 'local_candidate_0',
                    'candidate_kind': 'MEASUREMENT',
                    'assignment_score': 0.82,
                    'center_distance_mm': 4.0,
                    'cad_angle_gap_deg': 2.0,
                    'direction_alignment': 0.97,
                    'visibility_score': 0.7,
                    'evidence_score': 0.72,
                    'measurement_confidence_score': 0.9,
                    'measurement_success': True,
                },
                {
                    'candidate_id': 'null_candidate',
                    'candidate_kind': 'NULL',
                    'assignment_score': 0.14,
                },
            ]
        },
    }
    if extra:
        match.update(extra)
    return {
        'part_key': 'PART_A',
        'scan_name': f'{scan_state}_scan.ply',
        'expectation': {
            'state': scan_state,
            'expected_total_bends': 1,
            'expected_completed_bends': 1 if scan_state == 'full' else None,
        },
        'details': {
            'cad_strategy': 'detector_mesh',
            'scan_quality_status': 'GOOD',
            'scan_point_count': 1000,
            'scan_coverage_pct': 88.0,
            'scan_density_pts_per_cm2': 18.0,
        },
        'summary': {
            'completed_bends': 1 if status != 'NOT_DETECTED' else 0,
            'remaining_bends': 0 if status != 'NOT_DETECTED' else 1,
        },
        'matches': [{**match, 'status': status}],
    }


def test_extract_match_features_includes_candidate_summary():
    payload = _case_payload()
    features = extract_match_features(payload, payload['matches'][0])
    assert features['candidate_count'] == 2
    assert features['measurement_candidate_count'] == 1
    assert features['candidate_margin_to_null'] > 0.0
    assert features['selected_candidate_kind'] == 'MEASUREMENT'


def test_build_examples_generates_visibility_and_state_examples():
    payloads = [
        _case_payload(scan_state='full', status='PASS'),
        _case_payload(scan_state='partial', status='NOT_DETECTED', observability_state='UNOBSERVED', extra={
            'assignment_candidate_id': 'null_candidate',
            'assignment_candidate_kind': 'NULL',
            'assignment_candidate_score': 0.7,
            'assignment_null_score': 0.7,
            'visibility_score': 0.75,
            'measurement_context': {
                'assignment_candidates': [
                    {'candidate_id': 'null_candidate', 'candidate_kind': 'NULL', 'assignment_score': 0.7}
                ]
            },
        }),
    ]
    visibility_examples, state_examples = build_examples(payloads)
    assert len(visibility_examples) == 2
    assert {example.label for example in visibility_examples} == {0, 1}
    assert len(state_examples) == 2
    assert {example.label for example in state_examples} == {0, 1}


def test_train_and_score_unary_models_bootstrap():
    payloads = []
    for idx in range(6):
        payloads.append(_case_payload(scan_state='full', bend_id=f'CAD_P{idx}', status='PASS', extra={'confidence': 0.9, 'visibility_score': 0.85}))
    for idx in range(6):
        payloads.append(_case_payload(scan_state='partial', bend_id=f'CAD_N{idx}', status='NOT_DETECTED', observability_state='UNOBSERVED', extra={
            'assignment_candidate_id': 'null_candidate',
            'assignment_candidate_kind': 'NULL',
            'assignment_candidate_score': 0.72,
            'assignment_null_score': 0.72,
            'visibility_score': 0.72,
            'local_support_score': 0.58,
            'measurement_context': {'assignment_candidates': [{'candidate_id': 'null_candidate', 'candidate_kind': 'NULL', 'assignment_score': 0.72}]},
        }))
        payloads[-1]['part_key'] = f'PART_NEG_{idx%3}'
    bundle, summary = train_unary_models(payloads)
    assert bundle['visibility_head'] is not None
    assert bundle['state_head'] is not None
    assert summary.visibility_head is not None
    assert summary.state_head is not None
    assert not any('observability_state' in feature for feature in summary.visibility_head.features)

    annotated = score_case_payload(_case_payload(), bundle)
    preds = annotated['matches'][0]['unary_predictions']
    assert 0.0 <= preds['visibility_probability'] <= 1.0
    assert 0.0 <= preds['formed_probability'] <= 1.0


def test_score_case_payload_adds_confusable_atoms_and_hard_exclusivity():
    payload = _case_payload()
    payload['part_key'] = 'PART_EXCL_A'
    base_match = payload['matches'][0]
    base_match['bend_id'] = 'CAD_A'
    base_match['measurement_context']['assignment_candidates'][0]['measurement_index'] = 7
    competitor = {
        **base_match,
        'bend_id': 'CAD_B',
        'status': 'FAIL',
        'assignment_candidate_id': 'local_candidate_0',
        'assignment_candidate_kind': 'MEASUREMENT',
        'assignment_candidate_score': 0.81,
        'measurement_context': {
            'assignment_candidates': [
                {
                    'candidate_id': 'local_candidate_0',
                    'candidate_kind': 'MEASUREMENT',
                    'measurement_index': 7,
                    'assignment_score': 0.81,
                    'measurement_success': True,
                    'measurement_confidence_score': 0.8,
                },
                {
                    'candidate_id': 'null_candidate',
                    'candidate_kind': 'NULL',
                    'assignment_score': 0.12,
                },
            ]
        },
    }
    payload['matches'].append(competitor)

    negatives = []
    for idx in range(6):
        neg = _case_payload(
            scan_state='partial',
            bend_id=f'CAD_NEG{idx}',
            status='NOT_DETECTED',
            observability_state='PARTIALLY_OBSERVED',
            extra={
                'assignment_candidate_id': 'null_candidate',
                'assignment_candidate_kind': 'NULL',
                'assignment_candidate_score': 0.7,
                'assignment_null_score': 0.7,
                'visibility_score': 0.7,
                'local_support_score': 0.4,
                'local_point_count': 120,
                'measurement_context': {
                    'assignment_candidates': [{'candidate_id': 'null_candidate', 'candidate_kind': 'NULL', 'assignment_score': 0.7}]
                },
            },
        )
        neg['part_key'] = f'PART_NEG_{idx}'
        negatives.append(neg)

    positives = []
    for idx in range(6):
        pos = _case_payload(scan_state='full', bend_id=f'CAD_POS{idx}', status='PASS')
        pos['part_key'] = f'PART_POS_{idx}'
        positives.append(pos)

    bundle, _summary = train_unary_models([payload, *negatives, *positives])
    annotated = score_case_payload(payload, bundle)

    assert 'structured_context' in annotated
    assert annotated['structured_context']['confusable_group_count'] >= 1
    assert len(annotated['structured_context']['confusable_atoms']) >= 1
    rejected = [m for m in annotated['matches'] if m['structured_predictions']['exclusive_rejected']]
    kept = [m for m in annotated['matches'] if not m['structured_predictions']['exclusive_rejected']]
    assert len(rejected) == 1
    assert len(kept) == 1
    count_posterior = annotated['structured_context']['count_posterior']
    assert abs(sum(count_posterior['count_distribution']) - 1.0) < 1e-5
    assert 0 <= count_posterior['median_completed_bends'] <= len(annotated['matches'])
    assert 'joint_confusable_components' in annotated['structured_context']
    assert all('joint_exclusive_formed_probability' in m['structured_predictions'] for m in annotated['matches'])


def test_joint_candidate_exclusivity_rejects_lower_scoring_competitor():
    payload = _case_payload()
    base_match = payload['matches'][0]
    base_match['bend_id'] = 'CAD_A'
    base_match['measurement_context']['assignment_candidates'] = [
        {
            'candidate_id': 'local_candidate_a',
            'candidate_kind': 'MEASUREMENT',
            'measurement_index': 3,
            'assignment_score': 0.92,
            'evidence_score': 0.88,
            'visibility_score': 0.8,
            'measurement_confidence_score': 0.9,
            'measurement_success': True,
        },
        {
            'candidate_id': 'null_candidate',
            'candidate_kind': 'NULL',
            'assignment_score': 0.1,
        },
    ]
    competitor = {
        **base_match,
        'bend_id': 'CAD_B',
        'assignment_candidate_id': 'local_candidate_b',
        'assignment_candidate_score': 0.6,
        'measurement_context': {
            'assignment_candidates': [
                {
                    'candidate_id': 'local_candidate_b',
                    'candidate_kind': 'MEASUREMENT',
                    'measurement_index': 3,
                    'assignment_score': 0.6,
                    'evidence_score': 0.55,
                    'visibility_score': 0.7,
                    'measurement_confidence_score': 0.7,
                    'measurement_success': True,
                },
                {
                    'candidate_id': 'null_candidate',
                    'candidate_kind': 'NULL',
                    'assignment_score': 0.3,
                },
            ]
        },
    }
    payload['matches'] = [base_match, competitor]

    negatives = []
    for idx in range(6):
        neg = _case_payload(
            scan_state='partial',
            bend_id=f'CAD_NEGJ{idx}',
            status='NOT_DETECTED',
            observability_state='PARTIALLY_OBSERVED',
            extra={
                'assignment_candidate_id': 'null_candidate',
                'assignment_candidate_kind': 'NULL',
                'assignment_candidate_score': 0.8,
                'assignment_null_score': 0.8,
                'visibility_score': 0.75,
                'local_support_score': 0.45,
                'local_point_count': 90,
                'measurement_context': {'assignment_candidates': [{'candidate_id': 'null_candidate', 'candidate_kind': 'NULL', 'assignment_score': 0.8}]},
            },
        )
        neg['part_key'] = f'PART_NEGJ_{idx}'
        negatives.append(neg)
    positives = []
    for idx in range(6):
        pos = _case_payload(scan_state='full', bend_id=f'CAD_POSJ{idx}', status='PASS')
        pos['part_key'] = f'PART_POSJ_{idx}'
        positives.append(pos)

    bundle, _summary = train_unary_models([payload, *negatives, *positives])
    annotated = score_case_payload(payload, bundle)
    by_bend = {m['bend_id']: m for m in annotated['matches']}
    assert by_bend['CAD_A']['structured_predictions']['joint_exclusive_formed_probability'] > 0.0
    assert by_bend['CAD_B']['structured_predictions']['joint_exclusive_formed_probability'] == 0.0
    assert by_bend['CAD_B']['structured_predictions']['joint_exclusive_rejected'] is True
