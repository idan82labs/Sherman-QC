from backend.f1_line_object_clustering import cluster_line_candidates


def _line(identifier, y, *, source, support):
    return {
        "bend_id": identifier,
        "span_endpoints": [[0.0, y, 0.0], [100.0, y, 0.0]],
        "candidate_source": source,
        "finite_support_fraction": support,
        "finite_support_longest_run_fraction": support,
        "endpoint_support_fraction": support,
    }


def test_cross_source_duplicates_form_one_line_object():
    objects = cluster_line_candidates(
        [
            _line("A", 0.0, source="owned", support=0.8),
            _line("B", 1.0, source="cylinder", support=1.0),
            _line("C", 20.0, source="plane", support=1.0),
        ]
    )

    assert len(objects) == 2
    assert objects[0]["representative_source"] == "cylinder"
    assert objects[0]["cluster_member_count"] == 2
    assert objects[0]["cluster_source_count"] == 2
    assert objects[1]["cluster_member_count"] == 1


def test_line_object_clustering_is_deterministic_under_input_order():
    lines = [
        _line("A", 0.0, source="owned", support=0.8),
        _line("B", 1.0, source="cylinder", support=1.0),
    ]

    assert cluster_line_candidates(lines) == cluster_line_candidates(list(reversed(lines)))
