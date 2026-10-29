from research.f1_cad_contextual_assignment_v1 import AssignmentConfig, assign, fuse


def line(name, y=0., x0=0., x1=40.):
    return {"bend_id": name, "line_start": [x0, y, 0.], "line_end": [x1, y, 0.]}


def test_assignment_is_one_to_one_and_conflicting_cad_event_becomes_null():
    cad = [line("C1", .5), line("C2", 1.)]
    scan = [line("S1")]
    fused, audit = fuse(cad, scan, AssignmentConfig())
    assert [row["bend_id"] for row in fused] == ["S1"]
    assert sum(row["decision"] == "SCAN_OWNS_ASSIGNED_EVENT" for row in audit["decisions"]) == 1
    assert sum(row["decision"] == "NULL_ONE_TO_ONE_CONFLICT" for row in audit["decisions"]) == 1


def test_unmatched_visible_cad_event_is_admitted():
    cad = [line("C1", 30.)]
    scan = [line("S1")]
    fused, audit = fuse(cad, scan, AssignmentConfig())
    assert [row["bend_id"] for row in fused] == ["S1", "CAD_C1"]
    assert audit["decisions"][0]["decision"] == "UNMATCHED_CAD_EVENT_ADMITTED"


def test_contextual_assignment_is_deterministic():
    cad = [line("C1", .5), line("C2", 15.)]
    scan = [line("S1"), line("S2", 15.5)]
    first = assign(cad, scan, AssignmentConfig())
    second = assign(cad, scan, AssignmentConfig())
    assert first == second
