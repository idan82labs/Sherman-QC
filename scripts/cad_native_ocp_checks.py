"""Fixed, in-memory OCP checks. No CAD reader, file argument or app import.

Run only inside check_cad_native_provenance.py's bounded isolated worker.
Matrix: 3 dimensions x 4 proper poses x 3 positive scales x 2 orientations
= 72 boxes; plus 12 located two-instance and 4 repeated-occurrence controls.
Mesh comparison is capture-on/off within OCP, NOT OCP versus gmsh/PythonOCC.
"""
import ast
from collections import Counter
from dataclasses import dataclass, field
import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple


def run_checks():
    import numpy as np
    import OCP
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.TopoDS import TopoDS_Compound
    from OCP.TopLoc import TopLoc_Location
    from OCP.gp import gp_Trsf

    source = Path(__file__).resolve().parents[1] / 'backend' / 'cad_import.py'
    tree = ast.parse(source.read_text())
    names = {'_select_native_cad_binding', '_tessellation_provenance', 'CADMesh', 'CADImporter'}
    nodes = [n for n in tree.body if
             isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith('MAX_') for t in n.targets)]
    methods = {'__init__', '_check_occ_available', '_load_occ', '_load_ocp', '_tessellate',
               '_capture_native_topology', '_calculate_face_normals'}
    for node in nodes:
        if isinstance(node, ast.ClassDef) and node.name == 'CADImporter':
            node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in methods]
            assert {item.name for item in node.body} == methods
    ns = dict(__name__=__name__, np=np, dataclass=dataclass, field=field, Any=Any,
              Dict=Dict, List=List, Optional=Optional, Tuple=Tuple, Path=Path,
              logger=SimpleNamespace(warning=lambda *a: None))
    # Only called methods are compiled; no reader/export method or app import.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    cls = ns['CADImporter']
    rotations = [np.eye(3), np.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]]),
                 np.array([[.6, -.8, 0], [.8, .6, 0], [0, 0, 1]]),
                 np.array([[-2., 2, 1], [2, 1, 2], [1, 2, -2]]) / 3]
    translations = [np.array(t, float) for t in [(0, 0, 0), (7, -11, 13), (-31, 17, 5), (100, -200, 300)]]
    counts = Counter()
    maximum_corner_error = 0.0

    def location(rotation, translation):
        trsf = gp_Trsf()
        trsf.SetValues(*[float(v) for row, off in zip(rotation, translation) for v in (*row, off)])
        return TopLoc_Location(trsf)

    def compound(children):
        builder, value = BRep_Builder(), TopoDS_Compound()
        builder.MakeCompound(value)
        for child in children:
            builder.Add(value, child)
        return value

    def check(shape, corners, boxes, owners, available=True):
        nonlocal maximum_corner_error
        active = cls(allow_ocp=True)
        active._load_occ()
        assert active.native_binding == 'ocp', 'fixture requires the installed OCP path'
        baseline = cls(allow_ocp=True)
        baseline._load_occ()
        baseline._native_topology_api_available = False
        old, new = baseline._tessellate(shape), active._tessellate(shape)
        for field_name in ('vertices', 'faces', 'normals'):
            assert np.array_equal(getattr(old, field_name), getattr(new, field_name))
        assert old.tessellation_provenance == new.tessellation_provenance
        assert old.native_topology['status'] == 'UNAVAILABLE'
        provenance = new.tessellation_provenance
        assert provenance['mapping_status'] == 'FACE_OCCURRENCE_RANGES'
        assert provenance['source_face_count'] == 6 * owners
        vstop = tstop = 0
        for record in provenance['face_records']:
            start, stop = record['vertex_range']
            first, last = record['triangle_range']
            assert start == vstop and first == tstop and stop > start and last > first
            assert np.all((new.faces[first:last] >= start) & (new.faces[first:last] < stop))
            vstop, tstop = stop, last
        assert (vstop, tstop) == (len(new.vertices), len(new.faces))
        # Independently transformed eight analytic corners, no kernel bbox oracle.
        error = max(float(np.max(np.abs(new.vertices.min(0) - corners.min(0)))),
                    float(np.max(np.abs(new.vertices.max(0) - corners.max(0)))))
        maximum_corner_error = max(maximum_corner_error, error)
        bound = 128 * np.finfo(float).eps * max(1., float(np.max(np.abs(corners))))
        assert error <= bound, 'transformed corner mismatch'
        covered = np.zeros(len(new.vertices), dtype=bool)
        for rotation, translation, lengths in boxes:
            local = (new.vertices - translation) @ rotation
            inside = np.all((local >= -bound) & (local <= lengths + bound), axis=1)
            on_boundary = np.any((np.abs(local) <= bound) | (np.abs(local - lengths) <= bound), axis=1)
            covered |= inside & on_boundary
        assert covered.all(), 'emitted vertex off all constructed box boundaries'
        top = new.native_topology
        if available:
            assert top['status'] == 'CAPTURED_IMPORT_LOCAL_INCIDENCE', top['reasons']
            assert len(top['face_records']) == 6 * owners and len(top['owner_records']) == owners
            assert all(o['kind'] == 'SOLID' for o in top['owner_records'])
            edge_counts = Counter()
            edge_orientations = {}
            owner_faces = Counter()
            for face in top['face_records']:
                owner_faces[face['owner_id']] += 1
                assert len(face['edge_uses']) == 4
                for use in face['edge_uses']:
                    assert use['edge_id'].startswith(face['owner_id'] + ':edge:')
                    assert use['orientation'] in ('FORWARD', 'REVERSED') and use['degenerate'] is False
                    edge_counts[use['edge_id']] += 1
                    edge_orientations.setdefault(use['edge_id'], []).append(use['orientation'])
            assert len(edge_counts) == 12 * owners and set(edge_counts.values()) == {2}
            assert set(owner_faces.values()) == {6} and len(owner_faces) == owners
            assert all(sorted(uses) == ['FORWARD', 'REVERSED'] for uses in edge_orientations.values())
            assert all(sum(edge.startswith(owner + ':edge:') for edge in edge_counts) == 12 for owner in owner_faces)
            counts['captured_cases'] += 1
        else:
            assert top['status'] == 'UNAVAILABLE' and not top['face_records'] and not top['owner_records']
            assert top['reasons'] == ['AMBIGUOUS_REPEATED_NATIVE_FACE_OCCURRENCE']
            counts['expected_ambiguity_abstentions'] += 1
        counts['paired_geometry_cases'] += 1
        counts['face_occurrences_checked'] += 6 * owners

    for dims, ri, scale, reversed_ in itertools.product(
            [(10., 20., 30.), (1., 1., 1.), (.001, 10., 100.)], range(4), [.125, 1., 8.], [False, True]):
        lengths = np.array(dims) * scale
        corners = np.array(list(itertools.product(*[(0., x) for x in lengths])))
        rotation, translation = rotations[ri], translations[ri] * scale
        assert np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-15)
        assert abs(np.linalg.det(rotation) - 1) < 1e-15
        shape = BRepPrimAPI_MakeBox(*lengths).Shape().Located(location(rotation, translation))
        if reversed_:
            shape = shape.Reversed()
        check(shape, corners @ rotation.T + translation, [(rotation, translation, lengths)], 1)

    for ri, scale in itertools.product(range(4), [.125, 1., 8.]):
        lengths = np.array([10., 20., 30.]) * scale
        corners = np.array(list(itertools.product(*[(0., x) for x in lengths])))
        base = BRepPrimAPI_MakeBox(*lengths).Shape()
        child_offsets = [np.array([0., 0., 0.]), np.array([70., -90., 110.]) * scale]
        shape = compound([base.Located(location(np.eye(3), t)) for t in child_offsets])
        r, t = rotations[ri], translations[ri] * scale
        shape = shape.Located(location(r, t))
        expected = np.concatenate([(corners + child) @ r.T + t for child in child_offsets])
        check(shape, expected, [(r, r @ child + t, lengths) for child in child_offsets], 2)

    for ri in range(4):
        base = BRepPrimAPI_MakeBox(10., 20., 30.).Shape()
        shape = compound([compound([base]), compound([base])]).Located(location(rotations[ri], translations[ri]))
        corners = np.array(list(itertools.product((0., 10.), (0., 20.), (0., 30.))))
        check(shape, corners @ rotations[ri].T + translations[ri],
              [(rotations[ri], translations[ri], np.array([10., 20., 30.]))], 2, False)
    assert counts['paired_geometry_cases'] == 88
    return dict(counts, ocp_version=OCP.__version__, maximum_corner_error=maximum_corner_error,
                source='in-memory native boxes only', file_readers_exercised=False,
                cross_binding_geometry_equivalence=False, grouping_eligible=False)
