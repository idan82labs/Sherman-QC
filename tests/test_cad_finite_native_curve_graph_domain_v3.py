import copy
import unittest

from backend.cad_finite_native_curve_graph_domain_v3 import certify_native_curve_graph_domain


def polygon(points):
    result = []
    for i, start in enumerate(points):
        end = points[(i+1) % len(points)]
        result.append(dict(native_edge_id=i+1, native_start_vertex_id=i+1,
                           native_end_vertex_id=(i+1) % len(points)+1,
                           kernel_vertex_tolerance_mm=1e-7, poles=[list(start), list(end)], weights=[1., 1.],
                           basis=dict(kind="LINE", degree=1, actual_interval_proven=True,
                                      actual_interval=[0., 1.], restricted_interval=[0., 1.], periodic=False)))
    return result


def certify(curves, **overrides):
    settings = dict(face_tolerance_mm=1e-7, native_brep_valid=True,
                    complete_outer_wire=True, chart_branch_proven=True, holes_present=False)
    settings.update(overrides)
    return certify_native_curve_graph_domain(curves, **settings)


class BoundaryChainTests(unittest.TestCase):
    def test_rectangle_preserves_nominal_uncertainty(self):
        curves = polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        saved = copy.deepcopy(curves)
        proof = certify(curves)
        self.assertEqual(proof["status"], "PROVED")
        self.assertTrue(proof["detection_only"])
        self.assertFalse(proof["unfold_certificate"])
        self.assertEqual(curves, saved)
        self.assertEqual(proof["kernel_vertex_uncertainty_retained"], [1e-7]*4)

    def test_multiple_edge_end_and_rail_chains(self):
        proof = certify(polygon([(0, 0), (5, 0), (10, 0), (10, 4), (10, 10), (0, 10)]))
        self.assertEqual(proof["status"], "PROVED")
        self.assertEqual(len(proof["safe_angular_components_metric_mm"]), 2)
        self.assertEqual(len(proof["native_rail_edge_ids"]), 3)

    def test_vertical_internal_connector_excluded(self):
        proof = certify(polygon([(0, 0), (5, 0), (5, 1), (10, 1), (10, 10), (0, 10)]))
        self.assertEqual(proof["status"], "PROVED")
        components = proof["safe_angular_components_metric_mm"]
        self.assertEqual(len(components), 2)
        self.assertTrue(all(not a <= 5 <= b for a, b in components))
        self.assertGreater(proof["safe_rectangular_interior_core_mm"][2], 1)

    def test_cyclic_rotation_and_full_orientation_reversal(self):
        points = [(0, 0), (5, 0), (10, 0), (10, 10), (0, 10)]
        for p in [points, points[2:]+points[:2], list(reversed(points))]:
            self.assertEqual(certify(polygon(p))["status"], "PROVED")

    def test_end_backtrack_refused(self):
        curves = polygon([(0, 0), (6, 0), (4, 1), (10, 1), (10, 10), (0, 10)])
        self.assertEqual(certify(curves)["reason"], "END_CHAIN_ANGULAR_BACKTRACK")

    def test_shared_vertex_identity_and_large_gap(self):
        curves = polygon([(0, 0), (5, 0), (10, 0), (10, 10), (0, 10)])
        curves[1]["native_start_vertex_id"] = 99
        self.assertEqual(certify(curves)["reason"], "SHARED_NATIVE_VERTEX_IDENTITY_REQUIRED")
        curves = polygon([(0, 0), (5, 0), (10, 0), (10, 10), (0, 10)])
        curves[1]["poles"][0][0] += 1e-4
        self.assertEqual(certify(curves)["reason"], "MEASURED_CHART_JOIN_EXCEEDS_FACE_TOLERANCE")

    def test_small_internal_gap_is_excluded_not_filled(self):
        curves = polygon([(0, 0), (5, 0), (10, 0), (10, 10), (0, 10)])
        curves[1]["poles"][0][0] += 5e-8
        proof = certify(curves)
        self.assertEqual(proof["status"], "PROVED")
        self.assertTrue(all(b < 5 or a > 5+5e-8 for a, b in proof["safe_angular_components_metric_mm"]))

    def test_nonmonotone_rational_controls_and_bad_knots(self):
        for bad in ["controls", "knots", "weights", "active"]:
            curves = polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
            c = curves[0]
            c["poles"] = [[0., 0.], [5., 0.], [10., 0.]]
            c["weights"] = [1., 2., 1.]
            c["basis"].update(kind="BSPLINE", degree=2, knots=[0., 1.], multiplicities=[3, 3])
            if bad == "controls": c["poles"][1][0] = 11.
            if bad == "knots": c["basis"]["multiplicities"] = [2, 3]
            if bad == "weights": c["weights"][1] = 0.
            if bad == "active": c["basis"]["restricted_interval"] = [0., .9]
            with self.subTest(bad=bad): self.assertEqual(certify(curves)["status"], "REFUSED")

    def test_valid_original_rational_bspline(self):
        curves = polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        curves[0]["poles"] = [[0., 0.], [5., 1.], [10., 0.]]
        curves[0]["weights"] = [1., 2., 1.]
        curves[0]["basis"].update(kind="BSPLINE", degree=2, knots=[0., 1.], multiplicities=[3, 3])
        self.assertEqual(certify(curves)["status"], "PROVED")

    def test_rail_chain_backtrack(self):
        proof = certify(polygon([(0, 0), (10, 0), (10, 7), (10, 5), (10, 10), (0, 10)]))
        self.assertEqual(proof["reason"], "RAIL_CHAIN_BACKTRACK")

    def test_shortened_end_endpoint_never_extended(self):
        curves = polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        curves[0]["poles"][-1][0] -= 5e-8
        proof = certify(curves)
        self.assertEqual(proof["status"], "PROVED")
        self.assertLess(proof["safe_rectangular_interior_core_mm"][1], 10-5e-8)

    def test_rail_join_backtrack_is_not_declared_monotone(self):
        curves = polygon([(0, 0), (10, 0), (10, 5), (10, 10), (0, 10)])
        curves[2]["poles"][0][1] -= 5e-8
        self.assertEqual(certify(curves)["reason"], "RAIL_CHAIN_JOIN_BACKTRACK")

    def test_holes_flags_budgets_and_nonfinite(self):
        curves = polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        self.assertEqual(certify(curves, holes_present=True)["reason"], "HOLES_UNSUPPORTED")
        self.assertEqual(certify(curves, native_brep_valid=False)["status"], "REFUSED")
        self.assertEqual(certify(curves*17)["reason"], "NATIVE_EDGE_COUNT_BUDGET")
        curves[0]["poles"][1][0] = float("nan")
        self.assertEqual(certify(curves)["reason"], "NONFINITE_COEFFICIENT")


if __name__ == "__main__":
    unittest.main()
