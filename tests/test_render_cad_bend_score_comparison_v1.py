import unittest
from scripts.render_cad_bend_score_comparison_v1 import classify_part, marker_line


def prediction(name, faces):
    return {"prediction_id": name, "face_ids": faces}


def fixture():
    return {"reference_status": "partial", "unknown_face_ids": [7],
            "reference_bends": [{"bend_id": name, "inner_face_ids": [a], "outer_face_ids": [b]}
                                for name, a, b in [("a", 1, 2), ("b", 3, 4), ("c", 5, 6)]],
            "baseline": {"status": "ok", "predictions": [prediction("old-a", [1, 2])]},
            "challenger": {"status": "ok", "predictions": [prediction("new-a", [2, 1]),
                                                                  prediction("new-b", [3, 4]),
                                                                  prediction("unknown", [7])]}}


class ComparisonTests(unittest.TestCase):
    def test_retained_recovered_missing_and_unknown(self):
        result = classify_part(fixture(), set(range(1, 8)))
        self.assertEqual(result["counts"], {"retained": 1, "recovered": 1, "missing": 1})
        self.assertEqual(result["stages"]["challenger"]["unmatched"][0]["category"], "unknown_prediction")

    def test_duplicate_one_to_one(self):
        part = fixture()
        part["challenger"]["predictions"].append(prediction("duplicate", [1, 2]))
        result = classify_part(part, set(range(1, 8)))
        self.assertEqual(result["counts"]["retained"], 1)
        self.assertEqual(len(result["stages"]["challenger"]["unmatched"]), 2)

    def test_failure_is_all_missing(self):
        part = fixture()
        part["challenger"] = {"status": "failed", "predictions": []}
        self.assertEqual(classify_part(part, set(range(1, 8)))["counts"]["missing"], 3)

    def test_invalid_ids_and_failure_payload(self):
        for faces in [[0], [99], [True], [1, 1], []]:
            part = fixture()
            part["challenger"]["predictions"] = [prediction("bad", faces)]
            with self.subTest(faces=faces), self.assertRaises(ValueError):
                classify_part(part, set(range(1, 8)))
        part = fixture()
        part["challenger"]["status"] = "failed"
        with self.assertRaises(ValueError):
            classify_part(part, set(range(1, 8)))

    def test_merged_prediction_is_not_reference_match(self):
        part = fixture()
        part["challenger"]["predictions"] = [prediction("merge", [1, 2, 3, 4])]
        result = classify_part(part, set(range(1, 8)))
        self.assertEqual(result["counts"]["missing"], 3)
        self.assertEqual(result["stages"]["challenger"]["unmatched"][0]["category"], "unmatched")

    def test_incomplete_reference_not_full_accuracy(self):
        part = fixture()
        part["reference_status"] = "complete"
        with self.assertRaises(ValueError):
            classify_part(part, set(range(1, 8)))

    def test_optional_actual_marker_extent(self):
        p = prediction("p", [1, 2])
        self.assertIsNone(marker_line(p))
        p.update(line_start=[1, 2, 3], line_end=[4, 2, 3], marker_extent_semantics="finite_core_marker")
        result = marker_line(p)
        self.assertEqual(result["line_end"], [4, 2, 3])
        part = fixture()
        part["challenger"]["predictions"][0].update(p)
        self.assertEqual(classify_part(part, set(range(1, 8)))["stages"]["challenger"]["actual_marker_lines"], [result])

    def test_invalid_marker_lines(self):
        for start, end, semantics in [([1, 2], [3, 4, 5], "core"),
                                      ([1, 2, 3], None, "core"),
                                      ([1, 2, float("nan")], [3, 4, 5], "core"),
                                      ([1, 2, 3], [1, 2, 3], "core"),
                                      ([True, 2, 3], [3, 4, 5], "core"),
                                      ([1, 2, 3], [3, 4, 5], None)]:
            p = prediction("p", [1, 2])
            p.update(line_start=start, line_end=end, marker_extent_semantics=semantics)
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                marker_line(p)


if __name__ == "__main__":
    unittest.main()
