import unittest

from backend.cad_bend_reference_scoring_v1 import score_parts


def part(**overrides):
    result = dict(part_id="synthetic", face_ids=[1, 2, 3], reference_status="complete",
                  unknown_face_ids=[], verified_nonbend_face_ids=[3],
                  reference_bends=[dict(bend_id="b", inner_face_ids=[1], outer_face_ids=[2])],
                  predictions=[dict(prediction_id="p", face_ids=[2, 1])])
    result.update(overrides)
    return result


class ScoringTests(unittest.TestCase):
    def test_exact_and_duplicate(self):
        score = score_parts([part(predictions=[dict(prediction_id="z", face_ids=[1, 2]),
                                              dict(prediction_id="a", face_ids=[2, 1])])])
        self.assertEqual(score["full_reference_micro"]["tp"], 1)
        self.assertEqual(score["full_reference_micro"]["fp"], 1)
        self.assertEqual(score["parts"][0]["matches"][0]["prediction_id"], "a")
        self.assertEqual(score["full_reference_micro"]["f1"], 2 / 3)

    def test_partial_unknown_is_unscored(self):
        score = score_parts([part(reference_status="partial", unknown_face_ids=[3],
                                 verified_nonbend_face_ids=[],
                                 predictions=[dict(prediction_id="u", face_ids=[3])])])
        self.assertEqual(score["partial_verified_positive"]["recall"], 0)
        self.assertIsNone(score["parts"][0]["fp"])
        self.assertIsNone(score["parts"][0]["fn"])
        self.assertIsNone(score["full_reference_micro"]["f1"])
        self.assertEqual(score["parts"][0]["unscored_prediction_ids"], ["u"])

    def test_failure_retained(self):
        score = score_parts([part(predictions=[], prediction_status="failed", failure_reason="synthetic failure")])
        self.assertEqual(score["full_reference_micro"]["fn"], 1)
        self.assertFalse(score["parts"][0]["exact_count_correct"])
        self.assertEqual(score["parts"][0]["failure_reason"], "synthetic failure")

    def test_empty_truth(self):
        score = score_parts([part(reference_bends=[], predictions=[], verified_nonbend_face_ids=[1, 2, 3])])
        self.assertIsNone(score["full_reference_micro"]["precision"])
        self.assertIsNone(score["full_reference_micro"]["recall"])
        self.assertEqual(score["exact_count_accuracy"]["accuracy"], 1)

    def test_invalid_face_ids_and_partition(self):
        for changes in [dict(predictions=[dict(prediction_id="p", face_ids=[99])]),
                        dict(face_ids=[True, 2, 3]), dict(face_ids=[0, 1, 2]),
                        dict(verified_nonbend_face_ids=[]),
                        dict(unknown_face_ids=[3]), dict(verified_nonbend_face_ids=[1, 3])]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                score_parts([part(**changes)])

    def test_empty_corpus(self):
        self.assertIsNone(score_parts([])["coverage"]["full_reference_part_fraction"])

    def test_merged_prediction(self):
        score = score_parts([part(face_ids=[1, 2, 3, 4], verified_nonbend_face_ids=[],
                                 reference_bends=[dict(bend_id="a", inner_face_ids=[1], outer_face_ids=[2]),
                                                  dict(bend_id="b", inner_face_ids=[3], outer_face_ids=[4])],
                                 predictions=[dict(prediction_id="merged", face_ids=[1, 2, 3, 4])])])
        micro = score["full_reference_micro"]
        self.assertEqual((micro["tp"], micro["fp"], micro["fn"]), (0, 1, 2))
        self.assertEqual(score["full_reference_macro"]["f1"], {"value": 0, "defined_part_denominator": 1})

    def test_failed_empty_truth_is_not_count_agreement(self):
        score = score_parts([part(reference_bends=[], predictions=[], prediction_status="failed",
                                 verified_nonbend_face_ids=[1, 2, 3])])
        self.assertFalse(score["parts"][0]["exact_count_correct"])

    def test_partial_duplicate_is_unscored(self):
        score = score_parts([part(reference_status="partial", unknown_face_ids=[3],
                                 verified_nonbend_face_ids=[],
                                 predictions=[dict(prediction_id="a", face_ids=[1, 2]),
                                              dict(prediction_id="b", face_ids=[2, 1])])])
        self.assertEqual(score["parts"][0]["tp"], 1)
        self.assertEqual(score["parts"][0]["unscored_prediction_ids"], ["b"])
        self.assertIsNone(score["parts"][0]["fp"])
        self.assertIsNone(score["full_reference_macro"]["recall"]["value"])

    def test_combined_known_positive_recall(self):
        score = score_parts([part(), part(part_id="partial", reference_status="partial",
                                         unknown_face_ids=[3], verified_nonbend_face_ids=[], predictions=[])])
        combined = score["all_verified_positive_recall"]
        self.assertEqual((combined["matched"], combined["reference_denominator"], combined["recall"]), (1, 2, .5))
        self.assertIsNone(combined["precision"])
        self.assertIsNone(combined["f1"])


if __name__ == "__main__":
    unittest.main()
