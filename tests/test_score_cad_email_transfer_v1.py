"""Synthetic exact-face adapter tests; no native or artifact fixtures."""
from unittest import TestCase
from scripts import score_cad_email_transfer_v1 as s


class ScoreAdapter(TestCase):
    def setUp(self):
        self.reference = {"part_id": "synthetic", "input_sha256": "input", "reference_frozen": True,
            "face_ids": [1, 2, 3], "reference_status": "complete", "unknown_face_ids": [],
            "verified_nonbend_face_ids": [3], "reference_bends": [
                {"bend_id": "known", "inner_face_ids": [1], "outer_face_ids": [2]}]}
        self.plan = {"candidate_frozen_sha256": "candidate", "input_sha256": "input"}
        self.post = {"decision": "GREEN", "independent_non_author": True,
                     "mode": "extraction", "frozen_sha256": "candidate"}
        self.terminal = {"mode": "extraction", "cleanup_known": True, "outer_exit": 0, "reason": None}

    def test_exact_union_and_duplicate(self):
        e = {"input_sha256": "input", "bend_axes": [
            {"feature_id": "a", "face_indices": [2, 1]}, {"feature_id": "b", "face_indices": [1, 2]}]}
        result = s.score(self.reference, e, self.terminal, self.post, self.plan)
        self.assertEqual(result["parts"][0]["tp"], 1)
        self.assertEqual(result["parts"][0]["fp"], 1)

    def test_failed_abstention_retained(self):
        terminal = {**self.terminal, "outer_exit": 1, "reason": "TIME_BUDGET"}
        result = s.score(self.reference, None, terminal, self.post, self.plan)
        self.assertEqual(result["parts"][0]["fn"], 1)
        self.assertEqual(result["parts"][0]["prediction_status"], "failed")
        with self.assertRaises(ValueError):
            s.score(self.reference, {}, terminal, self.post, self.plan)

    def test_unknown_cleanup_refused(self):
        with self.assertRaises(ValueError):
            s.score(self.reference, None, {**self.terminal, "cleanup_known": False}, self.post, self.plan)

    def test_missing_structural_part_id(self):
        reference = dict(self.reference)
        del reference["part_id"]
        e = {"input_sha256": "input", "bend_axes": []}
        result = s.score(reference, e, self.terminal, self.post, self.plan)
        self.assertEqual(result["parts"][0]["part_id"], "email-transfer-01")
