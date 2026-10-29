"""Pure supervisor contract tests; no native or corpus access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import evaluate_cad_reference_evidence_v1 as campaign


class SupervisorTests(unittest.TestCase):
    def test_final_byte_accounting_caps_and_source_inclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "source.step").write_bytes(b"12345")
            (folder / "stdout.log").write_bytes(b"123")
            (folder / "stderr.log").write_bytes(b"12")
            (folder / "EVIDENCE.json").write_bytes(b"12345")
            limits = {**campaign.LIMITS, "artifact_bytes": 15, "log_bytes": 5}
            with patch.object(campaign, "LIMITS", limits):
                self.assertEqual(campaign.budget_accounting(folder),
                                 dict(log_bytes=5, artifact_bytes=15, reason=None))
                (folder / "EVIDENCE.json").write_bytes(b"123456")
                self.assertEqual(campaign.budget_accounting(folder)["reason"], "CASE_ARTIFACT_BUDGET")
                (folder / "stdout.log").write_bytes(b"1234")
                self.assertEqual(campaign.budget_accounting(folder)["reason"], "CASE_LOG_BUDGET")

    def test_failed_child_retained_in_complete_census(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            pre = attempt / "pre.json"
            raw = campaign.encode(dict(decision="GREEN", frozen_sha256="freeze",
                                       independent_non_author=True))
            pre.write_bytes(raw)
            rows = [dict(case_index=i, outer_exit=0 if i else 1,
                         reason=None, cleanup_known=True) for i in range(20)]
            with patch.object(campaign, "verify", return_value={"cases": [{}]*20}), \
                 patch.object(campaign, "run_case", side_effect=rows):
                self.assertEqual(campaign.execute(attempt, "freeze", pre, campaign.sha(raw)), 1)
            terminal = json.loads((attempt / "CAMPAIGN_TERMINAL.json").read_text())
            self.assertEqual(terminal["status"], "COMPLETE_WITH_FAILURES")
            self.assertEqual(terminal["failed_case_indices"], [0])
            self.assertEqual(len(terminal["receipts"]), 20)

    def test_unknown_cleanup_halts_all_remaining_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            pre = attempt / "pre.json"
            raw = campaign.encode(dict(decision="GREEN", frozen_sha256="freeze",
                                       independent_non_author=True))
            pre.write_bytes(raw)
            with patch.object(campaign, "verify", return_value={"cases": [{}]*20}), \
                 patch.object(campaign, "run_case", return_value=dict(
                     case_index=0, outer_exit=1, cleanup_known=False, reason="failure")) as run:
                self.assertEqual(campaign.execute(attempt, "freeze", pre, campaign.sha(raw)), 1)
            self.assertEqual(run.call_count, 1)
            terminal = json.loads((attempt / "CAMPAIGN_TERMINAL.json").read_text())
            self.assertEqual(len(terminal["receipts"]), 20)
            self.assertEqual(terminal["halt_reason"], "CLEANUP_UNKNOWN")
            self.assertEqual(terminal["receipts"][19]["status"], "NOT_EXECUTED")

    def test_execution_claim_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            campaign.write(attempt / "EXECUTION.json", {})
            with self.assertRaises(FileExistsError):
                campaign.write(attempt / "EXECUTION.json", {})


if __name__ == "__main__":
    unittest.main()
