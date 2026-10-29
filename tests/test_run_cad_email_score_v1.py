"""Synthetic supervisor contracts: no subprocess jobs or real receipts."""
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
from scripts import run_cad_email_score_v1 as s


class SupervisorContracts(TestCase):
    def test_command_onepart_not_corpus(self):
        a = SimpleNamespace(out=Path("synthetic"), plan=Path("p"), plan_sha256="pin",
                            reference=Path("r"), terminal=Path("t"), terminal_sha256="t",
                            post=Path("post"), post_sha256="post")
        cmd = s.command(a)
        self.assertIn("score_cad_email_transfer_v1.py", cmd[1])
        self.assertEqual(cmd[-1], "synthetic/SCORE.json")
        self.assertNotIn("--extraction", cmd)

    def test_accounting_output_log_and_symlink(self):
        root = Path("synthetic")
        files = [root/"stdout.log", root/"SCORE.json"]
        with mock.patch.object(Path, "rglob", return_value=files), \
             mock.patch.object(Path, "is_symlink", return_value=False), \
             mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(Path, "stat", return_value=SimpleNamespace(st_size=1_000_001)):
            self.assertEqual(s.accounting(root)["reason"], "LOG_BUDGET")
        with mock.patch.object(Path, "rglob", return_value=files), \
             mock.patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaises(ValueError):
                s.accounting(root)
