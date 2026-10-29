"""Synthetic storage contracts; no filesystem fixtures or native imports."""
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
from scripts import evaluate_cad_email_transfer_v2 as subject


class StorageContracts(TestCase):
    def test_separate_fixed_caps(self):
        self.assertIsNone(subject.storage_reason(1_000_000, 262_163_291, 0))
        self.assertEqual(subject.storage_reason(0, 300_000_001, 0), "FROZEN_ARCHIVE_BUDGET")
        self.assertEqual(subject.storage_reason(40_000_000, 0, 0), "AGGREGATE_ARTIFACT_BUDGET")
        self.assertEqual(subject.storage_reason(0, 0, 1_000_001), "AGGREGATE_LOG_BUDGET")

    def test_only_exact_root_archive_files_exempt(self):
        root = Path("synthetic")
        sizes = {root/"SOURCE.tar": 262_000_000, root/"SOURCE_MANIFEST.json": 100,
                 root/"FROZEN.json": 100, root/"input"/"source.step": 657890,
                 root/"neutral"/"SOURCE.tar": 40, root/"arbitrary.tar": 50,
                 root/"neutral"/"stdout.log": 30}
        with mock.patch.object(Path, "rglob", return_value=list(sizes)), \
             mock.patch.object(Path, "is_symlink", return_value=False), \
             mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(Path, "stat", autospec=True, side_effect=lambda p: SimpleNamespace(st_size=sizes[p])):
            result = subject.accounting(root)
        self.assertEqual(result["frozen_archive_bytes"], 262_000_200)
        self.assertEqual(result["artifact_bytes"], 658010)
        self.assertEqual(result["log_bytes"], 30)
        self.assertIsNone(result["reason"])

    def test_symlink_never_exempt(self):
        with mock.patch.object(Path, "rglob", return_value=[Path("synthetic/SOURCE.tar")]), \
             mock.patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "SYMLINK"):
                subject.accounting(Path("synthetic"))

    def test_successor_identity_and_fixed_limits(self):
        self.assertIn("V2", subject.PROFILE)
        self.assertEqual(subject.LIMITS["case_seconds"], 60)
        self.assertEqual(subject.LIMITS["rss_kib"], 4*1024**2)
        self.assertEqual(subject.FROZEN_ARCHIVE_FILES, {"SOURCE.tar", "SOURCE_MANIFEST.json", "FROZEN.json"})
