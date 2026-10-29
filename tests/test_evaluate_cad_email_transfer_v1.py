"""Pure synthetic gate tests. No native imports, raw inputs, or actual receipts."""
import json
import subprocess
import sys
from pathlib import Path
from unittest import TestCase, mock

from scripts import evaluate_cad_email_transfer_v1 as subject


def receipt():
    return {"filename": subject.INPUT_NAME, "sha256": subject.INPUT_SHA256,
            "file_bytes": subject.INPUT_BYTES,
            "approved_corpus_sha256": [f"{i:064x}" for i in range(20)]}


class TransferContracts(TestCase):
    def test_selected_binding_and_duplicate(self):
        self.assertEqual(subject.validate_input_receipt(receipt()), receipt())
        bad = receipt()
        bad["approved_corpus_sha256"][0] = subject.INPUT_SHA256
        with self.assertRaisesRegex(ValueError, "DUPLICATE"):
            subject.validate_input_receipt(bad)
        for key, value in (("filename", "other.step"), ("file_bytes", 1), ("sha256", "0"*64)):
            bad = receipt()
            bad[key] = value
            with self.assertRaises(ValueError):
                subject.validate_input_receipt(bad)

    def test_corpus_census_not_boolean_assertion(self):
        for hashes in ([], ["0"*64]*20, ["z"*64]+[f"{i:064x}" for i in range(19)]):
            bad = receipt()
            bad["approved_corpus_sha256"] = hashes
            with self.assertRaises(ValueError):
                subject.validate_input_receipt(bad)

    def test_pre_is_mode_scoped_and_independent(self):
        good = {"decision": "GREEN", "independent_non_author": True,
                "frozen_sha256": "pin", "mode": "neutral"}
        subject.validate_pre(good, "pin", "neutral")
        for key, value in (("decision", "RED"), ("independent_non_author", 1),
                           ("frozen_sha256", "other"), ("mode", "extraction")):
            with self.assertRaises(ValueError):
                subject.validate_pre({**good, key: value}, "pin", "neutral")

    def test_extraction_requires_success_post_and_reference(self):
        evidence = b'{"synthetic":true}'
        post = {"decision": "GREEN", "independent_non_author": True,
                "frozen_sha256": "pin", "mode": "neutral"}
        reference = {"input_sha256": subject.INPUT_SHA256,
                     "evidence_sha256": subject.sha(evidence),
                     "independent_non_author": True, "reference_frozen": True,
                     "reference_bends": [], "feature_complete": True,
                     "reference_status": "complete", "limitations": ["Technical, not manufacturing truth"],
                     "face_ids": [1], "unknown_face_ids": [], "verified_nonbend_face_ids": [1]}
        terminal = {"outer_exit": 0, "reason": None, "cleanup_known": True}
        def invoke(post_value=post, ref_value=reference, terminal_value=terminal):
            data = {"post": subject.encode(post_value), "ref": subject.encode(ref_value),
                    "CASE_TERMINAL.json": subject.encode(terminal_value), "EVIDENCE.json": evidence}
            with mock.patch.object(subject, "read", side_effect=lambda p, *args: data[Path(p).name]):
                subject.extraction_gate(Path("synthetic"), "pin", Path("post"),
                                        subject.sha(data["post"]), Path("ref"), subject.sha(data["ref"]))
        invoke()
        with self.assertRaises(ValueError):
            invoke(terminal_value={**terminal, "cleanup_known": False})
        with self.assertRaises(ValueError):
            invoke(ref_value={**reference, "reference_frozen": False})
        with self.assertRaises(ValueError):
            invoke(ref_value={**reference, "evidence_sha256": "wrong"})
        with self.assertRaises(ValueError):
            invoke(post_value={**post, "mode": "extraction"})
        for key in ("reference_bends", "feature_complete", "limitations"):
            bad = dict(reference)
            del bad[key]
            with self.assertRaises(ValueError):
                invoke(ref_value=bad)

    def test_cli_help_without_native_or_input(self):
        result = subprocess.run([sys.executable, str(subject.ROOT/"scripts/evaluate_cad_email_transfer_v1.py"),
                                 "--help"], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--native-child", result.stdout)

    def test_no_retry_job_folder(self):
        good = subject.encode({"decision": "GREEN", "independent_non_author": True,
                               "frozen_sha256": "pin", "mode": "neutral"})
        with mock.patch.object(subject, "verify"), mock.patch.object(subject, "read", return_value=good), \
             mock.patch.object(subject, "accounting", return_value={"reason": None}), \
             mock.patch.object(Path, "exists", return_value=False), \
             mock.patch.object(Path, "mkdir", side_effect=FileExistsError), \
             mock.patch.object(subject, "run_job") as run:
            with self.assertRaises(FileExistsError):
                subject.execute(Path("synthetic"), "neutral", "pin", Path("pre"), subject.sha(good))
            run.assert_not_called()
