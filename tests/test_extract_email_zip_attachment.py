from email.message import EmailMessage
from pathlib import Path

import pytest

from scripts.extract_email_zip_attachment import extract_zip_attachments


def _write_message(path: Path, *, filename: str = "part.zip", payload: bytes = b"PK\x03\x04data") -> None:
    message = EmailMessage()
    message["From"] = "eyal@example.com"
    message["To"] = "qc@example.com"
    message["Subject"] = "part"
    message.set_content("attached")
    message.add_attachment(payload, maintype="application", subtype="zip", filename=filename)
    path.write_bytes(message.as_bytes())


def test_extract_zip_attachment_sanitizes_filename_and_hashes_payload(tmp_path):
    eml = tmp_path / "message.eml"
    _write_message(eml, filename="../part.zip")

    extracted = extract_zip_attachments(eml, tmp_path / "out")

    assert extracted[0]["filename"] == "part.zip"
    assert (tmp_path / "out" / "part.zip").read_bytes() == b"PK\x03\x04data"
    assert len(str(extracted[0]["sha256"])) == 64


def test_extract_zip_attachment_refuses_different_existing_payload(tmp_path):
    eml = tmp_path / "message.eml"
    _write_message(eml)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "part.zip").write_bytes(b"different")

    with pytest.raises(FileExistsError):
        extract_zip_attachments(eml, output_dir)
