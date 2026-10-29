"""C07 artifact serving: allowlist, workspace containment, headers, caching.

``GET /api/line/jobs/{job_id}/artifacts/{name}`` serves only names in
``contracts.production.layout.ARTIFACT_ALLOWLIST`` from inside
``<data_root>/jobs/<job_id>``; anything else is 404. Files are tmp fixtures,
never real scans.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from apps.api.services import production_line_service as svc
from contracts.production.layout import ARTIFACT_ALLOWLIST
from tests.test_prod_line_routes import OTHER, SKU, STATION, LineHarness, build_harness, h


@pytest.fixture()
def harness(tmp_path: Path) -> Iterator[LineHarness]:
    hs = build_harness(tmp_path)
    yield hs
    hs.client.close()
    svc.reset_line_api_caches()


def _url(job_id: str, name: str) -> str:
    return f"/api/line/jobs/{job_id}/artifacts/{name}"


def test_allowlisted_artifacts_are_served_with_cache_headers(harness: LineHarness) -> None:
    job = harness.completed_job(STATION)
    ws = harness.ws(job["job_id"])
    (ws / "result" / "overview.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 8)
    (ws / "result" / "stage_mesh.ply").write_bytes(b"ply\nend_header\n")
    (ws / "result" / "final_mesh.ply").write_bytes(b"ply\nend_header\n")
    for name, rel in ARTIFACT_ALLOWLIST.items():
        r = harness.client.get(_url(job["job_id"], name), headers=h("tok-viewer"))
        assert r.status_code == 200, name
        assert r.content == (ws / rel).read_bytes(), name
        assert r.headers["cache-control"] == "private, max-age=3600"
        assert r.headers["etag"].startswith('"') and r.headers["x-content-type-options"] == "nosniff"
        assert int(r.headers["content-length"]) == (ws / rel).stat().st_size
    ply = harness.client.get(_url(job["job_id"], "viewer_scan.ply"), headers=h("tok-viewer"))
    assert ply.headers["content-type"] == "application/octet-stream"
    png = harness.client.get(_url(job["job_id"], "overview.png"), headers=h("tok-viewer"))
    assert png.headers["content-type"] == "image/png"


@pytest.mark.parametrize(
    "name",
    [
        "..%2Fjob.json",
        "../job.json",
        "recipe.json",
        "job.json",
        "engine/bend_inspection_report.json",
        "engine%2Fbend_inspection_report.json",
        "result/report.pdf",
        "shopfloor_result.json",
        "result%2Fshopfloor_result.json",
        "REPORT.PDF",
        "report.pdf%00",
        "%2E%2E%2F%2E%2E%2Fdb%2Fline.sqlite3",
    ],
)
def test_non_allowlisted_names_are_404(harness: LineHarness, name: str) -> None:
    job = harness.completed_job(STATION)
    (harness.ws(job["job_id"]) / "recipe.json").write_text("{}")
    r = harness.client.get(f"/api/line/jobs/{job['job_id']}/artifacts/{name}", headers=h("tok-admin"))
    assert r.status_code == 404, (name, r.status_code)
    assert b"job_id" not in r.content and b"{}" != r.content


def test_f32_headers(harness: LineHarness) -> None:
    job = harness.completed_job(STATION)
    path = harness.ws(job["job_id"]) / "result" / "viewer_deviations.f32"
    path.write_bytes(b"\x00\x00\x80\x3f" * 37)
    r = harness.client.get(_url(job["job_id"], "viewer_deviations.f32"), headers=h("tok-viewer"))
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert int(r.headers["x-deviation-count"]) == path.stat().st_size // 4 == 37
    assert r.headers["x-deviation-units"] == "mm"
    # ENGINE_IO §4.1 sign convention (the result's viewer_assets.deviation_sign).
    assert r.headers["x-deviation-sign"] == "SIGNED_ALONG_STAGE_MODEL_OUTWARD_NORMAL"
    assert r.headers["x-deviation-encoding"] == "float32_le_nan_missing"

    path.write_bytes(b"\x00" * 6)  # not a whole number of float32 values
    r = harness.client.get(_url(job["job_id"], "viewer_deviations.f32"), headers=h("tok-viewer"))
    assert r.status_code == 500 and r.json()["detail"]["code"] == "ARTIFACT_INVALID"


def test_pdf_content_disposition(harness: LineHarness) -> None:
    job = harness.completed_job(STATION, serial="SN 42/7")
    r = harness.client.get(_url(job["job_id"], "report.pdf"), headers=h("tok-viewer"))
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    date = job["finished_at"][:10].replace("-", "")
    assert r.headers["content-disposition"] == f'attachment; filename="{SKU}_SN-42-7_{date}.pdf"'

    no_serial = harness.completed_job(STATION, serial=None)
    r = harness.client.get(_url(no_serial["job_id"], "report.pdf"), headers=h("tok-viewer"))
    job8 = no_serial["job_id"][5:13]
    assert r.headers["content-disposition"] == f'attachment; filename="{SKU}_{job8}_{date}.pdf"'
    assert re.fullmatch(r'attachment; filename="[A-Za-z0-9._-]+\.pdf"', r.headers["content-disposition"])


def test_etag_304_and_change(harness: LineHarness) -> None:
    job = harness.completed_job(STATION)
    url = _url(job["job_id"], "report.pdf")
    r = harness.client.get(url, headers=h("tok-viewer"))
    etag = r.headers["etag"]
    again = harness.client.get(url, headers={**h("tok-viewer"), "If-None-Match": etag})
    assert again.status_code == 304 and not again.content and again.headers["etag"] == etag
    path = harness.ws(job["job_id"]) / "result" / "report.pdf"
    path.write_bytes(b"%PDF-1.4\n%re-rendered\n")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    changed = harness.client.get(url, headers={**h("tok-viewer"), "If-None-Match": etag})
    assert changed.status_code == 200 and changed.headers["etag"] != etag
    assert changed.content == b"%PDF-1.4\n%re-rendered\n"
    # The ETag binds the job id: the same bytes in another job differ.
    other = harness.completed_job(STATION, serial="SN2")
    assert harness.client.get(_url(other["job_id"], "report.pdf"), headers=h("tok-viewer")).headers["etag"] != changed.headers["etag"]


def test_symlink_out_of_workspace_is_refused(harness: LineHarness, tmp_path: Path) -> None:
    job = harness.completed_job(STATION)
    secret = tmp_path / "outside.ply"
    secret.write_bytes(b"OUTSIDE-SECRET")
    target = harness.ws(job["job_id"]) / "result" / "viewer_scan.ply"
    target.unlink()
    target.symlink_to(secret)
    r = harness.client.get(_url(job["job_id"], "viewer_scan.ply"), headers=h("tok-admin"))
    assert r.status_code == 404 and b"OUTSIDE-SECRET" not in r.content
    # Pointing into another job's workspace is refused too.
    other = harness.completed_job(STATION, serial="SN3")
    target.unlink()
    target.symlink_to(harness.ws(other["job_id"]) / "result" / "viewer_scan.ply")
    assert harness.client.get(_url(job["job_id"], "viewer_scan.ply"), headers=h("tok-admin")).status_code == 404
    # A whole result/ directory symlinked elsewhere is refused.
    result_dir = harness.ws(other["job_id"]) / "result"
    moved = tmp_path / "moved-result"
    result_dir.rename(moved)
    result_dir.symlink_to(moved, target_is_directory=True)
    assert harness.client.get(_url(other["job_id"], "report.pdf"), headers=h("tok-admin")).status_code == 404


def test_directory_in_place_of_file_is_404(harness: LineHarness) -> None:
    job = harness.completed_job(STATION)
    target = harness.ws(job["job_id"]) / "result" / "viewer_scan.ply"
    target.unlink()
    target.mkdir()
    assert harness.client.get(_url(job["job_id"], "viewer_scan.ply"), headers=h("tok-admin")).status_code == 404


def test_missing_artifact_and_unfinished_job(harness: LineHarness) -> None:
    job = harness.completed_job(STATION, artifacts=False)
    r = harness.client.get(_url(job["job_id"], "report.pdf"), headers=h("tok-viewer"))
    assert r.status_code == 404 and r.json()["detail"]["code"] == "ARTIFACT_NOT_FOUND"
    running = harness.start(harness.queue_job(STATION))
    harness.write_workspace(running, None)
    r = harness.client.get(_url(running["job_id"], "report.pdf"), headers=h("tok-viewer"))
    assert r.status_code == 409 and r.json()["detail"]["code"] == "JOB_NOT_COMPLETED"
    assert harness.client.get(_url("line_" + "e" * 32, "report.pdf"), headers=h("tok-viewer")).status_code == 404
    assert harness.client.get(_url("line_XYZ", "report.pdf"), headers=h("tok-viewer")).status_code == 404


def test_device_scope_on_artifacts(harness: LineHarness) -> None:
    mine = harness.completed_job(STATION)
    other = harness.completed_job(OTHER)
    _, device = harness.device_token(STATION)
    assert harness.client.get(_url(mine["job_id"], "viewer_deviations.f32"), headers=h(device)).status_code == 200
    r = harness.client.get(_url(other["job_id"], "viewer_deviations.f32"), headers=h(device))
    assert r.status_code == 404 and r.json()["detail"]["code"] == "JOB_NOT_FOUND"
    assert harness.client.get(_url(other["job_id"], "report.pdf")).status_code == 401


def test_never_reads_outside_data_root_jobs(harness: LineHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every file the artifact route opens lies under <data_root>/jobs/<job_id>."""
    job = harness.completed_job(STATION)
    opened: list[str] = []

    class _OsSpy:
        """``os`` for the service module only, recording every ``open``."""

        def __getattr__(self, name: str):  # noqa: ANN204
            return getattr(os, name)

        def open(self, path, flags, *args, **kwargs):  # noqa: ANN001, ANN201
            opened.append(os.fspath(path))
            return os.open(path, flags, *args, **kwargs)

    monkeypatch.setattr(svc, "os", _OsSpy())
    for name in ("report.pdf", "viewer_scan.ply", "viewer_deviations.f32", "bend_geometry.json"):
        assert harness.client.get(_url(job["job_id"], name), headers=h("tok-viewer")).status_code == 200
    root = os.path.realpath(harness.ws(job["job_id"])) + os.sep
    assert opened and all(os.path.realpath(p).startswith(root) for p in opened)
