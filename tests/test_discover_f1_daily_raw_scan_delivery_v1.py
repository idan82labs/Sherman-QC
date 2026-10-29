from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import pytest

from scripts.discover_f1_daily_raw_scan_delivery_v1 import (
    EXTERNAL_KIND,
    SCAN_ONLY_KIND,
    DiscoveryError,
    build_discovery_plan,
    canonical_json_bytes,
    materialize_plan,
    run_discovery,
)


DATE = "2026-08-31"
ZONE = "Asia/Jerusalem"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _ply(seed: int, *, comment: str | None = None) -> bytes:
    comment_row = f"comment {comment}\n" if comment else ""
    return (
        "ply\nformat ascii 1.0\n"
        f"{comment_row}"
        "element vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"end_header\n{seed} 0 0\n"
    ).encode("ascii")


def _stl(seed: int) -> bytes:
    return (
        f"solid raw_{seed}\n"
        "facet normal 0 0 1\nouter loop\n"
        "vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n"
        "endloop\nendfacet\nendsolid raw\n"
    ).encode("ascii")


def _set_today(path: Path) -> None:
    stamp = datetime(2026, 8, 31, 12, 0, tzinfo=ZoneInfo(ZONE)).timestamp()
    os.utime(path, (stamp, stamp))


def _set_old(path: Path) -> None:
    stamp = datetime(2026, 8, 30, 12, 0, tzinfo=ZoneInfo(ZONE)).timestamp()
    os.utime(path, (stamp, stamp))


def _freeze(tmp_path: Path) -> Path:
    path = tmp_path / "freeze.json"
    path.write_bytes(
        canonical_json_bytes(
            {
                "local_cohort_date": DATE,
                "local_timezone": ZONE,
                "status": "SEALED_BEFORE_2026_08_31_DELIVERY_DISCOVERY",
            }
        )
    )
    return path


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    incoming = tmp_path / "incoming"
    known = tmp_path / "known"
    incoming.mkdir()
    known.mkdir()
    return incoming, known


def _inventory_digest(members: list[dict]) -> str:
    rows = sorted(
        (
            {
                "link_sha256": row["link_sha256"],
                "member_name_sha256": row["member_name_sha256"],
            }
            for row in members
        ),
        key=lambda row: (row["member_name_sha256"], row["link_sha256"]),
    )
    return _sha_bytes(canonical_json_bytes(rows))


def _external(
    tmp_path: Path,
    *,
    members: list[dict] | None = None,
    observations: list[dict] | None = None,
) -> Path:
    inventory = None
    if members is not None:
        inventory = {
            "captured_at": "2026-08-31T12:00:00+03:00",
            "inventory_sha256": _inventory_digest(members),
            "members": members,
        }
    path = tmp_path / "external.json"
    path.write_bytes(
        canonical_json_bytes(
            {
                "cohort_date": DATE,
                "inventory": inventory,
                "kind": EXTERNAL_KIND,
                "observations": observations or [],
                "schema_version": 1,
                "timezone": ZONE,
            }
        )
    )
    return path


def _run(
    tmp_path: Path,
    incoming: Path,
    known: Path,
    *,
    external: Path | None = None,
    known_inventory_digests: tuple[str, ...] = (),
    materialize: bool = False,
) -> tuple[dict, Path, Path, Path]:
    output = tmp_path / "new_discovery.json"
    opaque = tmp_path / "sealed" / "opaque_scans"
    scan_manifest = tmp_path / "sealed" / "scan_only.json"
    result = run_discovery(
        cohort_date_value=DATE,
        timezone_name=ZONE,
        discovery_roots=[incoming],
        known_corpus_roots=[known],
        freeze_path=_freeze(tmp_path),
        external_manifest_path=external,
        known_inventory_digests=known_inventory_digests,
        output_manifest=output,
        materialize=materialize,
        opaque_root=opaque if materialize else None,
        scan_only_manifest=scan_manifest if materialize else None,
    )
    return result, output, opaque, scan_manifest


def test_zero_discovery_is_canonical_idempotent_and_writes_no_opaque_artifacts(
    tmp_path: Path,
) -> None:
    incoming, known = _roots(tmp_path)
    freeze = _freeze(tmp_path)
    freeze_before = freeze.read_bytes()
    result, output, opaque, scan_manifest = _run(
        tmp_path, incoming, known, materialize=True
    )
    first = output.read_bytes()
    result2, _, _, _ = _run(tmp_path, incoming, known, materialize=True)
    assert output.read_bytes() == first == canonical_json_bytes(result)
    assert result2 == result
    assert result["status"] == "ZERO_ELIGIBLE_SCANS"
    assert result["eligible_scan_count"] == 0
    assert result["materialization"]["status"] == "NOT_RUN_EMPTY_COHORT"
    assert not opaque.exists()
    assert not scan_manifest.exists()
    assert freeze.read_bytes() == freeze_before


def test_novel_scans_are_sha_sorted_materialized_opaque_and_idempotent(
    tmp_path: Path,
) -> None:
    incoming, known = _roots(tmp_path)
    source_rows = []
    for name, payload in (("customer-B.PLY", _ply(2)), ("customer-A.stl", _stl(1))):
        path = incoming / name
        path.write_bytes(payload)
        _set_today(path)
        source_rows.append((path, payload))
    result, output, opaque, scan_manifest = _run(
        tmp_path, incoming, known, materialize=True
    )
    expected = sorted(source_rows, key=lambda row: (_sha_bytes(row[1]), len(row[1])))
    assert result["status"] == "ELIGIBLE_SCANS_MATERIALIZED"
    assert result["eligible_scan_count"] == 2
    assert [row["case_id"] for row in result["eligible_scans"]] == ["T001", "T002"]
    assert [row["scan_sha256"] for row in result["eligible_scans"]] == [
        _sha_bytes(payload) for _path, payload in expected
    ]
    for ordinal, (source, payload) in enumerate(expected, start=1):
        target = opaque / f"T{ordinal:03d}{source.suffix.lower()}"
        assert target.read_bytes() == payload
    sealed = json.loads(scan_manifest.read_text(encoding="utf-8"))
    assert set(sealed) == {"cases", "kind", "schema_version"}
    assert sealed["kind"] == SCAN_ONLY_KIND
    assert all(set(row) == {"scan_path", "scan_sha256"} for row in sealed["cases"])
    assert not any("customer" in str(row).lower() for row in sealed["cases"])
    first_discovery = output.read_bytes()
    first_scan_manifest = scan_manifest.read_bytes()
    result2, _, _, _ = _run(tmp_path, incoming, known, materialize=True)
    assert result2 == result
    assert output.read_bytes() == first_discovery
    assert scan_manifest.read_bytes() == first_scan_manifest


def test_known_old_derived_synthetic_and_invalid_scans_are_rejected(
    tmp_path: Path,
) -> None:
    incoming, known = _roots(tmp_path)
    known_scan = known / "historical.ply"
    known_scan.write_bytes(_ply(1))
    duplicate = incoming / "fresh.ply"
    duplicate.write_bytes(known_scan.read_bytes())
    derived = incoming / "aligned_result.ply"
    derived.write_bytes(_ply(2))
    synthetic = incoming / "fresh2.ply"
    synthetic.write_bytes(_ply(3, comment="synthetic benchmark fixture"))
    invalid = incoming / "fresh3.ply"
    invalid.write_bytes(b"not a point cloud")
    old = incoming / "old.ply"
    old.write_bytes(_ply(4))
    for path in (duplicate, derived, synthetic, invalid):
        _set_today(path)
    _set_old(old)
    result, _, _, _ = _run(tmp_path, incoming, known)
    reasons = {row["reason"] for row in result["rejections"]}
    assert result["eligible_scan_count"] == 0
    assert result["status"] == "ZERO_ELIGIBLE_SCANS"
    assert {
        "KNOWN_CORPUS_DUPLICATE",
        "DERIVED_OR_SYNTHETIC_PATH",
        "SYNTHETIC_OR_DERIVED_CONTENT_MARKER",
        "INVALID_PLY_HEADER",
        "OUTSIDE_COHORT_WINDOW",
    }.issubset(reasons)


def test_derived_directory_is_pruned_without_opening_its_payload(tmp_path: Path) -> None:
    incoming, known = _roots(tmp_path)
    derived = incoming / "output"
    derived.mkdir()
    unreadable_like = derived / "would_be_scan.ply"
    unreadable_like.write_bytes(_ply(1))
    _set_today(unreadable_like)
    result, _, _, _ = _run(tmp_path, incoming, known)
    assert result["eligible_scan_count"] == 0
    assert result["summary"]["pruned_derived_directory_count"] == 1
    assert result["summary"]["geometry_file_count"] == 0


def test_unchanged_external_name_link_inventory_is_historical_and_nonblocking(
    tmp_path: Path,
) -> None:
    incoming, known = _roots(tmp_path)
    members = [
        {
            "content_sha256": None,
            "link_sha256": _sha_bytes(b"link-a"),
            "local_path": None,
            "member_name_sha256": _sha_bytes(b"49205000_A.ply"),
        },
        {
            "content_sha256": None,
            "link_sha256": _sha_bytes(b"link-b"),
            "local_path": None,
            "member_name_sha256": _sha_bytes(b"41017000_B.ply"),
        },
    ]
    external = _external(tmp_path, members=members)
    digest = json.loads(external.read_text())["inventory"]["inventory_sha256"]
    result, _, opaque, _ = _run(
        tmp_path,
        incoming,
        known,
        external=external,
        known_inventory_digests=(digest,),
        materialize=True,
    )
    inventory = result["external_evidence"]["inventory"]
    assert result["status"] == "ZERO_ELIGIBLE_SCANS"
    assert result["summary"]["blocking_issue_count"] == 0
    assert inventory["status"] == "UNCHANGED_KNOWN_NAME_LINK_INVENTORY"
    assert {row["classification"] for row in inventory["classifications"]} == {
        "UNCHANGED_KNOWN_NAME_LINK_MEMBER"
    }
    assert not opaque.exists()


def test_unrecognized_known_name_without_bytes_blocks_instead_of_assuming_duplicate(
    tmp_path: Path,
) -> None:
    incoming, known = _roots(tmp_path)
    historical = known / "historical.ply"
    historical.write_bytes(_ply(1))
    members = [
        {
            "content_sha256": None,
            "link_sha256": _sha_bytes(b"new-link"),
            "local_path": None,
            "member_name_sha256": _sha_bytes(historical.name.encode()),
        }
    ]
    external = _external(tmp_path, members=members)
    result, _, _, _ = _run(tmp_path, incoming, known, external=external)
    assert result["status"] == "BLOCKED_FAIL_CLOSED"
    assert result["summary"]["blocking_issue_count"] == 1
    assert result["external_evidence"]["inventory"]["classifications"][0][
        "classification"
    ] == "KNOWN_NAME_CONTENT_UNVERIFIED"
    assert result["rejections"][0]["reason"] == "EXTERNAL_KNOWN_NAME_CONTENT_UNVERIFIED"


def test_declared_known_content_and_verified_local_duplicate_are_classified(
    tmp_path: Path,
) -> None:
    incoming, known = _roots(tmp_path)
    payload = _ply(7)
    historical = known / "historical.ply"
    historical.write_bytes(payload)
    delivered = incoming / "daily.ply"
    delivered.write_bytes(payload)
    _set_old(delivered)  # External cohort evidence, not mtime, makes it inspectable.
    members = [
        {
            "content_sha256": _sha_bytes(payload),
            "link_sha256": _sha_bytes(b"link"),
            "local_path": str(delivered.resolve()),
            "member_name_sha256": _sha_bytes(delivered.name.encode()),
        }
    ]
    external = _external(tmp_path, members=members)
    result, _, _, _ = _run(tmp_path, incoming, known, external=external)
    assert result["status"] == "ZERO_ELIGIBLE_SCANS"
    assert result["summary"]["blocking_issue_count"] == 0
    assert {row["reason"] for row in result["rejections"]} == {
        "KNOWN_CORPUS_DUPLICATE"
    }
    assert result["external_evidence"]["inventory"]["classifications"][0][
        "classification"
    ] == "DECLARED_KNOWN_CORPUS_CONTENT_SHA256"


def test_external_attestation_can_admit_old_mtime_novel_bytes(tmp_path: Path) -> None:
    incoming, known = _roots(tmp_path)
    delivered = incoming / "attachment.ply"
    delivered.write_bytes(_ply(11))
    _set_old(delivered)
    observation = {
        "link_sha256": None,
        "local_path": str(delivered.resolve()),
        "message_id_sha256": _sha_bytes(b"message"),
        "observed_at": "2026-08-31T09:00:00+03:00",
    }
    external = _external(tmp_path, observations=[observation])
    result, _, opaque, _ = _run(
        tmp_path, incoming, known, external=external, materialize=True
    )
    assert result["eligible_scan_count"] == 1
    assert result["eligible_scans"][0]["discovery_channels"] == ["external_manifest"]
    assert (opaque / "T001.ply").read_bytes() == delivered.read_bytes()


def test_declared_content_mismatch_blocks_and_materializes_nothing(tmp_path: Path) -> None:
    incoming, known = _roots(tmp_path)
    delivered = incoming / "daily.ply"
    delivered.write_bytes(_ply(9))
    _set_today(delivered)
    members = [
        {
            "content_sha256": _sha_bytes(b"different"),
            "link_sha256": _sha_bytes(b"link"),
            "local_path": str(delivered.resolve()),
            "member_name_sha256": _sha_bytes(delivered.name.encode()),
        }
    ]
    external = _external(tmp_path, members=members)
    result, _, opaque, scan_manifest = _run(
        tmp_path, incoming, known, external=external, materialize=True
    )
    assert result["status"] == "BLOCKED_FAIL_CLOSED"
    assert result["eligible_scan_count"] == 0
    assert result["rejections"][0]["reason"] == "DECLARED_CONTENT_HASH_MISMATCH"
    assert not opaque.exists()
    assert not scan_manifest.exists()


def test_external_inventory_digest_tamper_is_rejected(tmp_path: Path) -> None:
    incoming, known = _roots(tmp_path)
    members = [
        {
            "content_sha256": None,
            "link_sha256": _sha_bytes(b"link"),
            "local_path": None,
            "member_name_sha256": _sha_bytes(b"name"),
        }
    ]
    external = _external(tmp_path, members=members)
    payload = json.loads(external.read_text())
    payload["inventory"]["inventory_sha256"] = "0" * 64
    external.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DiscoveryError, match="does not bind"):
        build_discovery_plan(
            cohort_date_value=DATE,
            timezone_name=ZONE,
            discovery_roots=[incoming],
            known_corpus_roots=[known],
            freeze_path=_freeze(tmp_path),
            external_manifest_path=external,
        )


def test_external_candidate_outside_explicit_roots_blocks_without_reading_it(
    tmp_path: Path,
) -> None:
    incoming, known = _roots(tmp_path)
    outside = tmp_path / "outside.ply"
    outside.write_bytes(_ply(99))
    _set_today(outside)
    observation = {
        "link_sha256": None,
        "local_path": str(outside.resolve()),
        "message_id_sha256": _sha_bytes(b"message"),
        "observed_at": "2026-08-31T09:00:00+03:00",
    }
    external = _external(tmp_path, observations=[observation])
    result, _, _, _ = _run(tmp_path, incoming, known, external=external)
    assert result["status"] == "BLOCKED_FAIL_CLOSED"
    assert result["rejections"][0]["reason"] == "EXTERNAL_CANDIDATE_OUTSIDE_ROOTS"
    assert _sha(outside) == _sha_bytes(_ply(99))


def test_source_mutation_after_plan_aborts_before_any_copy(tmp_path: Path) -> None:
    incoming, known = _roots(tmp_path)
    scan = incoming / "daily.ply"
    scan.write_bytes(_ply(1))
    _set_today(scan)
    plan = build_discovery_plan(
        cohort_date_value=DATE,
        timezone_name=ZONE,
        discovery_roots=[incoming],
        known_corpus_roots=[known],
        freeze_path=_freeze(tmp_path),
        materialize_requested=True,
    )
    scan.write_bytes(_ply(2))
    opaque = tmp_path / "sealed" / "opaque_scans"
    with pytest.raises(DiscoveryError, match="changed after discovery"):
        materialize_plan(
            plan,
            opaque_root=opaque,
            scan_only_manifest=tmp_path / "sealed/scan_only.json",
            discovery_roots=[incoming],
        )
    assert not opaque.exists()


def test_stale_opaque_target_blocks_before_copying_new_cases(tmp_path: Path) -> None:
    incoming, known = _roots(tmp_path)
    scan = incoming / "daily.ply"
    scan.write_bytes(_ply(1))
    _set_today(scan)
    plan = build_discovery_plan(
        cohort_date_value=DATE,
        timezone_name=ZONE,
        discovery_roots=[incoming],
        known_corpus_roots=[known],
        freeze_path=_freeze(tmp_path),
        materialize_requested=True,
    )
    opaque = tmp_path / "sealed/opaque_scans"
    opaque.mkdir(parents=True)
    stale = opaque / "T999.ply"
    stale.write_bytes(_ply(999))
    with pytest.raises(DiscoveryError, match="unexpected stale"):
        materialize_plan(
            plan,
            opaque_root=opaque,
            scan_only_manifest=tmp_path / "sealed/scan_only.json",
            discovery_roots=[incoming],
        )
    assert list(opaque.iterdir()) == [stale]


def test_roots_must_be_disjoint_and_freeze_must_match(tmp_path: Path) -> None:
    incoming, _known = _roots(tmp_path)
    nested_known = incoming / "known"
    nested_known.mkdir()
    with pytest.raises(DiscoveryError, match="must be disjoint"):
        build_discovery_plan(
            cohort_date_value=DATE,
            timezone_name=ZONE,
            discovery_roots=[incoming],
            known_corpus_roots=[nested_known],
            freeze_path=_freeze(tmp_path),
        )
    wrong = tmp_path / "wrong.json"
    wrong.write_bytes(
        canonical_json_bytes(
            {
                "local_cohort_date": "2026-09-01",
                "local_timezone": ZONE,
                "status": "SEALED_BEFORE_DISCOVERY",
            }
        )
    )
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(DiscoveryError, match="cohort date mismatch"):
        build_discovery_plan(
            cohort_date_value=DATE,
            timezone_name=ZONE,
            discovery_roots=[incoming],
            known_corpus_roots=[other],
            freeze_path=wrong,
        )


def test_cli_zero_run_is_connector_free_and_preserves_unrelated_existing_manifest(
    tmp_path: Path,
) -> None:
    incoming, known = _roots(tmp_path)
    preserved = tmp_path / "DELIVERY_DISCOVERY_MANIFEST.json"
    preserved.write_bytes(b"historical-manifest-must-not-change\n")
    output = tmp_path / "new-monitor-state.json"
    runner = Path(__file__).resolve().parents[1] / "scripts/discover_f1_daily_raw_scan_delivery_v1.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(runner),
            "--cohort-date",
            DATE,
            "--timezone",
            ZONE,
            "--root",
            str(incoming),
            "--known-corpus-root",
            str(known),
            "--freeze",
            str(_freeze(tmp_path)),
            "--output-manifest",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "ZERO_ELIGIBLE_SCANS"
    assert output.is_file()
    assert preserved.read_bytes() == b"historical-manifest-must-not-change\n"


def test_external_manifest_rejects_raw_link_or_unknown_fields(tmp_path: Path) -> None:
    incoming, known = _roots(tmp_path)
    external = _external(tmp_path)
    payload = json.loads(external.read_text())
    payload["raw_url"] = "https://example.invalid/private"
    external.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DiscoveryError, match="unknown or missing"):
        build_discovery_plan(
            cohort_date_value=DATE,
            timezone_name=ZONE,
            discovery_roots=[incoming],
            known_corpus_roots=[known],
            freeze_path=_freeze(tmp_path),
            external_manifest_path=external,
        )
