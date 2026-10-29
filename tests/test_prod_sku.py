"""SKU canonicalisation and real scan-filename parsing (C01 sku.py). Synthetic names only."""
from __future__ import annotations

import pytest

from domains.production.recipe.sku import (
    SkuError,
    canonical_sku,
    find_canonical_collisions,
    parse_scan_filename,
)

KNOWN = ["11979003", "23553000", "38172000", "44211000", "44211000_A"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("44211000_A", "44211000_A"),
        ("  abc 1 ", "ABC-1"),
        ("abc   def\t1", "ABC-DEF-1"),
        ("ｓｙｎ－ｈａｔ", "SYN-HAT"),  # NFKC folds full-width forms
        ("a.b_c-d", "A.B_C-D"),
    ],
)
def test_canonical_sku(raw, expected):
    assert canonical_sku(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "_abc", "-1", "abc/1", "קוד", "A" * 65, "a#b"])
def test_canonical_sku_rejects(raw):
    with pytest.raises(SkuError) as exc:
        canonical_sku(raw)
    assert exc.value.code == "SKU_INVALID"


def test_canonical_collisions():
    collisions = find_canonical_collisions(["abc 1", "ABC-1", "x", "y"])
    assert collisions == {"ABC-1": ("ABC-1", "abc 1")}
    assert find_canonical_collisions(["x", "y"]) == {}
    with pytest.raises(SkuError):
        find_canonical_collisions(["ok", "not ok/"])


def test_survey_table():
    r = parse_scan_filename("44211000_A.ply", ["44211000_A"])
    assert (r.status, r.sku) == ("OK", "44211000_A")
    r = parse_scan_filename("44211000_A_SN0042_20261017T101500.PLY", ["44211000_A"])
    assert (r.status, r.sku, r.serial, r.scan_timestamp) == ("OK", "44211000_A", "SN0042", "2026-10-17T10:15:00")
    r = parse_scan_filename("44211000_001.ply", {"44211000", "44211000_A"})
    assert (r.status, r.sku, r.serial) == ("OK", "44211000", "001")
    assert parse_scan_filename("zz-unknown.ply", KNOWN).status == "UNMATCHED"
    assert parse_scan_filename(".ply", KNOWN).status == "UNPARSEABLE"
    assert parse_scan_filename("  (2).stl", KNOWN).status == "UNPARSEABLE"


def test_longest_known_sku_wins():
    r = parse_scan_filename("44211000_A_001.ply", KNOWN)
    assert (r.sku, r.serial) == ("44211000_A", "001")


def test_before_bending_prefix_and_double_space():
    r = parse_scan_filename("BEFORE BENDING  11979003_C.ply", KNOWN)
    assert (r.status, r.sku, r.revision, r.revision_status) == ("OK", "11979003", "C", "UNVERIFIED")
    assert r.scan_label == "BEFORE_BENDING" and r.prefix == "BEFORE BENDING" and r.serial is None


def test_full_scan_with_export_index():
    r = parse_scan_filename("23553000_B_FULL_SCAN 1.ply", KNOWN)
    assert (r.sku, r.revision, r.scan_label, r.export_index, r.serial, r.copy_index) == (
        "23553000", "B", "FULL_SCAN", 1, None, None)


def test_numeric_revision_and_scan_index():
    r = parse_scan_filename("38172000_01_SCAN_2 1.ply", KNOWN)
    assert (r.sku, r.revision, r.scan_label, r.export_index, r.serial) == ("38172000", "01", "SCAN_2", 1, None)


@pytest.mark.parametrize(
    ("name", "copy_index"),
    [
        ("23553000_B_FULL_SCAN 1 (1).ply", 1),
        ("23553000_B_FULL_SCAN 1(3).PLY", 3),
        ("23553000_B_FULL_SCAN 1 copy.ply", 1),
        ("23553000_B_FULL_SCAN 1 copy 2.stl", 2),
    ],
)
def test_duplicate_copy_markers(name, copy_index):
    r = parse_scan_filename(name, KNOWN)
    assert (r.status, r.sku, r.copy_index, r.scan_label, r.export_index) == ("OK", "23553000", copy_index, "FULL_SCAN", 1)


def test_directory_and_extension_are_stripped():
    for name in ("/inbox/cell-1/23553000_B.ply", "C:\\inbox\\23553000_B.STL", "23553000_B"):
        r = parse_scan_filename(name, KNOWN)
        assert (r.sku, r.revision) == ("23553000", "B")


def test_sku_may_appear_anywhere():
    r = parse_scan_filename("scan of part 23553000 rev B.ply", KNOWN)
    assert r.status == "OK" and r.sku == "23553000"
    # SKU glued to other text is not a token-boundary match: never guessed.
    assert parse_scan_filename("X23553000_B.ply", KNOWN).status == "UNMATCHED"
    assert parse_scan_filename("23553000B.ply", KNOWN).status == "UNMATCHED"


def test_ambiguous_when_two_distinct_skus_match():
    r = parse_scan_filename("11979003_23553000.ply", KNOWN)
    assert r.status == "AMBIGUOUS" and r.sku is None and r.candidates == ("11979003", "23553000")
    # Separator-insensitive twins are ambiguous too, never resolved by guessing.
    r = parse_scan_filename("ABC_1_SCAN.ply", ["ABC-1", "ABC.1"])
    assert r.status == "AMBIGUOUS"
    # The same SKU twice is not ambiguous.
    assert parse_scan_filename("11979003_C_11979003.ply", KNOWN).status == "OK"


def test_revision_mismatch_and_match_with_expected_revisions():
    expected = {"11979003": "C", "23553000": "A", "38172000": "01", "44211000": None}
    r = parse_scan_filename("BEFORE BENDING  11979003_C.ply", expected)
    assert (r.status, r.revision_status, r.expected_revision) == ("OK", "MATCH", "C")
    r = parse_scan_filename("23553000_B_FULL_SCAN 1.ply", expected)
    assert (r.status, r.revision, r.expected_revision, r.revision_status) == ("REVISION_MISMATCH", "B", "A", "MISMATCH")
    assert r.to_contract().status == "REVISION_MISMATCH"
    r = parse_scan_filename("38172000_01_SCAN_2 1.ply", expected)
    assert (r.status, r.revision_status) == ("OK", "MATCH")
    r = parse_scan_filename("38172000_SCAN_2.ply", expected)
    assert (r.status, r.revision, r.revision_status) == ("OK", None, "NOT_IN_NAME")
    r = parse_scan_filename("44211000_001.ply", expected)
    assert (r.status, r.revision, r.serial) == ("OK", None, "001")


def test_timestamp_forms():
    r = parse_scan_filename("44211000_A_20261017_1015.ply", ["44211000_A"])
    assert r.scan_timestamp == "2026-10-17T10:15:00" and r.serial is None
    r = parse_scan_filename("44211000_A_20261399T101500.ply", ["44211000_A"])  # invalid date -> not a timestamp
    assert r.scan_timestamp is None and r.serial == "20261399T101500"


def test_filename_pattern_mode():
    pattern = r"(?P<sku>\d{8})-(?P<revision>[A-Z])-(?P<serial>SN\d+)"
    r = parse_scan_filename("23553000-B-SN7.ply", KNOWN, pattern=pattern)
    assert (r.status, r.sku, r.revision, r.serial) == ("OK", "23553000", "B", "SN7")
    assert parse_scan_filename("23553000_B.ply", KNOWN, pattern=pattern).status == "UNPARSEABLE"
    assert parse_scan_filename("99999999-B-SN7.ply", KNOWN, pattern=pattern).status == "UNMATCHED"
    r = parse_scan_filename("23553000-B-SN7.ply", {"23553000": "A"}, pattern=pattern)
    assert r.status == "REVISION_MISMATCH"


def test_pattern_without_sku_group_raises():
    with pytest.raises(SkuError) as exc:
        parse_scan_filename("23553000_B.ply", KNOWN, pattern=r"(?P<part>\d+)_B")
    assert exc.value.code == "FILENAME_REGEX_INVALID"
    with pytest.raises(SkuError):
        parse_scan_filename("x.ply", KNOWN, pattern=r"(?P<sku>")


def test_invalid_known_sku_raises():
    with pytest.raises(SkuError):
        parse_scan_filename("x.ply", ["bad/sku"])


def test_contract_round_trip():
    r = parse_scan_filename("23553000_B_FULL_SCAN 1 (1).ply", KNOWN)
    c = r.to_contract()
    assert (c.status, c.sku, c.revision, c.scan_label, c.export_index, c.copy_index) == (
        "OK", "23553000", "B", "FULL_SCAN", 1, 1)
    assert r.to_dict()["candidates"] == []


# --------------------------------------------------------------------------
# Explicit revision markers, Windows copies, numeric revisions (review fixes)
# --------------------------------------------------------------------------

EXPECTED = {"11979003": "C", "38172000": "01", "44211000": None, "55500000": "1"}


@pytest.mark.parametrize(
    ("name", "status", "revision", "revision_status", "serial"),
    [
        # explicit REV marker: the mismatch is caught, "REV" is never a serial
        ("11979003-REV-D.ply", "REVISION_MISMATCH", "D", "MISMATCH", None),
        ("11979003_REV_C.ply", "OK", "C", "MATCH", None),
        ("11979003_REVD.ply", "REVISION_MISMATCH", "D", "MISMATCH", None),
        ("11979003 REVISION C.ply", "OK", "C", "MATCH", None),
        # marker after other tokens still counts
        ("11979003_C_SN0042_REV_D.ply", "REVISION_MISMATCH", "D", "MISMATCH", "SN0042"),
        # bare R marker only with a known expected revision
        ("11979003_R_D.ply", "REVISION_MISMATCH", "D", "MISMATCH", None),
        ("11979003_R_C.ply", "OK", "C", "MATCH", None),
        # numeric revisions compare by value; a single digit is a revision only off whitespace
        ("38172000_1_SCAN_2 1.ply", "OK", "1", "MATCH", None),
        ("38172000_2_SCAN_2 1.ply", "REVISION_MISMATCH", "2", "MISMATCH", None),
        ("38172000_REV01.ply", "OK", "01", "MATCH", None),
        ("55500000_01.ply", "OK", "01", "MATCH", None),
    ],
)
def test_explicit_and_numeric_revisions(name, status, revision, revision_status, serial):
    r = parse_scan_filename(name, EXPECTED)
    assert (r.status, r.revision, r.revision_status, r.serial) == (status, revision, revision_status, serial)


def test_single_digit_export_counter_is_never_a_revision():
    for name in ("38172000 1.ply", "55500000 1.ply"):
        r = parse_scan_filename(name, EXPECTED)
        assert (r.status, r.revision, r.revision_status, r.export_index) == ("OK", None, "NOT_IN_NAME", 1)


def test_two_conflicting_revision_readings_are_a_mismatch():
    r = parse_scan_filename("11979003_C_REV_D.ply", EXPECTED)
    assert (r.status, r.revision) == ("REVISION_MISMATCH", "D")
    # without an expected revision nothing is guessed
    r = parse_scan_filename("44211000_A_REV_B.ply", EXPECTED)
    assert (r.status, r.revision, r.revision_status) == ("OK", None, "UNVERIFIED")


@pytest.mark.parametrize(
    ("name", "copy_index"),
    [
        ("11979003_C - Copy.ply", 1),
        ("11979003_C - Copy (2).ply", 2),
        ("11979003_C copy 3.ply", 3),
        ("11979003_C (4).ply", 4),
    ],
)
def test_windows_and_mac_copy_markers(name, copy_index):
    r = parse_scan_filename(name, EXPECTED)
    assert (r.status, r.sku, r.revision, r.serial, r.copy_index) == ("OK", "11979003", "C", None, copy_index)


# -- second review: serial labels are not revisions; conflicting readings ---


@pytest.mark.parametrize(
    ("name", "known", "status", "revision", "revision_status", "serial"),
    [
        # 'SN' is a serial label, not revision 'SN': no false quarantine
        ("11979003_SN_0042.ply", EXPECTED, "OK", None, "NOT_IN_NAME", "0042"),
        ("44211000_SN_0042.ply", EXPECTED, "OK", None, "NOT_IN_NAME", "0042"),
        ("11979003_NO 0042.ply", EXPECTED, "OK", None, "NOT_IN_NAME", "0042"),
        ("11979003_C_SN_0042.ply", EXPECTED, "OK", "C", "MATCH", "0042"),
        # a real wrong revision before the serial is still caught
        ("11979003_D_SN_0042.ply", EXPECTED, "REVISION_MISMATCH", "D", "MISMATCH", "0042"),
        # an explicit marker still reads a label word, and the expected revision always matches
        ("11979003_REV_NO.ply", EXPECTED, "REVISION_MISMATCH", "NO", "MISMATCH", None),
        ("77700000_OK.ply", {"77700000": "OK"}, "OK", "OK", "MATCH", None),
        # no expected revision: two readings conflict -> UNVERIFIED with no revision (never a guess)
        ("11979003_C_REV_D.ply", ["11979003"], "OK", None, "UNVERIFIED", None),
        ("11979003_C_REV_C.ply", ["11979003"], "OK", "C", "UNVERIFIED", None),
    ],
)
def test_serial_labels_and_conflicting_readings(name, known, status, revision, revision_status, serial):
    r = parse_scan_filename(name, known)
    assert (r.status, r.revision, r.revision_status, r.serial) == (status, revision, revision_status, serial)
