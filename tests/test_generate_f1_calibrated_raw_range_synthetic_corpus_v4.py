from scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v4 import CAPTURE_INDICES, FAMILIES, TOKEN


def test_v4_family_freeze_is_new_and_complex() -> None:
    assert TOKEN.endswith("NON_CONFIRMATORY")
    assert CAPTURE_INDICES == (8, 9)
    assert len(FAMILIES) == 15
    assert len({row["id"] for row in FAMILIES}) == 15
    assert all(row["id"].startswith("PF") for row in FAMILIES)
    assert all(len(row["profile"]) >= 10 for row in FAMILIES)
    assert min(row["radius"] for row in FAMILIES) <= 0.5
    assert max(row["radius"] for row in FAMILIES) >= 9.5
    assert all(len(row["challenge"]) >= 24 for row in FAMILIES)
