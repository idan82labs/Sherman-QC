from scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v3 import (
    CAPTURE_INDICES,
    CORPUS_KIND,
    FAMILIES,
    TOKEN,
)


def test_v3_complex_geometry_contract_is_frozen() -> None:
    assert CAPTURE_INDICES == (6, 7)
    assert CORPUS_KIND.endswith("complex_geometry_v3")
    assert "NON_CONFIRMATORY" in TOKEN
    assert len(FAMILIES) == 15
    assert len({family["id"] for family in FAMILIES}) == 15
    assert min(float(family["radius"]) for family in FAMILIES) <= 0.55
    assert max(float(family["radius"]) for family in FAMILIES) >= 9.0
    assert max(len(family["profile"]) - 2 for family in FAMILIES) >= 11
    assert all(family["challenge"] for family in FAMILIES)
