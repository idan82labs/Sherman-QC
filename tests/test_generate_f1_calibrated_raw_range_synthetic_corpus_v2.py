from scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v2 import CAPTURE_INDICES, CORPUS_KIND


def test_v2_replication_identity_is_frozen() -> None:
    assert CAPTURE_INDICES == (2, 3)
    assert CORPUS_KIND.endswith("capture_replication_v2")
