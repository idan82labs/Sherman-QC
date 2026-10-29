from scripts.run_f1_v56_source_free_preflight import run_suite


def test_v56_fixed_source_free_suite() -> None:
    result = run_suite()
    assert result["status"] == "PASS"
    assert result["fixture_count"] >= 14
