from research.f1_gcdf_v1.config import CONFIG, ATTRITION_TOKENS, validate_config
from research.f1_gcdf_v1.core import ceil_sqrt_q16, lower_median, ppm
from research.f1_gcdf_v1.tamper import assert_tamper_suite
from scripts.run_f1_gcdf_v1_source_free_regressions import run_regressions


def test_configuration_is_frozen_and_target_closed() -> None:
    validate_config()
    assert CONFIG.sample_count == 200_000
    assert CONFIG.landmark_count == 160
    assert CONFIG.target_capability_opened is False
    assert len(ATTRITION_TOKENS) == 27


def test_exact_integer_primitives() -> None:
    assert ceil_sqrt_q16(1) == 1 << 16
    assert ceil_sqrt_q16(2) ** 2 >= 2 << 32
    assert ppm(3, 1000) == 3000
    assert lower_median([4, 1, 3, 2]) == 2


def test_tamper_suite_rejects_every_mutation() -> None:
    assert all(assert_tamper_suite().values())


def test_mandatory_source_free_regressions() -> None:
    result = run_regressions()
    assert result["source_opened"] is False
    assert result["target_gold_cad_benchmark_opened"] is False
    assert result["passed"] is True
