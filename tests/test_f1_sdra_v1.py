import numpy as np

from research.f1_sdra_v1.config import CONFIG,CONFIG_SHA256,validate_config
from research.f1_sdra_v1.core import canonicalize_sampled_xyz,exact_knn_kdtree,q32
from research.f1_sdra_v1.tamper import assert_tamper_suite
from scripts.run_f1_sdra_v1_source_free_regressions import run_regressions


def test_frozen_configuration_and_closed_capabilities() -> None:
    validate_config()
    assert CONFIG.sample_count == 200_000
    assert CONFIG.authoritative_png_count == 77
    assert CONFIG.target_capability_granted is False
    assert len(CONFIG_SHA256) == 64


def test_nearest_even_q32() -> None:
    assert q32(np.asarray([0.0, 0.5, 1.0])).tolist() == [0, 1 << 31, 1 << 32]


def test_exact_integer_neighbor_ties_use_point_id() -> None:
    u=np.arange(21,dtype=float)-10; x,y=np.meshgrid(u,u,indexing="ij"); xyz=np.stack((x.ravel(),y.ravel(),np.zeros(x.size)),axis=1)
    source=canonicalize_sampled_xyz(xyz)
    ids,squared=exact_knn_kdtree(source,anchors=np.asarray([220],np.uint32),k=384)
    assert list(zip(squared[0].tolist(),ids[0].tolist())) == sorted(zip(squared[0].tolist(),ids[0].tolist()))


def test_tamper_suite() -> None:
    assert all(assert_tamper_suite().values())


def test_complete_source_free_regression_package() -> None:
    result=run_regressions()
    assert result["source_opened"] is False
    assert result["target_gold_cad_benchmark_gcdf_opened"] is False
    assert result["passed"] is True
