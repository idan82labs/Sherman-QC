import copy
import json
import numpy as np
import pytest
from tests.test_vendor_cad_bend_candidates_v5 import feature
from scripts.extract_step_topology_bend_axes_v5 import certify_rectangle_segments
from backend.vendor_cad_bend_candidates_v6 import axis_rows_v6
from backend.vendor_unfold_contract_v2 import PreviewContractError


def test_in_memory_native_residual_matches_json_replay_without_mutation():
    axis = feature()
    # Source coefficient certificate, not a handwritten all-zero receipt.
    points = [[0., 0.], [3., 1e-9], [3., 40.], [0., 40.]]
    segments = [{"start": p, "end": q, "edge_id": i+1}
                for i, (p, q) in enumerate(zip(points, points[1:]+points[:1]))]
    proof = certify_rectangle_segments(segments, 1e-7)
    assert proof["status"] == "PROVED" and proof["maximum_residual_mm"] > 0
    for certificate in axis["domain_receipt"]["native_domain_certificates"]:
        certificate.update(proof)
        certificate["maximum_residual_mm"] = np.float64(certificate["maximum_residual_mm"])
    old = copy.deepcopy(axis)
    native = axis_rows_v6([axis])
    serialized = axis_rows_v6(json.loads(json.dumps([axis])))
    assert native == serialized and axis == old
    assert type(axis["domain_receipt"]["native_domain_certificates"][0]["maximum_residual_mm"]) is np.float64
    assert native[0]["v5_provenance"]["adapter_version"] == 6


@pytest.mark.parametrize("value", [np.float64(float("nan")), np.float64(float("inf")),
    np.float64(-1.), np.float64(1.), np.bool_(True), True, "0"])
def test_bad_residuals_remain_refused(value):
    axis = feature()
    axis["domain_receipt"]["native_domain_certificates"][0]["maximum_residual_mm"] = value
    with pytest.raises(PreviewContractError): axis_rows_v6([axis])


@pytest.mark.parametrize("field,value", [("face_index", np.float64(2)),
    ("face_index", np.int64(2)), ("wire_count", True), ("wire_count", np.float64(1))])
def test_identity_scalars_are_never_normalized(field, value):
    axis = feature()
    axis["domain_receipt"]["native_domain_certificates"][0][field] = value
    with pytest.raises(PreviewContractError): axis_rows_v6([axis])
