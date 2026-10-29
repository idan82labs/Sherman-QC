import copy
import pytest
from tests.test_vendor_cad_bend_candidates_v3 import axis
from backend.vendor_cad_bend_candidates_v4 import axis_rows_v4, extraction_to_candidates
from backend.vendor_unfold_contract_v2 import PreviewContractError


def feature():
    a = axis()
    frame = {"axis": [0., 0., 1.], "x_axis": [1., 0., 0.], "y_axis": [0., 1., 0.]}
    a["carrier_receipt"].update(angular_frame=frame,
        source_angular_frames=[{"face_index": i, "source_frame": copy.deepcopy(frame)} for i in [2, 5]])
    return a


def test_explicit_frame_and_source_ownership_survive_adapter():
    a = feature(); old = copy.deepcopy(a)
    row = axis_rows_v4([a])[0]
    assert row["v4_provenance"]["extractor_schema_version"] == 4
    assert row["v4_provenance"]["angular_frame"] == a["carrier_receipt"]["angular_frame"]
    assert "v3_provenance" not in row and a == old


@pytest.mark.parametrize("mutation", ["missing", "reflection", "nonunit", "ownership"])
def test_refuse_invalid_frame(mutation):
    a = feature(); r = a["carrier_receipt"]
    if mutation == "missing": del r["angular_frame"]
    elif mutation == "reflection": r["angular_frame"]["axis"] = [0., 0., -1.]
    elif mutation == "nonunit": r["angular_frame"]["x_axis"] = [2., 0., 0.]
    else: r["source_angular_frames"].pop()
    with pytest.raises(PreviewContractError): axis_rows_v4([a])


def test_schema_is_explicit_not_legacy_relabel():
    with pytest.raises(PreviewContractError):
        extraction_to_candidates({"schema_version": 3, "kind": "sherman_qc_step_topology_bend_axes", "bend_axes": [feature()]})
    assert len(extraction_to_candidates({"schema_version": 4, "kind": "sherman_qc_step_topology_bend_axes", "bend_axes": [feature()]})["candidates"]) == 1
