"""Pure marker identity adversaries, no native imports or physical inputs."""
import copy

import pytest

from backend.vendor_manufacturing_bridge import bind_vendor_bend_ids, ManufacturingBridgeRefused

HASH = "a" * 64


def rows():
    native = [{"pair_id": f"pair-{i}", "line_start": [i*10, 0, 0], "line_end": [i*10, 20, 0],
               "source_sha256": HASH} for i in range(3)]
    vendor = [{"feature_id": f"B{i+1}", "line_start": list(row["line_end"]), "line_end": list(row["line_start"]),
               "source_sha256": HASH} for i, row in enumerate(native)]
    return native, list(reversed(vendor))


def test_bijective_orientation_order_independent_source_identity():
    native, vendor = rows()
    result = bind_vendor_bend_ids(native, vendor, HASH)
    assert [row["feature_id"] for row in result["assignments"]] == ["B1", "B2", "B3"]
    assert result["all_native_and_vendor_bends_bijective"]
    assert not result["manufacturing_ready"]


@pytest.mark.parametrize("mutation", ["far", "tiny_drift", "duplicate", "source", "missing", "nan", "zero"])
def test_no_best_available_or_incomplete_mapping(mutation):
    native, vendor = copy.deepcopy(rows())
    if mutation in {"far", "tiny_drift"}:
        vendor[0]["line_start"][0] += 1000000 if mutation == "far" else 1e-5
    elif mutation == "duplicate":
        vendor[1]["line_start"] = vendor[0]["line_start"]
        vendor[1]["line_end"] = vendor[0]["line_end"]
    elif mutation == "source":
        vendor[0]["source_sha256"] = "b" * 64
    elif mutation == "missing":
        vendor.pop()
    elif mutation == "nan":
        vendor[0]["line_start"][0] = float("nan")
    elif mutation == "zero":
        native[0]["line_end"] = native[0]["line_start"]
    with pytest.raises(ManufacturingBridgeRefused):
        bind_vendor_bend_ids(native, vendor, HASH)
