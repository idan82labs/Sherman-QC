from __future__ import annotations

import json

import numpy as np
import pytest

from backend.f1_v34_artifact_scalar_normalizer import (
    ArtifactScalarNormalizationError,
    normalize_numpy_scalars,
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def test_nested_approved_numpy_scalars_match_manual_native_payload() -> None:
    numpy_payload = {
        "decision": {
            "passes": [np.bool_(True), np.bool_(False)],
            "counts": (np.int8(-2), np.uint64(9)),
            "scores": [np.float16(0.5), np.float64(1.25)],
        }
    }
    native_payload = {
        "decision": {
            "passes": [True, False],
            "counts": (-2, 9),
            "scores": [0.5, 1.25],
        }
    }

    assert canonical(normalize_numpy_scalars(numpy_payload)) == canonical(native_payload)


def test_python_native_payload_is_byte_identical_through_wrapper() -> None:
    payload = {
        "z": None,
        "a": [True, 7, -0.25, "unchanged"],
        "nested": {"tuple": (3, 2, 1)},
    }

    assert canonical(normalize_numpy_scalars(payload)) == canonical(payload)


def test_structure_order_fields_and_values_are_preserved() -> None:
    payload = {
        "third": [np.int32(3), np.int32(1), np.int32(2)],
        "first": (np.bool_(False), "sentinel"),
        "second": {"b": np.float32(2.5), "a": np.uint16(4)},
    }
    normalized = normalize_numpy_scalars(payload)

    assert list(normalized) == ["third", "first", "second"]
    assert list(normalized["second"]) == ["b", "a"]
    assert isinstance(normalized["first"], tuple)
    assert normalized == {
        "third": [3, 1, 2],
        "first": (False, "sentinel"),
        "second": {"b": 2.5, "a": 4},
    }


@pytest.mark.parametrize("value", [np.float32(np.nan), np.float64(np.inf), np.float64(-np.inf)])
def test_nonfinite_approved_scalars_still_fail_canonical_json(value: np.floating) -> None:
    normalized = normalize_numpy_scalars({"value": value})
    with pytest.raises(ValueError):
        canonical(normalized)


@pytest.mark.parametrize(
    "value",
    [
        np.asarray([1, 2], dtype=np.int64),
        np.complex64(1 + 2j),
        np.datetime64("2026-08-14"),
        np.timedelta64(3, "D"),
        np.str_("numpy-string"),
        np.bytes_(b"numpy-bytes"),
        np.void(b"opaque"),
        object(),
    ],
)
def test_unsupported_values_fail_closed(value: object) -> None:
    with pytest.raises(ArtifactScalarNormalizationError):
        normalize_numpy_scalars({"value": value})


@pytest.mark.parametrize("key", [1, np.int64(1), np.str_("key")])
def test_dictionary_keys_are_never_converted(key: object) -> None:
    with pytest.raises(ArtifactScalarNormalizationError):
        normalize_numpy_scalars({key: np.bool_(True)})
