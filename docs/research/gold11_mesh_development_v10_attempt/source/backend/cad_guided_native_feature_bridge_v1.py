"""Native feature bridge V1: explicit, lossless kernel-V1 -> kernel-V2 feature reconstruction.

Both accepted native adapter branches (`cad_guided_native_features` r0 and
`cad_guided_native_features_v1`) construct kernel-V1 `CADGuidedFeature` objects.
Support cover V3 / context measurement V2 require the kernel-V2 class by exact
identity (the two dataclasses have identical fields but different Python
identities), which refused the only V9 slot that reached measurement
(`INVALID_SUPPORT_COVER_FEATURES`). This adapter runs immediately after the
UNCHANGED factory returns: it requires exact kernel-V1 objects and rebuilds a
kernel-V2 object from every dataclass field in declared order. No class aliasing,
no geometry rederivation, no omitted or defaulted field, no truncation: tuple
order and count, feature ids, nominal source, signed sweep, orientation and span
are copied verbatim, and the factory's metadata (candidates, windows, owners,
faces, edges, trim exclusions, PARTIAL/unavailable reasons) is untouched.

The bridge establishes no material-side, unit/frame or measurement authority;
the existing context transform remains the sole feature-to-scan coordinate
conversion. Any wrong class, schema drift, duplicate id or over-limit set refuses
BEFORE costly registration with the native diagnostics retained and no partial
prefix. The frozen factories and the downstream exact-type checks are untouched.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json
import math

from .cad_guided_measurement import CADGuidedFeature as _KernelV1Feature
from .cad_guided_measurement_v2 import CADGuidedFeature as _KernelV2Feature

BRIDGE_IMPLEMENTATION = "NATIVE_FEATURE_BRIDGE_V1_EXACT_KERNEL_V1_TO_KERNEL_V2_LOSSLESS_FIELD_RECONSTRUCTION"
SOURCE_CLASS = "backend.cad_guided_measurement.CADGuidedFeature"
TARGET_CLASS = "backend.cad_guided_measurement_v2.CADGuidedFeature"
FIELD_NAMES = tuple(field.name for field in fields(_KernelV1Feature))
# The consumer boundary is the support cover V3 feature grammar (`1 <= len(features) <= 64`,
# cover V3 line 143; context V2 itself admits up to 512 records per call), NOT the standalone
# kernel entrypoint cap of 16: a cohort case with 17 native windows (CG4BDD in V9) must bridge
# whole. 65 or more refuse whole before any costly work; nothing is truncated. The boundary is
# asserted against the real cover in the factory-to-measurement test.
MAX_BRIDGED_FEATURES = 64
AUTHORITY = "NO_MATERIAL_SIDE_UNIT_FRAME_OR_MEASUREMENT_AUTHORITY_FIELD_COPY_ONLY"
COORDINATE_CONVERSION = "NONE_THE_CONTEXT_TRANSFORM_REMAINS_THE_SOLE_FEATURE_TO_SCAN_CONVERSION"


class BridgeRefusal(ValueError):
    """Raised before any costly work; the native factory result is retained unchanged."""


def _require(condition, reason):
    if not condition:
        raise BridgeRefusal(reason)


def _canonical(value):
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        _require(math.isfinite(value), "NON_FINITE_NATIVE_FEATURE_FIELD")
    return value


def _field_digest(values):
    encoded = json.dumps({name: _canonical(values[name]) for name in FIELD_NAMES}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def bridge_native_features_v1(extracted):
    """Return (kernel-V2 feature tuple, provenance record) for an unchanged factory result.

    `extracted` is the factory's own result dict (`status`, `features`, `metadata`) and is
    never mutated. An empty feature tuple bridges to an empty tuple (the caller keeps its
    unchanged NATIVE_FEATURES_UNAVAILABLE handling).
    """
    _require(type(extracted) is dict and type(extracted.get("status")) is str
             and type(extracted.get("metadata")) is dict, "INVALID_NATIVE_FACTORY_RESULT")
    features = extracted.get("features")
    _require(type(features) is tuple, "NATIVE_FEATURES_NOT_A_TUPLE")
    _require(len(features) <= MAX_BRIDGED_FEATURES, "NATIVE_FEATURE_SET_OVER_LIMIT_NOT_TRUNCATED")
    _require(tuple(field.name for field in fields(_KernelV2Feature)) == FIELD_NAMES,
             "KERNEL_FEATURE_FIELD_GRAMMAR_DRIFT")
    seen, rebuilt, records = set(), [], []
    for index, feature in enumerate(features):
        _require(type(feature) is _KernelV1Feature, "NATIVE_FEATURE_NOT_EXACT_KERNEL_V1_CLASS")
        values = {name: getattr(feature, name) for name in FIELD_NAMES}
        _require(feature.feature_id not in seen, "DUPLICATE_NATIVE_FEATURE_ID")
        seen.add(feature.feature_id)
        digest = _field_digest(values)
        target = _KernelV2Feature(**values)          # V2 __post_init__ re-validates; nothing is rederived
        _require(type(target) is _KernelV2Feature and asdict(target) == asdict(feature)
                 and tuple(asdict(target)) == FIELD_NAMES, "LOSSLESS_RECONSTRUCTION_MISMATCH")
        rebuilt.append(target)
        records.append(dict(index=index, feature_id=feature.feature_id, nominal_source=feature.nominal_source,
                            nominal_radius_mm=feature.nominal_radius_mm, nominal_sweep_deg=feature.nominal_sweep_deg,
                            span_mm=list(feature.span_mm), field_digest_sha256=digest))
    provenance = dict(implementation=BRIDGE_IMPLEMENTATION, source_class=SOURCE_CLASS, target_class=TARGET_CLASS,
        field_names=list(FIELD_NAMES), feature_count=len(rebuilt), features=records,
        native_status=extracted["status"], native_metadata_retained_unchanged=True,
        authority=AUTHORITY, coordinate_conversion=COORDINATE_CONVERSION,
        truncation="NEVER_OVER_LIMIT_SETS_REFUSE_WHOLE")
    return tuple(rebuilt), provenance
