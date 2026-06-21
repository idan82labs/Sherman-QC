from __future__ import annotations

from domains.manual_assistant.models import ManualProfile


PROFILE_LABELS: dict[ManualProfile, str] = {
    "cell_operation": "Cell Operation",
    "software": "Software",
}


def assert_profile(value: str) -> ManualProfile:
    if value not in PROFILE_LABELS:
        raise ValueError(f"Unsupported manual assistant profile: {value}")
    return value  # type: ignore[return-value]
