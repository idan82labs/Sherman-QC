from __future__ import annotations

import pytest

from scripts.render_f1_line_audit import _select_owned_regions


def test_select_owned_regions_preserves_manifest_order() -> None:
    regions = [{"bend_id": "OB1"}, {"bend_id": "OB2"}, {"bend_id": "OB3"}]

    assert _select_owned_regions(regions, {"OB3", "OB1"}) == [regions[0], regions[2]]


def test_select_owned_regions_fails_closed_for_unknown_identifier() -> None:
    with pytest.raises(ValueError, match="OB9"):
        _select_owned_regions([{"bend_id": "OB1"}], {"OB1", "OB9"})
