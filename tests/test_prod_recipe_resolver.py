"""C05 resolver: scan filename -> active recipe (C02 ports), recipe provider, cache invalidation.

Store-level tests: recipes from the C01 builder on synthetic dicts with a
recorded compile (no CAD, no subprocess). Filename rules come from C01
``parse_scan_filename`` through C02's ``FilenameRecipeResolver``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apps.production_line.ingest.ports import (
    ActiveRecipeCatalog,
    RecipeProvider,
    ScanRecipeResolver,
)
from apps.production_line.recipes.release_service import (
    add_release_listener,
    notify_release_listeners,
    remove_release_listener,
)
from apps.production_line.recipes.resolver import LineRecipeResolver, catalog_sku_source
from apps.production_line.recipes.store import CompiledRecord, RecipeStore
from tests.test_prod_recipe_store import compiled_for, make_recipe


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def env(tmp_path: Path):
    store = RecipeStore(tmp_path / "db" / "recipes.sqlite3")
    clock = Clock()
    resolver = LineRecipeResolver(store, data_root=tmp_path, monotonic=clock)
    return store, resolver, clock, tmp_path


def test_ports_are_implemented(env) -> None:
    _, resolver, _, _ = env
    assert isinstance(resolver, ScanRecipeResolver)
    assert isinstance(resolver, ActiveRecipeCatalog)
    assert isinstance(resolver, RecipeProvider)


def test_matched_unmatched_not_ready(env) -> None:
    store, resolver, _, _ = env
    recipe = make_recipe()
    store.insert_recipe(recipe, compiled=compiled_for(recipe))
    store.register_sku("OTHER-9", "p2")  # registered, never released
    out = resolver.resolve("SYN-HAT-1_SN7_20261017T101500.ply")
    assert out.status == "MATCHED"
    assert (out.recipe_id, out.recipe_version, out.recipe_sha256) == ("SYN-HAT-1@v1", 1, recipe.recipe_sha256)
    assert out.compiled_part_sha256 == "d" * 64 and out.serial == "SN7"
    assert out.scan_timestamp is not None and out.scan_timestamp.startswith("2026-10-17T10:15")
    assert resolver.resolve("NOPE.ply").status == "UNMATCHED"
    nr = resolver.resolve("OTHER-9_FULL_SCAN 1.ply")
    assert nr.status == "PART_NOT_READY" and nr.reason == "NO_ACTIVE_RECIPE"
    # Real ZEISS shapes resolve too (copy markers, prefix words, double spaces).
    for name in ("BEFORE BENDING  SYN-HAT-1_A.ply", "SYN-HAT-1_A_FULL_SCAN 1 (1).ply", "SYN-HAT-1 - Copy.stl"):
        assert resolver.resolve(name).status == "MATCHED", name


def test_revision_mismatch_is_reported(env) -> None:
    store, resolver, _, _ = env
    recipe = make_recipe(revision="C")
    store.insert_recipe(recipe, compiled=compiled_for(recipe))
    assert resolver.resolve("SYN-HAT-1_C_FULL_SCAN 1.ply").status == "MATCHED"
    out = resolver.resolve("SYN-HAT-1_D_FULL_SCAN 1.ply")
    assert out.status == "REVISION_MISMATCH" and out.recipe_id is None
    assert resolver.resolve("SYN-HAT-1-REV-D.ply").status == "REVISION_MISMATCH"


def test_only_retired_or_uncompiled_is_not_ready(env) -> None:
    store, resolver, clock, _ = env
    v1 = make_recipe()
    store.insert_recipe(v1, compiled=compiled_for(v1))
    store.retire("SYN-HAT-1@v1")
    clock.t += 11  # past the 10 s cache
    out = resolver.resolve("SYN-HAT-1_SN1.ply")
    assert out.status == "PART_NOT_READY" and out.reason == "NO_ACTIVE_RECIPE"
    v2 = make_recipe(version=2, flange_plus=0.9)
    store.insert_recipe(v2, compiled=compiled_for(v2, "BLOCKED"))
    clock.t += 11
    out = resolver.resolve("SYN-HAT-1_SN1.ply")
    assert out.status == "PART_NOT_READY" and out.reason == "RECIPE_COMPILE_BLOCKED"
    assert out.recipe_id == "SYN-HAT-1@v2"  # named for the operator, never enqueued
    v3 = make_recipe(version=3, flange_plus=1.1)
    store.insert_recipe(v3)  # no compile recorded at all
    clock.t += 11
    assert resolver.resolve("SYN-HAT-1_SN1.ply").reason == "RECIPE_COMPILE_PENDING"


def test_cache_ttl_and_release_invalidation(env) -> None:
    store, resolver, clock, _ = env
    assert resolver.resolve("SYN-HAT-1_SN1.ply").status == "UNMATCHED"
    recipe = make_recipe()
    store.insert_recipe(recipe, compiled=compiled_for(recipe))
    assert resolver.resolve("SYN-HAT-1_SN1.ply").status == "UNMATCHED"  # cached snapshot (10 s)
    add_release_listener(resolver.invalidate)
    try:
        notify_release_listeners(recipe)
        assert resolver.resolve("SYN-HAT-1_SN1.ply").status == "MATCHED"
    finally:
        remove_release_listener(resolver.invalidate)
    store.retire("SYN-HAT-1@v1")
    assert resolver.resolve("SYN-HAT-1_SN1.ply").status == "MATCHED"  # still cached
    clock.t += 10.5
    assert resolver.resolve("SYN-HAT-1_SN1.ply").status == "PART_NOT_READY"


def test_catalog_skus_are_known_but_not_ready(env) -> None:
    store, _, clock, root = env

    class Catalog:
        def list_parts(self, limit=100):
            return [{"part_number": "cat 77"}, {"part_number": "bad/sku"}, {"part_number": None}]

    resolver = LineRecipeResolver(store, data_root=root, extra_skus=catalog_sku_source(Catalog()), monotonic=clock)
    assert resolver.list_registered_skus() == ["CAT-77"]  # invalid part numbers skipped, never guessed
    out = resolver.resolve("CAT-77_FULL_SCAN 1.ply")
    assert out.status == "PART_NOT_READY"


def test_ambiguous_and_unparseable_and_regex(env) -> None:
    store, _, clock, root = env
    for sku, pid in (("AB-1", "p1"), ("CD-2", "p2")):
        r = make_recipe(sku=sku, part_id=pid)
        store.insert_recipe(r, compiled=compiled_for(r))
    resolver = LineRecipeResolver(store, data_root=root, monotonic=clock)
    assert resolver.resolve("AB-1_CD-2.ply").status == "AMBIGUOUS"
    assert resolver.resolve("").status in ("UNPARSEABLE", "UNMATCHED")
    patterned = LineRecipeResolver(store, r"PFX_(?P<sku>[A-Z0-9-]+?)_.*", data_root=root, monotonic=clock)
    assert patterned.resolve("PFX_CD-2_x.ply").recipe_id == "CD-2@v1"


def test_load_returns_absolute_paths_and_keyerror(env) -> None:
    store, resolver, _, root = env
    recipe = make_recipe()
    store.insert_recipe(recipe, compiled=compiled_for(recipe))
    loaded = resolver.load("SYN-HAT-1@v1")
    assert loaded.recipe_json_bytes == store.get("SYN-HAT-1@v1").recipe_json
    assert loaded.cad_path == (root / recipe.cad.stored_rel_path).resolve() and loaded.cad_path.is_absolute()
    assert loaded.compiled_dir == (root / f"recipes/compiled/{recipe.recipe_sha256}").resolve()
    assert loaded.compiled_part_sha256 == "d" * 64 and loaded.flat_path is None
    with pytest.raises(KeyError):
        resolver.load("SYN-HAT-1@v9")
    # A retired recipe stays loadable (jobs queued before retirement still run or fail explicitly).
    store.retire("SYN-HAT-1@v1")
    assert resolver.load("SYN-HAT-1@v1").extra["status"] == "retired"


def test_resolve_never_raises(env, monkeypatch) -> None:
    store, resolver, _, _ = env
    monkeypatch.setattr(store, "list_active", lambda: (_ for _ in ()).throw(RuntimeError("db gone")))
    out = resolver.resolve("SYN-HAT-1_SN1.ply")
    assert out.status == "UNPARSEABLE" and out.reason.startswith("RESOLVER_ERROR")


def test_store_expectations_sets_ready(env) -> None:
    store, resolver, _, _ = env
    recipe = make_recipe()
    store.insert_recipe(recipe, compiled=CompiledRecord(recipe_sha256=recipe.recipe_sha256, status="PENDING"))
    resolver.store_expectations(recipe.recipe_sha256, "e1-test", {"bends": []}, None)
    assert store.get_expectations(recipe.recipe_sha256, "e1-test")["status"] == "READY"
