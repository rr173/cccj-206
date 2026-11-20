"""Idempotency, source conflicts and dataset compatibility for the cube."""

from __future__ import annotations

import json

import pytest

from grib_asm.cube.errors import (
    CubeCompatibilityError, CubeConflictError,
)
from grib_asm.cube.importer import import_package
from grib_asm.cube.query import CubeReader
from grib_asm.cube.store import CUBE_DIR, CubeConfig

from cube_helpers import (
    build_package, make_request, real_package_from_message,
    standard_grid_digest,
)
from e2e_helpers import HAS_ECCODES

pytestmark = pytest.mark.skipif(
    not HAS_ECCODES, reason="ecCodes required to encode/decode fixtures")


def test_same_package_reimport_is_idempotent(tmp_path):
    grid_dig = standard_grid_digest()
    request = make_request("idem", grid_dig)
    pkg = build_package(tmp_path, 0, request)
    store = tmp_path / "cube"
    first = import_package(store, pkg)
    second = import_package(store, pkg)
    third = import_package(store, pkg)
    assert first.status == "committed"
    assert second.status == "idempotent"
    assert third.status == "idempotent"

    reader = CubeReader(store)
    assert reader.origins() == ["2026-01-01T00:00:00Z"]
    # Logical content unchanged: provenance still names the one digest.
    from grib_asm.identity import VariableId
    result = reader.select(VariableId(0, 0, 0, 29, 1, 7, 0))
    assert result.present.sum() == result.present.size
    sources = json.loads((store / CUBE_DIR / "registry"
                          / "sources.json").read_text())
    assert len(sources) == 1


def test_same_origin_different_digest_conflicts_and_keeps_old(tmp_path):
    grid_dig = standard_grid_digest()
    request = make_request("conf", grid_dig)
    pkg_a = build_package(tmp_path, 0, request)
    store = tmp_path / "cube"
    assert import_package(store, pkg_a).status == "committed"

    # Rebuild a byte-different package for the same reference time by using a
    # separate catalog/output path but identical messages is not enough (same
    # digest); instead mutate one encoded message via a different bitsPerValue,
    # which changes the message digest while preserving grid/identity.
    from cube_helpers import build_cycle_specs, write_collection
    from grib_asm.catalog import Catalog
    from grib_asm.assemble import assemble
    specs = build_cycle_specs(0)
    # Re-encode every message with different packing -> new message digests,
    # same Section 3 grid digest and same business keys.
    from fixturegen import encode_message
    mutated = []
    for li, mi, si, slot, spec in specs:
        spec.bits_per_value = 12
        mutated.append((li, mi, si, slot, spec))
    root, manifest = write_collection(tmp_path, "cyc0b", mutated)
    cat = Catalog(tmp_path / "catb.db")
    cat.ingest_file(manifest, manifest.files[0], root)
    pkg_b = assemble(
        cat, request, {"cyc0b": root}, tmp_path / "packages_b",
        {"cyc0b": manifest}).package_dir
    cat.close()

    marker_a = json.loads(
        (pkg_a / "PACKAGE_COMPLETE.json").read_text())
    marker_b = json.loads(
        (pkg_b / "PACKAGE_COMPLETE.json").read_text())
    assert marker_a["data_sha256"] != marker_b["data_sha256"]

    with pytest.raises(CubeConflictError) as exc:
        import_package(store, pkg_b)
    assert exc.value.reason == "source_digest_conflict"

    # The old cycle is still fully readable and unchanged.
    reader = CubeReader(store)
    assert reader.origins() == ["2026-01-01T00:00:00Z"]
    from grib_asm.identity import VariableId
    result = reader.select(VariableId(0, 0, 0, 29, 1, 7, 0))
    assert result.present.sum() == result.present.size
    assert np_all_pkg_digest(result, marker_a["data_sha256"])

    # Readable evidence of the rejected source is retained.
    conflict_files = list((store / CUBE_DIR / "conflicts").glob("*.json"))
    assert len(conflict_files) == 1
    evidence = json.loads(conflict_files[0].read_text())
    assert evidence["rejected"]["package_digest"] == marker_b["data_sha256"]
    assert reader.conflicts()[0]["reason"] == "source_digest_conflict"


def np_all_pkg_digest(result, digest):
    import numpy as np
    return bool(np.all(result.package_digest == digest))


def test_different_grid_is_rejected_before_writes(tmp_path):
    grid_dig = standard_grid_digest()
    request = make_request("grid-a", grid_dig)
    pkg_a = build_package(tmp_path, 0, request)
    store = tmp_path / "cube"
    import_package(store, pkg_a)

    # Second cycle on a shifted grid: different Section 3, different digest.
    shifted = dict(first_lat=55.0)
    request_b = make_request(
        "grid-b", standard_grid_digest_override(shifted),
        origin_hour=12)
    pkg_b = build_package(
        tmp_path, 1, request_b, grid_overrides=shifted)
    with pytest.raises(CubeCompatibilityError) as exc:
        import_package(store, pkg_b)
    assert exc.value.reason == "grid_definition_conflict"

    # Nothing from the rejected package is visible.
    reader = CubeReader(store)
    assert reader.origins() == ["2026-01-01T00:00:00Z"]
    assert reader.meta["grid_definition"][
        "latitudeOfFirstGridPointInDegrees"] == 40.0


def standard_grid_digest_override(overrides):
    import io
    from fixturegen import FixtureSpec, encode_message
    from grib_asm.identity import grid_digest
    from grib_asm.scanner import scan_file
    spec = FixtureSpec(ni=4, nj=3, **overrides)
    raw = encode_message(spec)
    res = scan_file(io.BytesIO(raw), len(raw))
    return grid_digest(res.messages[0].gds_bytes)


def test_different_variable_identity_is_separate_not_merged(tmp_path):
    grid_dig = standard_grid_digest()
    # Same shortName-capable variable slot, but parameterNumber 99: a
    # distinct numeric identity that must never merge with parameter 0.
    request0 = make_request("var0", grid_dig, pnum=0)
    request99 = make_request("var99", grid_dig, pnum=99)
    pkg0 = build_package(tmp_path, 0, request0, pnum=0)
    pkg99 = build_package(tmp_path, 1, request99, pnum=99)
    store = tmp_path / "cube"
    import_package(store, pkg0)
    import_package(store, pkg99)

    reader = CubeReader(store)
    variables = reader.variables()
    parameter_numbers = sorted(
        v["identity"]["parameterNumber"] for v in variables)
    assert parameter_numbers == [0, 99]
    assert len({v["key"] for v in variables}) == 2

    from grib_asm.identity import VariableId
    r0 = reader.select(VariableId(0, 0, 0, 29, 1, 7, 0))
    r99 = reader.select(VariableId(0, 0, 99, 29, 1, 7, 0))
    assert r0.variable_identity["parameterNumber"] == 0
    assert r99.variable_identity["parameterNumber"] == 99
    # Both variables span both committed origins.
    assert r0.present.shape == r99.present.shape


def test_validation_failure_changes_nothing(tmp_path):
    grid_dig = standard_grid_digest()
    request = make_request("val", grid_dig)
    pkg = build_package(tmp_path, 0, request)
    store = tmp_path / "cube"
    import_package(store, pkg)

    # Tamper with the data file: verification must fail before any write.
    data = pkg / "messages.grib2"
    blob = bytearray(data.read_bytes())
    blob[-10] ^= 0xFF
    data.write_bytes(bytes(blob))
    from grib_asm.cube.errors import PackageValidationError
    with pytest.raises(PackageValidationError):
        import_package(store, pkg)
    reader = CubeReader(store)
    assert reader.origins() == ["2026-01-01T00:00:00Z"]
    assert not list((store / CUBE_DIR / "txn").iterdir())


def test_tampered_completion_marker_is_refused(tmp_path):
    grid_dig = standard_grid_digest()
    request = make_request("marker", grid_dig)
    pkg = build_package(tmp_path, 0, request)
    store = tmp_path / "cube"
    import_package(store, pkg)

    # Rewrite the marker with a wrong digest; the independent re-hash must
    # catch it before any dataset byte changes.
    marker_path = pkg / "PACKAGE_COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["data_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker))
    from grib_asm.cube.errors import PackageValidationError
    with pytest.raises(PackageValidationError) as exc:
        import_package(store, pkg)
    assert exc.value.reason == "package_verification_failed"
    # The already committed cycle is untouched and readable.
    reader = CubeReader(store)
    assert reader.origins() == ["2026-01-01T00:00:00Z"]


def test_distinct_level_value_is_appended_not_overwritten(tmp_path):
    import io
    from grib_asm.identity import grid_digest
    from grib_asm.scanner import scan_file
    from fixturegen import FixtureSpec, encode_message

    grid_dig = standard_grid_digest()
    request0 = make_request("lvl850", grid_dig)
    pkg0 = build_package(tmp_path, 0, request0)
    store = tmp_path / "cube"
    import_package(store, pkg0)

    # A second origin carrying 100000 Pa instead of 85000 Pa: a distinct
    # exact-rational level key that must be appended, never merged into the
    # existing 85000 Pa coordinate.  Reference time 12Z gives its own origin.
    spec = FixtureSpec(
        pdt=0, first_surface_type=100, first_scale=0,
        first_scaled=100000, data_seed=5, data_time=1200)
    raw = encode_message(spec)
    scan = scan_file(io.BytesIO(raw), len(raw))
    other_dig = grid_digest(scan.messages[0].gds_bytes)
    assert other_dig == grid_dig      # Section 3 is unchanged

    pkg1 = real_package_from_message(
        tmp_path / "second", raw, other_dig, "lvl1000")
    import_package(store, pkg1)

    reader = CubeReader(store)
    rationals = {
        tuple(entry["identity"]["first"]["rational"])
        for entry in reader.levels()}
    assert rationals == {(85000, 1), (50000, 1), (100000, 1)}
    # The second package carries its own origin; both stay readable.
    assert len(reader.origins()) == 2


def test_level_semantics_conflict_is_rejected(tmp_path):
    # Same rational level key (type 100, value 85000/1) but an incompatible
    # stored surface identity: simulate a registry whose existing entry
    # carries different surface octets, proving the preflight guard fires
    # before any write.
    from grib_asm.cube.importer import _preflight
    from grib_asm.cube.loader import load_and_validate
    from grib_asm.cube.store import CubeStore, Registry
    from grib_asm.cube.coords import level_key
    from grib_asm.identity import make_surface, Level

    grid_dig = standard_grid_digest()
    pkg0 = build_package(
        tmp_path, 0, make_request("lvlbase", grid_dig))
    store_path = tmp_path / "cube"
    import_package(store_path, pkg0)

    # Corrupt only the stored level identity while keeping its key.
    registry = Registry(store_path)
    level = Level(first=make_surface(100, 0, 85000), second=None)
    lk = level_key(level)
    registry.levels[lk]["identity"]["first"]["scale_factor"] = 9
    registry.write()

    store = CubeStore(store_path)
    validated = load_and_validate(pkg0)
    fresh_registry = Registry(store_path)
    with pytest.raises(CubeCompatibilityError) as exc:
        _preflight(store, fresh_registry, validated)
    assert exc.value.reason == "level_semantics_conflict"
    # The published dataset was not modified by the failed preflight.
    reader = CubeReader(store_path)
    assert reader.origins() == ["2026-01-01T00:00:00Z"]
