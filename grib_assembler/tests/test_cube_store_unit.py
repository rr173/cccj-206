"""Lightweight unit tests for cube storage primitives (no GRIB decoding).

These tests exercise the transaction journal, registry snapshots and
coordinate keying directly, so they run even when ecCodes is unavailable.
"""

from __future__ import annotations

import pytest

zarr = pytest.importorskip("zarr")
np = pytest.importorskip("numpy")

from grib_asm.cube.coords import (
    level_key, member_key, slot_key, variable_key,
)
from grib_asm.cube.recovery import recover
from grib_asm.cube.store import (
    CUBE_DIR, CubeConfig, CubeStore, PendingCell, Registry, marker_name,
)
from grib_asm.identity import (
    Level, MemberId, VariableId, make_surface,
)
from grib_asm.timeutil import TimeSlot


class _FakeGrid:
    nj, ni = 3, 4
    latitude = np.linspace(40.0, 42.0, 3)
    longitude = np.linspace(10.0, 13.0, 4)

    def definition_doc(self):
        return {"gridDefinitionTemplateNumber": 0,
                "pointOrdering": "row-major [Nj, Ni], scan mode 0"}


def _variable():
    return VariableId(0, 0, 0, 29, 1, 7, 0)


def _level():
    return Level(first=make_surface(100, 0, 85000), second=None)


def _member():
    return MemberId(kind="perturbed", type_of_ensemble=1, number=1)


def _slot():
    return TimeSlot(10800, 21600, 1)


def test_registry_appends_are_indexed_stably(tmp_path):
    reg = Registry(tmp_path)
    from grib_asm.cube.units import resolve_units
    var = _variable()
    units = resolve_units(var, "K")
    vk = reg.ensure_variable(var, units, {"shortName_observed": "t"})
    lk = reg.ensure_level(_level())
    mk = reg.ensure_member(_member())
    sk = reg.ensure_slot(_slot())
    oi = reg.ensure_origin("2026-01-01T00:00:00Z")
    # Absolute validity bounds are stored per origin, not on the slot axis.
    reg.set_slot_validity(
        "2026-01-01T00:00:00Z", _slot(),
        "2026-01-01T03:00:00Z", "2026-01-01T06:00:00Z")

    assert reg.variable_index(var) == 0
    assert reg.level_index(_level()) == 0
    assert reg.member_index(_member()) == 0
    assert reg.slot_index(_slot()) == 0
    assert oi == 0
    # Re-ensuring the same coordinate is a no-op, not a duplicate.
    assert reg.ensure_level(_level()) == lk
    assert len(reg.levels) == 1
    reg.write()
    reloaded = Registry(tmp_path)
    assert reloaded.variables[vk]["identity"] == var.to_dict()
    assert reloaded.slot_validity_map["2026-01-01T00:00:00Z"][sk] == {
            "validity_start": "2026-01-01T03:00:00Z",
            "validity_end": "2026-01-01T06:00:00Z"}
    assert reloaded.origins == ["2026-01-01T00:00:00Z"]


def test_first_import_publishes_only_after_commit(tmp_path):
    store = CubeStore(tmp_path, CubeConfig(batch_size=2))
    registry = Registry(tmp_path)
    var = _variable()
    from grib_asm.cube.units import resolve_units
    registry.ensure_variable(
        var, resolve_units(var, "K"), {"shortName_observed": "t"})
    registry.ensure_level(_level())
    registry.ensure_member(_member())
    registry.ensure_slot(_slot())
    origin = "2026-01-01T00:00:00Z"
    txn = store.begin_transaction(
        origin, "a" * 64, registry, initial_grid=_FakeGrid(),
        initial_grid_digest="g" * 64)
    cell = PendingCell(
        var_key=variable_key(var), origin_index=0, level_index=0,
        member_index=0, slot_index=0,
        values=np.arange(12, dtype="float64").reshape(3, 4),
        message_digest="m" * 64, package_digest="a" * 64)
    store.flush(txn, [cell], registry)

    # Before commit, the published group has no committed arrays or marker.
    assert not list((tmp_path / CUBE_DIR / "committed").glob("*.json"))
    registry.sources[origin] = {"package_digest": "a" * 64}
    store.prepare(txn, registry, registry.sources[origin])
    store.commit(txn, registry)

    assert (tmp_path / CUBE_DIR / "committed"
            / f"{marker_name(origin)}.json").is_file()
    group = zarr.open_group(str(tmp_path), mode="r")
    values = group[f"vars/{variable_key(var)}/values"]
    assert values.shape == (1, 1, 1, 1, 3, 4)
    np.testing.assert_array_equal(
        values[0, 0, 0, 0], cell.values)
    assert group[f"vars/{variable_key(var)}/present"][0, 0, 0, 0] == 1
    assert str(group[f"vars/{variable_key(var)}/message_digest"][
        0, 0, 0, 0]) == "m" * 64
    np.testing.assert_allclose(group["coords/latitude"][:],
                               _FakeGrid.latitude)


def test_uncommitted_origin_is_hidden_and_cleaned(tmp_path):
    store = CubeStore(tmp_path, CubeConfig())
    registry = Registry(tmp_path)
    var = _variable()
    from grib_asm.cube.units import resolve_units
    registry.ensure_variable(
        var, resolve_units(var, "K"), {"shortName_observed": "t"})
    registry.ensure_level(_level())
    registry.ensure_member(_member())
    registry.ensure_slot(_slot())
    origin = "2026-01-01T00:00:00Z"
    txn = store.begin_transaction(
        origin, "a" * 64, registry, initial_grid=_FakeGrid(),
        initial_grid_digest="g" * 64)
    cell = PendingCell(
        var_key=variable_key(var), origin_index=0, level_index=0,
        member_index=0, slot_index=0,
        values=np.zeros((3, 4)), message_digest="m" * 64,
        package_digest="a" * 64)
    store.flush(txn, [cell], registry)
    # Simulate a hard interruption: staging exists, marker never written.
    report = recover(tmp_path, mode="cleanup")
    assert not list((tmp_path / CUBE_DIR / "txn").iterdir())
    assert not list((tmp_path / CUBE_DIR / "committed").glob("*.json"))
    # The transaction was removed before prepare; no committed origin exists.
    assert report.committed_origins == []
