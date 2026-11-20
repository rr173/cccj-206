"""Acceptance tests for the hand-off package -> Zarr analysis cube.

The expectations below are hand-authored independently of the production
coordinate constructors: the expected grid coordinates and every cell array
are recomputed here from documented constants and the seed formula in
``cube_helpers``.  The production code is never used to generate the
expected data.

Coverage required by the task statement:

* two reference-time cycles, two members, two levels;
* one instantaneous field and two distinct accumulation intervals
  (0-6 h must never masquerade as 3-6 h);
* one cycle missing a member/slot, expressed by an explicit mask, never
  filled from the neighbouring cycle;
* coordinates, values, interval boundaries, mask and provenance digests are
  checked cell by cell;
* idempotent re-import, same-origin/different-digest conflict, different
  grid and different variable identity all leave the old data readable.
"""

from __future__ import annotations

import numpy as np
import pytest

from grib_asm.cube.importer import import_package
from grib_asm.cube.query import CubeReader
from grib_asm.cube.store import CubeConfig

from cube_helpers import (
    ACC_0_6, ACC_3_6, ALL_SLOTS, INSTANT, LEVELS, NJ, NI,
    build_package, expected_lats, expected_lons, expected_values,
    make_request, ref_time, standard_grid_digest,
)
from e2e_helpers import HAS_ECCODES

pytestmark = pytest.mark.skipif(
    not HAS_ECCODES, reason="ecCodes required to encode/decode fixtures")


def _reader_and_packages(tmp_path):
    grid_dig = standard_grid_digest()
    # Full Cartesian-product request; cycle 1 will simply omit one cell.
    full_request = make_request("acc-full", grid_dig)
    pkg0 = build_package(tmp_path, 0, full_request)
    # Cycle 1 ships only the control member (the hand-off package's own
    # request Cartesian product is therefore control-only).  In the cube the
    # perturbed member of cycle 1 is expressed by an explicit missing mask;
    # values are never borrowed from cycle 0.
    control_only_request = make_request(
        "acc-partial", grid_dig, members=(0,))
    pkg1 = build_package(
        tmp_path, 1, control_only_request, members=(0,))
    store = tmp_path / "cube"
    r0 = import_package(store, pkg0, CubeConfig(batch_size=4))
    r1 = import_package(store, pkg1, CubeConfig(batch_size=4))
    assert r0.status == "committed"
    assert r1.status == "committed"
    return store, grid_dig, (pkg0, pkg1)


def test_hand_authored_grid_coordinates(tmp_path):
    store, grid_dig, _ = _reader_and_packages(tmp_path)
    reader = CubeReader(store)
    # Independently expected coordinates, computed with plain numpy here.
    np.testing.assert_allclose(reader._group["coords/latitude"][:],
                               expected_lats())
    np.testing.assert_allclose(reader._group["coords/longitude"][:],
                               expected_lons())
    assert reader.meta["grid_digest"] == grid_dig
    assert reader.meta["grid_shape"] == [NJ, NI]
    # Scanning/ordering metadata is preserved verbatim.
    assert reader.grid["pointOrdering"].startswith("row-major")


def test_full_cycle_matches_hand_authored_arrays(tmp_path):
    store, _, _ = _reader_and_packages(tmp_path)
    reader = CubeReader(store)
    from grib_asm.identity import VariableId
    var = VariableId(0, 0, 0, 29, 1, 7, 0)
    result = reader.select(var)
    # Axes: origin(2), level(2), member(2), slot(3), lat(3), lon(4).
    assert result.values.shape == (2, 2, 2, 3, NJ, NI)
    assert result.present.shape == (2, 2, 2, 3)

    for li in range(2):
        for mi in range(2):
            for si in range(3):
                np.testing.assert_array_equal(
                    result.values[0, li, mi, si],
                    expected_values(0, li, mi, si))
                assert result.present[0, li, mi, si] == 1


def test_missing_cell_is_explicit_mask_not_neighbour(tmp_path):
    store, _, _ = _reader_and_packages(tmp_path)
    reader = CubeReader(store)
    from grib_asm.identity import VariableId
    var = VariableId(0, 0, 0, 29, 1, 7, 0)
    result = reader.select(var)

    # Cycle 1 shipped only the control member: every perturbed (member 1)
    # cell of cycle 1, at both levels and all three slots, is an explicit
    # missing cell - including the 3-6 h accumulation slot.
    for li in range(2):
        for si in range(3):
            assert result.present[1, li, 1, si] == 0
            assert np.all(np.isnan(result.values[1, li, 1, si]))
            assert str(result.message_digest[1, li, 1, si]) == ""
            assert str(result.package_digest[1, li, 1, si]) == ""

    # Every cycle-1 control cell is present with genuine cycle-1 values;
    # the neighbouring cycle's values were not copied into the mask holes.
    for li in range(2):
        for si in range(3):
            assert result.present[1, li, 0, si] == 1
            np.testing.assert_array_equal(
                result.values[1, li, 0, si],
                expected_values(1, li, 0, si))

    # The perturbed member of cycle 0 is still intact and readable.
    for li in range(2):
        for si in range(3):
            assert result.present[0, li, 1, si] == 1
            np.testing.assert_array_equal(
                result.values[0, li, 1, si],
                expected_values(0, li, 1, si))


def test_instant_and_accumulation_intervals_are_distinct(tmp_path):
    store, _, _ = _reader_and_packages(tmp_path)
    reader = CubeReader(store)
    slots = reader.slots()
    by_key = {s["key"]: s for s in slots}

    instant = by_key["instant_0"]
    acc06 = by_key["stat1_0_21600"]
    acc36 = by_key["stat1_10800_21600"]

    # Instant vs interval is explicit in the stored coordinates.
    assert instant["is_instant"] is True
    assert instant["identity"] == {
        "start_seconds": 0, "end_seconds": 0, "statistic": None}
    for slot in (acc06, acc36):
        assert slot["is_instant"] is False
        assert slot["identity"]["statistic"] == 1

    # Absolute validity boundaries are per origin.  Cycle 0 reference 00Z:
    origin0 = ref_time(0)
    bounds_instant = reader.slot_validity(origin0, "instant_0")
    bounds_acc06 = reader.slot_validity(origin0, "stat1_0_21600")
    bounds_acc36 = reader.slot_validity(origin0, "stat1_10800_21600")
    assert bounds_instant == {
        "validity_start": origin0, "validity_end": origin0}
    assert bounds_acc06 == {
        "validity_start": "2026-01-01T00:00:00Z",
        "validity_end": "2026-01-01T06:00:00Z"}
    assert bounds_acc36 == {
        "validity_start": "2026-01-01T03:00:00Z",
        "validity_end": "2026-01-01T06:00:00Z"}

    # The same relative slot at the 12Z origin carries shifted absolute
    # bounds, proving boundaries are not collapsed into one global value.
    origin1 = ref_time(1)
    assert reader.slot_validity(origin1, "instant_0") == {
        "validity_start": origin1, "validity_end": origin1}
    assert reader.slot_validity(origin1, "stat1_0_21600") == {
        "validity_start": "2026-01-01T12:00:00Z",
        "validity_end": "2026-01-01T18:00:00Z"}

    # Same end time, different start -> distinct arrays, and 0-6 h values
    # can never fill the 3-6 h coordinate.
    from grib_asm.identity import VariableId
    result = reader.select(VariableId(0, 0, 0, 29, 1, 7, 0))
    for oi in range(2):
        for li in range(2):
            for mi in range(2):
                if result.present[oi, li, mi, 2] == 0:
                    continue
                assert not np.array_equal(
                    result.values[oi, li, mi, 1],
                    result.values[oi, li, mi, 2])


def test_member_and_level_identities_do_not_swap(tmp_path):
    store, _, _ = _reader_and_packages(tmp_path)
    reader = CubeReader(store)
    levels = reader.levels()
    members = reader.members()
    # Exact rational level surfaces are stored, not a float pressure guess.
    level_ids = [l["identity"] for l in levels]
    assert level_ids[0]["first"]["type"] == LEVELS[0][0]
    assert level_ids[0]["first"]["rational"] == [LEVELS[0][2], 1]
    assert level_ids[1]["first"]["rational"] == [LEVELS[1][2], 1]
    member_kinds = [m["identity"]["kind"] for m in members]
    assert member_kinds == ["control", "perturbed"]
    assert members[1]["identity"]["perturbationNumber"] == 1

    from grib_asm.identity import VariableId
    result = reader.select(VariableId(0, 0, 0, 29, 1, 7, 0))
    # Cell values at swapped indices must differ and match their own labels;
    # only present cells participate (cycle 1 has no perturbed member).
    for oi in range(2):
        for si in range(3):
            for li in range(2):
                if result.present[oi, li, 0, si] == 0 or \
                        result.present[oi, li, 1, si] == 0:
                    continue
                assert not np.array_equal(
                    result.values[oi, li, 0, si],
                    result.values[oi, li, 1, si])
            for mi in range(2):
                if result.present[oi, 0, mi, si] == 0 or \
                        result.present[oi, 1, mi, si] == 0:
                    continue
                assert not np.array_equal(
                    result.values[oi, 0, mi, si],
                    result.values[oi, 1, mi, si])


def test_provenance_digests_per_cell(tmp_path):
    store, _, pkgs = _reader_and_packages(tmp_path)
    pkg0, pkg1 = pkgs
    import json
    marker0 = json.loads((pkg0 / "PACKAGE_COMPLETE.json").read_text())
    marker1 = json.loads((pkg1 / "PACKAGE_COMPLETE.json").read_text())
    manifest0 = json.loads(
        (pkg0 / "package_manifest.json").read_text())
    reader = CubeReader(store)
    from grib_asm.identity import VariableId
    result = reader.select(VariableId(0, 0, 0, 29, 1, 7, 0))

    # Package digest provenance follows the origin, never the neighbour.
    assert np.all(
        result.package_digest[0] == marker0["data_sha256"])
    present1 = result.present[1] == 1
    assert np.all(
        np.asarray(result.package_digest[1])[present1]
        == marker1["data_sha256"])

    # Message digests match the actual hand-off manifest records in order.
    # Manifest order is variable(1), level, member, slot; our cube indices
    # use the same level/member/slot ordering.
    records = manifest0["records"]
    k = 0
    for li in range(2):
        for mi in range(2):
            for si in range(3):
                assert str(result.message_digest[0, li, mi, si]) == \
                    records[k]["output_sha256"]
                k += 1


def test_query_slice_filters_and_stable_order(tmp_path):
    store, _, _ = _reader_and_packages(tmp_path)
    reader = CubeReader(store)
    from grib_asm.identity import (
        Level, MemberId, VariableId, make_surface,
    )
    from grib_asm.timeutil import TimeSlot
    var = VariableId(0, 0, 0, 29, 1, 7, 0)
    level = Level(first=make_surface(100, 0, 85000), second=None)
    member = MemberId(kind="perturbed", type_of_ensemble=1, number=1)
    slot = TimeSlot(10800, 21600, 1)
    result = reader.select(
        var, levels=[level], members=[member], slots=[slot])
    assert result.values.shape == (2, 1, 1, 1, NJ, NI)
    assert result.axes["origin"] == [ref_time(0), ref_time(1)]
    assert result.axes["level"][0]["first"]["rational"] == [85000, 1]
    # Missing 3-6 h perturbed cell in cycle 1 remains masked.
    assert result.present[0, 0, 0, 0] == 1
    assert result.present[1, 0, 0, 0] == 0

    # Validity-time window filter on the 00Z origin: only the instant slot
    # ends at/before 03Z on that reference day (both accumulations end 06Z).
    # Windows are absolute UTC, so they are scoped to one origin.
    windowed = reader.select(
        var, origins=[ref_time(0)],
        validity_end_to="2026-01-01T03:00:00Z")
    slot_keys = [s["key"] for s in windowed.axes["slot"]]
    assert slot_keys == ["instant_0"]

    # At the 12Z origin the 0-6 h slot starts at 12Z and is excluded by a
    # 13Z start window; only the 3-6 h slot (starts 15Z) remains.
    afternoon = reader.select(
        var, origins=[ref_time(1)],
        validity_start_from="2026-01-01T13:00:00Z")
    assert {s["key"] for s in afternoon.axes["slot"]} == {
        "stat1_10800_21600"}
    assert "instant_0" not in {
        s["key"] for s in afternoon.axes["slot"]}


def test_variable_identity_keeps_full_numeric_codes(tmp_path):
    store, _, _ = _reader_and_packages(tmp_path)
    reader = CubeReader(store)
    entry = reader.variables()[0]
    assert entry["identity"] == {
        "discipline": 0,
        "parameterCategory": 0,
        "parameterNumber": 0,
        "masterTablesVersionNumber": 29,
        "localTablesVersionNumber": 1,
        "centre": 7,
        "subCentre": 0,
    }
    # Raw unit preserved; a CF mapping, when available, is additive only.
    assert entry["units"]["raw_unit"]
    assert entry["units"]["mapping_status"] in (
        "mapped", "raw_only_no_credible_mapping")
    # shortName is stored only as observed evidence, never as identity.
    assert "shortName_observed" in entry["attrs"]
