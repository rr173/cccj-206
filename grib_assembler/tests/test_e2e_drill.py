"""Main drill: independently authored four-cell matrices and selection.

Expected matrices are written by hand in each test; the production readiness
generator is never used to produce the expectations.
"""

import json
from datetime import datetime, timezone

import pytest

from grib_asm.catalog import Catalog
from grib_asm.readiness import build_readiness
from grib_asm.assemble import assemble
from grib_asm.verify import verify_package
from grib_asm.errors import BlockedError
from grib_asm.timeutil import TimeSlot

from e2e_helpers import (
    HAS_ECCODES, make_stat_spec, _write_collection, _build_request,
    _grid_digest_of, encode_message, _ingest, _four_cycle_files,
)

pytestmark = pytest.mark.skipif(
    not HAS_ECCODES, reason="ecCodes required for end-to-end fixtures")

UTC = timezone.utc


def test_00_complete_06_missing_one_cell_selects_00(tmp_path, grid_digest_value):
    # Independently authored expected matrix:
    #                 0-3h        3-6h
    # control(00):    present     present
    # perturbed(00):  present     present
    # control(06):    present     present
    # perturbed(06):  present     MISSING
    coll_00 = [("c00", _four_cycle_files(
        0, {"control": [0, 1], "perturbed": [0, 1]}))]
    coll_06 = [("c06", _four_cycle_files(
        6, {"control": [0, 1], "perturbed": [0]}))]
    cat, roots, manifests = _ingest(tmp_path, coll_00 + coll_06)
    req = _build_request(grid_digest_value)
    report = build_readiness(cat, req)

    assert not report.blocked
    assert report.selected_reference_time == "2026-01-01T00:00:00Z"
    doc = report.to_dict()
    cyc = {c["reference_time"]: c for c in doc["cycles"]}
    assert cyc["2026-01-01T00:00:00Z"]["complete"] is True
    c06 = cyc["2026-01-01T06:00:00Z"]
    assert c06["complete"] is False
    assert len(c06["reasons"]) == 1
    assert c06["reasons"][0]["reason"] == "missing"
    # The missing cell is the perturbed 3-6h slot.
    missing = c06["reasons"][0]["cell"]
    assert missing["member"]["kind"] == "perturbed"
    assert missing["slot"] == {"start_seconds": 10800,
                               "end_seconds": 21600, "statistic": 1}

    result = assemble(cat, req, roots, tmp_path / "out", manifests)
    assert result.selected_reference_time == "2026-01-01T00:00:00Z"
    assert result.record_count == 4
    cat.close()

    v = verify_package(result.package_dir)
    assert v.ok, v.problems


def test_after_06_completes_selection_switches_to_06(tmp_path, grid_digest_value):
    coll_00 = [("c00", _four_cycle_files(
        0, {"control": [0, 1], "perturbed": [0, 1]}))]
    # Now 06 has all four cells.
    coll_06 = [("c06", _four_cycle_files(
        6, {"control": [0, 1], "perturbed": [0, 1]}))]
    cat, roots, manifests = _ingest(tmp_path, coll_00 + coll_06)
    req = _build_request(grid_digest_value)
    report = build_readiness(cat, req)
    assert report.selected_reference_time == "2026-01-01T06:00:00Z"
    result = assemble(cat, req, roots, tmp_path / "out", manifests)
    assert result.selected_reference_time == "2026-01-01T06:00:00Z"
    # Every record comes from a 06 reference time.
    pkg = json.loads((result.package_dir / "package_manifest.json").read_text())
    for rec in pkg["records"]:
        assert rec["selected_reference_time"] == "2026-01-01T06:00:00Z"
    assert verify_package(result.package_dir).ok
    cat.close()


def test_conflict_in_06_falls_back_to_00_not_mix(tmp_path, grid_digest_value):
    coll_00 = [("c00", _four_cycle_files(
        0, {"control": [0, 1], "perturbed": [0, 1]}))]
    files06 = _four_cycle_files(
        6, {"control": [0, 1], "perturbed": [0, 1]})
    # Add a second, byte-different copy of the control 0-3h cell (same key,
    # different whole-message digest) by changing the data seed.
    conflict_spec = make_stat_spec(6, "control", 0, 3, data_seed=999)
    files06["conflict"] = [conflict_spec]
    cat, roots, manifests = _ingest(
        tmp_path, [("c00", _four_cycle_files(
            0, {"control": [0, 1], "perturbed": [0, 1]})),
                   ("c06", files06)])
    req = _build_request(grid_digest_value)
    report = build_readiness(cat, req)
    # 06 is conflicted -> fall back to complete 00.
    assert report.selected_reference_time == "2026-01-01T00:00:00Z"
    cyc06 = next(c for c in report.to_dict()["cycles"]
                 if c["reference_time"] == "2026-01-01T06:00:00Z")
    assert any(r["reason"] == "conflict" for r in cyc06["reasons"])

    result = assemble(cat, req, roots, tmp_path / "out", manifests)
    pkg = json.loads((result.package_dir / "package_manifest.json").read_text())
    # No 06 cell is smuggled in: all four records are from the 00 cycle.
    for rec in pkg["records"]:
        assert rec["selected_reference_time"] == "2026-01-01T00:00:00Z"
    assert verify_package(result.package_dir).ok
    cat.close()


def test_blocked_when_no_cycle_complete(tmp_path, grid_digest_value):
    # Only a partial 06 cycle exists; 00 absent.
    coll = [("c06", _four_cycle_files(
        6, {"control": [0, 1], "perturbed": [0]}))]
    cat, roots, manifests = _ingest(tmp_path, coll)
    req = _build_request(grid_digest_value)
    report = build_readiness(cat, req)
    assert report.blocked
    with pytest.raises(BlockedError):
        assemble(cat, req, roots, tmp_path / "out", manifests)
    # No success directory is published.
    assert not (tmp_path / "out").exists() or \
        all(p.suffix != ".grib2" for p in
            (tmp_path / "out").rglob("*") if p.is_file())
    cat.close()


def test_not_after_boundary_excludes_newer_cycle(tmp_path, grid_digest_value):
    coll_00 = [("c00", _four_cycle_files(
        0, {"control": [0, 1], "perturbed": [0, 1]}))]
    coll_06 = [("c06", _four_cycle_files(
        6, {"control": [0, 1], "perturbed": [0, 1]}))]
    cat, roots, manifests = _ingest(tmp_path, coll_00 + coll_06)
    # not_after at 03Z excludes the 06 cycle even though it is complete.
    req = _build_request(grid_digest_value,
                         not_after="2026-01-01T03:00:00Z")
    report = build_readiness(cat, req)
    assert report.selected_reference_time == "2026-01-01T00:00:00Z"
    cat.close()


def test_not_after_equal_to_cycle_time_is_inclusive(tmp_path,
                                                    grid_digest_value):
    coll_00 = [("c00", _four_cycle_files(
        0, {"control": [0, 1], "perturbed": [0, 1]}))]
    coll_06 = [("c06", _four_cycle_files(
        6, {"control": [0, 1], "perturbed": [0, 1]}))]
    cat, roots, manifests = _ingest(tmp_path, coll_00 + coll_06)
    # not_after exactly at 06Z must still include the 06Z cycle (inclusive).
    req = _build_request(grid_digest_value,
                         not_after="2026-01-01T06:00:00Z")
    report = build_readiness(cat, req)
    assert report.selected_reference_time == "2026-01-01T06:00:00Z"
    cat.close()


def test_same_endpoint_0_6_cannot_fill_3_6(tmp_path, grid_digest_value):
    # 06 cycle missing perturbed 3-6h; add a 0-6h accumulation instead.
    files = _four_cycle_files(
        6, {"control": [0, 1], "perturbed": [0]})
    files["all"].append(make_stat_spec(6, "perturbed", 0, 6, data_seed=7))
    cat, roots, _ = _ingest(tmp_path, [("c06", files)])
    req = _build_request(grid_digest_value)
    report = build_readiness(cat, req)
    cyc06 = report.to_dict()["cycles"][0]
    assert not cyc06["complete"]
    # The 3-6h cell stays missing; 0-6h is a distinct slot, not a substitute.
    missing_slots = [(r["cell"]["slot"]["start_seconds"],
                      r["cell"]["slot"]["end_seconds"])
                     for r in cyc06["reasons"] if r["reason"] == "missing"]
    assert (10800, 21600) in missing_slots
    cat.close()


def test_different_grid_cannot_substitute(tmp_path, grid_digest_value):
    files = _four_cycle_files(
        6, {"control": [0, 1], "perturbed": [0]})
    # Missing perturbed 3-6h supplied on a different grid (shifted first point).
    files["othergrid"] = [make_stat_spec(
        6, "perturbed", 3, 6, data_seed=8,
        grid_overrides=dict(first_lat=55.0))]
    cat, roots, _ = _ingest(tmp_path, [("c06", files)])
    req = _build_request(grid_digest_value)
    report = build_readiness(cat, req)
    cyc = report.to_dict()["cycles"][0]
    assert not cyc["complete"]
    assert any(r["reason"] == "grid_mismatch" for r in cyc["reasons"])
    cat.close()


def test_same_shortname_different_param_number_is_different_variable(
        tmp_path, grid_digest_value):
    files = _four_cycle_files(
        6, {"control": [0, 1], "perturbed": [0]})
    # Fill the gap with parameterNumber=99 instead of 0.
    files["othervar"] = [make_stat_spec(
        6, "perturbed", 3, 6, data_seed=9, parameter_number=99)]
    cat, roots, _ = _ingest(tmp_path, [("c06", files)])
    req = _build_request(grid_digest_value)
    report = build_readiness(cat, req)
    cyc = report.to_dict()["cycles"][0]
    assert not cyc["complete"]
    # It cannot match; the requested cell is simply missing.
    assert any(r["reason"] == "missing" for r in cyc["reasons"])
    cat.close()
