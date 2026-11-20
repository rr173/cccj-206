"""Profile edge cases verified with real ecCodes fixtures.

Covers minute/hour offset equivalence, instant vs statistical products,
rational level scaling, deterministic vs ensemble members, retransmission
(same digest, several sources), bad-tail files and whole-package copy
independence.
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone

import pytest

from grib_asm.catalog import Catalog
from grib_asm.readiness import build_readiness
from grib_asm.assemble import assemble
from grib_asm.verify import verify_package
from grib_asm.timeutil import TimeSlot
from grib_asm.identity import grid_digest
from grib_asm.request import (
    Request, RequestedLevel, RequestedMember, RequestedVariable,
)
from grib_asm.scanner import scan_file
from grib_asm.errors import BadFileError
import io

from fixturegen import FixtureSpec, encode_message
from e2e_helpers import (
    HAS_ECCODES, _write_collection, _build_request, _grid_digest_of,
    make_stat_spec,
)

pytestmark = pytest.mark.skipif(
    not HAS_ECCODES, reason="ecCodes required for profile edge-case fixtures")

UTC = timezone.utc


def test_minute_and_hour_offsets_are_equivalent(grid_digest_value):
    # Same 0-3h interval expressed in minutes (180 min) vs hours (3 h) must
    # normalise to the identical slot and thus the identical business key.
    in_minutes = make_stat_spec(0, "control", 0, 3, length_unit=0,
                                data_seed=1)
    # forecastTime unit also minutes for the start; start is 0 either way.
    in_minutes.time_range_unit = 0  # minute
    in_minutes.length_of_time_range = 180
    in_hours = make_stat_spec(0, "control", 0, 3, length_unit=1, data_seed=1)

    from grib_asm.classify import classify_message
    from grib_asm.headers import (open_handle_from_message_bytes,
                                  close_handle)

    def classify(raw):
        h = open_handle_from_message_bytes(raw)
        try:
            res = scan_file(io.BytesIO(raw), len(raw))
            rec = classify_message(h, raw, res.messages[0], "f", "c")
            return rec
        finally:
            close_handle(h)

    r_min = classify(encode_message(in_minutes))
    r_hr = classify(encode_message(in_hours))
    assert r_min.status == "ok" and r_hr.status == "ok"
    assert r_min.slot == r_hr.slot
    assert r_min.business_key.digest() == r_hr.business_key.digest()


def test_instant_vs_statistical_are_different_products(grid_digest_value):
    instant = FixtureSpec(pdt=1, forecast_time=3, data_seed=2,
                          ensemble_type=3, perturbation_number=0)
    stat = make_stat_spec(0, "control", 3, 6, data_seed=2)
    raw_i, raw_s = encode_message(instant), encode_message(stat)
    from grib_asm.classify import classify_message
    from grib_asm.headers import (open_handle_from_message_bytes,
                                  close_handle)

    def classify(raw):
        res = scan_file(io.BytesIO(raw), len(raw))
        h = open_handle_from_message_bytes(raw)
        try:
            return classify_message(h, raw, res.messages[0], "f", "c")
        finally:
            close_handle(h)

    ri, rs = classify(raw_i), classify(raw_s)
    assert ri.slot.is_instant and not rs.slot.is_instant
    assert ri.business_key.digest() != rs.business_key.digest()


def test_level_scaling_preserved_as_exact_rational(grid_digest_value):
    # 150 with scale factor 2 = 1.50 surface; verify the rational is exact.
    spec = FixtureSpec(pdt=0, first_surface_type=102,
                       first_scale=2, first_scaled=150)
    raw = encode_message(spec)
    from grib_asm.classify import classify_message
    from grib_asm.headers import (open_handle_from_message_bytes,
                                  close_handle)
    res = scan_file(io.BytesIO(raw), len(raw))
    h = open_handle_from_message_bytes(raw)
    try:
        rec = classify_message(h, raw, res.messages[0], "f", "c")
    finally:
        close_handle(h)
    from fractions import Fraction
    assert rec.level.first.value == Fraction(150, 100)
    assert rec.level.second is None  # undefined second surface stays missing


def test_deterministic_and_ensemble_members_distinct(grid_digest_value):
    det = FixtureSpec(pdt=0, forecast_time=0, data_seed=3)
    ctrl = FixtureSpec(pdt=1, forecast_time=0, data_seed=3,
                       ensemble_type=3, perturbation_number=0)
    from grib_asm.classify import classify_message
    from grib_asm.headers import (open_handle_from_message_bytes,
                                  close_handle)

    def classify(raw):
        res = scan_file(io.BytesIO(raw), len(raw))
        h = open_handle_from_message_bytes(raw)
        try:
            return classify_message(h, raw, res.messages[0], "f", "c")
        finally:
            close_handle(h)

    rd, rc = classify(encode_message(det)), classify(encode_message(ctrl))
    assert rd.member.kind == "deterministic"
    assert rc.member.kind == "control"
    assert rd.business_key.digest() != rc.business_key.digest()


def test_retransmission_same_digest_one_cell_many_sources(tmp_path,
                                                          grid_digest_value):
    # Two files carry byte-identical copies of one cell.
    spec = make_stat_spec(0, "control", 0, 3, data_seed=4)
    raw = encode_message(spec)
    root = tmp_path / "c"
    root.mkdir()
    entries = []
    for fid in ("f1", "f2"):
        rel = f"{fid}.grib2"
        (root / rel).write_bytes(raw)
        entries.append({"file_id": fid, "relative_path": rel,
                        "size_bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    from fixturegen import build_manifest
    from grib_asm.request import Manifest, ManifestFile
    mdoc = build_manifest("c", root, entries)
    manifest = Manifest("c", None,
                        [ManifestFile(**{k: e[k] for k in
                                         ("file_id", "relative_path",
                                          "size_bytes", "sha256")})
                         for e in entries], mdoc)
    cat = Catalog(tmp_path / "cat.db")
    for e in manifest.files:
        cat.ingest_file(manifest, e, root)
    # One filled cell with two sources sharing one digest.
    rows = cat._conn.execute("SELECT * FROM cells").fetchall()
    assert len(rows) == 1
    cell = cat.cell(rows[0]["bkey_digest"])
    assert cell["state"] == "filled"
    assert len(cell["sources"]) == 2
    assert cell["distinct_digests"] == [hashlib.sha256(raw).hexdigest()]
    cat.close()


def test_bad_tail_message_file_unusable(tmp_path, grid_digest_value):
    good = encode_message(FixtureSpec(pdt=0, data_seed=5))
    broken = good + b"TAIL"
    rel = "broken.grib2"
    root = tmp_path / "c"
    root.mkdir()
    (root / rel).write_bytes(broken)
    from grib_asm.request import Manifest, ManifestFile
    manifest = Manifest("c", None, [ManifestFile(
        "f", rel, len(broken), hashlib.sha256(broken).hexdigest())], {})
    cat = Catalog(tmp_path / "cat.db")
    res = cat.ingest_file(manifest, manifest.files[0], root)
    assert res.status == "failed"
    assert res.reason == "trailing_junk"
    cat.close()


def test_package_copy_verifies_elsewhere(tmp_path, grid_digest_value):
    # Build a complete 00 package, copy the directory elsewhere, and verify it
    # with no reference to the original input tree.
    from e2e_helpers import _four_cycle_files, _ingest
    cat, roots, manifests = _ingest(tmp_path, [("c00", _four_cycle_files(
        0, {"control": [0, 1], "perturbed": [0, 1]}))])
    req = _build_request(grid_digest_value)
    result = assemble(cat, req, roots, tmp_path / "out", manifests)
    cat.close()

    copied = tmp_path / "elsewhere" / "pkg"
    shutil.copytree(result.package_dir, copied)
    # Remove the original inputs to prove the package is self-contained.
    shutil.rmtree(roots["c00"])
    report = verify_package(copied)
    assert report.ok, report.problems
    assert report.record_count == 4


def test_in_progress_directory_is_not_a_package(tmp_path, grid_digest_value):
    from grib_asm.verify import VerificationError, COMPLETE_MARKER
    stage = tmp_path / "pkg.in-progress"
    stage.mkdir()
    (stage / "messages.grib2").write_bytes(b"x")
    with pytest.raises(VerificationError):
        verify_package(stage)
    # A directory lacking the completion marker is also refused.
    no_marker = tmp_path / "pkg2"
    no_marker.mkdir()
    with pytest.raises(VerificationError):
        verify_package(no_marker)
