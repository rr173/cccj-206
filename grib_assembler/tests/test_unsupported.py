"""Unsupported-template rules: well-formed messages outside the profile.

These messages are structurally valid GRIB2 but violate a profile constraint.
They must be recorded as ``unsupported`` with a stable reason, never crash and
never fabricate fields.
"""

import io

import pytest

from grib_asm.scanner import scan_file
from grib_asm.classify import classify_message
from grib_asm.headers import (
    open_handle_from_message_bytes, close_handle,
)

from fixturegen import FixtureSpec, encode_message
from e2e_helpers import HAS_ECCODES

pytestmark = pytest.mark.skipif(
    not HAS_ECCODES, reason="ecCodes required for profile fixtures")


def _classify(spec):
    raw = encode_message(spec)
    res = scan_file(io.BytesIO(raw), len(raw))
    h = open_handle_from_message_bytes(raw)
    try:
        return classify_message(h, raw, res.messages[0], "f", "c")
    finally:
        close_handle(h)


def test_unsupported_pdt_marked_not_crashed():
    # PDT 4.2 (derived forecast) is outside 4.0/4.1/4.8/4.11.
    spec = FixtureSpec(pdt=2)
    rec = _classify(spec)
    assert rec.status == "unsupported"
    assert rec.reason == "unsupported_pdt"
    assert rec.business_key is None


def test_unsupported_time_unit_for_instant():
    # Unit 3 = 3 hours (a composite/calendar unit), not in the allowed set.
    spec = FixtureSpec(pdt=0, forecast_time=1, time_unit=3)
    rec = _classify(spec)
    assert rec.status == "unsupported"
    assert rec.reason == "unsupported_time_unit"


def test_unsupported_statistical_process():
    # Process 4 (difference/other) is outside {0,1,2,3}.
    from e2e_helpers import stat_spec
    spec = stat_spec(0, "control", 0, 3, process=4)
    rec = _classify(spec)
    assert rec.status == "unsupported"
    assert rec.reason == "unsupported_statistical_process"


def test_unsupported_nonzero_time_increment():
    from e2e_helpers import stat_spec
    spec = stat_spec(0, "control", 0, 3)
    spec.time_increment = 1
    rec = _classify(spec)
    assert rec.status == "unsupported"
    assert rec.reason == "nonzero_time_increment"


def test_unsupported_wrong_type_of_time_increment():
    from e2e_helpers import stat_spec
    spec = stat_spec(0, "control", 0, 3)
    spec.type_of_time_increment = 1
    rec = _classify(spec)
    assert rec.status == "unsupported"
    assert rec.reason == "unsupported_type_of_time_increment"


def test_interval_end_mismatch_detected():
    from e2e_helpers import stat_spec
    spec = stat_spec(0, "control", 0, 3)
    # Lie about the explicit end hour so it disagrees with ref+forecast+range.
    spec.end_hour = spec.end_hour + 1
    rec = _classify(spec)
    assert rec.status == "unsupported"
    assert rec.reason == "interval_end_mismatch"


def test_supported_stat_processes_all_classify_ok():
    from e2e_helpers import stat_spec
    for process in (0, 1, 2, 3):
        rec = _classify(stat_spec(0, "control", 0, 3, process=process))
        assert rec.status == "ok", (process, rec.reason)
        assert rec.slot.statistic == process


def test_unsupported_message_still_kept_in_catalog_as_unsupported(
        tmp_path, grid_digest_value):
    # An unsupported message is persisted (status=unsupported) but never makes
    # a usable cell.
    import hashlib
    from grib_asm.catalog import Catalog
    from grib_asm.request import Manifest, ManifestFile
    spec = FixtureSpec(pdt=2)
    raw = encode_message(spec)
    root = tmp_path / "c"
    root.mkdir()
    rel = "u.grib2"
    (root / rel).write_bytes(raw)
    manifest = Manifest("c", None, [ManifestFile(
        "u", rel, len(raw), hashlib.sha256(raw).hexdigest())], {})
    cat = Catalog(tmp_path / "cat.db")
    res = cat.ingest_file(manifest, manifest.files[0], root)
    assert res.status == "ok"          # the file itself is intact
    assert res.usable_cells == 0       # but it contributes no usable cell
    row = cat._conn.execute(
        "SELECT status, reason FROM messages").fetchone()
    assert row["status"] == "unsupported"
    assert row["reason"] == "unsupported_pdt"
    assert cat._conn.execute("SELECT COUNT(*) c FROM cells").fetchone()[
        "c"] == 0
    cat.close()
