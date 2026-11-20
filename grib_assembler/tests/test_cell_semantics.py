"""Conflict vs retransmission cell semantics at the catalog layer."""

import hashlib

import pytest

from grib_asm.catalog import Catalog
from grib_asm.request import Manifest, ManifestFile

from e2e_helpers import HAS_ECCODES, stat_spec, encode_message

pytestmark = pytest.mark.skipif(
    not HAS_ECCODES, reason="ecCodes required to build cell-producing fixtures")


def _write(tmp_path, cid, fid, raw):
    root = tmp_path / cid
    root.mkdir(parents=True, exist_ok=True)
    rel = f"{fid}.grib2"
    (root / rel).write_bytes(raw)
    manifest = Manifest(cid, None, [ManifestFile(
        fid, rel, len(raw), hashlib.sha256(raw).hexdigest())], {})
    return root, manifest


def test_retransmission_one_cell_many_sources_same_digest(tmp_path):
    raw = encode_message(stat_spec(0, "control", 0, 3, data_seed=4))
    cat = Catalog(tmp_path / "cat.db")
    for fid in ("f1", "f2"):
        root, man = _write(tmp_path, "c", fid, raw)
        cat.ingest_file(man, man.files[0], root)
    cells = cat._conn.execute("SELECT COUNT(*) c FROM cells").fetchone()["c"]
    assert cells == 1
    row = cat._conn.execute("SELECT bkey_digest FROM cells").fetchone()
    cell = cat.cell(row["bkey_digest"])
    assert cell["state"] == "filled"
    assert len(cell["sources"]) == 2
    assert len(cell["distinct_digests"]) == 1
    cat.close()


def test_conflict_same_key_distinct_digest_unusable(tmp_path):
    a = encode_message(stat_spec(6, "control", 0, 3, data_seed=11))
    b = encode_message(stat_spec(6, "control", 0, 3, data_seed=22))
    assert hashlib.sha256(a).hexdigest() != hashlib.sha256(b).hexdigest()
    cat = Catalog(tmp_path / "cat.db")
    for fid, raw in (("a", a), ("b", b)):
        root, man = _write(tmp_path, "c", fid, raw)
        cat.ingest_file(man, man.files[0], root)
    row = cat._conn.execute("SELECT bkey_digest FROM cells").fetchone()
    cell = cat.cell(row["bkey_digest"])
    assert cell["state"] == "conflict"
    assert len(cell["distinct_digests"]) == 2
    assert len(cell["sources"]) == 2
    cat.close()


def test_packing_change_does_not_change_grid_identity(tmp_path):
    # Same grid, different bitsPerValue: different message bytes but the grid
    # digest (Section 3) must be identical.
    from grib_asm.identity import grid_digest
    import io
    from grib_asm.scanner import scan_file
    r1 = encode_message(stat_spec(0, "control", 0, 3, bits=12))
    r2 = encode_message(stat_spec(0, "control", 0, 3, bits=20))
    g1 = grid_digest(scan_file(io.BytesIO(r1), len(r1)).messages[0].gds_bytes)
    g2 = grid_digest(scan_file(io.BytesIO(r2), len(r2)).messages[0].gds_bytes)
    assert g1 == g2
    assert hashlib.sha256(r1).hexdigest() != hashlib.sha256(r2).hexdigest()
