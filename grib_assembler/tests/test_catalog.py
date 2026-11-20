"""Catalog ingest, manifest verification and cell state tests.

These use the byte-accurate raw GRIB2 frames so manifest-size/digest and
conflict/retransmission behaviour can be exercised without ecCodes.  Messages
that cannot be classified (no ecCodes headers here) simply never produce a
cell, which is enough to validate persistence and verification rules.
"""

import hashlib
import json

import pytest

from grib_asm.catalog import Catalog
from grib_asm.request import Manifest, ManifestFile

from raw_frames import make_message, sha

try:
    import eccodes  # noqa: F401
    HAS_ECCODES = True
except Exception:
    HAS_ECCODES = False

requires_eccodes = pytest.mark.skipif(
    not HAS_ECCODES, reason="ecCodes required for header classification")


def _manifest(collection_id, entries, root):
    files = []
    for fid, blob in entries:
        rel = f"{fid}.grib2"
        (root / rel).write_bytes(blob)
        files.append(ManifestFile(fid, rel, len(blob), sha(blob)))
    return Manifest(collection_id, None, files,
                    {"collection_id": collection_id, "files": []})


def test_missing_file_is_failed_with_evidence(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    man = Manifest("c1", None, [
        ManifestFile("f_missing", "nope.grib2", 10, "a" * 64)], {})
    db = tmp_path / "cat.db"
    cat = Catalog(db)
    res = cat.ingest_file(man, man.files[0], root)
    assert res.status == "failed"
    assert res.reason == "missing_file"
    failed = cat.failed_files("c1")
    assert len(failed) == 1
    ev = json.loads(failed[0]["evidence"])
    assert ev["relative_path"] == "nope.grib2"
    cat.close()


def test_size_mismatch_makes_file_unusable(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    blob = make_message(b"G")
    rel = "f.grib2"
    (root / rel).write_bytes(blob)
    man = Manifest("c1", None, [
        ManifestFile("f", rel, len(blob) + 99, sha(blob))], {})
    cat = Catalog(tmp_path / "cat.db")
    res = cat.ingest_file(man, man.files[0], root)
    assert res.reason == "size_mismatch"
    assert cat.failed_files("c1")[0]["status"] == "failed"
    cat.close()


def test_digest_mismatch_makes_file_unusable(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    blob = make_message(b"G")
    rel = "f.grib2"
    (root / rel).write_bytes(blob)
    man = Manifest("c1", None, [
        ManifestFile("f", rel, len(blob), "f" * 64)], {})
    cat = Catalog(tmp_path / "cat.db")
    res = cat.ingest_file(man, man.files[0], root)
    assert res.reason == "manifest_digest_mismatch"
    row = cat.failed_files("c1")[0]
    ev = json.loads(row["evidence"])
    assert ev["actual_sha256"] == sha(blob)
    assert ev["manifest_sha256"] == "f" * 64
    cat.close()


def test_broken_boundary_fails_and_keeps_evidence(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    good = make_message(b"G")
    blob = good + b"JUNK"
    rel = "broken.grib2"
    (root / rel).write_bytes(blob)
    man = Manifest("c1", None, [
        ManifestFile("f", rel, len(blob), sha(blob))], {})
    cat = Catalog(tmp_path / "cat.db")
    res = cat.ingest_file(man, man.files[0], root)
    assert res.status == "failed"
    assert res.reason == "trailing_junk"
    cat.close()


@requires_eccodes
def test_ok_file_persists_messages(tmp_path):
    from fixturegen import FixtureSpec, encode_message
    root = tmp_path / "in"
    root.mkdir()
    m1, m2 = (encode_message(FixtureSpec(data_seed=1)),
              encode_message(FixtureSpec(data_seed=2)))
    man = _manifest("c1", [("f", m1 + m2)], root)
    cat = Catalog(tmp_path / "cat.db")
    res = cat.ingest_file(man, man.files[0], root)
    assert res.status == "ok"
    assert res.message_count == 2
    rows = cat._conn.execute(
        "SELECT * FROM messages ORDER BY msg_index").fetchall()
    assert [r["offset"] for r in rows] == [0, len(m1)]
    assert rows[0]["message_sha256"] == sha(m1)
    cat.close()


@requires_eccodes
def test_reingest_is_idempotent(tmp_path):
    from fixturegen import FixtureSpec, encode_message
    root = tmp_path / "in"
    root.mkdir()
    man = _manifest("c1", [("f", encode_message(FixtureSpec(data_seed=3)))],
                    root)
    cat = Catalog(tmp_path / "cat.db")
    first = cat.ingest_file(man, man.files[0], root)
    second = cat.ingest_file(man, man.files[0], root)
    assert first.status == "ok"
    assert second.status == "cached"
    n = cat._conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
    assert n == 1
    cat.close()
