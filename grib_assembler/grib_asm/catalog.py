"""Disk-backed catalog of scanned files, messages and request cells.

All per-message state lives in a SQLite database so memory does not grow with
the archive size; only one message is materialised at a time.  The catalog is
append/evidence oriented:

* failed files keep their structural or manifest evidence and are never
  silently retried as usable;
* every message keeps its offset, length, whole-message SHA-256 and raw
  header snapshot;
* a business key with several sources sharing one message digest is a
  retransmission (one cell, many sources);
* a business key with more than one distinct message digest is a conflict and
  the cell is never usable, regardless of file timestamps.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .classify import NormalizedRecord, classify_message
from .errors import BadFileError
from .headers import close_handle, open_handle_from_file
from .request import Manifest, ManifestFile
from .scanner import MessageSpan, scan_file
from .timeutil import iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    collection_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    manifest_size INTEGER,
    actual_size INTEGER,
    manifest_sha256 TEXT,
    actual_sha256 TEXT,
    status TEXT NOT NULL,           -- ok | failed
    reason TEXT,
    evidence TEXT,
    scanned_at TEXT NOT NULL,
    PRIMARY KEY (collection_id, file_id)
);
CREATE TABLE IF NOT EXISTS messages (
    collection_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    msg_index INTEGER NOT NULL,
    offset INTEGER NOT NULL,
    length INTEGER NOT NULL,
    message_sha256 TEXT NOT NULL,
    grid_digest TEXT,
    pdt INTEGER,
    status TEXT NOT NULL,           -- ok | unsupported
    reason TEXT,
    reference_time TEXT,
    bkey_digest TEXT,
    identity TEXT,
    header_snapshot TEXT,
    PRIMARY KEY (collection_id, file_id, msg_index)
);
CREATE TABLE IF NOT EXISTS cells (
    bkey_digest TEXT PRIMARY KEY,
    reference_time TEXT NOT NULL,
    key_json TEXT NOT NULL,
    state TEXT NOT NULL,            -- filled | conflict
    distinct_digests TEXT NOT NULL  -- JSON list of message digests
);
CREATE TABLE IF NOT EXISTS cell_sources (
    bkey_digest TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    msg_index INTEGER NOT NULL,
    message_sha256 TEXT NOT NULL,
    PRIMARY KEY (bkey_digest, collection_id, file_id, msg_index)
);
CREATE INDEX IF NOT EXISTS idx_cells_ref ON cells(reference_time);
CREATE INDEX IF NOT EXISTS idx_msgs_bkey ON messages(bkey_digest);
"""


def _now() -> str:
    return iso(datetime.now(timezone.utc))


@dataclass
class FileIngestResult:
    file_id: str
    status: str
    reason: str | None
    message_count: int
    usable_cells: int


class Catalog:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------ ingest
    def already_ingested(self, collection_id: str, file_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM files WHERE collection_id=? AND file_id=?",
            (collection_id, file_id)).fetchone()
        return row is not None

    def ingest_file(self, manifest: Manifest, entry: ManifestFile,
                    input_root: Path, bufsize: int = 1024 * 1024,
                    skip_existing: bool = True) -> FileIngestResult:
        """Verify one file against the manifest, scan and classify it."""

        if skip_existing and self.already_ingested(manifest.collection_id,
                                                   entry.file_id):
            return FileIngestResult(entry.file_id, "cached", None, 0, 0)

        path = input_root / entry.relative_path
        evidence: dict = {"relative_path": entry.relative_path}
        if not path.is_file():
            return self._record_failed(manifest, entry, None, None,
                                       "missing_file", evidence)
        actual_size = path.stat().st_size
        if actual_size != entry.size_bytes:
            evidence.update(manifest_size=entry.size_bytes,
                            actual_size=actual_size)
            return self._record_failed(manifest, entry, actual_size, None,
                                       "size_mismatch", evidence)

        try:
            with open(path, "rb") as fh:
                scan = scan_file(fh, actual_size, bufsize=bufsize)
        except BadFileError as exc:
            evidence.update(exc.evidence)
            return self._record_failed(manifest, entry, actual_size, None,
                                       exc.reason, evidence)
        except OSError as exc:
            evidence.update(os_error=str(exc))
            return self._record_failed(manifest, entry, actual_size, None,
                                       "io_error", evidence)

        if scan.file_sha256 != entry.sha256:
            evidence.update(manifest_sha256=entry.sha256,
                            actual_sha256=scan.file_sha256)
            return self._record_failed(manifest, entry, actual_size,
                                       scan.file_sha256,
                                       "manifest_digest_mismatch", evidence)

        # File is intact and structurally valid: persist the file row, then
        # classify each message with at most one open ecCodes handle.
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (manifest.collection_id, entry.file_id, entry.relative_path,
                 entry.size_bytes, actual_size, entry.sha256,
                 scan.file_sha256, "ok", None, None, _now()))
            usable = 0
            for idx, span in enumerate(scan.messages):
                rec = self._classify_span(path, span, entry.file_id,
                                          manifest.collection_id, idx)
                self._store_message(conn, rec, idx)
                if rec.status == "ok":
                    self._store_cell(conn, rec, idx)
                    usable += 1
        return FileIngestResult(entry.file_id, "ok", None,
                                len(scan.messages), usable)

    def _classify_span(self, path: Path, span: MessageSpan,
                       file_id: str, collection_id: str,
                       msg_index: int) -> NormalizedRecord:
        # Open exactly this one encoded message straight from disk at its
        # offset and read only headers.  Numeric identity codes are parsed
        # from small section reads (gribbytes); ecCodes reads timing scalars.
        # No whole-message byte copy and no data array is held; the handle is
        # released before the next message.
        with open(path, "rb") as fh:
            handle = open_handle_from_file(fh, span.offset)
            try:
                return classify_message(
                    handle, None, span, file_id, collection_id,
                    fileobj=fh, msg_offset=span.offset,
                    msg_length=span.length)
            finally:
                close_handle(handle)

    def _record_failed(self, manifest: Manifest, entry: ManifestFile,
                       actual_size: int | None, actual_sha: str | None,
                       reason: str, evidence: dict) -> FileIngestResult:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (manifest.collection_id, entry.file_id, entry.relative_path,
                 entry.size_bytes, actual_size, entry.sha256, actual_sha,
                 "failed", reason, json.dumps(evidence, sort_keys=True),
                 _now()))
        return FileIngestResult(entry.file_id, "failed", reason, 0, 0)

    def _store_message(self, conn: sqlite3.Connection,
                       rec: NormalizedRecord, msg_index: int) -> None:
        identity = None
        bkey = None
        ref = None
        if rec.status == "ok" and rec.business_key is not None:
            identity = json.dumps(rec.business_key.to_dict(),
                                  sort_keys=True, ensure_ascii=False)
            bkey = rec.business_key.digest()
            ref = rec.business_key.reference_time
        conn.execute(
            "INSERT OR REPLACE INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec.collection_id, rec.file_id, msg_index,
             rec.span.offset, rec.span.length, rec.span.sha256,
             rec.grid_digest, rec.pdt, rec.status, rec.reason, ref, bkey,
             identity, json.dumps(rec.header_snapshot, sort_keys=True,
                                  ensure_ascii=False)))

    def _store_cell(self, conn: sqlite3.Connection,
                    rec: NormalizedRecord, msg_index: int) -> None:
        assert rec.business_key is not None
        digest = rec.business_key.digest()
        row = conn.execute("SELECT * FROM cells WHERE bkey_digest=?",
                           (digest,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO cells VALUES (?,?,?,?,?)", (
                digest, rec.business_key.reference_time,
                rec.business_key.canonical(), "filled",
                json.dumps([rec.span.sha256])))
        else:
            digests = set(json.loads(row["distinct_digests"]))
            digests.add(rec.span.sha256)
            state = "conflict" if len(digests) > 1 else "filled"
            conn.execute(
                "UPDATE cells SET state=?, distinct_digests=? WHERE bkey_digest=?",
                (state, json.dumps(sorted(digests)), digest))
        conn.execute(
            "INSERT OR REPLACE INTO cell_sources VALUES (?,?,?,?,?)",
            (digest, rec.collection_id, rec.file_id,
             msg_index, rec.span.sha256))

    # ------------------------------------------------------------------ queries
    def failed_files(self, collection_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM files WHERE collection_id=? AND status='failed' "
            "ORDER BY file_id", (collection_id,)).fetchall()
        return [dict(r) for r in rows]

    def ok_files(self, collection_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM files WHERE collection_id=? AND status='ok' "
            "ORDER BY file_id", (collection_id,)).fetchall()
        return [dict(r) for r in rows]

    def cell(self, bkey_digest: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM cells WHERE bkey_digest=?",
                                 (bkey_digest,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["distinct_digests"] = json.loads(out["distinct_digests"])
        out["sources"] = [dict(r) for r in self._conn.execute(
            "SELECT * FROM cell_sources WHERE bkey_digest=? ORDER BY "
            "collection_id, file_id, msg_index", (bkey_digest,)).fetchall()]
        return out

    def cycle_reference_times(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT reference_time FROM cells ORDER BY reference_time"
        ).fetchall()
        return [r["reference_time"] for r in rows]

    def message_row(self, collection_id: str, file_id: str,
                    msg_index: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM messages WHERE collection_id=? AND file_id=? "
            "AND msg_index=?", (collection_id, file_id, msg_index)).fetchone()
        return None if row is None else dict(row)
