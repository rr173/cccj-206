"""Disk-backed staging workspace (SQLite) for the streaming export.

The sample manifest and the three history streams are explicitly allowed to be
too large to hold in memory, so nothing here accumulates them as Python
collections.  Every parsed line is inserted into a temporary SQLite database:

* ``samples``     - one row per input sample, insertion order preserved;
* ``events``      - one row per publish/station/observation stream record,
                    keyed by its stream business key, with the full JSON doc;
* ``point_requests`` - populated by the resolution pass and consumed by the
                    chunk-bounded point reader, so scattered samples that need
                    the same value block are grouped on disk first.

Same-instant conflicts are detected incrementally with a bounded in-memory
dictionary of ``(business_key, visible_at, seq) -> payload`` tokens (one entry
per distinct timestamp/sequence, not per line).  SQLite's own page cache is
sized from the export configuration, which is what bounds the working set of
the resolution queries.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .errors import StreamError
from .model import (
    StreamRecord, epoch_iso, observation_record, publish_record,
    sample_from_dict, station_record, validate_sample_against_recipe,
)


@dataclass
class WorkspaceStats:
    samples: int = 0
    publish_events: int = 0
    station_events: int = 0
    observation_events: int = 0
    point_requests: int = 0


def _typed_columns(rec: StreamRecord) -> tuple:
    """Extract integer-typed key columns for clock/interval SQL comparisons."""

    doc = rec.doc
    if rec.stream == "station":
        return (str(doc["station_id"]),
                int(_iso_epoch(doc["effective_from"])),
                int(_iso_epoch(doc["effective_to"])),
                None, None, None, None)
    if rec.stream == "publish":
        return (None, None, None, int(_iso_epoch(doc["origin"])),
                None, None, None)
    if rec.stream == "observation":
        return (str(doc["station_id"]), None, None, None,
                str(doc["metric"]),
                int(_iso_epoch(doc["target_start"])),
                int(_iso_epoch(doc["target_end"])))
    return (None, None, None, None, None, None, None)


def _iso_epoch(text: str) -> int:
    from datetime import datetime, timezone
    return int(datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
               .replace(tzinfo=timezone.utc).timestamp())


class TrainingWorkspace:
    """Owns the temporary SQLite file and all bounded-memory staging."""

    def __init__(self, path: str | Path, *, sqlite_cache_kib: int = 64 * 1024,
                 commit_every: int = 1000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        # Bound SQLite's own memory; the rest of the pipeline streams.
        self.conn.execute(f"PRAGMA cache_size = -{int(sqlite_cache_kib)}")
        self.conn.execute("PRAGMA temp_store = MEMORY")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.commit_every = int(commit_every)
        self._create_tables()
        # (business_key, visible_at, seq) -> canonical payload, per stream,
        # for incremental resend/conflict detection.
        self._seen_tokens: dict[str, dict[tuple, str]] = {
            "publish": {}, "station": {}, "observation": {}}
        self._seen_sample_ids: set[str] = set()
        self._since_commit = 0
        self.stats = WorkspaceStats()

    # --------------------------------------------------------------- schema
    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE samples (
                sample_id      TEXT PRIMARY KEY,
                station_id     TEXT NOT NULL,
                metric         TEXT NOT NULL,
                target_start   INTEGER NOT NULL,
                target_end     INTEGER NOT NULL,
                decision_at    INTEGER NOT NULL,
                input_order    INTEGER NOT NULL
            );

            CREATE TABLE events (
                stream         TEXT NOT NULL,
                record_id      TEXT NOT NULL,
                business_key   TEXT NOT NULL,
                visible_at     INTEGER NOT NULL,
                seq            INTEGER NOT NULL,
                action         TEXT NOT NULL,
                doc_json       TEXT NOT NULL,
                -- Typed business-key columns (NULL where inapplicable) so
                -- interval/clock comparisons stay integer, never text.
                station_id     TEXT,
                effective_from INTEGER,
                effective_to   INTEGER,
                origin_epoch   INTEGER,
                metric         TEXT,
                target_start   INTEGER,
                target_end     INTEGER
            );
            CREATE INDEX idx_events_key
                ON events (stream, business_key, visible_at, seq);
            CREATE INDEX idx_station
                ON events (stream, station_id, visible_at, seq);
            CREATE INDEX idx_observation
                ON events (stream, station_id, metric, target_start,
                           target_end, visible_at, seq);

            CREATE TABLE point_requests (
                request_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id      TEXT NOT NULL,
                column_index   INTEGER NOT NULL,
                var_key        TEXT NOT NULL,
                origin_index   INTEGER NOT NULL,
                level_index    INTEGER NOT NULL,
                member_index   INTEGER NOT NULL,
                slot_index     INTEGER NOT NULL,
                lat_index      INTEGER NOT NULL,
                lon_index      INTEGER NOT NULL,
                origin_epoch   INTEGER NOT NULL
            );
            CREATE INDEX idx_requests_chunk
                ON point_requests (var_key, origin_index, level_index,
                                   member_index, slot_index);
            CREATE INDEX idx_requests_sample
                ON point_requests (sample_id, column_index);

            CREATE TABLE point_results (
                sample_id      TEXT NOT NULL,
                column_index   INTEGER NOT NULL,
                value          REAL,
                present        INTEGER NOT NULL,
                message_digest TEXT,
                package_digest TEXT,
                raw_unit       TEXT,
                PRIMARY KEY (sample_id, column_index)
            );
            """)

    def _maybe_commit(self) -> None:
        self._since_commit += 1
        if self._since_commit >= self.commit_every:
            self.conn.commit()
            self._since_commit = 0

    # ------------------------------------------------------------- ingestion
    def add_sample(self, doc: dict) -> None:
        sample = sample_from_dict(doc, self.stats.samples)
        if sample.sample_id in self._seen_sample_ids:
            raise StreamError("stream_record_invalid", {
                "context": "samples",
                "sample_id": sample.sample_id,
                "reason": "duplicate sample_id"})
        self._seen_sample_ids.add(sample.sample_id)
        # Recipe-dependent invariants are checked as samples arrive; the
        # recipe is attached via ``set_recipe`` before ingestion starts.
        validate_sample_against_recipe(sample, self._recipe)
        self.conn.execute(
            "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sample.sample_id, sample.station_id, sample.metric,
             sample.target_start_epoch, sample.target_end_epoch,
             sample.decision_epoch, sample.order))
        self.stats.samples += 1
        self._maybe_commit()

    def set_recipe(self, recipe) -> None:
        self._recipe = recipe

    def add_publish(self, doc: dict) -> None:
        rec = publish_record(doc, self.stats.publish_events)
        self._ingest_event(rec)
        self.stats.publish_events += 1

    def add_station(self, doc: dict) -> None:
        rec = station_record(doc, self.stats.station_events)
        self._ingest_event(rec)
        self.stats.station_events += 1

    def add_observation(self, doc: dict) -> None:
        rec = observation_record(doc, self.stats.observation_events)
        self._ingest_event(rec)
        self.stats.observation_events += 1

    def _ingest_event(self, rec: StreamRecord) -> None:
        tokens = self._seen_tokens[rec.stream]
        token = (rec.business_key, rec.visible_at_epoch, rec.seq)
        previous = tokens.get(token)
        if previous is None:
            tokens[token] = rec.payload
        elif previous != rec.payload:
            # Same clock time + sequence, different content: the input must
            # be fixed; line order must never pick a winner.
            raise StreamError("stream_duplicate_conflict", {
                "stream": rec.stream,
                "business_key": json.loads(rec.business_key),
                "visible_at": epoch_iso(rec.visible_at_epoch),
                "seq": rec.seq,
                "reason": "same visibility time and sequence but different "
                          "content"})
            # Identical resend: deduplicated (not inserted again).
        else:
            return
        self.conn.execute(
            """
            INSERT INTO events (
                stream, record_id, business_key, visible_at, seq, action,
                doc_json, station_id, effective_from, effective_to,
                origin_epoch, metric, target_start, target_end)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (rec.stream, rec.record_id, rec.business_key,
             rec.visible_at_epoch, rec.seq, rec.action, rec.payload,
             *_typed_columns(rec)))
        self._maybe_commit()

    # ------------------------------------------------------------- resolution
    def station_winner(self, station_id: str, target_start_epoch: int,
                       cutoff_epoch: int) -> sqlite3.Row | None:
        """Newest map/unmap visible at the cutoff and covering target start.

        The coverage test ``effective_from <= target_start < effective_to``
        applies to both actions: an ``unmap`` only masks the interval it
        names, so an older ``map`` whose effective range still covers the
        target start remains the winner when no newer covering record exists.
        """

        return self.conn.execute(
            """
            SELECT * FROM events
            WHERE stream = 'station'
              AND station_id = ?
              AND visible_at <= ?
              AND effective_from <= ?
              AND effective_to   >  ?
            ORDER BY visible_at DESC, seq DESC
            LIMIT 1
            """,
            (station_id, cutoff_epoch, target_start_epoch,
             target_start_epoch)).fetchone()

    def publish_winner(self, origin_epoch: int,
                       cutoff_epoch: int) -> sqlite3.Row | None:
        business_key = json.dumps(
            {"stream": "publish", "origin": origin_epoch},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return self.conn.execute(
            """
            SELECT * FROM events
            WHERE stream = 'publish' AND business_key = ?
              AND visible_at <= ?
            ORDER BY visible_at DESC, seq DESC
            LIMIT 1
            """,
            (business_key, cutoff_epoch)).fetchone()

    def observation_winner(self, station_id: str, metric: str,
                           target_start_epoch: int, target_end_epoch: int,
                           cutoff_epoch: int) -> sqlite3.Row | None:
        business_key = json.dumps(
            {"stream": "observation", "station_id": station_id,
             "metric": metric, "target_start": target_start_epoch,
             "target_end": target_end_epoch},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return self.conn.execute(
            """
            SELECT * FROM events
            WHERE stream = 'observation' AND business_key = ?
              AND visible_at <= ?
            ORDER BY visible_at DESC, seq DESC
            LIMIT 1
            """,
            (business_key, cutoff_epoch)).fetchone()
    def iter_samples(self):
        """Stream every sample row; output ordering is applied by the caller."""

        return self.conn.execute(
            "SELECT * FROM samples ORDER BY input_order")

    def sample_count(self) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) c FROM samples").fetchone()["c"])

    # ------------------------------------------------------- point scheduling
    def add_point_request(self, *, sample_id: str, column_index: int,
                          var_key: str, origin_index: int, level_index: int,
                          member_index: int, slot_index: int,
                          lat_index: int, lon_index: int,
                          origin_epoch: int) -> None:
        self.conn.execute(
            """
            INSERT INTO point_requests
                (sample_id, column_index, var_key, origin_index, level_index,
                 member_index, slot_index, lat_index, lon_index, origin_epoch)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (sample_id, column_index, var_key, origin_index, level_index,
             member_index, slot_index, lat_index, lon_index, origin_epoch))
        self._maybe_commit()

    def distinct_chunks(self):
        """Each distinct values block that must be decoded, exactly once."""

        return self.conn.execute(
            """
            SELECT DISTINCT var_key, origin_index, level_index,
                            member_index, slot_index
            FROM point_requests
            ORDER BY var_key, origin_index, level_index,
                     member_index, slot_index
            """)

    def requests_for_chunk(self, var_key: str, origin_index: int,
                           level_index: int, member_index: int,
                           slot_index: int):
        return self.conn.execute(
            """
            SELECT * FROM point_requests
            WHERE var_key = ? AND origin_index = ? AND level_index = ?
              AND member_index = ? AND slot_index = ?
            """,
            (var_key, origin_index, level_index, member_index,
             slot_index))

    def store_result(self, *, sample_id: str, column_index: int,
                     value, present: int, message_digest, package_digest,
                     raw_unit) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO point_results VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sample_id, column_index,
             None if value is None else float(value), int(present),
             message_digest, package_digest, raw_unit))
        self._maybe_commit()

    def result_for(self, sample_id: str, column_index: int):
        return self.conn.execute(
            "SELECT * FROM point_results WHERE sample_id = ? AND column_index = ?",
            (sample_id, column_index)).fetchone()

    # ------------------------------------------------------------ lifecycle
    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    def destroy(self) -> None:
        """Remove the database and its WAL/SHM sidecars (cleanable workspace)."""

        self.close()
        for suffix in ("", "-wal", "-shm"):
            p = self.path.with_name(self.path.name + suffix)
            try:
                p.unlink()
            except FileNotFoundError:
                pass
