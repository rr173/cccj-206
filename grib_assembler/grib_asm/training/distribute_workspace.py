"""Disk-backed connectivity workspace for distribution.

The two relation kinds (shared GRIB2 message, overlapping observation target
interval) must be unioned *transitively* while keeping memory bounded even when
a single field is referenced by a million rows or one station owns a deep stack
of nested intervals.  This workspace therefore keeps every row, reference and
edge in a SQLite database and maintains the connected components as a
**union-by-size spanning forest** inside the ``rows`` table itself.

Why this preserves transitivity without enumerating all related pairs
----------------------------------------------------------------------

* **Message relations.** Rows referencing one ``message_digest`` form a clique,
  but a clique needs no quadratic edge list: the references are sorted by
  digest on disk and every later reference is joined to the digest's first
  (smallest joint identity) reference, a hub-and-spoke star.  A digest cited
  ``k`` times produces exactly ``k - 1`` edges, so the total is bounded by the
  number of successful-feature references, not by the number of row pairs.
* **Observation relations.** Within each ``(station_id, metric)`` group, rows
  are swept in deterministic ``(target_start, joint_id)`` order with a
  disk-resident active set whose end is extended by ``gap_seconds``
  (``end + gap_seconds``).  Each row is joined only to the still-active row with
  the greatest extended end (ties broken by joint identity).  That single edge
  is enough: every other active row's interval is covered by the chosen one's,
  so connectivity of the interval-overlap graph is identical while the edge
  count stays below one per row.

  The expiration test is deliberately ``ext_end < start`` (strict).  Two
  provisional intervals ``[a, a_end+gap)`` and ``[b, b_end+gap)`` relate iff
  their intersection has positive length, i.e. ``a < b_end+gap``; an empty
  space of exactly ``gap_seconds`` (``a == b_end+gap``) does not relate, and
  with ``gap_seconds = 0`` touching endpoints (``a == b_end``) do not relate
  either.  Expiring only strictly before the new start keeps a row that merely
  *touches* the new interval as a possible bridge to earlier still-overlapping
  rows, preserving transitivity without ever creating a touching-pair edge.

Both builders stream through keyset-paginated queries: Python never holds a
whole group, component or evidence chain.  Built edges are at most
``O(rows + successful-feature references)``; the spanning-forest subset used
for explanations is at most ``rows - 1``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

EDGE_MESSAGE = "message"
EDGE_OBSERVATION = "observation"


@dataclass
class WorkspaceConfig:
    sqlite_cache_kib: int = 16 * 1024
    commit_every: int = 2000
    page_limit: int = 2000


class DistributionWorkspace:
    """Owns the SQLite index: rows, references, edges and the DSU forest."""

    def __init__(self, path: str | Path, *,
                 config: WorkspaceConfig | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config or WorkspaceConfig()
        self.conn = sqlite3.connect(str(self.path))
        self._configure_connection()
        self._create_tables()
        self._reset_counters()

    @classmethod
    def open_readonly(cls, path: str | Path, *,
                      config: WorkspaceConfig | None = None
                      ) -> "DistributionWorkspace":
        """Open a delivered distribution index without mutating it."""

        self = cls.__new__(cls)
        self.path = Path(path)
        self.config = config or WorkspaceConfig()
        self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self._configure_connection(allow_wal=False)
        self._reset_counters()
        return self

    def _reset_counters(self) -> None:
        self._since_commit = 0
        self.built_message_edges = 0
        self.built_observation_edges = 0
        self.forest_edges = 0

    def _configure_connection(self, *, allow_wal: bool = True) -> None:
        self.conn.row_factory = sqlite3.Row
        # Bound SQLite's cache and keep sorters/temp tables on disk: a million
        # references must not be pulled into RAM to be sorted.
        self.conn.execute(
            f"PRAGMA cache_size = -{int(self.config.sqlite_cache_kib)}")
        self.conn.execute("PRAGMA temp_store = FILE")
        if allow_wal:
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")

    # --------------------------------------------------------------- schema
    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE rows (
                joint_id       TEXT PRIMARY KEY,
                export_id      TEXT NOT NULL,
                sample_id      TEXT NOT NULL,
                station_id     TEXT NOT NULL,
                metric         TEXT NOT NULL,
                target_start   INTEGER NOT NULL,
                target_end     INTEGER NOT NULL,
                label_status   TEXT NOT NULL,
                candidate      INTEGER NOT NULL DEFAULT 0,
                window_name    TEXT,
                window_relation TEXT,
                reasons_json   TEXT NOT NULL DEFAULT '[]',
                destination    TEXT,
                group_id       TEXT,
                input_order    INTEGER NOT NULL,
                raw_json       TEXT NOT NULL,
                -- disjoint-set forest, rooted at a row whose parent == itself
                parent         TEXT NOT NULL,
                dsu_size       INTEGER NOT NULL DEFAULT 1,
                tree_min       TEXT NOT NULL,
                parent_edge_id INTEGER
            );

            CREATE TABLE message_refs (
                joint_id    TEXT NOT NULL,
                digest      TEXT NOT NULL,
                column_name TEXT NOT NULL,
                ref_order   INTEGER NOT NULL,
                PRIMARY KEY (joint_id, ref_order)
            );
            CREATE INDEX idx_refs_digest
                ON message_refs (digest, joint_id, ref_order);
            CREATE INDEX idx_rows_sweep
                ON rows (station_id, metric, target_start, joint_id);

            CREATE TABLE edges (
                edge_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                edge_kind   TEXT NOT NULL,
                a_id        TEXT NOT NULL,
                b_id        TEXT NOT NULL,
                is_forest   INTEGER NOT NULL DEFAULT 0,
                -- message-edge evidence
                digest      TEXT,
                column_a    TEXT,
                column_b    TEXT,
                -- observation-edge evidence
                station_id  TEXT,
                metric      TEXT,
                gap_seconds INTEGER,
                a_start     INTEGER,
                a_end       INTEGER,
                b_start     INTEGER,
                b_end       INTEGER
            );
            CREATE INDEX idx_edges_a ON edges (a_id);
            CREATE INDEX idx_edges_b ON edges (b_id);

            CREATE TABLE run_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """)

    def commit(self) -> None:
        self.conn.commit()
        self._since_commit = 0

    def _maybe_commit(self) -> None:
        self._since_commit += 1
        if self._since_commit >= self.config.commit_every:
            self.commit()

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
            (key, value))

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM run_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row["value"]

    # ------------------------------------------------------------- ingestion
    def insert_row(self, parsed, *, input_order: int) -> None:
        self.conn.execute(
            """
            INSERT INTO rows (joint_id, export_id, sample_id, station_id,
                              metric, target_start, target_end, label_status,
                              input_order, raw_json, parent, tree_min)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (parsed.joint_id, parsed.export_id, parsed.sample_id,
             parsed.station_id, parsed.metric, parsed.target_start,
             parsed.target_end, parsed.label_status, input_order,
             parsed.raw_text, parsed.joint_id, parsed.joint_id))
        for ref_order, (column, digest) in enumerate(parsed.message_refs):
            self.conn.execute(
                "INSERT INTO message_refs VALUES (?, ?, ?, ?)",
                (parsed.joint_id, digest, column, ref_order))
        self._maybe_commit()

    def row_count(self) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) c FROM rows").fetchone()["c"])

    def ref_count(self) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) c FROM message_refs").fetchone()["c"])

    # ----------------------------------------------------------- DSU forest
    def _parent(self, joint_id: str) -> str:
        return self.conn.execute(
            "SELECT parent FROM rows WHERE joint_id = ?",
            (joint_id,)).fetchone()["parent"]

    def find(self, joint_id: str) -> str:
        """Root lookup with path halving; depth is logarithmic (union by size)."""

        node = joint_id
        while True:
            parent = self._parent(node)
            if parent == node:
                return node
            grand = self._parent(parent)
            if grand != parent:
                self.conn.execute(
                    "UPDATE rows SET parent = ? WHERE joint_id = ?",
                    (grand, node))
                node = grand
            else:
                node = parent

    def _add_edge(self, *, kind: str, a_id: str, b_id: str,
                  evidence: dict) -> int:
        self.conn.execute(
            """
            INSERT INTO edges (edge_kind, a_id, b_id, digest, column_a,
                               column_b, station_id, metric, gap_seconds,
                               a_start, a_end, b_start, b_end)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (kind, a_id, b_id, evidence.get("digest"),
             evidence.get("column_a"), evidence.get("column_b"),
             evidence.get("station_id"), evidence.get("metric"),
             evidence.get("gap_seconds"), evidence.get("a_start"),
             evidence.get("a_end"), evidence.get("b_start"),
             evidence.get("b_end")))
        edge_id = self.conn.execute("SELECT last_insert_rowid() i"
                                    ).fetchone()["i"]
        return int(edge_id)

    def union(self, a_id: str, b_id: str, *, edge_id: int) -> bool:
        """Merge two trees deterministically; return False if already one.

        All tree metadata (sizes, minima) is read *before* any mutation, then
        the smaller root is attached under the larger one (equal sizes break
        the tie by each tree's smallest joint identity).  This keeps every
        tree's ``dsu_size`` equal to its actual membership count and the
        forest depth logarithmic regardless of edge insertion order.
        """

        root_a = self.find(a_id)
        root_b = self.find(b_id)
        if root_a == root_b:
            return False
        size_a = self._root_size(root_a)
        size_b = self._root_size(root_b)
        min_a = self._root_min(root_a)
        min_b = self._root_min(root_b)
        if size_a < size_b or (size_a == size_b and min_a > min_b):
            child, keep = root_a, root_b
            child_size, keep_size = size_a, size_b
        else:
            child, keep = root_b, root_a
            child_size, keep_size = size_b, size_a
        self.conn.execute(
            """
            UPDATE rows
               SET parent = ?, parent_edge_id = ?
             WHERE joint_id = ?
            """,
            (keep, edge_id, child))
        self.conn.execute(
            "UPDATE rows SET dsu_size = ? WHERE joint_id = ?",
            (keep_size + child_size, keep))
        self.conn.execute(
            "UPDATE rows SET tree_min = ? WHERE joint_id = ?",
            (min(min_a, min_b), keep))
        self.conn.execute("UPDATE edges SET is_forest = 1 WHERE edge_id = ?",
                          (edge_id,))
        self.forest_edges += 1
        self._maybe_commit()
        return True

    def _root_size(self, root: str) -> int:
        return int(self.conn.execute(
            "SELECT dsu_size FROM rows WHERE joint_id = ?",
            (root,)).fetchone()["dsu_size"])

    def _root_min(self, root: str) -> str:
        return self.conn.execute(
            "SELECT tree_min FROM rows WHERE joint_id = ?",
            (root,)).fetchone()["tree_min"]

    def connect(self, *, kind: str, a_id: str, b_id: str,
                evidence: dict) -> None:
        """Build one evidence-bearing edge and union through it."""

        edge_id = self._add_edge(kind=kind, a_id=a_id, b_id=b_id,
                                 evidence=evidence)
        if kind == EDGE_MESSAGE:
            self.built_message_edges += 1
        else:
            self.built_observation_edges += 1
        self.union(a_id, b_id, edge_id=edge_id)

    # ------------------------------------------------------- edge builders
    def build_message_edges(self) -> None:
        """Star-connect every digest's references around its first reference.

        The reference stream is ordered ``(digest, joint_id, ref_order)`` on
        disk.  The first reference of each digest - the smallest joint identity
        - is remembered in a temporary hub table, so hub state survives page
        boundaries without a per-digest Python map; every later reference gets
        one edge to its digest's hub.
        """

        self.conn.execute(
            """
            CREATE TEMP TABLE digest_hubs (
                digest   TEXT PRIMARY KEY,
                hub_id   TEXT NOT NULL,
                hub_col  TEXT NOT NULL
            )
            """)
        limit = self.config.page_limit
        last_digest, last_joint, last_ref = None, None, -1
        while True:
            if last_digest is None:
                rows = self.conn.execute(
                    """
                    SELECT joint_id, digest, column_name, ref_order
                      FROM message_refs
                     ORDER BY digest, joint_id, ref_order
                     LIMIT ?
                    """, (limit,)).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT joint_id, digest, column_name, ref_order
                      FROM message_refs
                     WHERE (digest > ?)
                        OR (digest = ? AND joint_id > ?)
                        OR (digest = ? AND joint_id = ? AND ref_order > ?)
                     ORDER BY digest, joint_id, ref_order
                     LIMIT ?
                    """, (last_digest, last_digest, last_joint,
                          last_digest, last_joint, last_ref, limit)).fetchall()
            if not rows:
                break
            for row in rows:
                digest, joint = row["digest"], row["joint_id"]
                hub = self.conn.execute(
                    "SELECT hub_id, hub_col FROM digest_hubs WHERE digest = ?",
                    (digest,)).fetchone()
                if hub is None:
                    # First (smallest joint identity) reference of the digest:
                    # it becomes the hub and needs no edge to itself.
                    self.conn.execute(
                        "INSERT INTO digest_hubs VALUES (?, ?, ?)",
                        (digest, joint, row["column_name"]))
                elif hub["hub_id"] != joint:
                    self.connect(
                        kind=EDGE_MESSAGE, a_id=hub["hub_id"], b_id=joint,
                        evidence={"digest": digest, "column_a": hub["hub_col"],
                                  "column_b": row["column_name"]})
                last_digest, last_joint, last_ref = (
                    digest, joint, row["ref_order"])
            self._maybe_commit()
            if len(rows) < limit:
                break
        self.conn.execute("DROP TABLE digest_hubs")
        self.commit()

    def build_observation_edges(self, gap_seconds: int) -> None:
        """Sweep each station/metric group with a disk-resident active set."""

        self.conn.execute(
            """
            CREATE TEMP TABLE active_intervals (
                station_id TEXT NOT NULL,
                metric     TEXT NOT NULL,
                ext_end    INTEGER NOT NULL,
                joint_id   TEXT NOT NULL
            )
            """)
        self.conn.execute(
            "CREATE INDEX idx_active ON active_intervals "
            "(station_id, metric, ext_end, joint_id)")
        groups = self.conn.execute(
            "SELECT DISTINCT station_id, metric FROM rows "
            "ORDER BY station_id, metric")
        for group in groups:
            self._sweep_one_group(group["station_id"], group["metric"],
                                  gap_seconds)
            self.conn.execute(
                "DELETE FROM active_intervals WHERE station_id = ? AND "
                "metric = ?", (group["station_id"], group["metric"]))
        self.conn.execute("DROP TABLE active_intervals")
        self.commit()

    def _sweep_one_group(self, station_id: str, metric: str,
                         gap_seconds: int) -> None:
        limit = self.config.page_limit
        last_start, last_joint = None, None
        while True:
            if last_start is None:
                rows = self.conn.execute(
                    """
                    SELECT joint_id, target_start, target_end FROM rows
                     WHERE station_id = ? AND metric = ?
                     ORDER BY target_start, joint_id
                     LIMIT ?
                    """, (station_id, metric, limit)).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT joint_id, target_start, target_end FROM rows
                     WHERE station_id = ? AND metric = ?
                       AND (target_start > ?
                            OR (target_start = ? AND joint_id > ?))
                     ORDER BY target_start, joint_id
                     LIMIT ?
                    """, (station_id, metric, last_start, last_start,
                          last_joint, limit)).fetchall()
            if not rows:
                break
            for row in rows:
                joint, start, end = (row["joint_id"], row["target_start"],
                                     row["target_end"])
                # Expiration uses the *extended* end and is strict: a row whose
                # ext_end equals this start merely touches the new interval
                # (empty gap of exactly gap_seconds), so it must not get an
                # edge to the new row - but it stays available as a bridge to
                # earlier rows whose provisional intervals still overlap.
                self.conn.execute(
                    """
                    DELETE FROM active_intervals
                     WHERE station_id = ? AND metric = ? AND ext_end < ?
                    """, (station_id, metric, start))
                parent = self.conn.execute(
                    """
                    SELECT joint_id, ext_end FROM active_intervals
                     WHERE station_id = ? AND metric = ?
                       AND ext_end > ?
                     ORDER BY ext_end DESC, joint_id ASC
                     LIMIT 1
                    """, (station_id, metric, start)).fetchone()
                if parent is not None:
                    parent_row = self.conn.execute(
                        "SELECT target_start, target_end FROM rows "
                        "WHERE joint_id = ?",
                        (parent["joint_id"],)).fetchone()
                    self.connect(
                        kind=EDGE_OBSERVATION, a_id=parent["joint_id"],
                        b_id=joint,
                        evidence={"station_id": station_id, "metric": metric,
                                  "gap_seconds": gap_seconds,
                                  "a_start": parent_row["target_start"],
                                  "a_end": parent_row["target_end"],
                                  "b_start": start, "b_end": end})
                self.conn.execute(
                    "INSERT INTO active_intervals VALUES (?, ?, ?, ?)",
                    (station_id, metric, end + gap_seconds, joint))
                last_start, last_joint = start, joint
            self._maybe_commit()
            if len(rows) < limit:
                break

    # --------------------------------------------------------- finalization
    def assign_components(self) -> None:
        """Resolve every row to its component and name it by minimum identity.

        Two ordered passes, both keyset-paginated so no component's membership
        is held in Python:

        1. iterate all rows in joint-id order and write each row's current
           DSU root into ``group_id`` (path halving keeps this logarithmic);
        2. in the same order, the first row encountered for any root is the
           component's smallest joint identity, so every row of that root is
           renamed to it.  A temporary ``component_names`` table makes the
           rename independent of correlated-subquery semantics.
        """

        self.conn.execute(
            "CREATE TEMP TABLE component_names ("
            "root TEXT PRIMARY KEY, min_id TEXT NOT NULL)")
        last_joint = None
        limit = self.config.page_limit
        while True:
            if last_joint is None:
                rows = self.conn.execute(
                    "SELECT joint_id FROM rows ORDER BY joint_id LIMIT ?",
                    (limit,)).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT joint_id FROM rows WHERE joint_id > ? "
                    "ORDER BY joint_id LIMIT ?",
                    (last_joint, limit)).fetchall()
            if not rows:
                break
            for row in rows:
                joint = row["joint_id"]
                root = self.find(joint)
                self.conn.execute(
                    "UPDATE rows SET group_id = ? WHERE joint_id = ?",
                    (root, joint))
                # Joint ids are visited ascending: the first sighting of a
                # root is automatically its smallest member identity.
                self.conn.execute(
                    "INSERT OR IGNORE INTO component_names VALUES (?, ?)",
                    (root, joint))
            self.commit()
            last_joint = rows[-1]["joint_id"]
            if len(rows) < limit:
                break
        self.conn.execute(
            """
            UPDATE rows
               SET group_id = (
                    SELECT min_id FROM component_names
                     WHERE component_names.root = rows.group_id)
            """)
        self.conn.execute("DROP TABLE component_names")
        self.commit()

    def group_window_counts(self):
        """One row per component with its candidate counts per window."""

        return self.conn.execute(
            """
            SELECT group_id,
                   SUM(CASE WHEN candidate = 1 AND window_name = 'train'
                            THEN 1 ELSE 0 END) AS n_train,
                   SUM(CASE WHEN candidate = 1 AND window_name = 'validation'
                            THEN 1 ELSE 0 END) AS n_validation,
                   SUM(CASE WHEN candidate = 1 AND window_name = 'holdout'
                            THEN 1 ELSE 0 END) AS n_holdout,
                   SUM(CASE WHEN candidate = 1 THEN 1 ELSE 0 END) AS n_cand
              FROM rows
             GROUP BY group_id
             ORDER BY group_id
            """)

    # ------------------------------------------------------------- explain
    def get_row(self, joint_id: str):
        return self.conn.execute(
            "SELECT * FROM rows WHERE joint_id = ?",
            (joint_id,)).fetchone()

    def get_edge(self, edge_id: int):
        return self.conn.execute(
            "SELECT * FROM edges WHERE edge_id = ?",
            (edge_id,)).fetchone()

    def neighbors(self, joint_id: str) -> list[sqlite3.Row]:
        """Every evidence edge incident on one node, via the edge indexes.

        Both endpoint indexes are used, so an explanation expands only the
        asked row's component instead of scanning the whole table.  Rows are
        returned in a fixed ``edge_id`` order for deterministic BFS traversal.
        """

        return self.conn.execute(
            """
            SELECT * FROM edges WHERE a_id = ?
            UNION ALL
            SELECT * FROM edges WHERE b_id = ?
            ORDER BY edge_id
            """, (joint_id, joint_id)).fetchall()

    def ancestors(self, joint_id: str) -> list[str]:
        """Root path; logarithmic because unions are union-by-size."""

        path = []
        node = joint_id
        while True:
            path.append(node)
            row = self.get_row(node)
            if row["parent"] == node:
                return path
            node = row["parent"]

    def close(self) -> None:
        self.conn.commit()
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
