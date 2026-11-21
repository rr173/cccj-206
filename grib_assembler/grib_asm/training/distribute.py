"""Offline distribution of already-exported training rows.

The engine consumes *only* the JSONL rows produced by one or more
``cube-training-export`` runs.  It never opens a cube, never re-fetches grid
points, never re-runs forecast publish / station registration / observation
revision matching and never trains anything.  Its four jobs are:

1. parse every exported row under its manifest-assigned ``export_id`` and
   reject malformed identities, intervals or allegedly successful features
   that lack a legal message digest;
2. build the two relation kinds (shared GRIB2 message digest, overlapping
   observation target intervals under the ``gap_seconds`` rule) directly into
   the disk-backed union forest of
   :mod:`grib_asm.training.distribute_workspace`, so transitivity across both
   sources is retained without enumerating related pairs;
3. apply the delivery plan: label-``ok`` rows whose target interval lies fully
   in one window are candidates; a component whose candidates name exactly one
   window is delivered to that shard, every other row is isolated with one or
   more explicit reasons;
4. emit three shard files, one isolation file, a per-row decision journal, a
   count summary and a self-contained explanation index.

Every output is written to a ``.tmp`` path and atomically replaced, mirroring
the export stage: a failed run leaves nothing that looks like a result.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .distribute_errors import ManifestError, RowError
from .distribute_plan import (
    DeliveryPlan, ExportEntry, load_delivery_plan, load_export_manifest,
)
from .distribute_rows import (
    ALL_ISOLATION_REASONS, REASON_CROSSES_WINDOWS,
    REASON_GROUP_SPANS_MULTIPLE_WINDOWS, REASON_LABEL_UNAVAILABLE,
    REASON_OUTSIDE_WINDOWS, parse_export_row,
)
from .distribute_workspace import (
    DistributionWorkspace, WorkspaceConfig,
)

DECISION_TRAIN = "train"
DECISION_VALIDATION = "validation"
DECISION_HOLDOUT = "holdout"
DECISION_ISOLATED = "isolated"
SHARD_DECISIONS = (DECISION_TRAIN, DECISION_VALIDATION, DECISION_HOLDOUT)
DECISIONS = SHARD_DECISIONS + (DECISION_ISOLATED,)

INDEX_NAME = "distribution-index.sqlite"
INDEX_MANIFEST_NAME = "distribution-index.json"
SCHEMA_VERSION = 1


@dataclass
class DistributionConfig:
    work_dir: str | None = None
    keep_workspace: bool = False
    sqlite_cache_kib: int = 16 * 1024
    commit_every: int = 2000
    page_limit: int = 2000


@dataclass
class DistributionReport:
    output_dir: str
    index_path: str
    row_count: int
    group_count: int
    shard_counts: dict = field(default_factory=dict)
    isolated_count: int = 0
    reason_counts: dict = field(default_factory=dict)
    group_stats: dict = field(default_factory=dict)
    relation_scale: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "output_dir": self.output_dir,
            "index_path": self.index_path,
            "row_count": self.row_count,
            "group_count": self.group_count,
            "shard_counts": self.shard_counts,
            "isolated_count": self.isolated_count,
            "reason_counts": self.reason_counts,
            "group_stats": self.group_stats,
            "relation_scale": self.relation_scale,
            "files": self.files,
        }


def _stream_jsonl(path: Path, *, export_id: str):
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                doc = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RowError("row_record_invalid", {
                    "export_id": export_id, "path": str(path),
                    "line_no": line_no, "reason": f"invalid JSON: {exc.msg}"})
            yield line_no, doc


class Distributor:
    """One full distribution run over a manifest and a delivery plan."""

    def __init__(self, plan: DeliveryPlan, entries: list[ExportEntry],
                 config: DistributionConfig):
        self.plan = plan
        self.entries = entries
        self.config = config
        self.shard_counts = {d: 0 for d in SHARD_DECISIONS}
        self.reason_counts = {r: 0 for r in ALL_ISOLATION_REASONS}
        self.isolated_count = 0

    # ------------------------------------------------------------------ run
    def run(self, workspace: DistributionWorkspace, out_dir: Path
            ) -> DistributionReport:
        self._ingest(workspace)
        workspace.commit()
        self._classify_rows(workspace)
        workspace.build_message_edges()
        workspace.build_observation_edges(self.plan.gap_seconds)
        # The forest is now final; resolve every row to its component and name
        # each component after its smallest joint identity.  Doing this before
        # all unions would freeze stale roots.
        workspace.assign_components()
        group_stats = self._apply_group_decisions(workspace)
        files = self._emit(workspace, out_dir)
        relation_scale = {
            "rows": workspace.row_count(),
            "successful_feature_refs": workspace.ref_count(),
            "message_edges": workspace.built_message_edges,
            "observation_edges": workspace.built_observation_edges,
            "total_edges": (workspace.built_message_edges
                            + workspace.built_observation_edges),
            "forest_edges": workspace.forest_edges,
            "edge_bound": "message_edges <= successful_feature_refs; "
                          "observation_edges < rows; total edges = O(rows + "
                          "successful_feature_refs)",
        }
        group_count = int(workspace.conn.execute(
            "SELECT COUNT(DISTINCT group_id) c FROM rows").fetchone()["c"])
        workspace.set_meta("schema_version", str(SCHEMA_VERSION))
        workspace.set_meta("gap_seconds", str(self.plan.gap_seconds))
        workspace.set_meta("relation_scale", json.dumps(relation_scale))
        workspace.commit()
        return DistributionReport(
            output_dir=str(out_dir),
            index_path=str(out_dir / INDEX_NAME),
            row_count=workspace.row_count(), group_count=group_count,
            shard_counts=dict(self.shard_counts),
            isolated_count=self.isolated_count,
            reason_counts=dict(self.reason_counts),
            group_stats=group_stats, relation_scale=relation_scale,
            files=files)

    # --------------------------------------------------------------- ingest
    def _ingest(self, workspace: DistributionWorkspace) -> None:
        input_order = 0
        for entry in self.entries:
            path = Path(entry.path)
            if not path.is_file():
                raise ManifestError("manifest_file_missing", {
                    "export_id": entry.export_id, "path": entry.path})
            for line_no, doc in _stream_jsonl(path, export_id=entry.export_id):
                parsed = parse_export_row(
                    doc, export_id=entry.export_id, line_no=line_no)
                try:
                    workspace.insert_row(parsed, input_order=input_order)
                except sqlite3.IntegrityError:
                    # The primary key is the joint identity, so this is a
                    # duplicate sample_id inside one export.
                    raise RowError("duplicate_joint_identity", {
                        "export_id": entry.export_id,
                        "sample_id": parsed.sample_id,
                        "path": entry.path, "line_no": line_no})
                input_order += 1

    # ------------------------------------------------------------ classify
    def _classify_rows(self, workspace: DistributionWorkspace) -> None:
        """Window/label classification, streamed in joint-id pages."""

        conn = workspace.conn
        last_joint = None
        limit = self.config.page_limit
        while True:
            if last_joint is None:
                rows = conn.execute(
                    "SELECT joint_id, target_start, target_end, label_status "
                    "FROM rows ORDER BY joint_id LIMIT ?",
                    (limit,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT joint_id, target_start, target_end, label_status "
                    "FROM rows WHERE joint_id > ? ORDER BY joint_id LIMIT ?",
                    (last_joint, limit)).fetchall()
            if not rows:
                break
            for row in rows:
                reasons = []
                if row["label_status"] != "ok":
                    reasons.append(REASON_LABEL_UNAVAILABLE)
                window_name, relation = self.plan.classify(
                    row["target_start"], row["target_end"])
                if relation == "crosses_windows":
                    reasons.append(REASON_CROSSES_WINDOWS)
                elif relation == "outside_windows":
                    reasons.append(REASON_OUTSIDE_WINDOWS)
                candidate = int(
                    row["label_status"] == "ok"
                    and relation == "in_window")
                conn.execute(
                    """
                    UPDATE rows
                       SET candidate = ?, window_name = ?,
                           window_relation = ?, reasons_json = ?
                     WHERE joint_id = ?
                    """,
                    (candidate, window_name, relation,
                     json.dumps(reasons, ensure_ascii=False),
                     row["joint_id"]))
            workspace.commit()
            last_joint = rows[-1]["joint_id"]
            if len(rows) < limit:
                break

    # ------------------------------------------------------------- decisions
    def _apply_group_decisions(self, workspace: DistributionWorkspace) -> dict:
        stats = {"groups_total": 0, "groups_single_window": 0,
                 "groups_multiple_windows": 0, "groups_without_candidates": 0}
        conn = workspace.conn
        for group in workspace.group_window_counts():
            stats["groups_total"] += 1
            windows_present = [name for name in ("train", "validation",
                                                 "holdout")
                               if group[f"n_{name}"] > 0]
            if group["n_cand"] == 0:
                stats["groups_without_candidates"] += 1
                # No shard data; every row keeps its direct isolation reasons.
                conn.execute(
                    "UPDATE rows SET destination = ? WHERE group_id = ?",
                    (DECISION_ISOLATED, group["group_id"]))
                continue
            if len(windows_present) == 1:
                stats["groups_single_window"] += 1
                window = windows_present[0]
                # Candidates go to the shard; the group's non-candidates stay
                # isolated but remain members of the same group.
                conn.execute(
                    "UPDATE rows SET destination = ? WHERE group_id = ? "
                    "AND candidate = 1",
                    (window, group["group_id"]))
                conn.execute(
                    "UPDATE rows SET destination = ? WHERE group_id = ? "
                    "AND candidate = 0",
                    (DECISION_ISOLATED, group["group_id"]))
            else:
                stats["groups_multiple_windows"] += 1
                # No majority/first/priority rule: every candidate is isolated
                # and gains the group-spanning reason (in addition to any
                # direct reason it already carries).
                conn.execute(
                    "UPDATE rows SET destination = ? WHERE group_id = ? "
                    "AND candidate = 0",
                    (DECISION_ISOLATED, group["group_id"]))
                last_joint = None
                while True:
                    if last_joint is None:
                        candidates = conn.execute(
                            "SELECT joint_id, reasons_json FROM rows "
                            "WHERE group_id = ? AND candidate = 1 "
                            "ORDER BY joint_id LIMIT ?",
                            (group["group_id"], self.config.page_limit)
                            ).fetchall()
                    else:
                        candidates = conn.execute(
                            "SELECT joint_id, reasons_json FROM rows "
                            "WHERE group_id = ? AND candidate = 1 "
                            "AND joint_id > ? ORDER BY joint_id LIMIT ?",
                            (group["group_id"], last_joint,
                             self.config.page_limit)).fetchall()
                    if not candidates:
                        break
                    for cand in candidates:
                        reasons = json.loads(cand["reasons_json"])
                        if REASON_GROUP_SPANS_MULTIPLE_WINDOWS not in reasons:
                            reasons.append(
                                REASON_GROUP_SPANS_MULTIPLE_WINDOWS)
                        conn.execute(
                            "UPDATE rows SET destination = ?, reasons_json = ? "
                            "WHERE joint_id = ?",
                            (DECISION_ISOLATED,
                             json.dumps(reasons, ensure_ascii=False),
                             cand["joint_id"]))
                        last_joint = cand["joint_id"]
                    if len(candidates) < self.config.page_limit:
                        break
            workspace.commit()
        # Tally from the final destinations; reasons may be multi-valued.
        for decision in DECISIONS:
            count = int(conn.execute(
                "SELECT COUNT(*) c FROM rows WHERE destination = ?",
                (decision,)).fetchone()["c"])
            if decision == DECISION_ISOLATED:
                self.isolated_count = count
            else:
                self.shard_counts[decision] = count
        for reason in ALL_ISOLATION_REASONS:
            self.reason_counts[reason] = int(conn.execute(
                "SELECT COUNT(*) c FROM rows WHERE EXISTS ("
                "SELECT 1 FROM json_each(rows.reasons_json) WHERE value = ?)",
                (reason,)).fetchone()["c"])
        workspace.commit()
        return stats

    # ----------------------------------------------------------------- emit
    def _emit(self, workspace: DistributionWorkspace, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        conn = workspace.conn
        targets = {
            DECISION_TRAIN: out_dir / "train.jsonl",
            DECISION_VALIDATION: out_dir / "validation.jsonl",
            DECISION_HOLDOUT: out_dir / "holdout.jsonl",
            DECISION_ISOLATED: out_dir / "isolated.jsonl",
        }
        tmp_paths = [str(p) + ".tmp" for p in targets.values()]
        tmp_paths.append(str(out_dir / "decisions.jsonl.tmp"))
        tmp_paths.append(str(out_dir / "summary.json.tmp"))
        for tmp in tmp_paths:
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
        handles = {d: open(str(targets[d]) + ".tmp", "w", encoding="utf-8")
                   for d in DECISIONS}
        decisions_handle = open(out_dir / "decisions.jsonl.tmp", "w",
                                encoding="utf-8")
        try:
            # Joint-id UTF-8 byte order makes every output independent of
            # manifest order, file line order and paging/batch sizes.  A
            # default-size streaming cursor means the whole row set never has
            # to fit in Python memory; each row is written and released.
            cursor = conn.execute(
                "SELECT * FROM rows ORDER BY joint_id")
            for row in cursor:
                envelope = json.dumps(
                    {"export_id": row["export_id"],
                     "row": json.loads(row["raw_json"])},
                    ensure_ascii=False)
                handles[row["destination"]].write(envelope + "\n")
                group_export, group_sample = row["group_id"].split("\x1f", 1)
                decision_doc = {
                    "export_id": row["export_id"],
                    "sample_id": row["sample_id"],
                    "group": {"export_id": group_export,
                              "sample_id": group_sample},
                    "decision": row["destination"],
                    "window": row["window_name"],
                    "window_relation": row["window_relation"],
                    "label_status": row["label_status"],
                    "reasons": json.loads(row["reasons_json"]),
                }
                decisions_handle.write(
                    json.dumps(decision_doc, ensure_ascii=False) + "\n")
        finally:
            for handle in handles.values():
                handle.close()
            decisions_handle.close()

        summary = {
            "schema_version": SCHEMA_VERSION,
            "gap_seconds": self.plan.gap_seconds,
            "windows": {name: self.plan.windows[name].to_dict()
                        for name in ("train", "validation", "holdout")},
            "exports": [{"export_id": e.export_id, "path": e.path}
                        for e in self.entries],
            "row_count": workspace.row_count(),
            "shard_counts": dict(self.shard_counts),
            "isolated_count": self.isolated_count,
            "reason_counts": dict(self.reason_counts),
            "relation_scale": {
                "rows": workspace.row_count(),
                "successful_feature_refs": workspace.ref_count(),
                "message_edges": workspace.built_message_edges,
                "observation_edges": workspace.built_observation_edges,
                "forest_edges": workspace.forest_edges},
        }
        # group_stats are computed by the caller before emit; re-derive here
        # from stored destinations for a single source of truth.
        summary["group_stats"] = self._group_stats_from_db(conn)
        (out_dir / "summary.json.tmp").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8")
        for target in list(targets.values()) + [
                out_dir / "decisions.jsonl", out_dir / "summary.json"]:
            os.replace(str(target) + ".tmp", target)
        return {d: str(targets[d]) for d in DECISIONS} | {
            "decisions": str(out_dir / "decisions.jsonl"),
            "summary": str(out_dir / "summary.json")}

    @staticmethod
    def _group_stats_from_db(conn) -> dict:
        total = int(conn.execute(
            "SELECT COUNT(DISTINCT group_id) c FROM rows").fetchone()["c"])
        multi = int(conn.execute(
            "SELECT COUNT(*) c FROM rows WHERE EXISTS ("
            "SELECT 1 FROM json_each(rows.reasons_json) "
            "WHERE value = ?)",
            (REASON_GROUP_SPANS_MULTIPLE_WINDOWS,)).fetchone()["c"])
        multi_groups = int(conn.execute(
            "SELECT COUNT(DISTINCT group_id) c FROM rows WHERE EXISTS ("
            "SELECT 1 FROM json_each(rows.reasons_json) "
            "WHERE value = ?)",
            (REASON_GROUP_SPANS_MULTIPLE_WINDOWS,)).fetchone()["c"])
        no_candidate_groups = int(conn.execute(
            "SELECT COUNT(DISTINCT group_id) c FROM rows "
            "WHERE group_id NOT IN (SELECT DISTINCT group_id FROM rows "
            "WHERE candidate = 1)").fetchone()["c"])
        return {
            "groups_total": total,
            "groups_single_window": total - multi_groups - no_candidate_groups,
            "groups_multiple_windows": multi_groups,
            "groups_without_candidates": no_candidate_groups,
            "rows_isolated_by_group_spanning": multi,
        }


def run_distribution(manifest_path: str | Path, plan_path: str | Path,
                     out_dir: str | Path,
                     config: DistributionConfig | None = None
                     ) -> DistributionReport:
    """Validate inputs, build the index and publish every delivery artifact."""

    config = config or DistributionConfig()
    entries = load_export_manifest(manifest_path)
    plan = load_delivery_plan(plan_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    workspace_root = config.work_dir
    created_root = False
    if workspace_root is None:
        workspace_root = tempfile.mkdtemp(prefix="cube-distribute-")
        created_root = True
    workspace_root = Path(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    db_path = workspace_root / "distribution.sqlite"
    for suffix in ("", "-wal", "-shm"):
        try:
            (Path(str(db_path) + suffix)).unlink()
        except FileNotFoundError:
            pass

    ws_config = WorkspaceConfig(
        sqlite_cache_kib=config.sqlite_cache_kib,
        commit_every=config.commit_every, page_limit=config.page_limit)
    workspace = DistributionWorkspace(db_path, config=ws_config)
    try:
        report = Distributor(plan, entries, config).run(workspace, out_dir)
        # Merge the WAL into the main database and switch back to rollback
        # journal mode, so the published index is one self-contained file with
        # no -wal/-shm sidecars that explanations could miss or outlive.
        workspace.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        workspace.conn.execute("PRAGMA journal_mode = DELETE")
        index_target = out_dir / INDEX_NAME
        index_manifest = out_dir / INDEX_MANIFEST_NAME
        tmp_index = str(index_target) + ".tmp"
        try:
            os.remove(tmp_index)
        except FileNotFoundError:
            pass
        shutil.copyfile(db_path, tmp_index)
        index_manifest_doc = {
            "schema_version": SCHEMA_VERSION,
            "gap_seconds": plan.gap_seconds,
            "windows": {name: plan.windows[name].to_dict()
                        for name in ("train", "validation", "holdout")},
            "exports": [{"export_id": e.export_id, "path": e.path}
                        for e in entries],
            "row_count": report.row_count,
            "group_count": report.group_count,
            "relation_scale": report.relation_scale,
            "tables": ["rows", "message_refs", "edges", "run_meta"],
        }
        Path(str(index_manifest) + ".tmp").write_text(
            json.dumps(index_manifest_doc, indent=2, ensure_ascii=False),
            encoding="utf-8")
        os.replace(tmp_index, index_target)
        os.replace(str(index_manifest) + ".tmp", index_manifest)
        return report
    except BaseException:
        # Never leave partial delivery artifacts behind; remove an output
        # directory this run created if it holds nothing but failed outputs.
        for name in ("train.jsonl", "validation.jsonl", "holdout.jsonl",
                     "isolated.jsonl", "decisions.jsonl", "summary.json",
                     INDEX_NAME, INDEX_MANIFEST_NAME):
            for suffix in ("", ".tmp"):
                try:
                    (out_dir / f"{name}{suffix}").unlink()
                except FileNotFoundError:
                    pass
        try:
            # Only remove when empty so a pre-existing directory with other
            # content is never touched.
            out_dir.rmdir()
        except OSError:
            pass
        raise
    finally:
        if config.keep_workspace:
            workspace.close()
        else:
            workspace.destroy()
            if created_root:
                shutil.rmtree(workspace_root, ignore_errors=True)
