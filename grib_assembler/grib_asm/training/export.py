"""Point-in-time training export orchestration.

The pipeline is deliberately split into disk-backed stages so that neither the
sample manifest nor the three streams ever have to fit in memory:

1. **ingest** - stream every JSONL line through the strict parsers into a
   temporary SQLite workspace.  Same-instant conflicts and all clock/interval
   invariants fail here, before any output file is created.
2. **resolve** - one streaming pass over the samples.  For every sample and
   every recipe column, adjudicate station (feature clock), publish (feature
   clock) and field/point eligibility, applying the required
   station -> publish -> field/point status precedence.  Eligible point reads
   become rows in ``point_requests``; labels are resolved against the separate
   label clock and persisted alongside the partially built rows.
3. **fetch** - group the scheduled requests by distinct values chunk on
   disk, decode each chunk once, extract every requested grid point and write
   the results back to the workspace.
4. **emit** - re-read the rows in ``sample_id`` UTF-8 byte order and write the
   JSONL and the count summary atomically (temp files + ``os.replace``).

Hard input-evidence errors abort before stage 4, so a failed run leaves no
file a caller could mistake for this run's successful output.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..cube.coords import slot_key, variable_key
from .cubeaccess import CubePointReader
from .errors import CubeEvidenceError, TrainingExportError
from .model import (
    Recipe, Sample, epoch_iso, load_recipe,
)
from .workspace import TrainingWorkspace

# Feature cell statuses (stable strings written to the JSONL and summary).
STATION_NOT_REGISTERED = "station_not_registered"
STATION_UNMAPPED = "station_unmapped"
FORECAST_NOT_PUBLISHED = "forecast_not_published"
FORECAST_WITHDRAWN = "forecast_withdrawn"
FIELD_MISSING = "field_missing"
POINT_MISSING = "point_missing"
OK = "ok"

FEATURE_STATUSES = (
    STATION_NOT_REGISTERED, STATION_UNMAPPED, FORECAST_NOT_PUBLISHED,
    FORECAST_WITHDRAWN, FIELD_MISSING, POINT_MISSING, OK,
)

# Label statuses.
LABEL_NOT_RECEIVED = "label_not_received"
LABEL_RETRACTED = "label_retracted"
LABEL_UNIT_MISMATCH = "label_unit_mismatch"
LABEL_OK = "ok"
LABEL_STATUSES = (
    LABEL_NOT_RECEIVED, LABEL_RETRACTED, LABEL_UNIT_MISMATCH, LABEL_OK)


@dataclass
class ExportConfig:
    """Tunables bounding the on-disk workspace and buffering."""

    work_dir: str | None = None
    keep_workspace: bool = False
    sqlite_cache_kib: int = 64 * 1024
    commit_every: int = 1000
    # Maximum number of point requests buffered when draining one chunk; the
    # decoded full-grid chunk plus this many request rows bounds fetch memory.
    point_batch: int = 4096


@dataclass
class ExportReport:
    output_jsonl: str
    output_summary: str
    sample_count: int
    value_block_reads: int
    value_block_keys: list[str]
    workspace: str
    feature_status_counts: dict = field(default_factory=dict)
    label_status_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "output_jsonl": self.output_jsonl,
            "output_summary": self.output_summary,
            "sample_count": self.sample_count,
            "value_block_reads": self.value_block_reads,
            "value_block_keys": self.value_block_keys,
            "workspace": self.workspace,
            "feature_status_counts": self.feature_status_counts,
            "label_status_counts": self.label_status_counts,
        }


def _stream_jsonl(path: str | Path):
    """Yield parsed JSON objects line by line without loading the file."""

    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise TrainingExportError("stream_record_invalid", {
                    "path": str(path), "line_no": line_no,
                    "reason": f"invalid JSON: {exc.msg}"})


class TrainingExporter:
    """Holds the assembled collaborators for one export run."""

    def __init__(self, store_path: str | Path, recipe: Recipe,
                 config: ExportConfig):
        self.recipe = recipe
        self.config = config
        self.reader = CubePointReader(store_path)
        self.feature_counts = {s: 0 for s in FEATURE_STATUSES}
        self.label_counts = {s: 0 for s in LABEL_STATUSES}
        # Rows staged on disk keyed by sample_id (one JSON doc per sample).
        self._row_paths: dict[str, str] = {}

    # ------------------------------------------------------------ stage 1+2
    def run(self, samples_path, publish_path, station_path, obs_path,
            workspace: TrainingWorkspace, out_jsonl: str,
            out_summary: str) -> ExportReport:
        self._ingest_streams(workspace, samples_path, publish_path,
                             station_path, obs_path)
        workspace.commit()
        rows_dir = Path(workspace.path).parent / "rows"
        rows_dir.mkdir(parents=True, exist_ok=True)
        self._resolve_all(workspace, rows_dir)
        self._fetch_all(workspace)
        report = self._emit(workspace, rows_dir, out_jsonl, out_summary)
        return report

    def _ingest_streams(self, workspace: TrainingWorkspace, samples_path,
                        publish_path, station_path, obs_path) -> None:
        for doc in _stream_jsonl(publish_path):
            workspace.add_publish(doc)
        for doc in _stream_jsonl(station_path):
            workspace.add_station(doc)
        for doc in _stream_jsonl(obs_path):
            workspace.add_observation(doc)
        # Samples last: their per-row validation needs the recipe already set
        # and benefits from the streams being fully staged.
        for doc in _stream_jsonl(samples_path):
            workspace.add_sample(doc)

    # ------------------------------------------------------------- resolve
    def _resolve_all(self, workspace: TrainingWorkspace,
                     rows_dir: Path) -> None:
        for srow in workspace.iter_samples():
            sample = Sample(
                sample_id=srow["sample_id"], station_id=srow["station_id"],
                metric=srow["metric"], target_start_epoch=srow["target_start"],
                target_end_epoch=srow["target_end"],
                decision_epoch=srow["decision_at"], order=srow["input_order"])
            station = self._resolve_station(workspace, sample)
            features = []
            for col_index, column in enumerate(self.recipe.features):
                features.append(self._resolve_feature(
                    workspace, sample, column, col_index, station))
            label = self._resolve_label(workspace, sample)
            row = {
                "sample_id": sample.sample_id,
                "station_id": sample.station_id,
                "metric": sample.metric,
                "target_interval": [sample.target_start, sample.target_end],
                "decision_at": sample.decision_at,
                "features": features,
                "label": label,
            }
            row_path = rows_dir / f"{len(self._row_paths):012d}.json"
            row_path.write_text(
                json.dumps(row, ensure_ascii=False), encoding="utf-8")
            # Map sample_id -> the row file; byte-order traversal happens at
            # emit time.
            self._row_paths[sample.sample_id] = str(row_path)

    def _resolve_station(self, workspace: TrainingWorkspace, sample: Sample):
        """Resolve the single station location used by every feature column."""

        winner = workspace.station_winner(
            sample.station_id, sample.target_start_epoch,
            sample.decision_epoch)
        if winner is None:
            return {"status": STATION_NOT_REGISTERED, "evidence": None}
        doc = json.loads(winner["doc_json"])
        record_identity = {
            "record_id": winner["record_id"],
            "visible_at": epoch_iso(winner["visible_at"]),
            "seq": winner["seq"],
            "business_key": json.loads(winner["business_key"]),
        }
        if doc["action"] == "unmap":
            # The newest covering record cancels the mapping for this target.
            return {"status": STATION_UNMAPPED,
                    "evidence": {"winning_record": record_identity}}
        # A winning map must point at this cube's grid and legal indices;
        # anything else is contradictory input evidence, not a missing value.
        self.reader.validate_grid_point(
            doc["grid_digest"], int(doc["lat_index"]), int(doc["lon_index"]))
        return {
            "status": "mapped",
            "evidence": {"winning_record": record_identity},
            "grid_digest": doc["grid_digest"],
            "lat_index": int(doc["lat_index"]),
            "lon_index": int(doc["lon_index"]),
        }

    def _resolve_feature(self, workspace: TrainingWorkspace, sample: Sample,
                         column, col_index: int, station: dict) -> dict:
        """Adjudicate one feature cell with station->publish->field precedence."""

        origin_epoch = column.derive_origin_epoch(sample.target_start_epoch)
        origin_iso = epoch_iso(origin_epoch)
        feature_window = [
            epoch_iso(sample.target_start_epoch + column.offset_start),
            epoch_iso(sample.target_start_epoch + column.offset_end)]
        base = {
            "column": column.name,
            "origin": origin_iso,
            "slot": column.slot.to_dict(),
            "feature_window": feature_window,
        }

        # 1) station gate --------------------------------------------------
        if station["status"] == STATION_NOT_REGISTERED:
            self.feature_counts[STATION_NOT_REGISTERED] += 1
            return {**base, "status": STATION_NOT_REGISTERED, "value": None,
                    "evidence": {"station": station}}
        if station["status"] == STATION_UNMAPPED:
            self.feature_counts[STATION_UNMAPPED] += 1
            return {**base, "status": STATION_UNMAPPED, "value": None,
                    "evidence": {"station": station}}

        # 2) publish gate --------------------------------------------------
        publish = workspace.publish_winner(origin_epoch,
                                           sample.decision_epoch)
        if publish is None:
            self.feature_counts[FORECAST_NOT_PUBLISHED] += 1
            return {**base, "status": FORECAST_NOT_PUBLISHED, "value": None,
                    "evidence": {"station": station, "publish": None}}
        pub_doc = json.loads(publish["doc_json"])
        pub_identity = {
            "record_id": publish["record_id"],
            "visible_at": epoch_iso(publish["visible_at"]),
            "seq": publish["seq"],
            "business_key": {"stream": "publish",
                             "origin": pub_doc["origin"]},
        }
        if pub_doc["action"] == "withdraw":
            self.feature_counts[FORECAST_WITHDRAWN] += 1
            return {**base, "status": FORECAST_WITHDRAWN, "value": None,
                    "evidence": {"station": station,
                                 "publish": {"winning_record": pub_identity,
                                             "action": "withdraw"}}}

        # A visible publish must reference a cycle this cube actually
        # committed, with the exact package digest the cube recorded.
        if not self.reader.is_origin_committed(origin_iso):
            raise CubeEvidenceError("publish_origin_not_committed", {
                "column": column.name, "sample_id": sample.sample_id,
                "origin": origin_iso, "publish_record": pub_identity})
        cube_digest = self.reader.source_digest_of(origin_iso)
        claimed_digest = pub_doc["package_summary"]["package_digest"]
        if cube_digest != claimed_digest:
            raise CubeEvidenceError("publish_digest_mismatch", {
                "column": column.name, "sample_id": sample.sample_id,
                "origin": origin_iso, "publish_record": pub_identity,
                "claimed_package_digest": claimed_digest,
                "cube_package_digest": cube_digest})

        # 3) field / point gate -------------------------------------------
        coords = self.reader.resolve_coordinates(column, origin_iso)
        if coords is None:
            # The variable identity is absent from this cube entirely (other
            # variables may well exist); the requested identity stays in the
            # evidence so the gap is fully explainable.
            self.feature_counts[FIELD_MISSING] += 1
            return {**base, "status": FIELD_MISSING, "value": None,
                    "evidence": {
                        "station": station,
                        "publish": {"winning_record": pub_identity,
                                    "action": "publish",
                                    "package_digest": claimed_digest},
                        "field": {
                            "variable_key": variable_key(column.variable),
                            "origin": origin_iso,
                            "slot_key": slot_key(column.slot),
                            "registry_indices": None,
                            "present": False,
                            "message_digest": "",
                            "package_digest": "",
                            "raw_unit": "",
                            "point_index": None}}}
        cell_meta = self.reader.cell_metadata(coords)
        if not cell_meta["present"]:
            # The field exists as a registered variable but this cell was
            # never written (explicit missing slab).
            self.feature_counts[FIELD_MISSING] += 1
            return {**base, "status": FIELD_MISSING, "value": None,
                    "evidence": self._field_evidence(
                        station, pub_identity, claimed_digest, coords,
                        cell_meta, sample, column, origin_iso,
                        point_index=None)}
        # The committed cell's provenance must match the published package.
        if cell_meta["package_digest"] != claimed_digest:
            raise CubeEvidenceError("cell_provenance_conflict", {
                "column": column.name, "sample_id": sample.sample_id,
                "origin": origin_iso,
                "cell_package_digest": cell_meta["package_digest"],
                "publish_package_digest": claimed_digest,
                "message_digest": cell_meta["message_digest"]})

        # All gates passed: schedule the actual value-block read.  Nothing
        # above decoded a values chunk.
        workspace.add_point_request(
            sample_id=sample.sample_id, column_index=col_index,
            var_key=coords.var_key, origin_index=coords.origin_index,
            level_index=coords.level_index, member_index=coords.member_index,
            slot_index=coords.slot_index,
            lat_index=station["lat_index"], lon_index=station["lon_index"],
            origin_epoch=origin_epoch)
        return {
            **base, "status": "__pending__", "value": None,
            "evidence": self._field_evidence(
                station, pub_identity, claimed_digest, coords, cell_meta,
                sample, column, origin_iso,
                point_index={"lat_index": station["lat_index"],
                             "lon_index": station["lon_index"]}),
            "_workspace_key": [sample.sample_id, col_index],
        }

    def _field_evidence(self, station, pub_identity, claimed_digest, coords,
                        cell_meta, sample, column, origin_iso,
                        point_index) -> dict:
        return {
            "station": station,
            "publish": {"winning_record": pub_identity, "action": "publish",
                        "package_digest": claimed_digest},
            "field": {
                "variable_key": coords.var_key if coords else None,
                "origin": origin_iso,
                "slot_key": slot_key(column.slot),
                "registry_indices": None if coords is None else {
                    "origin_index": coords.origin_index,
                    "level_index": coords.level_index,
                    "member_index": coords.member_index,
                    "slot_index": coords.slot_index},
                "present": cell_meta["present"],
                "message_digest": cell_meta["message_digest"],
                "package_digest": cell_meta["package_digest"],
                "raw_unit": cell_meta["raw_unit"],
                "point_index": point_index,
            },
        }

    # --------------------------------------------------------------- label
    def _resolve_label(self, workspace: TrainingWorkspace, sample: Sample):
        winner = workspace.observation_winner(
            sample.station_id, sample.metric,
            sample.target_start_epoch, sample.target_end_epoch,
            self.recipe.label_cutoff_epoch)
        if winner is None:
            self.label_counts[LABEL_NOT_RECEIVED] += 1
            return {"status": LABEL_NOT_RECEIVED, "value": None, "unit": None,
                    "winning_record": None, "cutoff": self.recipe.label_cutoff}
        doc = json.loads(winner["doc_json"])
        identity = {
            "record_id": winner["record_id"],
            "visible_at": epoch_iso(winner["visible_at"]),
            "seq": winner["seq"],
            "business_key": {
                "stream": "observation",
                "station_id": doc["station_id"],
                "metric": doc["metric"],
                "target_start": doc["target_start"],
                "target_end": doc["target_end"]},
        }
        if doc["action"] == "retract":
            self.label_counts[LABEL_RETRACTED] += 1
            return {"status": LABEL_RETRACTED, "value": None, "unit": None,
                    "winning_record": identity,
                    "cutoff": self.recipe.label_cutoff}
        expected_unit = self.recipe.metric_units[sample.metric]
        if doc["unit"] != expected_unit:
            # A different unit must never silently stand in for the expected
            # one; this is a label status, not a hard export error.
            self.label_counts[LABEL_UNIT_MISMATCH] += 1
            return {"status": LABEL_UNIT_MISMATCH, "value": None,
                    "unit": doc["unit"], "expected_unit": expected_unit,
                    "winning_record": identity,
                    "cutoff": self.recipe.label_cutoff}
        self.label_counts[LABEL_OK] += 1
        return {"status": LABEL_OK, "value": float(doc["value"]),
                "unit": doc["unit"], "winning_record": identity,
                "cutoff": self.recipe.label_cutoff}

    # --------------------------------------------------------------- fetch
    def _fetch_all(self, workspace: TrainingWorkspace) -> None:
        """Decode each distinct values chunk once and fan points out."""

        for chunk in workspace.distinct_chunks():
            coords = self._coords_from_row(chunk)
            self.reader.clear_chunk_cache()
            grid = self.reader.read_chunk_values(coords)
            cell_meta = self.reader.cell_metadata(coords)
            processed = 0
            for req in workspace.requests_for_chunk(
                    chunk["var_key"], chunk["origin_index"],
                    chunk["level_index"], chunk["member_index"],
                    chunk["slot_index"]):
                lat, lon = req["lat_index"], req["lon_index"]
                raw = grid[lat, lon]
                present = not math.isnan(raw)
                workspace.store_result(
                    sample_id=req["sample_id"],
                    column_index=req["column_index"],
                    value=None if not present else float(raw),
                    present=1 if present else 0,
                    message_digest=cell_meta["message_digest"],
                    package_digest=cell_meta["package_digest"],
                    raw_unit=cell_meta["raw_unit"])
                processed += 1
                if processed >= self.config.point_batch:
                    # Commit periodically; the chunk grid stays the only big
                    # resident object regardless of request count.
                    workspace.commit()
                    processed = 0
            workspace.commit()

    def _coords_from_row(self, row):
        from .cubeaccess import ChunkCoordinates
        return ChunkCoordinates(
            var_key=row["var_key"], origin_index=row["origin_index"],
            level_index=row["level_index"], member_index=row["member_index"],
            slot_index=row["slot_index"])

    # ---------------------------------------------------------------- emit
    def _emit(self, workspace: TrainingWorkspace, rows_dir: Path,
              out_jsonl: str, out_summary: str) -> ExportReport:
        out_jsonl = str(out_jsonl)
        out_summary = str(out_summary)
        tmp_jsonl = out_jsonl + ".tmp"
        tmp_summary = out_summary + ".tmp"
        # Never append onto a stale temp file from an interrupted attempt.
        for tmp in (tmp_jsonl, tmp_summary):
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
        # UTF-8 byte order over sample_id.
        ordered_ids = sorted(self._row_paths,
                             key=lambda sid: sid.encode("utf-8"))
        # Create the JSONL up front so an empty sample manifest still yields
        # a valid (empty) output file rather than a missing temp path.
        open(tmp_jsonl, "w", encoding="utf-8").close()
        for sid in ordered_ids:
            row_path = self._row_paths[sid]
            row = json.loads(Path(row_path).read_text(encoding="utf-8"))
            self._finalize_pending(workspace, row)
            with open(tmp_jsonl, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = {
            "recipe_id": self.recipe.recipe_id,
            "label_cutoff": self.recipe.label_cutoff,
            "sample_count": len(ordered_ids),
            "feature_columns": [c.name for c in self.recipe.features],
            "feature_status_counts": dict(self.feature_counts),
            "label_status_counts": dict(self.label_counts),
            "value_block_reads": self.reader.value_block_read_count(),
            "value_block_keys": self.reader.value_block_read_keys(),
        }
        Path(tmp_summary).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8")
        os.replace(tmp_jsonl, out_jsonl)
        os.replace(tmp_summary, out_summary)
        return ExportReport(
            output_jsonl=out_jsonl, output_summary=out_summary,
            sample_count=len(ordered_ids),
            value_block_reads=self.reader.value_block_read_count(),
            value_block_keys=self.reader.value_block_read_keys(),
            workspace=str(workspace.path),
            feature_status_counts=dict(self.feature_counts),
            label_status_counts=dict(self.label_counts))

    def _finalize_pending(self, workspace: TrainingWorkspace, row: dict):
        for feature in row["features"]:
            if feature["status"] != "__pending__":
                continue
            sample_id, col_index = feature.pop("_workspace_key")
            result = workspace.result_for(sample_id, col_index)
            if result is None:
                raise TrainingExportError("internal_pending_without_result", {
                    "sample_id": sample_id, "column": feature["column"]})
            if result["present"] == 1:
                feature["status"] = OK
                feature["value"] = result["value"]
                self.feature_counts[OK] += 1
            else:
                # The cell was present/provenanced but the grid point itself
                # is an explicit missing value.
                feature["status"] = POINT_MISSING
                feature["value"] = None
                self.feature_counts[POINT_MISSING] += 1
            feature["evidence"]["field"]["point_result"] = {
                "present": int(result["present"]),
                "message_digest": result["message_digest"],
                "package_digest": result["package_digest"],
                "raw_unit": result["raw_unit"]}


def export_training_table(store_path: str | Path, recipe_path: str | Path,
                          samples_path: str | Path, publish_path: str | Path,
                          station_path: str | Path, observation_path: str | Path,
                          out_jsonl: str | Path, out_summary: str | Path,
                          config: ExportConfig | None = None
                          ) -> ExportReport:
    """Run the full export; on any error no success output is published."""

    config = config or ExportConfig()
    recipe = load_recipe(str(recipe_path))
    # Pre-create output temp paths' parent and make sure stale temp files from
    # an interrupted earlier attempt cannot be mistaken for results.
    out_jsonl, out_summary = str(out_jsonl), str(out_summary)
    for target in (out_jsonl, out_summary,
                   out_jsonl + ".tmp", out_summary + ".tmp"):
        try:
            os.remove(target)
        except FileNotFoundError:
            pass

    workspace_root = config.work_dir
    created_root = False
    if workspace_root is None:
        workspace_root = tempfile.mkdtemp(prefix="cube-training-")
        created_root = True
    workspace_root = Path(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    db_path = workspace_root / "training.sqlite"

    workspace = TrainingWorkspace(
        db_path, sqlite_cache_kib=config.sqlite_cache_kib,
        commit_every=config.commit_every)
    workspace.set_recipe(recipe)
    exporter = TrainingExporter(store_path, recipe, config)
    try:
        report = exporter.run(
            samples_path, publish_path, station_path, observation_path,
            workspace, out_jsonl, out_summary)
    except BaseException:
        # Never leave temp output that could be read as a successful result.
        for tmp in (out_jsonl + ".tmp", out_summary + ".tmp"):
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
        raise
    finally:
        if config.keep_workspace:
            workspace.close()
        else:
            workspace.destroy()
            import shutil
            if created_root:
                # We created this directory: remove it and the staged rows.
                shutil.rmtree(workspace_root, ignore_errors=True)
            else:
                # Caller-supplied directory: remove only the artifacts this
                # run created (SQLite file + row staging), keep the directory.
                shutil.rmtree(workspace_root / "rows", ignore_errors=True)
    return report
