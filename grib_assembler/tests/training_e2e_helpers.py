"""Shared builders for point-in-time export tests against a real cube.

Every helper here produces a genuinely imported Zarr cube (via the project's
own catalog -> assembler -> importer) and hand-written JSONL streams.  The
expected values used in assertions are constants declared by the tests, never
read back from the exporter.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from grib_asm.timeutil import TimeSlot
from grib_asm.training.export import ExportConfig, export_training_table
from training_fixtures import (
    NI, NJ, POINT_A, POINT_B, build_cube,
    encode_stat_message, feature_column_doc, publish_record,
    sample_record, station_record, write_jsonl,
)

UTC = timezone.utc
DAY = datetime(2026, 1, 1, tzinfo=UTC)
SLOT_3_6 = TimeSlot(3 * 3600, 6 * 3600, 1)
SLOT_0_6 = TimeSlot(0, 6 * 3600, 1)

# Hand-authored values (independent of exporter logic).
VALUE_A = 21.5
VALUE_B = 42.25


def at(hour: int, minute: int = 0) -> str:
    return (DAY + timedelta(hours=hour, minutes=minute)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def write_world(tmp_path: Path, *, params=(0, 7, 9),
                missing_param=9, absent_param=11,
                value_a=VALUE_A, value_b=VALUE_B):
    """Build one origin (06:00) cube with several variables and two slots."""

    origin = DAY.replace(hour=6)
    origin_iso = at(6)
    messages = []
    grids = {}
    for pnum in params:
        for slot, sh, eh in ((SLOT_3_6, 3, 6), (SLOT_0_6, 0, 6)):
            grid = np.full((NJ, NI), float(100 + pnum + eh), dtype="float64")
            if pnum == 0 and slot is SLOT_3_6:
                grid[POINT_A] = value_a
                grid[POINT_B] = value_b
            if pnum == missing_param and slot is SLOT_3_6:
                grid[POINT_B] = np.nan  # explicit bitmap-missing point
            packing = "grid_ieee" if np.isnan(grid).any() else "grid_simple"
            bits = 32 if packing == "grid_ieee" else 24
            raw = encode_stat_message(
                origin_dt=origin, start_hour=sh, end_hour=eh, values=grid,
                parameter_number=pnum, packing=packing, bits_per_value=bits)
            messages.append((raw, slot, pnum))
            grids[(pnum, slot is SLOT_3_6)] = grid
    store, grid_dig, pkg = build_cube(
        tmp_path,
        [{"collection_id": "c06", "origin": origin_iso,
          "messages": messages}])
    return {
        "store": str(store), "grid_digest": grid_dig,
        "package_digest": pkg[origin_iso], "origin": origin_iso,
        "missing_param": missing_param, "absent_param": absent_param,
        "grids": grids,
    }


def write_two_origin_world(tmp_path: Path):
    """Two origins (00:00 and 06:00), each its own package digest.

    Used to prove a publish whose digest names the wrong committed package is
    an evidence error rather than a silently trusted value.
    """

    collections = []
    for hour in (0, 6):
        origin = DAY.replace(hour=hour)
        origin_iso = at(hour)
        grid = np.full((NJ, NI), float(hour + 1), dtype="float64")
        grid[POINT_A] = VALUE_A
        raw = encode_stat_message(
            origin_dt=origin, start_hour=3, end_hour=6, values=grid)
        collections.append({"collection_id": f"c{hour:02d}",
                            "origin": origin_iso,
                            "messages": [(raw, SLOT_3_6, 0)]})
    store, grid_dig, pkg = build_cube(tmp_path, collections)
    return {
        "store": str(store), "grid_digest": grid_dig,
        "package_by_origin": pkg,
        "origin_00": at(0), "origin_06": at(6),
        "missing_param": 9, "absent_param": 11,
    }


def base_recipe(*, columns, label_cutoff=at(18),
                metric_units=None) -> dict:
    return {
        "recipe_id": "test-recipe",
        "label_cutoff": label_cutoff,
        "metric_units": metric_units or {"temperature": "K"},
        "features": columns,
    }


def col_3_6(name="f36", pnum=0):
    return feature_column_doc(
        name, offset_start=0, offset_end=3 * 3600,
        slot=SLOT_3_6, parameter_number=pnum)


def col_0_6(name="f06", pnum=0):
    return feature_column_doc(
        name, offset_start=-3 * 3600, offset_end=3 * 3600,
        slot=SLOT_0_6, parameter_number=pnum)


def standard_station_streams(world, *, station_id="ST1",
                             map_a_to=at(18), map_b=True,
                             grid_digest=None, a_point=POINT_A,
                             b_point=POINT_B, ghost=False):
    """The canonical A->B migration timeline plus publish/withdraw."""

    grid_digest = grid_digest or world["grid_digest"]
    publish = [
        publish_record(record_id="P1", origin=world["origin"],
                       visible_at=at(8), seq=0, action="publish",
                       package_digest=world["package_digest"]),
        publish_record(record_id="P2", origin=world["origin"],
                       visible_at=at(9), seq=0, action="withdraw"),
    ]
    station = [
        station_record(record_id="M1", station_id=station_id,
                       visible_at=at(7, 30), seq=0, action="map",
                       effective_from=at(0), effective_to=map_a_to,
                       grid_digest=grid_digest,
                       lat_index=a_point[0], lon_index=a_point[1])]
    if map_b:
        station.append(station_record(
            record_id="M2", station_id=station_id,
            visible_at=at(8, 30), seq=0, action="map",
            effective_from=at(9), effective_to=at(12),
            grid_digest=grid_digest,
            lat_index=b_point[0], lon_index=b_point[1]))
    return publish, station


def run_export(tmp_path, world, *, recipe, samples, publish, station,
              observations, config=None):
    d = tmp_path / "streams"
    paths = {
        "recipe": d / "recipe.json",
        "samples": d / "samples.jsonl",
        "publish": d / "publish.jsonl",
        "station": d / "station.jsonl",
        "observation": d / "observation.jsonl"}
    d.mkdir(parents=True, exist_ok=True)
    paths["recipe"].write_text(
        json.dumps(recipe, ensure_ascii=False), encoding="utf-8")
    write_jsonl(paths["samples"], samples)
    write_jsonl(paths["publish"], publish)
    write_jsonl(paths["station"], station)
    write_jsonl(paths["observation"], observations)
    report = export_training_table(
        store_path=world["store"], recipe_path=paths["recipe"],
        samples_path=paths["samples"], publish_path=paths["publish"],
        station_path=paths["station"],
        observation_path=paths["observation"],
        out_jsonl=tmp_path / "out.jsonl",
        out_summary=tmp_path / "summary.json",
        config=config or ExportConfig())
    rows = [json.loads(line)
            for line in (tmp_path / "out.jsonl").read_text(
        encoding="utf-8").splitlines()]
    summary = json.loads((tmp_path / "summary.json").read_text(
        encoding="utf-8"))
    return rows, summary, report
