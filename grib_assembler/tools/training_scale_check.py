#!/usr/bin/env python3
"""Independent scale and bounded-memory check for the training export.

This script is standalone and writes only into a fresh temp directory.  It

1. builds one small real imported cube with a handful of value blocks (the
   same catalog -> assembler -> importer path as production);
2. generates a large, deliberately shuffled sample manifest plus shuffled
   publish/station/observation streams, spreading many samples over a few
   stations and one origin so distant sample batches reuse the same fields;
3. runs the real export at several input sizes and buffer configurations;
4. asserts full row counts, sampled feature values, that every accessed
   values block is decoded at most once, and that peak resident memory stays
   bounded by chunk/buffer configuration rather than growing with the number
   of samples.

Expectations (row counts, grid values) are hand-specified constants here;
nothing is reverse-engineered from the exporter.  Run::

    python -I tools/training_scale_check.py --sizes 200 800 3200
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from grib_asm.timeutil import TimeSlot  # noqa: E402
from grib_asm.training.export import ExportConfig, export_training_table  # noqa: E402
from training_fixtures import (  # noqa: E402
    NI, NJ, POINT_A, POINT_B, build_cube, encode_stat_message,
    feature_column_doc, observation_record, publish_record, sample_record,
    station_record, write_jsonl,
)

UTC = timezone.utc
DAY = datetime(2026, 1, 1, tzinfo=UTC)
SLOT_3_6 = TimeSlot(3 * 3600, 6 * 3600, 1)
# Hand-authored grid values used by every expectation below.
VALUE_A, VALUE_B = 21.5, 42.25
N_STATIONS = 8


def _at(hour: int, minute: int = 0) -> str:
    return (DAY + timedelta(hours=hour, minutes=minute)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_small_cube(tmp: Path):
    origin = DAY.replace(hour=6)
    grid = np.zeros((NJ, NI), dtype="float64")
    grid[POINT_A] = VALUE_A
    grid[POINT_B] = VALUE_B
    raw = encode_stat_message(
        origin_dt=origin, start_hour=3, end_hour=6, values=grid)
    origin_iso = _at(6)
    store, grid_dig, pkg = build_cube(
        tmp, [{"collection_id": "c06", "origin": origin_iso,
               "messages": [(raw, SLOT_3_6, 0)]}])
    return store, grid_dig, pkg[origin_iso], origin_iso


def generate_streams(tmp: Path, n_samples: int, *, grid_dig: str,
                     package_digest: str, origin_iso: str, seed: int = 1234):
    rng = random.Random(seed)
    stations = [f"ST{i:02d}" for i in range(N_STATIONS)]
    stream_dir = tmp / "streams"
    stream_dir.mkdir(parents=True, exist_ok=True)

    # One map per station, alternating points A/B, all visible well before
    # the decisions and covering the whole day.
    station = []
    for i, sid in enumerate(stations):
        point = POINT_A if i % 2 == 0 else POINT_B
        station.append(station_record(
            record_id=f"M-{sid}", station_id=sid, visible_at=_at(7),
            seq=0, action="map", effective_from=_at(0),
            effective_to=_at(23, 59), grid_digest=grid_dig,
            lat_index=point[0], lon_index=point[1]))
    # One publish + one later withdraw that no sample's decision reaches
    # (all decisions are 08:30), so every sample resolves to an OK feature.
    publish = [publish_record(
        record_id="P1", origin=origin_iso, visible_at=_at(8), seq=0,
        action="publish", package_digest=package_digest)]

    samples, observations = [], []
    for k in range(n_samples):
        sid = stations[k % N_STATIONS]
        # All samples share the 09-12 target so they hit one value block.
        samples.append(sample_record(
            sample_id=f"sample-{k:08d}", station_id=sid,
            metric="temperature", target_start=_at(9), target_end=_at(12),
            decision_at=_at(8, 30)))
        # Distinct visibility instant and sequence per record keeps the
        # (business key, visible_at, seq) token unique even when several
        # samples share a station and target interval.
        minute = 1 + (k % 59)
        observations.append(observation_record(
            record_id=f"O-{k:08d}", station_id=sid, metric="temperature",
            visible_at=_at(12, minute), seq=k, action="set",
            target_start=_at(9), target_end=_at(12),
            value=float(k % 100), unit="K"))

    # Shuffle every stream: line order must be irrelevant.
    for seq in (station, publish, samples, observations):
        rng.shuffle(seq)

    recipe = {
        "recipe_id": "scale-recipe", "label_cutoff": _at(18),
        "metric_units": {"temperature": "K"},
        "features": [feature_column_doc(
            "f36", offset_start=0, offset_end=3 * 3600, slot=SLOT_3_6)]}
    paths = {
        "recipe": stream_dir / "recipe.json",
        "samples": stream_dir / "samples.jsonl",
        "publish": stream_dir / "publish.jsonl",
        "station": stream_dir / "station.jsonl",
        "observation": stream_dir / "observation.jsonl"}
    paths["recipe"].write_text(json.dumps(recipe), encoding="utf-8")
    write_jsonl(paths["samples"], samples)
    write_jsonl(paths["publish"], publish)
    write_jsonl(paths["station"], station)
    write_jsonl(paths["observation"], observations)
    return paths, stations


def run_one(tmp: Path, n_samples: int, point_batch: int,
            sqlite_cache_kib: int):
    store, grid_dig, package_digest, origin_iso = build_small_cube(
        tmp / f"cube-{n_samples}-{point_batch}")
    paths, stations = generate_streams(
        tmp / f"in-{n_samples}-{point_batch}", n_samples,
        grid_dig=grid_dig, package_digest=package_digest,
        origin_iso=origin_iso)
    out_jsonl = tmp / f"out-{n_samples}-{point_batch}.jsonl"
    out_summary = tmp / f"sum-{n_samples}-{point_batch}.json"
    base_rss = _rss_mb()
    report = export_training_table(
        store_path=store, recipe_path=paths["recipe"],
        samples_path=paths["samples"], publish_path=paths["publish"],
        station_path=paths["station"],
        observation_path=paths["observation"],
        out_jsonl=out_jsonl, out_summary=out_summary,
        config=ExportConfig(point_batch=point_batch,
                            sqlite_cache_kib=sqlite_cache_kib))
    peak_rss = _rss_mb()
    rows = [json.loads(line) for line in out_jsonl.read_text(
        encoding="utf-8").splitlines()]
    summary = json.loads(out_summary.read_text(encoding="utf-8"))

    # Full row count and ordering.
    assert len(rows) == n_samples
    ids = [r["sample_id"] for r in rows]
    assert ids == sorted(ids, key=lambda s: s.encode("utf-8"))
    # Every feature resolved OK with the hand-authored point value.
    expected = {sid: (VALUE_A if i % 2 == 0 else VALUE_B)
                for i, sid in enumerate(stations)}
    for row in rows:
        feature = row["features"][0]
        assert feature["status"] == "ok", (row["sample_id"], feature["status"])
        assert feature["value"] == expected[row["station_id"]]
        assert row["label"]["status"] == "ok"
    # All samples target one origin/slot/level/member cell: exactly one
    # distinct values block, decoded exactly once regardless of sample count.
    assert report.value_block_reads == 1, report.value_block_reads
    assert summary["value_block_reads"] == 1
    return {
        "samples": n_samples, "point_batch": point_batch,
        "sqlite_cache_kib": sqlite_cache_kib,
        "rows": len(rows),
        "value_block_reads": report.value_block_reads,
        "base_rss_mb": round(base_rss, 2),
        "peak_rss_mb": round(peak_rss, 2),
        "delta_rss_mb": round(peak_rss - base_rss, 2),
        "ok_cells": summary["feature_status_counts"].get("ok", 0),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[200, 800, 3200])
    parser.add_argument("--small-batch", type=int, default=256)
    parser.add_argument("--large-batch", type=int, default=4096)
    parser.add_argument("--sqlite-cache-kib", type=int, default=16 * 1024)
    args = parser.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="training-scale-"))
    measurements = []
    try:
        for n in args.sizes:
            for batch in (args.small_batch, args.large_batch):
                measurements.append(run_one(
                    tmp / f"run-{n}-{batch}", n, batch,
                    args.sqlite_cache_kib))
        # Flatness: per-sample marginal RSS between the smallest and largest
        # run at the same batch setting must stay near zero (disk-backed, not
        # accumulating rows in Python memory).
        slopes = {}
        for batch in (args.small_batch, args.large_batch):
            runs = [m for m in measurements if m["point_batch"] == batch]
            small, large = runs[0], runs[-1]
            extra = max(large["samples"] - small["samples"], 1)
            extra_mb = max(large["delta_rss_mb"] - small["delta_rss_mb"], 0.0)
            slopes[f"batch{batch}_per_1000_samples_mb"] = round(
                extra_mb / extra * 1000, 4)
        report = {
            "sizes": args.sizes,
            "measurements": measurements,
            "slopes": slopes,
            "value_block_rule": "each accessed values block decoded at most "
                                "once; all runs observed exactly 1 read for 1 "
                                "shared cell",
            "conclusion": "peak memory bounded by SQLite cache, point batch "
                          "and one grid chunk, not by sample count",
        }
        print(json.dumps(report, indent=2))
        # A genuinely flat pipeline adds only a few MB per thousand extra
        # samples (SQLite/process overhead), never a grid's worth per sample.
        for value in slopes.values():
            assert value < 5.0, value
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
