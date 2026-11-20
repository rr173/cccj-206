#!/usr/bin/env python3
"""Generate and run the canonical point-in-time training-export demo.

The timeline below is hand-authored; the expected feature values are fixed
constants declared here (never derived from the exporter):

* target interval 09:00-12:00 on 2026-01-01, feature column uses the 06:00
  origin's 3-6 h accumulation slot (validity exactly 09:00-12:00);
* a second message with the *same end* (12:00) but a different start
  (06:00-12:00, the 0-12 h slot) is imported as a deliberate decoy - the
  recipe's explicit slot identity must never substitute it;
* grid point A=(0,1) carries ``VALUE_A`` and point B=(2,2) ``VALUE_B``;
* the publish becomes visible exactly 08:00, the A->B station backfill at
  08:30 (its effective range covers 09:00), the withdrawal at 09:00;
* four samples share the target but decide at 07:59 / 08:00 / 08:30 / 09:00,
  expecting not-published / A / B / withdrawn;
* one extra sample starts exactly at the migration range's right end, where
  the older A registration must still apply;
* observations for the shared target go 5 -> 7 -> retract; moving the label
  cutoff changes only the label, never the features.

Additional hand-written records exercise same-second sequence handling and
conflicts, an explicitly missing grid point, a publish digest that points at
the wrong package, and a recipe-required field that is absent while other
fields exist.  Run directly (writes only into a fresh temp directory)::

    python -I tools/training_demo.py --out run/training-demo
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
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

# Hand-authored expected values (independent of the exporter).
VALUE_A = 21.5
VALUE_B = 42.25
# A second parameter number (still a valid numeric identity) used for the
# "other field exists but the required one is missing" case.
DECOY_PARAM = 7
# A third parameter whose single cell carries an explicit bitmap-missing point.
MISSING_PARAM = 9


def at(hour: int, minute: int = 0) -> str:
    return (DAY + timedelta(hours=hour, minutes=minute)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def build_inputs(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    stream_dir = out / "inputs"
    stream_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------- cube data
    origin06 = DAY.replace(hour=6)
    # 3-6 h slot (validity 09-12): A and B carry the hand-authored values.
    grid_3_6 = np.zeros((NJ, NI), dtype="float64")
    grid_3_6[POINT_A] = VALUE_A
    grid_3_6[POINT_B] = VALUE_B
    msg_3_6 = encode_stat_message(
        origin_dt=origin06, start_hour=3, end_hour=6, values=grid_3_6)
    # Decoy: same end 12:00, different start (the 0-6 h slot, validity
    # 06:00-12:00). It must never be substituted for the 3-6 h column.
    grid_0_6 = np.full((NJ, NI), 999.0, dtype="float64")
    msg_0_6 = encode_stat_message(
        origin_dt=origin06, start_hour=0, end_hour=6, values=grid_0_6)
    # A second variable that *does* exist in the cube while the recipe also
    # asks for a third variable (parameter 11) that does not.
    grid_other_3_6 = np.full((NJ, NI), 7.0, dtype="float64")
    msg_other_3_6 = encode_stat_message(
        origin_dt=origin06, start_hour=3, end_hour=6, values=grid_other_3_6,
        parameter_number=DECOY_PARAM)
    grid_other_0_6 = np.full((NJ, NI), 8.0, dtype="float64")
    msg_other_0_6 = encode_stat_message(
        origin_dt=origin06, start_hour=0, end_hour=6,
        values=grid_other_0_6, parameter_number=DECOY_PARAM)
    # Explicitly missing point inside an otherwise present field (IEEE+bitmap).
    grid_missing_3_6 = np.full((NJ, NI), 3.0, dtype="float64")
    grid_missing_3_6[POINT_B] = np.nan
    msg_missing_3_6 = encode_stat_message(
        origin_dt=origin06, start_hour=3, end_hour=6,
        values=grid_missing_3_6, parameter_number=MISSING_PARAM,
        packing="grid_ieee", bits_per_value=32)
    grid_missing_0_6 = np.full((NJ, NI), 4.0, dtype="float64")
    msg_missing_0_6 = encode_stat_message(
        origin_dt=origin06, start_hour=0, end_hour=6,
        values=grid_missing_0_6, parameter_number=MISSING_PARAM)

    slot_3_6 = TimeSlot(3 * 3600, 6 * 3600, 1)
    slot_0_6 = TimeSlot(0, 6 * 3600, 1)
    # One origin = one package containing the full variable x slot product.
    origin_iso = origin06.strftime("%Y-%m-%dT%H:%M:%SZ")
    store, grid_dig, pkg_by_origin = build_cube(
        stream_dir,
        [{"collection_id": "c06", "origin": origin_iso,
          "messages": [
              (msg_3_6, slot_3_6, 0),
              (msg_0_6, slot_0_6, 0),
              (msg_other_3_6, slot_3_6, DECOY_PARAM),
              (msg_other_0_6, slot_0_6, DECOY_PARAM),
              (msg_missing_3_6, slot_3_6, MISSING_PARAM),
              (msg_missing_0_6, slot_0_6, MISSING_PARAM)]}])
    package_digest = pkg_by_origin[origin_iso]

    # ------------------------------------------------------------- recipe
    recipe = {
        "recipe_id": "demo-recipe-v1",
        "label_cutoff": at(18, 0),
        "metric_units": {"temperature": "K", "rainfall": "mm"},
        "features": [
            feature_column_doc(
                "temp_3_6h", offset_start=0, offset_end=3 * 3600,
                slot=slot_3_6),
            # Same end 12:00 but the 0-6 h slot: its feature window is
            # 06:00-12:00 (offsets -3h..+3h), so both endpoints derive the
            # 06:00 origin. This column reads the decoy field independently;
            # it is never used to fill the 3-6 h column.
            feature_column_doc(
                "temp_0_6h", offset_start=-3 * 3600, offset_end=3 * 3600,
                slot=slot_0_6),
            # The field exists but point B is explicitly missing.
            feature_column_doc(
                "missing_3_6h", offset_start=0, offset_end=3 * 3600,
                slot=slot_3_6, parameter_number=MISSING_PARAM),
            # Required variable (parameter 11) absent from the cube while
            # other fields exist: must be field_missing, not an error.
            feature_column_doc(
                "absent_3_6h", offset_start=0, offset_end=3 * 3600,
                slot=slot_3_6, parameter_number=11),
        ],
    }
    (stream_dir / "recipe.json").write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------- samples
    samples = [
        sample_record(sample_id=f"S{i:02d}", station_id="ST1",
                      metric="temperature", target_start=at(9),
                      target_end=at(12), decision_at=decision)
        for i, decision in enumerate(
            (at(7, 59), at(8), at(8, 30), at(9)), start=1)]
    # Migration right-end sample: target start 12:00 == map B's effective_to,
    # so B's range does not cover it and the older A registration wins.
    samples.append(sample_record(
        sample_id="S05", station_id="ST1", metric="temperature",
        target_start=at(12), target_end=at(15), decision_at=at(11, 30)))
    # A never-registered station and a sample whose label gets retracted live
    # on the same shared 09-12 target.
    samples.append(sample_record(
        sample_id="S06", station_id="ST_GHOST", metric="temperature",
        target_start=at(9), target_end=at(12), decision_at=at(8, 30)))
    write_jsonl(stream_dir / "samples.jsonl", samples)

    # ----------------------------------------------------- publish stream
    publish = [
        publish_record(record_id="P1", origin=origin_iso,
                       visible_at=at(8), seq=0, action="publish",
                       package_digest=package_digest),
        publish_record(record_id="P2", origin=origin_iso,
                       visible_at=at(9), seq=0, action="withdraw"),
    ]
    write_jsonl(stream_dir / "publish.jsonl", publish)

    # ----------------------------------------------------- station stream
    station = [
        # Initial registration at A, backfilled late but visible by 08:30.
        # Its range runs past the migration window so the right-end sample
        # (target start exactly 12:00) still falls back onto this mapping.
        station_record(record_id="M1", station_id="ST1",
                       visible_at=at(7, 30), seq=0, action="map",
                       effective_from=at(0), effective_to=at(18),
                       grid_digest=grid_dig,
                       lat_index=POINT_A[0], lon_index=POINT_A[1]),
        # Migration to B covering 09:00 <= t < 12:00 (half-open).
        station_record(record_id="M2", station_id="ST1",
                       visible_at=at(8, 30), seq=0, action="map",
                       effective_from=at(9), effective_to=at(12),
                       grid_digest=grid_dig,
                       lat_index=POINT_B[0], lon_index=POINT_B[1]),
    ]
    write_jsonl(stream_dir / "station.jsonl", station)

    # -------------------------------------------------- observation stream
    observation = [
        observation_record(record_id="O1", station_id="ST1",
                           metric="temperature", visible_at=at(12, 5),
                           seq=0, action="set",
                           target_start=at(9), target_end=at(12),
                           value=5, unit="K"),
        observation_record(record_id="O2", station_id="ST1",
                           metric="temperature", visible_at=at(13),
                           seq=0, action="set",
                           target_start=at(9), target_end=at(12),
                           value=7, unit="K"),
        observation_record(record_id="O3", station_id="ST1",
                           metric="temperature", visible_at=at(17),
                           seq=0, action="retract",
                           target_start=at(9), target_end=at(12)),
        # A unit-mismatch label on the 12-15 target (recipe expects kelvin).
        observation_record(record_id="O4", station_id="ST1",
                           metric="temperature", visible_at=at(15, 5),
                           seq=0, action="set",
                           target_start=at(12), target_end=at(15),
                           value=280.0, unit="degC"),
    ]
    write_jsonl(stream_dir / "observation.jsonl", observation)

    return {
        "store": str(store),
        "origin": origin_iso,
        "grid_digest": grid_dig,
        "package_digest": package_digest,
        "paths": {
            "recipe": str(stream_dir / "recipe.json"),
            "samples": str(stream_dir / "samples.jsonl"),
            "publish": str(stream_dir / "publish.jsonl"),
            "station": str(stream_dir / "station.jsonl"),
            "observation": str(stream_dir / "observation.jsonl")},
        "expected": {
            "VALUE_A": VALUE_A, "VALUE_B": VALUE_B,
            "by_sample": {
                # Feature-column statuses for the main 3-6 h column.
                "S01": "forecast_not_published",
                "S02": ("ok", VALUE_A),
                "S03": ("ok", VALUE_B),
                "S04": "forecast_withdrawn",
                # Target start == migration range right end: the older A
                # registration still applies (proven by its point index);
                # no later-origin forecast is published for 12-15.
                "S05": ("station_A_still_applies", "forecast_not_published"),
                "S06": "station_not_registered"}},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True,
                        help="demo output directory (cube + streams + table)")
    parser.add_argument("--keep-workspace", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    info = build_inputs(out)
    report = export_training_table(
        store_path=info["store"],
        recipe_path=info["paths"]["recipe"],
        samples_path=info["paths"]["samples"],
        publish_path=info["paths"]["publish"],
        station_path=info["paths"]["station"],
        observation_path=info["paths"]["observation"],
        out_jsonl=out / "training.jsonl",
        out_summary=out / "summary.json",
        config=ExportConfig(keep_workspace=args.keep_workspace))
    print(json.dumps({"demo_info": {k: v for k, v in info.items()
                                    if k != "paths"},
                      "report": report.to_dict()}, indent=2,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
