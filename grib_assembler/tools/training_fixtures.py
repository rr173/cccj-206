"""Hand-built fixture generation for the point-in-time training export.

Everything here is *test/demo tooling*.  It produces:

1. real GRIB2 messages (regular lat/lon, GDT 3.0, scan mode 0) that travel
   through the project's own catalog -> assembler -> cube importer, so the
   export always reads a genuinely committed Zarr cube - never a synthetic
   in-memory array;
2. the four order-independent JSONL streams (samples, publish, station,
   observation) and the feature recipe.

The expected feature *values* are deliberately specified up front by the
caller as plain grid arrays (``A_VALUE`` / ``B_VALUE`` constants in the demo
script), not reverse-engineered from the exporter.  Missing grid points use
IEEE packing with an explicit bitmap, which is the one packing this ecCodes
build round-trips into the cube's ``present`` mask faithfully.

Only small grids are generated; large, shuffled manifests are materialised at
runtime under a temporary directory by ``training_scale_check.py``.  Nothing
large is committed to the repository.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eccodes import (
    codes_get_message, codes_grib_new_from_samples, codes_release, codes_set,
    codes_set_values,
)

from grib_asm.assemble import assemble
from grib_asm.catalog import Catalog
from grib_asm.identity import grid_digest
from grib_asm.cube.importer import import_package
from grib_asm.cube.store import CubeConfig
from grib_asm.request import (
    Manifest, Request, RequestedLevel, RequestedMember, RequestedVariable,
)
from grib_asm.scanner import scan_file
from grib_asm.timeutil import TimeSlot

UTC = timezone.utc

# Hand-authored grid geometry (matches the rest of the project's demo data).
NI, NJ = 4, 3
FIRST_LAT, FIRST_LON = 40.0, 10.0
DELTA_LAT, DELTA_LON = 1.0, 1.0
# Grid points used by the station timeline demo.
POINT_A = (0, 1)          # (lat_index, lon_index)
POINT_B = (2, 2)

DISC, CAT, NUM = 0, 0, 0
CENTRE, SUBCENTRE = 7, 0
MTV, LTV = 29, 1
LEVEL_TYPE, LEVEL_SCALE, LEVEL_FACTOR = 100, 85000, 0


# --------------------------------------------------------------------------- #
# low-level GRIB2 authoring
# --------------------------------------------------------------------------- #
def _set_all(handle, pairs):
    for key, value in pairs:
        try:
            codes_set(handle, key, value)
        except Exception:
            # Build-specific alias for the statistical process code.
            if key == "statisticalProcess":
                codes_set(handle, "typeOfStatisticalProcessing", value)
            else:
                raise


def encode_stat_message(*, origin_dt: datetime, start_hour: int,
                        end_hour: int, values: np.ndarray,
                        parameter_number: int = NUM,
                        ensemble_type: int = 3, perturbation_number: int = 0,
                        statistic: int = 1,
                        packing: str = "grid_simple",
                        bits_per_value: int = 24) -> bytes:
    """Encode one PDT 4.11 interval-statistics message.

    ``values`` is the full ``(NJ, NI)`` grid; NaN entries become explicit
    bitmap-missing points when IEEE packing is requested.
    """

    flat = np.asarray(values, dtype="float64").reshape(-1)
    has_missing = bool(np.isnan(flat).any())
    bitmap_present = 1 if has_missing else 0
    handle = codes_grib_new_from_samples("GRIB2")
    pairs = [
        ("discipline", DISC), ("parameterCategory", CAT),
        ("parameterNumber", int(parameter_number)),
        ("tablesVersion", MTV), ("localTablesVersion", LTV),
        ("centre", CENTRE), ("subCentre", SUBCENTRE),
        ("dataDate", int(origin_dt.strftime("%Y%m%d"))),
        ("dataTime", int(origin_dt.strftime("%H%M"))),
        ("gridDefinitionTemplateNumber", 0),
        ("Ni", NI), ("Nj", NJ),
        ("latitudeOfFirstGridPointInDegrees", FIRST_LAT),
        ("longitudeOfFirstGridPointInDegrees", FIRST_LON),
        ("iDirectionIncrementInDegrees", DELTA_LON),
        ("jDirectionIncrementInDegrees", DELTA_LAT),
        ("longitudeOfLastGridPointInDegrees",
         FIRST_LON + (NI - 1) * DELTA_LON),
        ("latitudeOfLastGridPointInDegrees",
         FIRST_LAT - (NJ - 1) * DELTA_LAT),
        ("scanningMode", 0),
        ("productDefinitionTemplateNumber", 11),
        ("typeOfFirstFixedSurface", LEVEL_TYPE),
        ("scaleFactorOfFirstFixedSurface", LEVEL_FACTOR),
        ("scaledValueOfFirstFixedSurface", LEVEL_SCALE),
        ("typeOfEnsembleForecast", ensemble_type),
        ("perturbationNumber", perturbation_number),
        ("indicatorOfUnitOfTimeRange", 1),
        ("forecastTime", start_hour),
        ("numberOfTimeRange", 1),
        ("typeOfTimeIncrement", 2),
        ("indicatorOfUnitForTimeIncrement", 1),
        ("timeIncrement", 0),
        ("numberOfMissingInStatisticalProcess", 0),
        ("statisticalProcess", statistic),
        ("indicatorOfUnitForTimeRange", 1),
        ("lengthOfTimeRange", end_hour - start_hour),
        ("yearOfEndOfOverallTimeInterval", origin_dt.year),
        ("monthOfEndOfOverallTimeInterval", origin_dt.month),
        ("dayOfEndOfOverallTimeInterval", origin_dt.day),
        ("hourOfEndOfOverallTimeInterval",
         origin_dt.hour + end_hour),
        ("minuteOfEndOfOverallTimeInterval", 0),
        ("secondOfEndOfOverallTimeInterval", 0),
        ("packingType", packing),
        ("bitsPerValue", bits_per_value),
    ]
    _set_all(handle, pairs)
    if bitmap_present:
        # IEEE packing preserves NaN through the bitmap into an explicit
        # missing point; simple packing cannot represent it in this build.
        codes_set(handle, "bitmapPresent", 1)
    codes_set_values(handle, flat)
    raw = codes_get_message(handle)
    codes_release(handle)
    return raw


# --------------------------------------------------------------------------- #
# real cube construction
# --------------------------------------------------------------------------- #
def build_cube(tmp: Path, collections: list[dict], *,
               batch_size: int = 8):
    """Publish collections through catalog+assembler and import the cube.

    A collection is ``{"collection_id", "origin", "messages": [(raw, slot,
    parameter_number), ...]}`` and represents one origin's hand-off package:
    several variables/slots may share one package digest, but re-importing the
    same origin with a different digest is (correctly) a source conflict, so
    every message for one origin travels in one collection.

    Returns ``(store_path, grid_digest, package_digest_by_origin)``.
    """

    package_dirs = []
    grid_dig = None
    package_by_origin: dict[str, str] = {}
    for collection in collections:
        collection_id = collection["collection_id"]
        root = tmp / collection_id
        root.mkdir(parents=True, exist_ok=True)
        rel = "all.grib2"
        blob = b"".join(raw for raw, _slot, _pnum in collection["messages"])
        (root / rel).write_bytes(blob)
        digest = hashlib.sha256(blob).hexdigest()

        class _Entry:
            pass

        entry = _Entry()
        entry.file_id = "all"
        entry.relative_path = rel
        entry.size_bytes = len(blob)
        entry.sha256 = digest
        manifest_doc = {
            "collection_id": collection_id,
            "created_at": "2026-01-01T00:00:00Z",
            "files": [{"file_id": "all", "relative_path": rel,
                       "size_bytes": len(blob), "sha256": digest}]}
        manifest = Manifest(collection_id, "2026-01-01T00:00:00Z",
                            [entry], manifest_doc)
        cat = Catalog(tmp / f"{collection_id}.db")
        cat.ingest_file(manifest, entry, root)
        scan = scan_file(open(root / rel, "rb"), len(blob))
        dig = grid_digest(scan.messages[0].gds_bytes)
        if grid_dig is None:
            grid_dig = dig

        # Build the request Cartesian product from the distinct identities
        # actually present in this collection's messages.
        pnums = sorted({pnum for _r, _s, pnum in collection["messages"]})
        slots = []
        for _r, slot, _p in collection["messages"]:
            if not any(slot.start_seconds == s.start_seconds
                       and slot.end_seconds == s.end_seconds
                       and slot.statistic == s.statistic for s in slots):
                slots.append(slot)
        variables = [RequestedVariable(DISC, CAT, pnum, MTV, LTV,
                                       CENTRE, SUBCENTRE) for pnum in pnums]
        level = RequestedLevel(LEVEL_TYPE, LEVEL_FACTOR, LEVEL_SCALE,
                               None, None, None)
        member = RequestedMember("control", 3, 0)
        request = Request(
            request_id=f"req-{collection_id}",
            origin_from=datetime(2026, 1, 1, tzinfo=UTC),
            origin_to=datetime(2026, 12, 31, tzinfo=UTC),
            not_after=datetime(2027, 1, 1, tzinfo=UTC),
            expected_grid_digest=grid_dig,
            variables=variables, levels=[level], members=[member],
            slots=slots)
        result = assemble(cat, request, {collection_id: root},
                          tmp / "packages", {collection_id: manifest})
        cat.close()
        package_dirs.append(result.package_dir)
        marker = json.loads((result.package_dir
                             / "PACKAGE_COMPLETE.json").read_text(
            encoding="utf-8"))
        package_by_origin[collection["origin"]] = marker["data_sha256"]

    store = tmp / "cube"
    for pkg in package_dirs:
        import_package(store, pkg, CubeConfig(batch_size=batch_size))
    return store, grid_dig, package_by_origin


# --------------------------------------------------------------------------- #
# JSONL authoring helpers
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def publish_record(*, record_id: str, origin: str, visible_at: str, seq: int,
                   action: str, package_digest: str | None = None) -> dict:
    rec = {"record_id": record_id, "stream": "publish", "action": action,
           "origin": origin, "visible_at": visible_at, "seq": seq}
    if action == "publish":
        rec["package_summary"] = {"package_digest": package_digest}
    return rec


def station_record(*, record_id: str, station_id: str, visible_at: str,
                   seq: int, action: str, effective_from: str,
                   effective_to: str, grid_digest: str | None = None,
                   lat_index: int | None = None,
                   lon_index: int | None = None) -> dict:
    rec = {"record_id": record_id, "stream": "station", "action": action,
           "station_id": station_id, "visible_at": visible_at, "seq": seq,
           "effective_from": effective_from, "effective_to": effective_to}
    if action == "map":
        rec.update({"grid_digest": grid_digest,
                    "lat_index": lat_index, "lon_index": lon_index})
    return rec


def observation_record(*, record_id: str, station_id: str, metric: str,
                       visible_at: str, seq: int, action: str,
                       target_start: str, target_end: str,
                       value=None, unit: str | None = None) -> dict:
    rec = {"record_id": record_id, "stream": "observation", "action": action,
           "station_id": station_id, "metric": metric,
           "visible_at": visible_at, "seq": seq,
           "target_start": target_start, "target_end": target_end}
    if action == "set":
        rec.update({"value": value, "unit": unit})
    return rec


def sample_record(*, sample_id: str, station_id: str, metric: str,
                  target_start: str, target_end: str,
                  decision_at: str) -> dict:
    return {"sample_id": sample_id, "station_id": station_id,
            "metric": metric, "target_start": target_start,
            "target_end": target_end, "decision_at": decision_at}


def feature_column_doc(name: str, *, offset_start: int, offset_end: int,
                       slot: TimeSlot, parameter_number: int = NUM) -> dict:
    return {
        "name": name,
        "variable": {
            "discipline": DISC, "parameterCategory": CAT,
            "parameterNumber": int(parameter_number),
            "masterTablesVersionNumber": MTV,
            "localTablesVersionNumber": LTV,
            "centre": CENTRE, "subCentre": SUBCENTRE},
        "level": {"first": {"type": LEVEL_TYPE,
                            "scale_factor": LEVEL_FACTOR,
                            "scaled_value": LEVEL_SCALE},
                  "second": None},
        "member": {"kind": "control",
                   "typeOfEnsembleForecast": 3,
                   "perturbationNumber": 0},
        "slot": slot.to_dict(),
        "feature_offset_start_seconds": int(offset_start),
        "feature_offset_end_seconds": int(offset_end),
    }


def standard_grid_digest() -> str:
    """Grid digest of one standard-grid message (independent of the cube)."""

    spec_raw = encode_stat_message(
        origin_dt=datetime(2026, 1, 1, tzinfo=UTC), start_hour=0, end_hour=1,
        values=np.zeros((NJ, NI), dtype="float64"))
    import io
    scan = scan_file(io.BytesIO(spec_raw), len(spec_raw))
    return grid_digest(scan.messages[0].gds_bytes)
