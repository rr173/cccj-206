"""Deterministic GRIB2 fixture generation via ecCodes.

This is *test tooling only*.  It is the single place allowed to touch grid
point values, and even here it writes a tiny synthetic array so ecCodes can
emit a legal data section.  The assembler core never imports this module and
never reads ``values`` back.

The generator encodes every profile-relevant shape the exercises need:

* PDT 4.0/4.1 instantaneous and 4.8/4.11 single-range statistical products;
* control / perturbed / deterministic member identities;
* exact rational fixed surfaces with missing second surfaces;
* second/minute/hour/day step units;
* alternative packings (bits-per-value) that must not change grid identity;
* same-header/different-data messages that create whole-message conflicts;
* unsupported templates, units and statistics;
* truncated and junk-tail files.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import numpy as np
    from eccodes import (
        codes_grib_new_from_samples, codes_set, codes_set_values,
        codes_get_message, codes_release,
    )
    DEPS_AVAILABLE = True
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised when deps are absent
    np = None
    codes_grib_new_from_samples = None
    codes_set = codes_set_values = codes_get_message = codes_release = None
    DEPS_AVAILABLE = False
    IMPORT_ERROR = exc
MISSING = 255


@dataclass
class FixtureSpec:
    name: str = "fixture"
    pdt: int = 0
    discipline: int = 0
    parameter_category: int = 0
    parameter_number: int = 0
    master_tables_version: int = 29
    local_tables_version: int = 1
    centre: int = 7          # ecmwf default table value
    sub_centre: int = 0
    data_date: int = 20260101
    data_time: int = 0
    # Grid (lat/lon template 0).
    ni: int = 4
    nj: int = 3
    first_lat: float = 40.0
    first_lon: float = 10.0
    delta_lat: float = 1.0
    delta_lon: float = 1.0
    # Level.
    first_surface_type: int = 100            # isobaric surface (Pa)
    first_scale: int = 0
    first_scaled: int = 85000
    second_surface_type: int = MISSING
    second_scale: int = MISSING
    second_scaled: int = 0xFFFFFFFF
    # Ensemble.
    ensemble_type: int | None = None
    perturbation_number: int | None = None
    # Timing.
    forecast_time: int = 0
    time_unit: int = 1                       # hour
    # Statistical (PDT 4.8/4.11).
    number_of_time_range: int = 1
    type_of_time_increment: int = 2
    time_increment_unit: int = 1
    time_increment: int = 0
    number_missing: int = 0
    statistical_process: int = 0
    length_of_time_range: int = 0
    time_range_unit: int = 1
    end_year: int = 2026
    end_month: int = 1
    end_day: int = 1
    end_hour: int = 0
    end_minute: int = 0
    end_second: int = 0
    # Packing / data.
    bits_per_value: int = 16
    data_seed: int = 1
    values: list | None = None


def encode_message(spec: FixtureSpec) -> bytes:
    h = codes_grib_new_from_samples("GRIB2")
    try:
        # Identification section.  Key names differ across ecCodes builds:
        # tablesVersion/localTablesVersion are the common aliases for the
        # WMO master/local table version numbers.
        codes_set(h, "discipline", spec.discipline)
        try:
            codes_set(h, "tablesVersion", spec.master_tables_version)
        except Exception:
            codes_set(h, "masterTablesVersionNumber",
                      spec.master_tables_version)
        try:
            codes_set(h, "localTablesVersion", spec.local_tables_version)
        except Exception:
            codes_set(h, "localTablesVersionNumber",
                      spec.local_tables_version)
        codes_set(h, "centre", spec.centre)
        codes_set(h, "subCentre", spec.sub_centre)
        codes_set(h, "dataDate", spec.data_date)
        codes_set(h, "dataTime", spec.data_time)

        # Grid definition section (template 0, lat/lon).
        codes_set(h, "gridDefinitionTemplateNumber", 0)
        codes_set(h, "Ni", spec.ni)
        codes_set(h, "Nj", spec.nj)
        codes_set(h, "latitudeOfFirstGridPointInDegrees", spec.first_lat)
        codes_set(h, "longitudeOfFirstGridPointInDegrees", spec.first_lon)
        codes_set(h, "iDirectionIncrementInDegrees", spec.delta_lon)
        codes_set(h, "jDirectionIncrementInDegrees", spec.delta_lat)
        codes_set(h, "scanningMode", 0)

        # Product definition template.
        codes_set(h, "productDefinitionTemplateNumber", spec.pdt)
        codes_set(h, "parameterCategory", spec.parameter_category)
        codes_set(h, "parameterNumber", spec.parameter_number)

        # Fixed surfaces.
        codes_set(h, "typeOfFirstFixedSurface", spec.first_surface_type)
        if spec.first_scale != MISSING:
            codes_set(h, "scaleFactorOfFirstFixedSurface", spec.first_scale)
            codes_set(h, "scaledValueOfFirstFixedSurface", spec.first_scaled)
        codes_set(h, "typeOfSecondFixedSurface", spec.second_surface_type)
        if spec.second_surface_type != MISSING:
            codes_set(h, "scaleFactorOfSecondFixedSurface", spec.second_scale)
            codes_set(h, "scaledValueOfSecondFixedSurface",
                      spec.second_scaled)

        # Ensemble membership.
        if spec.pdt in (1, 11):
            assert spec.ensemble_type is not None
            codes_set(h, "typeOfEnsembleForecast", spec.ensemble_type)
            codes_set(h, "perturbationNumber", spec.perturbation_number)

        # Timing.
        codes_set(h, "indicatorOfUnitOfTimeRange", spec.time_unit)
        codes_set(h, "forecastTime", spec.forecast_time)

        if spec.pdt in (8, 11):
            # The numeric PDT already selects a statistical process template;
            # avoid build-specific stepType concept strings and set the
            # statistical scalars directly.
            codes_set(h, "numberOfTimeRange", spec.number_of_time_range)
            codes_set(h, "typeOfTimeIncrement", spec.type_of_time_increment)
            codes_set(h, "indicatorOfUnitForTimeIncrement",
                      spec.time_increment_unit)
            codes_set(h, "timeIncrement", spec.time_increment)
            codes_set(h, "numberOfMissingInStatisticalProcess",
                      spec.number_missing)
            try:
                codes_set(h, "statisticalProcess", spec.statistical_process)
            except Exception:
                codes_set(h, "typeOfStatisticalProcessing",
                          spec.statistical_process)
            codes_set(h, "indicatorOfUnitForTimeRange", spec.time_range_unit)
            codes_set(h, "lengthOfTimeRange", spec.length_of_time_range)
            codes_set(h, "yearOfEndOfOverallTimeInterval", spec.end_year)
            codes_set(h, "monthOfEndOfOverallTimeInterval", spec.end_month)
            codes_set(h, "dayOfEndOfOverallTimeInterval", spec.end_day)
            codes_set(h, "hourOfEndOfOverallTimeInterval", spec.end_hour)
            codes_set(h, "minuteOfEndOfOverallTimeInterval", spec.end_minute)
            codes_set(h, "secondOfEndOfOverallTimeInterval", spec.end_second)
        # PDT 0/1 are instantaneous point-in-time templates by construction;
        # no stepType concept string is needed.

        # Packing (section 5); GDS is untouched so grid identity is stable.
        codes_set(h, "bitsPerValue", spec.bits_per_value)

        # Tiny synthetic data section.
        npoints = spec.ni * spec.nj
        if spec.values is not None:
            arr = np.asarray(spec.values, dtype=np.float64)
        else:
            arr = (np.arange(npoints, dtype=np.float64)
                   + 1000.0 * spec.data_seed)
        codes_set_values(h, arr)

        return codes_get_message(h)
    finally:
        codes_release(h)


def write_file(path: Path, messages: list[bytes]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = b"".join(messages)
    path.write_bytes(blob)
    return {
        "relative_path": str(path).split(os.sep, 1)[-1]
        if os.sep in str(path) else str(path),
        "size_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def build_manifest(collection_id: str, root: Path,
                   entries: list[dict], created_at: str | None = None) -> dict:
    files = []
    for e in entries:
        files.append({
            "file_id": e["file_id"],
            "relative_path": e["relative_path"],
            "size_bytes": e["size_bytes"],
            "sha256": e["sha256"],
        })
    return {
        "collection_id": collection_id,
        "created_at": created_at,
        "files": files,
    }


def truncate_tail(message: bytes, cut: int) -> bytes:
    """Return a message with its final ``cut`` bytes removed."""

    return message[:-cut]


def append_junk(message: bytes, junk: bytes = b"JUNK") -> bytes:
    return message + junk
