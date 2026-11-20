"""ecCodes decoding of one verified hand-off message into cube coordinates.

This module is the *only* part of the whole project (alongside the test-only
fixture generator) that touches grid point values.  It is deliberately
restrictive:

* only GRIB2 grid definition template 3.0 - a regular latitude/longitude
  grid - is accepted; reduced grids, spectral fields, unstructured grids and
  every other template get a stable ``unsupported_grid_template`` reason;
* only the canonical row-major scanning mode (octet value 0) is accepted so
  the stored array order is unambiguously ``[nj, ni]``;
* the single message must decode completely through ecCodes; any failure
  becomes ``undecodable_message`` with the original exception as evidence;
* latitude/longitude coordinates are computed independently from the Section
  3 start/increment octets and then cross-checked against the coordinates
  ecCodes itself decodes; disagreement is never papered over.

No interpolation, unit conversion or re-projection happens here.  Values are
returned as the float64 array ecCodes decodes, preserving GRIB missing points
as IEEE NaN (the cube stores an explicit presence mask separately).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..errors import GribAsmError
from .errors import UnsupportedCubeMessage

# GRIB2 GDT 3.0 = latitude/longitude (also called "regular lat/lon").
REGULAR_LL_TEMPLATE = 0
# Scanning mode 0: points increase in i then j, west->east, north->south.
ROW_MAJOR_SCANNING_MODE = 0


@dataclass(frozen=True)
class GridGeometry:
    """Regular lat/lon grid definition in exact decoded form."""

    ni: int
    nj: int
    first_lat: float
    first_lon: float
    last_lat: float
    last_lon: float
    delta_lat: float
    delta_lon: float
    scanning_mode: int
    latitude: np.ndarray       # 1-D, length nj, degrees north
    longitude: np.ndarray      # 1-D, length ni, degrees east
    gdt_number: int

    def definition_doc(self) -> dict:
        return {
            "gridDefinitionTemplateNumber": self.gdt_number,
            "Ni": self.ni,
            "Nj": self.nj,
            "latitudeOfFirstGridPointInDegrees": self.first_lat,
            "longitudeOfFirstGridPointInDegrees": self.first_lon,
            "latitudeOfLastGridPointInDegrees": self.last_lat,
            "longitudeOfLastGridPointInDegrees": self.last_lon,
            "iDirectionIncrementInDegrees": self.delta_lon,
            "jDirectionIncrementInDegrees": self.delta_lat,
            "scanningMode": self.scanning_mode,
            "pointOrdering": "row-major [Nj, Ni], scan mode 0",
        }


@dataclass
class DecodedField:
    values: np.ndarray            # float64, shape (nj, ni); NaN at missing pts
    missing_points: int
    grid: GridGeometry
    raw_unit: str
    short_name: str | None
    name: str | None


def _codes_int(handle, key: str) -> int | None:
    from eccodes import codes_get, codes_is_defined
    try:
        if not codes_is_defined(handle, key):
            return None
        return int(codes_get(handle, key))
    except Exception:
        return None


def _codes_float(handle, key: str) -> float | None:
    from eccodes import codes_get, codes_is_defined
    try:
        if not codes_is_defined(handle, key):
            return None
        return float(codes_get(handle, key))
    except Exception:
        return None


def _codes_str(handle, key: str) -> str | None:
    from eccodes import codes_get, codes_is_defined
    try:
        if not codes_is_defined(handle, key):
            return None
        return str(codes_get(handle, key, ktype="str"))
    except Exception:
        return None


def _regular_coordinates(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1-D lat/lon coordinates independently of ecCodes arrays.

    Scan mode 0 walks west->east and north->south, so the longitude axis
    increases while the latitude axis decreases from the first grid point.
    Direction is taken from the decoded last-point scalars rather than
    assumed, keeping this construction independent of ecCodes' arrays while
    faithful to the Section 3 definition.
    """

    steps_j = np.arange(grid["Nj"], dtype=np.float64)
    steps_i = np.arange(grid["Ni"], dtype=np.float64)
    if grid["latitudeOfLastGridPointInDegrees"] >= \
            grid["latitudeOfFirstGridPointInDegrees"]:
        lats = (grid["latitudeOfFirstGridPointInDegrees"]
                + steps_j * grid["jDirectionIncrementInDegrees"])
    else:
        lats = (grid["latitudeOfFirstGridPointInDegrees"]
                - steps_j * grid["jDirectionIncrementInDegrees"])
    if grid["longitudeOfLastGridPointInDegrees"] >= \
            grid["longitudeOfFirstGridPointInDegrees"]:
        lons = (grid["longitudeOfFirstGridPointInDegrees"]
                + steps_i * grid["iDirectionIncrementInDegrees"])
    else:
        lons = (grid["longitudeOfFirstGridPointInDegrees"]
                - steps_i * grid["iDirectionIncrementInDegrees"])
    return lats, lons


def decode_message(handle) -> DecodedField:
    """Decode one open ecCodes handle into a :class:`DecodedField`."""

    gdt = _codes_int(handle, "gridDefinitionTemplateNumber")
    if gdt != REGULAR_LL_TEMPLATE:
        raise UnsupportedCubeMessage("unsupported_grid_template", {
            "gridDefinitionTemplateNumber": gdt,
            "supported": [REGULAR_LL_TEMPLATE],
        })

    scanning = _codes_int(handle, "scanningMode")
    if scanning != ROW_MAJOR_SCANNING_MODE:
        raise UnsupportedCubeMessage("unsupported_scanning_mode", {
            "scanningMode": scanning,
            "supported": [ROW_MAJOR_SCANNING_MODE],
        })

    needed = ("Ni", "Nj",
              "latitudeOfFirstGridPointInDegrees",
              "longitudeOfFirstGridPointInDegrees",
              "iDirectionIncrementInDegrees",
              "jDirectionIncrementInDegrees")
    grid_scalars: dict[str, float] = {}
    for key in needed:
        val = _codes_float(handle, key)
        if val is None:
            raise UnsupportedCubeMessage("undecodable_message", {
                "missing_scalar_key": key})
        grid_scalars[key] = val
    ni, nj = int(grid_scalars["Ni"]), int(grid_scalars["Nj"])
    if ni <= 0 or nj <= 0:
        raise UnsupportedCubeMessage("unsupported_grid_template", {
            "reason": "non-positive Ni/Nj", "Ni": ni, "Nj": nj})

    try:
        from eccodes import codes_get_array
        flat = np.asarray(codes_get_array(handle, "values"), dtype=np.float64)
    except Exception as exc:  # pragma: no cover - depends on libeccodes
        raise UnsupportedCubeMessage("undecodable_message", {
            "error": f"{type(exc).__name__}: {exc}"}) from exc

    if flat.size != ni * nj:
        raise UnsupportedCubeMessage("point_count_mismatch", {
            "decoded_points": int(flat.size), "Ni": ni, "Nj": nj,
            "expected": ni * nj})

    # Independent coordinate construction, then cross-check against the
    # coordinates ecCodes decodes from the same Section 3.
    last_lat = _codes_float(handle, "latitudeOfLastGridPointInDegrees")
    last_lon = _codes_float(handle, "longitudeOfLastGridPointInDegrees")
    if last_lat is None or last_lon is None:
        raise UnsupportedCubeMessage("undecodable_message", {
            "missing_scalar_key": "last grid point"})
    grid = GridGeometry(
        ni=ni, nj=nj,
        first_lat=grid_scalars["latitudeOfFirstGridPointInDegrees"],
        first_lon=grid_scalars["longitudeOfFirstGridPointInDegrees"],
        last_lat=last_lat, last_lon=last_lon,
        delta_lat=grid_scalars["jDirectionIncrementInDegrees"],
        delta_lon=grid_scalars["iDirectionIncrementInDegrees"],
        scanning_mode=scanning,
        latitude=np.empty(0), longitude=np.empty(0),
        gdt_number=int(gdt),
    )
    lats, lons = _regular_coordinates(grid.definition_doc())
    try:
        ecc_lats = np.asarray(
            codes_get_array(handle, "distinctLatitudes"), dtype=np.float64)
        ecc_lons = np.asarray(
            codes_get_array(handle, "distinctLongitudes"), dtype=np.float64)
    except Exception as exc:
        raise UnsupportedCubeMessage("undecodable_message", {
            "error": f"coordinates: {type(exc).__name__}: {exc}"}) from exc
    # Some builds return the unordered grid arrays; compare as sets within a
    # tight tolerance, and keep our independently ordered 1-D coordinates.
    if (ecc_lats.size != nj or ecc_lons.size != ni
            or not np.allclose(np.sort(ecc_lats), np.sort(lats), atol=1e-9)
            or not np.allclose(np.sort(ecc_lons), np.sort(lons), atol=1e-9)):
        raise UnsupportedCubeMessage("grid_coordinate_mismatch", {
            "computed_lat_range": [float(lats[0]), float(lats[-1])],
            "eccodes_lat_size": int(ecc_lats.size),
            "computed_lon_range": [float(lons[0]), float(lons[-1])],
            "eccodes_lon_size": int(ecc_lons.size),
        })

    grid = GridGeometry(
        ni=ni, nj=nj, first_lat=grid.first_lat, first_lon=grid.first_lon,
        last_lat=last_lat, last_lon=last_lon,
        delta_lat=grid.delta_lat, delta_lon=grid.delta_lon,
        scanning_mode=scanning, latitude=lats, longitude=lons,
        gdt_number=int(gdt))

    values = flat.reshape(nj, ni)
    missing_points = int(np.count_nonzero(np.isnan(values)))

    raw_unit = _codes_str(handle, "units") or ""
    short_name = _codes_str(handle, "shortName")
    name = _codes_str(handle, "name")
    return DecodedField(
        values=values, missing_points=missing_points, grid=grid,
        raw_unit=raw_unit, short_name=short_name, name=name)
