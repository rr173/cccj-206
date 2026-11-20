"""ecCodes header access layer.

Only *scalar* header keys are ever requested.  Grid point arrays (``values``,
``packedValues``, ...) are explicitly refused, and an open message handle is
closed before opening the next one, so memory stays flat regardless of grid
size.  This module performs no decoding of data values and no re-encoding.
"""

from __future__ import annotations

from typing import Any, Iterable

from .errors import GribAsmError

# Keys that would force decoding of the full data section.  Refuse them even
# if a caller asks by accident.
_FORBIDDEN_ARRAY_KEYS = frozenset({
    "values", "packedValues", "packedData", "codedValues",
    "latitudes", "longitudes", "distinctLatitudes", "distinctLongitudes",
    "bits", "bitmap", "unpackedData",
})

# Scalar header keys we rely on.  Kept explicit so the audit trail is stable.
_SCALAR_KEYS = (
    "edition", "discipline",
    "masterTablesVersionNumber", "localTablesVersionNumber",
    "centre", "subCentre",
    "dataDate", "dataTime",
    "productDefinitionTemplateNumber",
    "parameterCategory", "parameterNumber",
    "typeOfFirstFixedSurface", "scaleFactorOfFirstFixedSurface",
    "scaledValueOfFirstFixedSurface",
    "typeOfSecondFixedSurface", "scaleFactorOfSecondFixedSurface",
    "scaledValueOfSecondFixedSurface",
    "typeOfEnsembleForecast", "perturbationNumber",
    "forecastTime", "indicatorOfUnitOfTimeRange",
    "numberOfTimeRange", "statisticalProcess",
    "typeOfStatisticalProcessing",
    "typeOfTimeIncrement", "timeIncrement",
    "numberOfMissingInStatisticalProcess",
    "lengthOfTimeRange", "indicatorOfUnitForTimeRange",
    "indicatorOfUnitForTimeIncrement",
    "yearOfEndOfOverallTimeInterval", "monthOfEndOfOverallTimeInterval",
    "dayOfEndOfOverallTimeInterval", "hourOfEndOfOverallTimeInterval",
    "minuteOfEndOfOverallTimeInterval", "secondOfEndOfOverallTimeInterval",
    "year", "month", "day", "hour", "minute", "second",
    "gridDefinitionTemplateNumber",
)


class HeaderAccessError(GribAsmError):
    pass


def _check_key(key: str) -> None:
    if key in _FORBIDDEN_ARRAY_KEYS:
        raise HeaderAccessError(
            f"refusing array/data key {key!r}: grid point decoding is disabled")


class MessageHeader:
    """Lazily read scalar view over one open ecCodes message handle."""

    def __init__(self, handle: Any):
        self._h = handle

    def int(self, key: str, default: int | None = None) -> int:
        _check_key(key)
        from eccodes import codes_get
        try:
            value = codes_get(self._h, key)
        except Exception:
            if default is not None:
                return default
            raise HeaderAccessError(f"missing integer key {key!r}")
        # Some ecCodes builds return None for an explicit ktype="int"; the
        # default call returns the native scalar, which is normally an int.
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            # A code-table key may come back as a symbolic name rather than a
            # number.  Re-request the numeric form; if unavailable this key is
            # genuinely not numerically defined in the local tables.
            try:
                num = codes_get(self._h, key, ktype="int")
                if isinstance(num, int):
                    return int(num)
            except Exception:
                pass
        if default is not None:
            return default
        raise HeaderAccessError(
            f"key {key!r} did not yield an integer, got {value!r}")

    def str(self, key: str, default: str | None = None) -> str:
        _check_key(key)
        from eccodes import codes_get
        try:
            return str(codes_get(self._h, key, ktype="str"))
        except Exception:
            if default is not None:
                return default
            raise HeaderAccessError(f"missing string key {key!r}")

    def has(self, key: str) -> bool:
        from eccodes import codes_is_defined
        try:
            return bool(codes_is_defined(self._h, key))
        except Exception:
            return False

    # Some ecCodes builds rename GRIB2 section-1 table keys.  Aliases let the
    # profile keep its canonical WMO names while reading what the local build
    # actually exposes, without inventing values.
    ALIASES = {
        "masterTablesVersionNumber": ("masterTablesVersionNumber",
                                      "tablesVersion"),
        "localTablesVersionNumber": ("localTablesVersionNumber",
                                     "localTablesVersion"),
    }

    def int_any(self, key: str, default: int | None = None) -> int:
        """Read an integer, trying build-specific aliases in order."""

        for candidate in self.ALIASES.get(key, (key,)):
            if self.has(candidate):
                return self.int(candidate)
        if default is not None:
            return default
        raise HeaderAccessError(
            f"missing integer key {key!r} (aliases tried: "
            f"{self.ALIASES.get(key, (key,))})")

    def snapshot(self, keys: Iterable[str] | None = None) -> dict:
        """Return a JSON-safe dict of scalar header keys for the audit trail."""

        out: dict[str, Any] = {}
        for key in list(keys) if keys is not None else _SCALAR_KEYS:
            # Resolve build-specific aliases but always store the canonical
            # WMO key name so the audit trail is portable across builds.
            if key in self.ALIASES:
                try:
                    out[key] = self.int_any(key)
                except HeaderAccessError:
                    pass
                continue
            if not self.has(key):
                continue
            try:
                out[key] = self.int(key)
            except HeaderAccessError:
                continue
            except Exception:
                try:
                    out[key] = self.str(key)
                except Exception:
                    continue
        return out


def open_handle_from_message_bytes(raw: bytes) -> Any:
    """Open one fully-in-memory encoded message.

    Used only on the single message currently being classified (and only for
    its headers).  The handle is closed by the caller.
    """

    from eccodes import codes_new_from_message
    return codes_new_from_message(raw)


def open_handle_from_file(f: Any, offset: int) -> Any:
    """Open one message starting at ``offset`` of an open binary file.

    ecCodes reads only what it needs from the file; grid point values are not
    decoded unless an array key is requested, which this package never does.
    The handle is closed by the caller.
    """

    from eccodes import codes_grib_new_from_file
    f.seek(offset)
    handle = codes_grib_new_from_file(f)
    if handle is None:
        raise HeaderAccessError(f"no GRIB message at file offset {offset}")
    return handle


def close_handle(handle: Any) -> None:
    from eccodes import codes_release
    codes_release(handle)
