"""Time handling: exact integer seconds, UTC reference time, slot identity.

Only GRIB2 code table 4.4 units that map to an exact whole number of seconds
are accepted: second (13), minute (0), hour (1), day (2).  Composite or
calendar units (3/6/12 hours, month, year, decade, normal, ...) are reported as
unsupported rather than approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .errors import UnsupportedTemplate

# GRIB2 Table 4.4 - Indicator of unit of time range.  Only exact, fixed-length
# units are permitted by this profile.
UNIT_SECONDS = {
    13: 1,    # second
    0: 60,    # minute
    1: 3600,  # hour
    2: 86400,  # day
}


def unit_to_seconds(unit: int, value: int) -> int:
    """Convert ``value`` in GRIB2 time unit ``unit`` to exact integer seconds."""

    if unit not in UNIT_SECONDS:
        raise UnsupportedTemplate("unsupported_time_unit", {
            "indicator_of_unit_of_time_range": unit,
            "value": value,
        })
    return int(value) * UNIT_SECONDS[unit]


def reference_datetime(data_date: int, data_time: int) -> datetime:
    """Build the complete UTC reference datetime from GRIB2 date/time keys.

    ``data_date`` is YYYYMMDD and ``data_time`` is HHMM.  Seconds are always
    zero in GRIB2 section 1 reference time.
    """

    yyyy = data_date // 10000
    mm = (data_date // 100) % 100
    dd = data_date % 100
    hour = data_time // 100
    minute = data_time % 100
    return datetime(yyyy, mm, dd, hour, minute, 0, tzinfo=timezone.utc)


def explicit_datetime(year: int, month: int, day: int,
                      hour: int, minute: int, second: int) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def iso(dt: datetime) -> str:
    """Canonical UTC string ending in ``Z``."""

    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class TimeSlot:
    """A request slot: relative offsets from the reference time.

    For an instantaneous product ``start_seconds == end_seconds`` and
    ``statistic`` is ``None``.  Statistical slots carry the GRIB2 statistical
    process code (table 4.10, restricted here to 0/1/2/3).
    """

    start_seconds: int
    end_seconds: int
    statistic: int | None = None

    def __post_init__(self):
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError(f"invalid time slot: {self}")
        if self.statistic is None and self.start_seconds != self.end_seconds:
            raise ValueError(
                f"instant slot must have start==end, got {self}")
        if self.statistic is not None and self.start_seconds == self.end_seconds:
            raise ValueError(
                f"statistical slot must span a non-empty range, got {self}")

    @property
    def is_instant(self) -> bool:
        return self.statistic is None

    def key(self) -> str:
        if self.is_instant:
            return f"instant@{self.start_seconds}"
        return f"stat{self.statistic}:{self.start_seconds}-{self.end_seconds}"

    def to_dict(self) -> dict:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "statistic": self.statistic,
        }


def slot_from_dict(d: dict) -> TimeSlot:
    return TimeSlot(
        start_seconds=int(d["start_seconds"]),
        end_seconds=int(d["end_seconds"]),
        statistic=None if d.get("statistic") is None else int(d["statistic"]),
    )
