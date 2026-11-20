"""Time unit normalisation and slot tests."""

import pytest

from grib_asm.errors import UnsupportedTemplate
from grib_asm.timeutil import (
    TimeSlot, iso, reference_datetime, unit_to_seconds,
)


def test_allowed_units_to_exact_seconds():
    assert unit_to_seconds(13, 5) == 5          # seconds
    assert unit_to_seconds(0, 3) == 180         # minutes
    assert unit_to_seconds(1, 6) == 21600       # hours
    assert unit_to_seconds(2, 2) == 172800      # days


def test_minute_hour_equivalence():
    # 60 minutes and 1 hour are the same exact offset.
    assert unit_to_seconds(0, 60) == unit_to_seconds(1, 1) == 3600
    assert unit_to_seconds(0, 180) == unit_to_seconds(1, 3) == 10800


@pytest.mark.parametrize("bad_unit", [3, 4, 5, 6, 7, 10, 11, 12])
def test_unsupported_calendar_units(bad_unit):
    with pytest.raises(UnsupportedTemplate) as ei:
        unit_to_seconds(bad_unit, 1)
    assert ei.value.reason == "unsupported_time_unit"


def test_reference_datetime_utc():
    dt = reference_datetime(20260101, 600)
    assert iso(dt) == "2026-01-01T06:00:00Z"
    dt0 = reference_datetime(20261231, 0)
    assert iso(dt0) == "2026-12-31T00:00:00Z"


def test_instant_slot_requires_start_equals_end():
    TimeSlot(3600, 3600, None)  # valid instant
    with pytest.raises(ValueError):
        TimeSlot(0, 3600, None)


def test_statistical_slot_requires_range():
    TimeSlot(0, 10800, 1)  # 0-3h accumulation
    with pytest.raises(ValueError):
        TimeSlot(3600, 3600, 1)


def test_slot_key_distinguishes_same_endpoint_ranges():
    # 0-6h and 3-6h share an endpoint but are different products.
    a = TimeSlot(0, 21600, 1)
    b = TimeSlot(10800, 21600, 1)
    c = TimeSlot(0, 10800, 1)
    assert a.key() != b.key()
    assert b.key() != c.key()
    assert a.key() != c.key()


def test_different_statistics_are_different_slots():
    avg = TimeSlot(0, 10800, 0)
    acc = TimeSlot(0, 10800, 1)
    assert avg.key() != acc.key()
