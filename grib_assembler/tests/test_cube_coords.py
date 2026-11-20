"""Unit tests for cube coordinate keys and the conservative unit mapping."""

from __future__ import annotations

import pytest

from grib_asm.cube.coords import (
    level_key, member_key, slot_key, validity_bounds, variable_key,
)
from grib_asm.cube.units import (
    MAPPED, RAW_ONLY, resolve_units,
)
from grib_asm.identity import (
    Level, MemberId, VariableId, make_surface,
)
from grib_asm.timeutil import TimeSlot
from datetime import datetime, timezone


def test_variable_key_keeps_full_numeric_identity():
    a = VariableId(0, 0, 0, 29, 1, 7, 0)
    b = VariableId(0, 0, 99, 29, 1, 7, 0)   # different parameter number
    c = VariableId(0, 0, 0, 28, 1, 7, 0)    # different master table
    keys = {variable_key(a), variable_key(b), variable_key(c)}
    assert len(keys) == 3
    assert "p0" in variable_key(a) and "p99" in variable_key(b)
    assert "m29" in variable_key(a) and "m28" in variable_key(c)


def test_level_key_uses_exact_rational_and_missing_surface():
    ground = Level(first=make_surface(1, 0, 0), second=None)
    pressure = Level(first=make_surface(100, 0, 85000), second=None)
    layer = Level(first=make_surface(100, 0, 100000),
                  second=make_surface(100, 0, 85000))
    keys = {level_key(ground), level_key(pressure), level_key(layer)}
    assert len(keys) == 3
    assert "tNAvNA" in level_key(ground)
    assert "v85000d1" in level_key(pressure)


def test_member_key_distinguishes_kinds_and_numbers():
    det = MemberId(kind="deterministic")
    ctl = MemberId(kind="control", type_of_ensemble=3, number=0)
    p1 = MemberId(kind="perturbed", type_of_ensemble=1, number=1)
    p2 = MemberId(kind="perturbed", type_of_ensemble=1, number=2)
    assert member_key(det) == "deterministic"
    assert member_key(ctl) == "control"
    assert member_key(p1) == "perturbed_1"
    assert member_key(p2) == "perturbed_2"


def test_slot_keys_and_validity_bounds():
    instant = TimeSlot(0, 0, None)
    acc06 = TimeSlot(0, 21600, 1)
    acc36 = TimeSlot(10800, 21600, 1)
    assert slot_key(instant) == "instant_0"
    assert slot_key(acc06) == "stat1_0_21600"
    assert slot_key(acc36) == "stat1_10800_21600"

    ref = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert validity_bounds(ref, instant) == (
        "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
    assert validity_bounds(ref, acc06) == (
        "2026-01-01T00:00:00Z", "2026-01-01T06:00:00Z")
    assert validity_bounds(ref, acc36) == (
        "2026-01-01T03:00:00Z", "2026-01-01T06:00:00Z")


def test_unit_mapping_is_explicit_never_guessed():
    mapped_var = VariableId(0, 0, 0, 29, 1, 7, 0)
    info = resolve_units(mapped_var, "K")
    assert info.mapping_status == MAPPED
    assert info.cf_standard_name == "air_temperature"
    assert info.raw_unit == "K"

    unknown = VariableId(0, 5, 42, 29, 1, 7, 0)
    raw = resolve_units(unknown, "some-native-unit")
    assert raw.mapping_status == RAW_ONLY
    assert raw.cf_standard_name is None
    assert raw.cf_canonical_unit is None
    assert raw.raw_unit == "some-native-unit"

    # A different centre defeats even the otherwise matching table entry.
    other_centre = VariableId(0, 0, 0, 29, 1, 99, 0)
    assert resolve_units(other_centre, "K").mapping_status == RAW_ONLY
