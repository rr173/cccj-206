"""Identity: rational levels, missing surfaces, members, grid digest."""

from fractions import Fraction

from grib_asm.identity import (
    FixedSurface, Level, MemberId, VariableId, BusinessKey,
    deterministic_member, ensemble_member, grid_digest, make_surface,
    scaled_value, MISSING_BYTE, MISSING_LONG,
)
from grib_asm.timeutil import TimeSlot


def test_scaled_value_exact_rational():
    assert scaled_value(0, 85000) == Fraction(85000)
    assert scaled_value(2, 150) == Fraction(150, 100)
    assert scaled_value(-1, 850) == Fraction(8500)


def test_surface_preserves_raw_factor_and_scaled_value():
    # The audit record must round-trip the original GRIB2 octets, not a
    # re-reduced fraction that loses the declared scale factor.
    surf = make_surface(102, 2, 150)
    doc = surf.to_dict()
    assert doc["scale_factor"] == 2
    assert doc["scaled_value"] == 150
    assert doc["rational"] == (3, 2)
    assert surf.value == Fraction(150, 100)


def test_missing_surface_is_none_not_zero():
    assert scaled_value(MISSING_BYTE, 0) is None
    assert scaled_value(0, MISSING_LONG) is None
    surf = make_surface(255, MISSING_BYTE, MISSING_LONG)
    assert surf.value is None


def test_surface_differs_from_zero_pressure():
    ground = make_surface(1, 0, 0)       # ground/surface, value 0
    zero_pa = make_surface(100, 0, 0)    # 0 Pa isobaric surface
    assert ground != zero_pa
    # Different surface types never collapse even with equal values.
    assert ground.type != zero_pa.type


def test_level_keeps_undefined_second_surface():
    first = make_surface(100, 0, 85000)
    lvl = Level(first=first, second=None)
    assert lvl.second is None
    doc = lvl.to_dict()
    assert doc["second"] is None


def test_level_with_two_surfaces_roundtrip():
    first = make_surface(100, 0, 100000)
    second = make_surface(100, 0, 85000)
    lvl = Level(first=first, second=second)
    doc = lvl.to_dict()
    assert doc["second"]["type"] == 100


def test_variable_identity_uses_full_code_set():
    a = VariableId(0, 0, 0, 29, 1, 7, 0)
    b = VariableId(0, 0, 1, 29, 1, 7, 0)   # different parameterNumber
    c = VariableId(0, 0, 0, 29, 1, 7, 99)  # different subCentre
    assert a != b
    assert a != c


def test_member_kinds():
    det = deterministic_member()
    ctrl = ensemble_member(3, 0)
    pert = ensemble_member(1, 4)
    assert det.kind == "deterministic"
    assert ctrl.kind == "control"
    assert pert.kind == "perturbed" and pert.number == 4
    # Deterministic is its own member kind, not silently a control member.
    assert MemberId("deterministic") != MemberId("control", 3, 0)


def test_grid_digest_stable_and_packing_independent():
    gds = b"section-three-bytes"
    d1 = grid_digest(gds)
    d2 = grid_digest(gds)
    assert d1 == d2
    assert grid_digest(gds + b"") == d1
    assert grid_digest(b"other-gds") != d1


def test_business_key_includes_all_dimensions_and_stable():
    var = VariableId(0, 0, 0, 29, 1, 7, 0)
    lvl = Level(make_surface(100, 0, 85000), None)
    mem = deterministic_member()
    slot = TimeSlot(0, 10800, 1)
    k1 = BusinessKey("2026-01-01T00:00:00Z", "griddigest", var, lvl, mem, slot)
    k2 = BusinessKey("2026-01-01T00:00:00Z", "griddigest", var, lvl, mem, slot)
    assert k1.digest() == k2.digest()
    # Different grid -> different key even if the rest matches.
    k3 = BusinessKey("2026-01-01T00:00:00Z", "other-grid", var, lvl, mem, slot)
    assert k1.digest() != k3.digest()
    # Different reference time -> different key.
    k4 = BusinessKey("2026-01-01T06:00:00Z", "griddigest", var, lvl, mem, slot)
    assert k1.digest() != k4.digest()
