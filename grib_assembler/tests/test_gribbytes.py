"""Direct byte-level GRIB2 section parser tests (real ecCodes fixtures)."""

import pytest

from grib_asm.gribbytes import (
    parse_identification, parse_product, locate_sections,
)
from fixturegen import FixtureSpec, encode_message
from e2e_helpers import HAS_ECCODES

pytestmark = pytest.mark.skipif(
    not HAS_ECCODES, reason="ecCodes required to build fixtures")


def test_identification_fields():
    raw = encode_message(FixtureSpec(
        centre=7, sub_centre=0, master_tables_version=29,
        local_tables_version=1, data_date=20260101, data_time=600))
    ident = parse_identification(raw)
    assert ident.centre == 7
    assert ident.sub_centre == 0
    assert ident.master_tables_version == 29
    assert ident.local_tables_version == 1
    assert (ident.data_date, ident.data_time) == (20260101, 600)


def test_product_instant_deterministic():
    raw = encode_message(FixtureSpec(
        pdt=0, discipline=0, parameter_category=0, parameter_number=0,
        first_surface_type=100, first_scaled=85000))
    prod = parse_product(raw)
    assert prod.template_number == 0
    assert prod.discipline == 0
    assert prod.first_surface_type == 100
    assert prod.first_scaled_value == 85000
    assert prod.second_surface_type == 255
    assert prod.type_of_ensemble_forecast is None


@pytest.mark.parametrize("pdt,kind,et,num", [
    (1, "control", 3, 0), (1, "perturbed", 1, 1),
    (11, "control", 3, 0), (11, "perturbed", 1, 1),
])
def test_product_ensemble_codes(pdt, kind, et, num):
    spec = FixtureSpec(
        pdt=pdt, ensemble_type=et, perturbation_number=num,
        forecast_time=3 if pdt == 1 else 0)
    raw = encode_message(spec)
    prod = parse_product(raw)
    assert prod.template_number == pdt
    assert prod.type_of_ensemble_forecast == et
    assert prod.perturbation_number == num


def test_different_parameter_number_distinguished():
    a = parse_product(encode_message(FixtureSpec(parameter_number=0)))
    b = parse_product(encode_message(FixtureSpec(parameter_number=99)))
    assert a.parameter_number == 0
    assert b.parameter_number == 99
    assert a.parameter_number != b.parameter_number


def test_ground_vs_zero_pressure_distinguished():
    # Ground surface type 1 vs isobaric 100, both with scaled value 0.
    ground = parse_product(encode_message(FixtureSpec(
        first_surface_type=1, first_scaled=0)))
    zero_pa = parse_product(encode_message(FixtureSpec(
        first_surface_type=100, first_scaled=0)))
    assert ground.first_surface_type != zero_pa.first_surface_type
    assert ground.first_scaled_value == zero_pa.first_scaled_value == 0


def test_locate_sections_has_required_sections():
    raw = encode_message(FixtureSpec(pdt=8))
    sections = locate_sections(raw)
    assert {1, 3, 4, 5, 7}.issubset(sections.keys())
