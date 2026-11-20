"""Pytest-free shared helpers for ecCodes fixture and demo generation.

The test suite imports the same constants/spec builder so the committed demo
data and the in-test fixtures stay identical.
"""

from __future__ import annotations

import io

from fixturegen import FixtureSpec, encode_message
from grib_asm.scanner import scan_file
from grib_asm.identity import grid_digest

# (start_hour, end_hour) accumulation slots for the two-member drill.
CONTROL_SLOTS = ((0, 3), (3, 6))
PERTURBED_SLOTS = ((0, 3), (3, 6))

DISC, CAT, NUM = 0, 0, 0
CENTRE, SUBCENTRE = 7, 0
MTV, LTV = 29, 1
LEVEL_TYPE = 100
LEVEL_SCALED = 85000
REFERENCE_DATE = 20260101


def stat_spec(reference_hour, member_kind, start_h, end_h, process=1,
              data_seed=1, bits=16, parameter_number=NUM,
              grid_overrides=None, time_unit=1, length_unit=1):
    if member_kind == "control":
        pdt, ens_type, ens_num = 11, 3, 0
    elif member_kind == "perturbed":
        pdt, ens_type, ens_num = 11, 1, 1
    else:
        pdt, ens_type, ens_num = 8, None, None

    kw = dict(
        name=f"{reference_hour:02d}-{member_kind}-{start_h}-{end_h}",
        pdt=pdt,
        discipline=DISC, parameter_category=CAT,
        parameter_number=parameter_number,
        master_tables_version=MTV, local_tables_version=LTV,
        centre=CENTRE, sub_centre=SUBCENTRE,
        data_date=REFERENCE_DATE, data_time=reference_hour * 100,
        first_surface_type=LEVEL_TYPE, first_scale=0,
        first_scaled=LEVEL_SCALED,
        ensemble_type=ens_type, perturbation_number=ens_num,
        forecast_time=start_h, time_unit=time_unit,
        statistical_process=process,
        length_of_time_range=(end_h - start_h),
        time_range_unit=length_unit,
        end_year=2026, end_month=1, end_day=1,
        end_hour=reference_hour + end_h, end_minute=0, end_second=0,
        bits_per_value=bits, data_seed=data_seed,
    )
    if grid_overrides:
        kw.update(grid_overrides)
    return FixtureSpec(**kw)


def instant_spec(reference_hour, member_kind="deterministic",
                 forecast_hour=0, data_seed=1, **overrides):
    if member_kind == "control":
        pdt, ens_type, ens_num = 1, 3, 0
    elif member_kind == "perturbed":
        pdt, ens_type, ens_num = 1, 1, 1
    else:
        pdt, ens_type, ens_num = 0, None, None
    kw = dict(
        name=f"instant-{reference_hour:02d}-{member_kind}-{forecast_hour}",
        pdt=pdt,
        discipline=DISC, parameter_category=CAT, parameter_number=NUM,
        master_tables_version=MTV, local_tables_version=LTV,
        centre=CENTRE, sub_centre=SUBCENTRE,
        data_date=REFERENCE_DATE, data_time=reference_hour * 100,
        first_surface_type=LEVEL_TYPE, first_scale=0,
        first_scaled=LEVEL_SCALED,
        ensemble_type=ens_type, perturbation_number=ens_num,
        forecast_time=forecast_hour, time_unit=1,
        data_seed=data_seed,
    )
    kw.update(overrides)
    return FixtureSpec(**kw)


def grid_digest_of(spec_kwargs=None):
    spec = FixtureSpec(**(spec_kwargs or {}))
    raw = encode_message(spec)
    res = scan_file(io.BytesIO(raw), len(raw))
    return grid_digest(res.messages[0].gds_bytes), raw
