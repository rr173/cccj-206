"""Unsupported grid / undecodable message handling for cube imports.

A message the verifier accepts structurally may still be outside the cube's
representable profile.  Such messages must surface a *stable* reason code and
must never be silently skipped while the cycle is reported complete.

The assembler's classifier deliberately ignores the grid template and scan
order (it only hashes Section 3), so messages on e.g. a rotated lat/lon grid
can flow through a real, fully verified hand-off package and must then be
rejected by the cube decoder - which is the boundary under test here.
"""

from __future__ import annotations

import io
from datetime import timezone

import numpy as np
import pytest

from grib_asm.cube.errors import UnsupportedCubeMessage
from grib_asm.cube.importer import import_package
from grib_asm.cube.store import CubeConfig
from grib_asm.gribbytes import locate_sections
from grib_asm.headers import (
    close_handle, open_handle_from_message_bytes,
)
from grib_asm.identity import grid_digest
from grib_asm.scanner import scan_file
from grib_asm.classify import classify_message

from fixturegen import FixtureSpec, encode_message
from cube_helpers import real_package_from_message
from e2e_helpers import HAS_ECCODES

pytestmark = pytest.mark.skipif(
    not HAS_ECCODES, reason="ecCodes required to encode/decode fixtures")

UTC = timezone.utc


def _gds_digest(raw: bytes) -> str:
    sections = locate_sections(raw)
    off, length = sections[3]
    return grid_digest(raw[off:off + length])


def _encode_rotated_ll(spec: FixtureSpec) -> bytes:
    from eccodes import (
        codes_grib_new_from_samples, codes_set, codes_set_values,
        codes_get_message, codes_release,
    )
    h = codes_grib_new_from_samples("GRIB2")
    try:
        # Identification section: explicit 2026 reference and the same
        # numeric identity the rest of the cube test suite uses, so the
        # package request built from the classified message is in range.
        codes_set(h, "discipline", spec.discipline)
        codes_set(h, "tablesVersion", spec.master_tables_version)
        codes_set(h, "localTablesVersion", spec.local_tables_version)
        codes_set(h, "centre", spec.centre)
        codes_set(h, "subCentre", spec.sub_centre)
        codes_set(h, "dataDate", spec.data_date)
        codes_set(h, "dataTime", spec.data_time)
        codes_set(h, "gridDefinitionTemplateNumber", 1)
        codes_set(h, "Ni", spec.ni)
        codes_set(h, "Nj", spec.nj)
        codes_set(h, "latitudeOfFirstGridPointInDegrees", spec.first_lat)
        codes_set(h, "longitudeOfFirstGridPointInDegrees", spec.first_lon)
        codes_set(h, "iDirectionIncrementInDegrees", spec.delta_lon)
        codes_set(h, "jDirectionIncrementInDegrees", spec.delta_lat)
        codes_set(h, "latitudeOfSouthernPoleInDegrees", 0.0)
        codes_set(h, "longitudeOfSouthernPoleInDegrees", 0.0)
        codes_set(h, "angleOfRotationInDegrees", 0.0)
        codes_set(h, "scanningMode", 0)
        codes_set(h, "productDefinitionTemplateNumber", spec.pdt)
        codes_set(h, "parameterCategory", spec.parameter_category)
        codes_set(h, "parameterNumber", spec.parameter_number)
        codes_set(h, "typeOfFirstFixedSurface", spec.first_surface_type)
        codes_set(h, "scaleFactorOfFirstFixedSurface", spec.first_scale)
        codes_set(h, "scaledValueOfFirstFixedSurface", spec.first_scaled)
        codes_set(h, "typeOfSecondFixedSurface", 255)
        codes_set(h, "forecastTime", spec.forecast_time)
        codes_set(h, "indicatorOfUnitOfTimeRange", spec.time_unit)
        codes_set(h, "bitsPerValue", spec.bits_per_value)
        n = spec.ni * spec.nj
        codes_set_values(
            h, np.arange(n, dtype=np.float64) + 1000.0 * spec.data_seed)
        return codes_get_message(h)
    finally:
        codes_release(h)


def test_unsupported_grid_template_fails_real_import(tmp_path):
    # GDT 3.1 rotated lat/lon: a verified package, not a regular lat/lon grid.
    spec = FixtureSpec(pdt=0)
    raw = _encode_rotated_ll(spec)
    dig = _gds_digest(raw)
    pkg = real_package_from_message(tmp_path, raw, dig, "rotated")

    # Direct decoder boundary: stable reason and explicit evidence.
    handle = open_handle_from_message_bytes(raw)
    try:
        with pytest.raises(UnsupportedCubeMessage) as exc:
            from grib_asm.cube.decode import decode_message
            decode_message(handle)
    finally:
        close_handle(handle)
    assert exc.value.reason == "unsupported_grid_template"
    assert exc.value.evidence["gridDefinitionTemplateNumber"] == 1

    # End-to-end: the whole cycle import fails and nothing is published.
    with pytest.raises(UnsupportedCubeMessage) as exc2:
        import_package(tmp_path / "cube", pkg, CubeConfig(batch_size=2))
    assert exc2.value.reason == "unsupported_grid_template"
    assert not (tmp_path / "cube").exists() or not list(
        (tmp_path / "cube" / ".cube" / "committed").glob("*.json"))


def test_unsupported_scanning_mode_has_stable_reason(tmp_path):
    from eccodes import (
        codes_grib_new_from_samples, codes_set, codes_set_values,
        codes_get_message, codes_release,
    )
    h = codes_grib_new_from_samples("GRIB2")
    try:
        codes_set(h, "discipline", 0)
        codes_set(h, "tablesVersion", 29)
        codes_set(h, "localTablesVersion", 1)
        codes_set(h, "centre", 7)
        codes_set(h, "subCentre", 0)
        codes_set(h, "dataDate", 20260101)
        codes_set(h, "dataTime", 0)
        codes_set(h, "gridDefinitionTemplateNumber", 0)
        codes_set(h, "Ni", 4)
        codes_set(h, "Nj", 3)
        # Bit 1 set: j points decrease south->north; not the canonical order.
        codes_set(h, "scanningMode", 0b01000000)
        codes_set(h, "productDefinitionTemplateNumber", 0)
        codes_set(h, "parameterCategory", 0)
        codes_set(h, "parameterNumber", 0)
        codes_set(h, "typeOfFirstFixedSurface", 100)
        codes_set(h, "scaleFactorOfFirstFixedSurface", 0)
        codes_set(h, "scaledValueOfFirstFixedSurface", 85000)
        codes_set(h, "typeOfSecondFixedSurface", 255)
        codes_set(h, "forecastTime", 0)
        codes_set(h, "indicatorOfUnitOfTimeRange", 1)
        codes_set(h, "bitsPerValue", 24)
        codes_set_values(h, np.arange(12, dtype=np.float64))
        raw = codes_get_message(h)
    finally:
        codes_release(h)

    handle = open_handle_from_message_bytes(raw)
    try:
        with pytest.raises(UnsupportedCubeMessage) as exc:
            from grib_asm.cube.decode import decode_message
            decode_message(handle)
    finally:
        close_handle(handle)
    assert exc.value.reason == "unsupported_scanning_mode"

    # The same stable reason aborts the full import via a verified package.
    dig = _gds_digest(raw)
    pkg = real_package_from_message(tmp_path, raw, dig, "scanmode")
    with pytest.raises(UnsupportedCubeMessage) as exc2:
        import_package(tmp_path / "cube2", pkg)
    assert exc2.value.reason == "unsupported_scanning_mode"


def test_unsupported_pdt_classifier_reason_is_stable():
    # PDT 4.2 never reaches a package; prove the classifier's stable reason.
    spec = FixtureSpec(pdt=2)
    raw = encode_message(spec)
    from grib_asm.classify import classify_message
    scan = scan_file(io.BytesIO(raw), len(raw))
    h = open_handle_from_message_bytes(raw)
    try:
        rec = classify_message(h, raw, scan.messages[0], "f", "c")
    finally:
        close_handle(h)
    assert rec.status == "unsupported"
    assert rec.reason == "unsupported_pdt"


def test_undecodable_message_reason_is_stable(monkeypatch):
    import eccodes
    from grib_asm.cube import decode as decode_mod

    class Boom:
        pass

    def boom(handle, key):
        raise RuntimeError("libeccodes failure")

    monkeypatch.setattr(eccodes, "codes_get_array", boom)
    with pytest.raises(UnsupportedCubeMessage) as exc:
        decode_mod.decode_message(Boom())
    # Scalar reads may also fail; either way the result is a stable code,
    # never a silent skip.
    assert exc.value.reason in (
        "unsupported_grid_template", "undecodable_message")
