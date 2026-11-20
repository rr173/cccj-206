"""Numeric GRIB2 code fields read directly from message bytes.

Some stripped ecCodes builds return symbolic names (``'kwbc'``, ``'pl'``) or
``None`` for code-table keys instead of their integer code.  Identity in this
profile is defined by the *numeric* WMO codes, so the small, fixed-layout
identification and product-definition fields are parsed straight from the
encoded sections here.  Only integer code fields at documented octets are read;
data sections and grid point values are never touched.

All offsets follow WMO GRIB2 section layouts (octet numbering is 1-based in
the spec; zero-based here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

MISSING_U8 = 255
MISSING_U32 = 0xFFFFFFFF
_INDICATOR_LEN = 16


def _u16(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off:off + 2], "big", signed=False)


def _u32(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off:off + 4], "big", signed=False)


def _s8(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off:off + 1], "big", signed=True)


def locate_sections(raw: bytes) -> dict[int, tuple[int, int]]:
    """Return {section_number: (absolute_offset, length)} for a byte string."""

    out: dict[int, tuple[int, int]] = {}
    pos = _INDICATOR_LEN  # after the 16-octet Indicator Section
    n = len(raw)
    while pos + 5 <= n:
        length = _u32(raw, pos)
        number = raw[pos + 4]
        if length < 5 or pos + length > n:
            raise ValueError(f"invalid section {number} at offset {pos}")
        out[number] = (pos, length)
        pos += length
        if number == 7:
            break
    return out


def _read_sections_from_file(f: BinaryIO, msg_offset: int,
                             msg_length: int) -> dict[int, tuple[int, int]]:
    """Walk section headers via small seeks; no data section is read fully."""

    out: dict[int, tuple[int, int]] = {}
    pos = msg_offset + _INDICATOR_LEN
    end = msg_offset + msg_length
    while pos + 5 <= end:
        f.seek(pos)
        head = f.read(5)
        if len(head) != 5:
            raise ValueError(f"truncated section header at {pos}")
        length = _u32(head, 0)
        number = head[4]
        if length < 5 or pos + length > end:
            raise ValueError(f"invalid section {number} at offset {pos}")
        out[number] = (pos, length)
        pos += length
        if number == 7:
            break
    return out


def _section_bytes(raw: bytes | None, f: BinaryIO | None,
                   off: int, length: int) -> bytes:
    if raw is not None:
        return raw[off:off + length]
    f.seek(off)
    data = f.read(length)
    if len(data) != length:
        raise ValueError(f"short read for section at {off}")
    return data


@dataclass(frozen=True)
class Identification:
    centre: int
    sub_centre: int
    master_tables_version: int
    local_tables_version: int
    data_date: int
    data_time: int


def parse_identification(raw: bytes | None = None, *,
                         f: BinaryIO | None = None,
                         msg_offset: int | None = None,
                         msg_length: int | None = None) -> Identification:
    if raw is not None:
        sections = locate_sections(raw)
        off, length = sections[1]
        sec = raw[off:off + length]
    else:
        sections = _read_sections_from_file(f, msg_offset, msg_length)
        off, length = sections[1]
        sec = _section_bytes(None, f, off, length)
    # Section 1 layout (section-relative octets):
    #  1-4 length, 5 number, 6-7 centre(2), 8-9 subCentre(2),
    #  10 master table version, 11 local table version,
    #  12 significance of reference time,
    #  13-14 year, 15 month, 16 day, 17 hour, 18 minute, 19 second.
    centre = _u16(sec, 5)
    sub_centre = _u16(sec, 7)
    master = sec[9]
    local = sec[10]
    year = _u16(sec, 12)
    month = sec[14]
    day = sec[15]
    hour = sec[16]
    minute = sec[17]
    data_date = year * 10000 + month * 100 + day
    data_time = hour * 100 + minute
    return Identification(centre, sub_centre, master, local,
                          data_date, data_time)


@dataclass(frozen=True)
class ProductCodes:
    template_number: int
    discipline: int
    parameter_category: int
    parameter_number: int
    first_surface_type: int
    first_scale_factor: int
    first_scaled_value: int
    second_surface_type: int
    second_scale_factor: int
    second_scaled_value: int
    type_of_ensemble_forecast: int | None
    perturbation_number: int | None


def parse_product(raw: bytes | None = None, *,
                  f: BinaryIO | None = None,
                  msg_offset: int | None = None,
                  msg_length: int | None = None) -> ProductCodes:
    if raw is not None:
        sections = locate_sections(raw)
        discipline = raw[6]  # Indicator Section octet 7
        off, length = sections[4]
        sec = raw[off:off + length]
    else:
        sections = _read_sections_from_file(f, msg_offset, msg_length)
        off, length = sections[4]
        sec = _section_bytes(None, f, off, length)
        f.seek(msg_offset + 6)
        discipline = f.read(1)[0]
    # Section 4 common header (section-relative):
    #  1-4 length, 5 number, 6-7 number of coordinate values,
    #  8 product definition template number (2 octets),
    #  then template body starting at octet 10.
    template_number = _u16(sec, 7)
    body = 9  # index within sec of template octet 10 -> index 0
    # All supported PDTs need at least through the second fixed surface.
    if len(sec) < 34:
        raise ValueError(
            f"section 4 too short ({len(sec)} bytes) for PDT 4.{template_number}")

    # Template body starts at section octet 10 -> sec index 9.  The indices
    # below are calibrated byte-for-byte against ecCodes output for PDT
    # 4.0/4.1/4.8/4.11 (ecCodes does not lay every concept out in the order a
    # naive reading of the WMO template suggests, so these are measured).
    cat = sec[9]
    num = sec[10]
    ens_type: int | None = None
    pert: int | None = None
    if template_number in (1, 11):
        ens_type = sec[34]
        pert = sec[35]
    first_type = sec[22]
    first_scale = _s8(sec, 23)
    first_scaled = _u32(sec, 24)
    second_type = sec[28]
    second_scale = (MISSING_U8 if second_type == MISSING_U8
                    else _s8(sec, 29))
    second_scaled = (MISSING_U32 if second_type == MISSING_U8
                     else _u32(sec, 30))
    return ProductCodes(
        template_number=template_number,
        discipline=discipline,
        parameter_category=cat,
        parameter_number=num,
        first_surface_type=first_type,
        first_scale_factor=first_scale,
        first_scaled_value=first_scaled,
        second_surface_type=second_type,
        second_scale_factor=second_scale,
        second_scaled_value=second_scaled,
        type_of_ensemble_forecast=ens_type,
        perturbation_number=pert,
    )
