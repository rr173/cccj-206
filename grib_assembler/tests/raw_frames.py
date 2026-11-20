"""Helpers to build minimal valid GRIB2 byte streams without ecCodes.

Only the framing the scanner relies on is reproduced: Section 0 (16 octets),
Section 1, optional Section 2, Section 3 and the Section 8 end marker.  These
are not meteorologically meaningful, but they are byte-accurate boundary
fixtures for scanner tests and contain no data sections at all.
"""

from __future__ import annotations

import hashlib


def section(number: int, body: bytes = b"") -> bytes:
    length = 5 + len(body)
    return length.to_bytes(4, "big") + bytes([number]) + body


def raw_message(sections: list[bytes], discipline: int = 0,
                edition: int = 2, length_override: int | None = None,
                end_marker: bytes = b"7777") -> bytes:
    body = b"".join(sections)
    total = length_override if length_override is not None else 16 + len(body) + 4
    # Section 0 (16 octets): 'GRIB' (4) + reserved (2) + discipline (1)
    # + edition (1) + total length (8 big-endian).
    header = (b"GRIB" + b"\x00\x00"
              + bytes([discipline & 0xFF, edition])
              + total.to_bytes(8, "big"))
    return header + body + end_marker


def standard_sections(gds_body: bytes = b"GDS-PAYLOAD",
                      with_local: bool = False) -> list[bytes]:
    secs = [section(1, b"IDENT")]
    if with_local:
        secs.append(section(2, b"LOCAL"))
    secs.append(section(3, gds_body))
    return secs


def make_message(gds_body: bytes = b"GDS-PAYLOAD",
                 with_local: bool = False, **kw) -> bytes:
    return raw_message(standard_sections(gds_body, with_local), **kw)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
