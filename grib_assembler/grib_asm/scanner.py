"""Streaming GRIB2 message boundary scanner.

The scanner walks a continuous GRIB2 file in fixed-size buffers so its working
memory never grows with the file.  In a single pass it locates every complete
message (offset + length) and computes both a whole-file SHA-256 and a
per-message SHA-256.  A message that spans many buffers is hashed
incrementally; at most one buffer, a <=16-byte header carry and a 4-byte end
window are held.

It uses only the GRIB2 wrapper (discipline-independent):

* a message starts with the ASCII magic ``GRIB``, two reserved octets, the
  2-octet discipline, the edition octet and the 8-byte big-endian total
  message length (the 16-octet Indicator Section);
* a message ends with the 4-octet end marker ``7777``.

Truncation, a length that runs past EOF, a missing end marker, trailing junk
between messages or an unparseable header all make the *whole file* unusable
(the profile forbids salvaging individual messages from a broken boundary);
the caller keeps the structural evidence and excludes the file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import BinaryIO, Iterator

from .errors import BadFileError

MAGIC = b"GRIB"
END_MARKER = b"7777"
# GRIB2 Indicator Section (Section 0), 16 octets:
#   1-4  'GRIB'
#   5-6  reserved
#   7    discipline (1 octet)
#   8    GRIB edition number (2)
#   9-16 total message length (8-octet big-endian)
EDITION_OFFSET = 7          # zero-based index of the edition octet
LENGTH_OFFSET = 8           # zero-based index of the 8-octet length
HEADER_LEN = 16
DEFAULT_BUFSIZE = 1024 * 1024


@dataclass
class MessageSpan:
    """One complete GRIB2 message located inside a scanned file."""

    offset: int
    length: int
    sha256: str
    # Raw Section 3 (grid definition section) bytes, captured while scanning so
    # the grid identity can be computed without re-reading the file later.
    gds_bytes: bytes = field(repr=False, default=b"")
    gds_offset: int = -1


@dataclass
class ScanResult:
    file_size: int
    file_sha256: str
    messages: list[MessageSpan]


def _u64be(buf: bytes, pos: int) -> int:
    return int.from_bytes(buf[pos:pos + 8], "big", signed=False)


def _u32be(buf: bytes, pos: int) -> int:
    return int.from_bytes(buf[pos:pos + 4], "big", signed=False)


def _read_exact(f: BinaryIO, offset: int, n: int) -> bytes:
    f.seek(offset)
    raw = f.read(n)
    if len(raw) != n:
        raise BadFileError("truncated_section", {
            "offset": offset, "expected": n, "got": len(raw),
        })
    return raw


def _peek_section_number(f: BinaryIO, pos: int, sec_len: int) -> int:
    if sec_len < 5:
        raise BadFileError("bad_section_length",
                           {"offset": pos, "section_length": sec_len})
    return _read_exact(f, pos + 4, 1)[0]


def _extract_gds(f: BinaryIO, msg_offset: int, msg_length: int) -> tuple[bytes, int]:
    """Return the raw Section 3 bytes of one message.

    Only section lengths/numbers are inspected; grid point values
    (Sections 5-7) are never read.  Every read is bounded.
    """

    pos = msg_offset + HEADER_LEN  # Section 0 (indicator) is 16 octets.
    sec_len = _u32be(_read_exact(f, pos, 4), 0)
    if _peek_section_number(f, pos, sec_len) != 1:
        raise BadFileError("bad_section_order",
                           {"offset": pos, "expected_section": 1})
    pos += sec_len
    sec_len = _u32be(_read_exact(f, pos, 4), 0)
    sec_no = _peek_section_number(f, pos, sec_len)
    if sec_no == 2:  # optional local section
        pos += sec_len
        sec_len = _u32be(_read_exact(f, pos, 4), 0)
        sec_no = _peek_section_number(f, pos, sec_len)
    if sec_no != 3:
        raise BadFileError("missing_grid_definition_section",
                           {"offset": pos, "got_section": sec_no})
    if pos + sec_len > msg_offset + msg_length:
        raise BadFileError("section_runs_past_message", {
            "section_offset": pos, "section_length": sec_len,
            "message_end": msg_offset + msg_length,
        })
    return _read_exact(f, pos, sec_len), pos


def scan_file(f: BinaryIO, file_size: int, bufsize: int = DEFAULT_BUFSIZE) -> ScanResult:
    """Scan an open binary GRIB2 stream once, in fixed-size buffers."""

    if bufsize < 64 * 1024:
        bufsize = 64 * 1024

    file_hash = hashlib.sha256()
    messages: list[MessageSpan] = []

    # Stream position of the next byte that still needs framing logic.
    pos = 0
    # Bytes carried because a header straddled a chunk boundary (<= 15).
    hold = b""
    # Open (in-progress) message state.
    open_offset = -1
    open_remaining = 0
    open_hasher: hashlib._Hash | None = None
    last4 = b""

    def fail(reason: str, evidence: dict) -> None:
        raise BadFileError(reason, evidence)

    f.seek(0)
    while True:
        chunk = f.read(bufsize)
        if not chunk:
            break
        file_hash.update(chunk)
        data = hold + chunk
        base = pos - len(hold)        # absolute offset of data[0]
        hold = b""
        i = 0
        n = len(data)

        while i < n:
            if open_hasher is not None:
                # Inside a message: length drives the boundary, not magic.
                take = min(open_remaining, n - i)
                open_hasher.update(data[i:i + take])
                last4 = (last4 + data[i:i + take])[-4:]
                open_remaining -= take
                i += take
                if open_remaining == 0:
                    if last4 != END_MARKER:
                        fail("missing_end_marker", {
                            "offset": open_offset,
                            "final_bytes": last4.hex(),
                        })
                    msg_len = pos_of_end - open_offset
                    resume = f.tell()
                    gds_bytes, gds_off = _extract_gds(f, open_offset, msg_len)
                    f.seek(resume)
                    messages.append(MessageSpan(
                        offset=open_offset,
                        length=msg_len,
                        sha256=open_hasher.hexdigest(),
                        gds_bytes=gds_bytes,
                        gds_offset=gds_off,
                    ))
                    open_hasher = None
                    open_offset = -1
                    last4 = b""
                continue

            # Looking for the next message magic.
            magic_idx = data.find(MAGIC, i)
            if magic_idx == -1:
                # Keep a <=15-byte suffix in case the magic straddles the edge.
                tail_len = min(len(MAGIC) - 1, n - i)
                hold = data[n - tail_len:] if tail_len else b""
                pos = base + n
                i = n
                break
            if magic_idx > i:
                fail("junk_between_messages", {
                    "offset": base + i,
                    "length": magic_idx - i,
                })
            if n - magic_idx < HEADER_LEN:
                # Header straddles the boundary; carry the rest.
                hold = data[magic_idx:]
                pos = base + n
                i = n
                break
            if data[magic_idx + EDITION_OFFSET] != 2:
                fail("unsupported_grib_edition", {
                    "offset": base + magic_idx,
                    "edition": data[magic_idx + EDITION_OFFSET],
                })
            msg_len = _u64be(data, magic_idx + LENGTH_OFFSET)
            if msg_len < HEADER_LEN + 4:
                fail("bad_message_length", {
                    "offset": base + magic_idx, "length": msg_len,
                })
            open_offset = base + magic_idx
            open_remaining = msg_len
            pos_of_end = open_offset + msg_len
            if pos_of_end > file_size:
                fail("message_runs_past_eof", {
                    "offset": open_offset,
                    "message_length": msg_len,
                    "file_size": file_size,
                })
            open_hasher = hashlib.sha256()
            last4 = b""
            # Re-enter the loop at the magic to consume message bytes.
            i = magic_idx

        if hold:
            pos = base + n
        else:
            pos = base + n

    # EOF checks.
    if open_hasher is not None:
        fail("truncated_message", {
            "offset": open_offset,
            "remaining_bytes": open_remaining,
        })
    if hold:
        if MAGIC in hold:
            fail("truncated_message_header", {"offset": file_size - len(hold)})
        fail("trailing_junk", {
            "offset": file_size - len(hold), "length": len(hold),
        })
    if not messages:
        fail("no_grib_messages", {"file_size": file_size})

    return ScanResult(
        file_size=file_size,
        file_sha256=file_hash.hexdigest(),
        messages=messages,
    )


def iter_message_bytes(f: BinaryIO, span: MessageSpan,
                       bufsize: int = DEFAULT_BUFSIZE) -> Iterator[bytes]:
    """Yield the raw bytes of one message in fixed-size buffers."""

    remaining = span.length
    f.seek(span.offset)
    while remaining:
        chunk = f.read(min(bufsize, remaining))
        if not chunk:
            raise BadFileError("truncated_message_on_copy", {
                "offset": span.offset, "remaining": remaining,
            })
        remaining -= len(chunk)
        yield chunk
