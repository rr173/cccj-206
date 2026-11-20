"""Scanner boundary and streaming-hash tests (no ecCodes dependency)."""

import io

import pytest

from grib_asm.errors import BadFileError
from grib_asm.scanner import scan_file, iter_message_bytes

from raw_frames import make_message, sha, section, raw_message


def test_scans_multiple_messages_and_hashes(tmp_path):
    m1 = make_message(b"G1")
    m2 = make_message(b"G2")
    blob = m1 + m2
    res = scan_file(io.BytesIO(blob), len(blob), bufsize=64 * 1024)
    assert len(res.messages) == 2
    assert res.messages[0].offset == 0
    assert res.messages[0].length == len(m1)
    assert res.messages[1].offset == len(m1)
    assert res.messages[0].sha256 == sha(m1)
    assert res.messages[1].sha256 == sha(m2)
    assert res.file_sha256 == sha(blob)
    # GDS extraction locates section 3 exactly.
    assert res.messages[0].gds_bytes == section(3, b"G1")


def test_message_spanning_many_buffers():
    # A message much larger than the buffer forces incremental hashing.
    big_gds = b"X" * 300_000
    msg = make_message(big_gds)
    assert len(msg) > 200_000
    res = scan_file(io.BytesIO(msg), len(msg), bufsize=64 * 1024)
    assert len(res.messages) == 1
    assert res.messages[0].sha256 == sha(msg)
    assert res.messages[0].gds_bytes == section(3, big_gds)


def test_two_large_messages_back_to_back():
    m1 = make_message(b"A" * 200_000)
    m2 = make_message(b"B" * 200_000)
    blob = m1 + m2
    res = scan_file(io.BytesIO(blob), len(blob), bufsize=64 * 1024)
    assert [m.length for m in res.messages] == [len(m1), len(m2)]
    assert res.messages[1].offset == len(m1)
    assert res.messages[1].sha256 == sha(m2)
    assert res.file_sha256 == sha(blob)


def test_truncated_final_message_rejected():
    msg = make_message(b"G")
    blob = msg[:-7]  # removes the end marker and some body
    with pytest.raises(BadFileError) as ei:
        scan_file(io.BytesIO(blob), len(blob), bufsize=64 * 1024)
    assert ei.value.reason in {
        "truncated_message", "message_runs_past_eof",
        "truncated_grid_definition_section"}


def test_declared_length_past_eof_rejected():
    msg = make_message(b"G")
    # Lie about the total length so it runs beyond EOF.
    bogus = raw_message([section(1, b"ID"), section(3, b"G")],
                        length_override=len(msg) + 500)
    with pytest.raises(BadFileError) as ei:
        scan_file(io.BytesIO(bogus), len(bogus), bufsize=64 * 1024)
    assert ei.value.reason == "message_runs_past_eof"


def test_missing_end_marker_rejected():
    msg = make_message(b"G")
    broken = msg[:-4] + b"8888"
    with pytest.raises(BadFileError) as ei:
        scan_file(io.BytesIO(broken), len(broken), bufsize=64 * 1024)
    assert ei.value.reason == "missing_end_marker"


def test_trailing_junk_rejected():
    msg = make_message(b"G")
    with pytest.raises(BadFileError) as ei:
        scan_file(io.BytesIO(msg + b"JUNK"), len(msg) + 4,
                  bufsize=64 * 1024)
    assert ei.value.reason == "trailing_junk"


def test_empty_file_rejected():
    with pytest.raises(BadFileError) as ei:
        scan_file(io.BytesIO(b""), 0, bufsize=64 * 1024)
    assert ei.value.reason == "no_grib_messages"


def test_local_section_then_gds():
    msg = make_message(b"GDS", with_local=True)
    res = scan_file(io.BytesIO(msg), len(msg), bufsize=64 * 1024)
    assert res.messages[0].gds_bytes == section(3, b"GDS")


def test_iter_message_bytes_roundtrip():
    m1 = make_message(b"G1")
    m2 = make_message(b"G2")
    blob = m1 + m2
    res = scan_file(io.BytesIO(blob), len(blob), bufsize=64 * 1024)
    for span, original in zip(res.messages, [m1, m2]):
        out = b"".join(iter_message_bytes(io.BytesIO(blob), span,
                                          bufsize=128))
        assert out == original
        assert sha(out) == span.sha256


def test_unsupported_edition_rejected():
    msg = make_message(b"G", edition=1)
    with pytest.raises(BadFileError) as ei:
        scan_file(io.BytesIO(msg), len(msg), bufsize=64 * 1024)
    assert ei.value.reason == "unsupported_grib_edition"


def test_buffer_smaller_than_message_with_magic_inside():
    # GDS body containing the magic must not confuse boundary detection.
    msg = make_message(b"GRIB" + b"Z" * 100_000)
    res = scan_file(io.BytesIO(msg), len(msg), bufsize=64 * 1024)
    assert len(res.messages) == 1
    assert res.messages[0].sha256 == sha(msg)


@pytest.mark.parametrize("bufsize", [4096, 65536, 131072])
def test_second_header_straddles_buffer_boundary(bufsize):
    m1 = make_message(b"A" * 70_000)
    m2 = make_message(b"B" * 10)
    blob = m1 + m2
    res = scan_file(io.BytesIO(blob), len(blob), bufsize=bufsize)
    assert len(res.messages) == 2
    assert res.messages[1].offset == len(m1)
    assert res.messages[1].sha256 == sha(m2)
    assert res.file_sha256 == sha(blob)
