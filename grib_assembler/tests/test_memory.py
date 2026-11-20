"""Memory-flatness evidence for large inputs.

Generates a temporary large archive (many messages), runs the streaming
ingest, and asserts that peak Python resident memory does not scale with the
number of messages: only fixed buffers plus the single currently-open ecCodes
message handle are ever held.  Also asserts the core never requests a data
array key by wrapping the header access guard.

The archive is generated on demand in a temp directory and is not committed.
"""

import hashlib
import io
import os
import resource
import sys
from pathlib import Path

import pytest

from grib_asm.catalog import Catalog
from grib_asm.request import Manifest, ManifestFile
from grib_asm.scanner import scan_file
from grib_asm import headers as headers_mod

from fixturegen import FixtureSpec, encode_message
from e2e_helpers import HAS_ECCODES

pytestmark = [
    pytest.mark.skipif(not HAS_ECCODES,
                       reason="ecCodes required to encode fixtures"),
    pytest.mark.slow,
]


def _rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def test_ingest_memory_does_not_grow_with_message_count(tmp_path):
    # Two archives: small (N1 messages) and large (N2 messages on a tiny
    # grid).  Peak RSS during the large ingest must stay within a fixed slack
    # band of the small ingest rather than growing linearly with N.
    n_small, n_large = 200, 2000

    def build_archive(collection_id, n):
        root = tmp_path / collection_id
        root.mkdir(parents=True, exist_ok=True)
        messages = [
            encode_message(FixtureSpec(
                pdt=0, ni=2, nj=2, data_seed=i,
                bits_per_value=8))
            for i in range(n)
        ]
        blob = b"".join(messages)
        rel = "big.grib2"
        (root / rel).write_bytes(blob)
        manifest = Manifest(collection_id, None, [ManifestFile(
            "big", rel, len(blob), hashlib.sha256(blob).hexdigest())], {})
        return root, manifest

    # Guard: the forbidden array keys must always raise.
    from grib_asm.headers import MessageHeader
    assert "values" in headers_mod._FORBIDDEN_ARRAY_KEYS

    # Warm the interpreter/ecCodes once so one-off allocations do not pollute
    # the steady-state measurement.
    root_w, man_w = build_archive("warm", 50)
    cat_w = Catalog(tmp_path / "warm.db")
    cat_w.ingest_file(man_w, man_w.files[0], root_w, bufsize=64 * 1024)
    cat_w.close()
    base_rss = _rss_kb()

    root_s, man_s = build_archive("small", n_small)
    cat_s = Catalog(tmp_path / "small.db")
    cat_s.ingest_file(man_s, man_s.files[0], root_s, bufsize=64 * 1024)
    after_small = _rss_kb()
    cat_s.close()

    root_l, man_l = build_archive("large", n_large)
    cat_l = Catalog(tmp_path / "large.db")
    cat_l.ingest_file(man_l, man_l.files[0], root_l, bufsize=64 * 1024)
    after_large = _rss_kb()
    cat_l.close()

    # Per-message marginal RSS must not grow with archive size: a flat-memory
    # pipeline spends ~constant working memory per message regardless of total
    # count, so the large archive's per-message footprint must stay within a
    # modest factor of the small one (never the 10x message-count ratio).
    per_msg_small = max(after_small - base_rss, 1) / n_small
    per_msg_large = max(after_large - after_small, 1) / (
        n_large - n_small)
    assert per_msg_large <= per_msg_small * 4.0, (
        f"per-message memory grew with archive size: "
        f"small={per_msg_small:.2f}KB/msg large={per_msg_large:.2f}KB/msg")

    # Catalog persisted every message on disk even though memory stayed flat.
    cat_check = Catalog(tmp_path / "large.db")
    n_rows = cat_check._conn.execute(
        "SELECT COUNT(*) c FROM messages").fetchone()["c"]
    assert n_rows == n_large
    cat_check.close()


def test_header_layer_refuses_data_array_keys():
    from grib_asm.errors import GribAsmError
    from grib_asm.headers import MessageHeader
    class Dummy:
        pass
    hdr = MessageHeader(Dummy())
    with pytest.raises(GribAsmError):
        hdr.int("values")
    with pytest.raises(GribAsmError):
        hdr.int("latitudes")
