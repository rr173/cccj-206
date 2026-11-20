"""End-to-end test helpers: request building and collection writing.

Fixture/spec construction lives in ``tools/demo_common.py`` so the committed
demo generator and the tests share one implementation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from grib_asm.catalog import Catalog
from grib_asm.request import (
    Manifest, Request, RequestedLevel, RequestedMember,
    RequestedVariable,
)
from grib_asm.timeutil import TimeSlot
from grib_asm.identity import grid_digest

from fixturegen import encode_message, build_manifest
from demo_common import (
    CONTROL_SLOTS, PERTURBED_SLOTS, stat_spec, grid_digest_of,
    LEVEL_TYPE, LEVEL_SCALED, DISC, CAT, NUM, MTV, LTV, CENTRE, SUBCENTRE,
)

# Re-export for tests that import these names from here.
__all__ = [
    "HAS_ECCODES", "make_stat_spec", "encode_message", "build_manifest",
    "grid_digest", "_write_collection", "_build_request", "_grid_digest_of",
    "CONTROL_SLOTS", "PERTURBED_SLOTS",
]

try:
    import eccodes  # noqa: F401
    HAS_ECCODES = True
except Exception:
    HAS_ECCODES = False

make_stat_spec = stat_spec
_grid_digest_of = grid_digest_of
UTC = timezone.utc


class _Entry:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _write_collection(tmp_path, collection_id, specs_by_file):
    root = tmp_path / collection_id
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for fid, specs in specs_by_file.items():
        messages = [encode_message(s) for s in specs]
        blob = b"".join(messages)
        rel = f"{fid}.grib2"
        (root / rel).write_bytes(blob)
        entries.append({
            "file_id": fid,
            "relative_path": rel,
            "size_bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        })
    manifest_doc = build_manifest(collection_id, root, entries,
                                  created_at="2026-01-01T00:00:00Z")
    mpath = tmp_path / f"{collection_id}.manifest.json"
    mpath.write_text(json.dumps(manifest_doc), encoding="utf-8")
    manifest = Manifest(
        collection_id, "2026-01-01T00:00:00Z",
        [_Entry(**e) for e in entries], manifest_doc)
    return root, manifest


def _four_cycle_files(hour, present):
    """Build file specs for a cycle; ``present`` maps member->[slot indices]."""
    specs = []
    for member in ("control", "perturbed"):
        for slot_idx, (s, e) in enumerate(CONTROL_SLOTS):
            if slot_idx in present[member]:
                specs.append(stat_spec(
                    hour, member, s, e,
                    data_seed=100 * hour
                    + (1 if member == "perturbed" else 0) + slot_idx))
    return {"all": specs}


def _ingest(tmp_path, collections, dbname="cat.db"):
    cat = Catalog(tmp_path / dbname)
    roots = {}
    manifests = {}
    for cid, files in collections:
        root, manifest = _write_collection(tmp_path, cid, files)
        roots[cid] = root
        manifests[cid] = manifest
        for entry in manifest.files:
            cat.ingest_file(manifest, entry, root)
    return cat, roots, manifests


def _build_request(grid_dig, not_after="2026-01-01T12:00:00Z"):
    var = RequestedVariable(DISC, CAT, NUM, MTV, LTV, CENTRE, SUBCENTRE)
    lvl = RequestedLevel(LEVEL_TYPE, 0, LEVEL_SCALED, None, None, None)
    members = [
        RequestedMember("control", 3, 0),
        RequestedMember("perturbed", 1, 1),
    ]
    slots = [
        TimeSlot(0, 3 * 3600, 1),
        TimeSlot(3 * 3600, 6 * 3600, 1),
    ]
    return Request(
        request_id="drill",
        origin_from=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        origin_to=datetime(2026, 1, 1, 6, 0, tzinfo=UTC),
        not_after=datetime.strptime(not_after, "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=UTC),
        expected_grid_digest=grid_dig,
        variables=[var], levels=[lvl], members=members, slots=slots,
    )
