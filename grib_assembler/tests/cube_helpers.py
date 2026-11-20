"""Runtime GRIB2/hand-off package builders for cube tests.

These helpers generate small, fully reproducible GRIB2 messages with the
test-only encoder (:mod:`fixturegen`), ingest them through the real catalog
and publish real hand-off packages with :mod:`grib_asm.assemble`.  No large
binary fixture is ever committed; everything lives under the pytest tmp
directory.

The expected *values* used by the acceptance tests are never produced from
these helpers: tests compute their hand-authored expected arrays with plain
numpy arithmetic over the documented seed formula below.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from grib_asm.assemble import assemble
from grib_asm.catalog import Catalog
from grib_asm.identity import grid_digest
from grib_asm.request import (
    Manifest, Request, RequestedLevel, RequestedMember, RequestedVariable,
)
from grib_asm.scanner import scan_file
from grib_asm.timeutil import TimeSlot, iso

from fixturegen import FixtureSpec, encode_message

UTC = timezone.utc

# Hand-documented fixture constants (tests repeat these independently).
DISC, CAT, NUM = 0, 0, 0
CENTRE, SUBCENTRE = 7, 0
MTV, LTV = 29, 1
NI, NJ = 4, 3
FIRST_LAT, FIRST_LON = 40.0, 10.0
DELTA_LAT, DELTA_LON = 1.0, 1.0
LEVELS = [(100, 0, 85000), (100, 0, 50000)]      # type, scale, scaled Pa
REFERENCE_HOURS = (0, 12)
REFERENCE_DATE = 20260101
# Slot codes used everywhere in the cube tests.
INSTANT = ("instant", 0, 0, None)
ACC_0_6 = ("acc", 0, 6 * 3600, 1)
ACC_3_6 = ("acc", 3 * 3600, 6 * 3600, 1)
ALL_SLOTS = (INSTANT, ACC_0_6, ACC_3_6)


def expected_seed(cycle_idx: int, level_idx: int, member_idx: int,
                  slot_idx: int) -> int:
    """Documented, hand-authored seed formula for one cell's values."""

    return cycle_idx * 100 + level_idx * 10 + member_idx * 4 + slot_idx


def expected_values(cycle_idx, level_idx, member_idx, slot_idx):
    """Hand-authored expected (Nj, Ni) array, independent of product code."""

    import numpy as np
    seed = expected_seed(cycle_idx, level_idx, member_idx, slot_idx)
    return (np.arange(NI * NJ, dtype="float64")
            + 1000.0 * seed).reshape(NJ, NI)


def expected_lats():
    # Hand-authored: scan mode 0 descends north->south from the first point.
    import numpy as np
    return FIRST_LAT - np.arange(NJ, dtype="float64") * DELTA_LAT


def expected_lons():
    import numpy as np
    return FIRST_LON + np.arange(NI, dtype="float64") * DELTA_LON


def standard_grid_digest() -> str:
    spec = FixtureSpec(ni=NI, nj=NJ, first_lat=FIRST_LAT,
                       first_lon=FIRST_LON, delta_lat=DELTA_LAT,
                       delta_lon=DELTA_LON)
    import io
    raw = encode_message(spec)
    res = scan_file(io.BytesIO(raw), len(raw))
    return grid_digest(res.messages[0].gds_bytes)


def _spec_for(cycle_idx: int, level_idx: int, member_idx: int,
              slot_idx: int, slot: tuple, *, pnum=NUM,
              grid_overrides=None):
    ref_hour = REFERENCE_HOURS[cycle_idx]
    surf_type, surf_scale, surf_scaled = LEVELS[level_idx]
    member_kind = "control" if member_idx == 0 else "perturbed"
    ens_type = 3 if member_idx == 0 else 1
    ens_num = 0 if member_idx == 0 else 1
    seed = expected_seed(cycle_idx, level_idx, member_idx, slot_idx)
    kw = dict(
        name=f"c{cycle_idx}-l{level_idx}-m{member_idx}-s{slot_idx}",
        discipline=DISC, parameter_category=CAT, parameter_number=pnum,
        master_tables_version=MTV, local_tables_version=LTV,
        centre=CENTRE, sub_centre=SUBCENTRE,
        data_date=REFERENCE_DATE, data_time=ref_hour * 100,
        ni=NI, nj=NJ, first_lat=FIRST_LAT, first_lon=FIRST_LON,
        delta_lat=DELTA_LAT, delta_lon=DELTA_LON,
        first_surface_type=surf_type, first_scale=surf_scale,
        first_scaled=surf_scaled,
        # 24 bits per value keeps every synthetic integer exactly
        # representable through the encode/decode round trip.
        bits_per_value=24, data_seed=seed,
    )
    if grid_overrides:
        kw.update(grid_overrides)
    kind, start_s, end_s, statistic = slot
    if kind == "instant":
        kw.update(pdt=1 if member_kind != "deterministic" else 0,
                  ensemble_type=ens_type, perturbation_number=ens_num,
                  forecast_time=start_s // 3600, time_unit=1)
    else:
        start_h = start_s // 3600
        length_h = (end_s - start_s) // 3600
        end_hour = ref_hour + end_s // 3600
        kw.update(
            pdt=11, ensemble_type=ens_type, perturbation_number=ens_num,
            forecast_time=start_h, time_unit=1,
            number_of_time_range=1, type_of_time_increment=2,
            time_increment_unit=1, time_increment=0, number_missing=0,
            statistical_process=statistic,
            length_of_time_range=length_h, time_range_unit=1,
            end_year=2026, end_month=1, end_day=1,
            end_hour=end_hour, end_minute=0, end_second=0)
    return FixtureSpec(**kw)


def build_cycle_specs(cycle_idx: int, *, members=(0, 1),
                      slots=ALL_SLOTS, levels=(0, 1), pnum=NUM,
                      grid_overrides=None):
    specs = []
    for li in levels:
        for mi in members:
            for si, slot in enumerate(slots):
                specs.append((li, mi, si, slot, _spec_for(
                    cycle_idx, li, mi, si, slot, pnum=pnum,
                    grid_overrides=grid_overrides)))
    return specs


def write_collection(tmp_path: Path, collection_id: str, specs,
                     created_at="2026-01-01T00:00:00Z"):
    root = tmp_path / collection_id
    root.mkdir(parents=True, exist_ok=True)
    messages = [encode_message(s) for *_rest, s in specs]
    blob = b"".join(messages)
    rel = "all.grib2"
    (root / rel).write_bytes(blob)
    manifest_doc = {
        "collection_id": collection_id,
        "created_at": created_at,
        "files": [{
            "file_id": "all",
            "relative_path": rel,
            "size_bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }],
    }
    mpath = tmp_path / f"{collection_id}.manifest.json"
    mpath.write_text(json.dumps(manifest_doc), encoding="utf-8")
    entry = type("E", (), dict(
        file_id="all", relative_path=rel, size_bytes=len(blob),
        sha256=hashlib.sha256(blob).hexdigest()))()
    manifest = Manifest(collection_id, created_at, [entry], manifest_doc)
    return root, manifest


def make_request(request_id: str, grid_dig: str, *, members=(0, 1),
                 slots=ALL_SLOTS, levels=(0, 1), pnum=NUM,
                 origin_hour=0):
    var = RequestedVariable(DISC, CAT, pnum, MTV, LTV, CENTRE, SUBCENTRE)
    req_levels = []
    for li in levels:
        st, sf, sv = LEVELS[li]
        req_levels.append(RequestedLevel(st, sf, sv, None, None, None))
    req_members = []
    for mi in members:
        if mi == 0:
            req_members.append(RequestedMember("control", 3, 0))
        else:
            req_members.append(RequestedMember("perturbed", 1, 1))
    req_slots = []
    for kind, start_s, end_s, statistic in slots:
        req_slots.append(TimeSlot(start_s, end_s, statistic))
    return Request(
        request_id=request_id,
        origin_from=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        origin_to=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        not_after=datetime(2026, 1, 2, 0, 0, tzinfo=UTC),
        expected_grid_digest=grid_dig,
        variables=[var], levels=req_levels, members=req_members,
        slots=req_slots)


def build_package(tmp_path: Path, cycle_idx: int, request: Request, *,
                  members=(0, 1), slots=ALL_SLOTS, levels=(0, 1),
                  pnum=NUM, grid_overrides=None,
                  catalog_db=None):
    """Ingest one cycle and publish a verified hand-off package."""

    collection_id = f"cyc{cycle_idx}"
    specs = build_cycle_specs(
        cycle_idx, members=members, slots=slots, levels=levels, pnum=pnum,
        grid_overrides=grid_overrides)
    root, manifest = write_collection(tmp_path, collection_id, specs)
    db = Path(catalog_db) if catalog_db else tmp_path / f"cat{cycle_idx}.db"
    cat = Catalog(db)
    cat.ingest_file(manifest, manifest.files[0], root)
    out_dir = tmp_path / "packages"
    result = assemble(cat, request, {collection_id: root}, out_dir,
                      {collection_id: manifest})
    cat.close()
    return result.package_dir


def ref_time(cycle_idx: int) -> str:
    return iso(datetime(2026, 1, 1, REFERENCE_HOURS[cycle_idx], 0, 0,
                        tzinfo=UTC))


def real_package_from_message(tmp_path: Path, raw: bytes,
                              grid_dig: str, request_id: str):
    """Publish one arbitrary verified message through catalog + assembler.

    Used for shapes (rotated grids, alternative scan modes, unusual levels)
    that the regular Cartesian builder does not cover.  The request is built
    from the catalog's own classified identity so the package verifies.
    """

    import hashlib
    root = tmp_path / "coll"
    root.mkdir(parents=True)
    rel = "all.grib2"
    (root / rel).write_bytes(raw)
    manifest_doc = {
        "collection_id": "coll",
        "created_at": "2026-01-01T00:00:00Z",
        "files": [{
            "file_id": "all", "relative_path": rel,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}],
    }
    entry = type("E", (), dict(
        file_id="all", relative_path=rel, size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest()))()
    manifest = Manifest("coll", "2026-01-01T00:00:00Z", [entry],
                        manifest_doc)
    cat = Catalog(tmp_path / "cat.db")
    ingest = cat.ingest_file(manifest, manifest.files[0], root)
    assert ingest.status == "ok"

    row = cat._conn.execute("SELECT identity FROM messages").fetchone()
    identity = json.loads(row["identity"])
    var_doc = identity["variable"]
    first = identity["level"]["first"]
    mem_doc = identity["member"]
    slot_doc = identity["slot"]
    var = RequestedVariable(
        var_doc["discipline"], var_doc["parameterCategory"],
        var_doc["parameterNumber"],
        var_doc["masterTablesVersionNumber"],
        var_doc["localTablesVersionNumber"],
        var_doc["centre"], var_doc["subCentre"])
    second = identity["level"].get("second")
    level = RequestedLevel(
        first["type"], first["scale_factor"], first["scaled_value"],
        None if second is None else second["type"],
        None if second is None else second["scale_factor"],
        None if second is None else second["scaled_value"])
    if mem_doc["kind"] == "deterministic":
        member = RequestedMember("deterministic", None, None)
    else:
        member = RequestedMember(
            mem_doc["kind"], mem_doc["typeOfEnsembleForecast"],
            mem_doc["perturbationNumber"])
    slot = TimeSlot(slot_doc["start_seconds"], slot_doc["end_seconds"],
                    slot_doc["statistic"])
    request = Request(
        request_id=request_id,
        origin_from=datetime(2026, 1, 1, tzinfo=UTC),
        origin_to=datetime(2026, 12, 31, tzinfo=UTC),
        not_after=datetime(2027, 1, 1, tzinfo=UTC),
        expected_grid_digest=grid_dig,
        variables=[var], levels=[level], members=[member], slots=[slot])
    result = assemble(cat, request, {"coll": root},
                      tmp_path / "packages", {"coll": manifest})
    cat.close()
    return result.package_dir
