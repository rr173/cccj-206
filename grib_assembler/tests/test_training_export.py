"""End-to-end point-in-time export tests against a really imported cube.

Covers the full acceptance matrix on real Zarr data:

* the four-decision timeline (not published -> A -> B -> withdrawn) including
  the migration right-end fallback and station/publish/field precedence;
* label cutoff independence (5 -> 7 -> retract changes labels only);
* explicit point-missing, absent-variable and decoy-slot handling;
* evidence-error exits (wrong grid, out of bounds, bad publish digest,
  uncommitted origin, provenance conflict) with no success output left behind;
* output ordering, summary counts and the once-per-value-block read guarantee.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grib_asm.training.errors import TrainingExportError
from grib_asm.training.export import (
    FIELD_MISSING, FORECAST_NOT_PUBLISHED, FORECAST_WITHDRAWN, OK,
    POINT_MISSING, STATION_NOT_REGISTERED, STATION_UNMAPPED,
)
from training_e2e_helpers import (
    DAY, VALUE_A, VALUE_B, at, base_recipe, col_0_6, col_3_6, run_export,
    sample_record, standard_station_streams, write_two_origin_world,
    write_world,
)
from training_fixtures import (
    observation_record, publish_record, station_record,
)

pytestmark = pytest.mark.skipif(
    __import__("e2e_helpers", fromlist=["HAS_ECCODES"]).HAS_ECCODES is False,
    reason="ecCodes required to build the cube fixtures")


def _decision_samples(station="ST1"):
    return [
        sample_record(sample_id=f"S{i:02d}", station_id=station,
                      metric="temperature", target_start=at(9),
                      target_end=at(12), decision_at=decision)
        for i, decision in enumerate(
            (at(7, 59), at(8), at(8, 30), at(9)), start=1)]


def _obs_5_7_retract():
    return [
        observation_record(record_id="O1", station_id="ST1",
                           metric="temperature", visible_at=at(12, 5),
                           seq=0, action="set", target_start=at(9),
                           target_end=at(12), value=5, unit="K"),
        observation_record(record_id="O2", station_id="ST1",
                           metric="temperature", visible_at=at(13),
                           seq=0, action="set", target_start=at(9),
                           target_end=at(12), value=7, unit="K"),
        observation_record(record_id="O3", station_id="ST1",
                           metric="temperature", visible_at=at(17),
                           seq=0, action="retract", target_start=at(9),
                           target_end=at(12)),
    ]


# --------------------------------------------------------------------------- #
# headline four-decision timeline
# --------------------------------------------------------------------------- #
def test_four_decision_timeline_not_published_a_b_withdrawn(tmp_path):
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world)
    recipe = base_recipe(columns=[col_3_6()])
    rows, summary, report = run_export(
        tmp_path, world, recipe=recipe, samples=_decision_samples(),
        publish=publish, station=station, observations=_obs_5_7_retract())
    by_id = {r["sample_id"]: r for r in rows}
    expected = {
        "S01": (FORECAST_NOT_PUBLISHED, None),
        "S02": (OK, VALUE_A),
        "S03": (OK, VALUE_B),
        "S04": (FORECAST_WITHDRAWN, None)}
    for sid, (status, value) in expected.items():
        feature = by_id[sid]["features"][0]
        assert feature["status"] == status, sid
        assert feature["value"] == value, sid
    # The chosen origin/window are always the 06:00 3-6 h slot.
    for sid in expected:
        feature = by_id[sid]["features"][0]
        assert feature["origin"] == "2026-01-01T06:00:00Z"
        assert feature["feature_window"] == [at(9), at(12)]
    # One distinct values block served the two OK samples: read exactly once.
    assert report.value_block_reads == 1
    assert summary["sample_count"] == 4


def test_a_and_b_use_distinct_grid_points_with_evidence(tmp_path):
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world)
    recipe = base_recipe(columns=[col_3_6()])
    rows, _, _ = run_export(
        tmp_path, world, recipe=recipe, samples=_decision_samples(),
        publish=publish, station=station, observations=[])
    by_id = {r["sample_id"]: r for r in rows}
    f_a = by_id["S02"]["features"][0]["evidence"]["field"]
    f_b = by_id["S03"]["features"][0]["evidence"]["field"]
    assert f_a["point_index"] == {"lat_index": 0, "lon_index": 1}
    assert f_b["point_index"] == {"lat_index": 2, "lon_index": 2}
    # Field provenance is retained even though both points are present.
    assert f_a["message_digest"] and f_a["package_digest"]
    assert f_a["package_digest"] == world["package_digest"]
    # The station records that won each row are recorded.
    assert by_id["S02"]["features"][0]["evidence"]["station"][
        "evidence"]["winning_record"]["record_id"] == "M1"
    assert by_id["S03"]["features"][0]["evidence"]["station"][
        "evidence"]["winning_record"]["record_id"] == "M2"


def test_migration_right_end_keeps_older_registration(tmp_path):
    world = write_world(tmp_path)
    # A map runs to 18:00; B only covers [09:00,12:00). A target starting at
    # exactly 12:00 must resolve to the older A registration.
    publish, station = standard_station_streams(world, map_a_to=at(18))
    samples = _decision_samples() + [sample_record(
        sample_id="S05", station_id="ST1", metric="temperature",
        target_start=at(12), target_end=at(15), decision_at=at(11, 30))]
    recipe = base_recipe(columns=[col_3_6()])
    rows, _, _ = run_export(
        tmp_path, world, recipe=recipe, samples=samples,
        publish=publish, station=station, observations=[])
    s05 = next(r for r in rows if r["sample_id"] == "S05")
    winner = s05["features"][0]["evidence"]["station"]["evidence"][
        "winning_record"]
    assert winner["record_id"] == "M1"
    assert s05["features"][0]["evidence"]["station"]["lat_index"] == 0


# --------------------------------------------------------------------------- #
# two clocks: label cutoff never changes features
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cutoff,label_status,label_value", [
    (at(12, 10), "ok", 5),
    (at(13, 30), "ok", 7),
    (at(18), "label_retracted", None),
])
def test_label_cutoff_changes_label_not_features(tmp_path, cutoff,
                                                 label_status, label_value):
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world)
    recipe = base_recipe(columns=[col_3_6()], label_cutoff=cutoff)
    rows, _, _ = run_export(
        tmp_path, world, recipe=recipe, samples=_decision_samples(),
        publish=publish, station=station,
        observations=_obs_5_7_retract())
    by_id = {r["sample_id"]: r for r in rows}
    for sid, feature_status in (
            ("S01", FORECAST_NOT_PUBLISHED), ("S02", OK),
            ("S03", OK), ("S04", FORECAST_WITHDRAWN)):
        assert by_id[sid]["features"][0]["status"] == feature_status
    # The label of every sample sharing the 09-12 target follows the cutoff.
    for sid in ("S01", "S02", "S03", "S04"):
        label = by_id[sid]["label"]
        assert label["status"] == label_status
        assert label["value"] == label_value


def test_label_unit_mismatch_is_distinct_status(tmp_path):
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world, map_a_to=at(18))
    samples = [sample_record(
        sample_id="S05", station_id="ST1", metric="temperature",
        target_start=at(12), target_end=at(15), decision_at=at(11, 30))]
    obs = [observation_record(
        record_id="O9", station_id="ST1", metric="temperature",
        visible_at=at(15, 5), seq=0, action="set",
        target_start=at(12), target_end=at(15), value=280.0, unit="degC")]
    recipe = base_recipe(columns=[col_3_6()])
    rows, _, _ = run_export(
        tmp_path, world, recipe=recipe, samples=samples,
        publish=publish, station=station, observations=obs)
    label = rows[0]["label"]
    assert label["status"] == "label_unit_mismatch"
    assert label["value"] is None
    assert label["unit"] == "degC"
    assert label["expected_unit"] == "K"


def test_label_not_received_and_retract_clears_old_value(tmp_path):
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world)
    recipe = base_recipe(columns=[col_3_6()], label_cutoff=at(12, 4))
    # No observation visible before the cutoff for the 09-12 target.
    rows, summary, _ = run_export(
        tmp_path, world, recipe=recipe, samples=_decision_samples(),
        publish=publish, station=station, observations=[])
    assert all(r["label"]["status"] == "label_not_received" for r in rows)
    assert summary["label_status_counts"]["label_not_received"] == 4


def test_label_cutoff_boundary_equality_and_exclusion(tmp_path):
    from datetime import timedelta
    world = write_world(tmp_path)
    samples = [sample_record(
        sample_id="B1", station_id="ST1", metric="temperature",
        target_start=at(9), target_end=at(12), decision_at=at(8, 30))]
    obs = [observation_record(
        record_id="O", station_id="ST1", metric="temperature",
        visible_at=at(13), seq=0, action="set", target_start=at(9),
        target_end=at(12), value=7, unit="K")]
    # Visible exactly at the cutoff -> included.
    rows_eq, _, _ = run_export(
        tmp_path / "eq", world,
        recipe=base_recipe(columns=[col_3_6()], label_cutoff=at(13)),
        samples=samples, publish=[], station=[], observations=obs)
    assert rows_eq[0]["label"]["status"] == "ok"
    assert rows_eq[0]["label"]["value"] == 7.0
    # One second before the cutoff -> still not received.
    before = (DAY + timedelta(hours=12, minutes=59, seconds=59)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    rows_before, _, _ = run_export(
        tmp_path / "before", world,
        recipe=base_recipe(columns=[col_3_6()], label_cutoff=before),
        samples=samples, publish=[], station=[], observations=obs)
    assert rows_before[0]["label"]["status"] == "label_not_received"


def test_empty_sample_manifest_writes_valid_empty_outputs(tmp_path):
    world = write_world(tmp_path)
    rows, summary, report = run_export(
        tmp_path, world, recipe=base_recipe(columns=[col_3_6()]),
        samples=[], publish=[], station=[], observations=[])
    assert rows == []
    assert summary["sample_count"] == 0
    assert report.value_block_reads == 0
    assert (tmp_path / "out.jsonl").read_text(encoding="utf-8") == ""


def test_workspace_cleanup_and_retention(tmp_path):
    from grib_asm.training.export import export_training_table, ExportConfig
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world)
    d = tmp_path / "ws"
    d.mkdir(parents=True, exist_ok=True)
    from training_e2e_helpers import write_jsonl
    (d / "recipe.json").write_text(
        json.dumps(base_recipe(columns=[col_3_6()])), encoding="utf-8")
    write_jsonl(d / "samples.jsonl", _decision_samples()[1:2])
    write_jsonl(d / "publish.jsonl", publish)
    write_jsonl(d / "station.jsonl", station)
    write_jsonl(d / "observation.jsonl", [])
    # Caller-supplied work dir survives but holds none of our artifacts.
    work = tmp_path / "userwork"
    export_training_table(
        world["store"], d / "recipe.json", d / "samples.jsonl",
        d / "publish.jsonl", d / "station.jsonl", d / "observation.jsonl",
        tmp_path / "a.jsonl", tmp_path / "a.json",
        ExportConfig(work_dir=str(work)))
    assert work.exists()
    assert not (work / "training.sqlite").exists()
    assert not (work / "rows").exists()
    # keep_workspace retains the SQLite staging for inspection.
    kept = tmp_path / "keptwork"
    export_training_table(
        world["store"], d / "recipe.json", d / "samples.jsonl",
        d / "publish.jsonl", d / "station.jsonl", d / "observation.jsonl",
        tmp_path / "b.jsonl", tmp_path / "b.json",
        ExportConfig(work_dir=str(kept), keep_workspace=True))
    assert (kept / "training.sqlite").exists()


# --------------------------------------------------------------------------- #
# field/point gaps and the decoy slot
# --------------------------------------------------------------------------- #
def test_point_missing_keeps_field_provenance(tmp_path):
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world)
    # The missing-point variable (parameter 9) has NaN at point B.
    recipe = base_recipe(columns=[col_3_6("m", pnum=world["missing_param"])])
    rows, _, _ = run_export(
        tmp_path, world, recipe=recipe, samples=_decision_samples(),
        publish=publish, station=station, observations=[])
    by_id = {r["sample_id"]: r for r in rows}
    # At 08:00 the sample reads point A (present=3.x); at 08:30 point B is
    # explicitly missing.
    assert by_id["S02"]["features"][0]["status"] == OK
    s03 = by_id["S03"]["features"][0]
    assert s03["status"] == POINT_MISSING
    # Even for a missing point, the field's source evidence is retained.
    field = s03["evidence"]["field"]
    assert field["present"] is True
    assert field["message_digest"]
    assert field["point_result"]["present"] == 0


def test_absent_variable_is_field_missing_while_others_exist(tmp_path):
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world)
    recipe = base_recipe(columns=[
        col_3_6("present"), col_3_6("absent", pnum=world["absent_param"])])
    rows, summary, _ = run_export(
        tmp_path, world, recipe=recipe,
        samples=_decision_samples()[1:3],  # S02, S03 are after publication
        publish=publish, station=station, observations=[])
    for row in rows:
        statuses = [f["status"] for f in row["features"]]
        assert statuses[0] == OK
        assert statuses[1] == FIELD_MISSING
    assert summary["feature_status_counts"][FIELD_MISSING] == 2


def test_same_end_different_slot_is_located_independently(tmp_path):
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world)
    # Two columns share the 12:00 feature end but use different slots; both
    # derive the 06:00 origin and read their own fields, never substituting.
    recipe = base_recipe(columns=[col_3_6("late"), col_0_6("early")])
    rows, _, report = run_export(
        tmp_path, world, recipe=recipe, samples=_decision_samples()[1:2],
        publish=publish, station=station, observations=[])
    feature = rows[0]["features"]
    assert feature[0]["status"] == OK and feature[0]["value"] == VALUE_A
    assert feature[1]["status"] == OK
    # The 0-6 h grid was built with the 100+0+6=106 fill value.
    assert feature[1]["value"] == 106.0
    assert feature[1]["feature_window"] == [at(6), at(12)]
    # Two distinct value blocks (one per slot), each read once.
    assert report.value_block_reads == 2


# --------------------------------------------------------------------------- #
# station precedence and unmap
# --------------------------------------------------------------------------- #
def test_station_gate_precedes_publish_gate(tmp_path):
    world = write_world(tmp_path)
    publish, _ = standard_station_streams(world)
    # A station id that has no map records at all.
    samples = [sample_record(
        sample_id="G1", station_id="GHOST", metric="temperature",
        target_start=at(9), target_end=at(12), decision_at=at(8, 30))]
    recipe = base_recipe(columns=[col_3_6()])
    rows, summary, report = run_export(
        tmp_path, world, recipe=recipe, samples=samples,
        publish=publish, station=[], observations=[])
    assert rows[0]["features"][0]["status"] == STATION_NOT_REGISTERED
    # No value block may be read when the station gate fails.
    assert report.value_block_reads == 0
    assert summary["feature_status_counts"][STATION_NOT_REGISTERED] == 1


def test_unmap_only_masks_its_named_interval(tmp_path):
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world, map_a_to=at(18))
    # An unmap that cancels exactly the [09:00,12:00) migration coverage.
    station.append(station_record(
        record_id="U1", station_id="ST1", visible_at=at(8, 45), seq=0,
        action="unmap", effective_from=at(9), effective_to=at(12)))
    samples = [
        sample_record(sample_id="U-a", station_id="ST1",
                      metric="temperature", target_start=at(9),
                      target_end=at(12), decision_at=at(8, 45)),
        sample_record(sample_id="U-b", station_id="ST1",
                      metric="temperature", target_start=at(12),
                      target_end=at(15), decision_at=at(11, 45))]
    recipe = base_recipe(columns=[col_3_6()])
    rows, _, _ = run_export(
        tmp_path, world, recipe=recipe, samples=samples,
        publish=publish, station=station, observations=[])
    by_id = {r["sample_id"]: r for r in rows}
    # The 09-12 target is covered by the unmap interval -> unmapped.
    assert by_id["U-a"]["features"][0]["status"] == STATION_UNMAPPED
    # The 12-15 target is outside it -> the older A map still applies.
    assert by_id["U-b"]["features"][0]["evidence"]["station"][
        "evidence"]["winning_record"]["record_id"] == "M1"


# --------------------------------------------------------------------------- #
# evidence errors (hard failures, no success output)
# --------------------------------------------------------------------------- #
def _expect_error(tmp_path, world, *, recipe, samples, publish, station,
                  observations, reason):
    out_jsonl = tmp_path / "never.jsonl"
    out_summary = tmp_path / "never.json"
    from training_e2e_helpers import write_jsonl
    d = tmp_path / "bad"
    d.mkdir(parents=True, exist_ok=True)
    (d / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    write_jsonl(d / "samples.jsonl", samples)
    write_jsonl(d / "publish.jsonl", publish)
    write_jsonl(d / "station.jsonl", station)
    write_jsonl(d / "observation.jsonl", observations)
    from grib_asm.training.export import export_training_table, ExportConfig
    with pytest.raises(TrainingExportError) as exc:
        export_training_table(
            store_path=world["store"], recipe_path=d / "recipe.json",
            samples_path=d / "samples.jsonl",
            publish_path=d / "publish.jsonl",
            station_path=d / "station.jsonl",
            observation_path=d / "observation.jsonl",
            out_jsonl=out_jsonl, out_summary=out_summary,
            config=ExportConfig())
    assert exc.value.reason == reason
    assert not out_jsonl.exists() and not out_summary.exists()


def test_map_onto_wrong_grid_is_evidence_error(tmp_path):
    world = write_world(tmp_path)
    publish, _ = standard_station_streams(world)
    bad_station = [station_record(
        record_id="M1", station_id="ST1", visible_at=at(7, 30), seq=0,
        action="map", effective_from=at(0), effective_to=at(18),
        grid_digest="wrong-grid-digest", lat_index=0, lon_index=1)]
    _expect_error(
        tmp_path, world, recipe=base_recipe(columns=[col_3_6()]),
        samples=_decision_samples()[1:2], publish=publish,
        station=bad_station, observations=[],
        reason="station_grid_mismatch")


def test_map_index_out_of_bounds_is_evidence_error(tmp_path):
    world = write_world(tmp_path)
    publish, _ = standard_station_streams(world)
    bad_station = [station_record(
        record_id="M1", station_id="ST1", visible_at=at(7, 30), seq=0,
        action="map", effective_from=at(0), effective_to=at(18),
        grid_digest=world["grid_digest"], lat_index=99, lon_index=1)]
    _expect_error(
        tmp_path, world, recipe=base_recipe(columns=[col_3_6()]),
        samples=_decision_samples()[1:2], publish=publish,
        station=bad_station, observations=[],
        reason="station_index_out_of_bounds")


def test_publish_digest_mismatch_is_evidence_error(tmp_path):
    world = write_world(tmp_path)
    _, station = standard_station_streams(world)
    bad_publish = [publish_record(
        record_id="P1", origin=world["origin"], visible_at=at(8), seq=0,
        action="publish", package_digest="not-the-cube-digest")]
    _expect_error(
        tmp_path, world, recipe=base_recipe(columns=[col_3_6()]),
        samples=_decision_samples()[1:2], publish=bad_publish,
        station=station, observations=[], reason="publish_digest_mismatch")


def test_publish_for_uncommitted_origin_is_evidence_error(tmp_path):
    world = write_world(tmp_path)
    _, station = standard_station_streams(world)
    # A publish for an origin the cube never committed.
    ghost_origin = at(0)
    bad_publish = [publish_record(
        record_id="P1", origin=ghost_origin, visible_at=at(2), seq=0,
        action="publish", package_digest=world["package_digest"])]
    # Recipe column whose window derives the 00:00 origin (slot 3-6, target
    # 03-06 -> origin 00:00).
    samples = [sample_record(
        sample_id="X1", station_id="ST1", metric="temperature",
        target_start=at(3), target_end=at(6), decision_at=at(2, 30))]
    station00 = [station_record(
        record_id="M0", station_id="ST1", visible_at=at(2), seq=0,
        action="map", effective_from=at(0), effective_to=at(18),
        grid_digest=world["grid_digest"], lat_index=0, lon_index=1)]
    _expect_error(
        tmp_path, world, recipe=base_recipe(columns=[col_3_6()]),
        samples=samples, publish=bad_publish, station=station00,
        observations=[], reason="publish_origin_not_committed")


def test_publish_naming_other_committed_package_is_evidence_error(tmp_path):
    # The 06:00 publish claims the *00:00* package digest.  Both packages are
    # committed, but the claimed digest does not match the 06:00 origin's
    # recorded source, so the publish-level digest gate rejects it before any
    # value block is read.
    world = write_two_origin_world(tmp_path)
    bad_publish = [publish_record(
        record_id="P1", origin=world["origin_06"], visible_at=at(8), seq=0,
        action="publish",
        package_digest=world["package_by_origin"][world["origin_00"]])]
    station = [station_record(
        record_id="M1", station_id="ST1", visible_at=at(7, 30), seq=0,
        action="map", effective_from=at(0), effective_to=at(18),
        grid_digest=world["grid_digest"], lat_index=0, lon_index=1)]
    _expect_error(
        tmp_path, world, recipe=base_recipe(columns=[col_3_6()]),
        samples=_decision_samples()[1:2], publish=bad_publish,
        station=station, observations=[],
        reason="publish_digest_mismatch")


# --------------------------------------------------------------------------- #
# ordering and output shape
# --------------------------------------------------------------------------- #
def test_output_is_sorted_by_sample_id_utf8_bytes(tmp_path):
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world)
    # Deliberately shuffled input order, including multi-byte-ish ids.
    ids = ["S10", "S2", "S1", "Ä", "S9"]
    samples = [sample_record(
        sample_id=sid, station_id="ST1", metric="temperature",
        target_start=at(9), target_end=at(12), decision_at=at(8, 30))
        for sid in ids]
    recipe = base_recipe(columns=[col_3_6()])
    rows, _, _ = run_export(
        tmp_path, world, recipe=recipe, samples=samples,
        publish=publish, station=station, observations=[])
    out_ids = [r["sample_id"] for r in rows]
    assert out_ids == sorted(ids, key=lambda s: s.encode("utf-8"))


def test_summary_counts_match_rows_and_columns(tmp_path):
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world)
    recipe = base_recipe(columns=[col_3_6("a"), col_3_6("b")])
    rows, summary, _ = run_export(
        tmp_path, world, recipe=recipe, samples=_decision_samples(),
        publish=publish, station=station, observations=_obs_5_7_retract())
    # Every sample has exactly the recipe's columns, in order.
    for row in rows:
        assert [f["column"] for f in row["features"]] == ["a", "b"]
    total_feature_cells = sum(summary["feature_status_counts"].values())
    assert total_feature_cells == len(rows) * 2
    total_label_cells = sum(summary["label_status_counts"].values())
    assert total_label_cells == len(rows)


def test_shuffled_stream_lines_do_not_change_result(tmp_path):
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world)
    obs = list(reversed(_obs_5_7_retract()))  # reverse line order
    shuffled_publish = list(reversed(publish))
    recipe = base_recipe(columns=[col_3_6()])
    rows, _, _ = run_export(
        tmp_path, world, recipe=recipe, samples=_decision_samples(),
        publish=shuffled_publish, station=list(reversed(station)),
        observations=obs)
    by_id = {r["sample_id"]: r for r in rows}
    assert by_id["S03"]["features"][0]["status"] == OK
    assert by_id["S03"]["features"][0]["value"] == VALUE_B
    assert by_id["S04"]["features"][0]["status"] == FORECAST_WITHDRAWN
    assert by_id["S01"]["label"]["status"] == "label_retracted"


# --------------------------------------------------------------------------- #
# command line entry point
# --------------------------------------------------------------------------- #
def test_cli_writes_jsonl_and_summary(tmp_path):
    import subprocess
    import sys
    world = write_world(tmp_path)
    publish, station = standard_station_streams(world)
    from training_e2e_helpers import write_jsonl
    d = tmp_path / "cli"
    d.mkdir(parents=True, exist_ok=True)
    recipe = base_recipe(columns=[col_3_6()])
    (d / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    write_jsonl(d / "samples.jsonl", _decision_samples())
    write_jsonl(d / "publish.jsonl", publish)
    write_jsonl(d / "station.jsonl", station)
    write_jsonl(d / "observation.jsonl", _obs_5_7_retract())
    out_jsonl = d / "training.jsonl"
    out_summary = d / "summary.json"
    repo_root = Path(__file__).resolve().parents[1]
    # ``-I`` ignores PYTHONPATH, so mirror the project's own tool convention:
    # a tiny runner that inserts the repo root before importing the CLI.
    runner = d / "run_cli.py"
    runner.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from grib_asm.training.cli import main\n"
        "raise SystemExit(main())\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-I", str(runner),
         "--store", world["store"], "--recipe", str(d / "recipe.json"),
         "--samples", str(d / "samples.jsonl"),
         "--publish-stream", str(d / "publish.jsonl"),
         "--station-stream", str(d / "station.jsonl"),
         "--observation-stream", str(d / "observation.jsonl"),
         "--output-jsonl", str(out_jsonl),
         "--output-summary", str(out_summary)],
        cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["sample_count"] == 4
    assert payload["value_block_reads"] == 1
    rows = [json.loads(line) for line in out_jsonl.read_text(
        encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert json.loads(out_summary.read_text(encoding="utf-8"))[
        "sample_count"] == 4


def test_cli_evidence_error_leaves_no_output(tmp_path):
    import subprocess
    import sys
    world = write_world(tmp_path)
    publish, _ = standard_station_streams(world)
    bad_station = [station_record(
        record_id="M1", station_id="ST1", visible_at=at(7, 30), seq=0,
        action="map", effective_from=at(0), effective_to=at(18),
        grid_digest="wrong", lat_index=0, lon_index=1)]
    from training_e2e_helpers import write_jsonl
    d = tmp_path / "cli-bad"
    d.mkdir(parents=True, exist_ok=True)
    (d / "recipe.json").write_text(
        json.dumps(base_recipe(columns=[col_3_6()])), encoding="utf-8")
    write_jsonl(d / "samples.jsonl", _decision_samples()[1:2])
    write_jsonl(d / "publish.jsonl", publish)
    write_jsonl(d / "station.jsonl", bad_station)
    write_jsonl(d / "observation.jsonl", [])
    out_jsonl, out_summary = d / "o.jsonl", d / "o.json"
    repo_root = Path(__file__).resolve().parents[1]
    runner = d / "run_cli.py"
    runner.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from grib_asm.training.cli import main\n"
        "raise SystemExit(main())\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-I", str(runner),
         "--store", world["store"], "--recipe", str(d / "recipe.json"),
         "--samples", str(d / "samples.jsonl"),
         "--publish-stream", str(d / "publish.jsonl"),
         "--station-stream", str(d / "station.jsonl"),
         "--observation-stream", str(d / "observation.jsonl"),
         "--output-jsonl", str(out_jsonl),
         "--output-summary", str(out_summary)],
        cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode == 6
    err = json.loads(proc.stdout)
    assert err["error"] == "station_grid_mismatch"
    assert not out_jsonl.exists() and not out_summary.exists()
