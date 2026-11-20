"""Unit tests for the point-in-time clock, recipe and stream-record rules.

These tests never open a cube: they exercise the pure integer-second clock,
the two-endpoint origin derivation, the visibility/sequence ordering and the
strict input validation.  End-to-end behaviour against a real imported cube
lives in ``test_training_export.py``.
"""

from __future__ import annotations

import json

import pytest

from grib_asm.identity import (
    Level, MemberId, VariableId, make_surface,
)
from grib_asm.timeutil import TimeSlot
from grib_asm.training.errors import RecipeError, StreamError
from grib_asm.training.model import (
    FeatureColumn, epoch_iso, observation_record as model_observation,
    parse_epoch, publish_record as model_publish, recipe_from_dict,
    sample_from_dict, station_record as model_station,
    validate_sample_against_recipe,
)
from grib_asm.training.timeline import latest_visible
from grib_asm.training.workspace import TrainingWorkspace

UTC_EPOCH_2026 = 1767225600  # 2026-01-01T00:00:00Z


def _column(name="c", start=3 * 3600, end=6 * 3600,
            off_start=0, off_end=3 * 3600):
    var = VariableId(0, 0, 0, 29, 1, 7, 0)
    level = Level(first=make_surface(100, 0, 85000), second=None)
    member = MemberId(kind="control", type_of_ensemble=3, number=0)
    slot = TimeSlot(start, end, 1)
    return FeatureColumn(name, var, level, member, slot, off_start, off_end)


# --------------------------------------------------------------------------- #
# clock
# --------------------------------------------------------------------------- #
def test_parse_epoch_is_whole_second_integer():
    assert parse_epoch("2026-01-01T09:00:00Z", context="t") == \
        UTC_EPOCH_2026 + 9 * 3600
    assert epoch_iso(UTC_EPOCH_2026 + 9 * 3600) == "2026-01-01T09:00:00Z"


@pytest.mark.parametrize("bad", [
    "2026-01-01T09:00:00.5Z", "2026-01-01 09:00:00", "2026-01-01T09:00Z",
    "not-a-time", 1767258000,
])
def test_parse_epoch_rejects_subsecond_and_noncanonical(bad):
    with pytest.raises(StreamError):
        parse_epoch(bad, context="t")


# --------------------------------------------------------------------------- #
# two-endpoint origin rule
# --------------------------------------------------------------------------- #
def test_both_endpoints_derive_same_origin():
    col = _column(start=3 * 3600, end=6 * 3600,
                  off_start=0, off_end=3 * 3600)
    target_start = UTC_EPOCH_2026 + 9 * 3600
    # Feature window 09-12; slot 3-6 -> origin 06:00 from both endpoints.
    assert col.derive_origin_epoch(target_start) == \
        UTC_EPOCH_2026 + 6 * 3600


def test_differing_endpoints_is_recipe_error():
    # Slot length 3 h but feature window length 6 h: the endpoints derive
    # different origins, which the recipe rejects at validation time.
    doc = _recipe_doc()
    doc["features"][0]["feature_offset_end_seconds"] = 6 * 3600
    with pytest.raises(RecipeError) as exc:
        recipe_from_dict(doc)
    assert exc.value.reason == "recipe_origin_mismatch"


def test_distinct_columns_locate_independent_slots():
    # Same feature end (12:00) but a 0-6 h slot and -3h..+3h window: origin
    # must still be 06:00, never inferred from another slot.
    col_0_6 = _column("early", start=0, end=6 * 3600,
                      off_start=-3 * 3600, off_end=3 * 3600)
    col_3_6 = _column("late", start=3 * 3600, end=6 * 3600,
                      off_start=0, off_end=3 * 3600)
    target_start = UTC_EPOCH_2026 + 9 * 3600
    assert col_0_6.derive_origin_epoch(target_start) == \
        col_3_6.derive_origin_epoch(target_start) == UTC_EPOCH_2026 + 6 * 3600


# --------------------------------------------------------------------------- #
# recipe validation
# --------------------------------------------------------------------------- #
def _recipe_doc(**overrides):
    doc = {
        "recipe_id": "r1",
        "label_cutoff": "2026-01-01T18:00:00Z",
        "metric_units": {"temperature": "K"},
        "features": [{
            "name": "c1",
            "variable": {"discipline": 0, "parameterCategory": 0,
                         "parameterNumber": 0,
                         "masterTablesVersionNumber": 29,
                         "localTablesVersionNumber": 1,
                         "centre": 7, "subCentre": 0},
            "level": {"first": {"type": 100, "scale_factor": 0,
                                "scaled_value": 85000}, "second": None},
            "member": {"kind": "control",
                       "typeOfEnsembleForecast": 3,
                       "perturbationNumber": 0},
            "slot": {"start_seconds": 10800, "end_seconds": 21600,
                     "statistic": 1},
            "feature_offset_start_seconds": 0,
            "feature_offset_end_seconds": 10800}],
    }
    doc.update(overrides)
    return doc


def test_recipe_accepts_interval_slot():
    recipe = recipe_from_dict(_recipe_doc())
    assert recipe.features[0].name == "c1"
    assert recipe.label_cutoff == "2026-01-01T18:00:00Z"


def test_recipe_rejects_instant_slot():
    doc = _recipe_doc()
    doc["features"][0]["slot"] = {"start_seconds": 0, "end_seconds": 0,
                                  "statistic": None}
    with pytest.raises(RecipeError):
        recipe_from_dict(doc)


def test_recipe_rejects_duplicate_column_names():
    doc = _recipe_doc()
    doc["features"].append(json.loads(json.dumps(doc["features"][0])))
    with pytest.raises(RecipeError):
        recipe_from_dict(doc)


def test_recipe_rejects_float_offsets():
    doc = _recipe_doc()
    doc["features"][0]["feature_offset_end_seconds"] = 10800.0
    with pytest.raises(RecipeError):
        recipe_from_dict(doc)


# --------------------------------------------------------------------------- #
# sample invariants
# --------------------------------------------------------------------------- #
def _sample(decision="2026-01-01T08:30:00Z",
            start="2026-01-01T09:00:00Z",
            end="2026-01-01T12:00:00Z"):
    return sample_from_dict({
        "sample_id": "S1", "station_id": "ST1", "metric": "temperature",
        "target_start": start, "target_end": end,
        "decision_at": decision}, 0)


def test_decision_must_not_follow_target_start():
    recipe = recipe_from_dict(_recipe_doc())
    sample = _sample(decision="2026-01-01T09:00:01Z")
    with pytest.raises(RecipeError) as exc:
        validate_sample_against_recipe(sample, recipe)
    assert exc.value.reason == "decision_after_target_start"


def test_decision_equal_to_target_start_is_allowed():
    recipe = recipe_from_dict(_recipe_doc())
    sample = _sample(decision="2026-01-01T09:00:00Z")
    validate_sample_against_recipe(sample, recipe)  # boundary equality OK


def test_label_cutoff_must_cover_target_end():
    doc = _recipe_doc(label_cutoff="2026-01-01T11:59:59Z")
    recipe = recipe_from_dict(doc)
    with pytest.raises(RecipeError) as exc:
        validate_sample_against_recipe(_sample(), recipe)
    assert exc.value.reason == "label_cutoff_too_early"


def test_unknown_metric_rejected():
    recipe = recipe_from_dict(_recipe_doc())
    sample = sample_from_dict({
        "sample_id": "S2", "station_id": "ST1", "metric": "rainfall",
        "target_start": "2026-01-01T09:00:00Z",
        "target_end": "2026-01-01T12:00:00Z",
        "decision_at": "2026-01-01T08:00:00Z"}, 1)
    with pytest.raises(RecipeError):
        validate_sample_against_recipe(sample, recipe)


# --------------------------------------------------------------------------- #
# stream record validation
# --------------------------------------------------------------------------- #
def test_publish_cannot_precede_origin():
    with pytest.raises(StreamError) as exc:
        model_publish(
            {"record_id": "P", "action": "publish",
             "origin": "2026-01-01T06:00:00Z",
             "visible_at": "2026-01-01T05:59:59Z", "seq": 0,
             "package_summary": {"package_digest": "x"}}, 0)
    assert exc.value.reason == "stream_visibility_before_reference"


def test_publish_requires_digest():
    with pytest.raises(StreamError):
        model_publish(
            {"record_id": "P", "action": "publish",
             "origin": "2026-01-01T06:00:00Z",
             "visible_at": "2026-01-01T08:00:00Z", "seq": 0}, 0)


def test_observation_visibility_not_before_interval_end():
    with pytest.raises(StreamError) as exc:
        model_observation(
            {"record_id": "O", "action": "set", "station_id": "ST1",
             "metric": "temperature",
             "visible_at": "2026-01-01T11:59:59Z", "seq": 0,
             "target_start": "2026-01-01T09:00:00Z",
             "target_end": "2026-01-01T12:00:00Z",
             "value": 5, "unit": "K"}, 0)
    assert exc.value.reason == "stream_visibility_before_reference"


def test_observation_set_requires_finite_value_and_unit():
    base = {"record_id": "O", "action": "set", "station_id": "ST1",
            "metric": "temperature",
            "visible_at": "2026-01-01T12:00:00Z", "seq": 0,
            "target_start": "2026-01-01T09:00:00Z",
            "target_end": "2026-01-01T12:00:00Z"}
    with pytest.raises(StreamError):
        model_observation({**base, "value": float("nan"), "unit": "K"}, 0)
    with pytest.raises(StreamError):
        model_observation({**base, "value": 5, "unit": ""}, 0)


def test_station_requires_non_negative_indices_and_valid_interval():
    base = {"record_id": "M", "action": "map", "station_id": "ST1",
            "visible_at": "2026-01-01T08:00:00Z", "seq": 0,
            "effective_from": "2026-01-01T09:00:00Z",
            "effective_to": "2026-01-01T12:00:00Z",
            "grid_digest": "g", "lat_index": 0, "lon_index": 0}
    model_station(dict(base), 0)  # valid
    with pytest.raises(StreamError):
        model_station({**base, "lat_index": -1}, 0)
    with pytest.raises(StreamError):
        model_station({**base,
                       "effective_to": "2026-01-01T09:00:00Z"}, 0)


def test_negative_sequence_rejected():
    with pytest.raises(StreamError):
        model_publish(
            {"record_id": "P", "action": "withdraw",
             "origin": "2026-01-01T06:00:00Z",
             "visible_at": "2026-01-01T08:00:00Z", "seq": -1}, 0)


# --------------------------------------------------------------------------- #
# ordering / dedup / conflicts via the disk workspace
# --------------------------------------------------------------------------- #
def _ws(tmp_path):
    recipe = recipe_from_dict(_recipe_doc())
    ws = TrainingWorkspace(tmp_path / "ws.db")
    ws.set_recipe(recipe)
    return ws


def _pub_doc(record_id, seq, action, *, digest=None):
    doc = {"record_id": record_id, "action": action,
           "origin": "2026-01-01T06:00:00Z",
           "visible_at": "2026-01-01T08:00:00Z", "seq": seq}
    if action == "publish":
        doc["package_summary"] = {"package_digest": digest or "d"}
    return doc


def test_identical_resend_is_deduplicated(tmp_path):
    ws = _ws(tmp_path)
    rec = _pub_doc("P", 0, "publish", digest="d")
    ws.add_publish(rec)
    ws.add_publish(json.loads(json.dumps(rec)))  # byte-identical resend
    rows = ws.conn.execute("SELECT COUNT(*) c FROM events").fetchone()
    assert rows["c"] == 1
    ws.destroy()


def test_same_instant_same_seq_different_content_conflicts(tmp_path):
    ws = _ws(tmp_path)
    ws.add_publish(_pub_doc("P1", 0, "publish", digest="d1"))
    with pytest.raises(StreamError) as exc:
        ws.add_publish(_pub_doc("P2", 0, "publish", digest="d2"))
    assert exc.value.reason == "stream_duplicate_conflict"
    ws.destroy()


def test_same_instant_higher_sequence_wins_independent_of_line_order(tmp_path):
    ws = _ws(tmp_path)
    origin = "2026-01-01T06:00:00Z"
    # Insert the higher-sequence record first; line order must not matter.
    high = _pub_doc("P-high", 5, "withdraw")
    low = _pub_doc("P-low", 1, "publish", digest="d")
    ws.add_publish(high)
    ws.add_publish(low)
    cutoff = parse_epoch("2026-01-01T08:00:00Z", context="t")
    winner = ws.publish_winner(parse_epoch(origin, context="t"), cutoff)
    assert winner["record_id"] == "P-high"
    ws.destroy()


def test_latest_visible_respects_cutoff_equality():
    # Pure-function sanity check of the boundary rule.
    recs = []
    from grib_asm.training.model import StreamRecord
    for vis, seq, rid in ((100, 0, "a"), (200, 0, "b"), (300, 0, "c")):
        recs.append(StreamRecord(
            "publish", rid, "k", vis, seq, "publish",
            json.dumps({"rid": rid}), {"rid": rid}))
    assert latest_visible(recs, 200).record_id == "b"   # equality visible
    assert latest_visible(recs, 199).record_id == "a"
    assert latest_visible(recs, 50) is None


def test_duplicate_sample_id_rejected(tmp_path):
    ws = _ws(tmp_path)
    doc = {"sample_id": "S1", "station_id": "ST1",
           "metric": "temperature",
           "target_start": "2026-01-01T09:00:00Z",
           "target_end": "2026-01-01T12:00:00Z",
           "decision_at": "2026-01-01T08:00:00Z"}
    ws.add_sample(doc)
    with pytest.raises(StreamError):
        ws.add_sample(json.loads(json.dumps(doc)))
    ws.destroy()
