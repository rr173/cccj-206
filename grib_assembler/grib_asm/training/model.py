"""Dataclasses and strict parsing for recipe, samples and the three streams.

Every timestamp is a UTC instant truncated to a whole second.  Internally all
instants are stored as integer epoch seconds so that visibility comparisons,
half-open interval tests and origin/offset arithmetic use only integer
operations - there is deliberately no floating point anywhere in the clock
logic.

The accepted on-the-wire shapes are documented in ``docs/TRAINING.md``.  The
parsers are intentionally strict: a missing key, a non-integer sequence
number, a sub-second timestamp or a backwards interval fails loudly with a
:class:`~grib_asm.training.errors.StreamError` /
:class:`~grib_asm.training.errors.RecipeError` rather than being guessed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..identity import Level, MemberId, VariableId, make_surface
from ..timeutil import TimeSlot
from .errors import RecipeError, StreamError

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


# --------------------------------------------------------------------------- #
# clock helpers (integer epoch seconds, whole UTC seconds only)
# --------------------------------------------------------------------------- #
def parse_epoch(text, *, context: str) -> int:
    """Parse a canonical ``...Z`` UTC string into integer epoch seconds."""

    if not isinstance(text, str) or not text.endswith("Z") or len(text) != 20:
        raise StreamError("stream_record_invalid", {
            "context": context,
            "timestamp": text,
            "reason": "timestamp must be YYYY-MM-DDTHH:MM:SSZ (whole second)",
        })
    try:
        dt = datetime.strptime(text, ISO_FMT).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise StreamError("stream_record_invalid", {
            "context": context, "timestamp": text, "reason": str(exc)})
    return int(dt.timestamp())


def epoch_iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(ISO_FMT)


def _require_int(value, *, context: str, field_name: str,
                 allow_negative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StreamError("stream_record_invalid", {
            "context": context, "field": field_name, "value": value,
            "reason": "integer required"})
    if not allow_negative and value < 0:
        raise StreamError("stream_record_invalid", {
            "context": context, "field": field_name, "value": value,
            "reason": "non-negative integer required"})
    return value


# --------------------------------------------------------------------------- #
# recipe
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureColumn:
    """One named feature column: full field identity + feature-window offsets.

    ``offset_start`` / ``offset_end`` are whole seconds relative to a sample's
    target start, delimiting the half-open feature window
    ``[target_start + offset_start, target_start + offset_end)``.  The column
    only ever reads the exact slot ``slot`` (relative start/end seconds plus
    statistical process) of the unique origin whose validity equals that
    window.
    """

    name: str
    variable: VariableId
    level: Level
    member: MemberId
    slot: TimeSlot
    offset_start: int
    offset_end: int

    def derive_origin_epoch(self, target_start_epoch: int) -> int:
        """Return the unique forecast origin consistent with both endpoints."""

        feature_start = target_start_epoch + self.offset_start
        feature_end = target_start_epoch + self.offset_end
        origin_from_start = feature_start - self.slot.start_seconds
        origin_from_end = feature_end - self.slot.end_seconds
        if origin_from_start != origin_from_end:
            # Static validation in ``Recipe`` makes this unreachable in
            # practice, but the two-endpoint rule is enforced here too.
            raise RecipeError("recipe_origin_mismatch", {
                "column": self.name,
                "target_start": epoch_iso(target_start_epoch),
                "origin_from_start": epoch_iso(origin_from_start),
                "origin_from_end": epoch_iso(origin_from_end)})
        return origin_from_start


@dataclass
class Recipe:
    recipe_id: str
    label_cutoff_epoch: int
    metric_units: dict[str, str]
    features: list[FeatureColumn]
    raw: dict = field(default_factory=dict)

    @property
    def label_cutoff(self) -> str:
        return epoch_iso(self.label_cutoff_epoch)


def _parse_variable(doc: dict, context: str) -> VariableId:
    try:
        return VariableId(
            discipline=int(doc["discipline"]),
            parameter_category=int(doc["parameterCategory"]),
            parameter_number=int(doc["parameterNumber"]),
            master_tables_version=int(doc["masterTablesVersionNumber"]),
            local_tables_version=int(doc["localTablesVersionNumber"]),
            centre=int(doc["centre"]),
            sub_centre=int(doc["subCentre"]))
    except KeyError as exc:
        raise RecipeError("recipe_invalid",
                          {"context": context, "missing": str(exc)})


def _parse_level(doc: dict, context: str) -> Level:
    try:
        first_doc = doc["first"]
        first = make_surface(int(first_doc["type"]),
                             int(first_doc["scale_factor"]),
                             int(first_doc["scaled_value"]))
        second = None
        if doc.get("second") is not None:
            sd = doc["second"]
            second = make_surface(int(sd["type"]), int(sd["scale_factor"]),
                                  int(sd["scaled_value"]))
        return Level(first=first, second=second)
    except KeyError as exc:
        raise RecipeError("recipe_invalid",
                          {"context": context, "missing": str(exc)})


def _parse_member(doc: dict, context: str) -> MemberId:
    try:
        return MemberId(
            kind=str(doc["kind"]),
            type_of_ensemble=(None if doc.get("typeOfEnsembleForecast") is None
                              else int(doc["typeOfEnsembleForecast"])),
            number=(None if doc.get("perturbationNumber") is None
                    else int(doc["perturbationNumber"])))
    except KeyError as exc:
        raise RecipeError("recipe_invalid",
                          {"context": context, "missing": str(exc)})


def load_recipe(path: str) -> Recipe:
    data = json.loads(open(path, encoding="utf-8").read())
    return recipe_from_dict(data)


def recipe_from_dict(data: dict) -> Recipe:
    context = "recipe"
    for key in ("recipe_id", "label_cutoff", "metric_units", "features"):
        if key not in data:
            raise RecipeError("recipe_invalid",
                              {"missing": key, "context": context})
    cutoff = parse_epoch(data["label_cutoff"], context=context)
    metric_units = data["metric_units"]
    if not isinstance(metric_units, dict) or not metric_units or \
            not all(isinstance(k, str) and isinstance(v, str)
                    for k, v in metric_units.items()):
        raise RecipeError("recipe_invalid", {
            "context": context,
            "reason": "metric_units must be a non-empty metric -> unit map"})

    features: list[FeatureColumn] = []
    seen_names: set[str] = set()
    for pos, col_doc in enumerate(data["features"]):
        cctx = f"{context}.features[{pos}]"
        name = col_doc.get("name")
        if not isinstance(name, str) or not name:
            raise RecipeError("recipe_invalid",
                              {"context": cctx, "reason": "name required"})
        if name in seen_names:
            raise RecipeError("recipe_invalid",
                              {"context": cctx, "reason": "duplicate name",
                               "name": name})
        seen_names.add(name)
        slot_doc = col_doc["slot"]
        slot = TimeSlot(
            start_seconds=int(slot_doc["start_seconds"]),
            end_seconds=int(slot_doc["end_seconds"]),
            statistic=None if slot_doc.get("statistic") is None
            else int(slot_doc["statistic"]))
        # This export only serves explicitly specified interval-statistics
        # slots; instantaneous fields are a different product.
        if slot.is_instant:
            raise RecipeError("recipe_invalid", {
                "context": cctx, "name": name,
                "reason": "only interval-statistics slots are supported"})
        offset_start = _recipe_int(col_doc, "feature_offset_start_seconds",
                                   cctx, name)
        offset_end = _recipe_int(col_doc, "feature_offset_end_seconds",
                                 cctx, name)
        if offset_end < offset_start:
            raise RecipeError("recipe_invalid", {
                "context": cctx, "name": name,
                "reason": "feature window end precedes start"})
        # The two-endpoint origin rule must hold for every sample; because
        # both windows are fixed-length offsets this is equivalent to equal
        # window lengths, checked once here.
        feature_len = offset_end - offset_start
        slot_len = slot.end_seconds - slot.start_seconds
        if feature_len != slot_len:
            raise RecipeError("recipe_origin_mismatch", {
                "context": cctx, "name": name,
                "feature_window_seconds": feature_len,
                "slot_seconds": slot_len,
                "reason": "the two feature-window endpoints derive different "
                          "forecast origins"})
        features.append(FeatureColumn(
            name=name,
            variable=_parse_variable(col_doc["variable"], cctx),
            level=_parse_level(col_doc["level"], cctx),
            member=_parse_member(col_doc["member"], cctx),
            slot=slot,
            offset_start=offset_start,
            offset_end=offset_end))
    if not features:
        raise RecipeError("recipe_invalid", {
            "context": context, "reason": "at least one feature required"})
    return Recipe(
        recipe_id=str(data["recipe_id"]),
        label_cutoff_epoch=cutoff,
        metric_units={str(k): str(v) for k, v in metric_units.items()},
        features=features,
        raw=data)


def _recipe_int(doc: dict, key: str, context: str, name: str) -> int:
    value = doc.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecipeError("recipe_invalid", {
            "context": context, "name": name, "field": key, "value": value,
            "reason": "integer seconds required"})
    return value


# --------------------------------------------------------------------------- #
# samples
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Sample:
    sample_id: str
    station_id: str
    metric: str
    target_start_epoch: int
    target_end_epoch: int
    decision_epoch: int
    order: int  # input order (output is re-sorted by sample_id bytes)

    @property
    def target_start(self) -> str:
        return epoch_iso(self.target_start_epoch)

    @property
    def target_end(self) -> str:
        return epoch_iso(self.target_end_epoch)

    @property
    def decision_at(self) -> str:
        return epoch_iso(self.decision_epoch)


def sample_from_dict(doc: dict, order: int) -> Sample:
    context = f"samples[{order}]"
    for key in ("sample_id", "station_id", "metric", "target_start",
                "target_end", "decision_at"):
        if key not in doc:
            raise StreamError("stream_record_invalid",
                              {"context": context, "missing": key})
    sample_id = str(doc["sample_id"])
    station_id = str(doc["station_id"])
    metric = str(doc["metric"])
    t0 = parse_epoch(doc["target_start"], context=context)
    t1 = parse_epoch(doc["target_end"], context=context)
    if t1 <= t0:
        raise StreamError("stream_record_invalid", {
            "context": context, "sample_id": sample_id,
            "reason": "target interval must be a non-empty half-open range"})
    decision = parse_epoch(doc["decision_at"], context=context)
    return Sample(sample_id, station_id, metric, t0, t1, decision, order)


def validate_sample_against_recipe(sample: Sample, recipe: Recipe) -> None:
    if sample.decision_epoch > sample.target_start_epoch:
        raise RecipeError("decision_after_target_start", {
            "sample_id": sample.sample_id,
            "decision_at": sample.decision_at,
            "target_start": sample.target_start})
    if recipe.label_cutoff_epoch < sample.target_end_epoch:
        raise RecipeError("label_cutoff_too_early", {
            "sample_id": sample.sample_id,
            "label_cutoff": recipe.label_cutoff,
            "target_end": sample.target_end})
    if sample.metric not in recipe.metric_units:
        raise RecipeError("recipe_invalid", {
            "sample_id": sample.sample_id, "metric": sample.metric,
            "reason": "recipe has no expected unit for this metric"})


# --------------------------------------------------------------------------- #
# stream records (publish / station / observation)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StreamRecord:
    """One normalized JSONL line from one of the three history streams."""

    stream: str                 # publish | station | observation
    record_id: str
    business_key: str           # canonical JSON of the stream's business key
    visible_at_epoch: int
    seq: int
    action: str
    payload: str                # canonical JSON of the full record (dedup)
    doc: dict = field(repr=False)


def _base_fields(doc: dict, context: str) -> tuple[str, int, int]:
    for key in ("record_id", "visible_at", "seq"):
        if key not in doc:
            raise StreamError("stream_record_invalid",
                              {"context": context, "missing": key})
    record_id = str(doc["record_id"])
    visible = parse_epoch(doc["visible_at"], context=context)
    seq = _require_int(doc["seq"], context=context, field_name="seq",
                       allow_negative=False)
    return record_id, visible, seq


def publish_record(doc: dict, line_no: int) -> StreamRecord:
    context = f"publish[{line_no}]"
    record_id, visible, seq = _base_fields(doc, context)
    action = doc.get("action")
    if action not in ("publish", "withdraw"):
        raise StreamError("stream_record_invalid", {
            "context": context, "action": action,
            "reason": "publish action must be publish|withdraw"})
    if "origin" not in doc:
        raise StreamError("stream_record_invalid",
                          {"context": context, "missing": "origin"})
    origin = parse_epoch(doc["origin"], context=context)
    if visible < origin:
        raise StreamError("stream_visibility_before_reference", {
            "context": context, "record_id": record_id,
            "visible_at": epoch_iso(visible), "origin": epoch_iso(origin)})
    if action == "publish":
        summary = doc.get("package_summary")
        if not isinstance(summary, dict) or \
                not isinstance(summary.get("package_digest"), str):
            raise StreamError("stream_record_invalid", {
                "context": context, "record_id": record_id,
                "reason": "publish requires package_summary.package_digest"})
    business_key = json.dumps(
        {"stream": "publish", "origin": origin},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return StreamRecord("publish", record_id, business_key, visible, seq,
                        action, _canonical(doc), doc)


def station_record(doc: dict, line_no: int) -> StreamRecord:
    context = f"station[{line_no}]"
    record_id, visible, seq = _base_fields(doc, context)
    action = doc.get("action")
    if action not in ("map", "unmap"):
        raise StreamError("stream_record_invalid", {
            "context": context, "action": action,
            "reason": "station action must be map|unmap"})
    for key in ("station_id", "effective_from", "effective_to"):
        if key not in doc:
            raise StreamError("stream_record_invalid",
                              {"context": context, "missing": key})
    station_id = str(doc["station_id"])
    eff_from = parse_epoch(doc["effective_from"], context=context)
    eff_to = parse_epoch(doc["effective_to"], context=context)
    if eff_to <= eff_from:
        raise StreamError("stream_record_invalid", {
            "context": context, "record_id": record_id,
            "reason": "effective interval must be a non-empty half-open range"})
    if action == "map":
        for key in ("grid_digest", "lat_index", "lon_index"):
            if key not in doc:
                raise StreamError("stream_record_invalid",
                                  {"context": context, "missing": key})
        if not isinstance(doc["grid_digest"], str):
            raise StreamError("stream_record_invalid", {
                "context": context, "field": "grid_digest",
                "reason": "string required"})
        _require_int(doc["lat_index"], context=context,
                     field_name="lat_index", allow_negative=False)
        _require_int(doc["lon_index"], context=context,
                     field_name="lon_index", allow_negative=False)
    business_key = json.dumps(
        {"stream": "station", "station_id": station_id},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return StreamRecord("station", record_id, business_key, visible, seq,
                        action, _canonical(doc), doc)


def observation_record(doc: dict, line_no: int) -> StreamRecord:
    context = f"observation[{line_no}]"
    record_id, visible, seq = _base_fields(doc, context)
    action = doc.get("action")
    if action not in ("set", "retract"):
        raise StreamError("stream_record_invalid", {
            "context": context, "action": action,
            "reason": "observation action must be set|retract"})
    for key in ("station_id", "metric", "target_start", "target_end"):
        if key not in doc:
            raise StreamError("stream_record_invalid",
                              {"context": context, "missing": key})
    station_id = str(doc["station_id"])
    metric = str(doc["metric"])
    t0 = parse_epoch(doc["target_start"], context=context)
    t1 = parse_epoch(doc["target_end"], context=context)
    if t1 <= t0:
        raise StreamError("stream_record_invalid", {
            "context": context, "record_id": record_id,
            "reason": "observation interval must be non-empty"})
    if visible < t1:
        raise StreamError("stream_visibility_before_reference", {
            "context": context, "record_id": record_id,
            "visible_at": epoch_iso(visible),
            "interval_end": epoch_iso(t1),
            "reason": "measurement visibility cannot precede interval end"})
    if action == "set":
        if "value" not in doc or "unit" not in doc:
            raise StreamError("stream_record_invalid", {
                "context": context, "record_id": record_id,
                "reason": "set requires value and unit"})
        value = doc["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise StreamError("stream_record_invalid", {
                "context": context, "record_id": record_id,
                "field": "value", "value": value,
                "reason": "finite numeric value required"})
        if not isinstance(doc["unit"], str) or not doc["unit"]:
            raise StreamError("stream_record_invalid", {
                "context": context, "record_id": record_id,
                "field": "unit", "reason": "non-empty unit string required"})
    business_key = json.dumps(
        {"stream": "observation", "station_id": station_id,
         "metric": metric, "target_start": t0, "target_end": t1},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return StreamRecord("observation", record_id, business_key, visible, seq,
                        action, _canonical(doc), doc)


def _canonical(doc: dict) -> str:
    """Stable full-record serialization used for resend/conflict decisions."""

    return json.dumps(doc, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
