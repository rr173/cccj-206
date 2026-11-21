"""Strict, allocation-light parsing of already-exported training rows.

A distribution run never re-runs the point-in-time matching: it only consumes
the rows that ``cube-training-export`` already wrote.  This module therefore
does just enough structural validation to make the grouping and delivery
decisions trustworthy:

* the joint identity ``(export_id, sample_id)`` must be legal and unique
  within one export;
* ``(station_id, metric)`` and the half-open integer ``target_interval`` must
  be present and legal;
* a feature whose status is ``ok`` *must* carry the original GRIB2 message
  identity in ``evidence.field.message_digest`` and that identity must look
  like a real SHA-256 digest.  A missing/illegal digest on an allegedly
  successful feature is a hard error, never treated as "unrelated".

Digests retained by *unsuccessful* features are deliberately ignored: a
missing/withdrawn cell is not evidence that two rows consumed the same field.
Likewise equal values or shared package digests never create a relation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .distribute_errors import RowError
from .distribute_plan import is_legal_message_digest

# Feature status that proves a row actually consumed a field value.
FEATURE_OK = "ok"
LABEL_OK = "ok"

# Stable per-row isolation reasons (also emitted by the engine).
REASON_LABEL_UNAVAILABLE = "label_unavailable"
REASON_CROSSES_WINDOWS = "crosses_windows"
REASON_OUTSIDE_WINDOWS = "outside_windows"
REASON_GROUP_SPANS_MULTIPLE_WINDOWS = "group_spans_multiple_windows"
ALL_ISOLATION_REASONS = (
    REASON_LABEL_UNAVAILABLE, REASON_CROSSES_WINDOWS,
    REASON_OUTSIDE_WINDOWS, REASON_GROUP_SPANS_MULTIPLE_WINDOWS)


@dataclass
class ParsedRow:
    """The handful of typed fields the engine needs; the raw text is retained."""

    export_id: str
    sample_id: str
    station_id: str
    metric: str
    target_start: int
    target_end: int
    label_status: str
    # ``(column_name, message_digest)`` for every successful feature, in row
    # order.  One row may reference the same digest through several columns.
    message_refs: list[tuple[str, str]]
    raw_text: str
    line_no: int

    @property
    def joint_id(self) -> str:
        return joint_identity(self.export_id, self.sample_id)


def joint_identity(export_id: str, sample_id: str) -> str:
    """Compose the unique row identity.

    ``\\x1f`` (ASCII unit separator) cannot appear inside a validated
    ``export_id`` or ``sample_id``, so the concatenation is unambiguous and its
    UTF-8 byte order is the same order the engine uses for group naming.
    """

    return f"{export_id}\x1f{sample_id}"


def split_joint_identity(joint_id: str) -> tuple[str, str]:
    export_id, sample_id = joint_id.split("\x1f", 1)
    return export_id, sample_id


def _require_nonempty_str(value, *, context: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RowError("row_identity_invalid",
                       {"context": context, "field": field, "value": value,
                        "reason": "non-empty string required"})
    if "\x1f" in value or "\x00" in value:
        raise RowError("row_identity_invalid",
                       {"context": context, "field": field, "value": value,
                        "reason": "control bytes are not allowed in identities"})
    return value


def _epoch_int(value, *, context: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RowError("row_interval_invalid",
                       {"context": context, "field": field, "value": value,
                        "reason": "integer epoch second required"})
    return value


def parse_export_row(doc, *, export_id: str, line_no: int) -> ParsedRow:
    """Validate one decoded export row and extract its typed decision fields."""

    context = f"export {export_id!r} line {line_no}"
    if not isinstance(doc, dict):
        raise RowError("row_record_invalid",
                       {"context": context, "reason": "JSON object required"})
    sample_id = _require_nonempty_str(
        doc.get("sample_id"), context=context, field="sample_id")
    station_id = _require_nonempty_str(
        doc.get("station_id"), context=context, field="station_id")
    metric = _require_nonempty_str(
        doc.get("metric"), context=context, field="metric")

    interval = doc.get("target_interval")
    if not isinstance(interval, list) or len(interval) != 2:
        raise RowError("row_interval_invalid", {
            "context": context, "sample_id": sample_id,
            "reason": "target_interval must be a [start, end] list"})
    target_start = _epoch_int(interval[0], context=context,
                              field="target_interval[0]")
    target_end = _epoch_int(interval[1], context=context,
                            field="target_interval[1]")
    if target_end <= target_start:
        raise RowError("row_interval_invalid", {
            "context": context, "sample_id": sample_id,
            "target_interval": interval,
            "reason": "target interval must be a non-empty half-open range"})

    label = doc.get("label")
    if not isinstance(label, dict) or not isinstance(label.get("status"), str):
        raise RowError("row_record_invalid", {
            "context": context, "sample_id": sample_id,
            "reason": "label object with string status required"})
    label_status = label["status"]

    features = doc.get("features")
    if not isinstance(features, list) or not features:
        raise RowError("row_record_invalid", {
            "context": context, "sample_id": sample_id,
            "reason": "non-empty features list required"})
    message_refs: list[tuple[str, str]] = []
    for col_pos, feature in enumerate(features):
        fctx = f"{context}.features[{col_pos}]"
        if not isinstance(feature, dict):
            raise RowError("row_record_invalid",
                          {"context": fctx, "reason": "feature object required"})
        if feature.get("status") != FEATURE_OK:
            # Digests on missing/withdrawn/unmapped cells are not evidence of
            # a shared consumed field and are intentionally not extracted.
            continue
        column = feature.get("column")
        if not isinstance(column, str) or not column:
            raise RowError("row_record_invalid", {
                "context": fctx, "reason": "successful feature needs a column"})
        digest = (((feature.get("evidence") or {}).get("field") or {})
                  .get("message_digest"))
        if not is_legal_message_digest(digest):
            raise RowError("successful_feature_without_message_digest", {
                "context": fctx, "export_id": export_id,
                "sample_id": sample_id, "column": column,
                "message_digest": digest,
                "reason": "an ok feature must name the original GRIB2 "
                          "message as a 64-hex SHA-256 message_digest; a "
                          "missing digest must not be read as 'unrelated'"})
        message_refs.append((column, digest))

    return ParsedRow(
        export_id=export_id, sample_id=sample_id, station_id=station_id,
        metric=metric, target_start=target_start, target_end=target_end,
        label_status=label_status, message_refs=message_refs,
        raw_text=json.dumps(doc, ensure_ascii=False), line_no=line_no)
