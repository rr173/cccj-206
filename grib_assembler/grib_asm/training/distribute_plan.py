"""Strict parsing and validation for the distribution manifest and plan.

Two small on-the-wire documents drive a distribution run:

* the **export manifest** names every already-produced ``cube-training-export``
  JSONL and assigns each one a unique, stable ``export_id``.  The
  ``(export_id, sample_id)`` pair is the only joint row identity: identically
  named samples in different exports are never merged.
* the **delivery plan** names three mutually exclusive, half-open UTC windows
  (``train``, ``validation``, ``holdout``) and the non-negative
  ``gap_seconds`` used when deciding whether two observation target intervals
  of the same ``(station_id, metric)`` are related.

Everything is parsed here, up front, so that a bad document is rejected with a
:class:`PlanError`/:class:`ManifestError` before any row is read.  All times
are UTC whole seconds stored as integer epoch values, matching the export
stage: interval arithmetic stays integer-only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .distribute_errors import ManifestError, PlanError
from .model import epoch_iso, parse_epoch

SHARDS = ("train", "validation", "holdout")
# 64 lowercase hex characters, matching hashlib.sha256(...).hexdigest().
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# An export_id must stay usable inside the lexicographic joint identity and
# inside file names; reject empty values and the record separator.
EXPORT_ID_RE = re.compile(r"^[^\x1f\x00]+$")


@dataclass(frozen=True)
class Window:
    """One half-open UTC delivery window ``[start_epoch, end_epoch)``."""

    name: str
    start_epoch: int
    end_epoch: int

    def contains(self, start_epoch: int, end_epoch: int) -> bool:
        """True when the whole half-open target interval lies in the window."""

        return self.start_epoch <= start_epoch and end_epoch <= self.end_epoch

    def intersects(self, start_epoch: int, end_epoch: int) -> bool:
        """True when the target interval has any positive-length overlap."""

        return self.start_epoch < end_epoch and start_epoch < self.end_epoch

    def to_dict(self) -> dict:
        return {"name": self.name, "start": epoch_iso(self.start_epoch),
                "end": epoch_iso(self.end_epoch)}


@dataclass(frozen=True)
class DeliveryPlan:
    gap_seconds: int
    windows: dict[str, Window]

    def classify(self, start_epoch: int, end_epoch: int) -> tuple[str, str | None]:
        """Return ``(window_name | None, window_relation)`` for an interval.

        ``window_relation`` is one of:

        * ``in_window`` - the whole interval falls in exactly one window;
        * ``crosses_windows`` - positive-length overlap with some window but
          the interval is not fully inside any single one;
        * ``outside_windows`` - no overlap with any window at all.
        """

        inside = [w for w in self.windows.values()
                  if w.contains(start_epoch, end_epoch)]
        if len(inside) == 1:
            return inside[0].name, "in_window"
        if any(w.intersects(start_epoch, end_epoch)
               for w in self.windows.values()):
            return None, "crosses_windows"
        return None, "outside_windows"


@dataclass(frozen=True)
class ExportEntry:
    export_id: str
    path: str
    order: int


def load_delivery_plan(path: str | Path) -> DeliveryPlan:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise PlanError("plan_invalid",
                        {"path": str(path), "reason": exc.msg})
    return plan_from_dict(data, source=str(path))


def plan_from_dict(data: dict, *, source: str = "<plan>") -> DeliveryPlan:
    context = f"plan {source}"
    if not isinstance(data, dict):
        raise PlanError("plan_invalid",
                        {"context": context, "reason": "object required"})
    gap = data.get("gap_seconds")
    if isinstance(gap, bool) or not isinstance(gap, int):
        raise PlanError("gap_seconds_invalid",
                        {"context": context, "value": gap,
                         "reason": "non-negative integer required"})
    if gap < 0:
        raise PlanError("gap_seconds_invalid",
                        {"context": context, "value": gap,
                         "reason": "non-negative integer required"})
    raw_windows = data.get("windows")
    if not isinstance(raw_windows, dict):
        raise PlanError("plan_invalid",
                        {"context": context,
                         "reason": "windows object required"})
    windows: dict[str, Window] = {}
    for name in SHARDS:
        doc = raw_windows.get(name)
        if not isinstance(doc, dict):
            raise PlanError("plan_window_missing",
                            {"context": context, "window": name})
        if "start" not in doc or "end" not in doc:
            raise PlanError("plan_window_invalid",
                            {"context": context, "window": name,
                             "reason": "start and end required"})
        start = parse_epoch(doc["start"], context=f"{context}.{name}")
        end = parse_epoch(doc["end"], context=f"{context}.{name}")
        if end <= start:
            raise PlanError("plan_window_invalid",
                            {"context": context, "window": name,
                             "start": doc["start"], "end": doc["end"],
                             "reason": "window must be a non-empty "
                                       "half-open range"})
        windows[name] = Window(name, start, end)
    # The plan promises mutually non-overlapping windows; reject any positive-
    # length intersection rather than silently picking one of them.
    ordered = [windows[n] for n in SHARDS]
    for left, right in zip(ordered, ordered[1:]):
        if left.intersects(right.start_epoch, right.end_epoch):
            raise PlanError("plan_windows_overlap", {
                "context": context,
                "left": left.to_dict(), "right": right.to_dict()})
    return DeliveryPlan(gap_seconds=gap, windows=windows)


def load_export_manifest(path: str | Path) -> list[ExportEntry]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ManifestError("manifest_invalid",
                            {"path": str(path), "reason": exc.msg})
    return manifest_from_list(data, source=str(path))


def manifest_from_list(data, *, source: str = "<manifest>") -> list[ExportEntry]:
    context = f"manifest {source}"
    if not isinstance(data, list) or not data:
        raise ManifestError("manifest_invalid", {
            "context": context,
            "reason": "non-empty list of export entries required"})
    entries: list[ExportEntry] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for order, doc in enumerate(data):
        ectx = f"{context}[{order}]"
        if not isinstance(doc, dict):
            raise ManifestError("manifest_invalid",
                                {"context": ectx, "reason": "object required"})
        export_id = doc.get("export_id")
        if not isinstance(export_id, str) or not export_id:
            raise ManifestError("manifest_export_id_missing",
                                {"context": ectx})
        if not EXPORT_ID_RE.match(export_id):
            raise ManifestError("manifest_export_id_not_stable", {
                "context": ectx, "export_id": export_id,
                "reason": "export_id must be a non-empty string without "
                          "NUL/unit-separator control bytes"})
        if export_id in seen_ids:
            raise ManifestError("duplicate_export_id",
                                {"context": ectx, "export_id": export_id})
        path = doc.get("path")
        if not isinstance(path, str) or not path:
            raise ManifestError("manifest_invalid",
                                {"context": ectx, "export_id": export_id,
                                 "reason": "path required"})
        if path in seen_paths:
            raise ManifestError("manifest_export_path_not_unique",
                                {"context": ectx, "path": path})
        seen_ids.add(export_id)
        seen_paths.add(path)
        entries.append(ExportEntry(export_id=export_id, path=path,
                                   order=order))
    return entries


def is_legal_message_digest(value) -> bool:
    """Legal evidence digest: a non-empty lowercase 64-hex SHA-256 string."""

    return isinstance(value, str) and bool(SHA256_RE.match(value))
