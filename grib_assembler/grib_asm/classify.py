"""Profile-aware message classification.

Turns one encoded GRIB2 message (located by :mod:`grib_asm.scanner`, read via
:mod:`grib_asm.headers`) into a normalised record whose identity follows the
profile:

* PDT 4.0/4.1 are instantaneous point-in-time products;
* PDT 4.8/4.11 are single-range statistical products with
  ``typeOfTimeIncrement=2``, ``timeIncrement=0``,
  ``numberOfMissingInStatisticalProcess=0`` and statistical process 0/1/2/3;
* only second/minute/hour/day time units are accepted;
* the explicit end-of-interval datetime must agree with
  reference + forecastTime + lengthOfTimeRange.

Anything well-formed but outside the profile is marked ``unsupported`` with a
stable reason code; it never fabricates missing fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .errors import UnsupportedTemplate
from .gribbytes import (
    MISSING_U32, MISSING_U8, parse_identification, parse_product,
)
from .headers import MessageHeader
from .identity import (
    BusinessKey, Level, MemberId, VariableId, grid_digest,
    deterministic_member, ensemble_member, make_surface,
)
from .scanner import MessageSpan
from .timeutil import (
    TimeSlot, explicit_datetime, iso, reference_datetime, unit_to_seconds,
)

SUPPORTED_PDT = (0, 1, 8, 11)
INSTANT_PDT = (0, 1)
STAT_PDT = (8, 11)
ALLOWED_STAT_PROCESSES = (0, 1, 2, 3)
REQUIRED_TYPE_OF_TIME_INCREMENT = 2


@dataclass
class NormalizedRecord:
    status: str                 # "ok" | "unsupported"
    reason: str | None
    span: MessageSpan
    file_id: str
    collection_id: str
    reference_time: datetime | None = None
    variable: VariableId | None = None
    level: Level | None = None
    member: MemberId | None = None
    slot: TimeSlot | None = None
    grid_digest: str | None = None
    pdt: int | None = None
    header_snapshot: dict = field(default_factory=dict)
    # Filled when status == "ok".
    business_key: BusinessKey | None = None

    def to_catalog_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "collection_id": self.collection_id,
            "file_id": self.file_id,
            "offset": self.span.offset,
            "length": self.span.length,
            "message_sha256": self.span.sha256,
            "grid_digest": self.grid_digest,
            "pdt": self.pdt,
            "reference_time": None if self.reference_time is None
            else iso(self.reference_time),
            "variable": None if self.variable is None
            else self.variable.to_dict(),
            "level": None if self.level is None else self.level.to_dict(),
            "member": None if self.member is None else self.member.to_dict(),
            "slot": None if self.slot is None else self.slot.to_dict(),
            "business_key_digest": None if self.business_key is None
            else self.business_key.digest(),
            "header_snapshot": self.header_snapshot,
        }


def _build_level(prod) -> Level:
    first = make_surface(prod.first_surface_type,
                         prod.first_scale_factor,
                         prod.first_scaled_value)
    second = None
    if prod.second_surface_type != MISSING_U8:
        second = make_surface(prod.second_surface_type,
                              prod.second_scale_factor,
                              prod.second_scaled_value)
    return Level(first=first, second=second)


def _build_member(prod) -> MemberId:
    if prod.template_number in (1, 11):
        return ensemble_member(prod.type_of_ensemble_forecast,
                               prod.perturbation_number)
    return deterministic_member()


def _build_variable(ident, prod) -> VariableId:
    return VariableId(
        discipline=prod.discipline,
        parameter_category=prod.parameter_category,
        parameter_number=prod.parameter_number,
        master_tables_version=ident.master_tables_version,
        local_tables_version=ident.local_tables_version,
        centre=ident.centre,
        sub_centre=ident.sub_centre,
    )


def _classify_instant(h: MessageHeader, ref: datetime) -> TimeSlot:
    forecast = unit_to_seconds(
        h.int("indicatorOfUnitOfTimeRange"),
        h.int("forecastTime"),
    )
    return TimeSlot(forecast, forecast, None)


def _classify_statistical(h: MessageHeader, ref: datetime) -> TimeSlot:
    pdt_range_count = h.int("numberOfTimeRange")
    if pdt_range_count != 1:
        raise UnsupportedTemplate("multiple_time_ranges", {
            "numberOfTimeRange": pdt_range_count,
        })

    inc_type = h.int("typeOfTimeIncrement")
    if inc_type != REQUIRED_TYPE_OF_TIME_INCREMENT:
        raise UnsupportedTemplate("unsupported_type_of_time_increment", {
            "typeOfTimeIncrement": inc_type,
        })

    inc_unit = h.int("indicatorOfUnitForTimeIncrement")
    inc_value = h.int("timeIncrement")
    # A zero increment is zero in any supported unit; non-zero increments are
    # outside this profile.
    if inc_value != 0:
        raise UnsupportedTemplate("nonzero_time_increment", {
            "indicatorOfUnitForTimeIncrement": inc_unit,
            "timeIncrement": inc_value,
        })

    missing = h.int("numberOfMissingInStatisticalProcess")
    if missing != 0:
        raise UnsupportedTemplate("missing_values_in_statistical_process", {
            "numberOfMissingInStatisticalProcess": missing,
        })

    process = (h.int("statisticalProcess", default=-1)
               if h.has("statisticalProcess") else -1)
    if process < 0 and h.has("typeOfStatisticalProcessing"):
        process = h.int("typeOfStatisticalProcessing")
    if process not in ALLOWED_STAT_PROCESSES:
        raise UnsupportedTemplate("unsupported_statistical_process", {
            "statisticalProcess": process,
        })

    start_offset = unit_to_seconds(
        h.int("indicatorOfUnitOfTimeRange"),
        h.int("forecastTime"),
    )
    length_seconds = unit_to_seconds(
        h.int("indicatorOfUnitForTimeRange"),
        h.int("lengthOfTimeRange"),
    )
    end_offset = start_offset + length_seconds

    # Explicit end-of-overall-interval must match ref + end_offset exactly.
    explicit_end = explicit_datetime(
        h.int("yearOfEndOfOverallTimeInterval"),
        h.int("monthOfEndOfOverallTimeInterval"),
        h.int("dayOfEndOfOverallTimeInterval"),
        h.int("hourOfEndOfOverallTimeInterval"),
        h.int("minuteOfEndOfOverallTimeInterval"),
        h.int("secondOfEndOfOverallTimeInterval"),
    )
    expected_end = ref + timedelta(seconds=end_offset)
    if explicit_end != expected_end:
        raise UnsupportedTemplate("interval_end_mismatch", {
            "explicit_end": iso(explicit_end),
            "computed_end": iso(expected_end),
            "start_offset_seconds": start_offset,
            "length_seconds": length_seconds,
        })

    return TimeSlot(start_offset, end_offset, int(process))


def classify_message(handle: Any, raw: bytes | None, span: MessageSpan,
                     file_id: str, collection_id: str, *,
                     fileobj=None, msg_offset: int | None = None,
                     msg_length: int | None = None) -> NormalizedRecord:
    """Classify one open message handle against the profile.

    Provide either ``raw`` (full message bytes) or ``fileobj`` with
    ``msg_offset``/``msg_length``.  Numeric identity codes come from
    :mod:`gribbytes` (build-independent); ecCodes supplies timing/statistical
    scalar headers and the audit snapshot.  Grid point arrays are never read.
    """

    h = MessageHeader(handle)
    if raw is not None:
        ident = parse_identification(raw)
        prod = parse_product(raw)
    else:
        ident = parse_identification(
            f=fileobj, msg_offset=msg_offset, msg_length=msg_length)
        prod = parse_product(
            f=fileobj, msg_offset=msg_offset, msg_length=msg_length)
    rec = NormalizedRecord(
        status="ok",
        reason=None,
        span=span,
        file_id=file_id,
        collection_id=collection_id,
    )
    rec.header_snapshot = h.snapshot()

    try:
        pdt = prod.template_number
        rec.pdt = pdt
        if pdt not in SUPPORTED_PDT:
            raise UnsupportedTemplate("unsupported_pdt",
                                      {"productDefinitionTemplateNumber": pdt})

        ref = reference_datetime(ident.data_date, ident.data_time)
        rec.reference_time = ref
        rec.variable = _build_variable(ident, prod)
        rec.level = _build_level(prod)
        rec.member = _build_member(prod)
        rec.grid_digest = grid_digest(span.gds_bytes)

        if pdt in INSTANT_PDT:
            rec.slot = _classify_instant(h, ref)
        else:
            rec.slot = _classify_statistical(h, ref)

        rec.business_key = BusinessKey(
            reference_time=iso(ref),
            grid_digest=rec.grid_digest,
            variable=rec.variable,
            level=rec.level,
            member=rec.member,
            slot=rec.slot,
        )
        return rec
    except UnsupportedTemplate as exc:
        rec.status = "unsupported"
        rec.reason = exc.reason
        rec.header_snapshot.setdefault("unsupported_evidence", exc.evidence)
        return rec
