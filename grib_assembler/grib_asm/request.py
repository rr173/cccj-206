"""Request and manifest models plus strict validation.

A request fixes the forecast origin range, ``not_after`` cutoff, one expected
grid and explicit ordered lists of variables, levels, members and time slots.
The full request space is their Cartesian product; assembly may only choose a
cycle that fills *every* cell.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .identity import Level, MemberId, VariableId, make_surface
from .timeutil import TimeSlot, slot_from_dict


def _parse_iso(text: str) -> datetime:
    if not text.endswith("Z"):
        raise ValueError(f"timestamps must be UTC and end with Z: {text}")
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=__import__("datetime").timezone.utc)


@dataclass(frozen=True)
class RequestedVariable:
    discipline: int
    parameter_category: int
    parameter_number: int
    master_tables_version: int
    local_tables_version: int
    centre: int
    sub_centre: int

    def matches(self, var: VariableId) -> bool:
        return (
            self.discipline == var.discipline
            and self.parameter_category == var.parameter_category
            and self.parameter_number == var.parameter_number
            and self.master_tables_version == var.master_tables_version
            and self.local_tables_version == var.local_tables_version
            and self.centre == var.centre
            and self.sub_centre == var.sub_centre
        )

    def as_variable_id(self) -> VariableId:
        return VariableId(
            discipline=self.discipline,
            parameter_category=self.parameter_category,
            parameter_number=self.parameter_number,
            master_tables_version=self.master_tables_version,
            local_tables_version=self.local_tables_version,
            centre=self.centre,
            sub_centre=self.sub_centre,
        )


@dataclass(frozen=True)
class RequestedLevel:
    first_type: int
    first_factor: int
    first_scaled: int
    second_type: int | None
    second_factor: int | None
    second_scaled: int | None

    def as_level(self) -> Level:
        first = make_surface(self.first_type, self.first_factor,
                             self.first_scaled)
        second = None
        if self.second_type is not None:
            second = make_surface(
                self.second_type,
                # None factors mean "undefined surface" in a request.
                self.second_factor if self.second_factor is not None else 255,
                self.second_scaled if self.second_scaled is not None
                else 0xFFFFFFFF,
            )
        return Level(first=first, second=second)


@dataclass(frozen=True)
class RequestedMember:
    kind: str
    type_of_ensemble: int | None
    number: int | None

    def as_member_id(self) -> MemberId:
        return MemberId(kind=self.kind,
                        type_of_ensemble=self.type_of_ensemble,
                        number=self.number)


@dataclass
class Request:
    request_id: str
    origin_from: datetime
    origin_to: datetime
    not_after: datetime
    expected_grid_digest: str
    variables: list[RequestedVariable]
    levels: list[RequestedLevel]
    members: list[RequestedMember]
    slots: list[TimeSlot]

    def expected_cells(self) -> int:
        return (len(self.variables) * len(self.levels)
                * len(self.members) * len(self.slots))

    def to_dict(self) -> dict:
        from .timeutil import iso
        levels = []
        for lv in self.levels:
            doc = {
                "first": {
                    "type": lv.first_type,
                    "scale_factor": lv.first_factor,
                    "scaled_value": lv.first_scaled,
                },
                "second": None,
            }
            if lv.second_type is not None:
                doc["second"] = {
                    "type": lv.second_type,
                    "scale_factor": lv.second_factor,
                    "scaled_value": lv.second_scaled,
                }
            levels.append(doc)
        return {
            "request_id": self.request_id,
            "origin_from": iso(self.origin_from),
            "origin_to": iso(self.origin_to),
            "not_after": iso(self.not_after),
            "expected_grid_digest": self.expected_grid_digest,
            "variables": [{
                "discipline": v.discipline,
                "parameterCategory": v.parameter_category,
                "parameterNumber": v.parameter_number,
                "masterTablesVersionNumber": v.master_tables_version,
                "localTablesVersionNumber": v.local_tables_version,
                "centre": v.centre,
                "subCentre": v.sub_centre,
            } for v in self.variables],
            "levels": levels,
            "members": [{
                "kind": m.kind,
                "typeOfEnsembleForecast": m.type_of_ensemble,
                "perturbationNumber": m.number,
            } for m in self.members],
            "slots": [s.to_dict() for s in self.slots],
        }


def _pick(d: dict, *keys):
    """Return the first present key's value (accepts camel/snake spellings)."""
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"none of {keys} present in {list(d)}")


def load_request(path: str | Path) -> Request:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ["request_id", "origin_from", "origin_to", "not_after",
                "expected_grid_digest", "variables", "levels", "members",
                "slots"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"request missing keys: {missing}")

    origin_from = _parse_iso(data["origin_from"])
    origin_to = _parse_iso(data["origin_to"])
    not_after = _parse_iso(data["not_after"])
    if origin_to < origin_from:
        raise ValueError("origin_to precedes origin_from")

    variables = [RequestedVariable(
        discipline=int(_pick(v, "discipline")),
        parameter_category=int(_pick(v, "parameterCategory",
                                     "parameter_category")),
        parameter_number=int(_pick(v, "parameterNumber",
                                   "parameter_number")),
        master_tables_version=int(_pick(v, "masterTablesVersionNumber",
                                        "master_tables_version")),
        local_tables_version=int(_pick(v, "localTablesVersionNumber",
                                       "local_tables_version")),
        centre=int(_pick(v, "centre")),
        sub_centre=int(_pick(v, "subCentre", "sub_centre")),
    ) for v in data["variables"]]

    levels = []
    for lv in data["levels"]:
        first = lv["first"]
        second = lv.get("second")
        levels.append(RequestedLevel(
            first_type=int(_pick(first, "type")),
            first_factor=int(_pick(first, "scale_factor", "scaleFactor")),
            first_scaled=int(_pick(first, "scaled_value", "scaledValue")),
            second_type=None if second is None else int(
                _pick(second, "type")),
            second_factor=None if second is None
            else (None if _pick(second, "scale_factor", "scaleFactor") is None
                  else int(_pick(second, "scale_factor", "scaleFactor"))),
            second_scaled=None if second is None
            else (None if _pick(second, "scaled_value", "scaledValue") is None
                  else int(_pick(second, "scaled_value", "scaledValue"))),
        ))

    members = [RequestedMember(
        kind=m["kind"],
        type_of_ensemble=(None if m.get("typeOfEnsembleForecast",
                                        m.get("type_of_ensemble")) is None
                          else int(m.get("typeOfEnsembleForecast",
                                         m.get("type_of_ensemble")))),
        number=(None if m.get("perturbationNumber",
                              m.get("number")) is None
                else int(m.get("perturbationNumber", m.get("number")))),
    ) for m in data["members"]]

    slots = [slot_from_dict(s) for s in data["slots"]]

    if not variables or not levels or not members or not slots:
        raise ValueError("request must contain at least one of each dimension")

    return Request(
        request_id=str(data["request_id"]),
        origin_from=origin_from,
        origin_to=origin_to,
        not_after=not_after,
        expected_grid_digest=str(data["expected_grid_digest"]),
        variables=variables,
        levels=levels,
        members=members,
        slots=slots,
    )


@dataclass
class ManifestFile:
    file_id: str
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass
class Manifest:
    collection_id: str
    created_at: str | None
    files: list[ManifestFile]
    raw: dict

    def ids(self) -> list[str]:
        return [f.file_id for f in self.files]


def load_manifest(path: str | Path) -> Manifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "collection_id" not in data or "files" not in data:
        raise ValueError("manifest requires collection_id and files")
    files = []
    seen = set()
    for entry in data["files"]:
        fid = str(entry["file_id"])
        if fid in seen:
            raise ValueError(f"duplicate file_id in manifest: {fid}")
        seen.add(fid)
        files.append(ManifestFile(
            file_id=fid,
            relative_path=str(entry["relative_path"]),
            size_bytes=int(entry["size_bytes"]),
            sha256=str(entry["sha256"]).lower(),
        ))
    return Manifest(
        collection_id=str(data["collection_id"]),
        created_at=data.get("created_at"),
        files=files,
        raw=data,
    )
