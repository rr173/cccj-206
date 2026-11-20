"""Orchestrate one verified hand-off package into one cube cycle.

The import is a strict validate-then-write pipeline:

1. crash recovery runs first so stale staging never participates;
2. the package is independently verified and re-decoded
   (:mod:`grib_asm.cube.loader`);
3. a *preflight* decoding pass streams every message once, proving the whole
   cycle is decodable on the supported grid and checking every compatibility
   rule before any published byte changes;
4. the transaction then streams the messages a second time and flushes
   batches bounded by ``CubeConfig.batch_size``;
5. the per-cycle commit marker is the last atomic operation.

Nothing in this module re-selects, interpolates, corrects or computes: the
values written are exactly what ecCodes decodes from the package's own
messages, in the package's own order and units.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..headers import close_handle, open_handle_from_file
from ..timeutil import iso
from .coords import (
    level_key, member_key, slot_doc, slot_key, validity_bounds,
    variable_key,
)
from .canonical import same_identity
from .decode import decode_message
from .errors import (
    CubeCompatibilityError, CubeConflictError, UnsupportedCubeMessage,
)
from .loader import ValidatedPackage, load_and_validate
from .recovery import recover
from .store import (
    CUBE_DIR, CubeConfig, CubeStore, PendingCell, Registry,
    _atomic_write_json, marker_name,
)
from .units import UnitInfo, resolve_units

# Named fault-injection points, enabled only via CUBE_FAIL_POINT for tests.
_FAIL_ENV = "CUBE_FAIL_POINT"


def _fail_point(name: str) -> None:
    if os.environ.get(_FAIL_ENV) == name:
        # Hard exit: simulates process kill / power loss, skipping cleanup.
        os._exit(42)


@dataclass
class ImportReport:
    store: str
    package_dir: str
    origin: str
    package_digest: str
    status: str                     # committed | idempotent | unsupported
    record_count: int
    new_variables: list[str] = field(default_factory=list)
    new_levels: list[str] = field(default_factory=list)
    new_members: list[str] = field(default_factory=list)
    new_slots: list[str] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "store": self.store,
            "package_dir": self.package_dir,
            "origin": self.origin,
            "package_digest": self.package_digest,
            "status": self.status,
            "record_count": self.record_count,
            "new_variables": self.new_variables,
            "new_levels": self.new_levels,
            "new_members": self.new_members,
            "new_slots": self.new_slots,
            "conflicts": self.conflicts,
        }


def _now() -> str:
    return iso(datetime.now(timezone.utc))


@dataclass
class _CellPlan:
    record_index: int
    var_key: str
    level_k: str
    member_k: str
    slot_k: str
    validity_start: str
    validity_end: str
    unit_info: UnitInfo
    observed_names: dict


def import_package(store_path: str | Path, package_dir: str | Path,
                   config: CubeConfig | None = None,
                   *, recovery_mode: str = "cleanup") -> ImportReport:
    """Import one complete hand-off package as one cube cycle."""

    store_path = Path(store_path)
    config = config or CubeConfig()
    # Never let an old half-transaction influence a new import.
    recover(store_path, mode=recovery_mode)

    validated = load_and_validate(package_dir)
    store = CubeStore(store_path, config)
    try:
        registry = Registry(store_path)
        origin = validated.selected_reference_time

        # --- idempotency / source conflict -------------------------------
        existing_source = registry.sources.get(origin)
        if existing_source is not None:
            if existing_source.get("package_digest") == \
                    validated.package_digest:
                return ImportReport(
                    store=str(store_path),
                    package_dir=str(package_dir), origin=origin,
                    package_digest=validated.package_digest,
                    status="idempotent",
                    record_count=validated.cell_count())
            conflict = _record_conflict(
                store_path, registry, origin, existing_source, validated)
            raise CubeConflictError("source_digest_conflict", conflict)

        # --- preflight: full decode + compatibility, no published writes -
        plan, first_grid, first_grid_digest = _preflight(
            store, registry, validated)
        # Coordinates are staged only for the dataset's first-ever import;
        # later cycles must not attempt to recreate the published axes.
        initial_grid = None if store.is_initialized() else first_grid
        initial_grid_digest = (
            None if store.is_initialized() else first_grid_digest)

        # Register coordinates in memory only; persistence happens at commit.
        ref_dt = _reference_dt(validated)
        for cell_plan, rec in zip(plan, validated.records):
            bk = rec.business_key
            registry.ensure_variable(
                bk.variable, cell_plan.unit_info,
                cell_plan.observed_names)
            registry.ensure_level(bk.level)
            registry.ensure_member(bk.member)
            registry.ensure_slot(bk.slot)
            registry.set_slot_validity(
                origin, bk.slot,
                cell_plan.validity_start, cell_plan.validity_end)

        # --- transactional import ----------------------------------------
        txn = store.begin_transaction(
            origin, validated.package_digest, registry,
            initial_grid=initial_grid,
            initial_grid_digest=initial_grid_digest)
        try:
            _stream_and_flush(store, txn, registry, validated, plan, config)
            source_doc = {
                "package_digest": validated.package_digest,
                "package_dir": str(validated.package_dir),
                "request_id": validated.request_doc["request_id"],
                "record_count": validated.cell_count(),
                "imported_at": _now(),
            }
            registry.sources[origin] = source_doc
            _fail_point("after_flush_before_prepare")
            store.prepare(txn, registry, source_doc)
            _fail_point("after_prepare_before_commit")
            # Commit-internal fail points are forwarded as an explicit
            # argument so the same store API serves production and tests.
            internal_point = None
            fp = os.environ.get(_FAIL_ENV)
            if fp in ("during_array_write", "during_registry_swap",
                      "during_commit_marker"):
                internal_point = fp
            store.commit(txn, registry, fail_point=internal_point)
            _fail_point("after_commit")
        except BaseException:
            store.rollback(txn)
            raise
        return ImportReport(
            store=str(store_path), package_dir=str(package_dir),
            origin=origin, package_digest=validated.package_digest,
            status="committed", record_count=validated.cell_count(),
            new_variables=plan_meta(plan, "var"),
            new_levels=plan_meta(plan, "lvl"),
            new_members=plan_meta(plan, "mem"),
            new_slots=plan_meta(plan, "slot"))
    finally:
        del store


def plan_meta(plan: list[_CellPlan], kind: str) -> list[str]:
    attr = {"var": "var_key", "lvl": "level_k", "mem": "member_k",
            "slot": "slot_k"}[kind]
    return sorted({getattr(c, attr) for c in plan})


def _reference_dt(validated: ValidatedPackage):
    rec = validated.records[0].business_key
    # Reference time is identical for every record in a hand-off package.
    return datetime.strptime(
        rec.reference_time, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)


def _preflight(store: CubeStore, registry: Registry,
               validated: ValidatedPackage):
    """Decode every message once; build the coordinate plan and run checks."""

    ref_dt = _reference_dt(validated)
    plans: list[_CellPlan] = []
    first_grid = None
    first_grid_digest = None

    for rec in validated.records:
        # Fresh descriptor per message: one reused native stream handle
        # cannot walk consecutive messages in this ecCodes build.
        with open(validated.data_path, "rb") as fh:
            handle = open_handle_from_file(fh, rec.offset)
            try:
                decoded = decode_message(handle)
            finally:
                close_handle(handle)

            bk = rec.business_key
            grid_digest = bk.grid_digest
            if first_grid is None:
                first_grid = decoded.grid
                first_grid_digest = grid_digest
                if store.is_initialized():
                    if store.meta["grid_digest"] != grid_digest:
                        raise CubeCompatibilityError(
                            "grid_definition_conflict", {
                                "dataset_grid_digest":
                                    store.meta["grid_digest"],
                                "package_grid_digest": grid_digest})
                    if store.meta["grid_definition"] != \
                            decoded.grid.definition_doc():
                        raise CubeCompatibilityError(
                            "grid_definition_conflict", {
                                "reason": "grid definition attributes differ",
                                "dataset": store.meta["grid_definition"],
                                "package": decoded.grid.definition_doc()})
            elif grid_digest != first_grid_digest:
                raise UnsupportedCubeMessage(
                    "grid_definition_conflict_within_package", {
                        "record_index": rec.index,
                        "expected": first_grid_digest,
                        "got": grid_digest})

            var = bk.variable
            vk = variable_key(var)
            unit_info = resolve_units(var, decoded.raw_unit)
            if vk in registry.variables:
                existing = registry.variables[vk]
                if not same_identity(existing["identity"], var.to_dict()):
                    raise CubeCompatibilityError(
                        "variable_identity_conflict", {
                            "variable_key": vk,
                            "dataset": existing["identity"],
                            "package": var.to_dict()})
                if existing["units"] != unit_info.doc():
                    raise CubeCompatibilityError("unit_conflict", {
                        "variable_key": vk,
                        "dataset_units": existing["units"],
                        "package_units": unit_info.doc()})

            lk = level_key(bk.level)
            if lk in registry.levels:
                existing_level = registry.levels[lk]
                if not same_identity(existing_level["identity"], bk.level.to_dict()):
                    raise CubeCompatibilityError(
                        "level_semantics_conflict", {
                            "level_key": lk,
                            "dataset": existing_level["identity"],
                            "package": bk.level.to_dict()})
            mk = member_key(bk.member)
            if mk in registry.members:
                existing_member = registry.members[mk]
                if not same_identity(existing_member["identity"], bk.member.to_dict()):
                    raise CubeCompatibilityError(
                        "member_identity_conflict", {
                            "member_key": mk,
                            "dataset": existing_member["identity"],
                            "package": bk.member.to_dict()})
            vstart, vend = validity_bounds(ref_dt, bk.slot)
            sk = slot_key(bk.slot)
            if sk in registry.slots:
                existing_slot = registry.slots[sk]
                if bool(existing_slot["is_instant"]) != bk.slot.is_instant:
                    raise CubeCompatibilityError(
                        "slot_kind_conflict", {"slot_key": sk})
                if not same_identity(existing_slot["identity"],
                                     slot_doc(bk.slot)):
                    raise CubeCompatibilityError(
                        "slot_identity_conflict", {"slot_key": sk})

            plans.append(_CellPlan(
                record_index=rec.index, var_key=vk, level_k=lk,
                member_k=mk, slot_k=sk, validity_start=vstart,
                validity_end=vend, unit_info=unit_info,
                observed_names={
                    "shortName_observed": decoded.short_name,
                    "name_observed": decoded.name,
                }))

    return plans, first_grid, first_grid_digest


def _stream_and_flush(store: CubeStore, txn, registry: Registry,
                      validated: ValidatedPackage, plan: list[_CellPlan],
                      config: CubeConfig) -> None:
    """Second streaming pass: decode, batch and flush bounded by batch_size."""

    batch: list[PendingCell] = []
    for cell_plan, rec in zip(plan, validated.records):
        # Fresh descriptor per message (see loader for the ecCodes reason).
        with open(validated.data_path, "rb") as fh:
            handle = open_handle_from_file(fh, rec.offset)
            try:
                decoded = decode_message(handle)
            finally:
                close_handle(handle)
        batch.append(PendingCell(
            var_key=cell_plan.var_key,
            origin_index=txn.origin_index,
            level_index=registry.levels[cell_plan.level_k]["index"],
            member_index=registry.members[cell_plan.member_k]["index"],
            slot_index=registry.slots[cell_plan.slot_k]["index"],
            values=decoded.values,
            message_digest=rec.message_sha256,
            package_digest=validated.package_digest))
        if len(batch) >= config.batch_size:
            _fail_point("during_batch_flush")
            store.flush(txn, batch, registry)
            batch.clear()
    if batch:
        store.flush(txn, batch, registry)
        batch.clear()


def _record_conflict(store_path: Path, registry: Registry, origin: str,
                     existing_source: dict,
                     validated: ValidatedPackage) -> dict:
    """Persist readable evidence of a same-origin/different-digest attempt."""

    conflict = {
        "origin": origin,
        "reason": "source_digest_conflict",
        "existing": existing_source,
        "rejected": {
            "package_digest": validated.package_digest,
            "package_dir": str(validated.package_dir),
            "request_id": validated.request_doc["request_id"],
            "record_count": validated.cell_count(),
            "observed_at": _now(),
        },
    }
    conflicts_dir = store_path / CUBE_DIR / "conflicts"
    conflicts_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = conflicts_dir / (
        f"{marker_name(origin)}__{validated.package_digest[:12]}.json")
    _atomic_write_json(evidence_path, conflict)
    registry.conflicts.append(conflict)
    # The registry is otherwise untouched; only the conflict log is appended.
    _atomic_write_json(store_path / CUBE_DIR / "registry" / "conflicts.json",
                       {"entries": registry.conflicts})
    return conflict
