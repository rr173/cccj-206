# Analysis cube: hand-off package -> Zarr

This pipeline converts *completed, self-contained GRIB2 hand-off packages*
into an appendable, slice-readable Zarr analysis dataset. It is strictly
downstream of the byte-preserving assembler: it never re-selects messages,
interpolates, corrects data, computes ensemble statistics or produces a
forecast. The only transformation is decoding grid point values that the
assembler was forbidden to touch, through ecCodes.

## 1. What is written

```
<store>/
  .cube/
    CUBE_META.json               # grid digest + full grid definition
    registry/
      origins.json               # committed reference times (ordered)
      variables.json             # full numeric identity + units/attrs
      levels.json                # exact rational fixed surfaces
      members.json               # deterministic/control/perturbed
      slots.json                 # relative start/end seconds + validity
      sources.json               # origin -> package digest evidence
      conflicts.json             # rejected same-origin/different-digest
      committed/<origin>.json    # per-cycle commit markers
    conflicts/<origin>__<digest>.json
    txn/<txn_id>/                # uncommitted staging + journal
  coords/latitude, coords/longitude
  vars/<variable_key>/
    values          float64  (origin, level, member, slot, lat, lon)
    present         uint8    explicit missing mask
    message_digest  U80      per-cell source message SHA-256
    package_digest  U80      per-cell source package SHA-256
```

* Variable identity keeps discipline, parameter category/number, master and
  local table versions, centre and sub-centre. Two fields with the same
  short name but different numeric codes are different variables and
  different zarr groups.
* Levels keep the exact `scaledValue / 10^scaleFactor` rationals; an
  undefined second surface stays missing.
* Time slots store relative start/end seconds, the statistical process code
  and the absolute UTC validity boundaries. Instantaneous fields
  (`start == end`, `statistic = null`) and statistical intervals are
  distinguishable in the registry; a 0-6 h accumulation can never occupy the
  3-6 h coordinate.
* The raw GRIB unit is always retained. A CF standard name/unit is added
  only when an explicit, versioned rule matches the full numeric variable
  identity (`grib_asm/cube/units.py`); otherwise `mapping_status` is
  `raw_only_no_credible_mapping` and nothing is guessed.
* Missing cells are the `present == 0` mask with NaN values and empty
  provenance strings. Nothing is ever filled from a neighbouring cycle.

## 2. Import validation (all gates must pass)

1. completion marker exists and says `complete`; in-progress dirs refused;
2. whole `messages.grib2` SHA-256 equals the marker digest;
3. per-record offset/length framing and per-message SHA-256 recomputed and
   matched, contiguous offsets, full Cartesian product in request order;
4. re-decoded headers match each record's canonical business key and the
   stored request identity (including its expected grid digest);
5. every message re-classifies as profile-supported;
6. ecCodes can fully decode every message on a regular lat/lon grid
   (GDT 3.0, scanning mode 0); decoded point count equals `Ni * Nj`;
   independently computed coordinates match ecCodes' coordinates.

Stable unsupported reasons include `unsupported_grid_template`,
`unsupported_scanning_mode`, `undecodable_message`, `point_count_mismatch`
and `grid_coordinate_mismatch`. An unsupported message fails the whole
cycle import; it is never silently skipped while the cycle reports success.
Any validation failure leaves the published dataset untouched.

## 3. Compatibility and conflicts

A new package is accepted only when its grid definition equals the dataset
grid (digest and attributes) and every existing variable's identity and
unit mapping is unchanged. New variables/levels/members/slots may be
appended; conflicting existing ones raise `grid_definition_conflict`,
`variable_identity_conflict`, `level_semantics_conflict`, `unit_conflict`
or `slot_validity_boundary_conflict` before any write.

* Same origin + same package digest: idempotent no-op.
* Same origin + different package digest: `source_digest_conflict`; old
  values stay in place and readable, the rejected evidence is retained under
  `.cube/conflicts/` and in `conflicts.json`.

## 4. Per-cycle commit protocol

One package = one origin transaction under `.cube/txn/<txn_id>/`:

1. recovery reconciles stale transactions before opening a new one;
2. preflight decodes the entire package and runs every compatibility check;
3. messages are re-streamed and flushed in batches of `batch_size` into the
   staging zarr group (one open ecCodes handle, one full-grid array at a
   time);
4. the registry snapshot and source record are frozen (`prepared`);
5. at commit, staged cells merge into published arrays, registries swap
   atomically, and the per-origin marker is the last `os.replace`.

Readers only expose origins that have a commit marker. After a crash,
`recover(store, mode="cleanup")` removes staging and truncates the trailing
uncommitted origin; `mode="resume"` finishes a *prepared* transaction from
its frozen snapshot. Committed cycles are never rewritten.

## 5. Memory model

Peak Python memory is bounded by `batch_size` decoded messages plus the
single-message grid (`Nj * Ni` floats), never by the number of cycles.
Zarr chunks are `(1, 1, 1, 1, Nj, Ni)` for values and scalar chunks for the
mask/provenance arrays, matching one analysis slice. The independent check
is `tools/cube_stream_check.py` (also run via the `slow` pytest marker): it
streams many temporary cycles, reads every chunk back and asserts per-cycle
marginal memory stays below a fixed budget for both small and large batch
configurations. No large data is committed; all inputs are generated at
runtime.

## 6. Commands

```bash
# Import (validation + transactional per-cycle commit)
python -m grib_asm.cube.cli cube-import --store run/cube \
    --package-dir run/packages/drill__2026-01-01T000000Z --batch-size 8

# Recover after an interruption
python -m grib_asm.cube.cli cube-recover --store run/cube
python -m grib_asm.cube.cli cube-recover --store run/cube --recovery-mode resume

# Inspect committed coordinates and source provenance
python -m grib_asm.cube.cli cube-inspect --store run/cube

# Slice one variable (numeric identity from the package's request.json)
python -m grib_asm.cube.cli cube-query --store run/cube \
    --request run/packages/.../request.json --variable-index 0 \
    --member-index 1 --validity-end-to 2026-01-01T06:00:00Z --include-values
```

The same subcommands are available under `python -m grib_asm.cli` without a
catalog database, since the cube pipeline only needs the package directory.
