# Point-in-time training-table export (`cube-training-export`)

This stage turns a *frozen, stop-writes* Zarr analysis cube plus four
order-independent JSONL histories into a directly consumable training table.
It is data preparation only: it never trains a model, never produces a
forecast, never interpolates and never converts units.

The core problem it solves is **temporal leakage**.  Joining on validity time
alone would pull in forecasts published *after* a historical decision and
station migrations that were only *later* known.  The export instead rebuilds,
for every sample, exactly the features visible at that sample's
`decision_at`, while labels are resolved independently at a common
`label_cutoff` from the measurement corrections already received by then.

## 1. Inputs

All times are canonical UTC strings `YYYY-MM-DDTHH:MM:SSZ` truncated to whole
seconds; every comparison and offset is integer-second arithmetic.

### 1.1 Recipe (`recipe.json`)

```json
{
  "recipe_id": "demo-recipe-v1",
  "label_cutoff": "2026-01-01T18:00:00Z",
  "metric_units": {"temperature": "K"},
  "features": [
    {
      "name": "temp_3_6h",
      "variable": {"discipline": 0, "parameterCategory": 0,
                   "parameterNumber": 0, "masterTablesVersionNumber": 29,
                   "localTablesVersionNumber": 1, "centre": 7,
                   "subCentre": 0},
      "level": {"first": {"type": 100, "scale_factor": 0,
                          "scaled_value": 85000}, "second": null},
      "member": {"kind": "control", "typeOfEnsembleForecast": 3,
                 "perturbationNumber": 0},
      "slot": {"start_seconds": 10800, "end_seconds": 21600,
               "statistic": 1},
      "feature_offset_start_seconds": 0,
      "feature_offset_end_seconds": 10800
    }
  ]
}
```

* `label_cutoff` must be at or after every sample's target end.
* Each feature names the **full** variable/level/member identity and one
  interval-statistics slot, plus the feature window as offsets from the
  sample target start.
* **Two-endpoint origin rule.** With target start `T`, the feature window is
  `[T+off_start, T+off_end)` and the slot is `[s_start, s_end)` relative to an
  origin.  Both `(T+off_start) - s_start` and `(T+off_end) - s_end` must yield
  the same origin, i.e. the window and slot must have equal length; otherwise
  the recipe is rejected with `recipe_origin_mismatch`.  Example: target
  09–12, offsets 0–3 h, slot 3–6 h ⇒ only the 06:00 origin's 09–12 validity
  cell qualifies.  A same-end, different-start slot (e.g. 0–6 h) is a separate
  column located separately and is never substituted.
* Only the explicitly specified slot is ever read; no other cycle is searched
  to fill a gap.

### 1.2 Sample manifest (`samples.jsonl`)

One object per line: `sample_id` (unique), `station_id`, `metric`,
half-open `target_start`/`target_end`, and `decision_at` (must not be later
than `target_start`; equality is allowed).

### 1.3 Publish stream (`publish.jsonl`)

Per origin, `publish` (with `package_summary.package_digest`) and `withdraw`.
`visible_at` must not precede the origin.  The business key is the origin.

### 1.4 Station stream (`station.jsonl`)

Per station, `map` (with `grid_digest` and zero-based `lat_index`/
`lon_index`) and `unmap`, each carrying a half-open effective interval
`[effective_from, effective_to)`.

### 1.5 Observation stream (`observation.jsonl`)

Per `(station, metric, target interval)`, `set` (finite numeric `value` +
`unit`) and `retract`.  Visibility must not precede the interval end.

## 2. The two clocks

| Clock | Cutoff | Records considered |
| --- | --- | --- |
| feature clock | sample `decision_at` | publish + station streams, `visible_at <= decision_at` (equality visible) |
| label clock | recipe `label_cutoff` | observation stream for the exact measurement key, `visible_at <= label_cutoff` |

Observation visibility never influences feature selection, and forecast
visibility never influences labels.

Within one stream, records order by `(visible_at, seq)`; on an equal instant
the larger `seq` wins and file line order is never used.  An identical
retransmission (same business key, instant, sequence and payload) is
deduplicated; the same instant and sequence with different payloads is a hard
`stream_duplicate_conflict`.

### Feature adjudication (station → publish → field/point)

One station location is resolved per sample and shared by all its columns:
the newest visible `map`/`unmap` whose effective interval covers the target
start.  An `unmap` masks only its named interval, so an older `map` still
covers times outside it.  A winning `map` must reference this cube's grid
digest with in-bounds indices, else the export fails.  Then, per column:

1. station gate — `station_not_registered`, `station_unmapped`;
2. publish gate — `forecast_not_published`, `forecast_withdrawn`; a visible
   `publish` must name a committed origin whose recorded package digest
   matches, else `publish_origin_not_committed` / `publish_digest_mismatch`;
3. field/point gate — `field_missing` (variable or cell absent),
   `point_missing` (cell present with provenance but the grid point is
   explicitly missing), or `ok`.

When several gaps apply, station is reported before publish, and publish
before field/point.  Only requests that pass every gate reach the cube, so a
failed gate never decodes a values block.

### Label adjudication

The newest visible record for the exact `(station, metric, target interval)`
key at `label_cutoff` wins: no record ⇒ `label_not_received`; a winning
`retract` ⇒ `label_retracted` (the old value is not reused); a `set` whose
unit differs from the recipe unit ⇒ `label_unit_mismatch`; otherwise `ok`.

## 3. Outputs

* **JSONL**, one row per sample in `sample_id` UTF-8 byte order (left join, so
  every input sample appears even with no matching history).  Each feature
  carries its derived origin, slot, feature window, value or null, an
  explainable status and evidence: the winning station/publish record
  identities (business key, visibility time, sequence), the actual point
  indices, and—whenever the field exists—the package/message digests and raw
  unit even when the point itself is missing.  The label carries its status,
  value/null, unit and winning observation identity.
* **summary JSON** with per-feature-column/label status counts, the sample
  count and the value-block read audit (`value_block_reads`,
  `value_block_keys`).

Stable feature statuses: `station_not_registered`, `station_unmapped`,
`forecast_not_published`, `forecast_withdrawn`, `field_missing`,
`point_missing`, `ok`.  Label statuses: `label_not_received`,
`label_retracted`, `label_unit_mismatch`, `ok`.

## 4. Evidence errors versus ordinary gaps

A missing value is a row-level status; contradictory *input evidence* is a
hard non-zero exit that names the offending record and key, removes any
temporary output, and never touches the cube or source streams:

`station_grid_mismatch`, `station_index_out_of_bounds`,
`publish_origin_not_committed`, `publish_digest_mismatch`,
`cell_provenance_conflict`, `stream_duplicate_conflict`,
`stream_visibility_before_reference`, `recipe_origin_mismatch`,
`label_cutoff_too_early`, `decision_after_target_start`.

## 5. Streaming, buffering and the once-per-block read guarantee

The sample list and streams are never accumulated in Python.  Parsed lines go
straight into a temporary SQLite workspace (cleanable, default temp dir,
`--work-dir`/`--keep-workspace` to inspect).  Resolution streams the samples;
eligible point reads are grouped on disk by their
`(variable, origin, level, member, slot)` values chunk.  The fetch stage walks
the distinct chunks, decodes each full-grid chunk **once**, extracts all
requested points, writes results back, and releases the grid before the next
chunk.  A counting zarr store records every physical `values` chunk fetch, so
the summary's `value_block_reads` is independently auditable.

Working memory is bounded by:

* one decoded full-grid chunk (`Nj × Ni` float64), with at most one chunk
  resident at a time;
* the point-request drain batch (`--point-batch`, default 4096 rows);
* the SQLite page cache (`--sqlite-cache-kib`, default 64 MiB) and commit
  interval (`--commit-every`);
* the in-memory recipe (one entry per feature column), the cube's coordinate
  registries (metadata only), and the per-cell scalar evidence cache.

It does **not** grow with the number of samples, stations or stream records.
Run evidence:

```
python -I tools/training_scale_check.py --sizes 200 800 3200
```

reports full row counts, sampled values, exactly one values-block read per
distinct cell regardless of sample count, and a per-1000-sample RSS slope
near zero at two buffer settings.

## 6. Command

```bash
python -I -m grib_asm.training.cli cube-training-export \
    --store run/cube --recipe recipe.json \
    --samples samples.jsonl --publish-stream publish.jsonl \
    --station-stream station.jsonl --observation-stream observation.jsonl \
    --output-jsonl run/training.jsonl --output-summary run/summary.json
```

The same subcommand is registered under `grib_asm.cube.cli` and the top-level
`grib-asm` CLI.  The end-to-end hand-timed scenario (07:59/08:00/08:30/09:00
decisions, A→B migration, withdrawal, right-end fallback, 5→7→retract labels,
missing point, absent variable, decoy slot) is generated by
`python -I tools/training_demo.py --out run/training-demo`; only small
fixtures and the generation steps live in the repository.
