# Offline training-data distribution (`cube-training-distribute`)

This stage sits *after* one or more `cube-training-export` runs.  It consumes
the exported JSONL rows only and splits them, by time plan, into three real
sample files (`train`, `validation`, `holdout`) plus one isolation file.  It
never opens a cube, never re-fetches grid points, never re-runs forecast
publish / station registration / observation-revision matching and never
trains a model.

The problem it solves is **independence of the delivered shards**.  Two rows
that obey point-in-time visibility individually can still be inseparable: they
may have consumed the same original GRIB2 message, may use overlapping station
observation intervals, or may be connected only *transitively* through rows
that cannot themselves enter any shard.  Delivering connected rows to
different windows would leak information across shards.

## 1. Inputs

### 1.1 Export manifest

A JSON list assigning every existing export JSONL a unique, stable
`export_id`:

```json
[
  {"export_id": "export-2026-01", "path": "/data/exports/2026-01.jsonl"},
  {"export_id": "export-2026-02", "path": "/data/exports/2026-02.jsonl"}
]
```

The joint row identity is `(export_id, sample_id)`.  A `sample_id` repeated in
two different exports is **not** merged: the `export_id` keeps the rows
distinct.  A `sample_id` repeated inside one export is a hard
`duplicate_joint_identity` error.  Duplicate `export_id`s and duplicate paths
are rejected before any row is read, as is a missing export file.

### 1.2 Delivery plan

```json
{
  "gap_seconds": 3600,
  "windows": {
    "train":      {"start": "2026-01-01T01:00:00Z", "end": "2026-01-01T04:00:00Z"},
    "validation": {"start": "2026-01-01T04:00:00Z", "end": "2026-01-01T10:00:00Z"},
    "holdout":    {"start": "2026-01-01T12:00:00Z", "end": "2026-01-01T18:00:00Z"}
  }
}
```

* All timestamps are canonical UTC whole seconds; every comparison is integer
  arithmetic.
* `gap_seconds` is a non-negative integer.  Negative values are rejected.
* The three windows are half-open `[start, end)` and mutually
  non-overlapping; any positive-length intersection is a
  `plan_windows_overlap` error.

## 2. Which rows cannot be separated

Two relation kinds are unioned, and the union is transitive.

### 2.1 Shared original GRIB2 message

Only the digest at `evidence.field.message_digest` of a **successful**
(feature `status == "ok"`) feature identifies an original GRIB2 message.  Two
rows relate when an `ok` feature of each references the same digest.  The
relation holds across packages, column names and point coordinates.

None of these create a relation:

* equal feature values;
* the same `package_digest` (one package contains many distinct messages);
* a digest retained by an unsuccessful feature (`field_missing`,
  `forecast_withdrawn`, ...) - such a cell was never consumed;
* a missing or malformed digest on an allegedly `ok` feature.  That is a hard
  `successful_feature_without_message_digest` error, never an implicit
  "unrelated" verdict.  A legal digest is a 64-character lowercase hex
  SHA-256 string.

### 2.2 Overlapping station observation intervals

Rows relate on their own `(station_id, metric)` and `target_interval`,
ignoring the label's revision number and numeric value.  Each target interval
`[start, end)` is provisionally treated as `[start, end + gap_seconds)`.  Two
rows of the same station and metric relate when their provisional intervals
have a **positive-length** intersection:

```
a_start < b_end + gap_seconds  AND  b_start < a_end + gap_seconds
```

Consequences worth pinning down:

* an empty space of exactly `gap_seconds` does not relate (intersection is one
  point);
* with `gap_seconds = 0`, head-to-tail touching intervals do not relate;
* genuinely overlapping or nested intervals always relate.

### 2.3 Transitivity across both sources

Relations cross files and windows and are transitive.  A row with no usable
label - which can never be a shard candidate - still participates and can be
the bridge that joins candidates of two different windows.  The two relation
kinds are fed into one connected-component structure; they are never computed
separately and intersected afterward.

## 3. Applying the delivery plan

A row is a **candidate** for a window only when:

1. its label `status` is exactly `ok`; and
2. its whole target interval falls inside that one half-open window.

Other rows are isolated directly, with every applicable reason retained:

| Situation | Reason |
| --- | --- |
| label status is not `ok` | `label_unavailable` |
| positive overlap with a window but not fully inside one | `crosses_windows` |
| no overlap with any window | `outside_windows` |

Intervals are never trimmed to make them fit.

For each connected component:

* candidates naming exactly **one** window are delivered to that shard; the
  component's non-candidate rows stay isolated but remain recorded as members
  of the same group;
* candidates naming **two or three** windows cause **all** of the component's
  candidates to be isolated with `group_spans_multiple_windows`.  No
  majority, first-match or priority rule ever picks a window;
* a component without candidates produces no shard data.

Each component is named after its smallest joint identity, compared term by
term in UTF-8 byte order.

## 4. Outputs

All outputs land in one directory and are atomically replaced via temp files;
a failed run leaves no deliverable behind.

| File | Content |
| --- | --- |
| `train.jsonl`, `validation.jsonl`, `holdout.jsonl` | one `{"export_id": ..., "row": <original row>}` per delivered candidate; the original row content is preserved |
| `isolated.jsonl` | the same envelope for every isolated row |
| `decisions.jsonl` | one per-row document: joint identity, group, decision, window relation, label status and all reasons |
| `summary.json` | window definitions, per-shard and per-reason counts, group statistics and relation-scale counters |
| `distribution-index.sqlite` | self-contained explanation index (rows, message references, all edges, DSU forest, run metadata) |
| `distribution-index.json` | index manifest: schema version, plan, exports, scale counters |

Every joint identity appears in exactly one of the four row files.  File row
order is joint-id UTF-8 byte order, so results are identical under manifest
reordering, in-file line reordering and different paging/batch sizes.

## 5. Keeping transitivity without enumerating all pairs

A naive implementation would materialize every related pair.  One field
referenced by a million rows would imply ~5×10¹¹ pairs, and one station with
many nested intervals would be quadratic too.  The disk-backed workspace
builds only a linear number of edges while keeping the exact same connected
components.

### 5.1 Message relations: hub-and-spoke stars

All successful-feature references are sorted on disk by
`(message_digest, joint_id, ref_order)`.  The first (smallest joint identity)
reference of each digest becomes its **hub**; every later reference gets one
edge to that hub.  A digest cited `k` times produces exactly `k - 1` edges
instead of `k(k-1)/2` pairs, with identical connectivity.  Hub state lives in
a temporary SQLite table, so keyset pagination never loses it at page
boundaries.

### 5.2 Observation relations: an interval sweep with a disk active set

Within each `(station_id, metric)` group, rows are swept in
`(target_start, joint_id)` order.  A disk-resident active set holds each
row's extended end `end + gap_seconds`.  For a new row:

1. active entries whose extended end is strictly before the new start are
   expired (`ext_end < start`);
2. among the survivors, the row with the greatest extended end (tie: smallest
   joint identity) is joined to the new row, but only when its extended end is
   strictly greater than the new start - the exact-gap and touching cases
   therefore never create an edge;
3. the new row enters the active set.

The single chosen edge suffices because the chosen active row's provisional
interval covers every other active row's interval relative to the new row, so
the sweep graph has the same connected components as the full overlap graph.
Entries that only touch at a point are retained as possible bridges to older
still-overlapping rows, which preserves transitivity.

Each non-first row contributes at most one observation edge, so observation
edges stay below the row count.

### 5.3 Union forest and component naming

Every edge flows through a union-by-size disjoint-set forest stored in the
`rows` table (logarithmic depth, deterministic equal-size tie-breaking by the
tree's smallest identity).  Non-forest edges are still persisted, because they
carry evidence used by explanations.  After both sweeps, one ordered pass
resolves every row to its root and a temporary naming table renames each
component to its smallest joint identity; no full member list is ever held in
Python memory.

### 5.4 Measuring relation scale and memory

`summary.json` and the report expose:

```json
"relation_scale": {
  "rows": N,
  "successful_feature_refs": R,
  "message_edges": E_m,
  "observation_edges": E_o,
  "total_edges": E_m + E_o,
  "forest_edges": E_f
}
```

with the auditable bounds `E_m <= R`, `E_o < N`, `E_m + E_o <= N + R` and
`E_f <= N - 1`.  The on-disk footprint is the size of
`distribution-index.sqlite` (plus its sidecars while a run is in progress).
Working memory is bounded by the SQLite page cache
(`--sqlite-cache-kib`), the keyset page size (`--page-limit`) and the commit
interval (`--commit-every`); it does not grow with row count, hot-field fan-in
or component size.  `python -I tools/distribution_scale_check.py` generates a
hot digest and a deeply nested single-station data set in a temp directory and
asserts conservation, the edge bounds and a near-zero per-1000-row RSS slope
at two cache sizes.

## 6. Explaining an isolated row

```bash
python -I -m grib_asm.training.cli cube-training-explain \
    --index run/delivery/distribution-index.sqlite \
    --export-id export-2026-01 --sample-id A
```

For a row isolated because its group spans windows, the command returns one
chain per other window from the asked row to that window's smallest-id
candidate.  Each step names both endpoint identities and evidence that can be
re-checked against the index:

* **message step**: the two concrete feature columns that used one shared
  `message_digest`, plus the digest itself;
* **observation step**: the station/metric, both original target intervals,
  the configured `gap_seconds`, the extended ends and the empty gap between
  the intervals.

The primary search is deterministic component-local BFS over indexed
incident edges (the whole data set is never rescanned).  A
`--max-chain-nodes` budget caps resident BFS state; if a component is too
large to explore within it, the logarithmic spanning-forest path is used
instead - still evidence-backed and reproducible.  Because the index contains
the rows, references and edges, explanations keep working after the original
export files are deleted.  Asking about a row that was not isolated for
cross-window sharing returns a structured `explain_not_cross_window_isolated`
error rather than a fabricated chain.

## 7. Error handling

Hard errors name the offending location and refuse delivery (exit code 7);
missing evidence is never interpreted as independence:

`manifest_invalid`, `manifest_export_id_missing`,
`manifest_export_id_not_stable`, `manifest_export_path_not_unique`,
`duplicate_export_id`, `manifest_file_missing`, `plan_invalid`,
`plan_window_invalid`, `plan_window_missing`, `plan_windows_overlap`,
`gap_seconds_invalid`, `row_record_invalid`, `row_identity_invalid`,
`duplicate_joint_identity`, `row_interval_invalid`,
`successful_feature_without_message_digest`.

## 8. Commands

```bash
# Distribution (also wired into the grib-asm and grib-cube CLIs)
python -I -m grib_asm.training.cli cube-training-distribute \
    --manifest export-manifest.json --plan delivery-plan.json \
    --out-dir run/delivery

# Isolation explanation from the delivered, self-contained index
python -I -m grib_asm.training.cli cube-training-explain \
    --index run/delivery/distribution-index.sqlite \
    --export-id export-2026-01 --sample-id A

# Hand-authored demo and independent scale/memory check
python -I tools/distribution_demo.py --out run/distribution-demo
python -I tools/distribution_scale_check.py --sizes 500 2000 8000
```
