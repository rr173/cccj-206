# Package integrity specification

This document defines the on-disk hand-off package and the exact checks the
independent verifier performs. All identifiers are derived from raw GRIB2
numeric codes; short names are never used for identity.

## 1. Input manifest and file acceptance

Each immutable collection has a `collection_id` and a manifest listing every
file with `file_id`, `relative_path`, `size_bytes` and `sha256`. A file is
accepted only when:

1. it exists at the manifest-relative path;
2. its actual byte size equals `size_bytes`;
3. its streaming whole-file SHA-256 equals the manifest `sha256`;
4. the streaming boundary scan finds continuous, complete GRIB2 messages
   (magic `GRIB`, edition 2, declared length, end marker `7777`) with no
   truncation, overrun, missing end marker or inter/intra-file junk.

Any failure marks the **whole file** `failed` with a stable reason code and
evidence (offsets, expected/actual values); the file is excluded from assembly
but its evidence is retained in the catalog. Individual messages are never
salvaged from a broken boundary.

## 2. Per-message record

For every complete message the catalog stores input coordinates and hashes:

* `offset`, `length` within the input file;
* whole-message SHA-256 (incrementally computed while streaming);
* raw Section 3 bytes and the grid digest;
* PDT number, reference time, normalized variable/level/member/slot;
* a scalar header snapshot (no data arrays).

## 3. Canonical business key

```
reference_time (ISO UTC)
grid_digest     = SHA-256 over complete Section 3 bytes (packing excluded)
variable        = discipline, parameterCategory, parameterNumber,
                  masterTablesVersionNumber, localTablesVersionNumber,
                  centre, subCentre
level           = (first fixed surface, second fixed surface or null)
                  each surface = type + exact rational scaledValue
member          = deterministic | control(type 3) | perturbed(type 1,number)
slot            = (start_seconds, end_seconds, statistic|null)
```

* Level rational = `scaledValue / 10^scaleFactor`, exact `Fraction`. A missing
  scale/value yields `null`, never zero.
* Time units restricted to second(13)/minute(0)/hour(1)/day(2), converted to
  exact integer seconds; other units are `unsupported`.
* Instant products use `start == end` and `statistic = null`.
* Statistical products require single time range,
  `typeOfTimeIncrement = 2`, `timeIncrement = 0`,
  `numberOfMissingInStatisticalProcess = 0`, process in `{0,1,2,3}`, and the
  explicit interval end must equal reference + forecastTime + lengthOfTimeRange.

## 4. Cell states

* **retransmission** — same key, one message digest, several sources: one cell.
* **conflict** — same key, more than one distinct message digest: the cell is
  unusable; no source is picked for being newer.
* **missing** — no message with the full key.
* **grid_mismatch** — a message matches every dimension except the grid
  digest; it is evidence only and never substitutes.

Different reference time, grid, level or time interval (e.g. 0–6 h vs 3–6 h)
are always distinct cells.

## 5. Cycle selection

For each candidate reference time within `[origin_from, origin_to]` and
`<= not_after`, every request Cartesian-product cell is evaluated. A cycle is
complete only when all cells are `filled`. The newest complete cycle is
selected; otherwise the result is `blocked`, with per-cycle missing/conflict/
grid-mismatch reasons, and no success package is published. Cells are never
cherry-picked across cycles.

## 6. Hand-off directory

Published atomically (staged as `*.in-progress`, renamed only on completion):

```
<request_id>__<reference_time>/
  PACKAGE_COMPLETE.json      # status, data SHA-256, record/byte totals
  messages.grib2             # raw messages concatenated in request order
  package_manifest.json      # per-record offset/length/hash/source/header
  request.json               # the exact request
  readiness.json             # per-cycle readiness and selection rationale
  source_manifests/          # every source collection manifest (verbatim)
```

Messages are copied with a fixed 1 MiB buffer; packing is unchanged and nothing
is re-encoded. Each record records output offset/length/SHA-256, the canonical
business key, the selected reference time, all retransmission sources, the
original `(collection_id, file_id, message_index, input_offset, length)` and
the header snapshot.

## 7. Independent verification

From the package directory alone, the verifier:

1. requires a valid `PACKAGE_COMPLETE.json` and refuses in-progress dirs;
2. streams the data file and recomputes its whole SHA-256;
3. seeks each record offset, checks GRIB/`7777` framing, recomputes the
   per-message SHA-256 and compares it with both the output and source hashes;
4. hands exactly that encoded message to ecCodes and re-decodes scalar headers
   (never data arrays), checking variable codes, reference time, surface type
   and PDT/statistic against the request and the stored business key;
5. confirms records cover the full Cartesian product in request order with
   contiguous, byte-exact offsets.
