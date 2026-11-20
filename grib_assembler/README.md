# Offline GRIB2 Collection Assembler

An offline, byte-preserving GRIB2 assembly tool. It reads **only GRIB2 section
headers** with ecCodes and **copies encoded messages byte-for-byte**. It never
decodes grid point values, never interpolates, never corrects data and never
produces a weather forecast.

## What it guarantees

* **Same cycle, same grid, explicit identity.** A canonical business key binds
  reference time, full grid-definition digest, variable identity
  (discipline, parameter category/number, master/local tables, centre,
  sub-centre — never short name alone), exact rational level surfaces,
  ensemble member kind/number and the whole time slot.
* **No cycle mixing.** Assembly fills the request Cartesian product from one
  reference time. A newer cycle with a missing or conflicted cell is rejected
  and the tool falls back to the newest *complete* cycle; if none is complete
  it reports `blocked` and writes no success package.
* **Conflicts stay conflicts.** Same key + same whole-message digest is a
  retransmission (one cell, several sources). Same key + different digest is a
  conflict and the cell is unusable regardless of file recency.
* **Exact levels and units.** Fixed surfaces use scale factor + integer scaled
  value as exact rationals; an undefined second surface stays `missing` (never
  zero, and ground is never confused with 0 Pa). Only second/minute/hour/day
  step units are accepted and converted to exact integer seconds.
* **Statistical products are interval-aware.** 0–6 h and 3–6 h accumulations
  are different products; the explicit end-of-interval datetime and
  `lengthOfTimeRange` are cross-checked against reference + forecast time.
* **Self-contained hand-off.** Each package contains the concatenated raw
  messages in request order, per-record offsets/lengths/hashes/source
  coordinates/header snapshots, the source manifests and the request, plus a
  completion marker. A copied package can be re-seeked, re-decoded and
  re-verified without the original input tree.
* **Flat memory.** Files are scanned in fixed buffers; per-message state lives
  in a SQLite catalog; only one ecCodes message handle is open at a time and
  data array keys are hard-refused.

## Layout

```
grib_asm/
  scanner.py    streaming boundary scan + per-message/file SHA-256
  timeutil.py   exact-second units, reference time, time slots
  identity.py   variable/level/member/grid identities and business keys
  headers.py    ecCodes scalar-only header access (array keys refused)
  classify.py   profile rules (PDT 4.0/4.1/4.8/4.11, statistics, units)
  catalog.py    disk-backed SQLite catalog, manifest verification, evidence
  readiness.py  per-cycle readiness, missing/conflict reasons, selection
  assemble.py   fixed-buffer byte copy into an atomic hand-off package
  explain.py    single-cell explanation (independent of assemble)
  verify.py     independent package re-seek / header re-decode / re-hash
  cube/         hand-off package -> Zarr analysis cube (values decoded once)
    loader.py     package validation (marker, digests, order, identity)
    decode.py     ecCodes grid-point decoding (regular lat/lon only)
    units.py      conservative, full-identity CF standard-name mapping
    coords.py     canonical coordinate keys and validity boundaries
    store.py      per-cycle transactions, staging and zarr arrays
    recovery.py   crash recovery: cleanup or resume of prepared work
    importer.py   validate -> preflight -> batch stream -> commit
    query.py      slice reads with mask, coordinates and provenance
    cli.py        cube-import | cube-recover | cube-inspect | cube-query
  training/       point-in-time feature/label training-table export
    model.py        whole-second clock, recipe/sample/stream record parsing
    timeline.py     visibility + sequence ordering, resend/conflict rules
    workspace.py    disk-backed SQLite staging (inputs never held in memory)
    cubeaccess.py   chunk-once, provenance-annotated point reads
    export.py       staged resolve -> fetch -> emit pipeline and statuses
    errors.py       evidence-bearing error reasons
    cli.py          cube-training-export
tools/
  fixturegen.py        ecCodes fixture encoder (test-only place that writes values)
  demo_common.py       shared demo/test spec builders
  make_demo_data.py    regenerates the small committed demo dataset
  cube_stream_check.py independent streaming/peak-memory check
  training_fixtures.py real-cube + JSONL builders for the training export
  training_demo.py     hand-timed point-in-time export demo
  training_scale_check.py large shuffled manifest / chunk-read / memory check
  run_demo.sh          end-to-end CLI drill
tests/                 unit + end-to-end + memory-flatness + cube tests
```

## Setup

```bash
# A venv with eccodes + numpy (the eccodes manylinux wheel bundles libeccodes).
python -m pip install eccodes numpy pytest
```

## Commands

```bash
# 1. Ingest a collection (streaming; manifest size + SHA-256 verified).
python -m grib_asm.cli --catalog-db run/cat.db \
  --input-roots c00=demo_data/inputs/c00 \
  ingest --manifests demo_data/manifests/c00.json

# 2. Readiness across cycles (exit 3 when blocked).
python -m grib_asm.cli --catalog-db run/cat.db \
  readiness --request demo_data/request.json

# 3. Assemble a package (exit 4 when blocked; writes a *.blocked.json report).
python -m grib_asm.cli --catalog-db run/cat.db \
  --input-roots c00=... --input-roots c06=... \
  assemble --request demo_data/request.json \
  --manifests demo_data/manifests/c00.json demo_data/manifests/c06.json \
  --out-dir run/packages

# 4. Explain one cell independently.
python -m grib_asm.cli --catalog-db run/cat.db explain-slot \
  --request demo_data/request.json --reference-time 2026-01-01T06:00:00Z \
  --variable-index 0 --level-index 0 --member-index 1 --slot-index 1

# 5. Independently verify a package from its directory alone.
python -m grib_asm.cli --catalog-db run/cat.db verify-package \
  --package-dir run/packages/drill__2026-01-01T00-00-00Z
```

## Tests

```bash
# A venv with eccodes + numpy + zarr v2 + numcodecs.
python -m pip install eccodes numpy pytest "zarr<3" numcodecs
python -m pytest -q                 # unit + end-to-end (small samples)
python -m pytest -q -m slow         # include large temporary-archive memory test
bash tools/run_demo.sh              # full CLI drill on regenerated demo data
```

Large archives are generated in temporary directories by the test suite and
are never committed; only the small demo samples and their generation commands
live in the repository.

## Hand-off package -> analysis cube

`grib_asm.cube` converts completed hand-off packages into an appendable,
slice-readable Zarr dataset. It is the one component that decodes grid point
values (via ecCodes); it never re-selects messages, interpolates, corrects
data, computes ensemble statistics or produces a forecast.

* Imports verify the completion marker, whole-file digest, per-message
  offset/length/digest, message order and request identity before writing;
  any failure leaves the published dataset untouched.
* Only regular lat/lon grids (GDT 3.0, scan mode 0) whose single message
  fully decodes are accepted; other shapes carry stable
  `unsupported_*` reasons and never silently skip past an incomplete cycle.
* Arrays are keyed by full numeric variable identity, exact rational level,
  member kind/number, reference time and time slot. Slots keep relative
  start/end seconds, the statistical process and absolute validity
  boundaries, so instant fields and 0-6 h / 3-6 h accumulations stay
  distinct. Missing cells are an explicit mask, never neighbour-filled.
* Same origin + same package digest is idempotent; same origin + different
  digest is a retained source conflict and the old values stay readable.
  Different grid/level/variable semantics are rejected before any write.
* Commits are per cycle under `.cube/txn/` with a commit-marker publish
  boundary; after a crash, recovery cleans up or resumes prepared work and
  never rewrites a committed cycle.
* Memory is bounded by `--batch-size` and the single-message grid, not the
  number of cycles; chunks are one full grid per analysis slice.

See [docs/CUBE.md](docs/CUBE.md) for the full format, validation and
recovery specification.

```bash
python -m grib_asm.cube.cli cube-import --store run/cube \
    --package-dir run/packages/drill__2026-01-01T000000Z --batch-size 8
python -m grib_asm.cube.cli cube-inspect --store run/cube
python -m grib_asm.cube.cli cube-query --store run/cube \
    --request run/packages/drill__2026-01-01T000000Z/request.json \
    --include-values
```

## Point-in-time training-table export

`grib_asm.training` rebuilds, per sample, the features genuinely visible at the
sample's historical `decision_at` and attaches a label resolved independently
at a common `label_cutoff`, so later-published forecasts and back-filled
station migrations never leak into history.  It is data preparation only: no
modelling, forecasting, interpolation or unit conversion.

* Two clocks stay separate: the feature clock (`decision_at`) gates
  publish/station records; the label clock (`label_cutoff`) gates measurement
  corrections, which never influence feature selection.
* Each feature column names its full variable/level/member identity, its
  interval-statistics slot and its target-relative window; both window
  endpoints must derive the same origin.  Same-end, different-start slots are
  located independently, never substituted, and no other cycle is searched to
  fill a gap.
* Statuses distinguish not-registered, unmapped, not-published, withdrawn,
  missing field and an explicitly missing grid point, applying station →
  publish → field/point precedence.  Labels distinguish not-received,
  retracted and unit-mismatch.
* Contradictory evidence (wrong/unknown grid, out-of-bounds point, publish
  digest the cube cannot confirm, same-instant conflicting records) is a hard
  evidence error naming the key, with no success output left behind.
* Inputs stream through a cleanable SQLite workspace; each accessed `values`
  block is decoded at most once per run, and peak memory is bounded by one
  grid chunk, the point batch, the SQLite cache and recipe/registry metadata.

See [docs/TRAINING.md](docs/TRAINING.md) for the full contract and memory
model.

```bash
python -I -m grib_asm.training.cli cube-training-export \
    --store run/cube --recipe recipe.json \
    --samples samples.jsonl --publish-stream publish.jsonl \
    --station-stream station.jsonl --observation-stream observation.jsonl \
    --output-jsonl run/training.jsonl --output-summary run/summary.json

python -I tools/training_demo.py --out run/training-demo       # hand-timed demo
python -I tools/training_scale_check.py --sizes 200 800 3200   # memory/reads
```
