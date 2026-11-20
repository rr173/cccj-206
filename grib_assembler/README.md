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
  cli.py        ingest | readiness | assemble | explain-slot | verify-package
tools/
  fixturegen.py     ecCodes fixture encoder (test-only place that writes values)
  demo_common.py    shared demo/test spec builders
  make_demo_data.py regenerates the small committed demo dataset
  run_demo.sh       end-to-end CLI drill
tests/              unit + end-to-end + memory-flatness tests
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
python -m pytest -q                 # unit + end-to-end (small samples)
python -m pytest -q -m slow         # include large temporary-archive memory test
bash tools/run_demo.sh              # full CLI drill on regenerated demo data
```

Large archives are generated in temporary directories by the test suite and
are never committed; only the small demo samples and their generation commands
live in the repository.
