#!/usr/bin/env bash
# End-to-end CLI drill for the offline GRIB2 assembler.
#
# Regenerates the small demo dataset, ingests collections, shows readiness,
# assembles packages for the headline scenarios, explains a slot and finally
# independently verifies each produced package.  All state lives under
# ./demo_run so the run is fully reproducible and self-contained.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/workspace/venv/bin/python}"
if [ ! -x "$PY" ]; then PY="$(command -v python3)"; fi

DATA="$ROOT/demo_data"
RUN="$ROOT/demo_run"
DB="$RUN/catalog.db"
OUT="$RUN/packages"

rm -rf "$RUN"
mkdir -p "$OUT"

echo "== 1. generate small demo dataset =="
"$PY" "$ROOT/tools/make_demo_data.py" --out "$DATA"

ingest() {
  local cid="$1"
  "$PY" -m grib_asm.cli --catalog-db "$DB" \
    --input-roots "$cid=$DATA/inputs/$cid" \
    ingest --manifests "$DATA/manifests/$cid.json"
}

readiness() {
  "$PY" -m grib_asm.cli --catalog-db "$DB" \
    readiness --request "$DATA/request.json" || true
}

assemble() {
  local cids="$1"
  local mans=()
  local roots=()
  for cid in $cids; do
    mans+=("$DATA/manifests/$cid.json")
    roots+=(--input-roots "$cid=$DATA/inputs/$cid")
  done
  "$PY" -m grib_asm.cli --catalog-db "$DB" "${roots[@]}" \
    assemble --request "$DATA/request.json" \
    --manifests "${mans[@]}" --out-dir "$OUT" || true
}

echo "== 2. ingest complete 00Z and partial 06Z =="
ingest c00
ingest c06_partial

echo "== 3. readiness: expect selection of 00Z (06Z missing one cell) =="
readiness

echo "== 4. assemble: expect four cells all from 00Z =="
assemble "c00 c06_partial"

echo "== 5. explain the missing 06Z perturbed 3-6h cell =="
"$PY" -m grib_asm.cli --catalog-db "$DB" explain-slot \
  --request "$DATA/request.json" \
  --reference-time 2026-01-01T06:00:00Z \
  --variable-index 0 --level-index 0 \
  --member-index 1 --slot-index 1 || true

echo "== 6. independently verify the 00Z package =="
PKG="$(find "$OUT" -maxdepth 1 -type d -name 'drill__*' | head -n 1)"
"$PY" -m grib_asm.cli --catalog-db "$DB" verify-package --package-dir "$PKG"

echo "== 7. import the verified package into a Zarr analysis cube =="
CUBE="$RUN/cube"
"$PY" -m grib_asm.cube.cli cube-import \
  --store "$CUBE" --package-dir "$PKG" --batch-size 4

echo "== 8. inspect committed cube coordinates and sources =="
"$PY" -m grib_asm.cube.cli cube-inspect --store "$CUBE"

echo "== 9. query one variable slice with mask and provenance =="
"$PY" -m grib_asm.cube.cli cube-query --store "$CUBE" \
  --request "$DATA/request.json" --variable-index 0

echo
echo "Demo run complete. Packages under: $OUT, cube under: $CUBE"
