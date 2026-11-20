#!/usr/bin/env bash
# End-to-end point-in-time training-export drill.
#
# Builds a real imported cube plus the four shuffled JSONL histories, then
# runs cube-training-export and prints the training table and gap summary.
# The expected values (point A=21.5, point B=42.25) and the timeline are
# hand-authored in tools/training_demo.py, not derived from the exporter.
# All state lives under ./demo_run/training so the run is self-contained.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"

OUT="$ROOT/demo_run/training"
rm -rf "$OUT"
mkdir -p "$OUT"

echo "== build cube + streams and run the point-in-time export =="
"$PY" -I "$ROOT/tools/training_demo.py" --out "$OUT"

echo
echo "== training table (one JSON object per line) =="
"$PY" - "$OUT/training.jsonl" <<'PY'
import json
import sys
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    main = row["features"][0]
    print(f"{row['sample_id']}: {main['status']:<24} "
          f"value={main['value']!s:<8} label={row['label']['status']}")
PY

echo
echo "== gap/read summary =="
cat "$OUT/summary.json"

echo
echo "Demo complete. Table: $OUT/training.jsonl, summary: $OUT/summary.json"
