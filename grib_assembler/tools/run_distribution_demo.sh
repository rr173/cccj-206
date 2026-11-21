#!/usr/bin/env bash
# End-to-end connected-source distribution drill.
#
# Builds two hand-authored export-shaped JSONL files, an export manifest and a
# delivery plan, then runs cube-training-distribute and explains one row that
# is isolated because its connected group spans the train and validation
# windows.  No cube is opened and no history matching is re-run.  All state
# lives under ./demo_run/distribution so the run is self-contained.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"

OUT="$ROOT/demo_run/distribution"
rm -rf "$OUT"
mkdir -p "$OUT"

echo "== build hand-authored exports and run connected-source distribution =="
"$PY" -I "$ROOT/tools/distribution_demo.py" --out "$OUT"

echo
echo "== per-row decisions =="
"$PY" - "$OUT/delivery/decisions.jsonl" <<'PY'
import json
import sys
for line in open(sys.argv[1], encoding="utf-8"):
    doc = json.loads(line)
    print(f"{doc['export_id']}/{doc['sample_id']:<8} "
          f"{doc['decision']:<10} reasons={','.join(doc['reasons']) or '-'}")
PY

echo
echo "== relation-scale counters =="
"$PY" - "$OUT/delivery/summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(summary["relation_scale"], indent=2))
PY

echo
echo "Demo complete. Shards and index: $OUT/delivery"
