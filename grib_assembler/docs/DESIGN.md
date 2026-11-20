# Design notes and requirement coverage

## 1. Hard boundaries the implementation never crosses

* Grid point values are never decoded. `headers.py` keeps an explicit
  `_FORBIDDEN_ARRAY_KEYS` set (`values`, `latitudes`, `packedValues`, ...) and
  raises if one is requested. The only place that touches values is
  `tools/fixturegen.py`, test-only, which writes a tiny synthetic array so
  ecCodes can emit a legal Section 7.
* No interpolation, correction, re-encoding or forecast. Package export copies
  encoded messages through a fixed 1 MiB buffer with `open/seek/read/write`;
  packing is untouched.
* Memory does not grow with the archive. Scanning uses fixed buffers and an
  incremental hasher; per-message state goes to SQLite; at most one ecCodes
  handle is open; numeric identity is parsed from small section reads.
  `tests/test_memory.py` measures per-message marginal RSS over a 200-message
  vs a 2000-message archive and asserts sub-linear growth.

## 2. ecCodes build compatibility

The CI/runtime manylinux ecCodes build exposes some keys under non-canonical
names and returns symbolic names or `None` for certain code-table keys:

* `tablesVersion` / `localTablesVersion` instead of
  `masterTablesVersionNumber` / `localTablesVersionNumber`;
* `typeOfStatisticalProcessing` instead of `statisticalProcess`;
* symbolic values (`'kwbc'`, `'pl'`) for centre/surface code keys.

Identity is therefore not taken from those symbolic lookups. `gribbytes.py`
parses the numeric WMO codes directly from the encoded Section 1 and Section 4
octets (offsets calibrated byte-for-byte against ecCodes output), while
ecCodes remains authoritative for timing/statistical scalar headers and for
producing fixtures. `headers.MessageHeader.int_any` provides alias fallback for
the table-version keys.

## 3. Requirement traceability

| Requirement | Where enforced |
| --- | --- |
| Same cycle/grid, explicit variable/level/member/time interval | `identity.BusinessKey`, `readiness.iter_request_keys` |
| Fall back to a complete older cycle; never stitch cycles | `readiness.build_readiness`, `assemble.assemble` |
| Manifest size + SHA-256; bad file excluded with evidence | `catalog.Catalog.ingest_file` |
| Offset/length + whole-message SHA-256, bad boundaries fail file | `scanner.scan_file` |
| Unsupported complete templates marked, not fabricated | `classify.classify_message` status/`reason` |
| PDT 4.0/4.1/4.8/4.11 + single time range constraints | `classify` (`SUPPORTED_PDT`, `_classify_statistical`) |
| Full UTC reference datetime | `timeutil.reference_datetime` |
| Variable identity beyond shortName | `gribbytes.parse_product` + `identity.VariableId` |
| Exact rational levels; missing second surface stays missing | `identity.scaled_value`, `FixedSurface`, `gribbytes` |
| Deterministic/control/perturbed member kinds | `identity.MemberId`, `gribbytes` ensemble octets |
| Grid = full Section 3 digest; packing-independent | `identity.grid_digest`, `tests/test_cell_semantics.py` |
| Second/minute/hour/day to exact integer seconds | `timeutil.UNIT_SECONDS`, `unit_to_seconds` |
| Instant start==end; statistical process 0..3 and interval checks | `timeutil.TimeSlot`, `classify._classify_statistical` |
| 0-6h vs 3-6h not merged | `TimeSlot.key`, `test_e2e_drill.test_same_endpoint_0_6_cannot_fill_3_6` |
| Retransmission = one cell/many sources; conflict unusable | `catalog._store_cell`, `tests/test_cell_semantics.py` |
| Different grid/level/slot never substitute | `readiness._find_other_grid`, near-miss drill tests |
| `not_after` cutoff and origin range | `readiness.candidate_cycles` |
| Blocked => no success package, reasons retained | `assemble.assemble`, CLI exit 4 + `*.blocked.json` |
| Self-contained package, random-seek re-decode without inputs | `assemble` layout, `verify.verify_package`, copy test |
| In-progress dirs never read as results | `assemble` staging + `verify` marker guard |
| readiness/assemble/explain-slot/verify-package independent | `grib_asm/cli.py` subcommands |

## 4. Immutable collections

Collections are keyed by `collection_id` and treated as immutable. Ingest of a
`(collection_id, file_id)` already present is a no-op (`cached`); a new
collection adds or removes files without rewriting prior catalog rows or
already-published packages. Old hand-off directories are never modified; new
assemblies are published under new, content-addressed directory names.

## 5. Main drill outcomes (independently authored expectations)The four-cell matrix is one variable/level, two members (control, perturbed),
two accumulation slots (0-3h, 3-6h):

1. 00Z complete, 06Z missing perturbed 3-6h -> select 00Z, four cells.
2. After 06Z completes -> switch entirely to 06Z.
3. A byte-different copy of a 06Z cell creates a conflict -> fall back to 00Z;
   never three 06Z cells plus one 00Z cell.
4. A same-endpoint 0-6h accumulation, a different grid, or a same-shortName
   different parameter number cannot fill the missing 3-6h cell.
5. No complete cycle -> blocked, no success package.

These are asserted in `tests/test_e2e_drill.py` with hand-written matrices and
reproduced against real data by `tools/run_demo.sh`.

## 6. Analysis cube requirement traceability

The cube pipeline (`grib_asm/cube/`) is the only component allowed to decode
grid point values; everything else in the project remains header/copy-only.

| Requirement | Where enforced |
| --- | --- |
| Completion marker, whole-file digest, per-record offset/length/digest, order and request identity checked before any write | `cube/loader.load_and_validate` (delegates structural work to `verify.verify_package`) |
| Any validation failure leaves the published dataset untouched | `cube/importer.import_package` preflight runs before `begin_transaction`; rollback removes staging |
| Regular lat/lon (GDT 3.0) and scan mode 0 only; single-message full decode | `cube/decode.decode_message` |
| Stable unsupported reasons, no silent skip, cycle never reported complete | `cube/decode`, `cube/loader` raise `UnsupportedCubeMessage`; no partial commit exists |
| Full numeric variable identity; short names never merge variables | `cube/coords.variable_key`, `cube/store.Registry.ensure_variable` |
| Exact rational levels; missing second surface stays missing | `cube/coords.level_key` over `identity.Level` |
| Relative start/end seconds, statistic, absolute validity boundaries; instant vs interval distinct | `cube/coords.slot_key`/`validity_bounds`, `Registry.ensure_slot` |
| Raw units retained; CF mapping only on explicit full-identity rules | `cube/units.resolve_units` |
| Missing cells as explicit mask, never neighbour-filled | `vars/.../present` uint8 array; unwritten cells keep fill 0 and NaN |
| Only grid/identity/unit/level-compatible packages accepted | `cube/importer._preflight` compatibility checks |
| Same origin+digest idempotent; same origin+different digest conflict retained, old data readable | `import_package` idempotency branch, `_record_conflict`, `.cube/conflicts/` |
| Per-cycle commit; no half-registered cycle visible | `cube/store.commit` marker-last `os.replace`; `CubeReader` hides unmarked origins |
| Recovery cleans up or resumes; committed cycles never rewritten | `cube/recovery.recover` (cleanup/resume) |
| Peak memory bounded by batch size and single message, not cycle count | `CubeConfig.batch_size`, full-grid chunks, `tools/cube_stream_check.py` |
| Query by variable/member/level/validity window with stable order, mask and provenance | `cube/query.CubeReader.select` |
| No large generated data committed; reproducible small GRIB2 inputs | `tests/cube_helpers.py`, runtime generation only |
