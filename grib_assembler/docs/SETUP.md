# Environment setup (offline-friendly)

The project needs Python >= 3.11 with `eccodes`, `numpy`, `zarr` (v2) and
`numcodecs`. The manylinux `eccodes` wheel bundles `libeccodes` (via
`eccodeslib`/`eckitlib`), so no system package manager or root access is
required.

## Online

```bash
python -m venv .venv
. .venv/bin/bin/activate  # or: source .venv/bin/activate
python -m pip install eccodes numpy pytest "zarr<3" numcodecs
```

## Air-gapped / restricted network (what this repository was built with)

Download the wheels on a connected host and install them with
`--no-index --find-links`:

```bash
python -m venv .venv
python -m pip install --no-index --find-links ./wheels \
    numpy numcodecs zarr eccodes pytest
# transitive runtime deps if not already present:
python -m pip install attrs cffi findlibs pycparser pluggy iniconfig \
    packaging pygments asciitree fasteners deprecated wrapt eckitlib \
    eccodeslib
```

Verify the native library loads:

```bash
python -c "import eccodes, numpy, zarr, numcodecs; \
  print(eccodes.codes_get_api_version(), zarr.__version__)"
```

## Sanity

```bash
python -m pytest -q            # unit + end-to-end (small committed samples)
python -m pytest -q -m slow    # plus large temporary-archive memory evidence
bash tools/run_demo.sh         # full CLI drill on regenerated demo data
```

The interpreter is always invoked explicitly; when reading untrusted downloaded
archives use `python -I` and keep scripts outside the data directory.
