# Validation

`tetrabz` is release-checked against three reference layers:

1. Legacy shell-matrix parity for the `8^3`, `16^3 -> 8^3`, and `16^3` cases.
2. Direct parity against the installed legacy `libtetrabz` Python wrapper on
   same-grid tensor outputs.
3. Analytic reference checks for occupation, DOS, static response, real-
   frequency response, and Matsubara-axis complex response.

## Reproduce The Full Gate

From a development checkout:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
```

To include direct parity against the legacy Python wrapper:

```bash
pip install -e ./libtetra_original/python
python -m pytest -q tests/test_legacy_shell_matrix.py tests/test_legacy_wrapper_parity.py
```

## What Each Layer Covers

### Legacy Shell Matrix

`tests/test_legacy_shell_matrix.py` checks the automated equivalents of the
legacy shell cases:

- `8^3`
- `16^3 -> 8^3`
- `16^3`

These tests exercise the full public routine matrix across occupation, DOS, and
response workflows.

### Direct Wrapper Parity

`tests/test_legacy_wrapper_parity.py` compares `tetrabz` against the installed
legacy `libtetrabz` Python wrapper on same-grid outputs for the public
computational surface.

### Analytic Checks

The main analytic and structural checks live in:

- `tests/test_occupancy.py`
- `tests/test_dos.py`
- `tests/test_response.py`
- `tests/test_frequency_response.py`
- `tests/test_complex_frequency_response.py`

These include free-electron integrals, Lindhard limits, and Matsubara-anchor
checks in addition to output-shape and dtype validation.
