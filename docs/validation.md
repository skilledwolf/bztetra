# Validation

`bztetra` is release-checked against three reference layers:

1. Legacy shell-matrix parity for the `8^3`, `16^3 -> 8^3`, and `16^3` cases.
2. Direct parity against the installed legacy `libtetrabz` Python wrapper on
   same-grid tensor outputs.
3. Analytic reference checks for occupation, DOS, static response, real-
   frequency response, and Matsubara-axis complex response.

These checks currently cover the 3D tetrahedron path only. Strictly 2D support
now has an initial validated `bztetra.twod` occupation/DOS path, while the 2D
response family remains a separate planned track; see
[two_dimensional_plan.md](two_dimensional_plan.md).

Until the first full public release, treat these checks as part of normal use
for important calculations. If a result matters, compare against the original
[`libtetrabz`](https://github.com/mitsuaki1987/libtetrabz) implementation or
run the parity suite below.

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

`tests/test_legacy_wrapper_parity.py` compares `bztetra` against the installed
legacy `libtetrabz` Python wrapper on same-grid outputs for the public
computational surface.

### Analytic Checks

The main analytic and structural checks live in:

- `tests/test_occupancy.py`
- `tests/test_dos.py`
- `tests/test_response.py`
- `tests/test_frequency_response.py`
- `tests/test_complex_frequency_response.py`
- `tests/test_twod_geometry.py`
- `tests/test_twod_occupancy.py`
- `tests/test_twod_dos.py`

These include free-electron integrals, Lindhard limits, and Matsubara-anchor
checks in addition to output-shape and dtype validation.

The first 2D analytic checks live in:

- `tests/test_twod_geometry.py`
- `tests/test_twod_occupancy.py`
- `tests/test_twod_dos.py`

## Original Project

`bztetra` is a clean-room port informed by the original `libtetrabz` project by
Mitsuaki Kawamura and collaborators. For original source, manuals, and the
legacy Python wrapper, see:

- [github.com/mitsuaki1987/libtetrabz](https://github.com/mitsuaki1987/libtetrabz)
- [mitsuaki1987.github.io/libtetrabz/python/_build/html/index.html](https://mitsuaki1987.github.io/libtetrabz/python/_build/html/index.html)

If you use `bztetra` in research, also cite the original method paper:

- M. Kawamura, Y. Gohda, and S. Tsuneyuki, "Improved tetrahedron method for the
  Brillouin-zone integration applicable to response functions,"
  [Phys. Rev. B 89, 094515 (2014)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.094515)
