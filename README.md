# tetrabz

`tetrabz` is a clean-room Python and Numba port of the legacy `libtetrabz`
tetrahedron-integration library. The root package is being built around a
modern Python API, reproducible validation fixtures, and performance-oriented
array layouts instead of a literal line-by-line Fortran translation.

The legacy source tree remains in `libtetra_original/` for reference and is
intentionally ignored by the root git repository.

The implementation strategy is:

- NumPy for array orchestration and test/reference helpers.
- Numba for hot scalar and tensor kernels once the numerical contracts are
  locked down. The occupation, DOS, integrated-DOS, `dblstep`, `dbldelta`,
  `polstat`, `fermigr`, and `polcmplx` paths are now compiled.
- SciPy only where it materially helps validation, reference calculations, or
  tooling; it is not a planned dependency for the core runtime path right now.

## Dev Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

For plot-only installs without the rest of the dev tooling, use
`pip install -e '.[plot]'`.

For a quick human-review snapshot of the current geometry/indexing layer, the
shared cut-formula helpers, and one occupation-weight result, run:

```bash
.venv/bin/python examples/review_geometry_and_cuts.py
```

For a focused occupation / Fermi-search review against the legacy 8x8 toy
system, run:

```bash
.venv/bin/python examples/review_occupancy.py
```

For a physically meaningful DOS review plot, reproduce the legacy cubic
tight-binding example and write a figure under `build/review_plots/`:

```bash
.venv/bin/python examples/plot_tight_binding_dos.py
```

For a physically meaningful static-polarization review plot, reproduce the
legacy 3D free-electron Lindhard example and write a figure under
`build/review_plots/`:

```bash
.venv/bin/python examples/plot_lindhard.py
```

For a physically meaningful real-frequency review plot, reproduce the
free-electron Fermi-golden-rule toy model with the legacy `k^2` matrix element
and write a figure under `build/review_plots/`:

```bash
.venv/bin/python examples/plot_fermigr.py
```

For a physically meaningful complex-frequency review plot, reproduce the
broadened free-electron interband response with exact comparison channels and
write a figure under `build/review_plots/`:

```bash
.venv/bin/python examples/plot_polcmplx.py
```

For a numeric DOS / integrated-DOS review that compares the current 8x8
free-electron fixture against the analytic continuum target, run:

```bash
.venv/bin/python examples/review_dos.py
```

For a quick local timing baseline on the current hot paths (`occ`, `dos`,
`intdos`, `dblstep`, `dbldelta`, `polstat`, `fermigr`, `polcmplx`), run:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_hotpaths.py
```

For a more focused `polstat` timing sweep that isolates the multiband free-
electron case, the Lindhard small-`q` / `2k_F` branches, and the interpolation
path, run:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_polstat.py --grid 16 --weight-grid 8 --q-values 0.125,2.0
```
